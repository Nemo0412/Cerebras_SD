"""Top-k log-probs over the vocabulary in one read of the bf16 logits.

F.log_softmax(logits.float(), -1).topk(k) casts the [R, V] logits to fp32, writes the fp32
log_softmax and runs a radix top-k over it, several passes over R*V. lse_topk reads the logits
once: a Triton kernel gives, per row and vocabulary block, the block's max, its sum of
exp(x - max) and its top-k (value, index); the rows are then merged on the small per-block
results. The scores are (x - max) - log(sum), the fp32 formula of log_softmax, up to the order
of the fp32 sum. Ties are broken towards the lower token id.
"""
import torch
import triton
import triton.language as tl

BLOCK = 4096


@triton.jit
def _block_lse_topk(X, stride, V, M, S, CV, CI, NB, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    blk = tl.program_id(1)
    ar = tl.arange(0, BLOCK)
    offs = blk * BLOCK + ar
    x = tl.load(X + row.to(tl.int64) * stride + offs, mask=offs < V,
                other=float("-inf")).to(tl.float32)
    bm = tl.max(x, axis=0)
    bs = tl.sum(tl.exp(x - bm), axis=0)
    o = row.to(tl.int64) * NB + blk
    tl.store(M + o, bm)
    tl.store(S + o, bs)
    for j in tl.static_range(K):
        v = tl.max(x, axis=0)
        i = tl.argmax(x, axis=0, tie_break_left=True)
        tl.store(CV + o * K + j, v)
        tl.store(CI + o * K + j, blk * BLOCK + i)
        x = tl.where(ar == i, float("-inf"), x)


def lse_topk(logits, k):
    """(scores, tokens) like F.log_softmax(logits.float(), -1).topk(k, dim=-1); fp32 scores,
    int64 tokens, sorted by score."""
    shape, V = logits.shape[:-1], logits.size(-1)
    x = logits.reshape(-1, V)
    if x.stride(-1) != 1:
        x = x.contiguous()
    R = x.size(0)
    NB = triton.cdiv(V, BLOCK)
    dev = x.device
    m = torch.empty(R, NB, device=dev, dtype=torch.float32)
    s = torch.empty(R, NB, device=dev, dtype=torch.float32)
    cv = torch.empty(R, NB * k, device=dev, dtype=torch.float32)
    ci = torch.empty(R, NB * k, device=dev, dtype=torch.int32)
    _block_lse_topk[(R, NB)](x, x.stride(0), V, m, s, cv, ci, NB, K=k, BLOCK=BLOCK,
                             num_warps=8)
    mx = m.max(1, keepdim=True).values
    lse_rest = torch.log((s * torch.exp(m - mx)).sum(1, keepdim=True))
    # candidates are in token-id order within equal values, so a stable sort keeps the lower id
    v, o = cv.sort(dim=1, descending=True, stable=True)
    tok = ci.gather(1, o[:, :k]).long()
    sc = (v[:, :k] - mx) - lse_rest
    return sc.view(*shape, k), tok.view(*shape, k)
