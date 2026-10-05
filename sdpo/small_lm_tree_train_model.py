"""Small-LM tree on-policy rollout training, parameterized by tree depth.

Tree shape: branch=2, max depth D (D ≥ 1). Layout:
  - depth 0: 1 root (node 0)
  - depth d: 2^d nodes, indices [2^d - 1, 2^(d+1) - 1)
  - total N = 2^(D+1) - 1 nodes
  - total P = 2^D root-to-leaf paths, each of length (D+1) nodes

parents[i] = (i - 1) // 2 for i ≥ 1, parents[0] = -1.
path k's node at depth d = (2^d - 1) + (k >> (D - d)).

Per training step, per sample:
  1. Target prefix forward (frozen, KV cache, no grad).
  2. Draft prefix forward (KV cache).
  3. Pick anchor position (random valid in loss_mask).
  4. Draft tree rollout from anchor, D sequential forwards:
       Step 1: 1 root input → 1 prediction → top-2 = 2 depth-1 tokens
       Step d (d ≥ 2): 2^(d-1) depth-(d-1) inputs (with tree mask) → top-2 each.
  5. Target tree verify (single forward, tree attention mask).
  6. Per-node KL anchor loss over N nodes. "Prediction of node i's token" uses
     parent slot's logits (or prefix logits for root).
  7. Per-path EAL aux loss: P paths × (D+1) nodes each; β_k =
     P_target(draft_token_k) at parent slot, then EAL = Σ_k cumprod(β).
  8. total_loss = anchor_kl_coef * mean_kl - eal_aux_coef * mean_eal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None


def _build_tree_topology(D):
    """Return (parents, depths, paths, N) for a branch-2 tree of max depth D."""
    N = 2 ** (D + 1) - 1
    P = 2 ** D
    parents = [-1] * N
    depths = [0] * N
    for i in range(1, N):
        parents[i] = (i - 1) // 2
        depths[i] = depths[parents[i]] + 1
    paths = []
    for k in range(P):
        path = []
        for d in range(D + 1):
            path.append((2 ** d - 1) + (k >> (D - d)))
        paths.append(tuple(path))
    return tuple(parents), tuple(depths), tuple(paths), N


def _extract_kv_lists(past_kv):
    if past_kv is None:
        return [], []
    if hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache"):
        return list(past_kv.key_cache), list(past_kv.value_cache)
    if hasattr(past_kv, "layers"):
        keys, values = [], []
        for layer in past_kv.layers:
            if hasattr(layer, "keys"):
                keys.append(layer.keys)
                values.append(layer.values)
            elif hasattr(layer, "key_cache"):
                keys.append(layer.key_cache)
                values.append(layer.value_cache)
            else:
                raise RuntimeError(f"Unknown LayerCache: {type(layer)}")
        return keys, values
    if isinstance(past_kv, (tuple, list)):
        return [l[0] for l in past_kv], [l[1] for l in past_kv]
    raise RuntimeError(f"Unknown past_key_values type: {type(past_kv)}")


def _slice_cache_to_bs1(keys, values, b, kv_len):
    new_cache = DynamicCache()
    for layer_idx in range(len(keys)):
        k_sliced = keys[layer_idx][b:b+1, :, :kv_len, :].contiguous()
        v_sliced = values[layer_idx][b:b+1, :, :kv_len, :].contiguous()
        new_cache.update(k_sliced, v_sliced, layer_idx)
    return new_cache


class SmallLMTreeTrainModel(nn.Module):

    def __init__(self, target_path, draft_path, tree_depth=2,
                 dtype=torch.float16):
        super().__init__()
        self.tree_depth = int(tree_depth)
        assert self.tree_depth >= 1
        parents, depths, paths, N = _build_tree_topology(self.tree_depth)
        self.tree_parents = parents
        self.tree_depths = depths
        self.tree_paths = paths
        self.tree_n_nodes = N

        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_path, torch_dtype=dtype)
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_path, torch_dtype=dtype, attn_implementation="sdpa")
        for p in self.target_model.parameters():
            p.requires_grad = False
        self.target_model.eval()

        self.register_buffer(
            "paths", torch.tensor(paths, dtype=torch.long), persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self.target_model.eval()
        self.draft_model.train(mode)
        return self

    def _target_tree_mask(self, prefix_len, device, dtype):
        """4D mask for target tree verify forward; input = [node 0..N-1]."""
        N = self.tree_n_nodes
        parents = self.tree_parents
        min_val = torch.finfo(dtype).min
        mask = torch.full((1, 1, N, prefix_len + N), min_val,
                          device=device, dtype=dtype)
        mask[0, 0, :, :prefix_len] = 0.0
        for i in range(N):
            j = i
            while j >= 0:
                mask[0, 0, i, prefix_len + j] = 0.0
                j = parents[j]
        return mask

    def _draft_step_mask(self, d_step, prefix_len, device, dtype):
        """4D mask for draft step d_step (d_step ≥ 2).
        Input: 2^(d_step-1) depth-(d_step-1) tokens.
        Cache before this step: prefix + (2^(d_step-1) - 1) old tree nodes.
        Each new token i ∈ [0, 2^(d_step-1)) sees prefix + ancestor chain + self.
        """
        min_val = torch.finfo(dtype).min
        q_len = 2 ** (d_step - 1)
        cache_before = 2 ** (d_step - 1) - 1
        kv_len = prefix_len + cache_before + q_len
        mask = torch.full((1, 1, q_len, kv_len), min_val,
                          device=device, dtype=dtype)
        mask[0, 0, :, :prefix_len] = 0.0
        for q in range(q_len):
            for k in range(d_step - 1):    # ancestor depths 0..d_step-2
                anc_at_k = q >> (d_step - 1 - k)
                slot = prefix_len + (2 ** k - 1) + anc_at_k
                mask[0, 0, q, slot] = 0.0
            # self in new tokens region
            mask[0, 0, q, prefix_len + cache_before + q] = 0.0
        return mask

    def _rollout_one_anchor(self, b, anchor_pos,
                            drf_prefix_logits, tgt_prefix_logits,
                            drf_keys, drf_values, tgt_keys, tgt_values):
        device = drf_prefix_logits.device
        D = self.tree_depth
        N = self.tree_n_nodes
        kv_len = anchor_pos + 1
        anchor_abs = anchor_pos

        # ── Per-step parent prediction logits (for KL "label" + EAL β) ──
        # parent_pred_lg[i] = prediction of node i's token, lives at parent slot.

        # ── Step 0 (draft): root pred from prefix ──
        drf_root_pred_lg = drf_prefix_logits[b, anchor_pos].float()    # [V]
        tgt_root_pred_lg = tgt_prefix_logits[b, anchor_pos].float()    # [V]
        root_token = drf_root_pred_lg.argmax(-1)                       # scalar

        # ── Iterate D draft steps to grow tree ──
        drf_cache = _slice_cache_to_bs1(drf_keys, drf_values, b, kv_len)
        tokens_per_depth = [root_token.view(1)]   # tokens_per_depth[d] has 2^d tokens
        drf_pred_per_step = [drf_root_pred_lg]    # prediction logits used to pick tokens at depth d_step
        # drf_pred_per_step[d_step] is the prediction of depth d_step's tokens
        # (i.e., logits from step d_step's forward, before topk).
        # For each "parent" at depth d_step-1, we get 1 logit row.

        # Step 1: input root_token, no mask needed (just causal extend)
        if D >= 1:
            cur_input = root_token.view(1, 1)
            cur_pos = torch.tensor([[anchor_abs + 1]], device=device, dtype=torch.long)
            step_out = self.draft_model(
                input_ids=cur_input,
                position_ids=cur_pos,
                past_key_values=drf_cache,
                use_cache=True, return_dict=True,
            )
            depth1_pred_lg = step_out.logits[0, -1, :].float()    # [V]
            drf_pred_per_step.append(depth1_pred_lg.unsqueeze(0))    # [1, V] — parent at depth 0 (1 row)
            depth1_tokens = torch.topk(depth1_pred_lg, 2).indices    # [2]
            tokens_per_depth.append(depth1_tokens)
            drf_cache = step_out.past_key_values

        # Steps 2..D
        for d_step in range(2, D + 1):
            n_in = 2 ** (d_step - 1)
            mask = self._draft_step_mask(
                d_step, kv_len, device, self.draft_model.dtype)
            input_tokens = tokens_per_depth[d_step - 1].view(1, n_in)
            pos = torch.full((1, n_in), anchor_abs + d_step,
                             device=device, dtype=torch.long)
            step_out = self.draft_model(
                input_ids=input_tokens,
                attention_mask=mask,
                position_ids=pos,
                past_key_values=drf_cache,
                use_cache=(d_step < D), return_dict=True,
            )
            depth_pred_lg = step_out.logits[0].float()    # [n_in, V] — one pred per parent at depth d_step-1
            drf_pred_per_step.append(depth_pred_lg)
            depth_tokens = torch.topk(depth_pred_lg, 2, dim=-1).indices    # [n_in, 2]
            tokens_per_depth.append(depth_tokens.reshape(-1))    # [2 * n_in] = 2^d_step tokens
            if d_step < D:
                drf_cache = step_out.past_key_values

        # ── Assemble flat tree_tokens [N] in node-index order ──
        tree_tokens = torch.cat(tokens_per_depth, dim=0)    # [1 + 2 + 4 + ... + 2^D] = [N]

        # ── Per-node draft prediction (= prediction at parent slot) ──
        # Node i with depth d_i has parent at depth d_i - 1 (or none for root).
        # parent_idx_in_depth = parents[i] - (2^(d_i-1) - 1) (offset in parent depth's list)
        # drf_pred_per_step[d_i] is shape [2^(d_i-1), V]; row = parent_idx_in_depth.
        # For root (d_i=0): drf_root_pred_lg.
        drf_node_pred_lg_list = []
        for i in range(N):
            d_i = self.tree_depths[i]
            if d_i == 0:
                drf_node_pred_lg_list.append(drf_root_pred_lg)
            else:
                parent_node_idx = self.tree_parents[i]
                parent_offset = parent_node_idx - (2 ** (d_i - 1) - 1)
                # drf_pred_per_step[d_i] has shape [2^(d_i - 1), V] for d_i ≥ 2,
                # or [1, V] for d_i = 1 (we unsqueezed above).
                drf_node_pred_lg_list.append(
                    drf_pred_per_step[d_i][parent_offset])
        drf_node_pred_lg = torch.stack(drf_node_pred_lg_list, dim=0)    # [N, V]

        # ── Target tree verify ──
        tgt_cache0 = _slice_cache_to_bs1(tgt_keys, tgt_values, b, kv_len)
        node_positions = torch.tensor(
            [[anchor_abs + 1 + self.tree_depths[i] for i in range(N)]],
            device=device, dtype=torch.long)
        tree_mask = self._target_tree_mask(
            kv_len, device, self.target_model.dtype)
        with torch.no_grad():
            tgt_out = self.target_model(
                input_ids=tree_tokens.view(1, N),
                attention_mask=tree_mask,
                position_ids=node_positions,
                past_key_values=tgt_cache0,
                use_cache=False, return_dict=True,
            )
        # tgt_tree_lg[i] = target's prediction at slot i (= prediction of next-after-i).
        tgt_tree_lg = tgt_out.logits[0].float()    # [N, V]

        # Target per-node prediction-of-node-i-token (= parent's slot logits).
        # Root: prefix logits; non-root: tgt_tree_lg[parent(i)].
        tgt_node_pred_lg_list = []
        for i in range(N):
            if i == 0:
                tgt_node_pred_lg_list.append(tgt_root_pred_lg)
            else:
                tgt_node_pred_lg_list.append(tgt_tree_lg[self.tree_parents[i]])
        tgt_node_pred_lg = torch.stack(tgt_node_pred_lg_list, dim=0)    # [N, V]

        # ── KL per node ──
        tgt_logp = F.log_softmax(tgt_node_pred_lg, dim=-1)
        tgt_p = tgt_logp.exp()
        drf_logp = F.log_softmax(drf_node_pred_lg, dim=-1)
        kl_per_node = (tgt_p * (tgt_logp - drf_logp)).sum(-1).clamp(min=0.0)

        # ── EAL per path ──
        beta_per_node = tgt_p.gather(
            -1, tree_tokens.long().unsqueeze(-1)).squeeze(-1)    # [N]
        paths = self.paths.to(device)    # [P, D+1]
        path_beta = beta_per_node[paths]    # [P, D+1]
        eal_per_path = path_beta.cumprod(dim=-1).sum(-1)    # [P]

        return kl_per_node, eal_per_path

    def forward(self, input_ids, attention_mask, loss_mask,
                anchor_kl_coef=1.0, eal_aux_coef=0.1, **kwargs):
        B, L = input_ids.shape
        device = input_ids.device

        with torch.no_grad():
            tgt_out = self.target_model(
                input_ids=input_ids, attention_mask=attention_mask,
                use_cache=True, return_dict=True)
            tgt_prefix_logits = tgt_out.logits.float()
            tgt_keys, tgt_values = _extract_kv_lists(tgt_out.past_key_values)

        drf_out = self.draft_model(
            input_ids=input_ids, attention_mask=attention_mask,
            use_cache=True, return_dict=True)
        drf_prefix_logits = drf_out.logits.float()
        drf_keys, drf_values = _extract_kv_lists(drf_out.past_key_values)

        # ── Pick one anchor per sample ──
        anchors = []
        for b in range(B):
            valid = (loss_mask[b] == 1).nonzero(as_tuple=True)[0]
            if len(valid) == 0:
                continue
            r = int(torch.randint(0, len(valid), (1,)).item())
            anchors.append((b, int(valid[r].item())))

        if not anchors:
            z = drf_prefix_logits.sum() * 0.0
            return z, z.detach(), {
                "total_loss": 0.0, "kl_loss": 0.0, "aux_loss": 0.0,
                "eal_mean": 0.0, "num_valid": 0.0, "num_anchors": 0,
                "mean_tau": 0.0, "eagle_loss": 0.0,
            }

        all_kl = []
        all_eal = []
        for (b, ap) in anchors:
            kl_node, eal_path = self._rollout_one_anchor(
                b, ap, drf_prefix_logits, tgt_prefix_logits,
                drf_keys, drf_values, tgt_keys, tgt_values)
            all_kl.append(kl_node)
            all_eal.append(eal_path)

        kl_stack = torch.stack(all_kl, dim=0)
        eal_stack = torch.stack(all_eal, dim=0)
        mean_kl = kl_stack.mean()
        mean_eal = eal_stack.mean()
        total_loss = anchor_kl_coef * mean_kl - eal_aux_coef * mean_eal

        metrics = {
            "total_loss": total_loss.item(),
            "kl_loss": mean_kl.item(),
            "aux_loss": (-mean_eal).item(),
            "eal_mean": mean_eal.item(),
            "num_anchors": len(anchors),
            "num_valid": float(len(anchors)),
            "mean_tau": mean_eal.item(),
            "eagle_loss": mean_kl.item(),
        }
        return total_loss, total_loss.detach(), metrics
