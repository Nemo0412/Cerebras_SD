"""Print β = 2σ(gap/T) distribution for V4 vs V6 drafts on a sample batch.

β_t = 2σ((z_draft[target_argmax_t] - max_t(z_draft)) / T)  ∈ [0, 1]
   β ≈ 1 → draft argmax matches target (accept)
   β ≈ 0 → draft argmax far from target (hard reject)
   β ≈ 0.5 → borderline

For each draft, we compute:
  - per-step (γ=7 sliding window) mean β
  - per-step β histogram (10 bins)
  - cum_prod over window
  - hard_accept rate per step

Usage:
    python sdpo/diagnose_beta.py \\
        --draft-v4 /scratch/.../q8_q06_klv4_regen/state_2 \\
        --draft-v6 /scratch/.../q8_q06_klv6_regen/state_2 \\
        --num-samples 20
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_draft(path):
    for f in glob.glob(os.path.join(path, "*.bin")):
        state = torch.load(f, map_location="cpu")
        for prefix in ("draft.base.", "draft_model."):
            if any(k.startswith(prefix) for k in state):
                state = {k.removeprefix(prefix): v for k, v in state.items()}
                torch.save(state, f)
                break
    m = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16,
        attn_implementation="sdpa").cuda()
    m.eval()
    return m


@torch.no_grad()
def compute_betas(target, draft, input_ids, attention_mask, loss_mask,
                  gamma=7, temperature=0.1):
    """Return per-step β (shape [B, usable, γ]) + hard_accept + cum_prod."""
    B, L = input_ids.shape
    device = input_ids.device

    # Target forward (clean)
    tgt_out = target(input_ids=input_ids, attention_mask=attention_mask,
                     return_dict=True)
    target_lg = tgt_out.logits[:, :-1, :].float()
    target_argmax = target_lg.argmax(-1)              # [B, L-1]

    # Draft forward (clean, teacher-forced)
    d_out = draft(input_ids=input_ids, attention_mask=attention_mask,
                  return_dict=True)
    draft_lg = d_out.logits[:, :-1, :].float()

    # Per-position β
    z_t = draft_lg.gather(-1, target_argmax.unsqueeze(-1)).squeeze(-1)
    z_m = draft_lg.max(-1).values
    gap = (z_t - z_m) / temperature
    beta = 2.0 * torch.sigmoid(gap)                   # [B, L-1]
    hard_accept = (draft_lg.argmax(-1) == target_argmax).float()

    # Sliding γ-window
    Lp = L - 1
    usable = Lp - gamma + 1
    beta_stack = torch.stack(
        [beta[:, k:k + usable] for k in range(gamma)], dim=-1)        # [B, U, γ]
    accept_stack = torch.stack(
        [hard_accept[:, k:k + usable] for k in range(gamma)], dim=-1)
    cum_prod = torch.cumprod(beta_stack, dim=-1)

    # Mask to valid windows (loss_mask[1:1+usable])
    win_mask = loss_mask[:, 1:1 + usable].bool()      # [B, U]

    return beta_stack, accept_stack, cum_prod, win_mask


def summarize(name, beta_stack, accept_stack, cum_prod, win_mask, gamma=7):
    """Print per-step statistics averaged over all valid windows."""
    # Flatten to [N_valid_windows, γ]
    mask_flat = win_mask.reshape(-1)
    N = int(mask_flat.sum().item())
    if N == 0:
        print(f"[{name}] No valid windows.")
        return

    B, U, G = beta_stack.shape
    beta_flat = beta_stack.reshape(B * U, G)[mask_flat.reshape(-1)]     # [N, γ]
    acc_flat = accept_stack.reshape(B * U, G)[mask_flat.reshape(-1)]
    cp_flat = cum_prod.reshape(B * U, G)[mask_flat.reshape(-1)]

    print(f"\n━━━━ {name}  (N={N} valid γ-windows) ━━━━")
    print(f"{'step k':<8} | {'mean β':<8} | {'median β':<8} | {'β>0.9':<7} | "
          f"{'β<0.1':<7} | {'hard_acc':<9} | {'cum_prod':<9}")
    print("-" * 78)
    for k in range(gamma):
        b = beta_flat[:, k]
        acc = acc_flat[:, k].mean().item()
        cp = cp_flat[:, k].mean().item()
        hi = (b > 0.9).float().mean().item()
        lo = (b < 0.1).float().mean().item()
        print(f"  step {k:<3} | {b.mean().item():<8.4f} | "
              f"{b.median().item():<8.4f} | {hi:<7.3f} | {lo:<7.3f} | "
              f"{acc:<9.4f} | {cp:<9.4f}")

    # β distribution histogram per step (10 bins [0, 1])
    print(f"\n  β distribution histogram (per step, 10 bins over [0,1])")
    print(f"  {'step':<6}  " + " ".join(f"{f'[{i/10:.1f}-{(i+1)/10:.1f}]':<8}"
                                         for i in range(10)))
    for k in range(gamma):
        b = beta_flat[:, k].cpu().numpy()
        hist, _ = np.histogram(b, bins=10, range=(0, 1))
        row = " ".join(f"{h/N:<8.3f}" for h in hist)
        print(f"  {k:<6}  {row}")


def compare(v4_stats, v6_stats, gamma=7):
    """Side-by-side comparison of β statistics."""
    (b4, a4, cp4, m4) = v4_stats
    (b6, a6, cp6, m6) = v6_stats
    B, U, G = b4.shape
    m4f = m4.reshape(-1)
    m6f = m6.reshape(-1)
    N = int(m4f.sum().item())   # should equal m6f (same mask)

    b4f = b4.reshape(B * U, G)[m4f.reshape(-1)]
    b6f = b6.reshape(B * U, G)[m6f.reshape(-1)]
    cp4f = cp4.reshape(B * U, G)[m4f.reshape(-1)]
    cp6f = cp6.reshape(B * U, G)[m6f.reshape(-1)]
    a4f = a4.reshape(B * U, G)[m4f.reshape(-1)]
    a6f = a6.reshape(B * U, G)[m6f.reshape(-1)]

    print(f"\n━━━━ V4 vs V6 side-by-side  (N={N}) ━━━━")
    print(f"{'step':<6} | {'β V4 / V6':<18} | {'β>0.9 V4/V6':<15} | "
          f"{'hard_acc V4/V6':<16} | {'cum_prod V4/V6':<18}")
    print("-" * 85)
    for k in range(gamma):
        b4k = b4f[:, k].mean().item()
        b6k = b6f[:, k].mean().item()
        hi4 = (b4f[:, k] > 0.9).float().mean().item()
        hi6 = (b6f[:, k] > 0.9).float().mean().item()
        ac4 = a4f[:, k].mean().item()
        ac6 = a6f[:, k].mean().item()
        cp4k = cp4f[:, k].mean().item()
        cp6k = cp6f[:, k].mean().item()
        print(f"  {k:<4} | {b4k:.3f} / {b6k:.3f}      | "
              f"{hi4:.3f} / {hi6:.3f}    | {ac4:.3f} / {ac6:.3f}      | "
              f"{cp4k:.3f} / {cp6k:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target', default='Qwen/Qwen3-8B')
    p.add_argument('--draft-v4',
                   default='/scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b/q8_q06_klv4_regen/state_2')
    p.add_argument('--draft-v6',
                   default='/scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b/q8_q06_klv6_regen/state_2')
    p.add_argument('--data-path',
                   default='/scratch/yf3005/gto_data/sharegpt_qwen3_8b_regen.jsonl')
    p.add_argument('--num-samples', type=int, default=20)
    p.add_argument('--max-len', type=int, default=1024)
    p.add_argument('--gamma', type=int, default=7)
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--plot', action='store_true', default=True,
                   help='save PNG plots (default: on)')
    p.add_argument('--plot-dir', default='smalllm_tree_eval_results/beta_diagnose',
                   help='directory for plots')
    args = p.parse_args()

    print(f"Loading target {args.target} + tokenizer")
    tok = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.float16,
        attn_implementation="sdpa").cuda()
    target.eval()

    print(f"Loading V4 draft: {args.draft_v4}")
    draft_v4 = load_draft(args.draft_v4)
    print(f"Loading V6 draft: {args.draft_v6}")
    draft_v6 = load_draft(args.draft_v6)

    print(f"\nLoading data, preparing {args.num_samples} samples...")
    ds = load_dataset('json', data_files=args.data_path)['train']
    ds = ds.shuffle(seed=42).select(range(args.num_samples * 3))

    v4_agg = [[], [], [], []]
    v6_agg = [[], [], [], []]
    count = 0
    for ex in ds:
        src = ex['conversations']
        if not src or src[0]['from'] != 'human':
            continue
        msgs = [{"role": "user" if s['from'] == 'human' else "assistant",
                 "content": s['value']} for s in src[:4]]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=False,
                                           enable_thinking=True)
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=False)
        ids = tok(text, return_tensors="pt",
                  add_special_tokens=False).input_ids
        if ids.shape[1] < args.gamma + 10 or ids.shape[1] > args.max_len:
            continue
        ids = ids.cuda()
        attn = torch.ones_like(ids)
        # simple loss mask: everything after the first user turn
        # For diagnostic we just mask everything (look at all positions)
        loss_mask = torch.ones_like(ids)
        loss_mask[:, 0] = 0

        b4 = compute_betas(target, draft_v4, ids, attn, loss_mask,
                           args.gamma, args.temperature)
        b6 = compute_betas(target, draft_v6, ids, attn, loss_mask,
                           args.gamma, args.temperature)
        for i in range(4):
            v4_agg[i].append(b4[i])
            v6_agg[i].append(b6[i])
        count += 1
        if count >= args.num_samples:
            break

    # Concatenate along batch dimension (unequal seq length → pad by padding beta_stack)
    def cat_stats(agg):
        # each element: [1, U_i, γ] or [1, U_i]. Different U_i per sample.
        # Flatten all windows and concatenate
        bs, accs, cps, ms = agg
        B, U, G = bs[0].shape
        # just concatenate all windows across samples (U dim varies)
        beta_list = []
        acc_list = []
        cp_list = []
        m_list = []
        for b, a, c, m in zip(bs, accs, cps, ms):
            # b shape [1, U_i, γ] → [U_i, γ]
            beta_list.append(b.reshape(-1, G))
            acc_list.append(a.reshape(-1, G))
            cp_list.append(c.reshape(-1, G))
            m_list.append(m.reshape(-1))
        beta_all = torch.cat(beta_list, dim=0).unsqueeze(0)  # [1, totalU, γ]
        acc_all = torch.cat(acc_list, dim=0).unsqueeze(0)
        cp_all = torch.cat(cp_list, dim=0).unsqueeze(0)
        m_all = torch.cat(m_list, dim=0).unsqueeze(0)
        return beta_all, acc_all, cp_all, m_all

    v4_stats = cat_stats(v4_agg)
    v6_stats = cat_stats(v6_agg)

    summarize("V4 (klv4)", *v4_stats, gamma=args.gamma)
    summarize("V6 (klv6)", *v6_stats, gamma=args.gamma)
    compare(v4_stats, v6_stats, gamma=args.gamma)

    if args.plot:
        save_plots(v4_stats, v6_stats, args.gamma, args.plot_dir)


def save_plots(v4_stats, v6_stats, gamma, outdir):
    """Save PNGs: β histograms, per-step bars, cum_prod trajectory."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)

    def flatten(stats):
        b, a, cp, m = stats
        B, U, G = b.shape
        mf = m.reshape(-1)
        bf = b.reshape(B * U, G)[mf.reshape(-1)].cpu().numpy()
        af = a.reshape(B * U, G)[mf.reshape(-1)].cpu().numpy()
        cpf = cp.reshape(B * U, G)[mf.reshape(-1)].cpu().numpy()
        return bf, af, cpf

    v4_b, v4_a, v4_cp = flatten(v4_stats)
    v6_b, v6_a, v6_cp = flatten(v6_stats)

    # ── Plot 1: β histogram (log scale) per step, V4 vs V6 overlay ──
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
    axes = axes.flatten()
    for k in range(gamma):
        ax = axes[k]
        bins = np.linspace(0, 1, 21)
        ax.hist(v4_b[:, k], bins=bins, alpha=0.5, label='V4 (klv4)',
                color='#1f77b4', edgecolor='black', linewidth=0.3)
        ax.hist(v6_b[:, k], bins=bins, alpha=0.5, label='V6 (klv6)',
                color='#ff7f0e', edgecolor='black', linewidth=0.3)
        ax.set_title(f'step {k}  β distribution', fontsize=10)
        ax.set_xlabel('β = 2σ(gap/T)')
        if k % 4 == 0:
            ax.set_ylabel('count')
        ax.set_yscale('log')
        if k == 0:
            ax.legend(fontsize=9)
    axes[-1].axis('off')
    fig.suptitle('β distribution per step — V4 vs V6 (log scale)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{outdir}/beta_hist_per_step.png', dpi=130, bbox_inches='tight')
    plt.close()

    # ── Plot 2: stacked mass bars (β<0.1 / middle / β>0.9) per step ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, bf, name, color_lo, color_hi in [
        (axes[0], v4_b, 'V4', '#d62728', '#2ca02c'),
        (axes[1], v6_b, 'V6', '#d62728', '#2ca02c')]:
        lo = (bf < 0.1).mean(axis=0)
        hi = (bf > 0.9).mean(axis=0)
        mid = 1 - lo - hi
        ks = np.arange(gamma)
        ax.bar(ks, hi, label='β>0.9 (accept)', color=color_hi)
        ax.bar(ks, mid, bottom=hi, label='0.1≤β≤0.9 (mid)', color='#aaaaaa')
        ax.bar(ks, lo, bottom=hi + mid, label='β<0.1 (reject)', color=color_lo)
        for k in ks:
            ax.text(k, hi[k] - 0.03, f'{hi[k]:.3f}', ha='center', fontsize=8,
                    color='white', fontweight='bold')
            ax.text(k, hi[k] + mid[k] + lo[k] - 0.03, f'{lo[k]:.3f}',
                    ha='center', fontsize=8, color='white', fontweight='bold')
        ax.set_xlabel('step k')
        ax.set_title(f'{name}: β mass decomposition', fontsize=11)
        ax.set_xticks(ks)
        ax.legend(loc='center right', fontsize=8)
    axes[0].set_ylabel('fraction of windows')
    fig.suptitle('β bimodal mass (accept / mid / reject) per step',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{outdir}/beta_mass_decomp.png', dpi=130, bbox_inches='tight')
    plt.close()

    # ── Plot 3: cum_prod trajectory + β mean trajectory ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ks = np.arange(gamma)
    axes[0].plot(ks, v4_b.mean(0), 'o-', label='V4', color='#1f77b4', lw=2)
    axes[0].plot(ks, v6_b.mean(0), 's-', label='V6', color='#ff7f0e', lw=2)
    axes[0].set_xlabel('step k'); axes[0].set_ylabel('mean β')
    axes[0].set_title('Per-step mean β', fontsize=11)
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_xticks(ks)

    axes[1].plot(ks, v4_cp.mean(0), 'o-', label='V4 cum_prod', color='#1f77b4', lw=2)
    axes[1].plot(ks, v6_cp.mean(0), 's-', label='V6 cum_prod', color='#ff7f0e', lw=2)
    axes[1].set_xlabel('step k'); axes[1].set_ylabel('mean cum_prod')
    axes[1].set_title(r'cum_prod = $\prod_{j \leq k} \beta_j$', fontsize=11)
    axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_xticks(ks)
    # add relative diff
    ratios = v6_cp.mean(0) / v4_cp.mean(0)
    ax2 = axes[1].twinx()
    ax2.plot(ks, (1 - ratios) * 100, 'v--', color='#d62728', alpha=0.6,
             label='V6 deficit %')
    ax2.set_ylabel('V6 deficit vs V4 (%)', color='#d62728')
    ax2.tick_params(axis='y', labelcolor='#d62728')
    ax2.legend(loc='upper right', fontsize=8)

    fig.suptitle('V4 vs V6 — β and cum_prod trajectory',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{outdir}/beta_cumprod_traj.png', dpi=130, bbox_inches='tight')
    plt.close()

    # ── Plot 4: V4 β vs V6 β scatter (per position, step 0 as representative) ──
    # Use step 0 only (all steps give similar scatter due to teacher-forcing)
    b4_s0 = v4_b[:, 0]
    b6_s0 = v6_b[:, 0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: scatter with jitter so density is visible
    ax = axes[0]
    jitter = 0.01
    n_points = len(b4_s0)
    rng = np.random.default_rng(0)
    x_j = b4_s0 + rng.normal(0, jitter, n_points)
    y_j = b6_s0 + rng.normal(0, jitter, n_points)
    ax.scatter(x_j, y_j, s=2, alpha=0.15, color='#333333')
    ax.plot([0, 1], [0, 1], '--', color='red', lw=1.5, label='y = x')
    ax.set_xlabel('V4 β (step 0)'); ax.set_ylabel('V6 β (step 0)')
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_title(f'V4 β  vs  V6 β  (N={n_points} positions)')
    ax.legend()
    ax.grid(alpha=0.3)

    # Right: 2D heatmap of disagreement
    ax = axes[1]
    bins = np.linspace(0, 1, 21)
    h, xe, ye = np.histogram2d(b4_s0, b6_s0, bins=[bins, bins])
    # Log-scale heatmap
    h_log = np.log10(h + 1)
    im = ax.imshow(h_log.T, origin='lower', aspect='equal',
                   extent=[0, 1, 0, 1], cmap='viridis')
    ax.plot([0, 1], [0, 1], '--', color='red', lw=1.5)
    ax.set_xlabel('V4 β'); ax.set_ylabel('V6 β')
    ax.set_title('2D histogram (log scale)')
    plt.colorbar(im, ax=ax, label='log10(count+1)')

    # Annotate quadrants
    nA = ((b4_s0 > 0.9) & (b6_s0 > 0.9)).sum()
    nR = ((b4_s0 < 0.1) & (b6_s0 < 0.1)).sum()
    nV4onlyA = ((b4_s0 > 0.9) & (b6_s0 < 0.1)).sum()
    nV6onlyA = ((b4_s0 < 0.1) & (b6_s0 > 0.9)).sum()
    axes[0].text(0.95, 0.95, f'both accept: {nA}\nV4 only acc: {nV4onlyA}\n'
                 f'V6 only acc: {nV6onlyA}\nboth reject: {nR}',
                 transform=axes[0].transAxes, ha='right', va='top',
                 fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    fig.suptitle('V4 vs V6 β disagreement  (step 0, per position)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{outdir}/beta_scatter_v4_v6.png', dpi=130, bbox_inches='tight')
    plt.close()

    # ── Plot 5: τ (acceptance length per window) histogram, V4 vs V6 ──
    # τ = cumsum(hard_accept) until first False, capped at γ
    # = count of leading 1s in accept_stack row
    def compute_tau(af):
        # af shape [N, γ], int {0,1}
        # find first 0 index; if no 0, τ = γ
        first_zero = np.argmax(1 - af, axis=1)  # returns 0 if all zero
        all_accept = af.all(axis=1)
        tau = np.where(all_accept, gamma, first_zero)
        # but if first position is 0, tau=0 (first_zero=0 and not all_accept)
        # argmax on [0,...] returns 0, that's τ=0 correctly
        return tau

    tau_v4 = compute_tau(v4_a.astype(int))
    tau_v6 = compute_tau(v6_a.astype(int))

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.arange(gamma + 2) - 0.4
    ax.hist(tau_v4, bins=bins, alpha=0.5, label=f'V4 (mean τ={tau_v4.mean():.3f})',
            color='#1f77b4', edgecolor='black', width=0.8, align='mid')
    ax.hist(tau_v6, bins=bins, alpha=0.5, label=f'V6 (mean τ={tau_v6.mean():.3f})',
            color='#ff7f0e', edgecolor='black', width=0.8, align='mid')
    # numerical annotation
    for t in range(gamma + 1):
        n4 = (tau_v4 == t).sum()
        n6 = (tau_v6 == t).sum()
        ax.text(t, max(n4, n6) + len(tau_v4) * 0.005,
                f'V4:{n4/len(tau_v4):.3f}\nV6:{n6/len(tau_v4):.3f}',
                ha='center', fontsize=7)
    ax.set_xlabel('τ (acceptance length of γ-window)')
    ax.set_ylabel('window count')
    ax.set_title(f'τ distribution (γ={gamma}, N={len(tau_v4)} windows)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(range(gamma + 1))
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{outdir}/tau_hist.png', dpi=130, bbox_inches='tight')
    plt.close()

    # ── Plot 6: Gap (z_target - z_max) distribution — focus on borderline zone ──
    # gap = logit( (1-β/2)/(1+β/2) ) * T — but easier to derive from β:
    # β = 2σ(gap/T) → gap/T = log(β / (2 - β))
    # → gap = T * log(β / (2 - β))
    T = 0.1
    def beta_to_gap(b):
        b_safe = np.clip(b, 1e-6, 2 - 1e-6)
        return T * np.log(b_safe / (2 - b_safe))

    gap_v4 = beta_to_gap(v4_b[:, 0])
    gap_v6 = beta_to_gap(v6_b[:, 0])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Full range
    ax = axes[0]
    bins = np.linspace(min(gap_v4.min(), gap_v6.min()),
                       max(gap_v4.max(), gap_v6.max()), 60)
    ax.hist(gap_v4, bins=bins, alpha=0.5, label='V4', color='#1f77b4',
            edgecolor='black', linewidth=0.3)
    ax.hist(gap_v6, bins=bins, alpha=0.5, label='V6', color='#ff7f0e',
            edgecolor='black', linewidth=0.3)
    ax.axvline(0, color='red', linestyle='--', lw=1, label='gap=0 (accept boundary)')
    ax.set_xlabel('gap = z_target - z_max (logit space)')
    ax.set_ylabel('count')
    ax.set_title('Full gap distribution', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_yscale('log')

    # Zoom into borderline region [-1, 0.1]
    ax = axes[1]
    bins = np.linspace(-1.0, 0.1, 50)
    # Clip values outside this range
    gv4_clip = gap_v4[(gap_v4 >= -1.0) & (gap_v4 <= 0.1)]
    gv6_clip = gap_v6[(gap_v6 >= -1.0) & (gap_v6 <= 0.1)]
    ax.hist(gv4_clip, bins=bins, alpha=0.5,
            label=f'V4 (N={len(gv4_clip)})', color='#1f77b4',
            edgecolor='black', linewidth=0.3)
    ax.hist(gv6_clip, bins=bins, alpha=0.5,
            label=f'V6 (N={len(gv6_clip)})', color='#ff7f0e',
            edgecolor='black', linewidth=0.3)
    ax.axvline(0, color='red', linestyle='--', lw=1)
    ax.set_xlabel('gap (zoomed borderline region)')
    ax.set_ylabel('count')
    ax.set_title('Borderline zone [-1, 0.1] — where V6 should win if design worked',
                 fontsize=10)
    ax.legend(fontsize=9)

    fig.suptitle('Logit gap distribution — V4 vs V6 (step 0)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{outdir}/gap_hist.png', dpi=130, bbox_inches='tight')
    plt.close()

    print(f"\nSaved 6 PNGs to {outdir}/:")
    print(f"  beta_hist_per_step.png   — β distribution histogram per step (log y)")
    print(f"  beta_mass_decomp.png     — β>0.9 / mid / β<0.1 stacked bars")
    print(f"  beta_cumprod_traj.png    — mean β + cum_prod trajectory with V6 deficit %")
    print(f"  beta_scatter_v4_v6.png   — V4 β vs V6 β scatter + quadrant counts")
    print(f"  tau_hist.png             — τ (acceptance length) histogram")
    print(f"  gap_hist.png             — raw logit gap distribution + borderline zoom")


if __name__ == "__main__":
    main()
