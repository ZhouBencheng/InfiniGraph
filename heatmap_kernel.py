import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.linewidth': 0.8,
})

# Data: (hd, q_tok, batch, kv_len, best_kernel: 0=warp, 1=warpcta8pipe, speedup%)
raw = [
    (128,1,1,128,1,125.6),(128,1,1,512,1,134.5),(128,1,1,1024,1,136.1),(128,1,1,2048,1,134.4),(128,1,1,4096,1,145.0),
    (128,1,4,128,0,19.4),(128,1,4,512,0,24.5),(128,1,4,1024,0,8.8),(128,1,4,2048,0,3.3),(128,1,4,4096,0,5.1),
    (128,1,8,128,0,128.2),(128,1,8,512,0,102.8),(128,1,8,1024,0,103.4),(128,1,8,2048,0,105.4),(128,1,8,4096,0,104.7),
    (128,16,1,128,0,16.6),(128,16,1,512,0,24.2),(128,16,1,1024,0,22.4),(128,16,1,2048,0,22.5),
    (128,16,4,128,1,4.2),(128,16,4,512,1,6.7),(128,16,4,1024,1,8.9),(128,16,4,2048,1,17.8),
    (128,16,8,128,0,2.0),(128,16,8,512,1,0.1),(128,16,8,1024,1,5.6),
    (128,64,1,128,0,1.3),(128,64,1,512,1,5.8),(128,64,1,1024,1,8.4),(128,64,1,2048,1,6.9),
    (128,64,4,128,0,11.0),(128,64,4,512,0,6.1),(128,64,4,1024,0,3.4),
    (128,256,1,128,0,10.7),(128,256,1,512,0,5.9),(128,256,1,1024,0,4.9),(128,256,1,2048,0,6.1),
    (128,256,4,128,0,18.7),(128,256,4,512,0,7.0),
    (128,1024,1,128,0,2.1),(128,1024,1,512,0,8.5),(128,1024,1,1024,0,8.4),
    (64,1,1,128,1,177.7),(64,1,1,512,1,218.6),(64,1,1,1024,1,224.6),(64,1,1,2048,1,227.0),(64,1,1,4096,1,228.2),
    (64,1,4,128,1,47.6),(64,1,4,512,1,50.1),(64,1,4,1024,1,53.3),(64,1,4,2048,1,66.8),
    (64,1,8,128,0,22.5),(64,1,8,512,0,26.5),(64,1,8,1024,0,13.7),(64,1,8,2048,0,9.6),
    (64,16,1,512,1,177.9),(64,16,1,2048,1,189.1),
    (64,16,4,512,1,155.4),(64,16,4,2048,1,158.7),
    (64,64,1,512,1,155.8),(64,64,1,2048,1,158.5),
    (64,64,4,512,1,133.8),
    (64,256,1,512,1,134.2),(64,256,1,2048,1,140.8),
]

def make_heatmap(ax, hd, raw_data):
    subset = [(q, b, kv, w, sp) for (h, q, b, kv, w, sp) in raw_data if h == hd]
    row_keys = sorted(set((q, b) for q, b, kv, w, sp in subset), key=lambda x: (x[0], x[1]))
    col_keys = sorted(set(kv for q, b, kv, w, sp in subset))

    row_idx = {k: i for i, k in enumerate(row_keys)}
    col_idx = {k: i for i, k in enumerate(col_keys)}
    nr, nc = len(row_keys), len(col_keys)

    mat = np.full((nr, nc), np.nan)
    winners = np.full((nr, nc), -1, dtype=int)
    sp_vals = np.full((nr, nc), np.nan)

    for q, b, kv, w, sp in subset:
        r, c = row_idx[(q, b)], col_idx[kv]
        # positive = warpcta8pipe wins, negative = warp wins
        mat[r, c] = sp if w == 1 else -sp
        winners[r, c] = w
        sp_vals[r, c] = sp

    vmax = 230
    cmap = mcolors.LinearSegmentedColormap.from_list('custom', [
        '#2166ac', '#4393c3', '#92c5de',  # strong blue -> light blue (warp)
        '#f7f7f7',                          # neutral white
        '#f4a582', '#d6604d', '#b2182b',   # light red -> strong red (pipe)
    ], N=256)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')

    # Annotate
    for i in range(nr):
        for j in range(nc):
            if np.isnan(mat[i, j]):
                ax.text(j, i, '', ha='center', va='center', fontsize=7, color='#cccccc')
                # Draw hatching for missing
                rect = mpatches.FancyBboxPatch((j - 0.5, i - 0.5), 1, 1,
                    boxstyle="square,pad=0", facecolor='#f0f0f0', edgecolor='none', zorder=0)
                ax.add_patch(rect)
            else:
                sp = sp_vals[i, j]
                intensity = abs(mat[i, j]) / vmax
                color = 'white' if intensity > 0.45 else '#333333'
                txt = f"+{sp:.0f}%"
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=8, color=color, fontweight='bold')

    # Labels
    row_labels = []
    for q, b in row_keys:
        row_labels.append(f"q={q}, bs={b}")
    col_labels = [f"{kv}" for kv in col_keys]

    ax.set_xticks(range(nc))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(nr))
    ax.set_yticklabels(row_labels, fontsize=9, fontfamily='monospace')
    ax.set_xlabel('KV Sequence Length', fontsize=11, labelpad=8)

    # Grid lines
    for i in range(nr + 1):
        ax.axhline(i - 0.5, color='white', linewidth=2)
    for j in range(nc + 1):
        ax.axvline(j - 0.5, color='white', linewidth=2)

    # Group separators by q_tok
    prev_q = None
    for idx, (q, b) in enumerate(row_keys):
        if prev_q is not None and q != prev_q:
            ax.axhline(idx - 0.5, color='#333333', linewidth=1.5, linestyle='-')
        prev_q = q

    ax.set_title(f'head_size = {hd}', fontsize=14, fontweight='bold', pad=12)
    ax.tick_params(length=0)

    return im


fig, axes = plt.subplots(1, 2, figsize=(20, 13), gridspec_kw={'width_ratios': [0.85, 1.15]})
fig.patch.set_facecolor('white')

im1 = make_heatmap(axes[0], 64, raw)
axes[0].set_ylabel('(q_tokens, batch_size)', fontsize=11, labelpad=8)
im2 = make_heatmap(axes[1], 128, raw)

# Shared colorbar
cbar_ax = fig.add_axes([0.25, 0.04, 0.50, 0.02])
cbar = fig.colorbar(im2, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Speedup of best kernel over 2nd best (%)', fontsize=11, labelpad=8)
# Custom ticks
ticks = [-200, -150, -100, -50, 0, 50, 100, 150, 200]
cbar.set_ticks(ticks)
cbar.set_ticklabels([f'warp +{abs(t)}%' if t < 0 else (f'pipe +{t}%' if t > 0 else '0%') for t in ticks])
cbar.ax.tick_params(labelsize=9)

# Legend patches
warp_patch = mpatches.Patch(color='#2166ac', label='warp wins')
pipe_patch = mpatches.Patch(color='#b2182b', label='warpcta8pipe wins')
neutral_patch = mpatches.Patch(color='#f7f7f7', edgecolor='#cccccc', label='close (<10%)')
fig.legend(handles=[warp_patch, pipe_patch, neutral_patch],
           loc='upper right', fontsize=10, framealpha=0.9,
           bbox_to_anchor=(0.97, 0.97))

fig.suptitle(
    'Iluvatar BI-V150 Paged Attention: Kernel Selection Heatmap\n'
    'num_heads=32, num_kv_heads=8, block_size=256, FP16',
    fontsize=16, fontweight='bold', y=0.99,
    color='#222222'
)

plt.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.10, wspace=0.25)
plt.savefig('/Users/lxy/lxygit/kernel_heatmap.png', dpi=180, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved to kernel_heatmap.png")
