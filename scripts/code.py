import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

labels = [
    'R6: 32 tasks\ntemp=0, Jaccard',
    'R7-A: 32 tasks\ntemp=0, Semantic',
    'R7-B: 8 tasks\ntemp=0.7, Semantic'
]

# Replace these with your actual Fisher z p-values from your output
pvals = [0.044, 0.007, 0.002]
neg_log_p = [-np.log10(p) for p in pvals]
threshold = -np.log10(0.05)

colors = ['#E07B39', '#5A9E6F', '#C0392B']

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(labels, neg_log_p, color=colors, width=0.45, zorder=3)
ax.axhline(threshold, color='#333333', linestyle='--', linewidth=1.2,
           label='$p = 0.05$ threshold', zorder=4)

for bar, p in zip(bars, pvals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.04,
            f'$p = {p:.3f}$', ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='#222222')

ax.set_ylabel('$-\\log_{10}(p)$ (higher = more significant)', fontsize=11)
ax.set_title('H2.3 Dissociation significance across three independent tests',
             fontsize=12, fontweight='bold')
ax.set_ylim(0, 3.5)
ax.yaxis.grid(True, linestyle=':', alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.legend(fontsize=10, loc='upper left')
plt.tight_layout()
plt.savefig('dissociation_progression.png', dpi=150, bbox_inches='tight')