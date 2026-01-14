import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# load your per-sample AP/mIoU arrays
our = np.load("/home/yun/MIC-BEV_Official/data_dumping/mic_bbox_ap.npy")
bev = np.load("/home/yun/MIC-BEV_Official/data_dumping/bevformer_bbox_ap.npy")

# compute stats
stat, p = wilcoxon(our, bev)
n = len(our)
median_bev = np.mean(bev)
iqr_bev    = np.percentile(bev, [25,75])
median_our = np.mean(our)
iqr_our    = np.percentile(our, [25,75])
r = stat / (n*(n+1)/2)   # rank-biserial effect size

print(f"N={n}, median(our)= {median_our:.3f} [{iqr_our[0]:.3f}–{iqr_our[1]:.3f}], "
      f"median(bev)= {median_bev:.3f} [{iqr_bev[0]:.3f}–{iqr_bev[1]:.3f}]")
print(f"Wilcoxon W={stat}, p={p:.3e}, r={r:.2f}")

# boxplot & save
plt.figure(figsize=(6,4))
data = [bev, our]
plt.boxplot(data, labels=["BEVFormer","Our model"], showfliers=False)
plt.scatter(np.zeros(n)+1, bev, alpha=0.4)
plt.scatter(np.ones(n)+1, our, alpha=0.4)
for i in range(n):
    plt.plot([1,2], [bev[i], our[i]], color='gray', alpha=0.3, linewidth=0.5)
plt.ylabel("Per-sample mIoU")
plt.title("Wilcoxon Signed-Rank Comparison")
plt.tight_layout()

# **Save to file instead of plt.show()**
output_path = "/home/yun/MIC-BEV_Official/data_dumping/wilcoxon_map_comparison.png"
plt.savefig(output_path, dpi=300)
print(f"Figure saved to {output_path}")
