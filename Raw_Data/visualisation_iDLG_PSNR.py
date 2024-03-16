import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

data = pd.read_csv("iDLG.csv")

DLG = data[data["Label"] == "DLG"]
iDLG = data[data["Label"] == "iDLG"]

plt.scatter(
    np.random.uniform(0.9, 1.1, size=len(DLG)),
    DLG["PSNR"],
    label="DLG method",
    alpha=0.7,
    color="blue",
    s=100,
)
plt.scatter(
    np.random.uniform(1.9, 2.1, size=len(iDLG)),
    iDLG["PSNR"],
    label="iDLG method",
    alpha=0.7,
    color="orange",
    s=100,
)

# mean and variance
DLG_mean = DLG["PSNR"].mean()
DLG_std = DLG["PSNR"].std()
iDLG_mean = iDLG["PSNR"].mean()
iDLG_std = iDLG["PSNR"].std()

# draw mean and variance line
plt.errorbar(
    [1], DLG_mean, yerr=DLG_std, fmt="o", color="blue", label=" DLG PSNR Mean ± Std"
)
plt.errorbar(
    [2], iDLG_mean, yerr=iDLG_std, fmt="o", color="orange", label="iDLG PSNR Mean ± Std"
)

plt.xticks([1, 2], [f"DLG", f"iDLG"])

plt.xlabel("Attack Method")
plt.ylabel("PSNR")
plt.title("PSNR vs Attack Method")
plt.legend()

plt.grid(True)

# # import figures
# train_img = plt.imread("F2.png")
# eval_img = plt.imread("F1.png")

# imagebox_train = OffsetImage(train_img, zoom=0.15)
# imagebox_eval = OffsetImage(eval_img, zoom=0.15)

# ab_train = AnnotationBbox(imagebox_train, (1 +0.2, train_mean), frameon=False)
# ab_eval = AnnotationBbox(imagebox_eval, (2 - 0.2, eval_mean), frameon=False)

# plt.gca().add_artist(ab_train)
# plt.gca().add_artist(ab_eval)

plt.tight_layout()
plt.show()
