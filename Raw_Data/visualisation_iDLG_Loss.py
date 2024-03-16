import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

data = pd.read_csv("iDLG.csv")

DLG = data[data["Label"] == "DLG"]
iDLG = data[data["Label"] == "iDLG"]

plt.scatter(
    np.random.uniform(0.9, 1.1, size=len(DLG)),
    DLG["Gradient_Loss"],
    label="DLG method",
    alpha=0.7,
    color="blue",
    s=100,
)
plt.scatter(
    np.random.uniform(1.9, 2.1, size=len(iDLG)),
    iDLG["Gradient_Loss"],
    label="iDLG method",
    alpha=0.7,
    color="orange",
    s=100,
)

# mean and variance
DLG_mean = DLG["Gradient_Loss"].mean()
DLG_std = DLG["Gradient_Loss"].std()
iDLG_mean = iDLG["Gradient_Loss"].mean()
iDLG_std = iDLG["Gradient_Loss"].std()

# draw mean and variance line
plt.errorbar(
    [1], DLG_mean, yerr=DLG_std, fmt="o", color="blue", label=" DLG Loss Mean ± Std"
)
plt.errorbar(
    [2], iDLG_mean, yerr=iDLG_std, fmt="o", color="orange", label="iDLG Loss Mean ± Std"
)

plt.xticks([1, 2], [f"DLG", f"iDLG"])

plt.xlabel("Attack Method")
plt.ylabel("Gradient Loss")
plt.title("Gradient Loss vs Attack Method")
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
