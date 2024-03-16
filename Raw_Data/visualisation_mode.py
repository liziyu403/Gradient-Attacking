import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

data = pd.read_csv("Mode.csv")

train_data = data[data["Mode"] == "model.train()"]
eval_data = data[data["Mode"] == "model.eval()"]

plt.scatter(
    np.random.uniform(0.9, 1.1, size=len(train_data)),
    train_data["Gradient_Loss"],
    label="Train Data",
    alpha=0.5,
    color="blue",
    s=100,
)
plt.scatter(
    np.random.uniform(1.9, 2.1, size=len(eval_data)),
    eval_data["Gradient_Loss"],
    label="Eval Data",
    alpha=0.5,
    color="orange",
    s=100,
)

# mean and variance
train_mean = train_data["Gradient_Loss"].mean()
train_std = train_data["Gradient_Loss"].std()
eval_mean = eval_data["Gradient_Loss"].mean()
eval_std = eval_data["Gradient_Loss"].std()

eval_mean_zero = eval_data["Zero_Percentage"].mean()
train_mean_zero = train_data["Zero_Percentage"].mean()

# draw mean and variance line
plt.errorbar(
    [1], train_mean, yerr=train_std, fmt="o", color="blue", label="Train Mean ± Std"
)
plt.errorbar(
    [2], eval_mean, yerr=eval_std, fmt="o", color="orange", label="Eval Mean ± Std"
)

plt.xticks(
    [1, 2],
    [
        f"model.train() \n {round(train_mean_zero,1)}% zero gradient",
        f"model.eval() \n {round(eval_mean_zero,1)}% zero gradient",
    ],
)

plt.xlabel("ModelMode")
plt.ylabel("Gradient Loss")
plt.title("Gradient Loss vs Mode")
plt.legend()

plt.grid(True)

# import figures
train_img = plt.imread("F2.png")
eval_img = plt.imread("F1.png")

imagebox_train = OffsetImage(train_img, zoom=0.15)
imagebox_eval = OffsetImage(eval_img, zoom=0.15)

ab_train = AnnotationBbox(imagebox_train, (1 + 0.2, train_mean), frameon=False)
ab_eval = AnnotationBbox(imagebox_eval, (2 - 0.2, eval_mean), frameon=False)

plt.gca().add_artist(ab_train)
plt.gca().add_artist(ab_eval)

plt.tight_layout()
plt.show()
