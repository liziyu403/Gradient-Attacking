from itertools import cycle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("data_ReLU_Custom_ResNet.csv")
df = pd.read_csv("data_Sigmoid_McMahan_CNN.csv")
df = pd.read_csv("data_ReLU_McMahan_CNN.csv")


sns.set(style="whitegrid")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

init_methods = [
    "kaiming_uniform_",
    "kaiming_normal_",
    "xavier_uniform_",
    "xavier_normal_",
]
colors = cycle(["Blues", "Oranges", "Greens", "Reds"])
markers = cycle(["s", "D", "^", "o"])

for init_method in init_methods:
    data_subset = df[df["Init_Method"] == init_method]
    sns.lineplot(
        ax=axes[0],
        x="Batch_Size",
        y="Gradient_Loss",
        hue="Model_Architecture",
        style="Seed",
        palette=next(colors),
        markers=next(markers),
        data=data_subset,
    )

# assert
activation = df["Activation"].unique()
assert len(activation) == 1
activation = activation[0]
model = df["Model_Architecture"].unique()
assert len(model) == 1
model = model[0]

axes[0].set_title(
    f"Batch_Size vs Gradient_Loss ({model} with {activation})".replace("_", " "),
    fontweight="bold",
)
axes[0].grid(True)

colors = cycle(["Blues", "Oranges", "Greens", "Reds"])
markers = cycle(["s", "D", "^", "o"])

for init_method in init_methods:
    data_subset = df[df["Init_Method"] == init_method]
    sns.lineplot(
        ax=axes[1],
        x="Batch_Size",
        y="PSNR",
        hue="Model_Architecture",
        style="Seed",
        palette=next(colors),
        markers=next(markers),
        data=data_subset,
    )

axes[1].set_title(
    f"Batch_Size vs Reconstruction_Loss ({model} with {activation})".replace("_", " "),
    fontweight="bold",
)
axes[1].grid(True)

colors = cycle(["#5DADE2", "#F39C12", "#58D68D", "#E74C3C"])
markers = cycle(["s", "D", "^", "o"])
legend_labels = init_methods
legend_handles = [
    plt.Line2D(
        [0],
        [0],
        marker=next(markers),
        color=next(colors),
        label=label.replace("_", " "),
    )
    for label in legend_labels
]

axes[0].legend(
    title="Init_Method".replace("_", " "),
    handles=legend_handles,
    bbox_to_anchor=(1, 1),
    fontsize="large",
)
axes[1].legend(
    title="Init_Method".replace("_", " "),
    handles=legend_handles,
    bbox_to_anchor=(1, 1),
    fontsize="large",
)

# Remove top and right spines
for ax in axes:
    ax.spines["top"].set_color("None")
    ax.spines["right"].set_color("None")

# Increase linewidth of remaining spines
for ax in axes:
    ax.spines["left"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

plt.tight_layout()
plt.show()
