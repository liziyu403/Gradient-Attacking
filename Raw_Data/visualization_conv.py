import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import pickle
import numpy as np

with open('conv2_grad_eval.pickle', 'rb') as f:
    conv2_grad = pickle.load(f)
    
mean_gradients = np.abs(torch.mean(conv2_grad, dim=(1))  )

# 生成子图
num_kernels = mean_gradients.shape[0]
num_rows = 8  # 设定每行显示8个子图
num_cols = num_kernels // num_rows  # 根据卷积核数量计算列数
fig = plt.figure(figsize=(15, 15))

# 遍历每四个卷积核的平均值，并在子图中显示
for i in range(0, num_kernels, 4):
    kernel_end = min(i + 4, num_kernels)
    ax = fig.add_subplot(num_rows, num_cols, i//4 + 1, projection='3d')
    for k in range(i, kernel_end):
        x, y = torch.meshgrid(torch.arange(mean_gradients.shape[2]), torch.arange(mean_gradients.shape[1]))
        z = mean_gradients[k].numpy()
        ax.plot_surface(x, y, z, cmap='viridis', alpha=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    ax.set_title(f'Kernels {i}-{kernel_end-1}')

# 添加 colorbar
norm = mcolors.Normalize(vmin=mean_gradients.min(), vmax=mean_gradients.max())
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=fig.get_axes(), orientation='horizontal', fraction=0.05, pad=0.04)

# 添加大标题
# fig.suptitle('Average Gradients of Convolutional Kernels', fontsize=16)

plt.tight_layout()  # 调整子图布局
plt.show()
