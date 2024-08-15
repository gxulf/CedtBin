import numpy as np
import matplotlib.pyplot as plt

# 假设 errors 是前面计算的重构误差
k_values = range(1, 50)
errors = []

# 创建绘图
plt.figure(figsize=(10, 6), dpi=600)
plt.plot(k_values, errors, marker='o', linestyle='-', color='b', markersize=6, linewidth=2)

# 设置图形的标题和标签
plt.xlabel('Number of Components (k)', fontsize=12)
plt.ylabel('Reconstruction Error', fontsize=12)

# 设置坐标轴刻度
plt.xticks(np.arange(0, 51, 5), fontsize=10)
plt.yticks(fontsize=10)

# 增加网格
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# 优化图例
plt.legend(['Reconstruction Error'], fontsize=12)

# 增加注释（可选）
plt.annotate('Suitable Point', xy=(10, errors[9]), xytext=(15, errors[9] + 0.5),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             fontsize=12, fontstyle='italic')

# 去掉顶部和右边框线
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# 显示图形
plt.tight_layout()
# plt.savefig('nmf_plot.png', bbox_inches='tight')
plt.show()