import matplotlib.pyplot as plt
import numpy as np

# 数据
recall = []
metabat2 = []
vamb = []
cedtbin = []

# 颜色
colors = ['#4878D0', '#EE854A', '#6ACC64']

# 创建图表
plt.figure(figsize=(10, 6))

# 绘制曲线
plt.plot(recall, cedtbin, color=colors[0], marker='^', label='CedtBin')
plt.plot(recall, vamb, color=colors[1], marker='s', label='VAMB')
plt.plot(recall, metabat2, color=colors[2], marker='o', label='MetaBAT2')

# 设置标题和标签
plt.title('MetaHIT', fontsize=16)
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Number of recovered NC strains', fontsize=12)

# 设置x轴刻度
plt.xticks(recall)

# 设置y轴范围
plt.ylim(0, 120)

# # 添加网格
# plt.grid(True, linestyle='--', alpha=0.7)

# 添加图例，保持原有顺序
plt.legend(loc='upper right')

# 调整布局
plt.tight_layout()

# 保存为高分辨率PNG (600 DPI)
plt.savefig('MetaHIT_plot_600dpi.png', dpi=600, format='png')

# 保存为SVG
plt.savefig('MetaHIT_plot.svg', format='svg')

# 显示图表
plt.show()