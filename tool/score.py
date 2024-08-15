import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 样本数据，请用您的实际数据替换
features = ['TNF', 'Dec_TNF', 'Contig\nembedding', 'Contig embedding\n+ Dec_TNF']
metrics = ['Accuracy', 'Recall', 'F1-score']

# 示例数据，请替换为您的实际数据

data = np.array([
    [],  # 准确率
    [],  # 召回率
    []   # F1分数
])

x = np.arange(len(features))  # 特征的位置
width = 0.25  # 柱的宽度

fig, ax = plt.subplots(figsize=(10, 6), dpi=600)

# 使用更柔和的颜色
colors = ['#4878D0', '#EE854A', '#6ACC64']

# 绘制柱状图
rects = []
for i in range(3):
    rects.append(ax.bar(x + (i-1)*width, data[i], width, label=metrics[i], color=colors[i], edgecolor='black', linewidth=0))

# 添加一些文本元素
ax.set_ylabel('Scores', fontsize=20, fontweight='bold')
ax.set_title('Oral', fontsize=20, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(features, rotation=0, ha='center', fontsize=20)
# ax.legend(fontsize=12, loc='upper right')

# 设置y轴的范围和格式
ax.set_ylim(0, 0.9)  # 将y轴的范围设置为 0 到 0.8
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
ax.tick_params(axis='y', labelsize=20, length=8, width=1.5)  # 设置y轴刻度标签的字体大小

# 在柱上添加数值标签
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2%}',  # 使用 .2% 来显示百分比，保留两位小数
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12)

# for rect in rects:
#     autolabel(rect)

# 添加网格线
#ax.grid(True, linestyle='--', alpha=0.7, axis='y')

# 移除顶部和右侧的边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 调整布局
fig.tight_layout()

# 保存图片
plt.savefig('oral-feature_comparison2.png', dpi=600, bbox_inches='tight')
# # 保存图形为SVG和PDF格式
fig.savefig('oral-feature_comparison2.svg', format='svg')
fig.savefig('oral-feature_comparison2.pdf', format='pdf')

plt.show()
