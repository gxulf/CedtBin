import pandas as pd
import matplotlib.pyplot as plt

# 创建数据框
data = {
    "CedtBin": [],
    "VAMB": [],
    "MetaBAT2": []
}
index = ["Airways", "GI", "Oral", "Skin", "Urog"]

df = pd.DataFrame(data, index=index)

# 设置图形对象
fig, ax = plt.subplots(figsize=(10, 6))

# 定义颜色
colors = ['#4878D0', '#EE854A', '#6ACC64']

# 绘制柱状图
bar_width = 0.25
index = pd.Index(range(len(df.index)))
opacity = 1

rects1 = plt.bar(index, df['CedtBin'], bar_width, alpha=opacity, color=colors[0], label='CedtBin')
rects2 = plt.bar(index + bar_width, df['VAMB'], bar_width, alpha=opacity, color=colors[1], label='VAMB')
rects3 = plt.bar(index + 2 * bar_width, df['MetaBAT2'], bar_width, alpha=opacity, color=colors[2], label='MetaBAT2')

# 添加标题和轴标签Ratio of recovered NC strains
plt.ylabel('Number of recovered NC strains', fontsize=12 ,fontweight='bold')
plt.xticks(index + bar_width, df.index)
plt.legend()

# # 显示网格
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# 调整布局
plt.tight_layout()

# 保存图像
fig.savefig("nc-bar_chart.png", dpi=600, format='png')
fig.savefig("nc-bar_chart.svg", format='svg')

# 显示图像
plt.show()
