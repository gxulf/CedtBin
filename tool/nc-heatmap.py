import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 创建数据框
data = {
    "CedtBin": [],
    "VAMB": [],
    "MetaBAT2": []
}
index = ["Airways", "GI", "Oral", "Skin", "Urog"]

df = pd.DataFrame(data, index=index)

# 设置Seaborn的样式
sns.set(style="whitegrid")

# 创建图形对象
fig, ax = plt.subplots(figsize=(8, 6))

# 绘制热图
heatmap = sns.heatmap(
    df,
    annot=True,
    cmap="YlGnBu",
    linewidths=.5,
    annot_kws={"size": 10},
    ax=ax,
    fmt='g'  # 使用一般格式显示数字
)

# 添加标题和轴标签
ax.set_title('NC Strains', fontsize=16,fontweight='bold')

# 调整刻度标签
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

# 调整布局
plt.tight_layout()

# 保存图像
# fig.savefig("heatmap_improved.png", dpi=600, format='png')
# fig.savefig("heatmap.svg", format='svg')

# 显示图像
plt.show()
