import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt

# 加载数据集
data = pd.read_csv('../特征/tnf.csv')
# 假设数据已经被处理成合适的格式，如将特征转换为行，样本为列
X = data.values

# 计算NMF降维的肘部图和信息保留率
def plot_nmf_elbow(X, max_components=100):
    reconstruction_errors = []
    explained_variance_ratios = []

    for n_components in range(1, max_components + 1):
        nmf = NMF(n_components=n_components)
        nmf.fit(X)
        reconstruction_errors.append(nmf.reconstruction_err_)
        explained_variance_ratios.append(np.sum(nmf.components_) / np.sum(X))

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Number of components')
    ax1.set_ylabel('Reconstruction error', color=color)
    ax1.plot(range(1, max_components + 1), reconstruction_errors, color=color, marker='o', linestyle='-')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Explained variance ratio', color=color)
    ax2.plot(range(1, max_components + 1), np.cumsum(explained_variance_ratios), color=color, marker='o', linestyle='-')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('NMF Elbow Method with Reconstruction Error and Explained Variance Ratio')
    plt.show()


# 示例数据

# 绘制NMF的肘部图和信息保留率
plot_nmf_elbow(X.T, max_components=50)
