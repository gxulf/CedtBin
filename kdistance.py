import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from annoy import AnnoyIndex

# 加载 tnf.csv 文件
# tnf_data = pd.read_csv('../特征/dec_tnf.csv')

def kdistance(X):
    # 将数据转换为 NumPy 数组
    #X = tnf_data.values

    # 选择 k 值，通常为数据点数量的 2 到 4 倍
    k = 25

    # 计算每个点到其 k 个最近邻的距离
    nearest_neighbors = NearestNeighbors(n_neighbors=k)
    neighbors = nearest_neighbors.fit(X)
    distances, indices = neighbors.kneighbors(X)

    # 取第 k 个最近邻的距离
    k_distances = distances[:, k - 1]

    # 对距离排序
    k_distances = np.sort(k_distances)

    # 绘制 k-距离图
    plt.figure(figsize=(10, 6), dpi=600)
    plt.plot(k_distances)
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{k}-distance')
    plt.title('k-distance Graph')
    # 增加网格
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # 去掉顶部和右边框线
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.show()
    eps_value = k_distances[np.argmax(np.diff(k_distances))]
    print(f'Suggested eps value: {eps_value}')
    return eps_value*0.3


def annoy_kdistance(X):
    # 数据维度
    f = X.shape[1]

    k = 25
    n_trees = 15

    # 创建Annoy索引
    annoy_index = AnnoyIndex(f, 'euclidean')

    # 添加数据点到Annoy索引
    for i, vec in enumerate(X):
        annoy_index.add_item(i, vec)

    # 构建Annoy索引
    annoy_index.build(n_trees)

    # 计算每个数据点到其第k个最近邻的距离
    k_distances = []
    for i in range(X.shape[0]):
        distances = annoy_index.get_nns_by_item(i, k, include_distances=True)[1]
        k_distances.append(distances[-1])  # 获取第k个最近邻的距离

    # 对距离排序
    k_distances = np.sort(k_distances)

    # # 绘制k-距离图
    plt.figure(figsize=(10, 6), dpi=600)
    plt.plot(k_distances)
    plt.xlabel('Points sorted by distance')
    plt.ylabel('eps')
    plt.title('k-distance Graph')
    # 增加网格
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # 去掉顶部和右边框线
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.savefig('k_distance_plot2.png', bbox_inches='tight')
    plt.show()

    # 找到拐点
    eps_value = k_distances[np.argmax(np.diff(k_distances))]
    print(f'Suggested eps value: {eps_value}')
    return eps_value * 0.25

