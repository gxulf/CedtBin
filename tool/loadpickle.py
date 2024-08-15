import os

import torch
import pickle

# 指定 pickle 文件的路径
#file_path = './data/CAMI/DNABERT/cached_contigs/contig_S13C0.pickle'
#file_path = '../data/CAMI/DNABERT/val_tuples.pickle'
file_path = '../oral_batches.pickle'
# 使用 torch.load() 加载 pickle 文件
#data = torch.load(file_path)
if os.path.exists(file_path):
    with open(file_path, 'rb') as fp:
        data = pickle.load(fp)
        # 打印加载的数据
        print(data)

