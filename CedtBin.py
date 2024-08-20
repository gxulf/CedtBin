import math
import random

from memory_profiler import memory_usage
from sklearn.metrics.cluster import pair_confusion_matrix
from sklearn import metrics
import pytorch_lightning as pl
from collections import defaultdict
import vamb.vambtools as _vambtools
from tqdm import tqdm
from umap import UMAP
import sys
import hdbscan
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import pickle
import csv
import itertools
from train import GenomeKmerDataset, CedtBin
import argparse
import psutil
import pathlib
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
import vamb
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import time
import tool.kdistance as kdistance

random.seed(42)
torch.multiprocessing.set_sharing_strategy('file_system')

def evaluate(model, data_loader, name, file_name, layer, num_batches=None, visualize=True,
             contig_range=None, contig_file=None):
    # 定义文件路径
    pickle_file = 'oral_batches.pickle'

    # 如果pickle文件存在，直接读取
    if os.path.exists(pickle_file):
        new_labels = []
        genome_id = []
        contig_names = []

        collapsed_hidden_state = np.empty((len(data_loader.dataset), 768), dtype='float32')
        with open(pickle_file, 'rb') as fr:
            try:
                i = 0
                while True:
                    step_result = pickle.load(fr)
                    new_labels.extend(step_result["taxonomy"])
                    collapsed_hidden_state[i:i + data_loader.batch_size] = step_result["collapsed_hidden_state"]
                    genome_id.extend(step_result["genome_id"])
                    contig_names.extend(step_result["contig_names"])
                    i += data_loader.batch_size
            except EOFError:
                pass
    else:
        # pickle文件不存在，执行数据处理并保存
        t = tqdm(enumerate(data_loader), total=len(data_loader))
        for i, batch in t:
            print(f"{i}/{len(data_loader)}")

            if num_batches is not None and i >= num_batches:
                break

            outputs = model(batch[0].cuda())

            memory = psutil.virtual_memory().percent
            t.set_description(f"cpu_mem: {memory}")

            selected_hidden_state = outputs[1][layer].detach().cpu().numpy()
            collapsed_hidden_state = np.mean(selected_hidden_state, axis=-2)
            taxonomy_labels = batch[1]
            contig_names = batch[2]
            genome_id = batch[3]
            contig_length = batch[4]

            with open(pickle_file, 'ab') as fp:
                pickle.dump({
                    'collapsed_hidden_state': collapsed_hidden_state,
                    'taxonomy': taxonomy_labels,
                    'contig_names': contig_names,
                    'genome_id': genome_id,
                    'contig_length': contig_length
                }, fp)

        new_labels = []
        genome_id = []
        contig_names = []

        collapsed_hidden_state = np.empty((len(data_loader.dataset), 768), dtype='float32')
        with open(pickle_file, 'rb') as fr:
            try:
                i = 0
                while True:
                    step_result = pickle.load(fr)
                    new_labels.extend(step_result["taxonomy"])
                    collapsed_hidden_state[i:i + data_loader.batch_size] = step_result["collapsed_hidden_state"]
                    genome_id.extend(step_result["genome_id"])
                    contig_names.extend(step_result["contig_names"])
                    i += data_loader.batch_size
            except EOFError:
                pass

    print('len of collapsed_hidden_state', len(collapsed_hidden_state))
    print('len of contig names', len(contig_names))
    print('len of genome ids', len(set(genome_id)))
    print('len of labels', len(set(new_labels)))


    # #concatenate Eigen
    tnf = get_tnf(data_loader, contig_file)
    print(tnf.shape)
    # 使用 NMF 进行矩阵分解
    nmf = NMF(n_components=10, init='nndsvd', random_state=0)
    W = nmf.fit_transform(tnf.T)
    H = nmf.components_
    dec_tnf = H.T
    # dec_tnf = scaler.fit_transform(nmf.fit_transform(tnf))
    #dec_tnf = nmf.fit_transform(tnf)
    print('nmf分解完成')
    
    # collapsed_hidden_state2 = scaler.fit_transform(collapsed_hidden_state)
    # new_feature = np.concatenate((collapsed_hidden_state, dec_tnf), axis=1)
    # new_feature2 = scaler.fit_transform(new_feature)

    hidden_state_i = layer
    projection_dims = [2] if visualize else [24]  # [5, 15, 50, 75, 103, 200, 500]

    #for feature in [tnf, dec_tnf, collapsed_hidden_state, new_feature]:
    for feature in [collapsed_hidden_state]:
        for j in projection_dims:
            for projection_method in ["umap"]:  # ["umap"]:
                if projection_method == "pca":
                    pca = PCA(n_components=j, random_state=1)
                    pca.fit(feature)
                    projection = pca.transform(feature)
                elif projection_method == "umap":
                    umap = UMAP(
                        n_neighbors=40,
                        min_dist=0.3,
                        n_components=j,
                        random_state=42,
                    )
                    projection = umap.fit_transform(feature)
                elif projection_method == "none":
                    projection = feature
                sorted_labels = sorted(set(new_labels))
                genome_to_color_id = {k: i for i, k in enumerate(sorted_labels)}
                genome_keys = genome_to_color_id.keys()
                targets = list(genome_to_color_id[x] for x in new_labels)
                # if visualize:
                #     plt.figure(figsize=(7, 7))
                #     scatter = plt.scatter(projection[:, 0], projection[:, 1], alpha=0.9, s=5.0, c=targets, cmap='tab10')
                #     plt.legend(loc="upper left", prop={'size': 6}, handles=scatter.legend_elements()[0], labels=genome_keys)
                #     plt.axis('off')
                #
                #     os.makedirs(name, exist_ok=True)
                #     plt.savefig(f"{name}/viz_{hidden_state_i}_{projection_method}.png")
        print('降维完成')
        projection = np.concatenate((projection, dec_tnf), axis=1)
        # projection = scaler.fit_transform(projection)

        #k_centers = [int(len(set(genome_id)) * 0.3)]
        k_centers = [int(len(set(new_labels))),int(len(set(genome_id)) * 0.4)]
        k_methods = ["kmeans", "kmedoids"]#, "kmedoids"
        k_combos = itertools.product(k_methods, k_centers)

        cluster_sizes = [5, 10, 15, 20, 25]
        methods = ["dbscan"]  # , ""hdbscan",vamb_clustering", "dbscan"]

        cluster_combos = itertools.product(methods, cluster_sizes)
        #cluster_combos = [(method, size) for method, size in cluster_combos if method != "vamb_clustering" or size == 5]
        # print(cluster_combos)
        # methods = list(k_combos) + list(cluster_combos)
        methods = list(cluster_combos)
        print('methods', methods)
        for cluster_method, k in methods:
            print('')
            method_name = f"{cluster_method}_{k}_fn_layer{hidden_state_i}_{projection_method}{j}"
            print(method_name)

            if cluster_method == "kmeans":
                cluster_results = KMeans(n_clusters=k, init='k-means++', random_state=1).fit_predict(projection)
                clusters = defaultdict(list)
                for i, x in enumerate(cluster_results):
                    clusters[x].append(contig_names[i])
            elif cluster_method == "kmedoids":
                cluster_results = KMedoids(n_clusters=k, random_state=1).fit_predict(projection)
                clusters = defaultdict(list)
                for i, x in enumerate(cluster_results):
                    clusters[x].append(contig_names[i])
            elif cluster_method == "hdbscan":
                cluster_results = hdbscan.HDBSCAN(min_cluster_size=k).fit(projection).labels_
                clusters = defaultdict(list)
                for i, x in enumerate(cluster_results):
                    clusters[x].append(contig_names[i])
                # print('clusters', clusters)
                if -1 in clusters:
                    del clusters[-1]  # Remove "no bin" from dbscan
                #print('clusters', clusters)
                # 将聚类结果保存到TSV文件中
                with open('output/Cedt_{k}_clusters.tsv'.format(k=k), mode='w', newline='') as file:
                    writer = csv.writer(file, delimiter='\t')
                    for cluster_id, names in clusters.items():
                        label = names[0]  # 使用聚类中的第一个元素作为标签
                        for name in names:
                            writer.writerow([label, name])
                print('Cluster results have been saved to clusters.tsv')
            elif cluster_method == "dbscan":
                dbscan = DBSCAN(eps=kdistance.annoy_kdistance(projection), min_samples=k)
                cluster_results = dbscan.fit_predict(projection)
                clusters = defaultdict(list)
                for i, x in enumerate(cluster_results):
                    clusters[x].append(contig_names[i])
                # print('clusters', clusters)
                if -1 in clusters:
                    del clusters[-1]  # Remove "no bin" from dbscan
                # print('clusters', clusters)
                # 将聚类结果保存到TSV文件中
                with open('output/Cedt_{k}_clusters.tsv'.format(k=k), mode='w', newline='') as file:
                    writer = csv.writer(file, delimiter='\t')
                    for cluster_id, names in clusters.items():
                        label = names[0]  # 使用聚类中的第一个元素作为标签
                        for name in names:
                            writer.writerow([label, name])
                print('Cluster results have been saved to clusters.tsv')
            # elif cluster_method == "vamb_clustering":
            #     filtered_labels = [n for n in contig_names]
            #     projection = projection.astype(np.float32)
            #     vamb_clusters = vamb.cluster.cluster(projection, labels=filtered_labels)
            #     vamb_results = dict(vamb_clusters)
            #
            #     cluster_results_list = []
            #     for contig in filtered_labels:
            #         for cluster_key, cluster_value in vamb_results.items():
            #             if contig in cluster_value:
            #                 cluster_results_list.append(cluster_key)
            #                 break  # 找到对应的聚类结果后跳出循环，减少不必要的遍历
            #     # print(cluster_results_list)
            #     # 创建一个字典，将 contig 的聚类结果映射为数字
            #     cluster_to_number = {cluster: number for number, cluster in enumerate(set(cluster_results_list))}
            #     # 将原始的聚类结果列表映射为数字列表
            #     # print(cluster_to_number)
            #     cluster_results = [cluster_to_number[cluster] for cluster in cluster_results_list]
            #     # 打印结果
            #     # print(cluster_results)
            #
            #     sorted_labels = sorted(set(filtered_labels))
            #     genome_to_color_id = {k: i for i, k in enumerate(sorted_labels)}
            #     genome_keys = genome_to_color_id.keys()
            #     # cluster_results = list(genome_to_color_id[x] for x in filtered_labels)

            noise_indices = list(i for i in range(len(cluster_results)) if cluster_results[i] == -1)
            filtered_noise_targets = [x for i, x in enumerate(targets) if i not in noise_indices]
            filtered_noise_clusters = [x for i, x in enumerate(cluster_results) if i not in noise_indices]
            # print('targets', len(filtered_noise_targets))

            paired_confusion_matrix = pair_confusion_matrix(filtered_noise_clusters, filtered_noise_targets)
            confusion_matrix = [value for sublist in paired_confusion_matrix for value in sublist]

            tn = confusion_matrix[0]
            fn = confusion_matrix[2]
            fp = confusion_matrix[1]
            tp = confusion_matrix[3]

            recall = tp / (tp + fn)
            print('recall', recall)

            precision = tp / (tp + fp)
            print('precision', precision)

            # acc = (tp + tn) / (tp + fp + fn + tn)
            # print('acc', acc)

            f1 = 2 * tp / (2 * tp + fp + fn)
            print('f1', f1)

            if visualize:
                plt.figure(figsize=(7, 7))
                scatter = plt.scatter(projection[:, 0], projection[:, 1], alpha=0.9, s=5.0, c=cluster_results,
                                      cmap='tab10')
                plt.axis('off')
                os.makedirs(name, exist_ok=True)

                plt.savefig(f"{name}/viz_{hidden_state_i}_{projection_method}_{cluster_method}_{k}_cluster.png")
                plt.clf()

            with open('output/results_{n}_{x}.csv'.format(n=contig_range, x=file_name), 'a') as f:
                writer = csv.writer(f)
                writer.writerow([method_name] + [recall] + [precision])

            with open('output/f1_{n}_{x}.csv'.format(n=contig_range, x=file_name), 'a') as f:
                writer = csv.writer(f)
                writer.writerow([method_name] + [f1])

def get_tnf(dataloader, contig_file):
    if os.path.isfile('tnf.npy'):
        # 如果文件存在，直接读取
        tnf = np.load('tnf.npy')
        print("读取已保存的数组")
        return tnf
    else:
        # Retrive map from contig name to genome.
        contig_name_to_genome = {}
        species_to_contig_name = defaultdict(list)
        genomes_set = set()
        for batch in dataloader:
            taxonomy_labels = batch[1]
            contig_names = batch[2]
            genome_id = batch[3]

            for name, taxonomy, genome in zip(contig_names, taxonomy_labels, genome_id):
                contig_name_to_genome[name] = taxonomy
                species_to_contig_name[taxonomy].append(name)
                genomes_set.add(genome)

        print("tnf: dataset length", len(dataloader.dataset))

        file_list = create_contig_file_list(contig_file)

        # 处理所有 contig 文件，提取 tnf 频率等信息
        for fasta in file_list:
            with vamb.vambtools.Reader(fasta, 'rb') as tnffile:
                tnfs, contignames, contiglengths = vamb.parsecontigs.read_contigs(tnffile)

        tuple_of_tnfs_to_contig = list(zip(tnfs, contignames))

        # Get all species labels for each tnf
        filtered_tuples = [t for t in tuple_of_tnfs_to_contig if t[1] in contig_name_to_genome]

        tnf = []
        for t in filtered_tuples:
            tnf.append(t[0])
        tnf = np.array(tnf)
        # 保存数组到文件
        np.save('tnf.npy', tnf)
        return tnf

def plot(features, targets, legend_labels, name):
    pca = PCA(n_components=2)
    pca.fit(features)
    projection = pca.transform(features)

    plt.figure(figsize=(7, 7))
    scatter = plt.scatter(
        projection[:, 0],
        projection[:, 1],
        alpha=0.9,
        s=5.0,
        c=targets,
        cmap='tab10'
      )
    plt.legend(
        loc="upper left",
        prop={'size': 6},
        handles=scatter.legend_elements()[0],
        labels=legend_labels
    )

    plt.savefig(f"{name}.png")
    plt.clf()

def evaluate_tnf(dataloader, contig_file, file_name, contig_range=None):

    # Retrive map from contig name to genome.
    new_labels = []
    contig_name_to_genome = {}
    species_to_contig_name = defaultdict(list)
    genomes_set = set()
    for batch in dataloader:
        taxonomy_labels = batch[1]
        contig_names = batch[2]
        genome_id = batch[3]

        for name, taxonomy, genome in zip(contig_names, taxonomy_labels, genome_id):
            contig_name_to_genome[name] = taxonomy
            new_labels.extend(["taxonomy"])
            species_to_contig_name[taxonomy].append(name)
            genomes_set.add(genome)

    print("tnf: dataset length", len(dataloader.dataset))

    file_list = create_contig_file_list(contig_file)

    # 处理所有 contig 文件，提取 tnf 频率等信息
    for fasta in file_list:
        with vamb.vambtools.Reader(fasta, 'rb') as tnffile:
            tnfs, contignames, contiglengths = vamb.parsecontigs.read_contigs(tnffile)

    print("evaluate_tnf: tnfs length:", len(contignames))

    tuple_of_tnfs_to_contig = list(zip(tnfs, contignames))

    # Get all species labels for each tnf
    filtered_tuples = [t for t in tuple_of_tnfs_to_contig if t[1] in contig_name_to_genome]

    species_labels = []
    tnf = []
    for t in filtered_tuples:
        tnf.append(t[0])
        species_labels.append(t[1])
    tnf = np.array(tnf)
    # 保存数组到文件
    np.save('tnf.npy', tnf)

    scaler = StandardScaler()
    # tnf2 = scaler.fit_transform(tnf)
    nmf = NMF(n_components=6, init='nndsvd', random_state=0)
    # dec_tnf = scaler.fit_transform(nmf.fit_transform(tnf))
    W = nmf.fit_transform(tnf.T)
    H = nmf.components_
    dec_tnf = H.T

    # Convert all species labels to unique int id
    species_to_id = {k: i for i, k in enumerate(sorted(set(species_labels)))}
    species_id_labels = [species_to_id[x] for x in species_labels]
    print('len species', len(species_id_labels))

    # # Plot 10 genomes
    # index_for_contig = {}
    # for i in range(len(contignames)):
    #   index_for_contig[contignames[i]] = i
    #
    # tnfs_to_plot = []
    # species_label = []
    # plot_contigs = []
    # for species in SPECIES_TO_PLOT:
    #     contig_names = species_to_contig_name[species]
    #     for contig_name in contig_names:
    #         if contig_name in index_for_contig:
    #             contig_tnfs = tnfs[index_for_contig[contig_name]]
    #             tnfs_to_plot.append(contig_tnfs)
    #             species_label.append(species)
    #             plot_contigs.append(contig_name)
    #
    # tnfs_to_plot = np.stack(tnfs_to_plot)
    #
    # genome_to_color_id = {k: i for k, i in zip(sorted(set(species_label)), range(10))}
    # genome_keys = genome_to_color_id.keys()
    # targets = list(genome_to_color_id[x] for x in species_label)
    # plot(tnfs_to_plot, targets, genome_keys, name="tnf_gt_{n}_{dataset}".format(n=contig_range, dataset=file_name))

    # Cluster tnfs.
    #hdbscan_clusters = hdbscan.HDBSCAN(min_cluster_size=10).fit(tnf).labels_

    # Create reference file.
    to_evaluate_idxes = [
        i for i in range(len(contignames))
        if contignames[i] in contig_name_to_genome
    ]
    contigs_for_genome = defaultdict(list)

    genome_id = [contig_name_to_genome[contignames[i]] for i in to_evaluate_idxes]
    contignames = [contignames[i] for i in to_evaluate_idxes]
    contiglengths = [contiglengths[i] for i in to_evaluate_idxes]
    for contig_name, genome, contiglength in zip(
            contignames,
            genome_id,
            contiglengths
    ):
        contig = vamb.benchmark.Contig.subjectless(contig_name, contiglength)
        contigs_for_genome[genome].append(contig)

    genomes = []
    for genome_instance in set(genome_id):
        genome = vamb.benchmark.Genome(genome_instance)
        for contig_name in contigs_for_genome[genome_instance]:
            genome.add(contig_name)

        genomes.append(genome)

    print("Number of genomes:", len(genomes))

    # for genome in genomes:
    #     genome.update_breadth()
    #
    # reference = vamb.benchmark.Reference(genomes)
    #
    # taxonomy_path = './data/CAMI/oral/taxonomy.tsv'
    # with open(taxonomy_path) as taxonomy_file:
    #     reference.load_tax_file(taxonomy_file)

    # hdb_tnf_clusters = hdbscan.HDBSCAN(min_cluster_size=5).fit(tnf2).labels_
    # hdb_dec_tnf_clusters = hdbscan.HDBSCAN(min_cluster_size=5).fit(dec_tnf).labels_
    #
    # print("Finished hdb clustering")

    dbscan = DBSCAN(eps=kdistance.annoy_kdistance(tnf), min_samples=20)
    db_tnf_clusters = dbscan.fit_predict(tnf)
    dec_dbscan = DBSCAN(eps=kdistance.annoy_kdistance(dec_tnf), min_samples=20)
    db_dec_tnf_clusters = dec_dbscan.fit_predict(dec_tnf)

    print("Finished clustering")

    # hdb_tnf_bins = metrics.rand_score(species_id_labels, hdb_tnf_clusters)
    # print('hdb_tnf rand score', hdb_tnf_bins)
    #
    # hdb_dec_tnf_bins = metrics.rand_score(species_id_labels, hdb_dec_tnf_clusters)
    # print('hdb_dec_tnf rand score', hdb_dec_tnf_bins)

    tnf_bins = metrics.rand_score(species_id_labels, db_tnf_clusters)
    print('db_tnf rand score', tnf_bins)

    dec_tnf_bins = metrics.rand_score(species_id_labels, db_dec_tnf_clusters)
    print('db_dec_tnf rand score', dec_tnf_bins)

    with open(f'output/tnf_results.csv', 'a') as f:
        writer = csv.writer(f)
        # writer.writerow(["rand score", "hdb_tnf", hdb_tnf_bins, "hdb_dec_tnf", hdb_dec_tnf_bins, "db_tnf", tnf_bins, "db_dec_tnf", dec_tnf_bins, ])
        writer.writerow(["rand score", "db_tnf", tnf_bins, "db_dec_tnf", dec_tnf_bins])

    cluster_methods = {
        "db_tnf": db_tnf_clusters,
        "db_dec_tnf": db_dec_tnf_clusters
    }

    for method, clusters in cluster_methods.items():
        cluster_dict = defaultdict(list)

        for i, x in enumerate(clusters):
            cluster_dict[x].append(contignames[i])

        noise_indices = list(i for i in range(len(clusters)) if clusters[i] == -1)
        filtered_noise_targets = [x for i, x in enumerate(species_id_labels) if i not in noise_indices]
        filtered_noise_clusters = [x for i, x in enumerate(clusters) if i not in noise_indices]
        print(f"{method} clusters", len(filtered_noise_clusters))
        print(f"{method} clusters num", len(set(clusters)))

        paired_confusion_matrix = pair_confusion_matrix(filtered_noise_clusters, filtered_noise_targets)
        confusion_matrix = [value for sublist in paired_confusion_matrix for value in sublist]

        tn = confusion_matrix[0]
        fn = confusion_matrix[2]
        fp = confusion_matrix[1]
        tp = confusion_matrix[3]

        recall = tp / (tp + fn)
        print('recall', recall)

        precision = tp / (tp + fp)
        print('precision', precision)

        # acc = (tp + tn) / (tp + fp + fn + tn)
        # print('acc', acc)

        f1 = 2 * tp / (2 * tp + fp + fn)
        print('f1', f1)

        # Dynamically get the appropriate bins variable
        # Save results.
        with open(f'output/tnf_results.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([f"{method} acc", acc])

def create_contig_file_list(path_to_contig_file):
    contig_list = []
    with open(path_to_contig_file, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            line = line.rstrip()
            contig_list.append(line)
    return contig_list


def main():
    file_name = "oral"
    ckpt_path = 'lightning_logs/version_5/checkpoints/last.ckpt'
    contigs = './data/CAMI/oral.txt'

    max_contig_length = 0

    dataset = GenomeKmerDataset(contigs, cache_name="oral", genomes=None, random_segment=False)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8, drop_last=False)

    print('val length', len(dataloader))

    model = CedtBin.load_from_checkpoint(ckpt_path, kmer_dataset=dataset, val_dataset=dataset).cuda()
    model.eval()

    evaluate_tnf(dataloader, contigs, file_name=file_name, contig_range = max_contig_length)
    evaluate(model, dataloader, name=f"out/viz{pathlib.Path(ckpt_path).stem}", file_name=file_name, layer=12, num_batches=None, visualize = False, contig_range = max_contig_length, contig_file=contigs)

if __name__ == "__main__":
    main()
