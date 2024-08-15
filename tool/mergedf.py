from typing import Dict
import random
import pickle
from collections import OrderedDict, defaultdict
from functools import partial
#
# import lineflow as lf
# import lineflow.datasets as lfds
# import lineflow.cross_validation as lfcv

import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, BertModel, BertConfig, DNATokenizer, BertForMaskedLM, BertForPreTraining
import argparse
import os
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import Dict, List, Tuple
from copy import deepcopy

import glob
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
import psutil
genomes = None
taxonomy = '../data/CAMI/airway/taxonomy.tsv'
contig_to_genome = '../data/CAMI/airway/reference.tsv'

contig_to_genome_df = pd.read_csv(contig_to_genome, sep='\t', header=None)
contig_to_genome_df = contig_to_genome_df.rename(columns={0: 'contig_name', 1: 'genome'})
contig_to_genome_df = contig_to_genome_df.iloc[:, :2]

taxonomy_df = pd.read_csv(taxonomy, sep='\t', header=None)
taxonomy_df = taxonomy_df.rename(columns={0: 'genome', 1: 'species', 2: 'genus'})

merged_df = pd.merge(contig_to_genome_df, taxonomy_df, how="left", on=["genome"])

GROUP_KEY = "species"

i = 0
tax_dict = dict()
species_groups = list(merged_df.groupby(GROUP_KEY))
print('111:',species_groups)

random.seed(42)
random.shuffle(species_groups)
print('222:',species_groups)
for x_name, x in species_groups:

    # Skip genome if it isn't in the list of given genomes.
    if genomes is not None and isinstance(genomes, list):
        if x_name not in genomes:
            continue
        else:
            print(x_name)

    elif genomes is not None and i >= genomes:
        break

    contigs = x['contig_name'].tolist()
    print('contigs',contigs)
    genome = x['genome'].tolist()
    print('genome',genome)
    tax_dict[x_name] = zip(contigs, genome)
    print(tax_dict)
    i += 1

#[('Enterococcus casseliflavus', 'S27C241360', 'OTU_97.9469.0'), ('Enterococcus casseliflavus', 'S27C26623', 'OTU_97.9469.0')]
flatten_dict = [(tax_name, contig_name, genome_id) for tax_name, contig_genome in tax_dict.items() for
                contig_name, genome_id in contig_genome]
print(flatten_dict)
# now that we have the validation contigs, we go through and find the sequence and tokenize it and then store it to disk ready to be read from get_item.

