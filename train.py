import random
import pickle
from collections import OrderedDict, defaultdict
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
import pytorch_lightning as pl
from transformers import AdamW, BertModel, BertConfig, DNATokenizer, BertForMaskedLM
import os
from copy import deepcopy
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping



random.seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

class KmerDataset(torch.utils.data.Dataset):
    def __init__(self, contigs, k=4):
        self.tokenizer = DNATokenizer.from_pretrained('./data/DNABERT/dnabert-config/bert-config-4')
        self.train_tuples = []
        #if tuples already stored, read them in - note if any of the underlying val contig samples are deleted then make sure to remove the cache or if arguments change
        # CHANGE THE CACHE ,,,,,[('S13C0', './data/CAMI/DNABERT/cached_contigs/contig_S13C0.pickle'), ('S13C1', './data/CAMI/DNABERT/cached_contigs/contig_S13C1.pickle')]
        tuple_cache_file = './data/CAMI/train_tuples.pickle'
        if os.path.exists(tuple_cache_file):
            with open(tuple_cache_file, 'rb') as fp:
                self.train_tuples = pickle.load(fp)
            return 

        #contigs are coming in as a list of paths to the samples. we need to open all of the samples and retrieve the sequences by their contig_name
        contig_list = self.create_contig_file_list(contigs)
        sequence_by_contig_name = self.file2seq(contig_list)
        
        for key, value in sequence_by_contig_name.items():
            sequence = sequence_by_contig_name[key]
            kmers = self.seq2kmer(sequence, k)
            padded_kmers = self.create_padding(kmers)
            tokenized_kmers = self.tokenize_all(padded_kmers)
            cache_file = './data/CAMI/train_cached_contigs/contig_{idx}.pickle'.format(idx=key)
            if not os.path.exists(cache_file):
                # 文件不存在，执行写入操作
                with open(cache_file, 'wb') as fp:
                    torch.save(tokenized_kmers, cache_file)
            else:
                pass  # 文件已存在，不执行任何操作
            self.train_tuples.append((key, cache_file))

        with open(tuple_cache_file, 'wb') as fp:
            pickle.dump(self.train_tuples, fp, protocol=4)
            
        print('Length of train tuples', len(self.train_tuples))

    def __getitem__(self, idx):
        #print("Getting item index", idx)
        contig_cache_tuple = self.train_tuples[idx]
        contig_file_name = contig_cache_tuple[1]
        with open(contig_file_name, 'r') as fp:
            segments = torch.load(contig_file_name)
            segment = random.choice(segments)
        return segment

    def __len__(self):
        #print("Getting length")
        return len(self.train_tuples)

    def create_contig_file_list(self, path_to_contig_file):
        print('Creating contig list from assemblies1')
        contig_list = []
        with open(path_to_contig_file, 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                line = line.rstrip()
                contig_list.append(line)
        return contig_list

    def file2seq(self, contig_list):
        print('Creating sequence1')
        seq_dict = defaultdict(str)
        for sample_file in contig_list:
            with open(sample_file, 'r') as fp:
                lines = fp.readlines()
                for line in lines:
                    if line[0] == '>':
                        key = line[1:].strip('\n')
                    else:
                        seq_dict[key] += line.strip('\n')
        return seq_dict

    def seq2kmer(self, value, k):
        print("Converting sequence to kmers1")
        #for key, value in seq_dict.items():
        kmer = [value[x:x+k] for x in range(len(value)+1-k)]
        kmers = " ".join(kmer)
        return kmers
    
    def create_padding(self, kmers):
        print('Padding the sequences1')
        kmers_split = kmers.split() 
        token_inputs = [kmers_split[i:i+512] for i in range(0, len(kmers_split), 512)]
        num_to_pad = 512 - len(token_inputs[-1])
        token_inputs[-1].extend(['[PAD]'] * num_to_pad)
        return token_inputs
    
    def tokenize_all(self, kmers_512_segments):
        print('Tokenizing1')
        tokenized_512_segments = []
        for idx, segment in enumerate(kmers_512_segments):
            tokenized_sequence = self.tokenizer.encode_plus(segment, add_special_tokens=True, max_length=512)["input_ids"]
            tokenized_sequence = torch.tensor(tokenized_sequence, dtype=torch.long)
            tokenized_512_segments.append(tokenized_sequence)
        return tokenized_512_segments
    
 
NUM_LABELS = 1
MAX_LENGTH = 512

MASK_LIST = {
            "3": [-1, 1],
            "4": [-1, 1, 2],
            "5": [-2, -1, 1, 2],
            "6": [-2, -1, 1, 2, 3]
                        }
class CedtBin(pl.LightningModule):

    def __init__(self, kmer_dataset, val_dataset):
        super(CedtBin, self).__init__()
        print("Activating CedtBin")
        # dir_to_pretrained_model = './data/DNABERT/pretrained_models/4-new-12w-0'
        #
        # config = BertConfig.from_pretrained('./data/DNABERT/pretrained_models/4-new-12w-0/config.json', output_hidden_states=True, return_dict=True)
        # self.model = BertForMaskedLM.from_pretrained(dir_to_pretrained_model, config=config)
        # # 可选：重新初始化权重（如果你想从头开始训练）
        # self.model.init_weights()  # 重新初始化模型权重
        dir_to_pretrained_model = './data/CAMI/DNABERT/pretrained_models/bert-base-uncased'

        config = BertConfig.from_pretrained(dir_to_pretrained_model, output_hidden_states=True, return_dict=True)
        self.model = BertForMaskedLM.from_pretrained(dir_to_pretrained_model, config=config)

        self._train_dataloader = DataLoader(kmer_dataset, batch_size=64, shuffle=True, num_workers=8, drop_last=True)
        self._val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8, drop_last=False)
        print('train length', len(self._train_dataloader))
        print('val length', len(self._val_dataloader))
        
        self.train_tokenizer = kmer_dataset.tokenizer
        self.val_tokenizer = val_dataset.tokenizer

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
                {
                    "param": [p for n,p in self.model.named_parameters() if not any (nd in n for nd in no_decay)],
                    "weight_decay": 0.01,
                },
                {"params": [p for n,p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            ]

        optimizer = AdamW(
                self.model.parameters(),
                lr=2e-5,
                )
        return optimizer

    def mask_tokens(self, inputs, tokenizer, mlm_probability=0.15):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

        # 随机选择一个掩码尺度（1、2、4个k-mer）
        possible_scales = [1, 2, 4]
        selected_scale = random.choice(possible_scales)

        # 根据选择的尺度调整掩码列表
        if selected_scale == 1:
            mask_list = [0]  # 仅掩码中心的k-mer
        elif selected_scale == 2:
            mask_list = [-1, 1]  # 掩码中心前后各一个k-mer
        elif selected_scale == 4:
            mask_list = [-2, -1, 1, 2]  # 掩码中心前后各两个k-mer

        if tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone().cuda()
        probability_matrix = torch.full(labels.shape, mlm_probability).cuda()
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool, device=probability_matrix.device), value=0.0)
        if tokenizer._pad_token is not None:
            padding_mask = labels.eq(tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask.cuda(), value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()

        # 修改掩码索引
        masks = deepcopy(masked_indices)
        for i, masked_index in enumerate(masks):
            end = torch.where(probability_matrix[i] != 0)[0].tolist()[-1]
            mask_centers = set(torch.where(masked_index == 1)[0].tolist())
            new_centers = deepcopy(mask_centers)
            for center in mask_centers:
                for mask_number in mask_list:
                    current_index = center + mask_number
                    if current_index <= end and current_index >= 1:
                        new_centers.add(current_index)
            new_centers = list(new_centers)
            masked_indices[i][new_centers] = True

        labels[~masked_indices] = -100  # 仅计算被掩码的token的损失

        # 80%的时间，用[MASK]代替被掩码的输入token
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool().cuda() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10%的时间，用随机词代替被掩码的输入token
        indices_random = torch.bernoulli(
            torch.full(labels.shape, 0.5)).bool().cuda() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)

        indices_random = indices_random.to("cuda")
        inputs = inputs.to("cuda")
        random_words = random_words.to("cuda")
        inputs[indices_random] = random_words[indices_random]

        # 10%的时间，保持被掩码的输入token不变
        return inputs, labels

    def training_step(self, batch, batch_idx):
        #print("Start training")
        inputs, labels = self.mask_tokens(batch, self.train_tokenizer) 
        outputs = self.model(inputs, masked_lm_labels=labels) 
        train_loss = outputs[0]  # model outputs are always tuple in transformers

        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return train_loss 
        
    # def validation_step(self, batch, batch_idx):
    #     #print("Start validation")
    #     inputs, labels = self.mask_tokens(batch[0], self.val_tokenizer)
    #     outputs = self.model(inputs, masked_lm_labels=labels)
    #     val_loss = outputs[0]
    #     hidden_states = outputs[2]
    #     embedding_output = hidden_states[0]
    #     #attention_hidden_states = hidden_states[1:]
    #     last_hidden_state = hidden_states[1]
    #     cls = torch.mean(last_hidden_state, 1)
    #     self.log('val_loss', val_loss)
    #     taxonomy_labels = batch[1]
    #
    #     return {'loss': val_loss.cpu(), 'prediction': cls.cpu(), 'taxonomy': taxonomy_labels}
    #
    # def validation_epoch_end(self, validation_step_outputs):
    #     output_folder = 'output'
    #     if not os.path.exists(output_folder):
    #         os.makedirs(output_folder)
    #     pred = [x['prediction'] for x in validation_step_outputs]
    #     combined_feature_space = torch.cat(pred)
    #     print('feature shape', combined_feature_space.shape)
    #     labels = [x['taxonomy'] for x in validation_step_outputs]
    #     new_labels= [item for t in labels for item in t]
    #
    #     pca = PCA(n_components=2)
    #     pca.fit(combined_feature_space)
    #     projection = pca.transform(combined_feature_space)
    #
    #     genome_to_color_id = {k: i for k, i in zip(sorted(set(new_labels)), range(10))}
    #     print(genome_to_color_id)
    #     genome_keys = genome_to_color_id.keys()
    #     targets = list(genome_to_color_id[x] for x in new_labels)
    #     plt.figure(figsize=(7, 7))
    #     scatter = plt.scatter(projection[:, 0], projection[:, 1], alpha=0.9, s=5.0, c=targets, cmap='tab10')
    #     plt.text(3, 2.25, 'epoch:{nr}'.format(nr = self.current_epoch), color='black', fontsize=12)
    #     plt.legend(loc="upper left", prop={'size': 6}, handles=scatter.legend_elements()[0], labels=genome_keys)
    #
    #     plt.savefig(os.path.join(output_folder, 'epoch{nr}.png'.format(nr=self.current_epoch)))
    #     plt.clf()
    #
    #     # kmeans_pca = KMeans(n_clusters = len(genome_to_color_id), init = 'k-means++', random_state=None).fit(projection)
    #     # file_name = os.path.join(output_folder, 'kmeans_{nr}.txt'.format(nr=self.current_epoch))
    #     # with open(file_name, 'w') as fw:
    #     #     print(kmeans_pca.cluster_centers_, file=fw)

    # def test_step(self, batch):
    #     return self.validation_step(self, batch)
    #
    # def test_epoch_end(self, validation_step_outputs):
    #     return self.validation_epoch_end(self, validation_step_outputs)

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader


class GenomeKmerDataset(torch.utils.data.Dataset):
    def __init__(self, contigs, random_segment=True, k=4, genomes=None, cache_name="oral_tuples"):
        self.tokenizer = DNATokenizer.from_pretrained('./data/DNABERT/dnabert-config/bert-config-4')

        # if tuples already stored, read them in - note if any of the underlying val contig samples are deleted then make sure to remove the cache or if arguments change
        tuple_cache_file = f"./data/CAMI/{cache_name}.pickle"
        if os.path.exists(tuple_cache_file):
            with open(tuple_cache_file, 'rb') as fp:
                self.tuples = pickle.load(fp)
            return

            # contigs are coming in as a list of paths to the samples. we need to open all of the samples and retrieve the sequences by their contig_name
        contig_list = self.create_contig_file_list(contigs)
        sequence_by_contig_name = self.file2seq(contig_list)

        # genome information is stored in the taxonomy and gsa mapping files. We need to join these together and then sample 10 genomes and store their respective contigs.
        taxonomy = './data/CAMI/vamb/oral/taxonomy.tsv'
        contig_to_genome = './data/CAMI/vamb/oral/reference.tsv'

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

        random.seed(42)
        random.shuffle(species_groups)
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
            genome = x['genome'].tolist()
            tax_dict[x_name] = zip(contigs, genome)
            i += 1

        # [('Enterococcus casseliflavus', 'S27C241360', 'OTU_97.9469.0'), ('Enterococcus casseliflavus', 'S27C26623', 'OTU_97.9469.0')]
        flatten_dict = [(tax_name, contig_name, genome_id) for tax_name, contig_genome in tax_dict.items() for
                        contig_name, genome_id in contig_genome]

        # now that we have the validation contigs, we go through and find the sequence and tokenize it and then store it to disk ready to be read from get_item.
        self.tuples = []
        for tax_name, contig_name, genome_id in flatten_dict:
            if contig_name not in sequence_by_contig_name:
                continue

            sequence = sequence_by_contig_name[contig_name]
            contig_length = len(sequence)
            kmers = self.seq2kmer(sequence, k)
            padded_kmers = self.create_padding(kmers)
            tokenized_kmers = self.tokenize_all(padded_kmers)
            segment = random.choice(tokenized_kmers)
            cache_file = './data/CAMI/eval_contigs_oral/{idx}.pt'.format(idx=contig_name)
            if not os.path.exists(cache_file):
                # 文件不存在，执行写入操作
                with open(cache_file, 'wb') as fp:
                    torch.save(segment, cache_file)
            else:
                pass  # 文件已存在，不执行任何操作

            self.tuples.append((tax_name, contig_name, cache_file, genome_id, contig_length))

        random.shuffle(self.tuples)

        with open(tuple_cache_file, 'wb') as fp:
            pickle.dump(self.tuples, fp, protocol=4)

        print('Length of tuples', len(self.tuples))

    def __getitem__(self, idx):
        species_contig_tuple = self.tuples[idx]
        taxonomy_id = species_contig_tuple[0]
        contig_file_name = species_contig_tuple[2]
        contig_name = species_contig_tuple[1]
        genome_id = species_contig_tuple[3]
        contig_length = species_contig_tuple[4]
        with open(contig_file_name, 'r') as fp:
            segment = torch.load(contig_file_name)
        return segment, taxonomy_id, contig_name, genome_id, contig_length

    def __len__(self):
        # print("Getting length")
        return len(self.tuples)

    def create_contig_file_list(self, path_to_contig_file):
        print('Creating contig list from assemblies')
        contig_list = []
        with open(path_to_contig_file, 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                line = line.rstrip()
                contig_list.append(line)
        return contig_list

    def file2seq(self, contig_list):
        print('Creating sequence')
        seq_dict = defaultdict(str)
        for val_file in contig_list:
            with open(val_file, 'r') as fp:
                lines = fp.readlines()
                for line in lines:
                    if line[0] == '>':
                        key = line[1:].strip('\n')
                    else:
                        seq_dict[key] += line.strip('\n')
        return seq_dict

    def seq2kmer(self, value, k):
        print("Converting sequence to kmers")
        kmer = [value[x:x + k] for x in range(len(value) + 1 - k)]
        kmers = " ".join(kmer)
        return kmers

    def create_padding(self, kmers):
        print('Padding the sequences')
        kmers_split = kmers.split()
        token_inputs = [kmers_split[i:i + 512] for i in range(0, len(kmers_split), 512)]
        num_to_pad = 512 - len(token_inputs[-1])
        token_inputs[-1].extend(['[PAD]'] * num_to_pad)
        return token_inputs

    def tokenize_all(self, kmers_512_segments):
        print('Tokenizing')
        tokenized_512_segments = []
        for idx, segment in enumerate(kmers_512_segments):
            tokenized_sequence = self.tokenizer.encode_plus(segment, add_special_tokens=True, max_length=512)[
                "input_ids"]
            tokenized_sequence = torch.tensor(tokenized_sequence, dtype=torch.long)
            tokenized_512_segments.append(tokenized_sequence)
        return tokenized_512_segments


def main():
    contigs = './data/contigs-train.txt'
    val_contigs ='./data/contigs-val.txt'

    # 创建数据集
    kmers_dataset = KmerDataset(contigs)
    val_dataset = GenomeKmerDataset(val_contigs)

    # 设置回调函数
    checkpoint_callback = ModelCheckpoint(
        save_weights_only=False,
        verbose=False,
        monitor='train_loss',
        save_top_k=1,  # 仅保存一个最佳检查点
        save_last=True,
        mode='min'
    )
    # 设置早停策略
    early_stopping_callback = EarlyStopping(
        monitor='train_loss',
        patience=3,
        mode='min'
    )
    # 创建训练器
    trainer = pl.Trainer(
        #auto_scale_batch_size='power',
        gpus=-1,
        #callbacks=[checkpoint_callback],
        callbacks=[checkpoint_callback, early_stopping_callback],  # 加入早停策略
        num_sanity_val_steps=1,  # 仅运行一个验证步骤
        max_epochs=10
    )

    # 创建模型并进行训练
    model = CedtBin(kmer_dataset=kmers_dataset, val_dataset=val_dataset)
    trainer.fit(model)
if __name__ == "__main__":
    main()
