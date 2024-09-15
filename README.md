CedtBin
===
CedtBin is a metagenome binning tool that uses the BERT model to train on a metagenome dataset to obtain its encoding representation and concatenate it with the decomposed tetranucleotide frequencies as the final binning feature. In terms of clustering methods, CedtBin uses the Annoy algorithm and grid search strategy based on the DBSCAN algorithm to determine relevant parameters and improve the effect of binning.

Environment setup
===
Create a Conda Environment
---
```
conda create -n CedtBin python=3.8
conda activate CedtBin
```
Install the package and other requirements
---
```
conda install pytorch torchvision cudatoolkit=11.6 -c pytorch

git clone https://github.com/gxulf/CedtBin
cd CedtBin

pip install umap-learn
pip install numpy
pip install scikit-learn matplotlib 
...
pip install pytorch_lightning==1.5.1
```

Pre-train
===
We use the metagenomic data (FASTA format) to be trained as the input of CedtBin, modify the location of the dataset in the `train.py` script, and then run the script to learn and train contigs. The learned pre-trained model can be used for the new metagenomic dataset to obtain its encoding representation.
```
cd CedtBin
./train.py
```

Binning
===
We use the new metagenomic data (FASTA format) to be binned as the input of CedtBin, and modify the location of the corresponding dataset in the script `CedtBin.py`. And load the pre-trained model. After running, the script will use the pre-trained model to encode the metagenomic dataset and process the tetranucleotide frequency using the non-negative matrix decomposition algorithm. Before the DBSCAN algorithm is clustered, the script `kdistance.py` will be called to use the Annoy algorithm to determine the relevant parameters. Finally, the binning results are obtained.
```
cd CedtBin
./CedtBin.py
```

Reference
===
* [DNABERT](https://github.com/jerryji1993/DNABERT)
* [VAMB](https://github.com/RasmussenLab/vamb)
* [Huggingface's Transformers](https://github.com/huggingface/transformers)
* [Annoy](https://github.com/spotify/annoy)

