CedtBin
===
CedtBin is a metagenome binning tool that uses the BERT model to train on a metagenome dataset to obtain its encoding representation and concatenate it with the decomposed tetranucleotide frequencies as the final binning feature. In terms of clustering methods, CedtBin uses the Annoy algorithm based on the DBSCAN algorithm to determine relevant parameters and improve the binning effect.

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
We use the metagenomic data in FASTA format as the input of CedtBin, modify the location of the dataset in the `train.py` script, and then run the script to learn and train contigs. The learned pre-trained model can be used for the new metagenomic dataset to obtain its encoding representation.
```
cd CedtBin
./train.py
```


