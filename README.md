# Document-Embedding-Extractor
====
# Prerequisites
* Python 3.6 or above
* sklearn
* swifter
* nltk
* tqdm
* numpy
* pandas
* Pytorch 1.10 or above
* transformers
* sentence-transformers
* pytorch_metric_learning

# project structure (Updating)
```
├── run.py            # the main program + extract document matrixs from original text
├── model.py          # model definition
├── trainer.py        # optimize the model by predefined loss
├── evaluate.py       # compute Recall@K and visualize the embedding space
└── utils.py          # data processing and other utilities
```

# Usage
## Using default arguments
```shell
python run.py 
```
## Using self-defined arguments
```shell
python run.py --backbone nli-bert-base --pooling_ops max --batch_size 32 --margin 0.05 --is_debug False --device 0
```
