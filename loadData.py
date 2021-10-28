import pandas as pd
import numpy as np

datasets = {
    "train": "https://storage.googleapis.com/wnut-2017_ner-shared-task/wnut17train_clean_tagged.txt",
    "dev": "https://storage.googleapis.com/wnut-2017_ner-shared-task/wnut17dev_clean_tagged.txt",
    "test": "https://storage.googleapis.com/wnut-2017_ner-shared-task/wnut17test_clean_tagged.txt"
}

def loadData(dataset):
    wnuttrain = datasets[dataset]
    train = pd.read_table(wnuttrain, header=None, names=['token', 'label', 'bio_only', 'upos']).dropna()
    return train

print(loadData("train"))
