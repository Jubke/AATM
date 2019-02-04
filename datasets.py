"""
  Functions to load different datasets for AATM seminar topic I code base.
"""
import os
import json
from sklearn.externals.joblib import Memory
import numpy as np
import pandas as pd
import aatm_support

MEMORY = Memory(location='./tmp', verbose=0)

@MEMORY.cache
def load_book_summary_raw():
    """ Returns the original CMU Book Summary Dataset. """
    return pd.read_csv(
        './datasets/booksummaries/booksummaries.txt',
        header=None,
        sep='\t',
        names=[
            'wiki_id', 'firebase_id', 'title',
            'author', 'pub_date', 'genres', 'plot'
        ],
        dtype={
            'wiki_id': 'uint32',
            'author': 'category'
        }
    )

# helper to load datasets later
@MEMORY.cache
def load_constructed_dataset(num=None, base='.//datasets//book_summary'):
    if num:
        path = f'{base}_{num}.csv'
    else:
        path = aatm_support.last_file(base, '.csv')

    return pd.read_csv(
        path,
        sep=',',
        header=0,
        index_col=0
    )

def load_book_summary_1():
    return load_constructed_dataset(1)

def load_book_summary_2():
    return load_constructed_dataset(2)


# helper to load feature sets later
@MEMORY.cache
def load_feature_selection(name=None, base='.//Features//selected_features'):
    if name:
        path = f'{base}_{name}.txt'
    else:
        path = aatm_support.last_file(base)

    return pd.read_csv(
        path,
        sep=',',
        header=0,
        index_col=0
    )

# helper to load feaure sets later
def load_extracted_features(name=None, base='.//Features//ext_features'):
    if name:
        path = f'{base}_{name}.txt'
    else:
        path = aatm_support.last_file(base)

    data = pd.read_csv(
        path,
        sep=',',
        header=0,
        index_col=0
    )

    data[data == np.inf] = np.nan
    data.fillna(0, inplace=True)

    return data

def load_pan_data(name='training'):
    inputDir = f'./datasets/pan18/pan18-style-change-detection-{name}-dataset-2018-01-31'

    files = os.listdir(inputDir)
    data = []
    i = 0

    for file in files:
        # for text files only (ignoring truth files)
        if file.endswith(".txt"):
            filePrefix = os.path.splitext(file)[0]
            textFileName = inputDir + "/" + filePrefix + ".txt"
            truthFileName = inputDir + "/" + filePrefix + ".truth"

            sample = ['', 0]

            with open(textFileName) as textFile:
                sample[0] = textFile.read()

            with open(truthFileName) as truthFile:
                sample[1] = int(json.load(truthFile)['changes'])

            data.append(sample)

            i += 1

    return pd.DataFrame(data, columns=['text', 'label'])
