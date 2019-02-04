"""
  Functions to load different datasets for AATM seminar topic I code base.
"""
from sklearn.externals.joblib import Memory
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
def load_constructed_dataset(num=None, base='.//datasets//constructed'):
    if num:
        path = f'{base}_{num}.csv'
    else:
        path = aatm_support.last_file(base, '.csv')

    return pd.read_csv(
        path,
        sep = ',',
        header = 0,
        index_col = 0
    )

def load_book_summary_1():
    return load_constructed_dataset(1)

def load_book_summary_2():
    return load_constructed_dataset(2)
