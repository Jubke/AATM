"""
  Functions to load different datasets for AATM seminar topic I code base.
"""
from sklearn.externals.joblib import Memory
import pandas as pd

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
