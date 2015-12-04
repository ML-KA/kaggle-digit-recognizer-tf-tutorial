#!/usr/bin/env python

import pandas as pd


def input_kaggle(train_path, test_path):
    """
    Parameters
    ----------
    train_path : str
        Path to the train.csv
    test_path : str
        Path to the test.csv

    Returns
    -------
    """

    # train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(test_df.to_dict()[:100])

if __name__ == '__main__':
    input_kaggle('input/train.csv', 'input/test.csv')
