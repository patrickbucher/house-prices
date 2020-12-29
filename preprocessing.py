#!/usr/bin/env python3

import pandas as pd
import numpy as np


input_cols = [
    'LotArea',
    'LotFrontage',
    'OverallQual',
    'OverallCond',
    'YearBuilt',
    'GrLivArea',
    'MSZoning',
    'ExterQual',
    'ExterCond',
    'BsmtQual',
    'CentralAir',
]

def get_training_data(csv_path):
    data = pd.read_csv(csv_path)
    data = data.loc[:,input_cols+['SalePrice']].dropna()
    data = augment_features(data)
    data = add_polynomials(data, input_cols, 5)
    Y = data.loc[:,'SalePrice'].to_numpy()
    data.drop(['SalePrice'], axis=1, inplace=True)
    X, Y = filter_normal(data, Y, [0.05, 0.95])
    X = normalize(X).to_numpy()
    return X, Y


def get_test_data(csv_path):
    data = pd.read_csv(csv_path)
    id_col = data.loc[:,['Id']].to_numpy()
    data = augment_features(data)
    data = add_polynomials(data, input_cols, 5)
    data = data.loc[:,input_cols]
    data = normalize(test_data.fillna(0)).to_numpy()
    return data, id_col


def filter_normal(X, Y, quantiles):
    low, high = np.quantile(Y, quantiles)
    ok_indices = (Y > low) & (Y < high)
    Y = Y[ok_indices]
    X = X[ok_indices]
    return X, Y


def augment_features(df):
    df['MSZoning'] = df['MSZoning'].map(
        {'A': 0, 'C': 0, 'FV': 0, 'I': -10, 'RH': 10, 'RL': 50, 'RP': 100, 'RM': 25}
    ).fillna(0)

    exterior = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0}
    df['ExterQual'] = df['ExterQual'].map(exterior).fillna(0)
    df['ExterCond'] = df['ExterCond'].map(exterior).fillna(0)

    df['BsmtQual'] = df['BsmtQual'].map(
        {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
    ).fillna(0)

    df['CentralAir'] = df['CentralAir'].map({'N': 0, 'Y': 1}).fillna(0)

    return df


def add_polynomials(df, columns, max_grade):
    for col in columns:
        for p in range(2,max_grade+1):
            df[f'{col}{p}'] = df[col] ** p
    return df


def normalize(arr):
    return (arr - arr.mean()) / arr.std()


def rank_feature(df, x_col, y_col):
    # group df by x_col
    # calculate average of y_col
    pass

if __name__ == '__main__':
    data  = pd.read_csv('data/train.csv')
