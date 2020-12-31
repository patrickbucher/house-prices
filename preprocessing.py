#!/usr/bin/env python3

import pandas as pd
import numpy as np

n_polynomials = 15

category_cols = [
    'MSSubClass',
    'MSZoning',
    'Street',
    'Alley',
    'LotShape',
    'LandContour',
    'Utilities',
    'LotConfig',
    'LandSlope',
    'Neighborhood',
    'Condition1',
    'Condition2',
    'BldgType',
    'HouseStyle',
    'RoofStyle',
    'RoofMatl',
    'Exterior1st',
    'Exterior2nd',
    'MasVnrType',
    'Foundation',
    'BsmtCond',
    'BsmtExposure',
    'BsmtFinType1',
    'BsmtFinType2',
    'Heating',
    'Electrical',
    'Functional',
    'GarageType',
    'GarageFinish',
    'PavedDrive',
    'MiscFeature',
    'SaleType',
    'SaleCondition',
]

scalar_cols = [
    'LotArea',
    'LotFrontage',
    'OverallQual',
    'OverallCond',
    'YearBuilt',
    'YearRemodAdd',
    'GrLivArea',
    'ExterQual',
    'ExterCond',
    'BsmtQual',
    'CentralAir',
    'MasVnrArea',
    'BsmtFinSF1',
    'BsmtFinSF2',
    'BsmtUnfSF',
    'TotalBsmtSF',
    'HeatingQC',
    '1stFlrSF',
    '2ndFlrSF',
    'LowQualFinSF',
    'BsmtFullBath',
    'BsmtHalfBath',
    'FullBath',
    'HalfBath',
    'BedroomAbvGr',
    'KitchenAbvGr',
    'KitchenQual',
    'TotRmsAbvGrd',
    'Fireplaces',
    'FireplaceQu',
    'GarageYrBlt',
    'GarageCars',
    'GarageArea',
    'GarageQual',
    'GarageCond',
    'WoodDeckSF',
    'OpenPorchSF',
    'EnclosedPorch',
    '3SsnPorch',
    'ScreenPorch',
    'PoolArea',
    'PoolQC',
    'Fence',
    'MiscVal',
    'MoSold',
    'YrSold',
]

input_cols = category_cols + scalar_cols


def get_training_data(csv_path):
    data = pd.read_csv(csv_path)
    data = data.loc[:,input_cols+['SalePrice']]
    data = augment_features(data)
    data, mappings = weigh_features(data, category_cols, 'SalePrice')
    # print(data.columns[data.isna().any()].tolist())
    data = add_polynomials(data, input_cols, n_polynomials)
    Y = data.loc[:,'SalePrice'].to_numpy()
    data.drop(['SalePrice'], axis=1, inplace=True)
    X, Y = filter_normal(data, Y, [0.05, 0.95])
    return X, Y, mappings


def get_test_data(csv_path, mappings):
    data = pd.read_csv(csv_path)
    id_col = data.loc[:,['Id']].to_numpy()
    data = data.loc[:,input_cols]
    data = augment_features(data)
    for x_col, mapping in mappings.items():
        data[x_col] = data[x_col].map(mapping).fillna(0)
    data = add_polynomials(data, input_cols, n_polynomials)
    data = data.fillna(0).to_numpy()
    return data, id_col


def filter_normal(X, Y, quantiles):
    low, high = np.quantile(Y, quantiles)
    ok_indices = (Y > low) & (Y < high)
    Y = Y[ok_indices]
    X = X[ok_indices]
    return X, Y


def augment_features(df):
    quality = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0}
    quality_na = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
    df['ExterQual'] = df['ExterQual'].map(quality)
    df['ExterCond'] = df['ExterCond'].map(quality)
    df['BsmtQual'] = df['BsmtQual'].map(
        {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
    )
    df['CentralAir'] = df['CentralAir'].map({'N': 0, 'Y': 1})
    df['HeatingQC'] = df['HeatingQC'].map(
        {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    )
    df['KitchenQual'] = df['KitchenQual'].map(quality)
    df['FireplaceQu'] = df['FireplaceQu'].map(quality_na)
    df['GarageQual'] = df['GarageQual'].map(quality_na)
    df['GarageCond'] = df['GarageCond'].map(quality_na)
    df['PoolQC'] = df['PoolQC'].map({'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'NA': 0})
    df['Fence'] = df['Fence'].map({'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'NA': 0})

    for fill_col in scalar_cols:
        df[fill_col] = df[fill_col].fillna(df[fill_col].mean())

    return df


def add_polynomials(df, columns, max_grade):
    for col in columns:
        for p in range(2, max_grade+1):
            df[f'{col}^{p}'] = df[col] ** p
    return df


def normalize(arr):
    return (arr - arr.mean()) / arr.std()


def weigh_features(df, x_cols, y_col):
    mappings = {}
    for x_col in x_cols:
        aggregate = df.groupby(x_col).aggregate({
            y_col: 'mean'
        })
        mapping = aggregate.to_dict()[y_col]
        df[x_col] = df[x_col].map(mapping).fillna(0)
        mappings[x_col] = mapping

    return df, mappings


if __name__ == '__main__':
    training_data, labels, mappings = get_training_data('data/train.csv')
    test_data, id_col = get_test_data('data/test.csv', mappings)
