#!/usr/bin/env python3

import numpy as np
import pandas as pd


def augment(df):
    df['MSZoning'] = df['MSZoning'].map({'A': 0, 'C': 0, 'FV': 0, 'I': -10, 'RH': 10, 'RL': 50, 'RP': 100, 'RM': 25}).fillna(0)
    exterior = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0}
    df['ExterQual'] = df['ExterQual'].map(exterior).fillna(0)
    df['ExterCond'] = df['ExterCond'].map(exterior).fillna(0)
    df['BsmtQual'] = df['BsmtQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}).fillna(0)
    df['CentralAir'] = df['CentralAir'].map({'N': 0, 'Y': 1}).fillna(0)

def normalize(arr):
    return (arr - arr.mean()) / arr.std()


def add_polynomials(df, columns, max_grade=5):
    for col in columns:
        for p in range(2,max_grade+1):
            df[f'{col}{p}'] = df[col] ** p


data = pd.read_csv('data/train.csv')
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
data = data.loc[:,input_cols+['SalePrice']].dropna()
augment(data)

Y = data.loc[:,'SalePrice'].to_numpy()

# only retain "normal" 80%
low, high = np.quantile(Y, q=[0.1, 0.9])
ok_indices = (Y > low) & (Y < high)
Y = Y[ok_indices]

data.drop(['SalePrice'], axis=1, inplace=True)

add_polynomials(data, input_cols)

X_norm = normalize(data).to_numpy()
X_norm = X_norm[ok_indices,:]

theta = np.random.rand(X_norm.shape[1])
alpha = 0.1
lmbda = 1e-6
m = len(X_norm)
print(f'training with {m} examples')
iterations = int(1e6)
batch = iterations / 10

for i in range(iterations):
    pred = X_norm.dot(theta)
    diff = pred - Y
    grad = diff.dot(X_norm)
    theta -= (alpha/m) * (grad + (lmbda/m) * theta.sum())
    cost = 1/(2*m) * (diff ** 2).sum() + (lmbda/(2*m)) * (theta ** 2).sum()
    if i % batch == batch-1:
        print(f'cost after {i} iterations: {cost:.5f}')

print(f'weights: {theta}')

test_data = pd.read_csv('data/test.csv')
id_col = test_data.loc[:,['Id']].to_numpy()
test_data = test_data.loc[:,input_cols]

augment(test_data)
add_polynomials(test_data, input_cols)

X_norm = normalize(test_data.fillna(0)).to_numpy()
predictions = X_norm.dot(theta)
df = pd.DataFrame({'Id': id_col[:,0], 'SalePrice': predictions})
df.to_csv('data/submission.csv', index=False)
