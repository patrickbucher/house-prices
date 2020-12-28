#!/usr/bin/env python3

import numpy as np
import pandas as pd

def normalize(arr):
    return (arr - arr.mean()) / arr.std()

data = pd.read_csv('data/train.csv')
input_cols = [
    'LotArea',
    'LotFrontage',
    'OverallQual',
    'OverallCond',
    'YearBuilt',
    'GrLivArea',
]
data = data.loc[:,input_cols+['SalePrice']].dropna()

Y = data.loc[:,'SalePrice'].to_numpy()
data.drop(['SalePrice'], axis=1, inplace=True)

# polynomial features
for col in input_cols:
    for p in [2,3,4,5]:
        data[f'{col}{p}'] = data[col] ** p

X_norm = normalize(data).to_numpy()

theta = np.random.rand(X_norm.shape[1])
alpha = 1e-3
lmbda = 1e-6
m = len(X_norm)
print(f'training with {m} examples')
iterations = int(1e5)
try:
    for i in range(iterations):
        pred = X_norm.dot(theta)
        diff = pred - Y
        grad = diff.dot(X_norm)
        theta -= (alpha/m) * grad
        theta += (lmbda/m) * theta.sum()
        cost = 1/(2*m) * (diff ** 2).sum() + (lmbda/(2*m)) * (theta ** 2).sum()
        if (i % (iterations / 10)) == 0:
            print(f'cost: {cost:.5f}')
except KeyboardInterrupt:
    pass

print(f'weights: {theta}')

pred = X_norm.dot(theta)
print('predictions', pred)
print('actual', Y)
print(Y - pred)
