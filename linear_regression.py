#!/usr/bin/env python3

import numpy as np
import pandas as pd

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
]
data = data.loc[:,input_cols+['SalePrice']].dropna()

Y = data.loc[:,'SalePrice'].to_numpy()
data.drop(['SalePrice'], axis=1, inplace=True)

add_polynomials(data, input_cols)

X_norm = normalize(data).to_numpy()

theta = np.random.rand(X_norm.shape[1])
alpha = 1e-3
lmbda = 1e-6
m = len(X_norm)
print(f'training with {m} examples')
iterations = int(1e5)
for i in range(iterations):
    pred = X_norm.dot(theta)
    diff = pred - Y
    grad = diff.dot(X_norm)
    theta -= (alpha/m) * grad
    theta += (lmbda/m) * theta.sum()
    cost = 1/(2*m) * (diff ** 2).sum() + (lmbda/(2*m)) * (theta ** 2).sum()
    if (i % (iterations / 10)) == 0:
        print(f'cost: {cost:.5f}')

print(f'weights: {theta}')

test_data = pd.read_csv('data/test.csv')
id_col = test_data.loc[:,['Id']].to_numpy()
test_data = test_data.loc[:,input_cols]

add_polynomials(test_data, input_cols)

X_norm = normalize(test_data.fillna(0)).to_numpy()
predictions = X_norm.dot(theta)
df = pd.DataFrame({'Id': id_col[:,0], 'SalePrice': predictions})
df.to_csv('data/submission.csv', index=False)
