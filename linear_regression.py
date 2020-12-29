#!/usr/bin/env python3

import numpy as np
import pandas as pd

from preprocessing import get_training_data
from preprocessing import get_test_data

def main():

    X, Y = get_training_data('data/train.csv')

    theta = np.random.rand(X.shape[1])
    alpha = 1e-1
    lmbda = 1e-6
    m = len(X)
    print(f'training with {m} examples')
    iterations = int(1e3)
    batch = iterations / 10

    for i in range(iterations):
        pred = X.dot(theta)
        diff = pred - Y
        grad = diff.dot(X)
        theta -= (alpha/m) * (grad + (lmbda/m) * theta.sum())
        cost = 1/(2*m) * (diff ** 2).sum() + (lmbda/(2*m)) * (theta ** 2).sum()
        if i % batch == batch-1:
            print(f'cost after {i+1} iterations: {cost:.5f}')

    print(f'weights: {theta}')

    test_data, id_col = get_test_data('data/test.csv')

    predictions = test_data.dot(theta)
    df = pd.DataFrame({'Id': id_col[:,0], 'SalePrice': predictions})
    df.to_csv('data/submission.csv', index=False)


if __name__ == '__main__':
    main()
