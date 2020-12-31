#!/usr/bin/env python3

import numpy as np
import pandas as pd

from preprocessing import get_training_data
from preprocessing import get_test_data

def main():

    X, Y, mappings = get_training_data('data/train.csv')

    X_norm = (X - X.mean()) / X.std()
    Y_norm = (Y - Y.mean()) / Y.std()

    # TODO: split train/cv data sets
    # TODO: add regularization

    theta = np.random.rand(X.shape[1])
    alpha = 1e-2
    m = len(X)
    print(f'training with {m} examples')
    iterations = int(1e5)
    batch = iterations / 10

    for i in range(iterations):
        pred = X_norm.dot(theta)
        diff = pred - Y_norm
        grad = diff.dot(X_norm)
        theta -= (alpha/m) * grad
        cost = 1/(2*m) * (diff ** 2).sum()
        if i % batch == batch-1:
            print(f'cost after {i+1} iterations: {cost}')

    print(f'weights: {theta}')

    test_data, id_col = get_test_data('data/test.csv', mappings)
    test_data_norm = (test_data - test_data.mean()) / test_data.std()

    predictions_norm = test_data_norm.dot(theta)
    predictions = predictions_norm * Y.std() + Y.mean()
    df = pd.DataFrame({'Id': id_col[:,0], 'SalePrice': predictions})
    submission_path = 'data/submission.csv'
    df.to_csv(submission_path, index=False)
    print(f'saved submission data to {submission_path}')


if __name__ == '__main__':
    main()
