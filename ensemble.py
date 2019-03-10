import pandas as pd
import numpy as np
import random
import math
import csc665.metrics
import csc665.tree


class RandomForestRegressor:
    def __init__(self, n_estimators=10, sample_ratio=0.35, random_state=None):
        # Initialize and bounds check values.
        self.n_estimators = max(n_estimators, 0)
        self.sample_ratio = max(min(sample_ratio, 1.0), 0)
        self.is_random_sampling = True if random_state is None else False
        self.random_state = random_state
        self.is_built = False
        # print(self.n_estimators, " ", self.sample_ratio, " ", self.random_state)

    def fit(self, X: pd.DataFrame, y: np.array):

        # The list of DecisionTreeRegressors.
        self.forest = []
        # Check for correct datatype, fix if incorrect.
        if isinstance(y, pd.DataFrame):
            y = np.reshape(np.array(y), -1)
        # The number of samples we will be giving to each tree.
        n_samples_per_tree = int(self.sample_ratio * X.shape[0])

        # Make n_estimator amount of randomly seeded trees.
        for x in range(self.n_estimators):
            rs = None
            # Sort out the random seeding.
            if self.is_random_sampling:
                rs = np.random.RandomState(random.randint(0, 2**32 - 1))
            else:
                rs = np.random.RandomState(self.random_state)
            # Now shuffle and grab a range of records to give to the new tree.
            shuffled_indices = rs.permutation(X.shape[0])
            X_shuffled = X.iloc[shuffled_indices[0:n_samples_per_tree], :]
            y_shuffled = y[shuffled_indices[0:n_samples_per_tree]]
            # Make a new tree and give it the shuffled dataframes.
            new_dtr = csc665.tree.DecisionTreeRegressor()
            new_dtr.fit(X_shuffled, y_shuffled)
            # Add the tree to the object array.
            self.forest.append(new_dtr)

        # self.print_trees()
        self.is_built = True

        return

    def predict(self, X: pd.DataFrame):
        if self.is_built is False:
            print("Warning: Forest has not been built!")
            return -1

        # Setup the full list of predictions.
        y_pred_list = []
        # loop through X.shape[0] number of times.
        for x in range(X.shape[0]):
            # Grab the row.
            X_test = X.iloc[[x]]
            y_pred = []
            # Send the row to all of the trees and sum the results.
            for tree in self.forest:
                y_pred.append(tree.predict(X_test))
            # Average the results from all the trees into a final result.
            y_pred_list.append(np.average(y_pred))

        # print(y_pred_list)
        return np.array(y_pred_list)

    def score(self, X: pd.DataFrame, y: np.array):
        if self.is_built is False:
            print("Warning: Forest has not been built!")
            return -1

        return csc665.metrics.rsq(self.predict(X), y)

    def print_trees(self):
        for tree in self.forest:
            print(repr(tree))
