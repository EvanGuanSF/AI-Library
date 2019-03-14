import pandas as pd
import numpy as np
import random
import csc665.metrics
import csc665.tree
# import sklearn.tree


class RandomForestRegressor:
    def __init__(self, n_estimators=10, sample_ratio=0.1, random_state=None):
        # Initialize and bounds check values.
        self.n_estimators = max(n_estimators, 0)
        self.sample_ratio = max(min(sample_ratio, 1.0), 0)
        self.is_random_sampling = True if random_state is None else False
        self.random_state = random_state
        self.is_built = False

    def fit(self, X: pd.DataFrame, y: np.array):
        self.forest = []
        # Check for correct datatype, fix if incorrect.
        if isinstance(y, pd.DataFrame):
            y = np.reshape(np.array(y), -1)
        # Seed the rng.
        if self.is_random_sampling:
            np.random.seed(random.randint(0, 2**32 - 1))
        else:
            np.random.seed(self.random_state)

        n_samples_per_tree = int(self.sample_ratio * X.shape[0])
        # Make n_estimator amount of randomly seeded trees.
        for x in range(self.n_estimators):
            # Generate a selection of indices in-place (duplicate indices allowed)
            random_indices = np.random.choice(a=X.shape[0], size=n_samples_per_tree, replace=True)

            new_dtr = csc665.tree.DecisionTreeRegressor()
            # new_dtr = sklearn.tree.DecisionTreeRegressor()

            new_dtr.fit(X.iloc[random_indices], y[random_indices])

            # Add the tree to the object array.
            self.forest.append(new_dtr)

        # print("Printing trees...")
        # self.print_trees()
        self.is_built = True

        return

    def predict(self, X: pd.DataFrame):
        if self.is_built is False:
            print("Warning: Forest has not been built!")
            return -1

        # Setup the and create the full 2d array of predictions.
        y_pred_list = list()
        for tree in self.forest:
            y_pred_list.append(tree.predict(X))

        y_pred_list = np.array(y_pred_list)
        # Average the results from all the trees into a final result.
        return y_pred_list.mean(axis=0)

    def score(self, X: pd.DataFrame, y: np.array):
        if self.is_built is False:
            print("Warning: Forest has not been built!")
            return -1

        return csc665.metrics.rsq(self.predict(X), y)

    def print_trees(self):
        for tree in self.forest:
            print(repr(tree))
