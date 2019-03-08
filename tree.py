import pandas as pd
import numpy as np
import math
import warnings
from csc665 import metrics


class DecisionTreeRegressor:

    def __init__(self, max_depth, min_samples_leaf):
        self.is_built = False
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def __repr__(self):
        # The number of tabs equal to the level * 4, for formatting
        # Print the tree for debugging purposes, e.g.
        # 0: [value, mse, samples, split_col <= split_val, if any]
        #   1: [value, mse, samples, split_col <= split_val, if any]
        #   1: [value, mse, samples, split_col > split_val, if any]
        #       2: 1: [value, mse, samples, split_col <= split_val, if any]
        # etc..

        is_split_info: str
        if self.split_col != math.inf:
            is_split_info = '|Sc:{0:}|Sv:{1:.2f}]'.format(self.split_col, self.split_val)
        else:
            is_split_info = "]"

        if self.left is None:
            return ' ' * self.depth * 2 + str(self.depth) + ": " + \
                '[V:{0:.2f}|M:{1:.2f}|N:{2:}{3:}'.format(self.value, self.mse, self.N, is_split_info)
        else:
            repr(self.left)

        if self.right is None:
            return ' ' * self.depth * 2 + str(self.depth) + ": " + \
                '[V:{0:.2f}|M:{1:.2f}|N:{2:}{3:}'.format(self.value, self.mse, self.N, is_split_info)
        else:
            repr(self.right)

        return ' ' * self.depth * 2 + str(self.depth) + ": " + \
            '[V:{0:.2f}|M:{1:.2f}|N:{2:}{3:}'.format(self.value, self.mse, self.N, is_split_info) +\
            "\n" + repr(self.left) + "\n" + repr(self.right)

    def fit(self, X: pd.DataFrame, y: np.array):
        # N = number of rows in the dataframe (number of samples).
        self.N = X.shape[0]
        # Indices of rows to be used in the tree
        self.indices = np.array(range(self.N))
        # All rows will be used at the top level; then this array will depend
        # on the split
        self.internal_fit(X, y, self.N, depth=0)

        self.is_built = True

    def internal_fit(self, X, y, indices, depth):
        #  The dataframe.
        self.X = X
        #  The column we are interested in predicting for.
        self.y = y
        # N = number of rows in the dataframe (number of samples).
        self.N = X.shape[0]

        #  Workaround.
        #  Check to make sure the indices exist if we are calling internally.
        try:
            self.indices
        except AttributeError:
            self.indices = np.array(indices)

        # Calculate value
        #  This is our current depth.
        self.depth = depth
        #  This is the mean of the entire subtree's y_known.
        self.value = y.mean()
        #  This is the mse of the entire subtree's y_known.
        self.mse = ((self.value - y) ** 2).mean()

        # The following values will be set during training/splitting
        # Index of a column on which we split
        self.split_col = math.inf
        # Split value for the split_col
        self.split_val = math.inf
        # The mse for the current or best split
        self.split_mse = math.inf
        #  The row index of the split if applicable.
        self.split_row_index = math.inf
        #  The set of ordered indices for the best mse.
        self.split_optimal_indices = indices

        # Left and right subtrees, if not leaf
        self.left = None
        self.right = None

        self.split()

    def split(self):
        #  Check the current depth since we are now done initializing variables.
        #  Check if we are a leaf by comparing N against min_samples_leaf.
        if self.depth >= self.max_depth:
            # print()
            # print("At max depth, returning.")
            return
        if self.N <= self.min_samples_leaf:
            # print()
            # print("Min leaf samples, returning.")
            return

        # Iterate over every column in X and try to split
        for i in range(self.X.shape[1]):
            self.find_best_split(i)

        #  If we are not a leaf, continue splitting.
        # Once done with finding the split, actually split and
        # create two subtrees
        #  We only need to create subtrees at this point if we have found a good split,
        #  so check the variables for initialization before continuing, and return otherwise.
        if self.split_val is not None:
            #  Create new subtrees based on that information if row indices in range.
            #  Get sub-indices to iloc the sections that will be needed.
            opt_indices_left = np.array(self.split_optimal_indices[0:self.split_row_index + 1])
            opt_indices_right = np.array(self.split_optimal_indices[self.split_row_index + 1:])

            #  Setup the features.
            # l_features = self.X.iloc[opt_indices_left]
            # r_features = self.X.iloc[opt_indices_right]

            #  Setup the values.
            # l_vals = self.y[opt_indices_left]
            # r_vals = self.y[opt_indices_right]

            #  Spawn the left subtree.
            self.left = DecisionTreeRegressor(self.max_depth, self.min_samples_leaf)
            self.left.internal_fit(
                self.X.iloc[opt_indices_left],
                self.y[opt_indices_left],
                np.array(range(len(opt_indices_left))),
                self.depth + 1)

            #  Spawn the right subtree.
            #  For the right tree, make sure there are nodes remaining
            #  After spawning the left subtree.
            if len(opt_indices_right) > 0:
                self.right = DecisionTreeRegressor(self.max_depth, self.min_samples_leaf)
                self.right.internal_fit(
                    self.X.iloc[opt_indices_right],
                    self.y[opt_indices_right],
                    np.array(range(len(opt_indices_right))),
                    self.depth + 1)

    def find_best_split(self, i):
        # X contains the numeric values of the rows of the specified
        #  column within the dataframe.
        local_X = self.X.iloc[self.indices, i]
        # y contains the numeric values of the rows of the specified
        #  column within the interested values array.
        local_y = self.y.take(self.indices)

        #  Sort local_X, local_y, and indices using X as the key.
        local_X, local_y, self.indices = zip(*sorted(zip(local_X, local_y, self.indices)))
        self.indices = np.array(self.indices)

        '''
        print("sorting by: " + self.X.columns[i])
        for j in range(len(local_X)):
            print('|{:<3}|'.format(j) + 'X: {:<7.3f}|'.format(local_X[j]) + ' y: {:<7.3f}|'.format(local_y[j]))
        '''

        #  Check for 1-row base case and the case where we only have duplicate X's in the frame.
        if len(self.indices) <= 1:
            return

        # Calculate MSE values and decide if this the best split so far.
        # If yes, set the object values: self.split_col, self.split_val, etc.
        x = 0
        while x < self.N:
            #  Compare the current split's mse with the best so far.
            #  If the current mse is better, replace the running mse and set self.split_col.
            #  Visually, this looks like a sort of wave effect as we iterate over the array.

            #  If this and the next column have the same X value, then skip past all duplicates and
            #  head to the next iteration.
            #  We group together duplicate X's because X will determine what "bucket" an arbitrary
            #  datapoint falls into.
            #  However, we will only group
            skipped_values = False
            x_start_val = local_X[x]
            temp_x = x + 1
            while temp_x < self.N:
                if local_X[temp_x] == x_start_val:
                    skipped_values = True
                    temp_x += 1
                else:
                    break
            if skipped_values is True:
                x = temp_x - 1

            #  Calculate mse in chunks to improve readability and reduce function calls.
            #  Left section mse
            #  Use the x_plus value to bounds check indices.
            x_plus = x + 1 if x + 1 < self.N else x
            left_val = 0
            left_val = np.array(local_y[0:x_plus]).mean() if np.array(local_y[0:x_plus]).mean() > 0 else 0
            left_mse = 0
            left_mse = ((left_val - np.array(local_y[0:x_plus])) ** 2).mean()
            left_mse = left_mse if left_mse > 0 else 0
            #  Right section mse
            right_val = 0
            right_val = np.array(local_y[x_plus:]).mean() if np.array(local_y[x_plus:]).mean() > 0 else 0
            right_mse = 0
            right_mse = ((right_val - np.array(local_y[x_plus:])) ** 2).mean()
            right_mse = right_mse if right_mse > 0 else 0
            #  Combined mse
            current_mse = 0
            current_mse = left_mse + right_mse

            if len(self.indices) == 2:
                x_plus = x + 1 if x + 1 < self.N else x
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    self.split_mse = np.array(local_y[0:x]).mean()
                self.split_col = i
                self.split_val = (local_X[x] + local_X[x_plus]) / 2
                # print("Split val: " + '{0:<7.2f}'.format(self.split_val))
                # print('{:-<75}'.format(''))
                self.split_row_index = x
                self.split_optimal_indices = self.indices
                return
            '''
            #  Use the x_plus value to bounds check indices.
            x_plus = x + 1 if x + 1 < self.N else x
            print('|{:<2}|'.format(str(x)) +
                ' C  {0:<7.2f}|'.format(current_mse) + ' B  {0:<7.2f}|'.format(self.split_mse) +
                ' X  {0:<7.2f}|'.format(local_X[x]) + ' X+ {0:<7.2f}|'.format(local_X[x_plus]) +
                ' Yv {0:<7.2f}|'.format(local_y[x]) + " [" + str(i) + ", " + str(i) + "]")
            '''
            # print("[" + str(x) + ", " + str(i) + "]")
            #  If new current mse is equal to or better than the best mse,
            #  then we adjust varaiables and continue.
            if current_mse < self.split_mse:
                '''
                print('*{:<2}|'.format(str(x))  + ' C  {0:<7.2f}'.format(current_mse) + "<" +
                    ' B  {0:<7.2f}*'.format(self.split_mse) + " [" + str(i) + ", " + str(i) + "]")
                '''
                #  Use the x_plus value to bounds check indices.
                x_plus = x + 1 if x + 1 < self.N else x
                self.split_mse = current_mse
                self.split_col = i
                self.split_val = (local_X[x] + local_X[x_plus]) / 2
                self.split_row_index = x
                self.split_optimal_indices = self.indices
                # print("Split val: " + '{0:<7.2f}'.format(self.split_val) + " Split row: " + str(x))

            #  If the mse we just calculated is worse than the mse for the entire subtree,
            #  then we just return.
            else:
                return
            x += 1
        #  At this point, we return to the lop in split to check more columns
        #  for better mses.

    def predict(self, X: pd.DataFrame):
        if self.is_built is False:
            print("Warning! Decision Tree has not been built!")
            return

        #  Create the return np.array.
        y_pred = np.empty(0)

        tree_navigator = self

        #  For each row in X, check it through the tree.
        for i in range(X.shape[0]):
            tree_navigator = self
            while True:
                if tree_navigator.split_col is not math.inf:
                    X_val = X.iloc[i, int(tree_navigator.split_col)]
                    # print("Checking: " + str(X_val) + " against " + '{0:.2f}'.format(tree_navigator.split_val))
                    if X_val <= tree_navigator.split_val:
                        if tree_navigator.left is not None:
                            tree_navigator = tree_navigator.left
                            continue
                        else:
                            y_pred = np.append(y_pred, tree_navigator.value)
                            # print("Making prediction: ", y_pred)
                            break
                    if X_val > tree_navigator.split_val:
                        if tree_navigator.right is not None:
                            tree_navigator = tree_navigator.right
                            continue
                        else:
                            y_pred = np.append(y_pred, tree_navigator.value)
                            # print("Making prediction: ", y_pred)
                            break
                else:
                    y_pred = np.append(y_pred, tree_navigator.value)
                    break

        return np.array(y_pred)

    def score(self, X: pd.DataFrame, y: np.array):
        return metrics.rsq(self.predict(X), y)
