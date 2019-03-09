import pandas as pd
import numpy as np
import math
import warnings
from csc665 import metrics


class DecisionTreeRegressor:

    def __init__(self, max_depth, min_samples_leaf):
        # Is the tree ready to make predictions?
        self.is_built = False
        # Maximum depth of the tree, starting from root node=0.
        self.max_depth = max_depth
        # Minimum number of samples to consider a node a leaf.
        self.min_samples_leaf = min_samples_leaf

    def __repr__(self):
        # This string contains the formatted split information if the node is a parent node.
        is_split_info: str
        if self.split_val != math.inf:
            is_split_info = '|Sval:<={0:.2f}|Scol:{1:}|Srow:{2:}]'.format(
                self.split_val, self.X.columns[self.split_col], str(self.split_row_index)
                + "-" + str(self.split_row_index + 1))
        else:
            is_split_info = "]"

        # Recurse into left subtree.
        if self.left is None:
            return ' ' * self.depth * 2 + str(self.depth) + ": " + \
                '[Val:{0:.2f}|MSE:{1:.2f}|N:{2:}{3:}'.format(self.value, self.mse, self.N, is_split_info)
        else:
            repr(self.left)

        # Recurse into right subtree.
        if self.right is None:
            return ' ' * self.depth * 2 + str(self.depth) + ": " + \
                '[Val:{0:.2f}|MSE:{1:.2f}|N:{2:}{3:}'.format(self.value, self.mse, self.N, is_split_info)
        else:
            repr(self.right)

        # Return info about node 0 and all child nodes.
        return ' ' * self.depth * 2 + str(self.depth) + ": " + \
            '[Val:{0:.2f}|MSE:{1:.2f}|N:{2:}{3:}'.format(self.value, self.mse, self.N, is_split_info) +\
            "\n" + repr(self.left) +\
            "\n" + repr(self.right)

    def fit(self, X: pd.DataFrame, y: np.array):
        # N = number of rows in the dataframe (number of samples).
        self.N = X.shape[0]
        # Indices of rows to be used in the tree
        self.indices = np.array(range(self.N))
        # All rows will be used at the top level; then this array will depend
        # on the split
        self.internal_fit(X, y, self.N, depth=0)
        # Set the is_built bool to true to indicate that the tree is ready to make predictions.
        self.is_built = True

    def internal_fit(self, X, y, indices, depth):
        # The feature dataframe.
        self.X = X
        # The column we are interested in predicting for.
        self.y = y
        # N = number of rows in the dataframe (number of samples).
        self.N = X.shape[0]
        # Check to make sure self.indices exists if we are calling internally.
        try:
            self.indices
        except AttributeError:
            self.indices = np.array(indices)
        # Depth of current subtree.
        self.depth = depth
        # Mean of the entire subtree's y_known.
        self.value = y.mean()
        # MSE of the entire subtree's y_known.
        self.mse = ((self.value - y) ** 2).mean()

        # The following values will be set during training/splitting
        # Column index of the split if applicable.
        self.split_col = 0
        # Value of the split if applicable.
        self.split_val = math.inf
        # MSE of the split if applicable.
        self.split_mse = math.inf
        # Row index of the split if applicable.
        self.split_row_index = 0
        # The set of ordered indices for the best mse.
        self.split_optimal_indices = indices

        # Left and right subtrees.
        self.left = None
        self.right = None

        self.split()

    def split(self):
        # Check the current depth since we are now done initializing variables.
        # Check if we are a leaf by comparing N against min_samples_leaf.
        if self.depth >= self.max_depth or self.N <= self.min_samples_leaf:
            return

        # Iterate over every column in X and try to split
        for i in range(self.X.shape[1]):
            # print("Iterating on col: " + str(i))
            self.find_best_split(i)

        # If we are not a leaf, continue splitting.
        # Once done with finding the split, actually split and create two subtrees
        # We only need to create subtrees at this point if we have found a good split,
        # so check the variables for initialization before continuing, and return otherwise.
        if self.split_val is not None:
            # Create new subtrees based on that information if row indices in range.
            # Get sub-indices to iloc the sections that will be needed.
            opt_indices_left = np.array(self.split_optimal_indices[0:self.split_row_index + 1])
            opt_indices_right = np.array(self.split_optimal_indices[self.split_row_index + 1:])

            # Spawn the left subtree.
            self.left = DecisionTreeRegressor(self.max_depth, self.min_samples_leaf)
            self.left.internal_fit(
                self.X.iloc[opt_indices_left],
                self.y[opt_indices_left],
                np.array(range(len(opt_indices_left))),
                self.depth + 1)

            # Spawn the right subtree.
            # For the right tree, make sure there are nodes remaining
            # After spawning the left subtree.
            if len(opt_indices_right) > 0:
                self.right = DecisionTreeRegressor(self.max_depth, self.min_samples_leaf)
                self.right.internal_fit(
                    self.X.iloc[opt_indices_right],
                    self.y[opt_indices_right],
                    np.array(range(len(opt_indices_right))),
                    self.depth + 1)

    def find_best_split(self, i):

        # X contains the numeric values of the rows of the specified column within the dataframe.
        local_X = self.X.iloc[self.indices, i]
        # y contains the numeric values of the rows of the specified column within the interested values array.
        local_y = self.y.take(self.indices)
        # Sort local_X, local_y, and indices using X as the key.
        local_X, local_y, self.indices = zip(*sorted(zip(local_X, local_y, self.indices)))
        self.indices = np.array(self.indices)

        # Calculate MSE values and decide if this the best split so far.
        # If yes, set the object values: self.split_col, self.split_val, etc.
        # A manual for loop is required because we will on occasion skip row indices.
        ''' Print current X and y columns
        print("sorting by: " + self.X.columns[i])
        for j in range(len(local_X)):
            print('|{:<3}|'.format(j) + 'X: {:<7.3f}|'.format(local_X[j]) + ' y: {:<7.3f}|'.format(local_y[j]))
        # '''

        x = 0
        while x < self.N:
            # Check and skip past duplicate X-rows.
            # We group together duplicate X's because X will determine what "bucket" an arbitrary datapoint falls into.
            # Be sure to skip this processing if the split has two or fewer elements in it so that we can get to leaf nodes.
            if self.N > 2:
                skipped_values = False
                x_start_val = local_X[x]
                temp_x = x + 1
                while temp_x < self.N and local_X[temp_x] == x_start_val:
                    skipped_values = True
                    temp_x += 1
                if skipped_values is True:
                    x = temp_x - 1

            # Use the x_plus value to bounds check indices.
            x_plus = x + 1 if x + 1 < self.N else x
            # Combined mse
            current_mse = (((np.array(local_y[:x_plus]).mean() - np.array(local_y[:x_plus])) ** 2).mean()
                           + ((np.array(local_y[x_plus:]).mean() - np.array(local_y[x_plus:])) ** 2).mean())

            ''' Print state information of current row.
            print(
                '|{:<2}|'.format(str(x))
                + ' C  {0:<7.2f}|'.format(current_mse) + ' B  {0:<7.2f}|'.format(self.split_mse)
                + ' X  {0:<7.2f}|'.format(local_X[x]) + ' X+ {0:<7.2f}|'.format(local_X[x_plus])
                + ' Yv {0:<7.2f}|'.format(local_y[x]) + " [" + str(x) + ", " + str(x_plus) + "]")
            # '''

            # Special case when we only have two nodes in the split.
            if self.N == 2:
                # Check previous column. If that column did not contain duplicates, then we have no work to do.
                if i > 0 and self.X.iloc[x, i - 1] != self.X.iloc[x_plus, i - 1]:
                    return
                # Check for duplicate X values in the current column.
                elif local_X[x] != local_X[x_plus]:
                    # Supress empty slice warnings.
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        self.split_mse = np.array(local_y[0:x]).mean()
                    self.split_col = i
                    self.split_val = (local_X[x] + local_X[x_plus]) / 2
                    self.split_row_index = x
                    self.split_optimal_indices = self.indices
                return

            # If new current mse is equal to or better than the best mse,
            # then we adjust varaiables and continue.
            if current_mse < self.split_mse:
                self.split_mse = current_mse
                self.split_col = i
                self.split_val = (local_X[x] + local_X[x_plus]) / 2
                self.split_row_index = x
                self.split_optimal_indices = self.indices
                ''' Print information about the split information update.
                print("x: " + str(x) + " x_plus: " + str(x_plus))
                print(
                    "Split val: " + '{0:<7.2f}'.format(self.split_val) + " On row " + str(x)
                    + " with L/R values: L:" + '{0:<7.2f}'.format(local_X[x])
                    + " R:" + '{0:<7.2f}'.format(local_X[x_plus]) + "Sr: " + '{0:<7.2f}'.format(self.split_row_index))
                print('{:-<75}'.format(''))
                # '''
            # If the mse we just calculated is worse than the mse for the entire subtree,
            # then we just return.
            else:
                return
            # Iterate and go again.
            x += 1
        # If we reach this point, we return to the loop in split to check more columns
        # for better MSEs.
        return

    def predict(self, X: pd.DataFrame):
        # Check to see if the tree has been built.
        if self.is_built is False:
            print("Warning! Decision Tree has not been built!")
            return

        # Create the return np.array.
        y_pred = np.empty(0)

        # For each row in X, check it through the tree.
        for i in range(X.shape[0]):
            # tree_navigator is the "current node" as we travel down the tree.
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
