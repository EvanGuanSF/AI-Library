import pandas as pd
import numpy as np
import math
from csc665 import metrics


class DecisionTreeRegressor:

    def __init__(self, max_depth=math.inf, min_samples_leaf=1):
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
            is_split_info = '|Split: {0:} <= {1:.2f}|Srow:{2:}]'.format(
                self.X.columns[self.split_col], self.split_val,
                str(self.split_row_index) + "-" + str(self.split_row_index + 1))
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

        return self

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
            self.find_best_split(i)

        # If we are not a leaf, continue splitting.
        if self.split_val is not None:
            # Create new subtrees based on that information if row indices in range.
            # Get sub-indices to iloc the sections that will be needed.
            opt_indices_left = np.array(self.split_optimal_indices[:self.split_row_index + 1])
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

        # Delete X and y to free memory resources.
        # del self.X
        # self.X = None
        # del self.y
        # self.y = None

    def find_best_split(self, i):
        # X contains the numeric values of the rows of the specified column within the dataframe.
        local_X = self.X.iloc[self.indices, i]
        # y contains the numeric values of the rows of the specified column within the interested values array.
        local_y = np.array(self.y.take(self.indices))
        # Sort local_X, local_y, and indices using X as the key.
        local_X, local_y, self.indices = zip(*sorted(zip(local_X, local_y, self.indices)))
        self.indices = np.array(self.indices)

        # Calculate MSE values and decide if this the best split so far.
        # If yes, set the object values: self.split_col, self.split_val, etc.
        # A manual for loop is required because we will on occasion skip row indices.
        x = 0

        while x < self.N:
            # Check and skip past duplicate feature values.
            # We group together duplicate feature values because those values will determine what "bucket"
            # an arbitrary record falls into.
            # '''
            skipped_values = False
            x_start_val = local_X[x]
            temp_x = x + 1
            while temp_x < self.N and local_X[temp_x] == x_start_val:
                skipped_values = True
                temp_x += 1
            if skipped_values is True:
                x = temp_x - 1
            # '''

            # Use the x_plus value to bounds check indices.
            x_plus = x + 1 if x + 1 < self.N else x

            # Check if we have reached the end of the X column after following up on duplicates.
            # If we have, don't split, just go to the next column iteration.
            if x_plus == x:
                return

            #  Calculate mse in chunks to improve readability and reduce function calls.
            #  Left section mses
            left_val = np.array(local_y[:x_plus]).mean() if np.array(local_y[:x_plus]).mean() > 0 else 0
            #  Right section mses
            right_val = np.array(local_y[x_plus:]).mean() if np.array(local_y[x_plus:]).mean() > 0 else 0
            #  Averaged mse
            current_mse = np.average(np.concatenate((
                (left_val - np.array(local_y[:x_plus])) ** 2,
                (right_val - np.array(local_y[x_plus:])) ** 2)))

            # If new current mse is equal to or better than the best mse,
            # then we adjust varaiables and continue.
            if current_mse < self.split_mse:
                self.split_mse = current_mse
                self.split_col = i
                self.split_val = (local_X[x] + local_X[x_plus]) / 2
                self.split_row_index = x
                self.split_optimal_indices = self.indices
            # If the mse we just calculated is worse than the mse for the entire subtree,
            # then we just return.
            # else:
            #     return
            # Iterate and go again.
            x += 1
        # If we reach this point, we return to the loop in split to check more columns
        # for better MSEs.

    def predict(self, X: pd.DataFrame):
        # Check to see if the tree has been built.
        if self.is_built is False:
            print("Warning! Decision Tree has not been built!")
            return

        # Create the return np.array.
        y_pred = list()

        # For each row in X, check it through the tree.
        for i in range(X.shape[0]):
            # tree_navigator is the "current node" as we travel down the tree.
            tree_navigator = self
            while True:
                if tree_navigator.split_col is not math.inf:
                    X_val = X.iloc[i, int(tree_navigator.split_col)]
                    if X_val <= tree_navigator.split_val and tree_navigator.left is not None:
                        tree_navigator = tree_navigator.left
                    elif X_val > tree_navigator.split_val and tree_navigator.right is not None:
                        tree_navigator = tree_navigator.right
                    else:
                        y_pred.append(tree_navigator.value)
                        break
                else:
                    y_pred.append(tree_navigator.value)
                    break

        return np.array(y_pred)

    def score(self, X: pd.DataFrame, y: np.array):
        return metrics.rsq(self.predict(X), y)


class DecisionTreeClassifier:

    def __init__(self, max_depth=math.inf, min_samples_leaf=1, num_classes=0):
        # Is the tree ready to make predictions?
        self.is_built = False
        # Maximum depth of the tree, starting from root node=0.
        self.max_depth = max_depth
        # Minimum number of samples to consider a node a leaf.
        self.min_samples_leaf = min_samples_leaf
        # The number of classes in the entirety of the data.
        self.num_classes = num_classes

    def __repr__(self):
        # This string contains the formatted split information if the node is a parent node.
        is_split_info: str
        if self.split_val != math.inf:
            is_split_info = '|Split: {0:} <= {1:.2f}|Srow:{2:}]'.format(
                self.X.columns[self.split_col], self.split_val,
                str(self.split_row_index) + "-" + str(self.split_row_index + 1))
        else:
            is_split_info = '|Prediction:{:2}]'.format(int(self.prediction))
        # Recurse into left subtree.
        if self.left is None:
            return ' ' * self.depth * 2 + str(self.depth) + ": " + \
                '[Value:{0}|Gini: {1:.3f}|N:{2:}{3:}'.format(str(self.value), self.gini, int(self.N), is_split_info)
        else:
            repr(self.left)
        # Recurse into right subtree.
        if self.right is None:
            return ' ' * self.depth * 2 + str(self.depth) + ": " + \
                '[Value:{0}|Gini: {1:.3f}|N:{2:}{3:}'.format(str(self.value), self.gini, int(self.N), is_split_info)
        else:
            repr(self.right)
        # Return info about node 0 and all child nodes.
        return ' ' * self.depth * 2 + str(self.depth) + ": " + \
            '[Value:{0}|Gini: {1:.3f}|N:{2:}{3:}'.format(str(self.value), self.gini, int(self.N), is_split_info) +\
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

        return self

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
        # Count the number of classes in the node if required (usually first init only).
        max_class_number = 0
        if self.num_classes == 0:
            # We will assume that the classes are represented as integers starting from zero,
            # and that the full range of classes in the full data set are all represented correctly.
            # Get a count of the number of clases.
            for i in range(self.N):
                if int(y[i]) > max_class_number:
                    max_class_number = int(y[i])
            max_class_number += 1
            self.num_classes = max_class_number
        # Error checking.
        if self.num_classes == 0:
            return
        # Create a zeroed array that can hold the number of classes.
        self.value = np.zeros((self.num_classes,), dtype=int)
        # Loop through the y's again and count the number of each class in this node.
        for i in range(self.N):
            self.value[int(y[i])] += 1
        # Finally, we can calculate the gini for the node.
        self.gini = 1 - np.sum((self.value / self.N) ** 2)

        # The following values will be set during training/splitting
        # Column index of the split if applicable.
        self.split_col = 0
        # Value of the split if applicable.
        self.split_val = math.inf
        # Gini impurity of the split gini, used to find the optimal split.
        self.split_gini = math.inf
        # Row index of the split if applicable.
        self.split_row_index = 0
        # The class that this node will predict.
        self.prediction = -math.inf
        # The set of ordered indices for the best gini.
        self.split_optimal_indices = indices

        # Left and right subtrees.
        self.left = None
        self.right = None

        self.split()

    def split(self):
        # Check the current depth since we are now done initializing variables.
        # Check if we are a leaf by comparing N against min_samples_leaf.
        # Check if the current node's gini impurity is greater than zero.
        if self.depth >= self.max_depth or self.N <= self.min_samples_leaf or math.isclose(self.gini, 0.0):
            # Now, if we ARE a leaf node, find the most commonly occuring class and set it as our prediction.
            if math.isclose(self.gini, 0.0):
                self.gini = 0.0
            cur_most_common_class = 0
            for i in range(self.num_classes):
                if self.value[i] > cur_most_common_class:
                    self.prediction = i
                    cur_most_common_class = self.value[i]
            return

        # Iterate over every column in X and try to split
        for i in range(self.X.shape[1]):
            # print("Splitting on col:", i, " with vals: ", str(self.value))
            self.find_best_split(i)
            # print("Col ", i, " done. ", ('{:.2f} {}').format(self.split_gini, str(i)))

        # If we are not a leaf, continue splitting.
        if self.split_val is not None:
            # Create new subtrees based on that information if row indices in range.
            # Get sub-indices to iloc the sections that will be needed.
            opt_indices_left = np.array(self.split_optimal_indices[:self.split_row_index + 1])
            opt_indices_right = np.array(self.split_optimal_indices[self.split_row_index + 1:])

            # Spawn the left subtree.
            if len(opt_indices_left) > 0:
                self.left = DecisionTreeClassifier(self.max_depth, self.min_samples_leaf, self.num_classes)
                self.left.internal_fit(
                    self.X.iloc[opt_indices_left],
                    self.y[opt_indices_left],
                    np.array(range(len(opt_indices_left))),
                    self.depth + 1)

            # Spawn the right subtree.
            # For the right tree, make sure there are nodes remaining
            # After spawning the left subtree.
            if len(opt_indices_right) > 0:
                self.right = DecisionTreeClassifier(self.max_depth, self.min_samples_leaf, self.num_classes)
                self.right.internal_fit(
                    self.X.iloc[opt_indices_right],
                    self.y[opt_indices_right],
                    np.array(range(len(opt_indices_right))),
                    self.depth + 1)

        # Delete X and y to free memory resources.
        # del self.X
        # self.X = None
        # del self.y
        # self.y = None

    def find_best_split(self, i):
        # X contains the numeric values of the rows of the specified column within the dataframe.
        local_X = self.X.iloc[self.indices, i]
        # y contains the numeric values of the rows of the specified column within the interested values array.
        local_y = np.array(self.y.take(self.indices))
        # Sort local_X, local_y, and indices using X as the key.
        local_X, local_y, self.indices = zip(*sorted(zip(local_X, local_y, self.indices)))
        self.indices = np.array(self.indices)

        # Calculate gini impurity values and decide if this the best split so far.
        # If yes, set the object values: self.split_col, self.split_val, etc.
        x = 0

        while x < self.N:
            # Check and skip past duplicate feature values.
            # We group together duplicate feature values because those values will determine what "bucket"
            # an arbitrary record falls into.
            # '''
            skipped_values = False
            x_start_val = local_X[x]
            temp_x = x + 1
            while temp_x < self.N and local_X[temp_x] == x_start_val:
                skipped_values = True
                temp_x += 1
            if skipped_values is True:
                x = temp_x - 1
            # '''
            # Use the x_plus value to bounds check indices.
            x_plus = x + 1 if x + 1 < self.N else x

            # Check if we have reached the end of the X column after following up on duplicates.
            # If we have, don't split, just go to the next column iteration.
            if x_plus == x:
                return

            left_num_samples = x_plus
            right_num_samples = self.N - x_plus

            #  Calculate gini in chunks to improve readability and reduce function calls.
            #  Left section gini
            left_vals = np.zeros((self.num_classes,), dtype=int)
            for j in range(0, x_plus):
                left_vals[int(local_y[j])] += 1
            left_gini = 1 - np.sum((left_vals / left_num_samples) ** 2)
            #  Right section gini
            right_vals = np.zeros((self.num_classes,), dtype=int)
            for j in range(x_plus, self.N):
                right_vals[int(local_y[j])] += 1
            right_gini = 1 - np.sum((right_vals / right_num_samples) ** 2)
            #  Weighted gini
            current_gini = ((left_num_samples / self.N) * left_gini
                            + (right_num_samples / self.N) * right_gini)
            '''
            print(('{:2} {:.2f} {:3} {} {:.2f} {:3} {} {:.5f} {} {} {} {:.5f} {:.5f}').format(
                str(x),
                left_gini, int(left_num_samples), left_vals,
                right_gini, int(right_num_samples), right_vals,
                current_gini, str(self.split_col), int(self.split_row_index), self.split_val,
                local_X[x], local_X[x_plus]))
            # '''

            # If new current gini is equal to or better than the best gini,
            # then we adjust varaiables and continue.
            if current_gini < self.split_gini:
                self.split_gini = current_gini
                self.split_col = i
                self.split_val = (local_X[x] + local_X[x_plus]) / 2
                self.split_row_index = x
                self.split_optimal_indices = self.indices
                '''
                print(('{:2} {:.2f} {:3} {} {:.2f} {:3} {} {:.5f} {} {} {} {:.5f} {:.5f}').format(
                    str(x),
                    left_gini, int(left_num_samples), left_vals,
                    right_gini, int(right_num_samples), right_vals,
                    current_gini, str(self.split_col), int(self.split_row_index), self.split_val,
                    local_X[x], local_X[x_plus]))
                # '''
            # Iterate and go again.
            x += 1

    def predict(self, X: pd.DataFrame):
        # Check to see if the tree has been built.
        if self.is_built is False:
            print("Warning! Decision Tree has not been built!")
            return

        # Create the return np.array.
        y_pred = list()

        # For each row in X, check it through the tree.
        for i in range(X.shape[0]):
            # tree_navigator is the "current node" as we travel down the tree.
            tree_navigator = self
            while True:
                if tree_navigator.split_col is not math.inf:
                    X_val = X.iloc[i, int(tree_navigator.split_col)]
                    if X_val <= tree_navigator.split_val and tree_navigator.left is not None:
                        tree_navigator = tree_navigator.left
                    elif X_val > tree_navigator.split_val and tree_navigator.right is not None:
                        tree_navigator = tree_navigator.right
                    else:
                        y_pred.append(tree_navigator.prediction)
                        break
                else:
                    y_pred.append(tree_navigator.prediction)
                    break

        return np.array(y_pred)

    def score(self, X: pd.DataFrame, y: np.array):
        return metrics.accuracy_score(self.predict(X), y)
