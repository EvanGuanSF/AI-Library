import pandas as pd
import numpy as np
import random
import math

def train_test_split(X, y, test_size=0.25, shuffle=True, random_state=None):
    # Setup variables for usage.
    # Check and set the train-to-test split ratio first.
    test_size = test_size if (type(test_size) is float and 0.0 <= test_size <= 1.0) else 0.25
    first_split_size = math.ceil(len(y) * test_size)
    # Then check and set the random seed.
    random_state = random_state if random_state is not None else random.randint(0, 100000)

    # Then shuffle if needed.
    if shuffle == True:
        # Concat the dataframes back together, using the indices we already had,
        # in order to preserve row parity.
        Xy_pair = pd.concat([X, y], axis=1, join='outer', ignore_index=False)
        # Now shuffle.
        Xy_pair = Xy_pair.reindex(np.random.RandomState(seed=random_state).permutation(Xy_pair.index))
        # Finally, seperate the dataframe back to X and y parts.
        y = pd.DataFrame(Xy_pair[y.columns[0]])
        X = pd.DataFrame(Xy_pair.drop(y.columns[0], axis=1))

    # Now split the dataframes.
    # X_train split.
    X_train_split = X.head(len(y) - first_split_size)
    # X_test split.
    X_test_split = X.tail(first_split_size)
    # y_train split.
    y_train_split = pd.DataFrame(y.head(len(y) - first_split_size)).T
    y_train_split = y_train_split[y_train_split.columns].values
    y_train_split = y_train_split[0]
    # y_test split.
    y_test_split = pd.DataFrame(y.tail(first_split_size)).T
    y_test_split = y_test_split[y_test_split.columns].values
    y_test_split = y_test_split[0]
    
    return pd.DataFrame(X_train_split), pd.DataFrame(X_test_split), y_train_split, y_test_split


def create_categories(data_frame, list_columns):
    str_cols = list()

    # Filter out the string type columns first, and store them in a list.
    for col_name in list_columns:
        if pd.api.types.is_string_dtype(data_frame[col_name]):
            list.append(str_cols, col_name)

    # Now check to see if we have any datetime string in the list. Convert if we find one.
    for col_name in str_cols:
        data_frame[col_name] = data_frame[col_name].astype(np.datetime64, errors='ignore')
        if pd.api.types.is_datetime64_dtype(data_frame[col_name]):
            data_frame[col_name] = pd.to_datetime(data_frame[col_name], infer_datetime_format=True)
            data_frame[col_name] = data_frame[col_name].astype(np.int64)
            list.remove(str_cols, col_name)

    # Loop through the remaining str_cols list and convert to numeric categories.
    for col_name in str_cols:
        data_frame[col_name] = data_frame[col_name].astype(np.datetime64, errors='ignore')
        data_frame[col_name] = data_frame[col_name].astype('category').cat.codes

    # Return the data_frame in-place.
    return (data_frame)


def preprocess_ver_1(csv_df, target_col):
    # Remove rows with NaN values.
    rows_labeled_na = csv_df.isnull().any(axis=1)
    rows_with_na = csv_df[rows_labeled_na]
    rows_with_data = csv_df[~rows_labeled_na]

    # Seperate y from the dataframe.
    y_label = pd.DataFrame(rows_with_data[target_col])
    feature_data_frame = pd.DataFrame(rows_with_data.drop(target_col, axis=1))
    feature_data_frame.head()
    list_columns = list(feature_data_frame.columns.values)

    # Give category codes to remaining string columns and convert datetime.
    feature_data_frame = create_categories(feature_data_frame, list_columns)

    return feature_data_frame, y_label
