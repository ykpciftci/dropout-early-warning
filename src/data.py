import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path, target):
    """Load the CSV, keep dropouts and graduates, encode Dropout as 1."""
    df = pd.read_csv(path, sep=';')
    df.columns = df.columns.str.strip()
    df_binary = df[df[target].isin(['Dropout', 'Graduate'])]
    mapping_dict = {'Dropout': 1, 'Graduate': 0}
    df_binary[target] = df_binary[target].map(mapping_dict)
    return df_binary


def split_data(df, features, target, random_state, test_size):
    """Split df into stratified train and test sets using the given features."""
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=random_state,
        test_size=test_size,
        stratify=y
        )
    return X_train, X_test, y_train, y_test
