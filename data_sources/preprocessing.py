import pandas as pd

from sklearn.preprocessing import normalize

norm_feature = None


def preprocess(X):
    global norm_feature
    X, norm_feature = normalize(X, return_norm=True)
    return X


def preprocess_inverse(X):
    return np.multiply(X, norm_feature[:, None])


def preprocess_df(df):
    return pd.DataFrame(preprocess(df.values.T).T, columns=df.columns, index=df.index)
