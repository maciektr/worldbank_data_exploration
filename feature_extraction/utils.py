from sklearn.preprocessing import normalize


class Normalizer():
    def __init__(self):
        self.norms_ = None

    def fit_transform(self, X, y=None):
        X, norms = normalize(X, return_norm=True)
        self.norms_ = norms
        return X

    def inverse_transform(self, X):
        return X * self.norms_[:, None]
