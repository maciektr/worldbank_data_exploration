from sklearn.preprocessing import StandardScaler

from feature_extraction import Autoencoder


class AutoencoderExtractor:
    def __init__(self, feature, root=None, model='autoencoder', feature_set='main_feature_set', n_bottleneck=4,
                 input_len=21):
        if root is None:
            root = '..'
        model_path = f'{root}/feature_extraction/model_weights/{model}_{n_bottleneck}/{feature_set}/{feature}'
        self.autoencoder = Autoencoder(n_bottleneck, input_len=input_len, model_path=model_path)
        self.autoencoder.load_weights()

    def extract_features(self, X, include_norms=True, scale=True):
        Y = self.autoencoder.encode(X, include_norms=include_norms)
        if scale:
            Y = StandardScaler().fit_transform(Y)
        return Y

    def evaluate(self, X):
        return self.autoencoder.compile_and_evaluate(X)
