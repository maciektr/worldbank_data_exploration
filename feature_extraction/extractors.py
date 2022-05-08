from sklearn.preprocessing import StandardScaler

from feature_extraction import Autoencoder


class AutoencoderExtractor:
    """
    Enables easy access to autoencoders and feature extraction
    """

    def __init__(
        self,
        feature,
        root=None,
        model="autoencoder",
        n_bottleneck=4,
        input_len=21,
        trim_feature=True,
    ):
        """
        :param feature: Feature name
        :param root: Root directory of the repo
        :param model: Model name
        :param n_bottleneck: Number of neurons in the bottleneck
        :param input_len: Length of the time series
        :param trim_feature: Trim the feature length to 65 to avoid problems with saving/reading the model weights
        """
        if root is None:
            root = ".."
        if trim_feature:
            feature = feature[:65]
        model_path = (
            f"{root}/feature_extraction/model_weights/{model}_{n_bottleneck}/{feature}"
        )
        self.autoencoder = Autoencoder(
            n_bottleneck, input_len=input_len, model_path=model_path
        )
        self.autoencoder.load_weights()

    def extract_features(self, X, include_norms=True, scale=True):
        """
        Use the autoencoder to extract features from the time series
        :param X: An array of shape (n x m) with n time series, each of length m
        :param include_norms: Determines whether to put norms as the first feature. Requires normalize=True
        :param scale: If True, features will be transformed with StandardScaler
        :return: Extracted features
        """
        Y = self.autoencoder.encode(X, include_norms=include_norms)
        if scale:
            Y = StandardScaler().fit_transform(Y)
        return Y

    def evaluate(self, X):
        """
        Evaluates the autoencoder
        :param X: An array of shape (n x m) with n time series, each of length m
        :return: Loss value
        """
        return self.autoencoder.compile_and_evaluate(X)
