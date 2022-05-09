import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers as k_layers

from feature_extraction.utils import Normalizer


class Autoencoder:
    """
    Class implementing an autoencoder for time series feature extraction
    """

    def __init__(self, n_bottleneck, input_len=21, model_path=None):
        """
        :param n_bottleneck: Number of neurons in the bottleneck
        :param input_len: Length of the time series
        :param model_path: Path to save model weights and loss history. If None, will be set to
        f'model_weights/autoencoder/{n_bottleneck}'
        """
        self.n_bottleneck = n_bottleneck
        self.input_len = input_len
        self.model_path = model_path
        if model_path is None:
            self.model_path = f"model_weights/autoencoder_{n_bottleneck}"

        self.encoder, self.autoencoder = self._build_model()
        self.history = None

    def compile_and_train(
        self,
        X,
        batch_size=256,
        n_epochs=1000,
        lr=0.001,
        lr_patience=100,
        loss_scale=10_000,
        verbose=1,
        save=True,
        normalize=True,
    ):
        """
        Compile and train the autoencoder
        :param X: An array of shape (n x m) with n time series, each of length m
        :param batch_size: batch size
        :param n_epochs: number of epochs
        :param lr: learning rate. If assigned to None, and the model weights have been loaded (load_weights),
        the learning rate from the loaded model will be used.
        :param lr_patience: patience for the ReduceLROnPlateau callback. If None, learning rate is constant
        :param loss_scale: Loss will be multiplied by this number to increase readability
        :param verbose: verbose parameter for training the model
        :param save: True: save the weights and history to self.model_path; False: do not save
        :param normalize: Determines whether to normalize the data or not
        :return: self
        """
        if normalize:
            X = Normalizer().fit_transform(X)

        self.autoencoder.compile(
            tf.keras.optimizers.Adam(learning_rate=lr),
            loss="mae",
            loss_weights=loss_scale,
        )
        # for some reason, when compiling after load_weights, lr is set to the value of the model whose weights are
        # loaded. Therefore, lr is set again
        if lr is not None:
            self.autoencoder.optimizer.learning_rate.assign(lr)

        callbacks = []
        if lr_patience is not None:
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="loss", factor=0.5, patience=lr_patience
                )
            )

        self.history = self.autoencoder.fit(
            X,
            X,
            batch_size=batch_size,
            epochs=n_epochs,
            verbose=verbose,
            callbacks=callbacks,
        ).history
        if save:
            self.autoencoder.save_weights(f"{self.model_path}/weights")
            pd.DataFrame(self.history).to_csv(f"{self.model_path}/loss_log.csv")

        return self

    def load_weights(self, path=None, load_history=False):
        """
        Load weights and history from self.model_path
        :return: self
        """
        if path is None:
            path = self.model_path

        if load_history:
            self.history = pd.read_csv(f"{path}/loss_log.csv")
        self.autoencoder.load_weights(f"{path}/weights").expect_partial()

        return self

    def encode(self, X, normalize=True, include_norms=True):
        """
        Extract features
        :param X: Matrix (n x m) of n time series, each of length m
        :param normalize: Determines whether to normalize the data or not
        :param include_norms: Determines whether to put norms as the first feature. Requires normalize=True
        :return: Extracted features
        """
        normalizer = Normalizer()
        if normalize:
            X = normalizer.fit_transform(X)

        Y = self.encoder.predict(X)
        if normalize and include_norms:
            norms = normalizer.norms_.reshape((-1, 1))
            Y = np.hstack([norms, Y])

        return Y

    def predict(self, X, normalize=True):
        """
        Autoencodes the data
        :param X: Matrix (n x m) of n time series, each of length m
        :param normalize: Determines whether to normalize the data or not
        :return: Autoencoded data
        """
        normalizer = Normalizer()
        if normalize:
            X = normalizer.fit_transform(X)

        Y = self.autoencoder.predict(X)
        if normalize:
            Y = normalizer.inverse_transform(Y)

        return Y

    def compile_and_evaluate(self, X, normalize=True, loss_scale=10_000):
        """
        Compile and evaluate model on X
        :param X: Matrix (n x m) of n time series, each of length m
        :param normalize: Determines whether to normalize the data or not
        :param loss_scale: Loss will be multiplied by this number to increase readability
        :return: Loss value
        """
        self.autoencoder.compile("adam", loss="mae", loss_weights=loss_scale)
        if normalize:
            X = Normalizer().fit_transform(X)
        return self.autoencoder.evaluate(X, X)

    def summary(self):
        """
        :return: Summary of the full model
        """
        return self.autoencoder.summary()

    def _build_model(self):
        dense_len = (self.input_len + 8 - 1) // 8 * 128

        inputs = k_layers.Input((self.input_len,))
        x = k_layers.Reshape((-1, 1))(inputs)

        x = k_layers.Conv1D(32, 3, activation="relu", padding="same")(x)
        x = k_layers.MaxPooling1D(padding="same")(x)

        x = k_layers.Conv1D(64, 3, activation="relu", padding="same")(x)
        x = k_layers.MaxPooling1D(padding="same")(x)

        x = k_layers.Conv1D(128, 3, activation="relu", padding="same")(x)
        x = k_layers.MaxPooling1D(padding="same")(x)

        x = k_layers.Flatten()(x)
        encoded_outputs = k_layers.Dense(self.n_bottleneck)(x)
        x = k_layers.Dense(dense_len)(encoded_outputs)
        x = k_layers.Reshape((-1, 128))(x)

        x = k_layers.Conv1D(128, 3, activation="relu", padding="same")(x)
        x = k_layers.UpSampling1D()(x)

        x = k_layers.Conv1D(64, 3, activation="relu", padding="same")(x)
        x = k_layers.UpSampling1D()(x)

        x = k_layers.Conv1D(32, 3, activation="relu", padding="same")(x)
        x = k_layers.UpSampling1D()(x)

        x = k_layers.Flatten()(x)
        outputs = k_layers.Dense(self.input_len)(x)

        encoder = tf.keras.Model(inputs, encoded_outputs)
        autoencoder = tf.keras.Model(inputs, outputs)

        return encoder, autoencoder


if __name__ == "__main__":
    autoencoder = Autoencoder(4)
    print(autoencoder.summary())
    print(
        "=============================================================================="
    )
    print(
        "=============================================================================="
    )
    print(
        "=============================================================================="
    )
    print(autoencoder.encoder.summary())

    X = np.random.random((100, 21))
    autoencoder.compile_and_train(X, save=False, n_epochs=500)
    # autoencoder.compile_and_train(X, save=True, n_epochs=500)

    print(autoencoder.encode(X[:5]))
    # autoencoder.load_weights()
    # print(autoencoder.encode(X[:5]))

    print(
        "=============================================================================="
    )
    print(X[3])
    print(autoencoder.predict(X)[3])
