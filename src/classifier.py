import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


def compute_cov(mat):
    # calculer la moyenne
    mean = mat.mean(axis=0)  # compute the mean

    # normalizer les données
    normalized = mat - mean  # Subtract the mean from all observations

    cov = np.matmul(normalized.T, normalized) / (len(mat) - 1)

    return cov


class LDA(BaseEstimator, ClassifierMixin):
    def __init__(self):
        # self._weight = None
        # self._intercept = None
        # self.classes = None
        pass

    def fit(self, X, y):
        """
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values."""

        self.classes_ = np.unique(y)
        assert len(self.classes_) == 2

        X_a = X[np.where(y == self.classes_[0]), :].squeeze()
        X_b = X[np.where(y == self.classes_[1]), :].squeeze()

        mean_A = np.mean(X_a, axis=0)
        mean_B = np.mean(X_b, axis=0)

        cov_A = compute_cov(X_a)
        cov_B = compute_cov(X_b)

        cov = (cov_A + cov_B) / 2

        # computing estimation
        inv_cov = np.linalg.inv(cov)

        self.intercept_ = -(1 / 2) * (mean_A + mean_B).dot(inv_cov).dot(
            (mean_B - mean_A).T
        )
        self.weight_ = inv_cov.dot((mean_B - mean_A).T)

        return self

    def predict(self, X):
        """X : array-like of shape (n_samples, n_features)
        predicted data."""
        check_is_fitted(self)

        result = self.intercept_ + self.weight_.dot(X.T)
        predict_classes = np.where(result < 0, self.classes_[0], self.classes_[1])
        return predict_classes

    def score(self, X, y):
        """X : array-like of shape (n_samples, n_features)
            evaluating data.
        y : array-like of shape (n_samples,)
            Target values."""

        check_is_fitted(self)
        predict_classes = self.predict(X)
        accuracy = np.where(predict_classes == y, 1, 0).mean()

        return accuracy

    def transform(self, X):
        check_is_fitted(self)
        return self.intercept_ + self.weight_.dot(X.T)
