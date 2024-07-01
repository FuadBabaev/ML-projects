import numpy as np
import scipy
from scipy.special import expit
from scipy.special import logsumexp


class BaseLoss:
    """
    Base class for loss function.
    """

    def func(self, X, y, w):
        """
        Get loss function value at w for each sample.
        """

        weights, bias = w[1:], w[0]
        logits = -y * (X @ weights + bias)

        # Base case loss function
        loss_terms = np.log(1 + np.exp(logits))
        # We can ignore 1 for large numbers and set loss as logits
        loss_terms = np.where(logits > 100, logits, loss_terms)

        return np.mean(loss_terms)

    def grad(self, X, y, w):
        """
        Get loss function gradient value at w.
        """
        weights, bias = w[1:], w[0]
        logits = -y * (X @ weights + bias)
        X_augmented = np.hstack((np.ones((X.shape[0], 1)), X))
        return -(X_augmented.T @ ((y * np.exp(logits)) / (1 + np.exp(logits))))
        

class BinaryLogisticLoss(BaseLoss):
    """
    Loss function for binary logistic regression.
    It should support l2 regularization.
    """

    def __init__(self, l2_coef):
        """
        Parameters
        ----------
        l2_coef - l2 regularization coefficient
        """
        self.l2_coef = l2_coef

    def func(self, X, y, w):
        """
        Get loss function value for data X, target y and coefficient w; w = [bias, weights].

        Parameters
        ----------
        X : scipy.sparse.csr_matrix or numpy.ndarray
        y : 1d numpy.ndarray
        w : 1d numpy.ndarray

        Returns
        -------
        : float
        """
        base_loss = super().func(X, y, w)
        l2_penalty = self.l2_coef * np.sum(w[1:] ** 2)
        total_loss = base_loss + l2_penalty
        return total_loss

    def grad(self, X, y, w):
        """
        Get loss function gradient for data X, target y and coefficient w; w = [bias, weights].

        Parameters
        ----------
        X : scipy.sparse.csr_matrix or numpy.ndarray
        y : 1d numpy.ndarray
        w : 1d numpy.ndarray

        Returns
        -------
        : 1d numpy.ndarray
        """
        base_grad = super().grad(X, y, w)
        l2_grad = self.l2_coef * np.r_[0, w[1:]]
        total_grad = base_grad / len(y) + 2 * l2_grad
        return total_grad 

loss_function = BinaryLogisticLoss(l2_coef=1.0)
X = np.array([
    [1, 2],
    [3, 4],
    [-5, 6]
])
y = np.array([-1, 1, 1])
w = np.array([1, 2, 3])
right_gradient = np.array([0.33325, 4.3335 , 6.66634])
print(loss_function.grad(X, y, w))