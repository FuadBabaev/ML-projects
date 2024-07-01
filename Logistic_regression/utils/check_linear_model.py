#from modules.losses import BinaryLogisticLoss
import numpy as np
from scipy.special import expit
import time
import numpy.testing as npt

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


class LinearModel:
    def __init__(
        self,
        loss_function,
        batch_size=100,
        step_alpha=1,
        step_beta=0, 
        tolerance=1e-5,
        max_iter=1000,
        random_seed=153,
        **kwargs
    ):
        """
        Parameters
        ----------
        loss_function : BaseLoss inherited instance
            Loss function to use
        batch_size : int
        step_alpha : float
        step_beta : float
            step_alpha and step_beta define the learning rate behaviour
        tolerance : float
            Tolerace for stop criterio.
        max_iter : int
            Max amount of epoches in method.
        """
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_seed = random_seed

    def fit(self, X, y, w_0=None, trace=False, X_val=None, y_val=None):
        """

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, training set.
        y : numpy.ndarray
            1d vector, target values.
        w_0 : numpy.ndarray
            1d vector in binary classification.
            2d matrix in multiclass classification.
            Initial approximation for SGD method.
        trace : bool
            If True need to calculate metrics on each iteration.
        X_val : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, validation set.
        y_val: numpy.ndarray
            1d vector, target values for validation set.

        Returns
        -------
        : dict
            Keys are 'time', 'func', 'func_val'.
            Each key correspond to list of metric values after each training epoch.
        """

        if w_0 is not None:
            w = w_0
        else:
            np.random.seed(self.random_seed)
            w = np.random.randn(X.shape[1] + 1)

        if trace:
            history = {'time': [], 'func': [], 'func_val': [None] * self.max_iter}
            start_time = time.time()
        else:
            history = None
            
        for epoch in range(self.max_iter):
            # Shuffle the data
            perm = np.random.permutation(X.shape[0])
            X_shuffled = X[perm]
            y_shuffled = y[perm]

            for batch_start in range(0, X.shape[0], self.batch_size):
                # Create batch
                X_batch = X_shuffled[batch_start:batch_start+self.batch_size]
                y_batch = y_shuffled[batch_start:batch_start+self.batch_size]

                # Compute gradient
                grad = self.loss_function.grad(X_batch, y_batch, w)

                # Update learning rate
                alpha = self.step_alpha / (epoch + 1) ** self.step_beta

                # Update weights
                w -= alpha * grad

            # Save info for each epoch in trace
            if trace:
                elapsed_time = time.time() - start_time
                history['time'].append(elapsed_time)
                history['func'].append(self.loss_function.func(X, y, w))

                if X_val is not None and y_val is not None:
                    history['func_val'][epoch] = self.loss_function.func(X_val, y_val, w)

                # Check if we fitted the condition
                # Only check if trace is True, so history is not None
                if epoch > 0 and np.abs(history['func'][-1] - history['func'][-2]) < self.tolerance:
                    break

        self.w = w
        if trace:
            return history
        else:
            return w

    def predict(self, X, threshold=0):
        """

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, test set.
        threshold : float
            Chosen target binarization threshold.

        Returns
        -------
        : numpy.ndarray
            answers on a test set
        """
        # Caculate classifier
        classifier = X @ self.w[1:] + self.w[0]
        
        predictions = np.where(classifier >= threshold, 1, -1)
        return predictions

    def get_weights(self):
        """
        Get model weights

        Returns
        -------
        : numpy.ndarray
            1d vector in binary classification.
            2d matrix in multiclass classification.
            Initial approximation for SGD method.
        """
        return self.w[1:]

    def get_objective(self, X, y):
        """
        Get objective.

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix.
        y : numpy.ndarray
            1d vector, target values for X.

        Returns
        -------
        : float
        """
        loss = self.loss_function.func(X, y, self.w)
        return loss

    def get_bias(self):
        """
        Get model bias

        Returns
        -------
        : float
            model bias
        """
        return self.w[0]

def create_simple_dataset():
    X1 = np.random.randint(1, 4, (1000, 10))
    X2 = np.random.randint(-4, 0, (1000, 10))
    X = np.vstack((X1, X2))
    y = np.array([-1] * 1000 + [1] * 1000)
    return X, y

X, y = create_simple_dataset()
loss_function = BinaryLogisticLoss(l2_coef=0.1)
linear_model = LinearModel(
    loss_function=loss_function,
    batch_size=100,
    step_alpha=1,
    step_beta=0,
    tolerance=1e-100,
    max_iter=5,
)
history = linear_model.fit(X, y, trace=True)
print(history)
for key in ['time', 'func', 'func_val']:
    assert key in history
    assert len(history[key]) == 5