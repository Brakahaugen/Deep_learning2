import numpy as np
import utils
import math
import datetime
np.random.seed(1)


def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        new_X: images of shape [batch size, 785] in the range (0, 1)
    """
    new_X = np.zeros((X.shape[0], X.shape[1] + 1))
    for i in range(X.shape[0]):
        new_X[i] = np.append(np.true_divide(X[i], 255), 1)

    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"
    return new_X


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray) -> float:
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, 1]
        outputs: outputs of model of shape: [batch size, 1]
    Returns:
        Cross entropy error (float)
    """

    C = 0
    N = targets.shape[0]

    for i in range(N):
        C -= (targets[i] * math.log(outputs[i])) + ((1 - targets[i]) * math.log(1 - outputs[i]))
    C = C/N

    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    return C


class BinaryModel:

    def __init__(self, l2_reg_lambda: float):
        # Define number of input nodes
        self.I = 785
        self.w = np.zeros((self.I, 1))
        self.grad = np.zeros((self.I, 1))

        # Hyperparameter for task 3
        self.l2_reg_lambda = l2_reg_lambda
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, 1]
        """
        # Sigmoid

        y = np.zeros((X.shape[0], 1))

        for i in range(X.shape[0]):
            y[i] = 1/(1 + math.exp(-np.dot(np.transpose(self.w), X[i]))) 

        return y

    def backward(self, X: np.ndarray, outputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, 1]
            targets: labels/targets of each image of shape: [batch size, 1]
        """
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"
        self.grad = np.zeros_like(self.w)
        assert self.grad.shape == self.w.shape,\
            f"Grad shape: {self.grad.shape}, w: {self.w.shape}"

        for i in range(X.shape[1]):
            for j in range(X.shape[0]):
                # FOr each weight and for each training example do the logistic regression update:
                self.grad[i] += -(targets[j] - outputs[j])*X[j, i]
                #FOr each weight and for each training example add L2-regularization
                self.grad[i] += self.l2_reg_lambda*self.w[i]

            self.grad[i] = self.grad[i] / (j+1)
        


    def zero_grad(self) -> None:
        self.grad = None


def gradient_approximation_test(model: BinaryModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    w_orig = model.w.copy()
    epsilon = 1e-2
    for i in range(w_orig.shape[0]):
        orig = model.w[i].copy()
        model.w[i] = orig + epsilon
        logits = model.forward(X)
        cost1 = cross_entropy_loss(Y, logits)
        model.w[i] = orig - epsilon
        logits = model.forward(X)
        cost2 = cross_entropy_loss(Y, logits)
        gradient_approximation = (cost1 - cost2) / (2 * epsilon)
        model.w[i] = orig
        # Actual gradient
        logits = model.forward(X)
        model.backward(X, logits, Y)
        difference = gradient_approximation - model.grad[i, 0]
        assert abs(difference) <= epsilon**2,\
            f"Calculated gradient is incorrect. " \
            f"Approximation: {gradient_approximation}, actual gradient: {model.grad[i, 0]}\n" \
            f"If this test fails there could be errors in your cross entropy loss function, " \
            f"forward function or backward function"


if __name__ == "__main__":
    print(datetime.datetime.now())
    category1, category2 = 2, 3
    X_train, Y_train, *_ = utils.load_binary_dataset(category1, category2, 0.1)
    X_train = X_train[:100]
    X_train = pre_process_images(X_train)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    # Simple test for forward pass. Note that this does not cover all errors!
    model = BinaryModel(0.0)
    logits = model.forward(X_train)
    np.testing.assert_almost_equal(
        logits.mean(), .5,
        err_msg="Since the weights are all 0's, the sigmoid activation should be 0.5")

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for i in range(2):
        gradient_approximation_test(model, X_train, Y_train)
        model.w = np.random.randn(*model.w.shape)
    print(datetime.datetime.now())

