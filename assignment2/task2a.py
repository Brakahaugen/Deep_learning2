import numpy as np
import math
import utils
import typing
np.random.seed(1)


def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785]
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"

    mean, std = calculate_mean_std(X)

    X = (X - mean) / std

    new_X = np.ones((X.shape[0], X.shape[1]+1))
    new_X[:,:-1] = X

    return new_X

def calculate_mean_std(X: np.ndarray):
    """
    Args:
        X: X_train of shape [batch size, 784] in the range (0, 255)
    Returns:
        mean: Mean pixel value
        std: Standard deviation of the pixel value
    """

    mean = np.sum(X) / (X.shape[0]*X.shape[1])
    print(mean)
    std = math.sqrt(np.sum((X - mean)**2) / (X.shape[0]*X.shape[1]))
    print(std)
    return mean, std


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"

    ce = targets * np.log(outputs)
    ce = -np.sum(ce)/targets.shape[0]
    return ce


class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3a hyperparameter
                 use_improved_weight_init: bool  # Task 3c hyperparameter
                 ):
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid

        #Buffers for the backpropagation
        self.a = []
        

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Initialize the weights
        self.ws = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)

            if use_improved_weight_init:
                print("Initializing improved weight to shape:", w_shape)
                w = np.random.normal(0, 1/np.sqrt(prev), (w_shape))
            else:
                print("Initializing weight to shape:", w_shape)
                w = np.random.uniform(-1, 1, (w_shape))

            self.ws.append(w)
            prev = size

        self.grads = [None for i in range(len(self.ws))]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        #resetting the activation buffer
        self.a = []

        #Handling the first layer
        self.a.append(X.dot(self.ws[0]))
        
        for i in range(1, len(self.neurons_per_layer)):
            if self.use_improved_sigmoid:
                self.a[i - 1] = self.improved_sigmoid(self.a[i - 1])
            else:
                self.a[i - 1] = self.sigmoid(self.a[i - 1])

            self.a.append(self.a[i - 1].dot(self.ws[i]))

        #Activate with the softmax function for the output here
        y = self.softmax(self.a[-1])


        return y
        

    def sigmoid(self, x):
        return np.exp(x)/(np.exp(x) + 1)

    def improved_sigmoid(self, x):
        return 1.7159 * np.tanh(2*x/3)

    def softmax(self, x):
        return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"


        #Output layer backpropagation
        delta = -(targets - outputs)
        self.grads[-1] = (np.dot(delta.T, self.a[-2]).T) / X.shape[0]

        #Hidden layer backpropagation
        for i in range(1, len(self.ws) - 1):
            z = np.dot(self.a[-2-i], self.ws[-1-i])
            if self.use_improved_sigmoid:
                delta = np.dot(self.ws[-i], delta.T).T * self.improved_sigmoid_derivative(z) # * self.ws[1]
            else:
                delta = np.dot(self.ws[-i], delta.T).T * self.sigmoid_derivative(z)

            self.grads[-1-i] = (np.dot(delta.T, self.a[-2-i]).T) / X.shape[0]


        # First layer backpropagation
        z = np.dot(X, self.ws[0])
        if self.use_improved_sigmoid:
            delta = np.dot(self.ws[1], delta.T).T * self.improved_sigmoid_derivative(z) # * self.ws[1]
        else:
            delta = np.dot(self.ws[1], delta.T).T * self.sigmoid_derivative(z)

        self.grads[0] = np.dot(delta.T, X).T / X.shape[0]






        #Adding L2 regularization
        # self.grad += self.l2_reg_lambda*self.w


        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."


    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]

    def sigmoid_derivative(self, x):
        return np.exp(-x) / np.square((1 + np.exp(-x)))

    def improved_sigmoid_derivative(self, x):
        return 1.7159 * 2 / (3 * np.square(np.cosh(2*x/3)))

def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """

    new_Y = np.zeros((Y.shape[0], num_classes))
    #For each training example
    for i in range(Y.shape[0]):
        new_Y[i, (Y[i,0])] = 1

    return new_Y

def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited.
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**2,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist(0.1)
    print("X shape")
    print(X_train.shape)
    print("Y shape")
    print(Y_train.shape)

    mean_pixel_value, std_pixel = calculate_mean_std(X_train)

    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)

    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)
    logits = model.forward(X_train)
    np.testing.assert_almost_equal(
        logits.mean(), 1/10,
        err_msg="Since the weights are all 0's, the softmax activation should be 1/10")

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    gradient_approximation_test(model, X_train, Y_train)
