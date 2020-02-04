import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import cross_entropy_loss, BinaryModel, pre_process_images
import datetime
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: BinaryModel) -> float: 
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 1]
        model: model of class BinaryModel
    Returns:
        Accuracy (float)
    """
    # Task 2c


    predictions = model.forward(X)
    correct_predictions = 0
    total_predictions = X.shape[0]

    for i in range(X.shape[0]):
        #Find the prediction
        p = 0
        if predictions[i,0] >= 0.5:
            p = 1
        #does it equal the target? add one to corrects_predictions
        if p == targets[i,0]:
            correct_predictions += 1        

    accuracy = correct_predictions/total_predictions
    return accuracy

def early_stop(val_loss: []):
    """
    Args:
        X: val_loss: List that holds values for loss at each step.
    Returns:
        True: If we should stop training
        False: If we should keep going
    """

    #Current criteria:
        # - Consistently worse loss for 3 consecutive steps (one full iteration of the training set)
        # - A difference from best to worst on these consecutive steps of 0.001
    i = len(val_loss)
    loss = 0
    consecutive_steps = 0
    total_loss = 0
    while i > 0:
        i = i - 1
        if loss > val_loss[i]:
            #We have a new consecutive "worse" step
            consecutive_steps += 1
            total_loss += loss - val_loss[i] 
            print(consecutive_steps)
            print(total_loss)
            if consecutive_steps >= 3:
                if total_loss >= 0.007:
                    return True
        else:
            consecutive_steps = 0
            total_loss = 0

        loss = val_loss[i]
        
    
    return False

def calculate_l2_norm(weights: np.ndarray):
    print(weights.shape)
    l2_norm = 0
    for w in weights:
        l2_norm += w[0]**2
    return l2_norm
    
def plot_image(weigths: np.ndarray, l2):
    x, y = 28, 28
    im = np.zeros((x,y))

    for i in range(x):
        for j in range(y):
            im[i,j] = weigths[i*x+j]
    
    plt.imshow(im)
    plt.imsave("lambda=" + str(l2) + ".png", im)

    return im








def train(
        num_epochs: int,
        learning_rate: float,
        batch_size: int,
        l2_reg_lambda: float # Task 3 hyperparameter. Can be ignored before this.
        ):
    """
        Function that implements logistic regression through mini-batch
        gradient descent for the given hyperparameters
    """
    def log_reg():
            # X_batch = pre_process_images(X_batch)
            outputs = model.forward(X_batch)       
            # model.w = model.w - learning_rate * (1/batch_size) * 
            model.backward(X_batch, outputs, Y_batch)
            model.w = model.w - learning_rate * model.grad
    


    global X_train, X_val, X_test
    # Utility variables
    num_batches_per_epoch = X_train.shape[0] // batch_size
    num_steps_per_val = num_batches_per_epoch // 5
    train_loss = {}
    val_loss = {}
    train_accuracy = {}
    val_accuracy = {}
    model = BinaryModel(l2_reg_lambda)
    early_stop_list = []

    global_step = 0
    for epoch in range(num_epochs):
        for step in range(num_batches_per_epoch):
            # Select our mini-batch of images / labels
            start = step * batch_size
            end = start + batch_size
            X_batch, Y_batch = X_train[start:end], Y_train[start:end]
            
            #Do the gradient descent:
            outputs = model.forward(X_batch) 
            model.backward(X_batch, outputs, Y_batch)
            model.w = model.w - learning_rate * model.grad


            # Track training loss continuously
            _train_loss = cross_entropy_loss(Y_batch, outputs)
            train_loss[global_step] = _train_loss
            # Track validation loss / accuracy every time we progress 20% through the dataset
            if global_step % num_steps_per_val == 0:
                print("Epoch: " + str(epoch))
                _val_loss = cross_entropy_loss(Y_val, model.forward(X_val))
                early_stop_list.append(_val_loss[0])
                if len(early_stop_list) > 5:
                    early_stop_list.remove(early_stop_list[0])
                val_loss[global_step] = _val_loss

                stop = early_stop(early_stop_list)

                train_accuracy[global_step] = calculate_accuracy(
                    X_train, Y_train, model)
                val_accuracy[global_step] = calculate_accuracy(
                    X_val, Y_val, model)
                print("val_loss:")
                print(val_loss[global_step])

            global_step += 1

    return model, train_loss, val_loss, train_accuracy, val_accuracy


# Load dataset
category1, category2 = 2, 3
validation_percentage = 0.1
X_train, Y_train, X_val, Y_val, X_test, Y_test = utils.load_binary_dataset(
    category1, category2, validation_percentage)

# Preprocess dataset
X_train = pre_process_images(X_train)
X_test = pre_process_images(X_test)
X_val = pre_process_images(X_val)


print(datetime.datetime.now())



# hyperparameters
num_epochs = 25
learning_rate = 0.1
batch_size = 128
l2_reg_lambda = 0
l2_reg_lambdas = [1, 0.1, 0.01, 0.001]
val_accuracys = {}
l2_norms = {}

for l2_reg_lambda in l2_reg_lambdas:
    model, train_loss, val_loss, train_accuracy, val_accuracy = train(
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        l2_reg_lambda=l2_reg_lambda)
    
    val_accuracys[l2_reg_lambda] = val_accuracy 
    l2_norms[l2_reg_lambda] = calculate_l2_norm(model.w)

    im = plot_image(model.w, l2_reg_lambda)
    


print("Final Train Cross Entropy Loss:",
    cross_entropy_loss(Y_train, model.forward(X_train)))
print("Final  Test Entropy Loss:",
    cross_entropy_loss(Y_test, model.forward(X_test)))
print("Final Validation Cross Entropy Loss:",
    cross_entropy_loss(Y_val, model.forward(X_val)))


print("Train accuracy:", calculate_accuracy(X_train, Y_train, model))
print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model))
print("Test accuracy:", calculate_accuracy(X_test, Y_test, model))

# Plot loss
plt.ylim([0., .4]) 
utils.plot_loss(train_loss, "Training Loss")
utils.plot_loss(val_loss, "Validation Loss")
plt.legend()
plt.savefig("binary_train_loss.png")
plt.show()


# Plot accuracy
plt.ylim([0.91, .99])
utils.plot_loss(train_accuracy, "Training Accuracy")
for l2_reg_lambda in l2_reg_lambdas:
    utils.plot_loss(val_accuracys[l2_reg_lambda], "Lambda = " + str(l2_reg_lambda) + " Validation Accuracy")
plt.legend()
plt.savefig("binary_train_accuracy.png")
plt.show()

# Plot the L2_norm
utils.plot_loss(l2_norms, "Weight length")
# for l2_reg_lambda in l2_reg_lambdas:
#     utils.plot_loss(l2_norms[l2_reg_lambda], "Lambda = " + str(l2_reg_lambda) + " Weight size")
plt.legend()
plt.savefig("binary_weight_onLambda.png")
plt.show()





print(datetime.datetime.now())
