import numpy as np
from sklearn.metrics import accuracy_score

# Inputs
x = np.array([2, 3, 1, -3])
# Randomly choosen weights
w = np.array([[0.34, 0.1, 0.5, -0.7],
              [0.14, 0.4, 0.2, -0.5],
              [-0.34, -0.3, -0.5, 0.7]])
# Randomly choosen bias
b = np.array([0, 1, -2])
# Wanted Output
y = np.array([0.5, 0.1, 0.9])
# Estimated Output
y_est = np.array([np.zeros(len(y))])
# Learning Rate (for not overshooting)
learning_rate = 0.1


# Activation Function
def sigmoid(w, b, x):
    y_est = np.dot(w, x)+b
    return 1 / (1+np.exp(-y_est))


# Gradient Descent
def train(w, b, x):
    y_est = sigmoid(w, b, x)
    error = y - y_est
    error_continues = error * np.dot(y_est, 1-y_est)
    d_w = learning_rate * np.outer(error_continues, x)
    d_b = learning_rate * error_continues
    w = w+d_w
    b = b+d_b
    return w, b, y_est, error


def cost_func(y, y_est):
    err = list(map(lambda a, b: (a-b)**2, y, y_est))
    return np.sum(err)


def calc_accuracy(y, y_est):
    total_accuracy = []
    for a, b in zip(y, y_est):
        if(a == 0 or b == 0):
            total_accuracy.append(0)
        else:
            if(a > b):
                total_accuracy.append(b/a)
            else:
                total_accuracy.append(a/b)
    return np.prod(total_accuracy)


def print_results(w, b, y_est, iterations):
    print("Your new weights are: \n"+str(w)+"\n")
    print("Your new biases are: \n"+str(b)+"\n")
    print("Your new estimation is \n"+str(y_est)+"\n")
    print("Your models accuracy is: \n"+str(calc_accuracy(y, y_est)
                                            * 100) + " % with "+str(iterations)+" iterations")


"""Error Handling"""
if(not len(b) == len(y)):
    raise Exception("\nOutput not same dimension as bias ")
for output in y:
    if(output < 0 or output > 1):
        raise Exception("\nOutputs not between 0 and 1")
dim_y, dim_x = w.shape
if(not dim_y == len(y)):
    raise Exception("\nWeights rows not same as output length")
if(not dim_x == len(x)):
    raise Exception("\nWeights columns not same as input length")


""" Training """
score = 97
max_iterations = 1000
accuracy_score = calc_accuracy(y, sigmoid(w, b, x))
iterations = 0
while(accuracy_score*100 < score):
    w, b, y_est, error = train(w, b, x)
    accuracy_score = calc_accuracy(y, y_est)
    iterations += 1
    if(iterations == max_iterations):
        print("\nReached max. Iterations!")
        break

print_results(w, b, y_est, iterations)
calc_accuracy(y, y_est)
