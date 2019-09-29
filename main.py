import numpy as np

# Inputs
x = np.array([2, 3])
# Randomly choosen weights
w = np.array([0.34, 0.1])
# Randomly choosen bias
b = np.array([0.54])
# Wanted Output
y = np.array([0.5])
# Estimated Output
y_est = np.array([0])
# Learning Rate
learning_rate = 0.1


# Activation Function
def sigmoid(w, x, b):
    y_est = np.dot(w, x)+b
    result = 1 / (1+np.exp(-y_est))
    return y_est


# Gradient Descent
def train(w, x, b):
    y_est = sigmoid(w, x, b)
    error = calc_error(y, y_est)
    d_w = learning_rate * error * y_est * (1 - y_est) * x
    d_b = learning_rate * error * y_est * (1 - y_est)
    if(error < 0):
        w = w-d_w
        b = b-d_b
    else:
        w = w+d_w
        b = b+d_b

    return w, b, y_est, error


def calc_error(y, y_est):
    m = len(y)
    quad_err = list(map(lambda a, b: (a-b)**2, y, y_est))
    error = np.sum(quad_err) / m
    return error


if __name__ == "__main__":
    # Training until error is lower then threshold
    threshold = 0.0001
    while(calc_error(y, y_est) > threshold):
        w, b, y_est, error = train(w, x, b)

    print("Your new weights are: "+str(w))
    print("Your new biases are: "+str(b))
    accuracy = str((y_est/y*100))
    print("Your model is "+accuracy + "% accurate!")
