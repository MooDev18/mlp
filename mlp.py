import numpy as np
import matplotlib.pyplot as plt
import time


def f(x,y):
    return np.sin(np.sqrt(x**2+y**2)) + 0.5*np.cos(2*x+2*y)


def CalculerAltitude(m):
    a = np.zeros(len(m))
    for i in range(len(m)):
        a[i] = f(m[i][0], m[i][1])
    return a


# normalize data to [0,1]
def normalize(X):
    X_norm = np.zeros(X.shape)
    for i in range(X.shape[1]):
        X_norm[:,i] = (X[:,i] - np.min(X[:,i])) / (np.max(X[:,i]) - np.min(X[:,i]))
    return X_norm


# create random data
rng = np.random.default_rng()
data = rng.uniform(-5,5,(2000,2))

Za = CalculerAltitude(data)

X = normalize(data)
Y = Za.reshape(-1,1)


##############################################
# NETWORK INITIALIZATION
##############################################

def init(inputSize, hiddenLayers, outputSize):

    layers = [inputSize] + hiddenLayers + [outputSize]

    weights = []
    biases = []

    for i in range(len(layers)-1):
        w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2/layers[i])
        b = np.zeros((1,layers[i+1]))

        weights.append(w)
        biases.append(b)

    return weights, biases


##############################################
# ACTIVATION Linear(without activation function)
##############################################

def linear(x):
     return x
def linear_deriv(x):
    return np.ones_like(x)

################################################
# ACTIVATION Relu
################################################
def Relu(x):
    return np.maximum(0,x)

def relu_deriv(x):
    return (x > 0).astype(float)


##############################################
# LOSS
##############################################

def mse(y, y_pred):
    return np.mean((y - y_pred)**2)


##############################################
# FORWARD
##############################################

def forward(weights, biases, X):

    Zs = []
    As = [X]

    A = X

    for i in range(len(weights)-1):

        Z = A @ weights[i] + biases[i]
        A = Relu(Z)

        Zs.append(Z)
        As.append(A)

    # output layer (linear)
    Z = A @ weights[-1] + biases[-1]

    Zs.append(Z)
    As.append(Z)

    return Zs, As


##############################################
# BACKWARD
##############################################

def backward(weights, biases, Zs, As, X, y_true):

    m = X.shape[0]

    grad_w = [None]*len(weights)
    grad_b = [None]*len(weights)

    delta = (As[-1] - y_true) / m

    for i in reversed(range(len(weights))):

        A_prev = As[i]

        grad_w[i] = A_prev.T @ delta
        grad_b[i] = np.sum(delta, axis=0, keepdims=True)

        if i > 0:
            delta = delta @ weights[i].T
            delta = delta * relu_deriv(Zs[i-1])

    return grad_w, grad_b


##############################################
# UPDATE
##############################################

def update(weights, biases, grad_w, grad_b, lr):

    for i in range(len(weights)):
        weights[i] -= lr * grad_w[i]
        biases[i]  -= lr * grad_b[i]

    return weights, biases
##############################################
# UPDATE with Momentum
##############################################

def update_momentum(weights, biases, grad_w, grad_b, v_w, v_b, lr, beta):

    for i in range(len(weights)):

        v_w[i] = beta * v_w[i] + (1 - beta) * grad_w[i]
        v_b[i] = beta * v_b[i] + (1 - beta) * grad_b[i]

        weights[i] -= lr * v_w[i]
        biases[i]  -= lr * v_b[i]

    return weights, biases, v_w, v_b

##############################################
# TRAINING
##############################################

np.random.seed(0)

weights, biases = init(2, [64,64], 1)

# momentum buffers
v_w = [np.zeros_like(w) for w in weights]
v_b = [np.zeros_like(b) for b in biases]
lr = 0.01
beta = 0.9
epochs = 5000
batch_size = 256
start_time = time.time()
loss_history = []

print("Training...")

for epoch in range(epochs):

    idx = np.random.permutation(len(X))
    X_s = X[idx]
    Y_s = Y[idx]

    for start in range(0, len(X), batch_size):

        xb = X_s[start:start+batch_size]
        yb = Y_s[start:start+batch_size]

        Zs, As = forward(weights, biases, xb)

        gw, gb = backward(weights, biases, Zs, As, xb, yb)
        
        #weights, biases = update(weights, biases, gw, gb, lr) # simple update
        weights, biases, v_w, v_b = update_momentum(weights, biases, gw, gb, v_w, v_b, lr, beta)

    _, As_all = forward(weights, biases, X)

    l = mse(Y, As_all[-1])

    loss_history.append(l)

    if epoch % 50 == 0:
        print("epoch", epoch, "loss", l)

end_time = time.time()
print("Training time:", end_time - start_time)


##############################################
# PREDICTIONS
##############################################

_, As_full = forward(weights, biases, X)

Z_pred = As_full[-1].flatten()


##############################################
# PLOTS
##############################################

fig = plt.figure(figsize=(18,5))

ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(data[:,0], data[:,1], Za, s=2)
ax1.set_title("True Surface")


ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(data[:,0], data[:,1], Z_pred, s=2)
ax2.set_title("Predicted Surface")

ax3 = fig.add_subplot(133)
ax3.plot(loss_history)
ax3.set_title("Training Loss")

plt.show()