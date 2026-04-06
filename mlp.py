import numpy as np
import matplotlib.pyplot as plt

def f(x,y):
    return np.sin(np.sqrt(x**2+y**2))+0.5*np.cos(2*x+2*y)

def CalculerAltitude(m):
    a = np.zeros(2000)
    for i in range(2000):
            a[i]= f(m[i][0],m[i][1])
    return a

# normalize data to [0,1]         
def normalize(X):
    X_norm = np.zeros(X.shape)
    for i in range(X.shape[1]):
        X_norm[:,i] = (X[:,i] - np.min(X[:,i])) / (np.max(X[:,i]) - np.min(X[:,i]))
    return X_norm

# normalize data to [-1,1]
def nor(x):
     return x / 5


#create random data in the range [-5,5] 
rng = np.random.default_rng()
data = rng.uniform(-5,5,(2000,2))


data_n = nor(data)
Z = CalculerAltitude(data_n)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(data_n[:,0], data_n[:,1], Z)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Z')

plt.show()

