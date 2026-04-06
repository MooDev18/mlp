import numpy as np
import matplotlib.pyplot as plt

def f(x,y):
    return np.sin(np.sqrt(x**2+y**2))+0.5*np.cos(2*x+2*y)
def CalculerAltitude(m):
    a = np.zeros(2000)
    for i in range(2000):
            a[i]= f(m[i][0],m[i][1])
    return a         
rng = np.random.default_rng()
data = rng.uniform(-5,5,(2000,2))
print(data)
print(data.shape)
print (f(2,3))
Z = CalculerAltitude(data)
print(Z)
print(Z.shape)
print(Z.size)
print("DD")
