import numpy as np
import matplotlib.pyplot as plt

print("MLP from scratch")
a = np.linspace(1,10,9)
b=  np.array([[1,2,3],[2,3,3]])
d = b.reshape(b.size)
d = d.reshape(d.size,1)
c = d.shape
print(c)
print(d)
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
l = CalculerAltitude(data)
print(l)
print(l.shape)
print(l.size)
