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


data_n = data
Z = CalculerAltitude(data_n)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(data_n[:,0], data_n[:,1], Z)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Z')

#plt.show()
##############################################
def init(inputSize,hidenLayersVector,ouputSize):
     weights = []
     l = 0
     w0 = np.random.rand(inputSize,hidenLayersVector[0]) * np.sqrt(2/inputSize)
     weights.append( w0 )
     for i in range(len(hidenLayersVector)):
          if i == 0:
               continue
          weights.append(np.random.rand(hidenLayersVector[i-1],hidenLayersVector[i]) * np.sqrt(2/hidenLayersVector[i-1]))
          l = i
     weights.append(np.random.rand(hidenLayersVector[l],ouputSize) * np.sqrt(2/hidenLayersVector[l]))     
     return weights

s = init(2,[5,4],1)
print(s)
print(s[0])
print(s[0].shape)
def Relu(x):
     print(" RELU ",x)

     for i in range(len(x)):
          if x[i] < 0:
               x[i] = 0
          else :
               x[i] = x[i]
     return x          
           
def forward(w,input):
     Z = []
     
     i = input.reshape(1,2)
     z0 =  input.reshape(1,2) @ w[0]
     print("z0 = ",z0)
     print(" Z0 length ",len(z0))
     print("z0 shape",z0.shape)
     z01 = z0
     for i in range(len(z0)):
          z01[i] = Relu(z0[i])
     print("z01 ",z01)    
     print("z01 i ",z01[0])    
     Z.append(z01)
     j = 1 
     while(1):
          z =  Z[j-1] @ w[j]
          z012 = z
          for i in range(len(z)):
               z012[i] = Relu(z[i])
          Z.append(z012)
          j+=1
          if j == len(w)-1:
               print(" W DIMS ",w[j].shape)
               print(" Z DIMS ",Z[j-1].shape)
               #ddd
               z =  Z[j-1] @ w[j]
               Z.append(z)
               print("result ",z)
               z012 = z
               break
     return Z

c = forward(s,data_n[0])
print(c)
