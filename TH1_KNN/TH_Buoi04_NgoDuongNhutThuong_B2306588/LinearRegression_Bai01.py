import numpy as np
import matplotlib.pyplot as plt

X = np.array([1,2,4])
Y = np.array([3,4,7])
def LR1(X, Y, eta, lanlap, theta0, theta1):
    m = len(X)
    for t in range(lanlap):
        for i in range(m):
            h = theta0 + theta1 * X[i]
            theta0 = theta0 + eta * (Y[i] - h)
            theta1 = theta1 + eta * (Y[i] - h) * X[i]
    return [theta0, theta1]

eta = 0.1 #Toc do hoc

theta1 = LR1(X, Y, eta, 1, 0, 1) # 1 buoc lap
X1 = np.array([1, 6])
Y1 = theta1[0] + theta1[1] * X1

theta2 = LR1(X, Y, eta, 2, 0, 1) # 2 buoc lap
X2 = np.array([1, 6])
Y2 = theta2[0] + theta2[1] * X2

plt.axis=[0,7,0,10]
plt.scatter(X,Y)

plt.plot(X1, Y1, 'violet')
plt.plot(X2, Y2, 'green')

plt.xlabel('Gia tri cua X')
plt.ylabel('Gia tri cua Y')
plt.show()