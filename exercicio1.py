
import numpy as np
import math
import matplotlib.pyplot as plt

# abertura do arquivo de entrada
file = open('./entrada_exercicio1')
# leitura do arquivo
x = []
y = []

# first membership function
def mf1(x):
	return 0.67*x+0.33

# second membership function
def mf2(x):
	return -0.67*x+0.33

def createBeta(x):
	beta = np.zeros(shape=(len(x),2))
	for i, item in enumerate(x):
		w1 = mf1(item)
		w2 = mf2(item)
		beta[i][0] = w1 / (w1+w2)
		beta[i][1] = w2 / (w1+w2)
	return beta

def createX(B, input):
	X = np.zeros(shape=(len(B[:]),4))

	for i, value in enumerate(input):
		X[i][0] = B[i][0]
		X[i][1] = B[i][1]
		X[i][2] = B[i][0]*input[i]
		X[i][3] = B[i][1]*input[i]

	return X

for pair in file:
	x.append(float(pair.split(' ')[0]))
	y.append(float(pair.split(' ')[1]))

Y = np.array(y)
X = createX(createBeta(x), x)
Xinv = np.linalg.pinv(X)
P = np.dot(Xinv, Y)

Z = np.dot(X,P);

print(Z)

plt.figure(1)

plt.subplot(211)
plt.xlabel("x");
plt.ylabel("f(x)");
plt.plot(x,Y, label="Expected output - f(x) = x²-0.2x-0.1");
plt.legend()

plt.subplot(212)
plt.xlabel("x");
plt.ylabel("f(x)");
plt.plot(x,Z, label="Adjusted output");
plt.legend()
plt.show()

