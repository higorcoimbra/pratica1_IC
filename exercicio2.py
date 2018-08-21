
import numpy as np
import math
import matplotlib.pyplot as plt

P = {'xm1': 2, 'sig1': 1, 'xm2': -2, 'sig2': 1, 'p1': 2, 
			'q1': 0, 'p2': -2, 'q2': 0}
D = [0,0,0,0,0,0,0,0]

def yd(x):
	return x**2

def y1(x):
	return P['p1']*x+P['q1']

def y2(x):
	return P['p2']*x+P['q2']

def y(x):
	return ((w1(x)*y1(x))+(w2(x)*y2(x)))/(w1(x)+w2(x))

def w1(x):
	return math.exp((-1/2)*(((x-P['xm1'])/P['sig1'])**2))

def w2(x):
	return math.exp((-1/2)*(((x-P['xm2'])/P['sig2'])**2))

def error(x):
	return (1/2)*math.pow((y(x)-yd(x)), 2)

def derivative(x):
	D[0] = (y(x)-yd(x))*(w1(x)/(w1(x)+w2(x)))*x
	D[1] = (y(x)-yd(x))*(w2(x)/(w1(x)+w2(x)))*x
	D[2] = (y(x)-yd(x))*(w1(x)/(w1(x)+w2(x)))
	D[3] = (y(x)-yd(x))*(w2(x)/(w1(x)+w2(x)))
	D[4] = (y(x)-yd(x))*w2(x)*(y1(x)-y2(x))/((w1(x)+w2(x))**2)*w1(x)*(x-P['xm1']/(P['sig1']**2))
	D[5] = (y(x)-yd(x))*w1(x)*(y2(x)-y1(x))/((w1(x)+w2(x))**2)*w2(x)*(x-P['xm2']/(P['sig2']**2))
	D[6] = (y(x)-yd(x))*w2(x)*(y1(x)-y2(x))/((w1(x)+w2(x))**2)*w1(x)*((x-P['xm1'])**2/(P['sig1']**3))
	D[7] = (y(x)-yd(x))*w1(x)*(y2(x)-y1(x))/((w1(x)+w2(x))**2)*w2(x)*((x-P['xm2'])**2/(P['sig2']**3))

Xin = np.arange(-2, 2.04, 0.04)
time = 50
alpha = 0.01
E = []
for t in range(0, time):
	index = np.random.permutation(100)
	e = 0
	for i in index:
		e += error(Xin[i])
		print(e)
		derivative(Xin[i])
		j = 0
		for key in P:
			P[key] -= alpha*D[j]
			j += 1
	E.append(e)

plt.plot(E)
plt.show()	
"""
plt.xlabel("x");
plt.ylabel("f(x)");
plt.plot(x,Y, label="Expected output - f(x) = x²-0.2x-0.1");
plt.plot(x,Z, label="Adjusted output");
plt.legend()
plt.show();
"""