
import numpy as np
import math
import matplotlib.pyplot as plt

P = {'p1': 2.0, 'q1': 0, 'p2': -2.0, 'q2': 0, 'xm1': 2.0, 'sig1': 1.0, 'xm2': -2.0, 'sig2': 1.0}
D = {'p1': 0, 'q1': 0, 'p2': 0, 'q2': 0, 'xm1': 0, 'sig1': 0, 'xm2': 0, 'sig2': 0}

def yd(x):
	return x**2

def y1(x):
	return P['p1']*x+P['q1']

def y2(x):
	return P['p2']*x+P['q2']

def y(x):
	return ((w1(x)*y1(x))+(w2(x)*y2(x)))/(w1(x)+w2(x))

def w1(x):
	return math.exp((-1/2.0)*(((x-P['xm1'])/P['sig1'])**2))

def w2(x):
	return math.exp((-1/2.0)*(((x-P['xm2'])/P['sig2'])**2))

def error(x):
	return (1/2.0)*math.pow((y(x)-yd(x)), 2)

def derivative(x):
	D['p1'] = (y(x) - yd(x))*(w1(x)/float((w1(x)+w2(x))))*x
	D['p2'] = (y(x) - yd(x))*(w2(x)/float((w1(x)+w2(x))))*x
	D['q1'] = (y(x) - yd(x))*(w1(x)/float((w1(x)+w2(x))))
	D['q2'] = (y(x) - yd(x))*(w2(x)/float((w1(x)+w2(x))))
	D['xm1'] = (y(x) - yd(x))*w2(x)*(y1(x)-y2(x))/float((w1(x)+w2(x))**2)*w1(x)*(x-P['xm1']/float(P['sig1']**2))
	D['xm2'] = (y(x) - yd(x))*w1(x)*(y2(x)-y1(x))/float((w1(x)+w2(x))**2)*w2(x)*(x-P['xm2']/float(P['sig2']**2))
	D['sig1'] = (y(x) - yd(x))*w2(x)*(y1(x)-y2(x))/float((w1(x)+w2(x))**2)*w1(x)*((x-P['xm1'])**2/float(P['sig1']**3))
	D['sig2'] = (y(x) - yd(x))*w1(x)*(y2(x)-y1(x))/float((w1(x)+w2(x))**2)*w2(x)*((x-P['xm2'])**2/float(P['sig2']**3))

Xin = np.arange(-2, 2.04, 0.04)
time = 500
alpha = 0.01
E = []
for t in range(0, time):
	index = np.random.permutation(100)
	e = 0
	for i in index:
		e += error(Xin[i])
		derivative(Xin[i])
		for key in P:
			P[key] -= alpha*D[key]
	E.append(e)


plt.figure(1)

plt.plot(E)
plt.xlabel("Time");
plt.ylabel("Error");

Yd = []
Y = []

for x in Xin:
	Y.append(y(x))
	Yd.append(yd(x))

plt.figure(2)

plt.subplot(211)
plt.xlabel("x");
plt.ylabel("f(x)");
plt.plot(Xin,Yd, label="Expected output - f(x) = x²");
plt.legend()

plt.subplot(212)
plt.xlabel("x");
plt.ylabel("f(x)");
plt.plot(Xin,Y, label="Adjusted output");
plt.legend()
plt.show()