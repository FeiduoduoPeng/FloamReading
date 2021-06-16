import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

M = np.zeros([3,3])
V = np.zeros([3])
Xrange = [-5,5]
Yrange = [-5,5]
Zrange = [-5,5]
ax.plot(Xrange, [0,0], [0,0], c='r')
ax.plot([0,0], Yrange, [0,0], c='g')
ax.plot([0,0], [0,0], Zrange, c='b')
for i in range(100):
    v = np.array([[np.random.uniform(-0.1,0.1)],
                  [np.random.uniform(-2,2)],
                  [np.random.uniform(-5,5)]])
    v_ = np.squeeze(v)
    V += v_
    ax.plot([0,v_[0]], [0,v_[1]], [0,v_[2]], c='pink')
    m = v@v.transpose()
    print(v)
    print(m)
    M += m

w, v = np.linalg.eig(M)
# print(w)
# print(v)
w = list(w)
idx = w.index(max(w))
v_max = v[:,idx]
ax.plot([0,v_max[0]], [0,v_max[1]], [0,v_max[2]], c='k')
print(v_max)
V = V/np.linalg.norm(V)
ax.plot([0,V[0]], [0,V[1]], [0,V[2]], c='y')
plt.show()