import numpy as np 
import matplotlib.pyplot as plt

t=np.array([-5,-3,0,0.698,1])
print (t.shape)

t1e5=np.array([72.11,72.11,97.23,97.29,97.29])
# t1e5=0.003/(t1e5*t)

t1e6=np.array([71.59,71.59,97.355,97.455,97.455])
# t1e6=0.012/(t1e6*t)

plt.figure(1)
fig, ax = plt.subplots()
plt.xlabel("log(C) base 10")
plt.ylabel("Percentage Accuracy(%)")
plt.title("Percentage Accuracy vs log(C)")
ax.plot(t,t1e5,'b--',label='test')
ax.plot(t,t1e6,'r--',label='cross-validation')
# ax.plot(t,t1e7,'g--',label='1e7')
# ax.plot(t,t1e8,'b--',label='1e8')
# ax.plot(t,t2e8,'m--',label='2e8')
legend = ax.legend(loc='upper center',fontsize='x-small')
legend.get_frame().set_facecolor('#00FFCC')
plt.savefig("Accuracy"+'.png')

# print t1e6
