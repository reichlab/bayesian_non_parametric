import matplotlib.pyplot as plt





plt.subplot(223)
plt.plot(range(1,5),[125,121,108,172], label="GP")
plt.plot(range(1,5),[140,142,142,143], label="KCDE")
# Place a legend to the right of this smaller subplot.
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel("Number of lags")
plt.ylabel("MSE")
plt.ylim([0,200])
plt.xlim([0,5])
plt.show()