from scipy.interpolate import spline
import matplotlib.pyplot as plt
import numpy as np

T = [6, 7, 8, 9,  12]
xnew = np.linspace(np.amin(T),np.amax(T),30) #300 represents number of points to make between T.min and T.max
power = [6, 7, 8, 9, 12]

plt.plot(T,power)
plt.show()
power_smooth = spline(T,power,xnew)

plt.plot(T,power_smooth,'ro')
plt.show()

