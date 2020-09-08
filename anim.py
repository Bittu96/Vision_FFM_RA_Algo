import numpy as np
from numpy import sin,cos,pi,tan
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()

x = np.arange(-20, 20, 1)

f = 50*sin(x) 
line, = ax.plot(x, f,lw = 2)
#line, = ax.plot([10,10+sin(x)], [10,10+cos(x)], 'black', lw=2)


def animate(t):
    line.set_ydata(t/1.000000) # update the data
    return line,


# Init only required for blitting to give a clean slate.
def init():
    line.set_ydata(np.ma.array(x, mask=True))
    return line,

ani = animation.FuncAnimation(fig, animate, np.arange(1, 200), init_func=init,
                              interval=25, blit=True)
plt.show()
