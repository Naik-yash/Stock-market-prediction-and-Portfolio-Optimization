#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 12:59:24 2018

@author: yash
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
style.use('ggplot')

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
def animate(i):
    graph_data = open('/Users/yash/Downloads/twitter-out.txt','r').read()
    lines = graph_data.split('/n')
    xs = []
    ys = []
    x=0
    y=0
    for l in lines:
        x+=1
        if "pos" in l:
            y+=1
        elif "neg" in l:
            y-=0.3
        
        xs.append(x)
        ys.append(y)
    ax1.clear()
    ax1.plot(xs, ys)
ani = animation.FuncAnimation(fig, animate, interval=500)
plt.show()