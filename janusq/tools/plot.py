'''
Author: name/jxhhhh� 2071379252@qq.com
Date: 2024-04-17 03:33:02
LastEditors: name/jxhhhh� 2071379252@qq.com
LastEditTime: 2024-04-21 05:16:24
FilePath: /JanusQ/janusq/tools/plot.py
Description: 

Copyright (c) 2024 by name/jxhhhh� 2071379252@qq.com, All Rights Reserved. 
'''
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import random


def plot_scatter(reals, predicts, durations, title = ''):
    '''
    description: plot scatter figure. the x-axis are real fidelities and the y-axis are predict fidelities.the coler of points are normalized durations

    '''
    par = np.polyfit(reals, predicts, 1, full=True)
    slope=par[0][0]
    intercept=par[0][1]
    x1 = [0.4, 1.0]
    y1 = [slope*xx + intercept  for xx in x1]
    colors = ["#FF3636", '#277C8E' ,"#1F77B4"]
    pos = [0, .5, 1]
    cmap = LinearSegmentedColormap.from_list('my_colormap', list(zip(pos, colors)))


    durations = np.array(durations)
    normalied_durations = (durations - durations.min())/(durations.max() - durations.min())

    random_index = list(range(len(reals)))
    random.shuffle(random_index)
    random_index = random_index[:1500]
    reals = np.array(reals)
    predicts = np.array(predicts)
    fig, axes = plt.subplots(figsize=(5, 5))  # 创建一个图形对象和一个子图对象
    axes.axis([0, 1, 0, 1])
    axes.scatter(reals[random_index], predicts[random_index], c= normalied_durations[random_index], cmap=cmap,alpha = 0.6, s=80 )
    axes.plot(x1,y1)
    axes.set_xlim(.5, 1)
    plt.title(title)
    axes.set_ylim(.5, 1)
    axes.set_xlabel('real ')
    axes.set_ylabel('predict')
    axes.plot([0,1],[0,1])
    # fig.colorbar(cm.ScalarMappable( cmap=cmap))
    # fig.savefig(name)
    print(slope, intercept)
    return fig, axes