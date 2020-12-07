#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
K-means Clustering
=========================================================

The plots display firstly what a K-means algorithm would yield
using three clusters. It is then shown what the effect of a bad
initialization is on the classification process:
By setting n_init to only 1 (default is 10), the amount of
times that the algorithm will be run with different centroid
seeds is reduced.
The next plot displays what using eight clusters would deliver
and finally the ground truth.

"""
print(__doc__)


# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
import pandas
# Though the following import is not directly being used, it is required
# for 3D projection to work
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn import datasets

from sklearn.datasets import load_svmlight_file

np.random.seed(5)


df = pandas.read_csv("CFB2019.csv")

names = ["First Downs", "Pass Touchdowns", "Penalties"]

X = df[[names[0]]].to_numpy()
X = X.reshape(X.shape[0],1)

y = df[[names[1]]].to_numpy()
y = y.reshape(X.shape[0],1)

z = df[[names[2]]].to_numpy()
z = z.reshape(X.shape[0], 1)

#iris = datasets.load_iris()
#X = iris.data
#y = iris.target

print("X values:")
print(X)
print("Y values:")
print(y)
print("Z values:")
print(z)

estimators = [('k_means_iris_8', KMeans(n_clusters=8)),
              ('k_means_iris_3', KMeans(n_clusters=3)),
              ('k_means_iris_bad_init', KMeans(n_clusters=3, n_init=1,
                                               init='random'))]

fignum = 1
titles = ['8 clusters', '3 clusters', '3 clusters, bad initialization']
for name, est in estimators:

    fig = plt.figure(fignum, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    est.fit(X)
    labels = est.labels_

    ax.scatter(X, y, z,
               c=labels.astype(np.float), edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.set_zlabel(names[2])
    ax.set_title(titles[fignum - 1])
    ax.dist = 12
    fig.savefig(titles[fignum - 1]+".png")
    fignum = fignum + 1


# Plot the ground truth
fig = plt.figure(fignum, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

#depending on our variables, this section will label each 'cluster'
#for name, label in [(names[0], 0),
#                    (names[1], 1),
#                    (names[2], 2)]:
#    ax.text3D(X.mean(),
#              y.mean(),
#              z.mean() + 2, name,
#              horizontalalignment='center',
#              bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))


# Reorder the labels to have colors matching the cluster results
#y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X, y, z, c=y, edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel(names[0])
ax.set_ylabel(names[1])
ax.set_zlabel(names[2])
ax.set_title('Ground Truth')
ax.dist = 12

fig.savefig('Ground Truth.png')
#fig.show()
