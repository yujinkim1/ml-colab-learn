# -*- coding: utf-8 -*-
"""
  Download a npy file to wget
  !wget https://bit.ly/fruits_300_data -O fruits_300.npy
  Loading saved a npy file
"""
import numpy as np

fruits = np.load("fruits_300.npy")
fruits_2d = fruits.reshape(-1, 100*100)
#checkout
print(fruits_2d.shape)

"""
  Using Principal Component Analysis class
"""
from sklearn.decomposition import PCA

pca = PCA(n_components=50)
pca.fit(fruits_2d)

"""
  Check the principal data
"""
#Checkout array size
print(pca.components_.shape)
print(pca.components_)

"""
  Draw a subplots to use `matplotlib.pyplot`
"""
import matplotlib.pyplot as plt

def draw_fruits(arr, ratio=1):
  n = len(arr)
  rows = int(np.ceil(n/10))
  colums = 10
  if rows < 2: colums = n
  fig, axs = plt.subplots(rows, colums, figsize=(colums*ratio, rows*ratio), squeeze=False)

  for i in range(rows):
    for j in range(colums):
      if i*10+j < n:
        axs[i, j].imshow(arr[i*10+j], cmap="gray_r")
      axs[i, j].axis("off")
  plt.show()

draw_fruits(pca.components_.reshape(-1, 100, 100))

"""
  Dimension reduction
"""
print(fruits_2d.shape)
#Checkout
fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)

"""
  Reconfigure source data
"""

print(fruits_pca.shape)
#Checkout
fruits_inverse=pca.inverse_transform(fruits_pca)
print(fruits_inverse.shape)

"""
  Reconstruct data for restored principal components
"""
fruits_reconstruct = fruits_inverse.reshape(-1, 100, 100)
for start in [0, 100, 200]:
  draw_fruits(fruits_reconstruct[start:start+100])
  print("\n")

"""
  Explained variance
"""
print(pca.explained_variance_ratio_)
print("\n")
#Checkout
print(np.sum(pca.explained_variance_ratio_))

"""
  Visualize the an explained variance graph
"""

import matplotlib.pyplot as plt

plt.plot(pca.explained_variance_ratio_)
plt.title("explained variance")
plt.show()

"""
  Use in a classification model
"""
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
target = np.array([0]*100 + [1]*100 +  [2]*100)
#create fruits_2d scores
from sklearn.model_selection import cross_validate
scores = cross_validate(lr, fruits_2d, target)
print(np.mean(scores["test_score"]))
print(np.mean(scores["fit_time"]))
#create fruits_pca scores
scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores["test_score"]))
print(np.mean(scores["fit_time"]))
#explained variance 50%
pca = PCA(n_components=0.5)
pca.fit(fruits_2d)
#Checkout
print(pca.n_components_)
fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)

"""
  Use in a cluster model
"""
from sklearn.cluster import KMeans

#Create model
km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_pca)
#Checkout
print(np.unique(km.labels_, return_counts=True))
#draw fruits
for label in range(0, 3):
  draw_fruits(fruits[km.labels_ == label])
  print("\n")

"""
  Data Visualization
"""
plt.figure(figsize=(10, 10))
for label in range(0, 3):
  data = fruits_pca[km.labels_ == label]
  plt.scatter(data[:,0], data[:,1])
plt.legend(["pineapple", "banana", "apple"])
plt.show()