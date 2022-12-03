# -*- coding: utf-8 -*-
"""
  Download a npy file to wget
  !wget https://bit.ly/fruits_300_data -O fruits_300.npy
"""

"""
  Loading saved a npy file
"""
import numpy as np

fruits = np.load("fruits_300.npy")
fruits_2d = fruits.reshape(-1, 100*100)
#checkout
print(fruits_2d.shape)

"""
  Create a KMeans class
"""
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_2d)

"""
  Check clustered results
"""
#n_cluster=3 ? labels_= [0, 1, 2], n_cluster=4 ? labels_= [0, 1, 2, 3]
print(km.labels_)

"""
  Check a Samples to fruits_2d
"""
print(np.unique(km.labels_, return_counts=True))

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

"""
  Boolean indexing and call the function
"""
# labels_=0 ? True, 1 and 2 False
draw_fruits(fruits[km.labels_==0])

# labels_=1 ? True, 0 and 2 False
draw_fruits(fruits[km.labels_==1])

#labels_=2 ? True, 0 and 1 False
draw_fruits(fruits[km.labels_==2])

"""
  Centroid details
"""
#return centroid
print(km.cluster_centers_)

#return transform to fruits_2d[100:101]
print(km.transform(fruits_2d[100:101]))

#return predict to fruits_2d[100:101]
print(km.predict(fruits_2d[100:101]))

#return iterate
print(km.n_iter_)

"""
  Output to centroid images
"""
draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio=4)

"""
  Using Elbow Method
"""
inertia = []
for k in range(2, 7):
  km = KMeans(n_clusters=k, random_state=42)
  km.fit(fruits_2d)
  inertia.append(km.inertia_)
#checkout
plt.figure(figsize=(20,5))
plt.plot(range(2, 7), inertia)
plt.xlabel("k")
plt.ylabel("inertia")
plt.title("elbow solution")
plt.show()