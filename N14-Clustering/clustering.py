# -*- coding: utf-8 -*-
"""
  Download a npy file to wget
  !wget https://bit.ly/fruits_300_data -O fruits_300.npy
"""

"""
  Import packages
  Loading saved a npy file
"""
import numpy as np
import matplotlib.pyplot as plt
fruits = np.load("fruits_300.npy")
#checkout
print(fruits.shape)

"""
  Output the first row of the first image
"""
#1
print(fruits[0, 0])
#2
print(fruits[0, 0, :])

"""
  Output the second row of the second image
"""
#1
print(fruits[1, 1])
#2
print(fruits[1, 1, :])

"""
  Visualize the first index image
"""
plt.imshow(fruits[0], cmap="gray")
plt.show()
#reversal
plt.imshow(fruits[0], cmap="gray_r")
plt.show()
plt.imshow(fruits[100], cmap="gray")
plt.show()
#reversal
plt.imshow(fruits[100], cmap="gray_r")
plt.show()
plt.imshow(fruits[200], cmap="gray")
plt.show()
#reversal
plt.imshow(fruits[200], cmap="gray_r")
plt.show()

"""
  Visualize multiple images
"""
fig, axs = plt.subplots(1, 3)
axs[0].imshow(fruits[0], cmap="gray_r")
axs[1].imshow(fruits[100], cmap="gray_r")
axs[2].imshow(fruits[200], cmap="gray_r")
plt.show()

"""
  Analyze the size of an array
"""
apple = fruits[0:100].reshape(-1, 100*100)
pineapple = fruits[100:200].reshape(-1, 100*100)
banana = fruits[200:300].reshape(-1, 100*100)

print(apple.shape)
print(pineapple.shape)
print(banana.shape)

#means of apple
print(apple.mean(axis=1))

#means of pineapple
print(pineapple.mean(axis=1))

#means of banana
print(banana.mean(axis=1))

"""
  Draw a histogram of the overall samples means
"""
plt.hist(np.mean(apple, axis=1), alpha=0.8)
plt.hist(np.mean(pineapple, axis=1), alpha=0.8)
plt.hist(np.mean(banana, axis=1), alpha=0.8)
plt.legend(["apple", "pineapple", "banana"])
plt.show()

plt.hist(np.mean(apple, axis=1), alpha=0.5)
plt.hist(np.mean(pineapple, axis=1), alpha=0.5)
plt.hist(np.mean(banana, axis=1), alpha=0.5)
plt.legend(["apple", "pineapple", "banana"])
plt.show()

"""
  Draw a bar-graph of the overall pixel means
"""
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].bar(range(10000), np.mean(apple, axis=0))
axs[1].bar(range(10000), np.mean(pineapple, axis=0))
axs[2].bar(range(10000), np.mean(banana, axis=0))

"""
  To output an average image
"""
apple_mean = np.mean(apple, axis=0).reshape(100, 100)
pineapple_mean = np.mean(pineapple, axis=0).reshape(100, 100)
banana_mean = np.mean(banana, axis=0).reshape(100, 100)

fig, axs = plt.subplots(1, 3, figsize=(20, 5))

axs[0].imshow(apple_mean, cmap="gray_r")
axs[1].imshow(pineapple_mean, cmap="gray_r")
axs[2].imshow(banana_mean, cmap="gray_r")
plt.show()

"""
  To output an image close to the means
"""
abs_diff = np.abs(fruits - apple_mean)
abs_mean = np.mean(abs_diff, axis=(1, 2))
print(abs_mean.shape)

apple_index = np.argsort(abs_mean)[:100]
fig, axs = plt.subplots(10, 10, figsize=(10, 10))
for i in range(10):
  for j in range(10):
    axs[i, j].imshow(fruits[apple_index[i*10 + j]], cmap="gray_r")
    axs[i, j].axis("off") #"on"= 좌표 축 생성
plt.show()

#pineapple
abs_diff = np.abs(fruits - pineapple_mean)
abs_mean = np.mean(abs_diff, axis=(1, 2))
print(abs_mean.shape)

pineapple_index = np.argsort(abs_mean)[:100]
fig, axs = plt.subplots(10, 10, figsize=(10, 10))
for i in range(10):
  for j in range(10):
    axs[i, j].imshow(fruits[pineapple_index[i*10 + j]], cmap="gray_r")
    axs[i, j].axis("on")
plt.show()

#banana
abs_diff = np.abs(fruits - banana_mean)
abs_mean = np.mean(abs_diff, axis=(1, 2))
print(abs_mean.shape)

banana_index = np.argsort(abs_mean)[:100]
fig, axs = plt.subplots(10, 10, figsize=(10, 10))
for i in range(10):
  for j in range(10):
    axs[i, j].imshow(fruits[banana_index[i*10 + j]], cmap="gray_r")
    axs[i, j].axis("off") #"on"= 좌표 축 생성
plt.show()