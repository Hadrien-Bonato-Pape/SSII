import os
import skimage
from skimage import io
from skimage import filters
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

filename = os.path.join('_DSC2966.jpg')
circuitOrig = io.imread(filename)

circuit = circuitOrig[:,:,0]

imageBruitGauss = skimage.util.random_noise(circuit, mode='gaussian')
imageBruitSP = skimage.util.random_noise(circuit, mode='s&p')

gaussGauss = filters.gaussian(imageBruitGauss)
gaussSP = filters.gaussian(imageBruitSP)

medGauss = filters.median(imageBruitGauss)
medSP = filters.median(imageBruitSP)

plt.subplot(331)
plt.imshow(circuit,cmap='gray')
plt.xlabel("Image d'origine")

plt.subplot(332)
plt.imshow(imageBruitGauss,cmap='gray')
plt.xlabel("Bruit Gauss")

plt.subplot(333)
plt.imshow(imageBruitSP,cmap='gray')
plt.xlabel("Bruit SP")

plt.subplot(334)
plt.imshow(gaussGauss,cmap='gray')
plt.xlabel("Filtre Gauss sur bruit Gauss")

plt.subplot(335)
plt.imshow(gaussSP,cmap='gray')
plt.xlabel("Filtre Gauss sur bruit S&P")

plt.subplot(336)
plt.imshow(medGauss,cmap='gray')
plt.xlabel("Filtre Median sur bruit Gauss")

plt.subplot(337)
plt.imshow(medSP,cmap='gray')
plt.xlabel("Filtre Median sur bruit S&P")

plt.show()
