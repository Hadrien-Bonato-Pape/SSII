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

k = np.array([[0,1,0], [0,1,0], [0,1,0]])

circuitBinaryErosion = ndimage.binary_erosion(circuit)
circuit_im_as_fl = filters.edges.img_as_float(circuit)

plt.subplot(221)
plt.imshow(circuit,cmap='gray')
plt.ylabel("Image d'origine")

plt.subplot(222)
plt.imshow(circuitBinaryErosion,cmap='gray')
plt.ylabel("Image avec convolution Binary Erosion")

circuitprewitt = filters.edges.prewitt(circuit)

plt.subplot(223)
plt.imshow(circuitprewitt,cmap='gray')
plt.ylabel("Image avec convolution Prewitt")

circuitsobel = filters.edges.sobel(circuit)

plt.subplot(224)
plt.imshow(circuitsobel,cmap='gray')
plt.ylabel("Image avec convolution Sobel")

plt.show()
