import os
import skimage
from skimage import io
import matplotlib.pyplot as plt
import numpy
from skimage.transform import rotate
from skimage.transform import warp
from skimage.transform import SimilarityTransform

filename = os.path.join('billietterie.jpeg')
circuitOrig = io.imread(filename)

circuit = rotate(circuitOrig, 30, True)

plt.subplot(221)
plt.imshow(circuit)
plt.ylabel("Image tournée de 30 deg")

tform = SimilarityTransform(translation=(0, -10))
circuit1 = warp(circuit, tform, order=3)
circuit2 = warp(circuit, tform, order=4)
circuit3 = warp(circuit, tform, order=5)

plt.subplot(222)
plt.imshow(circuit1)
plt.ylabel("Image avec mode 3")

plt.subplot(223)
plt.imshow(circuit2)
plt.ylabel("Image avec mode 4")

plt.subplot(224)
plt.imshow(circuit3)
plt.ylabel("Image avec mode 5")

plt.show()
