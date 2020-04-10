import os
import skimage
from skimage import io
import matplotlib.pyplot as plt
import numpy

filename = os.path.join('_DSC2966.jpg')
circuit = io.imread(filename)

plt.subplot(221)
plt.imshow(circuit)
plt.ylabel("Couleurs RGB")

rouge = circuit[:,:,0]
vert = circuit[:,:,1]
bleu = circuit[:,:,2]

plt.subplot(222)
plt.imshow(rouge,cmap='gray')
plt.colorbar()
plt.ylabel("Rouge")

plt.subplot(223)
plt.imshow(vert,cmap='gray')
plt.colorbar()
plt.ylabel("Vert")

plt.subplot(224)
plt.imshow(bleu,cmap='gray')
plt.colorbar()
plt.ylabel("Bleu")

plt.show()


from PIL import Image

#taille image
largeur = len(circuit[0,:,0])
hauteur = len(circuit[:,0,0])

# inversion couleur


img=Image.open("_DSC2966.jpg")
for x in range (largeur):
    for y in range (hauteur):
        r,v,b = img.getpixel((x,y))
        nr,nv,nb=255-r,255-v,255-b
        img.putpixel((x,y),(nr,nv,nb))
img.save("circuit2.jpeg")

filename = os.path.join('circuit2.jpeg')
circuit = io.imread(filename)

plt.imshow(circuit)
plt.ylabel("Couleurs inversées")
plt.show()


