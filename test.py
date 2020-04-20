from PIL import Image
import PIL
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import io
from skimage.transform import rotate,AffineTransform,warp

img = Image.open('./mao.jpg')
img = np.array(img)
#plt.imshow(img)
#plt.show()

#greyscale
gray= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.imshow(gray)
#plt.show()
imggrey = Image.fromarray(gray)
imggrey.save("grayscale.jpg")

#flip horizontally
flipped_img = np.fliplr(img)
plt.imshow(flipped_img)
imgflip = Image.fromarray(flipped_img)
imgflip.save("flipped_img.jpg")

#flip vertically
vertically_flipped_image = np.flipud(img)
plt.imshow(vertically_flipped_image)
imgflip_vert = Image.fromarray(vertically_flipped_image)
imgflip_vert.save("flipped_img_vert.jpg")
