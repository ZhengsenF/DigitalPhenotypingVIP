from PIL import Image
import PIL
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

img = Image.open('./mao.jpg')
img = np.array(img)
#plt.imshow(img)
#plt.show()

#flip
flipped_img = np.fliplr(img)
plt.imshow(flipped_img)
#plt.show()
np.save("flip.jpg",img)