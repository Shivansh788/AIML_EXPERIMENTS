#importing the image module from matplotlib library
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

#Pillow library
from PIL import Image
#importing open CV library
import cv2 

#loading an image through matplotlib.image module

img = mpimg.imread('puppy.jpg')

#type of image is ndarray
print(type(img))

print(img.shape)

img_plot = plt.imshow(img)

#resizing image
img = Image.open('puppy.jpg')
img_resized = img.resize((200,200))

#saving resized image
# img_resized.save('puppy_resized.jpg')

img_res = mpimg.imread('puppy_resized.jpg')
img_res_plot = plt.imshow(img_res)
plt.show()

#converting RGB Image to gray scale
img = cv2.imread('puppy.jpg')

#type of img is ndarray
# type(img)

grarscale_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#cv.imshow() will display the image. But this will not be allowed in Google Colab

 