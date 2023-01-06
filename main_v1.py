# Python program to illustrate  
# loading and showing an image 
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def mse(image_a, image_b):
  # the 'Mean Squared Error' between the two images is the
  # sum of the squared difference between the two images;
  # NOTE: the two images must have the same dimension
  err = np.sum((image_a.astype('float') - image_b.astype('float')) ** 2)
  err /= float(image_a.shape[0] * image_a.shape[1])

  # return the MSE, the lower the error, the more "similar"
  # the two images are
  return err

def compare_images(image_a, image_b, title):
  # compute the mean squared error and structural similarity
  # index for the images
  ssim_idx = ssim(image_a, image_b)
  mse_idx  = mse(image_a, image_b)
  
  # setup the figure
  fig = plt.figure(title)
  plt.suptitle("MSE: %.2f, SSIM: %.2f" % (mse_idx, ssim_idx))
  
  # show first image
  ax = fig.add_subplot(1, 2, 1)
  plt.imshow(image_a, cmap = plt.cm.gray)
  plt.axis("off")
  
  # show the second image
  ax = fig.add_subplot(1, 2, 2)
  plt.imshow(image_b, cmap = plt.cm.gray)
  plt.axis("off")
  
  # show the images
  plt.show()

arr = os.listdir('images')


f = open('result.txt', 'w')

for img_path in arr:
  # path to input images are specified and   
  # images are loaded with imread command
  # try img = cv2.imread('images/cats.jpg', )
  
  print('preprocessing images/%s' % img_path)
  
  img = cv2.imread('images/%s' % img_path, cv2.IMREAD_GRAYSCALE)
  
  # Convert BGR to RGB if not IMREAD_GRAYSCALE
  if len(img.shape) > 2:                  # if there are more than zero channels
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # covert BGR to RGB
  
  
  scale_percent = 2.0 # percent of original size
  width  = int(img.shape[1] * scale_percent / 100.0)
  height = int(img.shape[0] * scale_percent / 100.0)
  dim    = (width, height)
  
  # resize image
  resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

  # save image
  cv2.imwrite('preprocess/%s' % img_path, resized)

arr = os.listdir('preprocess')

print('compare images')

for i in range(len(arr)):
  img_a_path = arr[i]

  if arr[i] == 'moved':
    continue
    
  for j in range(i + 1, len(arr)):
    if arr[j] == 'moved':
      continue
      
    img_b_path = arr[j]
    
    img_a = cv2.imread('preprocess/%s' % img_a_path, cv2.IMREAD_GRAYSCALE)
    img_b = cv2.imread('preprocess/%s' % img_b_path, cv2.IMREAD_GRAYSCALE)

    if img_a.shape != img_b.shape:
      continue      

    # compare_images(img_a, img_b, '%s %s' % (img_a_path, img_b_path))

    ssim_value = ssim(img_a, img_b)

    print('%s %s %f' % (img_a_path, img_b_path, ssim_value))
    f.write('%s %s %f\n' % (img_a_path, img_b_path, ssim_value))

    if ssim_value > 0.95:
      os.rename('images/%s' % img_b_path, 'moved/%s' % img_b_path)
      os.remove('preprocess/%s' % img_b_path) 
      arr[j] = 'moved'
    

      
f.close()

'''
# the window showing output image 
plt.imshow(resized, cmap = plt.get_cmap('gray'))
plt.show()

# De-allocate any associated memory usage   
if cv2.waitKey(0) & 0xff == 27:  
    cv2.destroyAllWindows()
'''