import numpy as np
import scipy.ndimage
import cv2

def filter_over_Y(image, kernel):
  filtered_image = np.zeros_like(image).astype(np.float32)
  shape_of_image = np.shape(image)
  padded_image = np.zeros(shape=(shape_of_image[0] + kernel.size, shape_of_image[1]))
  padded_image[kernel.size//2:shape_of_image[0] + kernel.size//2, :] = image
  Y_dim = image.shape[0]
  for y in range(Y_dim):
    tmp_slice = np.zeros_like(image[y, :]).astype(np.float32)
    for i in range(kernel.size):
      tmp_slice += padded_image[y + i, :] * kernel[i]
    filtered_image[y, :] = tmp_slice
    print('.', end='', flush=True)
  print()
  return filtered_image

def filter_over_X(image, kernel):
  filtered_image = np.zeros_like(image).astype(np.float32)
  shape_of_image = np.shape(image)
  padded_image = np.zeros(shape=(shape_of_image[0], shape_of_image[1] + kernel.size))
  padded_image[:, kernel.size//2:shape_of_image[1] + kernel.size//2] = image
  X_dim = image.shape[1]
  for x in range(X_dim):
    tmp_slice = np.zeros_like(image[:, x]).astype(np.float32)
    for i in range(kernel.size):
      tmp_slice += padded_image[:, x + i] * kernel[i]
    filtered_image[:, x] = tmp_slice
    print('.', end='', flush=True)
  print()
  return filtered_image

def filter(image, kernel):
  print(f"image.shape={image.shape} kernel.shape={kernel.shape}")
  filtered_image_Y = filter_over_Y(image, kernel)
  filtered_image_YX = filter_over_X(filtered_image_Y, kernel)
  return filtered_image_YX
