import numpy as np
import scipy.ndimage
import cv2

N = 3

ofca_extension_mode = cv2.BORDER_REPLICATE

def warp_line(reference_line, flow):
  reference = np.stack([reference_line, reference_line, reference_line])
  height, width = flow.shape[:2]
  map_x = np.tile(np.arange(width), (height, 1))
  map_y = np.swapaxes(np.tile(np.arange(height), (width, 1)), 0, 1)
  map_xy = (flow + np.dstack((map_x, map_y))).astype('float32')
  return cv2.remap(reference, map_xy, None, interpolation=cv2.INTER_LINEAR, borderMode=ofca_extension_mode)[N//2, ...]

def get_flow(reference_line, target_line, l, w):
  # Farneback requires at least two lines
  reference = np.stack([reference_line for i in range(N)])
  #reference = np.stack([reference_line, reference_line, reference_line])
  print("get_flow: reference =", reference[1,...][0:10])
  #target = np.stack([target_line, target_line, target_line])
  target = np.stack([target_line for i in range(N)])
  print("get_flow: target    =", target[1,...][0:10])
  flow = cv2.calcOpticalFlowFarneback(prev=target, next=reference, flow=None, pyr_scale=0.5, levels=l, winsize=w, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
  # flow returns a field in which N lines are the same
  #flow = np.zeros((reference.shape[0], reference.shape[1], 2), dtype=np.float32)
  #print("get_flow: flow =", flow[1,...][0:10])
  print("get_flow: max =", np.max(flow))
  print("get_flow: min =", np.min(flow))
  return flow

def filter_over_Y(image, kernel, l, w):
  filtered_image = np.zeros_like(image).astype(np.float32)
  shape_of_image = np.shape(image)
  padded_image = np.zeros(shape=(shape_of_image[0] + kernel.size, shape_of_image[1]))
  padded_image[kernel.size//2:shape_of_image[0] + kernel.size//2, :] = image
  Y_dim = image.shape[0]
  for y in range(Y_dim):
    tmp_line = np.zeros_like(image[y, :]).astype(np.float32)
    for i in range(kernel.size):
      if i != kernel.size//2:
        #reference = np.stack([padded_image[y + i, :] for i in range(N)])
        #target = np.stack([image[y, :] for i in range(N)])
        flow = get_flow(padded_image[y + i, :], image[y, :], l, w)
        #flow = get_flow(reference, target, l, w)
        OF_compensated_line = warp_line(padded_image[y + i, :], flow)
        #OF_compensated_line = warp_slice(reference, flow)[N//2, ...]
        #tmp_slice += OF_compensated_slice * kernel[i]
        tmp_line += OF_compensated_line * kernel[i]
      else:
        # No OF is needed for this slice
        #tmp_slice += image[:, y - kernel.size//2, :] * kernel[kernel.size // 2]
        tmp_line += image[y, :] * kernel[i]
    filtered_image[y, :] = tmp_line
    print(y, end=' ', flush=True)
  print()
  return filtered_image

def filter_over_X(image, kernel, l, w):
  filtered_image = np.zeros_like(image).astype(np.float32)
  shape_of_image = np.shape(image)
  padded_image = np.zeros(shape=(shape_of_image[0], shape_of_image[1] + kernel.size))
  padded_image[:, kernel.size//2:shape_of_image[1] + kernel.size//2] = image
  X_dim = image.shape[1]
  for x in range(X_dim):
    tmp_line = np.zeros_like(image[:, x]).astype(np.float32)
    for i in range(kernel.size):
      if i != kernel.size//2:
        flow = get_flow(padded_image[:, x + i], image[:, x], l, w)
        OF_compensated_line = warp_line(padded_image[:, x + i], flow)
        tmp_slice += OF_compensated_line * kernel[i]
      else:
        # No OF is needed for this slice
        #tmp_slice += image[:, :, x - kernel.size//2] * kernel[kernel.size // 2]
        tmp_line += image[:, x] * kernel[i]
    filtered_image[:, x] = tmp_line
    print(x, end=' ', flush=True)
  print()
  return filtered_image

def filter(image, kernel, l, w):
  print(f"image.shape={image.shape} kernel.shape={kernel.shape} l={l} w={w}")
  filtered_image_Y = filter_over_Y(image, kernel, l, w)
  filtered_image_YX = filter_over_X(filtered_image_Y, kernel, l, w)
  return filtered_image_YX
