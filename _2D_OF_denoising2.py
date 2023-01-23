import numpy as np
import scipy.ndimage
import cv2

ofca_extension_mode = cv2.BORDER_REPLICATE

def warp_slice(reference_slice, flow):
  height, width = flow.shape[:2]
  map_x = np.tile(np.arange(width), (height, 1))
  map_y = np.swapaxes(np.tile(np.arange(height), (width, 1)), 0, 1)
  map_xy = (flow + np.dstack((map_x, map_y))).astype('float32')
  return cv2.remap(reference_slice, map_xy, None, interpolation=cv2.INTER_LINEAR, borderMode=ofca_extension_mode)

def get_flow(reference_slice, target_slice, l, w):
  flow = cv2.calcOpticalFlowFarneback(prev=target_slice, next=reference_slice, flow=None, pyr_scale=0.5, levels=l, winsize=w, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
  print("<", np.max(flow), ">")
  return flow

def filter_over_Y(stack, kernel, l, w):
  filtered_stack = np.zeros_like(stack).astype(np.float32)
  shape_of_stack = np.shape(stack)
  padded_stack = np.zeros(shape=(shape_of_stack[0], shape_of_stack[1] + kernel.size, shape_of_stack[2]))
  padded_stack[:, kernel.size//2:shape_of_stack[1] + kernel.size//2, :] = stack
  Y_dim = stack.shape[1]
  for y in range(0,Y_dim,1):
    tmp_slice = np.zeros_like(stack[:, y, :]).astype(np.float32)
    for i in range(kernel.size):
      if i != kernel.size//2:
        flow = get_flow(padded_stack[:, y + i, :], stack[:, y, :], l, w)
        OF_compensated_slice = warp_slice(padded_stack[:, y + i, :], flow)
        tmp_slice += OF_compensated_slice * kernel[i]
      else:
        # No OF is needed for this slice
        tmp_slice += stack[:, y, :] * kernel[i]
    filtered_stack[:, y, :] = tmp_slice
    print(y, end=' ', flush=True)
  print()
  return filtered_stack

def filter_over_X(stack, kernel, l, w):
  filtered_stack = np.zeros_like(stack).astype(np.float32)
  shape_of_stack = np.shape(stack)
  padded_stack = np.zeros(shape=(shape_of_stack[0], shape_of_stack[1], shape_of_stack[2] + kernel.size))
  padded_stack[:, :, kernel.size//2:shape_of_stack[2] + kernel.size//2] = stack
  X_dim = stack.shape[2]
  for x in range(0,X_dim,1):
    tmp_slice = np.zeros_like(stack[:, :, x]).astype(np.float32)
    for i in range(kernel.size):
      if i != kernel.size//2:
        flow = get_flow(padded_stack[:, :, x + i], stack[:, :, x], l, w)
        OF_compensated_slice = warp_slice(padded_stack[:, :, x + i], flow)
        tmp_slice += OF_compensated_slice * kernel[i]
      else:
        # No OF is needed for this slice
        tmp_slice += stack[:, :, x] * kernel[i]
    filtered_stack[:, :, x] = tmp_slice
    print(x, end=' ', flush=True)
  print()
  return filtered_stack

N = 17

def filter(img, kernel, l, w):
  print(f"img.shape={img.shape} kernel.shape={kernel.shape} l={l} w={w}")
  #img_zeros = np.zeros_like(img)
  #stack = np.stack([img_zeros, img, img_zeros])
  #stack = np.stack([img, img, img]
  stack = np.stack([np.roll(img, i, axis=0) for i in range(N)])
  filtered_Y_img = filter_over_Y(stack, kernel, l, w)[N//2, ...]
  filtered_Y_img = np.roll(filtered_Y_img, -N//2, axis=0) 
  #filtered_stack_Y = stack
  #stack = np.stack([np.roll(filtered_stack_Y,i) for i in range(N)])
  
  stack = np.stack([np.roll(filtered_Y_img, i, axis=1) for i in range(N)])
  filtered_YX_img = filter_over_X(stack, kernel, l, w)[N//2, ...]
  filtered_YX_img = np.roll(filtered_YX_img, -N//2, axis=1) 
  #filtered_stack_YX = filter_over_Y(filtered_stack_Y.T, kernel, l, w)
  #filtered_stack_YX = filtered_stack_Y
  return filtered_YX_img
  #return filtered_stack_YX.T[N//2, ...]
