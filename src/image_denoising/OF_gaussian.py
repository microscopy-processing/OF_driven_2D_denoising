import time
import numpy as np
import cv2
import scipy
import math
from . import kernels
import flow

#pip install "color_transforms @ git+https://github.com/vicente-gonzalez-ruiz/color_transforms"
from color_transforms import YCoCg as YUV

if __debug__:
    from matplotlib import pyplot as plt

def __get_flow(reference, target, l=3, w=5, prev_flow=None, sigma=0.5):
    flow = cv2.calcOpticalFlowFarneback(prev=target, next=reference, flow=prev_flow,
                                            pyr_scale=0.5, levels=l, winsize=w,
                                            iterations=3, poly_n=5, poly_sigma=sigma,
                                            flags=0)
    #flow[...] = 0.0
    print(np.max(np.abs(flow)), end=' ')
    return flow

def __warp_slice(reference, flow):
    height, width = flow.shape[:2]
    map_x = np.tile(np.arange(width), (height, 1))
    map_y = np.swapaxes(np.tile(np.arange(height), (width, 1)), 0, 1)
    map_xy = (flow + np.dstack((map_x, map_y))).astype('float32')
    warped_slice = cv2.remap(reference, map_xy, None,
                             #interpolation=cv2.INTER_LANCZOS4, #INTER_LINEAR,
                             interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    return warped_slice

def gray_vertical_OF_gaussian_filtering(img, kernel, l=3, w=5, sigma=0.5):
    print("v3")
    KL = kernel.size
    KL2 = KL//2
    w2 = w//2
    N_rows = img.shape[0]
    N_cols = img.shape[1]

    # Opción 0: Los márgenes son 128
    #extended_img = np.full(shape=(img.shape[0] + KL + w, img.shape[1] + w, img.shape[2]), fill_value=128, dtype=np.uint8)

    # Opción 1: Usando padding (no terminó de funcionar)
    #extended_img = np.empty(shape=(img.shape[0] + KL + w, img.shape[1] + w, img.shape[2]), dtype=np.uint8)
    #extended_img[..., 0] = np.pad(array=img[..., 0],
    #                              pad_width=(((KL + w)//2, (KL + w)//2), ((w + 1)//2, (w + 1)//2)),
    #                              mode="constant")
    #extended_img[..., 1] = np.pad(array=img[..., 1], pad_width=(KL2 + w2, w2), mode="constant")
    #extended_img[..., 2] = np.pad(array=img[..., 2], pad_width=(KL2 + w2, w2), mode="constant")

    # Opción 2: Los márgenes son la propia imagen, ampliada
    extended_img = cv2.resize(src = img, dsize = (img.shape[1] + w, img.shape[0] + KL + w))
    #print(extended_img.shape)
    #extended_img[KL2 + w2:img.shape[0] + KL2 + w2, w2:img.shape[1] + w2] = img[...]
    #extended_img[(KL + w)//2 - 1:img.shape[0] + (KL + w)//2 - 1, w2 - 1:img.shape[1] + w2 - 1] = img[...]
    extended_img[(KL + w)//2:img.shape[0] + (KL + w)//2, w2:img.shape[1] + w2] = img[...]
    extended_img = extended_img.astype(np.float32)
    extended_Y = extended_img
    filtered_img = []
    N_rows = img.shape[0]
    N_cols = img.shape[1]
    for y in range(N_rows):
        print(y, end=' ')
        horizontal_line = np.zeros(shape=(N_cols + w), dtype=np.float32)
        target_slice_Y = extended_Y[y + KL2:y + KL2 + w]
        #print("<", target_slice_Y.shape, ">")
        target_slice = extended_img[y + KL2:y + KL2 + w]
        for i in range(KL):
            reference_slice_Y = extended_Y[y + i:y + i + w]
            reference_slice = extended_img[y + i:y + i + w]
            flow = flow.get_flow_to_project_A_to_B(
                A=reference_slice_Y,
                B=target_slice_Y,
                l,
                w,
                prev_flow=None,
                sigma)
            OF_compensated_slice = flow.project(reference_slice, flow)
            OF_compensated_line = OF_compensated_slice[(w + 1) >> 1, :]
            OF_compensated_line = np.roll(OF_compensated_line, -w2)
            horizontal_line += OF_compensated_line * kernel[i]
        filtered_img.append(horizontal_line)
    filtered_img = np.stack(filtered_img, axis=0)[0:img.shape[0], 0:img.shape[1]]
    return filtered_img

def gray_OF_gaussian_filtering(img, kernel, l=3, w=5, sigma=0.5):
    filtered_img_in_vertical = gray_vertical_OF_gaussian_filtering(img, kernel, l, w, sigma)
    transposed_img = np.transpose(img, (1, 0))
    transposed_and_filtered_img_in_horizontal = gray_vertical_OF_gaussian_filtering(transposed_img, kernel, l, w, sigma)
    filtered_img_in_horizontal = np.transpose(transposed_and_filtered_img_in_horizontal, (1, 0))
    filtered_img = (filtered_img_in_vertical + filtered_img_in_horizontal)/2
    return filtered_img

def color_vertical_OF_gaussian_filtering(img, kernel, l=3, w=5, sigma=0.5):
    #print("v1")
    KL = kernel.size
    KL2 = KL//2
    w2 = w//2
    N_rows = img.shape[0]
    N_cols = img.shape[1]
    #print(f"KL={KL} l={l} w={w}")

    # Opción 0: Los márgenes son 128
    #extended_img = np.full(shape=(img.shape[0] + KL + w, img.shape[1] + w, img.shape[2]), fill_value=128, dtype=np.uint8)

    # Opción 1: Usando padding (no terminó de funcionar)
    #extended_img = np.empty(shape=(img.shape[0] + KL + w, img.shape[1] + w, img.shape[2]), dtype=np.uint8)
    #extended_img[..., 0] = np.pad(array=img[..., 0],
    #                              pad_width=(((KL + w)//2, (KL + w)//2), ((w + 1)//2, (w + 1)//2)),
    #                              mode="constant")
    #extended_img[..., 1] = np.pad(array=img[..., 1], pad_width=(KL2 + w2, w2), mode="constant")
    #extended_img[..., 2] = np.pad(array=img[..., 2], pad_width=(KL2 + w2, w2), mode="constant")

    # Opción 2: Los márgenes son la propia imagen, ampliada
    extended_img = cv2.resize(src = img, dsize = (img.shape[1] + w, img.shape[0] + KL + w))
    #print(extended_img.shape)
    extended_img[(KL + w)//2:img.shape[0] + (KL + w)//2, w2:img.shape[1] + w2] = img[...]
    extended_Y = YUV.from_RGB(extended_img.astype(np.int16))[..., 0]
    extended_Y = extended_Y.astype(np.float32)
    extended_img = extended_img.astype(np.float32)
    print(np.max(extended_Y), np.min(extended_Y))
    filtered_img = []
    N_rows = img.shape[0]
    N_cols = img.shape[1]
    for y in range(N_rows):
        print(y, end=' ')
        horizontal_line = np.zeros(shape=(N_cols + w, img.shape[2]), dtype=np.float32)
        target_slice_Y = extended_Y[y + KL2:y + KL2 + w]
        #print("<", target_slice_Y.shape, w, ">")
        target_slice = extended_img[y + KL2:y + KL2 + w, :]
        for i in range(KL):
            reference_slice_Y = extended_Y[y + i:y + i + w, :]
            reference_slice = extended_img[y + i:y + i + w, :]
            flow = get_flow_to_project_A_to_B(
                A=reference_slice_Y,
                B=target_slice_Y,
                l,
                w,
                prev_flow=None,
                sigma)
            OF_compensated_slice = flow.project(reference_slice, flow)
            OF_compensated_line = OF_compensated_slice[(w + 1) >> 1, :, :]
            #OF_compensated_line = OF_compensated_slice[(w + 0) >> 1, :, :]
            OF_compensated_line = np.roll(a=OF_compensated_line, shift=-w2, axis=0)
            horizontal_line += OF_compensated_line * kernel[i]
        filtered_img.append(horizontal_line)
    filtered_img = np.stack(filtered_img, axis=0)[0:img.shape[0], 0:img.shape[1], :]
    return filtered_img

def color_gaussian_filtering(img, kernel, l=3, w=5, sigma=0.5):
    filtered_img_Y = color_vertical_gaussian_filtering(img, kernel, l, w, sigma)
    filtered_img_YX = color_vertical_gaussian_filtering(np.transpose(filtered_img_Y, (1, 0, 2)), kernel, l, w, sigma)
    OF_filtered_img = np.transpose(filtered_img_YX, (1, 0, 2))
    return OF_filtered_img

def filter_gray_image(img, sigma_kernel=2.5, N_iters=1, l=3, w=9, sigma_OF=2.5):
    kernel = kernels.get_gaussian_kernel(sigma_kernel)
    denoised = noisy.copy()
    for i in range(N_iters):
        denoised = gray_gaussian_filtering(denoised, kernel, l, w, sigma_OF)
        if __debug__:
            fig, axs = plt.subplots(1, 2, figsize=(10, 20))
            axs[0].imshow(normalize(denoised), cmap="gray")
            axs[0].set_title(f"iter {i}")
            axs[1].imshow(normalize(denoised - prev + 128), cmap="gray")
            axs[1].set_title(f"diff")
            plt.show()
            print(f"\niter={i}")
    return denoised
