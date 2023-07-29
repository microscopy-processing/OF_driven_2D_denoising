import time
import numpy as np
import cv2
import scipy
import math
import kernels
#pip install "color_transforms @ git+https://github.com/vicente-gonzalez-ruiz/color_transforms"
from color_transforms import YCoCg as YUV

if __debug__:
    from matplotlib import pyplot as plt

def vertical_gaussian_filtering(img, kernel, mean):
    KL = kernel.size
    KL2 = KL//2
    extended_img = np.full(fill_value=mean, shape=(img.shape[0] + KL, img.shape[1]))
    extended_img[KL2:img.shape[0] + KL2, :] = img[:, :]
    filtered_img = []
    #filtered_img = np.empty_like(img, dtype=np.float32)
    N_rows = img.shape[0]
    N_cols = img.shape[1]
    #horizontal_line = np.empty(N_cols, dtype=np.float32)
    #print(horizontal_line.shape)
    for y in range(N_rows):
        #horizontal_line.fill(0)
        horizontal_line = np.zeros(N_cols, dtype=np.float32)
        for i in range(KL):
            horizontal_line += extended_img[y + i, :] * kernel[i]
        filtered_img.append(horizontal_line)
        #filtered_img[y, :] = horizontal_line[:]
    filtered_img = np.stack(filtered_img, axis=0)
    return filtered_img

def horizontal_gaussian_filtering(img, kernel, mean):
    KL = kernel.size
    KL2 = KL//2
    extended_img = np.full(fill_value=mean, shape=(img.shape[0], img.shape[1] + KL))
    extended_img[:, KL2:img.shape[1] + KL2] = img[:, :]
    #filtered_img = []
    filtered_img = np.empty_like(img, dtype=np.float32)
    N_rows = img.shape[0]
    N_cols = img.shape[1]
    vertical_line = np.empty(N_rows, dtype=np.float32)
    for x in range(N_cols):
        #vertical_line = np.zeros(N_rows, dtype=np.float32)
        vertical_line.fill(0)
        for i in range(KL):
            vertical_line += extended_img[:, x + i] * kernel[i]
        #filtered_img.append(vertical_line)
        filtered_img[:, x] = vertical_line[:]
    #filtered_img = np.stack(filtered_img, axis=1)
    return filtered_img

def gray_gaussian_filtering(img, kernel):
    mean = np.average(img)
    #t0 = time.perf_counter()
    filtered_img_Y = vertical_gaussian_filtering(img, kernel, mean)
    #t1 = time.perf_counter()
    #print(t1 - t0)
    filtered_img_YX = horizontal_gaussian_filtering(filtered_img_Y, kernel, mean)
    #t2 = time.perf_counter()
    #print(t2 - t1)
    return filtered_img_YX

def color_gaussian_filtering(img, kernel):
    filtered_img_R = gray_gaussian_filtering(img[..., 0], kernel)
    filtered_img_G = gray_gaussian_filtering(img[..., 1], kernel)
    filtered_img_B = gray_gaussian_filtering(img[..., 2], kernel)
    return np.stack([filtered_img_R, filtered_img_G, filtered_img_B], axis=2)

def filter_gray_image(img, sigma=2.5, N_iters=1.0):
    kernel = kernels.get_gaussian_kernel(sigma)
    denoised = noisy.copy()
    for i in range(N_iters):
        denoised = gray_gaussian_filtering(denoised, kernel)
        if __debug__:
            fig, axs = plt.subplots(1, 2, figsize=(10, 20))
            axs[0].imshow(normalize(denoised), cmap="gray")
            axs[0].set_title(f"iter {i}")
            axs[1].imshow(normalize(denoised - prev + 128), cmap="gray")
            axs[1].set_title(f"diff")
            plt.show()
            print(f"\niter={i}")
    return denoised

def filter_color_image(img, sigma=2.5, N_iters=1.0):
    kernel = kernels.get_gaussian_kernel(sigma)
    denoised = noisy.copy()
    for i in range(N_iters):
        denoised = color_gaussian_filtering(denoised, kernel)
        if __debug__:
            fig, axs = plt.subplots(1, 2, figsize=(10, 20))
            axs[0].imshow(normalize(denoised), cmap="gray")
            axs[0].set_title(f"iter {i}")
            axs[1].imshow(normalize(denoised - prev + 128), cmap="gray")
            axs[1].set_title(f"diff")
            plt.show()
            print(f"\niter={i}")
    return denoised
