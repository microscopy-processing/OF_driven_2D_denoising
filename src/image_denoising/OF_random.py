import numpy as np
import cv2
from . import flow_estimation
#pip install "color_transforms @ git+https://github.com/vicente-gonzalez-ruiz/color_transforms"
from color_transforms import YCoCg as YUV
#pip install "information_theory @ git+https://github.com/vicente-gonzalez-ruiz/information_theory"
from information_theory.distortion import PSNR
import information_theory
#import image_denoising

import logging
logger = logging.getLogger(__name__)
#logging.basicConfig(format="[%(filename)s:%(lineno)s %(funcName)s()] %(message)s")
#logger.setLevel(logging.CRITICAL)
#logger.setLevel(logging.ERROR)
logger.setLevel(logging.WARNING)
#logger.setLevel(logging.INFO)

if logger.getEffectiveLevel() < logging.WARNING:
    from matplotlib import pyplot as plt

    def normalize(img):
        min_img = np.min(img)
        max_img = np.max(img)
        return 255*((img - min_img)/(max_img - min_img))

def randomize(image, mean=0, std_dev=1.0):
    height, width = image.shape[:2]
    x_coords, y_coords = np.meshgrid(range(width), range(height)) # Create a grid of coordinates
    flattened_x_coords = x_coords.flatten()
    flattened_y_coords = y_coords.flatten()
    displacements_x = np.random.normal(mean, std_dev, flattened_x_coords.shape)
    displacements_y = np.random.normal(mean, std_dev, flattened_y_coords.shape)
    #displacements_x *= max_distance_x # Scale the displacements by the maximum distance
    #displacements_y *= max_distance_y
    displacements_x = displacements_x.astype(np.int32)
    displacements_y = displacements_y.astype(np.int32)
    
    logger.debug(np.max(displacements_x), np.max(displacements_y))
    randomized_x_coords = flattened_x_coords + displacements_x
    randomized_y_coords = flattened_y_coords + displacements_y
    randomized_x_coords = np.clip(randomized_x_coords, 0, width - 1) # Clip the randomized coordinates to stay within image bounds
    randomized_y_coords = np.clip(randomized_y_coords, 0, height - 1)
    #randomized_x_coords = np.mod(randomized_x_coords, width) # Apply periodic extension to handle border pixels
    #randomized_y_coords = np.mod(randomized_y_coords, height)
    randomized_image = np.zeros_like(image)
    randomized_image[randomized_y_coords, randomized_x_coords] = image[flattened_y_coords, flattened_x_coords]
    return randomized_image

def RGB_warp_B_to_A(A, B, l=3, w=15, prev_flow=None, sigma=1.5):
    A_luma = YUV.from_RGB(A.astype(np.int16))[..., 0]
    B_luma = YUV.from_RGB(B.astype(np.int16))[..., 0]
    #A_luma = np.log(YUV.from_RGB(A.astype(np.int16))[..., 0] + 1)
    #B_luma = np.log(YUV.from_RGB(B.astype(np.int16))[..., 0] + 1)
    flow = flow_estimation.get_flow_to_project_A_to_B(A_luma, B_luma, l, w, prev_flow, sigma)
    return flow_estimation.project(B, flow)

def warp_B_to_A(A, B, l=3, w=15, prev_flow=None, sigma=1.5):
    flow = flow_estimation.get_flow_to_project_A_to_B(A, B, l, w, prev_flow, sigma)
    return flow_estimation.project(B, flow)

def filter_image(
        warp_B_to_A,
        noisy_image,
        N_iters=50,
        mean_RD=0.0,
        sigma_RD=1.0,
        l=3,
        w=2,
        sigma_OF=0.3,
        GT=None):

    logger.info(f"N_iters={N_iters} mean_RD={mean_RD} sigma_RD={sigma_RD} l={l} w={w} sigma_OF={sigma_OF}")
    if logger.getEffectiveLevel() < logging.WARNING:
        PSNR_vs_iteration = []

    acc_image = np.zeros_like(noisy_image, dtype=np.float32)
    acc_image[...] = noisy_image
    denoised_image = noisy_image
    for i in range(N_iters):
        print(f"{i}/{N_iters}", end=' ')
        if logger.getEffectiveLevel() < logging.WARNING:
            fig, axs = plt.subplots(1, 2)
            prev = denoised_image
        denoised_image = acc_image/(i+1)
        if logger.getEffectiveLevel() < logging.WARNING:
            if GT != None:
                _PSNR = information_theory.distortion.PSNR(denoised_image, GT)
            else:
                _PSNR = 0.0
            PSNR_vs_iteration.append(_PSNR)
            axs[0].imshow(denoised_image.astype(np.uint8))
            axs[0].set_title(f"iter {i} " + f"({_PSNR:4.2f}dB)")
            axs[1].imshow(normalize(prev - denoised_image + 128).astype(np.uint8), cmap="gray")
            axs[1].set_title(f"diff")
            plt.show()
        randomized_noisy_image = randomize(noisy_image, mean_RD, sigma_RD).astype(np.float32)
        randomized_and_compensated_noisy_image = warp_B_to_A(
            A=randomized_noisy_image,
            B=denoised_image,
            l=l,
            w=w,
            sigma=sigma_OF)
        acc_image += randomized_and_compensated_noisy_image
    denoised_image = acc_image/(N_iters + 1)
    print()

    if logger.getEffectiveLevel() < logging.WARNING:
        return denoised_image, PSNR_vs_iteration
    else:
        return denoised_image, None

def _denoise(warp_B_to_A, noisy_image, N_iters=50, mean_RD=0.0, sigma_RD=1.0, l=3, w=2, sigma_OF=0.3):
    logger.info(f"N_iters={N_iters} mean_RD={mean_RD} sigma_RD={sigma_RD} l={l} w={w} sigma_OF={sigma_OF}")
    acc_image = np.zeros_like(noisy_image, dtype=np.float32)
    acc_image[...] = noisy_image
    for i in range(N_iters):
        print(f"iter={i}", end=' ')
        denoised_image = acc_image/(i+1)
        randomized_noisy_image = randomize(noisy_image, mean_RD, sigma_RD).astype(np.float32)
        randomized_and_compensated_noisy_image = warp_B_to_A(A=randomized_noisy_image, B=denoised_image, l=l, w=w, sigma=sigma_OF)
        acc_image += randomized_and_compensated_noisy_image
    denoised_image = acc_image/(N_iters + 1)
    print()
    return denoised_image
