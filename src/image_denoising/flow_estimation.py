import numpy as np
import cv2
import image_denoising
from motion_estimation import farneback
from motion_estimation import predict
import motion_estimation

def get_flow_to_project_A_to_B(A, B, l=3, w=15, prev_flow=None, sigma=1.5):
    # projection(next, flow) ~ prev
    flow = motion_estimation.farneback.get_flow(
        reference=A,
        target=B,
        prev_flow=prev_flow,
        pyr_scale=0.5,
        levels=l,
        winsize=w,
        iterations=5,
        poly_n=5,
        poly_sigma=sigma,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    #flow = cv2.calcOpticalFlowFarneback(prev=A, next=B, flow=prev_flow,
    #                                    pyr_scale=0.5, levels=l, winsize=w,
    #                                    iterations=5, poly_n=5, poly_sigma=sigma,
    #                                    flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    #                                   )
    #flow[...] = 0.0
    image_denoising.logger.info(f"avg_OF={np.average(np.abs(flow))} l={l} w={w} poly_sigma={sigma}")

    #print(f"avg_OF={np.average(np.abs(flow))}, l={l}, w={w}, sigma={sigma}", end=' ')
    return flow

def project(image, flow):
    warped_image = motion_estimation.predict.warp(
        reference=image,
        flow=flow,
        interpolation_mode=cv2.INTER_LINEAR,
        extension_mode=cv2.BORDER_REPLICATE)
    #height, width = flow.shape[:2]
    #map_x = np.tile(np.arange(width), (height, 1))
    #map_y = np.swapaxes(np.tile(np.arange(height), (width, 1)), 0, 1)
    #map_xy = (flow + np.dstack((map_x, map_y))).astype('float32')
    #warped_image = cv2.remap(image, map_xy, None,
    #                         #interpolation=cv2.INTER_LANCZOS4,
    #                         interpolation=cv2.INTER_LINEAR,
    #                         #interpolation=cv2.INTER_CUBIC,
    #                         #interpolation=cv2.INTER_NEAREST,
    #                         #interpolation=cv2.INTER_AREA,
    #                         borderMode=cv2.BORDER_REPLICATE)
    return warped_image
