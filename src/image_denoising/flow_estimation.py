import numpy as np
import cv2
#import image_denoising
#from motion_estimation import farneback
#from motion_estimation import predict
import logging
#import image_denoising

import logging
logger = logging.getLogger(__name__)
#logging.basicConfig(format="[%(filename)s:%(lineno)s %(funcName)s()] %(message)s")
#logger.setLevel(logging.CRITICAL)
#logger.setLevel(logging.ERROR)
logger.setLevel(logging.WARNING)
#logger.setLevel(logging.INFO)
#logger.setLevel(logging.DEBUG)

#image_denoising.logger.info(f"Logging level: {image_denoising.logger.getEffectiveLevel()}")

#pip install "color_transforms @ git+https://github.com/vicente-gonzalez-ruiz/motion_estimation"
import motion_estimation

class Farneback_Flow_Estimator(motion_estimation.frarneback.Estimator_in_CPU):

    def __init__(self,
                 levels=3,
                 pyr_scale=0.5,
                 fast_piramids=False,
                 win_side=15,
                 iters=5,
                 poly_n=5,
                 pyr_sigma=1.5,
                 flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN):
        
        super().__init__(levels,
                         pyr_scale,
                         fast_piramids,
                         win_side,
                         iters,
                         poly_n=,
                         pyr_sigma,
                         flags)
    
    def get_flow_to_project_A_to_B(A, B):
        flow = self.get_flow(target=A, reference=B, prev_flow=None)
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


'''
def get_flow_to_project_A_to_B(A, B, l=3, w=15, prev_flow=None, sigma=1.5, iterations=5):
    # projection(next, flow) ~ prev
    flow = motion_estimation.farneback.get_flow(
        reference=B,
        target=A,
        prev_flow=prev_flow,
        pyr_scale=0.5,
        levels=l,
        winsize=w,
        iterations=iterations,
        poly_n=5,
        poly_sigma=sigma,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    #flow = cv2.calcOpticalFlowFarneback(prev=A, next=B, flow=prev_flow,
    #                                    pyr_scale=0.5, levels=l, winsize=w,
    #                                    iterations=5, poly_n=5, poly_sigma=sigma,
    #                                    flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    #                                   )
    #flow[...] = 0.0
    logger.info(f"avg_OF={np.average(np.abs(flow)):4.2f}")
    logger.debug(f"l={l} w={w} sigma={sigma} iters={iterations}")

    #print(f"avg_OF={np.average(np.abs(flow))}, l={l}, w={w}, sigma={sigma}", end=' ')
    return flow
'''

