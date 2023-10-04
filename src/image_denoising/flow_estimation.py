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

class Farneback_Flow_Estimator(motion_estimation.farneback.Estimator_in_CPU):

    def __init__(self,
                 levels=3, # Pyramid slope. Multiply by 2^levels the searching area if the OFE
                 window_side=15, # Applicability window side
                 iters=3, # Number of iterations at each pyramid level
                 poly_n=5, # Size of the pixel neighborhood used to find polynomial expansion in each pixel
                 poly_sigma=1.5, # Standard deviation of the Gaussian basis used in the polynomial expansion
                 flags=0):#cv2.OPTFLOW_FARNEBACK_GAUSSIAN):
        
        super().__init__(levels=levels,
                         pyr_scale=0.5,
                         win_side=window_side,
                         iters=iters,
                         poly_n=poly_n,
                         poly_sigma=poly_sigma,
                         flags=flags)
    
    def get_flow_to_project_A_to_B(self, A, B):
        flow = self.get_flow(target=B, reference=A, prev_flow=None)
        #flow = cv2.calcOpticalFlowFarneback(prev=B, next=A, flow=None,
        #                                pyr_scale=0.5, levels=self.levels, winsize=self.win_side,
        #                                iterations=3, poly_n=5, poly_sigma=self.poly_sigma,
        #                                flags=0)
        return flow
    
def project(image, flow):
    projection = motion_estimation.project(
        image,
        flow,
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
    return projection


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

