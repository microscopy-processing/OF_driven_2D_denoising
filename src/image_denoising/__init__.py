import logging
logger = logging.getLogger(__name__)
#logging.basicConfig(format="[%(filename)s:%(lineno)s %(funcName)s()] %(message)s")
#logger.setLevel(logging.CRITICAL)
#logger.setLevel(logging.ERROR)
#logger.setLevel(logging.WARNING)
logger.setLevel(logging.INFO)
#logger.setLevel(logging.DEBUG)

logger.info(f"__init__ run")

from image_denoising import gaussian
from image_denoising import OF_gaussian
from image_denoising import OF_random
