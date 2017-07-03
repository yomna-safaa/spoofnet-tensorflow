
import numpy as np
import PIL.Image
import scipy.misc

############################################################################
def load_and_resize_image(path, height, width, mode='RGB'):
    """
    Returns an np.ndarray (height x width x channels)

    mode -- (RGB for color or L for grayscale)
    """

    image = PIL.Image.open(path)  # YY => RGB 8 bits, jpeg format, instance of Image class, not ndarray
    image = image.convert(mode)
    image = np.array(image)  # YY => ndarray, uint8 values bet 0 and 255, shape 240x320x3 (h x w x ch)
    if height > 0 and width > 0:
        image = scipy.misc.imresize(image, (height, width),
                                    'bilinear')  # YY => ndarray, uint8 values bet 0 and 255, shape (h2 x w2 x ch)

    return image

