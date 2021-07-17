import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import numpy as np

from .utils import to_rgb

# constants
import configparser
config = configparser.SafeConfigParser()
config.read("htorch/experiments/constants.cfg")
MAX_ITER =  config.getint("crf", "max_iter")
POS_W = config.getint("crf", "pos_w")
POS_XY_STD = config.getint("crf", "pos_xy_std")
BI_XY_STD = config.getint("crf", "bi_xy_std")
BI_RGB_STD = config.getint("crf", "bi_rgb_std")
BI_W = config.getint("crf", "bi_w")



def dense_crf(img, output_probs):
    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    img = np.ascontiguousarray(img)
    rgb = to_rgb(img)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=BI_XY_STD, srgb=BI_RGB_STD, rgbim=rgb, compat=BI_W)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))
    return Q

def dense_crf_wrapper(args):
    return dense_crf(args[0], args[1])