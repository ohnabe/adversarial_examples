import chainer.functions as F
from chainer import Variable
import numpy as np

def fast_gradient(chainer_model, chainer_image, target_ind_v, eps):
    target_img = Variable(chainer_image)
    loss = F.softmax_cross_entropy(chainer_model(target_img), target_ind_v)
    loss.backward()
    adv_part = np.sign(target_img.grad)
    adv_images = target_img.data + eps * adv_part

    return adv_images.astype(np.float32), adv_part