import chainer.functions as F
from chainer import Variable
import numpy as np


def iterative_least_likely(chainer_model, target_img, ll_ind_v, eps, iter_num, alpha):
    adv_images = np.copy(target_img)
    for _ in range(iter_num):
        adv_images = Variable(adv_images)
        loss = F.softmax_cross_entropy(chainer_model(adv_images), ll_ind_v)
        loss.backward()
        adv_part = np.sign(adv_images.grad)
        adv_images = adv_images.data - alpha * eps * adv_part
        adv_images = np.clip(adv_images, target_img - eps, target_img + eps)
    return adv_images.astype(np.float32), adv_part
