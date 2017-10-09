import chainer.functions as F
from chainer import Variable
import numpy as np
import utils.predict
import os
import sys


def iterative_least_likely(chainer_model, chainer_array, eps, iter_num, alpha):
    sys.path.append(os.getcwd())

    adv_images = np.copy(chainer_array)
    orig_prob, orig_ind = utils.predict.predict(chainer_model, Variable(adv_images))
    least_prob, least_ind = utils.predict.predict_least_likely(chainer_model, Variable(adv_images))

    for _ in range(iter_num):
        adv_images = Variable(adv_images)
        loss = F.softmax_cross_entropy(chainer_model(adv_images), Variable(np.array([least_ind.astype(np.int32)])))
        loss.backward()
        adv_part = np.sign(adv_images.grad)
        adv_images = adv_images.data - alpha * eps * adv_part
        adv_images = np.clip(adv_images, chainer_array - eps, chainer_array + eps)
    return adv_images.astype(np.float32), adv_part, orig_prob, orig_ind, least_prob, least_ind
