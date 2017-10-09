import chainer.functions as F
from chainer import Variable
import numpy as np


def iterative_least_likely(chainer_model, chainer_array, eps, alpha):
    adv_images = np.copy(chainer_array)
    predict_result = F.softmax(chainer_model(chainer_array)).data
    least_ind = np.argmin(predict_result)

    iter_num = int(min(eps+4, 1.25*eps))
    print("iteration_number {}".format(iter_num))
    for _ in range(iter_num):
        adv_images = Variable(adv_images)
        loss = F.softmax_cross_entropy(chainer_model(adv_images), Variable(np.array([least_ind.astype(np.int32)])))
        loss.backward()
        adv_part = np.sign(adv_images.grad)
        adv_images = adv_images.data - alpha * eps * adv_part
        adv_images = np.clip(adv_images, chainer_array - eps, chainer_array + eps)
    return adv_images.astype(np.float32), adv_part, predict_result
