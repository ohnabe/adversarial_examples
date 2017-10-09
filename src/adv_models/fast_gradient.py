import chainer.functions as F
from chainer import Variable
import numpy as np


def fast_gradient(chainer_model, chainer_array, eps):
    target_array = Variable(chainer_array)
    predict_result = F.softmax(chainer_model(chainer_array)).data
    orig_ind = np.argmax(predict_result)

    # create adv array
    loss = F.softmax_cross_entropy(chainer_model(target_array), Variable(np.array([orig_ind.astype(np.int32)])))
    loss.backward()
    adv_part = np.sign(target_array.grad)
    adv_array = target_array.data + eps * adv_part
    return adv_array.astype(np.float32), adv_part, predict_result