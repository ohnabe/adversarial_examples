import chainer.functions as F
from chainer import Variable
import numpy as np
import utils.predict
import os
import sys

def fast_gradient(chainer_model, chainer_array, eps):
    sys.path.append(os.getcwd())
    target_array = Variable(chainer_array)
    orig_prob, orig_ind = utils.predict.predict(chainer_model, target_array)

    # create adv array
    loss = F.softmax_cross_entropy(chainer_model(target_array), Variable(np.array([orig_ind.astype(np.int32)])))
    loss.backward()
    adv_part = np.sign(target_array.grad)
    adv_array = target_array.data + eps * adv_part
    return adv_array.astype(np.float32), adv_part, orig_prob, orig_ind