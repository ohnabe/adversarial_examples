import chainer.functions as F
from chainer import Variable
import numpy as np

def predict(chainer_model, target_array):
    target_result = F.softmax(chainer_model(target_array))
    target_prob = np.max(target_result.data)
    target_ind = np.argmax(target_result.data)
    return target_prob, target_ind


def predict_least_likely(chainer_model, target_array):
    target_result = F.softmax(chainer_model(target_array))
    target_prob = np.max(target_result.data)
    target_ind = np.argmin(target_result.data)
    return target_prob, target_ind
