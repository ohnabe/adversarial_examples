#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions
from chainer import initializers, optimizers, serializers, Variable
from chainer.links.caffe import CaffeFunction
import pickle
import models.Alex
from PIL import Image

from adv_models.fast_gradient import fast_gradient
from adv_models.iterative_gradient import iterative_gradient
from adv_models.iterative_least_likely import iterative_least_likely
import utils


IMAGENET_MEAN_FILE = "../data/ilsvrc_2012_mean.npy"
INPUT_IMAGE_SIZE = 227

def load_caffemodel(model_path):
    caffe_model = CaffeFunction(model_path)
    return caffe_model


def save_models(caffe_model, save_model_path):
    with open(save_model_path, 'wb') as f:
        pickle.dump(caffe_model, f)


def load_models(save_model_path):
    with open(save_model_path, 'rb') as f:
        model = pickle.load(f)
        return model


def copy_model(src, dst):
    assert isinstance(src, chainer.Chain)
    assert isinstance(dst, chainer.Chain)
    for child in src.children():
        if child.name not in dst.__dict__: continue
        dst_child = dst[child.name]
        if type(child) != type(dst_child): continue
        if isinstance(child, chainer.Chain):
            copy_model(child, dst_child)
        if isinstance(child, chainer.Link):
            match = True
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                if a[0] != b[0]:
                    match = False
                    break
                if a[1].data.shape != b[1].data.shape:
                    match = False
                    break
            if not match:
                print('Ignore %s because of parameter mismatch' % child.name)
                continue
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                b[1].data = a[1].data
            print('Copy %s' % child.name)


def create_mean_image_array(pic_mean_data_path, size_image):
    mean_data = np.load(pic_mean_data_path)
    mean_data = Image.fromarray(mean_data.astype(np.uint8), 'RGB').resize((size_image, size_image))
    mean_data = np.asarray(mean_data).astype(np.float32)
    return mean_data


def substract_mean_image(target_array, mean_array):
    # mean_value: 104 B
    # mean_value: 117 G
    # mean_value: 123 R
    result_array = target_array - mean_array
    return result_array


def add_mean_image(target_array, mean_array):
    result_array = target_array + mean_array
    return result_array


def resize_image(original_image_path):
    img = Image.open(original_image_path)
    print("original image format:{} {}".format(img.size, img.mode))

    img_resize = img.resize((INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))
    print("resize image format:{} {}".format(img_resize.size, img_resize.mode))
    return img_resize


def format2chainer(img_data):
    # RGB to GBR
    arrayImg = np.asarray(img_data).astype(np.float32)[:, :, ::-1]
    # HWC to CWH
    arrayImg = arrayImg.transpose(2, 0, 1)
    # 3-dimensions to 4-dimensions
    arrayImg = arrayImg.reshape((1,) + arrayImg.shape)
    return arrayImg


def format2orig(chainer_img):
    # CWH to HWC
    #orig_image = chainer_img.transpose(1, 2, 0).astype(np.uint8)
    orig_image = chainer_img.transpose(1, 2, 0)
    # BGR to RGB
    orig_image = orig_image[:,:,::-1]
    return orig_image


def create_label_list(label_file_path):
    label_d = {}
    with open(label_file_path, "r") as f:
        for line in f:
            line = line.rstrip("\n").strip(" ").split(":")
            if len(line) == 2:
               label_d[int(line[0])] = line[1].strip(" ")
    return label_d



if __name__ == '__main__':
    #model = load_caffemodel("models/bvlc_alexnet.caffemodel")
    #save_models(model, "models/alexnet.chainermodel")

    caffe_model = load_models("../models/alexnet.chainermodel")
    chainer_model = models.Alex.Alex()

    # get label dict
    label_d = create_label_list("../data/imagenet_label.txt")

    # copy caffe_model W, b to chainer_model
    copy_model(caffe_model, chainer_model)

    # create mean image array
    mean_image_array = create_mean_image_array(IMAGENET_MEAN_FILE, INPUT_IMAGE_SIZE)

    # predict target_image
    orig_img = resize_image("../data/panda.jpeg")
    orig_img.show()

    orig_array = np.asarray(orig_img)
    orig_array = substract_mean_image(orig_array, mean_image_array)


    chainer_array = format2chainer(orig_array)

    # apply gradient sign method
    #adv_array, adv_part_array, orig_prob, orig_ind = fast_gradient(chainer_model, chainer_array, eps=0.1)

    # apply iterative gradient sign method
    #adv_array, adv_part_array, orig_prob, orig_ind = iterative_gradient(chainer_model, chainer_array,
    #                                                                    eps=2.0, iter_num=10, alpha=1.0)

    # apply iterative least likely class method
    adv_array, adv_part_array, orig_prob, orig_ind, least_prob, least_ind = iterative_least_likely(chainer_model, chainer_array,
                                                                            eps=16.0, iter_num=4, alpha=1.0)
    print("least　likely category {}".format(label_d[least_ind]))

    # predict original image_result
    orig_label = label_d[orig_ind]
    print("predict_original_image: {} predict_prob: {}".format(orig_label.strip(" "), orig_prob))

    # predict adversarial_image
    part_prob, part_label_ind = utils.predict.predict(chainer_model, adv_part_array)
    part_label = label_d[part_label_ind]
    print("predict_adversarial_perturbations: {} predict_prob: {}".format(part_label, part_prob))

    adv_prob, adv_label_ind = utils.predict.predict(chainer_model, adv_array)
    adv_label = label_d[adv_label_ind]
    print("predict_adversarial_examples: {} predict_prob: {}".format(adv_label, adv_prob))

    # show adv_image
    adv_array = format2orig(adv_array[0])
    adv_part_array = format2orig(adv_part_array[0])
    adv_array = add_mean_image(adv_array, mean_image_array)
    adv_array = np.clip(adv_array, 0, 255)
    adv_array = adv_array.astype(np.uint8)
    Image.fromarray(adv_array, 'RGB').show()
    Image.fromarray(adv_part_array, 'RGB').show()

