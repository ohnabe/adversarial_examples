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


IMAGENET_MEAN_FILE = "../data/ilsvrc_2012_mean.npy"

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


def create_diff_mean_image(chainer_array, pic_mean_data_path):
    # mean_value: 104 B
    # mean_value: 117 G
    # mean_value: 123 R
    tmp_image = chainer_array.copy()
    mean_data = np.load(pic_mean_data_path)
    mean_data = Image.fromarray(mean_data.astype(np.uint8), 'RGB').resize((227, 227))
    mean_data = np.asarray(mean_data).transpose(2, 0, 1)
    #tmp_image = tmp_image - mean_data
    tmp_image[0] = tmp_image[0] - mean_data[0]
    tmp_image[1] = tmp_image[1] - mean_data[1]
    tmp_image[2] = tmp_image[2] - mean_data[2]
    return tmp_image


def resize_image(original_image_path):
    img = Image.open(original_image_path)
    print(img.size, img.mode)

    img_resize = img.resize((227, 227))
    print(img_resize.size, img_resize.mode)
    return img_resize


def format2chainer(img_data):
    arrayImg = np.asarray(img_data).astype(np.float32)[:, :, ::-1]
    arrayImg = arrayImg.transpose(2, 0, 1)
    arrayImg = create_diff_mean_image(arrayImg, IMAGENET_MEAN_FILE)
    arrayImg = arrayImg.reshape((1,) + arrayImg.shape)
    return arrayImg


def format2orig(chainer_img):
    orig_image = chainer_img.transpose(1, 2, 0).astype(np.uint8)
    orig_image = orig_image[:,:,::-1]
    return orig_image


def predict(target_image, label_d):
    chainer_img = format2chainer(target_image)
    raw_result = chainer_model(chainer_img)
    result = F.softmax(raw_result)
    target_ind = np.argmax(result.data)
    return np.max(result.data), target_ind, label_d[target_ind]


def create_label_list(label_file_path):
    label_d = {}
    with open(label_file_path, "r") as f:
        for line in f:
            line = line.rstrip("\n").strip(" ").split(":")
            if len(line) == 2:
               label_d[int(line[0])] = line[1]
    return label_d


def create_adv_sample(target_image, target_ind, eps):
    chainer_img = Variable(format2chainer(target_image))
    target_ind_v = Variable(np.array([target_ind.astype(np.int32)]))
    loss = F.softmax_cross_entropy(chainer_model(chainer_img), target_ind_v)
    loss.backward()
    adv_part = np.sign(chainer_img.grad)
    adv_images = chainer_img.data + eps * adv_part
    return adv_images.astype(np.float32), adv_part


if __name__ == '__main__':
    #model = load_caffemodel("models/bvlc_alexnet.caffemodel")
    #save_models(model, "models/alexnet.chainermodel")

    caffe_model = load_models("../models/alexnet.chainermodel")
    chainer_model = models.Alex.Alex()

    # get label dict
    label_d = create_label_list("../data/imagenet_label.txt")

    # copy caffe_model W, b to chainer_model
    copy_model(caffe_model, chainer_model)

    # predict target_image
    resize_img = resize_image("../data/panda2.jpeg")
    resize_img.show()
    #resize_img = create_diff_mean_image(resize_img, "./data/ilsvrc_2012_mean.npy")
    prob, label_ind, label = predict(resize_img, label_d)

    print(prob)
    print(label)

    # create adv_sample
    adv_image, adv_part = create_adv_sample(resize_img, label_ind, eps=0.2)

    # show adv_image
    adv_image = format2orig(adv_image[0])
    adv_part = format2orig(adv_part[0])
    Image.fromarray(adv_image, 'RGB').show()
    Image.fromarray(adv_part, 'RGB').show()

    adv_prob, adv_label_ind, adv_label = predict(adv_image, label_d)
    print(adv_prob)
    print(adv_label)

    part_prob, part_label_ind, part_label = predict(adv_part, label_d)
    print(part_prob)
    print(part_label)



    """
    tmp = format2chainer(resize_img)
    #print(tmp.shape)
    tmp1 = tmp[0].transpose(1, 2, 0).astype(np.uint8)
    Image.fromarray(tmp1, 'RGB').show()
    Image.fromarray(np.asarray(resize_img), 'RGB').show()
    tmp2 = np.asarray(tmp1) - np.asarray(resize_img)
    print(np.where(tmp2 !=0))
    #print(np.asarray(resize_img))
    #print("a")
    #print(tmp1)
    """

