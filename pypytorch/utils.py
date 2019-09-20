# -*- coding: utf-8 -*-

import numpy as np
import pypytorch as t

def ensure_tensor(data):
    if isinstance(data, t.Tensor):
        return data
    return t.Tensor(data)

def ensure_tuple_list(data):
    if isinstance(data, tuple) or isinstance(data, list):
        return data
    return (data, data)

def ensure_one_hot(data, label_num):
    data = ensure_tensor(data)
    if data.dim() == 2:
        return data
    data.data = one_hot(data.data, label_num)
    return data

def one_hot(data, label_num):
    if data.shape == ():
        out = np.zeros((1, label_num))
        out[0, int(data.tolist())] = 1
        return out

    out = np.zeros((len(data), label_num))
    for i in range(len(data)):
        out[i, int(data[i])] = 1
    return out

def pair(data):
    return ensure_tuple_list(data)

def pair_tuple(*data):
    return map(lambda x: pair(x), data)

def make_padding(x, padding):
    return np.pad(x,
                    ((0, 0), (0, 0), 
                    (padding[0], padding[0]), 
                    (padding[1], padding[1])), 'constant', constant_values=0)

def unwrap_padding(x, padding):
    if padding == (0, 0):
        return x
    return x[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]

def softmax(inputs, dim=0):
    exp = np.exp(inputs - np.max(inputs, axis=dim, keepdims=True))
    return exp / np.sum(exp, axis=dim, keepdims=True)

def adjust_lr(optimizer, epoch, initial_lr, lr_decay):
    lr = initial_lr / (1.0 + epoch * lr_decay)
    optimizer.lr = lr
    return lr

def _handle_padding(im, padding):
    if padding[0] == 0 and padding[1] == 0:
        return im
    if padding[0] == 0:
        return im[:, :, :, padding[1]:-padding[1]]
    if padding[1] == 0:
        return im[:, :, padding[0]:-padding[0], :]
    return im[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]

def im2col(im, kernel_size, stride, padding):
    batch, channels, height, width = im.shape
    im = np.pad(im, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant')
    im = np.transpose(im, (1, 2, 3, 0))
    out_height = (height - kernel_size[0] + 2 * padding[0]) // stride[0] + 1
    out_width = (width - kernel_size[1] + 2 * padding[1]) // stride[1] + 1
    col = np.zeros((kernel_size[0] * kernel_size[1] * channels, out_height * out_width * batch), dtype='int64')

    col_idx = 0

    for y in range(out_height):
        y_img_start = y * stride[0]
        y_img_end = y_img_start + kernel_size[0]
        for x in range(out_width):
            x_img_start = x * stride[1]
            x_img_end = x_img_start + kernel_size[1]
            col[:, col_idx:col_idx + batch] = im[:, y_img_start:y_img_end, x_img_start:x_img_end, :].reshape((col.shape[0], -1))
            col_idx += batch
    return col

def col2im(col, im, kernel_size, stride, padding):
    batch, channels, height, width = im.shape
    out_height = (height - kernel_size[0] + 2 * padding[0]) // stride[0] + 1
    out_width = (width - kernel_size[1] + 2 * padding[1]) // stride[1] + 1
    im = np.zeros_like(np.pad(im, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant'))
    im = np.transpose(im, (1, 2, 3, 0))

    col_idx = 0

    for y in range(out_height):
        y_img_start = y * stride[0]
        y_img_end = y_img_start + kernel_size[0]
        for x in range(out_width):
            x_img_start = x * stride[1]
            x_img_end = x_img_start + kernel_size[1]
            tmp = np.zeros_like(im)
            tmp[:, y_img_start:y_img_end, x_img_start:x_img_end, :] = col[:, col_idx:col_idx + batch].reshape(
                                                                        [-1, kernel_size[0], kernel_size[1], batch]
                                                                    )
            im += tmp
            col_idx += batch
    im = np.transpose(im, (3, 0, 1, 2))
    return _handle_padding(im, padding)


def fetch_args(args, num):
    values = [None] * num
    values[0:len(args)] = args
    return values