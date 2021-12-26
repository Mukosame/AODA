"""This module contains simple helper functions """
from __future__ import print_function

import logging
import os
from datetime import datetime

import numpy as np
import torch
from PIL import Image


####################
# image convert
####################


def tensor2im(input_image, imtype=np.uint8):
    """ "Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = (
            image_tensor[0].cpu().float().numpy()
        )  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (
            (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        )  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def get_timestamp():
    return datetime.now().strftime("%y%m%d-%H%M%S")


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + "_archived_" + get_timestamp()
        print("Path already exists. Rename it to [{:s}]".format(new_name))
        logger = logging.getLogger("base")
        logger.info(
            "Path already exists. Rename it to [{:s}]".format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def setup_logger(
    logger_name, root, phase, level=logging.INFO, screen=False, tofile=False
):
    """set up logger"""
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(
            root, phase + "_{}.log".format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)
