import os
import pickle

####################
# Files & IO
####################

# for scribble only


def scribble2idx(nake):
    if nake == "pineapple":
        A_nake = 0
    elif nake == "strawberry":
        A_nake = 1
    elif nake == "basketball":
        A_nake = 2
    elif nake == "chicken":
        A_nake = 3
    elif nake == "cookie":
        A_nake = 4
    elif nake == "cupcake":
        A_nake = 5
    elif nake == "moon":
        A_nake = 6
    elif nake == "orange":
        A_nake = 7
    elif nake == "soccer":
        A_nake = 8
    elif nake == "watermelon":
        A_nake = 9
    else:
        raise ValueError(
            "The input category name {} is not recognized by scribble.".format(
                nake)
        )
    return int(A_nake)


# for sketcycoco only


def sketchycoco2idx(nake):
    indices = [2, 3, 4, 5, 10, 11, 17, 18, 19, 20, 21, 22, 24, 25]
    idx = indices.index(int(nake))
    return int(idx)


###################### get image path list ######################
IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def _get_paths_from_images(path):
    """get image path list from image folder"""
    assert os.path.isdir(path), "{:s} is not a valid directory".format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, "{:s} has no valid image file".format(path)
    return images


def _get_paths_from_lmdb(dataroot):
    """get image path list from lmdb meta info"""
    meta_info = pickle.load(
        open(os.path.join(dataroot, "meta_info.pkl"), "rb"))
    paths = meta_info["keys"]
    sizes = meta_info["resolution"]
    if len(sizes) == 1:
        sizes = sizes * len(paths)
    return paths, sizes


def get_image_paths(data_type, dataroot):
    """get image path list
    support lmdb or image files"""
    paths, sizes = None, None
    if dataroot is not None:
        if data_type == "lmdb":
            paths, sizes = _get_paths_from_lmdb(dataroot)
        elif data_type == "img":
            paths = sorted(_get_paths_from_images(dataroot))
        else:
            raise NotImplementedError(
                "data_type [{:s}] is not recognized.".format(data_type)
            )
    return paths, sizes
