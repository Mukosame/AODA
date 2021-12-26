import os.path
import random

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image

from .util import scribble2idx, sketchycoco2idx


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        if opt.datarootA and opt.datarootB:
            self.dir_A = opt.datarootA
            self.dir_B = opt.datarootB
        else:
            # create a path '/path/to/data/trainA'
            self.dir_A = os.path.join(opt.dataroot, opt.phase + "A")
            # create a path '/path/to/data/trainB'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + "B")
        self.flag_scribble = ("ribble" in self.dir_A) or (opt.n_classes == 10)
        self.flag_sketchycoco = opt.n_classes == 14
        # load images from '/path/to/data/trainA'
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        # load images from '/path/to/data/trainB'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == "BtoA"
        # get the number of channels of input image
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc
        # get the number of channels of output image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """

        # make sure index is within then range
        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:  # make sure index is within then range
            index_B = index % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert("RGB")
        B_img = Image.open(B_path).convert("RGB")
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        if self.opt.n_classes > 0:
            A_nake = A_path.split("/")[-2]
            B_nake = B_path.split("/")[-2]
            if self.flag_scribble:
                A_idx = scribble2idx(A_nake)  # sketch
                B_idx = scribble2idx(B_nake)  # rgb
            elif self.flag_sketchycoco:
                A_idx = sketchycoco2idx(A_nake)
                B_idx = sketchycoco2idx(B_nake)
            else:
                A_idx = int(A_nake)
                B_idx = int(B_nake)
            return {
                "A": A,
                "B": B,
                "A_paths": A_path,
                "B_paths": B_path,
                "eta": B_idx,
                "eta_s": A_idx,
            }
        else:
            return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
