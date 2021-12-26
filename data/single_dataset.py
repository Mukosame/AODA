from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image

from .util import scribble2idx, sketchcoco2idx


class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        input_nc = (
            self.opt.output_nc if self.opt.direction == "BtoA" else self.opt.input_nc
        )
        self.transform = get_transform(opt, grayscale=(input_nc == 1))
        self.flag_scribble = "ribble" in opt.dataroot
        self.flag_sketchycoco = "Sketch/" in opt.dataroot

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert("RGB")
        A = self.transform(A_img)
        if self.opt.n_classes > 0:
            nake = A_path.split("/")[-2]
            if self.flag_scribble:
                idx = scribble2idx(nake)
            elif self.flag_sketchycoco:
                idx = sketchcoco2idx(nake)
            else:
                idx = int(nake)
            return {"A": A, "A_paths": A_path, "eta": idx}
        else:
            return {"A": A, "A_paths": A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
