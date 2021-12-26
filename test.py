"""General-purpose test script for image-to-image translation. 
Original code: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/test.py

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test an AODA model (one side only):
        python3 test.py --model_suffix _B --dataroot ./scribble_10class/testA --name scribble_aoda --model test --phase test --no_dropout --n_classes 10

    The option '--model test' is used for generating results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model aoda_gan' requires loading and generating results in both directions, and n_classes
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test an aoda model:
        python test.py --dataroot ./datasets/scribble --phase test --no_dropout --n_classes 10 --name scribble_aoda --model aoda_gan --direction BtoA

"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import time

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.serial_batches = True
    # no flip; comment this line if results on flipped images are needed.
    opt.no_flip = True
    # no visdom display; the test code saves the results to a HTML file.
    opt.display_id = -1
    # create a dataset given opt.dataset_mode and other options
    dataset = create_dataset(opt)
    # create a model given opt.model and other options
    model = create_model(opt)
    # regular setup: load and print networks; create schedulers
    model.setup(opt)
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(
        opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (
        opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.

    if opt.eval:
        model.eval()
    consume_time = []
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        start_t = time.time()
        model.test()           # run inference
        end_t = time.time()
        consume_time.append(end_t - start_t)
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path,
                    aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    webpage.save()  # save the HTML
    sum_time = sum(consume_time)
    avg_time = float(sum_time) / len(consume_time)
    print('Total runtime {}s for {} images, average runtime {}'.format(
        sum_time, len(consume_time), avg_time))
