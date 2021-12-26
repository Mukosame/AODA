"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import logging
import os
import time

import torch
import util.util as util
from data import create_dataset
from models import create_model
from options.train_options import TrainOptions
from util.visualizer import Visualizer

if __name__ == "__main__":
    opt = TrainOptions().parse()  # get training options
    # create a dataset given opt.dataset_mode and other options
    dataset = create_dataset(opt)
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print("The number of training images = %d" % dataset_size)
    # loading resume state if exists
    if opt.resume_state:
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt.resume_state, map_location=lambda storage, loc: storage.cuda(
                device_id)
        )
        # option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    # create a visualizer that display/save images and plots
    visualizer = Visualizer(opt)
    if resume_state is None:
        print("create model and training state directory ...")
        model_dir = os.path.join(opt.checkpoints_dir, opt.name, "models")
        state_dir = os.path.join(opt.checkpoints_dir, opt.name, "states")
        util.mkdirs([model_dir, state_dir])

    util.setup_logger(
        "base",
        os.path.join(opt.checkpoints_dir, opt.name),
        "train_" + opt.name,
        level=logging.INFO,
        screen=True,
        tofile=True,
    )
    logger = logging.getLogger("base")

    # create a model given opt.model and other options
    model = create_model(opt)
    # resume training
    if resume_state:
        logger.info(
            "Resuming training from epoch: {}, iter: {}.".format(
                resume_state["epoch"], resume_state["iter"]
            )
        )
        start_epoch = resume_state["epoch"]
        total_iters = resume_state["iter"]
        # regular setup: load and print networks; create schedulers
        model.setup(opt)
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        total_iters = 0
        start_epoch = opt.epoch_count
        # regular setup: load and print networks; create schedulers
        model.setup(opt)
    if opt.pretrained_path:
        model.load_pretrain_from_path(opt.pretrained_path, opt.epoch)

    if "ebug" in opt.name:  # Fast debugging mode
        opt.print_freq = 1
        opt.batch_size = 2
        opt.save_latest_freq = 8

    # training
    logger.info(
        "Start training from epoch: {:d}, iter: {:d}".format(
            start_epoch, total_iters)
    )
    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    for epoch in range(start_epoch, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        visualizer.reset()

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration

            total_iters += opt.batch_size
            # unpack data from dataset and apply preprocessing
            model.set_input(data)
            # calculate loss functions, get gradients, update network weights
            model.optimize_parameters()
            # model.evaluate_correct()

            if (
                total_iters % opt.display_freq == 0
            ):  # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(
                    model.get_current_visuals(), epoch, save_result
                )

            if (
                total_iters % opt.print_freq == 0
            ):  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                t_data = iter_start_time - iter_data_time
                # visualizer.print_current_losses(epoch, total_iters, losses, t_comp, t_data)
                # message = '(epoch: %d, iter: %d, time: %.3f, data: %.3f) ' % (epoch, total_iters, t_comp, t_data)
                message = "(epoch: %d, iter: %d, data: %.3f) " % (
                    epoch,
                    total_iters,
                    t_data,
                )
                for k, v in losses.items():
                    message += "%s: %.3f " % (k, v)
                logger.info(message)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(
                        epoch, float(total_iters) / dataset_size, losses
                    )

            if (
                total_iters % opt.save_latest_freq == 0
            ):  # cache our latest model every <save_latest_freq> iterations
                logger.info(
                    "saving the latest model (epoch %d, total_iters %d)"
                    % (epoch, total_iters)
                )
                save_suffix = "iter_%d" % total_iters if opt.save_by_iter else "latest"
                model.save_networks(save_suffix)
                model.save_training_state(epoch, total_iters)

            iter_data_time = time.time()
        if (
            epoch % opt.save_epoch_freq == 0
        ):  # cache our model every <save_epoch_freq> epochs
            logger.info(
                "saving the model at the end of epoch %d, iters %d"
                % (epoch, total_iters)
            )
            model.save_networks("latest")
            model.save_networks(epoch)
            model.save_training_state(epoch, total_iters)

        logger.info(
            "End of epoch %d / %d \t Time Taken: %d sec"
            % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time)
        )
        # update learning rates at the end of every epoch.
        model.update_learning_rate()
