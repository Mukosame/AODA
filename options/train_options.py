from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # visdom and HTML visualization parameters
        parser.add_argument(
            "--display_freq",
            type=int,
            default=400,
            help="frequency of showing training results on screen",
        )
        parser.add_argument(
            "--display_ncols",
            type=int,
            default=4,
            help="if positive, display all images in a single visdom web panel with certain number of images per row.",
        )
        parser.add_argument(
            "--display_id", type=int, default=1, help="window id of the web display"
        )
        parser.add_argument(
            "--display_server",
            type=str,
            default="http://localhost",
            help="visdom server of the web display",
        )
        parser.add_argument(
            "--display_env",
            type=str,
            default="main",
            help='visdom display environment name (default is "main")',
        )
        parser.add_argument(
            "--display_port",
            type=int,
            default=8097,
            help="visdom port of the web display",
        )
        parser.add_argument(
            "--update_html_freq",
            type=int,
            default=1000,
            help="frequency of saving training results to html",
        )
        parser.add_argument(
            "--print_freq",
            type=int,
            default=100,
            help="frequency of showing training results on console",
        )
        parser.add_argument(
            "--no_html",
            action="store_true",
            help="do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/",
        )
        # network saving and loading parameters
        parser.add_argument(
            "--save_latest_freq",
            type=int,
            default=5000,
            help="frequency of saving the latest results",
        )
        parser.add_argument(
            "--save_epoch_freq",
            type=int,
            default=10,
            help="frequency of saving checkpoints at the end of epochs",
        )
        parser.add_argument(
            "--save_by_iter",
            default=False,
            action="store_true",
            help="whether saves model by iteration",
        )
        parser.add_argument(
            "--resume_state", help="resume training from the given state file"
        )
        parser.add_argument(
            "--continue_train",
            action="store_true",
            help="continue training: load the latest model",
        )
        parser.add_argument(
            "--pretrained_path",
            default=None,
            help="specify the folder path to the pretrained weigths",
        )
        parser.add_argument(
            "--epoch_count",
            type=int,
            default=1,
            help="the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...",
        )
        parser.add_argument(
            "--phase", type=str, default="train", help="train, val, test, etc"
        )
        # training parameters
        parser.add_argument(
            "--n_epochs",
            type=int,
            default=50,
            help="number of epochs with the initial learning rate",
        )
        parser.add_argument(
            "--n_epochs_decay",
            type=int,
            default=150,
            help="number of epochs to linearly decay learning rate to zero",
        )
        parser.add_argument(
            "--niter", type=int, default=400, help="# of iter at starting learning rate"
        )
        parser.add_argument(
            "--niter_decay",
            type=int,
            default=100,
            help="# of iter to linearly decay learning rate to zero",
        )
        parser.add_argument(
            "--beta1", type=float, default=0.5, help="momentum term of adam"
        )
        parser.add_argument(
            "--lr", type=float, default=0.0002, help="initial learning rate for adam"
        )
        parser.add_argument(
            "--lr_g", type=float, default=1e-4, help="initial learning rate for adam"
        )
        parser.add_argument(
            "--lr_d", type=float, default=4e-4, help="initial learning rate for adam"
        )
        parser.add_argument(
            "--lambda_GAN", type=float, default=1.0, help="weight for GAN Loss"
        )
        parser.add_argument(
            "--lambda_A",
            type=float,
            default=10.0,
            help="weight for cycle loss (A -> B -> A)",
        )
        parser.add_argument(
            "--lambda_B",
            type=float,
            default=1.0,
            help="weight for cycle loss (B -> A -> B)",
        )
        parser.add_argument(
            "--lambda_identity",
            type=float,
            default=0,
            help="use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1",
        )
        parser.add_argument(
            "--gan_mode",
            type=str,
            default="lsgan",
            help="the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.",
        )
        parser.add_argument(
            "--pool_size",
            type=int,
            default=50,
            help="the size of image buffer that stores previously generated images",
        )
        parser.add_argument(
            "--lr_policy",
            type=str,
            default="step",
            help="learning rate policy. [linear | step | plateau | cosine]",
        )
        parser.add_argument(
            "--lr_decay_iters",
            type=int,
            default=50,
            help="multiply by a gamma every lr_decay_iters iterations",
        )
        parser.add_argument(
            "--identity",
            type=float,
            default=0.5,
            help="use identity mapping. Setting identity other than 1 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set optidentity = 0.1",
        )
        parser.add_argument(
            "--lambda_C", type=float, default=0.0, help="weight for vgg loss"
        )
        parser.add_argument(
            "--no_ganFeat_loss",
            action="store_true",
            help="if specified, do *not* use discriminator feature matching loss",
        )
        parser.add_argument(
            "--lambda_E", type=float, default=0, help="weight for eta related loss"
        )
        parser.add_argument(
            "--mix_rec", action="store_true", help="if to mix the recon as generation"
        )
        parser.add_argument(
            "--rand_p", type=float, default=0.5, help="above this value, substitute"
        )

        self.isTrain = True
        return parser
