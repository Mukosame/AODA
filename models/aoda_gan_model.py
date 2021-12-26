import itertools
import random

import torch
from util.image_pool import ImagePool, IClassPool

from . import networks
from .base_model import BaseModel
from .losses import FocalLoss


def rgb2gray(x):
    """
    x_shape: (B, C, H, W)
    C: RGB
    weighted grayscale by luminosity: 0.3*R + 0.59*G + 0.11*B
    """
    return 0.3 * x[:, 0, ...] + 0.59 * x[:, 1, ...] + 0.11 * x[:, 2, ...]


class AODAGANModel(BaseModel):
    """
    This class implements the adversarial open-domain adaptation GAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For aodaGAN, in addition to GAN losses, we also have lambda_A, lambda_C, lambda_E for the following losses:
        A -> B -> A
        Generators: G_A: A->B; G_B: B->A
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        A-consistency (like forward cycle loss): lambda_A * ||G_B(G_A(A)) - A||
        Context-loss: lambda_C * ||fea_A - fea_(G_B(B))||
        controlled-eta-loss: lambda_E * ||eta(G_B(B)) - \eta||
        """
        parser.set_defaults(
            no_dropout=True)  # default CycleGAN did not use dropout
        return parser

    def __init__(self, opt):
        """Initialize the AodaGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = [
            "D_A",
            "G_A",
            "D_B",
            "G_B",
        ]
        if self.isTrain:
            if self.opt.lambda_A > 0:
                self.loss_names.append("cons")
            if self.opt.lambda_E > 0 or self.opt.n_classes > 1:
                self.loss_names.append("eta")
            if self.opt.lambda_C > 0:
                self.loss_names.append("conx")
            if self.opt.lambda_identity > 0:
                self.loss_names = self.loss_names + ["idt_A", "idt_B"]
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ["real_A", "fake_B", "rec_A"]
        visual_names_B = ["real_B", "fake_A"]
        self.visual_names = (
            visual_names_A + visual_names_B
        )  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ["G_A", "G_B", "D_A", "D_B"]
            if self.opt.lambda_E > 0:
                self.model_names.append("R")
            if (
                ("rand" in self.opt.name)
                or ("pool" in self.opt.name)
                or ("rp" in self.opt.name)
            ):
                self.visual_names.append("real_B2")
        else:  # during test time, only load Gs
            self.model_names = ["G_A", "G_B"]

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(
            opt.input_nc,
            opt.output_nc,
            opt.ngf,
            opt.netG,
            opt.norm,
            not opt.no_dropout,
            opt.init_type,
            opt.init_gain,
            self.gpu_ids,
        )  # with dropout

        if opt.netGC:
            self.flag_B = 1
            self.netG_B = networks.define_GC(
                opt.n_classes,
                opt.output_nc,
                opt.input_nc,
                opt.ngf,
                opt.netGC,
                opt.norm,
                not opt.no_dropout,
                opt.init_type,
                opt.init_gain,
                self.gpu_ids,
            )
        else:
            self.flag_B = 0
            self.netG_B = networks.define_G(
                opt.output_nc,
                opt.input_nc,
                opt.ngf,
                opt.netG,
                opt.norm,
                not opt.no_dropout,
                opt.init_type,
                opt.init_gain,
                self.gpu_ids,
            )

        if opt.nz > 0:
            self.noise = self.Tensor(opt.batch_size, opt.nz)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(
                opt.output_nc,
                opt.ndf,
                opt.netD,
                opt.n_layers_D,
                opt.norm,
                opt.init_type,
                opt.init_gain,
                self.gpu_ids,
            )
            self.netD_B = networks.define_D(
                opt.input_nc,
                opt.ndf,
                opt.netD,
                opt.n_layers_D,
                opt.norm,
                opt.init_type,
                opt.init_gain,
                self.gpu_ids,
            )
            # feature loss, eta loss
            if self.opt.lambda_C > 0:
                self.netF = networks.define_F(self.gpu_ids, use_bn=False)
                self.netF.to(self.device)
            if self.opt.lambda_E > 0:
                self.netR = networks.define_R(opt, self.gpu_ids)
                self.netR.to(self.device)

            if (
                opt.lambda_identity > 0.0
            ):  # only works when input and output images have the same number of channels
                assert opt.input_nc == opt.output_nc
                self.criterionIdt = torch.nn.L1Loss()
                self.visual_names.append("idt_B")
                self.visual_names.append("idt_A")
            self.fake_A_pool = ImagePool(
                opt.pool_size
            )  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(
                opt.pool_size
            )  # create image buffer to store previously generated images

            self.under_pool = IClassPool(opt.pool_size)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(
                self.device
            )  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            # context loss function
            self.criterionContext = torch.nn.L1Loss().to(self.device)

            # eta loss
            self.criterionEta = FocalLoss().to(self.device)

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_A.parameters(),
                                self.netG_B.parameters()),
                lr=opt.lr_g,
                betas=(opt.beta1, 0.999),
            )
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD_A.parameters(),
                                self.netD_B.parameters()),
                lr=opt.lr_d,
                betas=(opt.beta1, 0.999),
            )

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            if self.opt.lambda_E > 0:
                self.optimizer_R = torch.optim.Adam(
                    self.netR.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
                )
                self.optimizers.append(self.optimizer_R)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        if self.flag_B != 0:
            self.eta = input["eta"].to(
                self.device
            )  # input of B is onehot encoding label

            self.eta_s = input["eta_s"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]
        if self.opt.nz > 0:
            self.noise.normal_(0, 1)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.real_B2 = self.real_B.clone()
        if self.isTrain:
            if "aoda" in self.opt.name:  # adopt the random-mixed sampling strategy
                p = random.uniform(0, 1)
                if (
                    p > self.opt.rand_p and self.eta.size(
                        0) == self.opt.batch_size
                ):  # skip the last batch
                    self.real_B2, self.eta_s = self.under_pool.query(
                        self.fake_B, self.eta
                    )
        if self.flag_B == 0:  # gan
            self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
            self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        elif self.flag_B == 1:  # naive conditional gan
            self.rec_A = self.netG_B(self.fake_B, self.eta)  # G_B(G_A(A))
            self.fake_A = self.netG_B(self.real_B2, self.eta_s)  # G_B(B)

        self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))

    def backward_eta(self):
        # fake
        eta_s_fake = self.netR(self.fake_A.detach())
        loss_eta_s_fake = self.criterionEta(eta_s_fake, self.eta_s)
        # real
        eta_real = self.netR(self.real_A.detach())
        loss_eta_real = self.criterionEta(eta_real, self.eta)
        # recon
        if self.opt.mix_rec:
            eta_rec = self.netR(self.rec_A.detach())
            loss_eta_rec = self.criterionEta(eta_rec, self.eta)
            loss_eta = (loss_eta_s_fake + loss_eta_real + loss_eta_rec) / 3
        else:
            loss_eta = (loss_eta_s_fake + loss_eta_real) * 0.5
        loss_eta.backward()
        return loss_eta

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_recon(self, netD, real, fake, rec):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # rec
        pred_rec = netD(rec.detach())
        loss_D_rec = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake + loss_D_rec) / 3
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(
            self.netD_A, self.real_B, self.fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_C = self.opt.lambda_C
        lambda_E = self.opt.lambda_E

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = (
                self.criterionIdt(self.idt_A, self.real_B) *
                lambda_B * lambda_idt
            )
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A, self.eta)
            self.loss_idt_B = (
                self.criterionIdt(self.idt_B, self.real_A) *
                lambda_A * lambda_idt
            )
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        if self.opt.mix_rec:
            self.loss_G_B_rec = self.criterionGAN(
                self.netD_B(self.rec_A), True)
        else:
            self.loss_G_B_rec = 0
        self.loss_G_B_gen = self.criterionGAN(self.netD_B(self.fake_A), True)
        self.loss_G_B = self.loss_G_B_rec + self.loss_G_B_gen
        # consistency loss || G_B(G_A(A)) - A||
        self.loss_cons = (
            self.criterionCycle(self.rec_A, self.real_A) * lambda_A
            + self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        )

        # Context-loss: lambda_C * ||fea_A - fea_(G_B(B))||
        if lambda_C > 0:
            fea_real = self.netF(self.real_A)
            fea_rec = self.netF(self.rec_A)
            self.loss_conx = self.criterionContext(
                fea_real, fea_rec) * lambda_C
        # controlled-eta-loss: lambda_E * ||eta(G_B(B)) - \eta||
        if lambda_E > 0:
            # fake
            eta_s_fake = self.netR(self.fake_A)
            loss_eta_fake = self.criterionEta(eta_s_fake, self.eta_s)
            # recon
            if self.opt.mix_rec:
                eta_rec = self.netR(self.rec_A)
                loss_eta_rec = self.criterionEta(eta_rec, self.eta)
                self.loss_eta = (loss_eta_fake + loss_eta_rec) * 0.5
            else:
                self.loss_eta = loss_eta_fake
        else:
            self.loss_eta = 0
        # combined loss and calculate gradients
        self.loss_G = (
            self.loss_G_A
            + self.loss_G_B
            + self.loss_cons
            + self.loss_eta
            + self.loss_idt_A
            + self.loss_idt_B
        )
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad(
            [self.netD_A, self.netD_B], False
        )  # Ds require no gradients when optimizing Gs
        if self.opt.lambda_E > 0:
            self.set_requires_grad([self.netR], False)
        if self.opt.lambda_C > 0:
            self.set_requires_grad([self.netF], False)

        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
        # # classifier
        if self.opt.lambda_E > 0:
            self.set_requires_grad([self.netR], True)
            self.optimizer_R.zero_grad()
            self.backward_eta()
            self.optimizer_R.step()
