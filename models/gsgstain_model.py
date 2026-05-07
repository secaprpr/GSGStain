import os
import torch
import torch.nn.functional as F

from .cut_model import CUTModel
from .graph_semantic import GraphSemanticRectificationModule


class GSGStainModel(CUTModel):
    """Graph-Semantic Guided virtual IHC staining.

    This model keeps CUT's adversarial and PatchNCE objectives, then adds the
    graph semantic rectification and dual-branch discriminator described by
    GSGStain for weakly paired H&E-to-IHC translation.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = CUTModel.modify_commandline_options(parser, is_train)
        parser.add_argument('--lambda_GSC', type=float, default=1.0, help='weight for graph semantic consistency loss')
        parser.add_argument('--lambda_R', type=float, default=0.1, help='weight for dual-branch ranking loss')
        parser.add_argument('--lambda_GSRM', type=float, default=1.0, help='weight for GSRM rectification losses')
        parser.add_argument('--graph_grid_size', type=int, default=8, help='grid nodes per side for graph construction')
        parser.add_argument('--graph_hidden_dim', type=int, default=128, help='hidden channels for the GSRM')
        parser.add_argument('--graph_he_dim', type=int, default=2048, help='H&E morphology feature dimension stored in precomputed graphs')
        parser.add_argument('--graph_layers', type=int, default=4, help='number of GSRM message passing layers')
        parser.add_argument('--graph_k', type=int, default=8, help='number of spatial neighbours for graph edges')
        parser.add_argument('--graph_pyramid_levels', type=int, default=3, help='HPC pyramid levels')
        parser.add_argument('--gsrm_conv', type=str, default='pna', choices=['pna', 'custom'],
                            help='message passing layer for GSRM; pna uses PyG PNAConv, custom uses the built-in lightweight fallback')
        parser.add_argument('--gsrm_pna_degree', type=int, default=9,
                            help='expected degree for PNA degree scaling, usually knn_k + self-loop')
        parser.add_argument('--gsg_stage', type=str, default='train_generator',
                            choices=['train_gsrm', 'train_generator', 'joint'],
                            help='two-stage GSGStain training stage')
        parser.add_argument('--gsg_pretrained_R_name', type=str, default=None,
                            help='checkpoint name that provides pretrained netR for train_generator')
        parser.add_argument('--gsg_R_epoch', type=str, default='latest',
                            help='which netR epoch to load for train_generator')

        parser.set_defaults(dataset_mode='gsgstain',
                            netD='dualbranch',
                            nce_idt=True,
                            lambda_NCE=1.0)
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.gsg_stage = opt.gsg_stage
        self.loaded_pretrained_R = False

        if self.isTrain:
            self.netR = GraphSemanticRectificationModule(
                grid_size=opt.graph_grid_size,
                hidden_dim=opt.graph_hidden_dim,
                he_dim=opt.graph_he_dim,
                num_layers=opt.graph_layers,
                k_neighbors=opt.graph_k,
                pyramid_levels=opt.graph_pyramid_levels,
                conv_type=opt.gsrm_conv,
                pna_degree=opt.gsrm_pna_degree,
            ).to(self.device)

            if self.gsg_stage == 'train_gsrm':
                self.loss_names = ['GSRM', 'HPC', 'IRC']
                self.visual_names = ['real_A', 'real_B']
                self.model_names = ['R']
                self.optimizers = []
                self.optimizer_R = torch.optim.Adam(self.netR.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
                self.optimizers.append(self.optimizer_R)
            elif self.gsg_stage == 'train_generator':
                self.loss_names += ['GSC']
                if opt.lambda_R > 0.0:
                    self.loss_names += ['D_R', 'G_R']
                self.model_names = ['G', 'F', 'D', 'R']
                self.set_requires_grad(self.netR, False)
            else:
                self.loss_names += ['GSC', 'GSRM', 'HPC', 'IRC']
                if opt.lambda_R > 0.0:
                    self.loss_names += ['D_R', 'G_R']
                self.model_names = ['G', 'F', 'D', 'R']
                self.optimizer_R = torch.optim.Adam(self.netR.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
                self.optimizers.append(self.optimizer_R)
        else:
            self.model_names = ['G']

    def setup(self, opt):
        super().setup(opt)
        if self.isTrain and self.gsg_stage == 'train_generator' and not self.loaded_pretrained_R:
            self.load_pretrained_R(opt)
            self.set_requires_grad(self.netR, False)

    def data_dependent_initialize(self, data):
        if self.gsg_stage == 'train_gsrm':
            self.set_input(data)
            return
        if self.gsg_stage == 'train_generator' and not self.loaded_pretrained_R:
            self.load_pretrained_R(self.opt)
            self.set_requires_grad(self.netR, False)
        super().data_dependent_initialize(data)

    def optimize_parameters(self):
        if self.gsg_stage == 'train_gsrm':
            self.optimizer_R.zero_grad()
            self.loss_GSRM = self.compute_R_loss()
            self.loss_GSRM.backward()
            self.optimizer_R.step()
            return

        self.forward()

        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        if self.gsg_stage == 'joint':
            self.optimizer_R.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()
        if self.gsg_stage == 'joint':
            self.optimizer_R.step()

    def set_input(self, input):
        super().set_input(input)
        if all(key in input for key in ['graph_x_he', 'graph_x_ihc', 'graph_centroids', 'graph_mask']):
            self.graph_data = {
                'x_he': input['graph_x_he'].to(self.device),
                'x_ihc': input['graph_x_ihc'].to(self.device),
                'centroids': input['graph_centroids'].to(self.device),
                'coords': input['graph_coords'].to(self.device) if 'graph_coords' in input else input['graph_centroids'].to(self.device),
                'adjacency': input['graph_adj'].to(self.device) if 'graph_adj' in input else None,
                'mask': input['graph_mask'].to(self.device),
            }
            if not hasattr(self, '_printed_graph_mode'):
                print('GSGStain graph mode: cell_graph')
                self._printed_graph_mode = True
        else:
            self.graph_data = None
            if not hasattr(self, '_printed_graph_mode'):
                print('GSGStain graph mode: grid_fallback')
                self._printed_graph_mode = True

    def compute_R_loss(self):
        graph_out = self.netR(self.real_A, self.real_B, graph_data=self.graph_data)
        self.loss_HPC = graph_out['loss_HPC']
        self.loss_IRC = graph_out['loss_IRC']
        self.loss_GSRM = graph_out['loss_GSRM'] * self.opt.lambda_GSRM
        return self.loss_GSRM

    def compute_D_loss(self):
        fake = self.fake_B.detach()
        pred_fake, score_fake = self.discriminate(fake)
        pred_real, score_real = self.discriminate(self.real_B)
        self.pred_real = pred_real

        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        loss_D_real = self.criterionGAN(pred_real, True)
        self.loss_D_real = loss_D_real.mean()
        self.loss_D_R = self.ranking_D_loss(score_real, score_fake) * self.opt.lambda_R
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 + self.loss_D_R
        return self.loss_D

    def compute_G_loss(self):
        fake = self.fake_B
        pred_fake, score_fake = self.discriminate(fake)
        if self.opt.lambda_GAN > 0.0:
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE = 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        graph_out = self.netR(self.real_A, self.real_B, self.fake_B, self.graph_data)
        if self.gsg_stage == 'joint':
            self.loss_HPC = graph_out['loss_HPC']
            self.loss_IRC = graph_out['loss_IRC']
            self.loss_GSRM = graph_out['loss_GSRM'] * self.opt.lambda_GSRM
        else:
            self.loss_GSRM = 0.0
        self.loss_GSC = self.netR_module().graph_semantic_consistency(
            graph_out['fake_semantics'], graph_out['rectified_semantics'], graph_out['node_mask']) * self.opt.lambda_GSC
        self.loss_G_R = self.ranking_G_loss(score_fake) * self.opt.lambda_R

        self.loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_GSC + self.loss_G_R
        if self.gsg_stage == 'joint':
            self.loss_G = self.loss_G + self.loss_GSRM
        return self.loss_G

    def netR_module(self):
        return self.netR.module if isinstance(self.netR, torch.nn.DataParallel) else self.netR

    def load_pretrained_R(self, opt):
        checkpoint_name = opt.gsg_pretrained_R_name or opt.name
        load_dir = os.path.join(opt.checkpoints_dir, checkpoint_name)
        load_path = os.path.join(load_dir, '%s_net_R.pth' % opt.gsg_R_epoch)
        if not os.path.exists(load_path):
            raise FileNotFoundError('pretrained GSRM checkpoint not found: %s' % load_path)
        print('loading pretrained GSRM from %s' % load_path)
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        self.netR.load_state_dict(state_dict)
        self.loaded_pretrained_R = True

    def discriminate(self, image):
        pred = self.netD(image)
        if isinstance(pred, tuple):
            return pred
        score = pred.view(pred.size(0), -1).mean(dim=1)
        return pred, score

    @staticmethod
    def ranking_D_loss(score_real, score_fake):
        return F.relu(1.0 - (score_real.mean() - score_fake.mean())).mean()

    @staticmethod
    def ranking_G_loss(score_fake):
        return -score_fake.mean()
