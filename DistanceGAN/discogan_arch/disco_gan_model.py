from itertools import chain

import scipy
import torch.optim as optim
from progressbar import ETA, Bar, Percentage, ProgressBar

from dataset import *
from discogan_arch_options.options import Options
from model import *


class DiscoGAN(object):

    def as_np(self, data):
        return data.cpu().data.numpy()

    def get_data(self):
        # celebA / edges2shoes / edges2handbags / ...
        if self.args.task_name == 'facescrub':
            data_A, data_B = get_facescrub_files(test=False, n_test=self.args.n_test)
            test_A, test_B = get_facescrub_files(test=True, n_test=self.args.n_test)

        elif self.args.task_name == 'celebA':
            data_A, data_B = get_celebA_files(style_A=self.args.style_A, style_B=self.args.style_B, constraint=self.args.constraint,
                                              constraint_type=self.args.constraint_type, test=False, n_test=self.args.n_test)
            test_A, test_B = get_celebA_files(style_A=self.args.style_A, style_B=self.args.style_B, constraint=self.args.constraint,
                                              constraint_type=self.args.constraint_type, test=True, n_test=self.args.n_test)

        elif self.args.task_name == 'edges2shoes':
            data_A, data_B = get_edge2photo_files(item='edges2shoes', test=False)
            test_A, test_B = get_edge2photo_files(item='edges2shoes', test=True)

        elif self.args.task_name == 'edges2handbags':
            data_A, data_B = get_edge2photo_files(item='edges2handbags', test=False)
            test_A, test_B = get_edge2photo_files(item='edges2handbags', test=True)

        elif self.args.task_name == 'handbags2shoes':
            data_A_1, data_A_2 = get_edge2photo_files(item='edges2handbags', test=False)
            test_A_1, test_A_2 = get_edge2photo_files(item='edges2handbags', test=True)

            data_A = np.hstack([data_A_1, data_A_2])
            test_A = np.hstack([test_A_1, test_A_2])

            data_B_1, data_B_2 = get_edge2photo_files(item='edges2shoes', test=False)
            test_B_1, test_B_2 = get_edge2photo_files(item='edges2shoes', test=True)

            data_B = np.hstack([data_B_1, data_B_2])
            test_B = np.hstack([test_B_1, test_B_2])

        return data_A, data_B, test_A, test_B

    def get_fm_loss(self, real_feats, fake_feats):
        losses = 0
        for real_feat, fake_feat in zip(real_feats, fake_feats):
            l2 = (real_feat.mean(0) - fake_feat.mean(0)) * (real_feat.mean(0) - fake_feat.mean(0))
            loss = self.feat_criterion(l2, Variable(torch.ones(l2.size())).cuda())
            losses += loss

        return losses

    def get_gan_loss(self, dis_real, dis_fake):
        labels_dis_real = Variable(torch.ones([dis_real.size()[0], 1]))
        labels_dis_fake = Variable(torch.zeros([dis_fake.size()[0], 1]))
        labels_gen = Variable(torch.ones([dis_fake.size()[0], 1]))

        if self.cuda:
            labels_dis_real = labels_dis_real.cuda()
            labels_dis_fake = labels_dis_fake.cuda()
            labels_gen = labels_gen.cuda()

        dis_loss = self.gan_criterion(dis_real, labels_dis_real) * 0.5 + self.gan_criterion(dis_fake, labels_dis_fake) * 0.5
        gen_loss = self.gan_criterion(dis_fake, labels_gen)

        return dis_loss, gen_loss

    def get_test_images(self):

        if self.args.task_name.startswith('edges2'):
            self.test_A = read_images(self.test_style_A, 'A', self.args.image_size)
            self.test_B = read_images(self.test_style_B, 'B', self.args.image_size)
        elif self.args.task_name == 'handbags2shoes' or self.args.task_name == 'shoes2handbags':
            self.test_A = read_images(self.test_style_A, 'B', self.args.image_size)
            self.test_B = read_images(self.test_style_B, 'B', self.args.image_size)
        else:
            self.test_A = read_images(self.test_style_A, None, self.args.image_size)
            self.test_B = read_images(self.test_style_B, None, self.args.image_size)

    def get_images(self):

        if self.args.task_name.startswith('edges2'):
            A = read_images(self.A_path, 'A', self.args.image_size)
            B = read_images(self.B_path, 'B', self.args.image_size)
        elif self.args.task_name == 'handbags2shoes' or self.args.task_name == 'shoes2handbags':
            A = read_images(self.A_path, 'B', self.args.image_size)
            B = read_images(self.B_path, 'B', self.args.image_size)
        else:
            A = read_images(self.A_path, None, self.args.image_size)
            B = read_images(self.B_path, None, self.args.image_size)

        return A, B

    def __init__(self):
        options = Options()
        options.initialize()
        self.args = options.parser.parse_args()

    def initialize(self):

        self.cuda = self.args.cuda
        if self.cuda == 'true':
            self.cuda = True
        else:
            self.cuda = False

        self.result_path = os.path.join(self.args.result_path, self.args.task_name)
        if self.args.style_A:
            self.result_path = os.path.join(self.result_path, self.args.style_A)
        self.result_path = os.path.join(self.result_path, self.args.model_arch)

        self.model_path = os.path.join(self.args.model_path, self.args.task_name)
        if self.args.style_A:
            self.model_path = os.path.join(self.model_path, self.args.style_A)
        self.model_path = os.path.join(self.model_path, self.args.model_arch)

        self.data_style_A, self.data_style_B, self.test_style_A, self.test_style_B = self.get_data()

        self.get_test_images()
        self.test_A = Variable(torch.FloatTensor(self.test_A), volatile=True)
        self.test_B = Variable(torch.FloatTensor(self.test_B), volatile=True)

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.generator_A = Generator(num_layers=self.args.num_layers)
        self.generator_B = Generator(num_layers=self.args.num_layers)
        self.discriminator_A = Discriminator()
        self.discriminator_B = Discriminator()

        if self.cuda:
            self.test_A = self.test_A.cuda()
            self.test_B = self.test_B.cuda()
            self.generator_A = self.generator_A.cuda()
            self.generator_B = self.generator_B.cuda()
            self.discriminator_A = self.discriminator_A.cuda()
            self.discriminator_B = self.discriminator_B.cuda()

        data_size = min(len(self.data_style_A), len(self.data_style_B))
        self.n_batches = (data_size // self.args.batch_size)

        self.recon_criterion = nn.MSELoss()
        self.gan_criterion = nn.BCELoss()
        self.feat_criterion = nn.HingeEmbeddingLoss()

        gen_params = chain(self.generator_A.parameters(), self.generator_B.parameters())
        dis_params = chain(self.discriminator_A.parameters(), self.discriminator_B.parameters())

        self.optim_gen = optim.Adam(gen_params, lr=self.args.learning_rate, betas=(0.5, 0.999), weight_decay=0.00001)
        self.optim_dis = optim.Adam(dis_params, lr=self.args.learning_rate, betas=(0.5, 0.999), weight_decay=0.00001)

    def run(self):

        self.initialize()
        self.iters = 0

        for epoch in range(self.args.epoch_size):
            data_style_A, data_style_B = shuffle_data(self.data_style_A, self.data_style_B)

            widgets = ['epoch #%d|' % epoch, Percentage(), Bar(), ETA()]
            pbar = ProgressBar(maxval=self.n_batches, widgets=widgets)
            pbar.start()

            for i in range(self.n_batches):

                pbar.update(i)

                self.generator_A.zero_grad()
                self.generator_B.zero_grad()
                self.discriminator_A.zero_grad()
                self.discriminator_B.zero_grad()

                self.A_path = data_style_A[i * self.args.batch_size: (i + 1) * self.args.batch_size]
                self.B_path = data_style_B[i * self.args.batch_size: (i + 1) * self.args.batch_size]

                A, B = self.get_images()
                A = Variable(torch.FloatTensor(A))
                B = Variable(torch.FloatTensor(B))

                if self.cuda:
                    A = A.cuda()
                    B = B.cuda()

                AB = self.generator_B(A)
                BA = self.generator_A(B)

                ABA = self.generator_A(AB)
                BAB = self.generator_B(BA)

                # Reconstruction Loss
                self.recon_loss_A = self.recon_criterion(ABA, A)
                self.recon_loss_B = self.recon_criterion(BAB, B)

                # Real/Fake GAN Loss (A)
                A_dis_real, A_feats_real = self.discriminator_A(A)
                A_dis_fake, A_feats_fake = self.discriminator_A(BA)

                self.dis_loss_A, self.gen_loss_A = self.get_gan_loss(A_dis_real, A_dis_fake)
                self.fm_loss_A = self.get_fm_loss(A_feats_real, A_feats_fake)

                # Real/Fake GAN Loss (B)
                B_dis_real, B_feats_real = self.discriminator_B(B)
                B_dis_fake, B_feats_fake = self.discriminator_B(AB)

                self.dis_loss_B, self.gen_loss_B = self.get_gan_loss(B_dis_real, B_dis_fake)
                self.fm_loss_B = self.get_fm_loss(B_feats_real, B_feats_fake)

                # Total Loss
                if self.iters < self.args.gan_curriculum:
                    rate = self.args.starting_rate
                else:
                    rate = self.args.default_rate

                self.gen_loss_A_total = (self.gen_loss_B * 0.1 + self.fm_loss_B * 0.9) * (1. - rate) + self.recon_loss_A * rate
                self.gen_loss_B_total = (self.gen_loss_A * 0.1 + self.fm_loss_A * 0.9) * (1. - rate) + self.recon_loss_B * rate

                if self.args.model_arch == 'discogan':
                    self.gen_loss = self.gen_loss_A_total + self.gen_loss_B_total
                    self.dis_loss = self.dis_loss_A + self.dis_loss_B
                elif self.args.model_arch == 'recongan':
                    self.gen_loss = self.gen_loss_A_total
                    self.dis_loss = self.dis_loss_B
                elif self.args.model_arch == 'recongan_reverse':
                    self.gen_loss = self.gen_loss_B_total
                    self.dis_loss = self.dis_loss_A
                elif self.args.model_arch == 'gan':
                    self.gen_loss = (self.gen_loss_B * 0.1 + self.fm_loss_B * 0.9)
                    self.dis_loss = self.dis_loss_B

                self.finish_iteration()
                self.iters += 1

    def finish_iteration(self):

        if self.iters % self.args.update_interval == 0:
            self.dis_loss.backward()
            self.optim_dis.step()
        else:
            self.gen_loss.backward()
            self.optim_gen.step()

        if self.iters % self.args.log_interval == 0:
            print "---------------------"
            print "GEN Loss:", self.as_np(self.gen_loss_A.mean()), self.as_np(self.gen_loss_B.mean())
            print "Feature Matching Loss:", self.as_np(self.fm_loss_A.mean()), self.as_np(self.fm_loss_B.mean())
            print "DIS Loss:", self.as_np(self.dis_loss_A.mean()), self.as_np(self.dis_loss_B.mean())
            print "RECON Loss:", self.as_np(self.recon_loss_A.mean()), self.as_np(self.recon_loss_B.mean())

        if self.iters % self.args.image_save_interval == 0:
            AB = self.generator_B(self.test_A)
            BA = self.generator_A(self.test_B)
            ABA = self.generator_A(AB)
            BAB = self.generator_B(BA)

            n_testset = min(self.test_A.size()[0], self.test_B.size()[0])
            subdir_path = os.path.join(self.result_path, str(self.iters / self.args.image_save_interval))

            if not os.path.exists(subdir_path):
                os.makedirs(subdir_path)

            for im_idx in range(n_testset):
                A_val = self.test_A[im_idx].cpu().data.numpy().transpose(1, 2, 0) * 255.
                B_val = self.test_B[im_idx].cpu().data.numpy().transpose(1, 2, 0) * 255.
                BA_val = BA[im_idx].cpu().data.numpy().transpose(1, 2, 0) * 255.
                ABA_val = ABA[im_idx].cpu().data.numpy().transpose(1, 2, 0) * 255.
                AB_val = AB[im_idx].cpu().data.numpy().transpose(1, 2, 0) * 255.
                BAB_val = BAB[im_idx].cpu().data.numpy().transpose(1, 2, 0) * 255.

                filename_prefix = os.path.join(subdir_path, str(im_idx))
                scipy.misc.imsave(filename_prefix + '.A.jpg', A_val.astype(np.uint8)[:, :, ::-1])
                scipy.misc.imsave(filename_prefix + '.B.jpg', B_val.astype(np.uint8)[:, :, ::-1])
                scipy.misc.imsave(filename_prefix + '.BA.jpg', BA_val.astype(np.uint8)[:, :, ::-1])
                scipy.misc.imsave(filename_prefix + '.AB.jpg', AB_val.astype(np.uint8)[:, :, ::-1])
                scipy.misc.imsave(filename_prefix + '.ABA.jpg', ABA_val.astype(np.uint8)[:, :, ::-1])
                scipy.misc.imsave(filename_prefix + '.BAB.jpg', BAB_val.astype(np.uint8)[:, :, ::-1])


        if self.iters % self.args.model_save_interval == 0:
            torch.save(self.generator_A,
                       os.path.join(self.model_path, 'model_gen_A-' + str(self.iters / self.args.model_save_interval)))
            torch.save(self.generator_B,
                       os.path.join(self.model_path, 'model_gen_B-' + str(self.iters / self.args.model_save_interval)))
            torch.save(self.discriminator_A,
                       os.path.join(self.model_path, 'model_dis_A-' + str(self.iters / self.args.model_save_interval)))
            torch.save(self.discriminator_B,
                       os.path.join(self.model_path, 'model_dis_B-' + str(self.iters / self.args.model_save_interval)))

if __name__ == "__main__":
    model = DiscoGAN()
    model.run()