from dataset import *
from disco_gan_model import DiscoGAN
from model import *
import scipy
from progressbar import ETA, Bar, Percentage, ProgressBar

class DistanceGAN(DiscoGAN):

    def distance(self, A, B):
        return torch.mean(torch.abs(A - B))

    def get_individual_distance_loss(self, A_i, A_j, AB_i, AB_j,
                                     B_i, B_j, BA_i, BA_j):

        distance_in_A = self.distance(A_i, A_j)
        distance_in_AB = self.distance(AB_i, AB_j)
        distance_in_B = self.distance(B_i, B_j)
        distance_in_BA = self.distance(BA_i, BA_j)

        if self.normalize_distances:
            distance_in_A = (distance_in_A - self.expectation_A) / self.std_A
            distance_in_AB = (distance_in_AB - self.expectation_B) / self.std_B
            distance_in_B = (distance_in_B - self.expectation_B) / self.std_B
            distance_in_BA = (distance_in_BA - self.expectation_A) / self.std_A

        return torch.abs(distance_in_A - distance_in_AB), torch.abs(distance_in_B - distance_in_BA)

    def get_self_distances(self, A, B, AB, BA):

        A_half_1, A_half_2 = torch.chunk(A, 2, dim=2)
        B_half_1, B_half_2 = torch.chunk(B, 2, dim=2)
        AB_half_1, AB_half_2 = torch.chunk(AB, 2, dim=2)
        BA_half_1, BA_half_2 = torch.chunk(BA, 2, dim=2)

        l_distance_A, l_distance_B = \
            self.get_individual_distance_loss(A_half_1, A_half_2,
                                              AB_half_1, AB_half_2,
                                              B_half_1, B_half_2,
                                              BA_half_1, BA_half_2)

        return l_distance_A, l_distance_B

    def get_distance_losses(self, A, B, AB, BA):

        As = torch.split(A, 1)
        Bs = torch.split(B, 1)
        ABs = torch.split(AB, 1)
        BAs = torch.split(BA, 1)

        loss_distance_A = 0.0
        loss_distance_B = 0.0
        num_pairs = 0
        min_length = min(len(As), len(Bs))

        for i in xrange(min_length - 1):
            for j in xrange(i + 1, min_length):
                num_pairs += 1
                loss_distance_A_ij, loss_distance_B_ij = \
                    self.get_individual_distance_loss(As[i], As[j],
                                                      ABs[i], ABs[j],
                                                      Bs[i], Bs[j],
                                                      BAs[i], BAs[j])

                loss_distance_A += loss_distance_A_ij
                loss_distance_B += loss_distance_B_ij

        loss_distance_A = loss_distance_A / num_pairs
        loss_distance_B = loss_distance_B / num_pairs

        return loss_distance_A, loss_distance_B

    def get_std(self, num_items, vars, expectation):

        num_pairs = 0
        std_sum = 0.0

        # If self distance computed std for top and bottom half
        if self.args.use_self_distance:
            for i in xrange(num_items):
                var_half_1, var_half_2 = torch.chunk(vars[i], 2, dim=2)
                std_sum += np.square(self.as_np(self.distance(var_half_1, var_half_2)) - expectation)
            return np.sqrt(std_sum / num_items)

        # Otherwise compute std for all pairs of images
        for i in xrange(num_items - 1):
            for j in xrange(i + 1, num_items):
                num_pairs += 1
                std_sum += np.square(self.as_np(self.distance(vars[i], vars[j])) - expectation)

        return np.sqrt(std_sum / num_pairs)

    def get_expectation(self, num_items, vars):

        num_pairs = 0
        distance_sum = 0.0

        # If self distance computed expectation for top and bottom half
        if self.args.use_self_distance:
            for i in xrange(num_items):
                # Split image to top and bottom half
                var_half_1, var_half_2 = torch.chunk(vars[i], 2, dim=2)
                distance_sum += self.as_np(self.distance(var_half_1, var_half_2))
            return distance_sum / num_items

        # Otherwise compute expectation for all pairs of images
        for i in xrange(num_items - 1):
            for j in xrange(i + 1, num_items):
                num_pairs += 1
                distance_sum += self.as_np(self.distance(vars[i], vars[j]))

        return distance_sum / num_pairs

    def set_expectation_and_std(self):

        max_items = self.args.max_items

        data_style_A, data_style_B = shuffle_data(self.data_style_A, self.data_style_B)

        if max_items < len(data_style_A):
            self.A_path = data_style_A[0:max_items]
        else:
            self.A_path = data_style_A

        if max_items < len(data_style_B):
            self.B_path = data_style_B[0:max_items]
        else:
            self.B_path = data_style_B

        dataset_A, dataset_B = self.get_images()

        A_vars = []
        num_vars_A = 0

        for step_data_a, data in enumerate(dataset_A):

            if step_data_a >= max_items:
                break

            A = Variable(torch.FloatTensor(data), volatile=True)
            if self.cuda:
                A = A.cuda()

            A_vars.append(A)
            num_vars_A += 1

        B_vars = []
        num_vars_B = 0
        for step_data_b, data in enumerate(dataset_B):

            if step_data_b >= max_items:
                break

            B = Variable(torch.FloatTensor(data), volatile=True)
            if self.cuda:
                B = B.cuda()

            B_vars.append(B)
            num_vars_B += 1


        self.expectation_A = self.get_expectation(num_vars_A, A_vars)[0].astype(float)
        self.expectation_B = self.get_expectation(num_vars_B, B_vars)[0].astype(float)
        self.std_A = self.get_std(num_vars_A, A_vars, self.expectation_A)[0].astype(float)
        self.std_B = self.get_std(num_vars_B, B_vars, self.expectation_B)[0].astype(float)

        print('Expectation for dataset A: %f' % self.expectation_A)
        print('Expectation for dataset B: %f' % self.expectation_B)
        print('Std for dataset A: %f' % self.std_A)
        print('Std for dataset B: %f' % self.std_B)

    def run(self):

        self.initialize()
        self.normalize_distances = not self.args.unnormalized_distances

        if self.normalize_distances:
            self.set_expectation_and_std()

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

                if self.args.use_reconst_loss:
                    ABA = self.generator_A(AB)
                    BAB = self.generator_B(BA)

                    # Reconstruction Loss
                    self.recon_loss_A = self.recon_criterion(ABA, A)
                    self.recon_loss_B = self.recon_criterion(BAB, B)

                if not self.args.use_self_distance:
                    self.loss_distance_A, self.loss_distance_B = self.get_distance_losses(A, B, AB, BA)
                else:
                    self.loss_distance_A, self.loss_distance_B = self.get_self_distances(A, B, AB, BA)

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

                non_gan_loss_A = self.loss_distance_A
                non_gan_loss_B = self.loss_distance_B

                if self.args.use_reconst_loss:
                    non_gan_loss_A += self.recon_loss_A
                    non_gan_loss_B += self.recon_loss_B

                self.gen_loss_A_total = (self.gen_loss_B * 0.1 + self.fm_loss_B * 0.9) * (1. - rate) + non_gan_loss_A * rate
                self.gen_loss_B_total = (self.gen_loss_A * 0.1 + self.fm_loss_A * 0.9) * (1. - rate) + non_gan_loss_B * rate

                if self.args.model_arch == 'distancegan':
                    self.gen_loss = self.gen_loss_A_total + self.gen_loss_B_total
                    self.dis_loss = self.dis_loss_A + self.dis_loss_B
                elif self.args.model_arch == 'distance_A_to_B':
                    self.gen_loss = self.gen_loss_A_total
                    self.dis_loss = self.dis_loss_B
                elif self.args.model_arch == 'distance_B_to_A':
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
            print "Distance Loss:", self.as_np(self.loss_distance_A.mean()), self.as_np(self.loss_distance_B.mean())
            if self.args.use_reconst_loss:
                print "RECON Loss:", self.as_np(self.recon_loss_A.mean()), self.as_np(self.recon_loss_B.mean())

        if self.iters % self.args.image_save_interval == 0:
            AB = self.generator_B(self.test_A)
            BA = self.generator_A(self.test_B)

            if self.args.use_reconst_loss:
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
                AB_val = AB[im_idx].cpu().data.numpy().transpose(1, 2, 0) * 255.

                filename_prefix = os.path.join(subdir_path, str(im_idx))
                scipy.misc.imsave(filename_prefix + '.A.jpg', A_val.astype(np.uint8)[:, :, ::-1])
                scipy.misc.imsave(filename_prefix + '.B.jpg', B_val.astype(np.uint8)[:, :, ::-1])
                scipy.misc.imsave(filename_prefix + '.BA.jpg', BA_val.astype(np.uint8)[:, :, ::-1])
                scipy.misc.imsave(filename_prefix + '.AB.jpg', AB_val.astype(np.uint8)[:, :, ::-1])

                if self.args.use_reconst_loss:
                    ABA_val = ABA[im_idx].cpu().data.numpy().transpose(1, 2, 0) * 255.
                    BAB_val = BAB[im_idx].cpu().data.numpy().transpose(1, 2, 0) * 255.
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
    model = DistanceGAN()
    model.run()