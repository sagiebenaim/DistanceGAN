from collections import OrderedDict

import numpy as np
import torch
from torch.autograd import Variable

from util import util as util
from .cycle_gan_model import CycleGANModel


class DistanceGANModel(CycleGANModel):
    def __init__(self, dataset):
        super(CycleGANModel, self).__init__()
        self.dataset = dataset

    def name(self):
        return 'DistanceGANModel'

    def initialize(self, opt):

        CycleGANModel.initialize(self, opt)

        self.use_self_distance = opt.use_self_distance
        self.normalize_distances = not opt.unnormalized_distances

        if self.isTrain and self.normalize_distances:
            self.set_expectation_and_std()


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

    def get_self_distances(self):

        A_half_1, A_half_2 = torch.chunk(self.real_A, 2, dim=2)
        B_half_1, B_half_2 = torch.chunk(self.real_B, 2, dim=2)
        AB_half_1, AB_half_2 = torch.chunk(self.fake_B, 2, dim=2)
        BA_half_1, BA_half_2 = torch.chunk(self.fake_A, 2, dim=2)

        l_distance_A, l_distance_B = \
            self.get_individual_distance_loss(A_half_1, A_half_2,
                                              AB_half_1, AB_half_2,
                                              B_half_1, B_half_2,
                                              BA_half_1, BA_half_2)

        return l_distance_A, l_distance_B

    def get_distance_losses(self):

        As = torch.split(self.real_A, 1)
        Bs = torch.split(self.real_B, 1)
        ABs = torch.split(self.fake_B, 1)
        BAs = torch.split(self.fake_A, 1)

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
        if self.use_self_distance:
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
        if self.use_self_distance:
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

        max_items = self.opt.max_items

        A_vars = []
        B_vars = []
        num_vars_A = 0
        num_vars_B = 0
        for i, data in enumerate(self.dataset):

            if (self.dataset.stop_A and self.dataset.stop_B) or i >= max_items:
                break

            if not self.dataset.stop_A:
                A = Variable(data['A'], volatile=True)

                # If reached end of dataset, variable sizes may be different
                # We check this and not take these variables into account
                # when calculating expectation and std
                if A.size()[0] != self.opt.batchSize:
                    continue

                A_vars.append(A)
                num_vars_A += 1

            if not self.dataset.stop_B:
                B = Variable(data['B'], volatile=True)

                if B.size()[0] != self.opt.batchSize:
                    continue

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

    def backward_G(self):

        # D_A(G_A(A))
        self.fake_B = self.netG_A.forward(self.real_A)
        pred_fake = self.netD_A.forward(self.fake_B)
        self.loss_G_A = self.criterionGAN(pred_fake, True)
        # D_B(G_B(B))
        self.fake_A = self.netG_B.forward(self.real_B)
        pred_fake = self.netD_B.forward(self.fake_A)
        self.loss_G_B = self.criterionGAN(pred_fake, True)

        if self.use_self_distance:
            self.loss_distance_A, self.loss_distance_B = self.get_self_distances()
        else:
            self.loss_distance_A, self.loss_distance_B = self.get_distance_losses()

        self.loss_distance_A *= self.opt.lambda_distance_A
        self.loss_distance_B *= self.opt.lambda_distance_B

        if self.A_to_B:
            self.loss_G = self.loss_G_A + self.loss_distance_A
        elif self.B_to_A:
            self.loss_G = self.loss_G_B + self.loss_distance_B
        else:
            self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_distance_A + self.loss_distance_B

        if self.opt.use_cycle_loss:
            # Forward cycle loss
            self.rec_A = self.netG_B.forward(self.fake_B)
            self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * self.opt.lambda_A
            # Backward cycle loss
            self.rec_B = self.netG_A.forward(self.fake_A)
            self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * self.opt.lambda_B

            self.loss_G += self.loss_cycle_A + self.loss_cycle_B

        self.loss_G.backward()

    def get_current_errors(self):
        if self.opt.use_cycle_loss:
            return CycleGANModel.get_current_errors(self)

        D_A = self.loss_D_A.data[0]
        G_A = self.loss_G_A.data[0]
        Dist_A = self.loss_distance_A.data[0]

        D_B = self.loss_D_B.data[0]
        G_B = self.loss_G_B.data[0]
        Dist_B = self.loss_distance_B.data[0]

        if self.A_to_B:
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Dist_A', Dist_A)])
        elif self.B_to_A:
            return OrderedDict([('D_B', D_B), ('G_B', G_B), ('Dist_B', Dist_B)])

        return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Dist_A', Dist_A),
                            ('D_B', D_B), ('G_B', G_B), ('Dist_B', Dist_B)])

    def get_current_visuals(self):
        if self.opt.use_cycle_loss:
            return CycleGANModel.get_current_visuals(self)

        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)

        real_B = util.tensor2im(self.real_B.data)
        fake_A = util.tensor2im(self.fake_A.data)

        if self.A_to_B:
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B)])
        elif self.B_to_A:
            return OrderedDict([('real_B', real_B), ('fake_A', fake_A)])

        return OrderedDict([('real_A', real_A), ('fake_B', fake_B),
                            ('real_B', real_B), ('fake_A', fake_A)])

    def test(self):
        if self.opt.use_cycle_loss:
            return CycleGANModel.test(self)

        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG_A.forward(self.real_A)

        self.real_B = Variable(self.input_B, volatile=True)
        self.fake_A = self.netG_B.forward(self.real_B)
