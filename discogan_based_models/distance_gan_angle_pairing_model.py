from dataset import *
from discogan_based_options.options import AnglePairingOptions
from distance_gan_model import DistanceGAN
from model import *


class DistanceGANAnglePairing(DistanceGAN):

    def get_data(self):
        if self.args.task_name == 'car2car':
            data_A = get_cars(test=False, ver=180, half='first', image_size=self.args.image_size)
            data_B = get_cars(test=False, ver=180, half='last', image_size=self.args.image_size)

            test_A = test_B = get_cars(test=True, ver=180, image_size=self.args.image_size)

        elif self.args.task_name == 'face2face':
            data_A = get_faces_3d(test=False, half='first')
            data_B = get_faces_3d(test=False, half='last')

            test_A = test_B = get_faces_3d(test=True)

        elif self.args.task_name == 'chair2chair':
            data_A = get_chairs(test=False, half='first', ver=360)
            data_B = get_chairs(test=False, half='last', ver=360)

            test_A = test_B = get_chairs(test=True, ver=360)

        elif self.args.task_name == 'chair2car':
            data_A = get_chairs(test=False, half=None, ver=180)
            data_B = get_cars(test=False, half=None, ver=180)

            test_A = get_chairs(test=True, ver=180)
            test_B = get_cars(test=True, ver=180)

        elif self.args.task_name == 'chair2face':
            data_A = get_chairs(test=False, half=None, ver=180)
            data_B = get_faces_3d(test=False, half=None)

            test_A = get_chairs(test=True, ver=180)
            test_B = get_faces_3d(test=True)

        elif self.args.task_name == 'car2face':
            data_A = get_cars(test=False, ver=180, half=None)
            data_B = get_faces_3d(test=False, half=None)

            test_A = get_cars(test=True, ver=180)
            test_B = get_faces_3d(test=True)

        return data_A, data_B, test_A, test_B

    def get_fm_loss(self, real_feats, fake_feats):
        losses = 0
        for real_feat, fake_feat in zip(real_feats[1:], fake_feats[1:]):
            l2 = (real_feat.mean(0) - fake_feat.mean(0)) * (real_feat.mean(0) - fake_feat.mean(0))
            loss = self.feat_criterion(l2, Variable(torch.ones(l2.size())).cuda())
            losses += loss

        return losses

    def get_test_images(self):
        if self.args.task_name.startswith('car') and self.args.task_name.endswith('car'):
            self.test_A = self.test_style_A
            self.test_B = self.test_style_B

        if self.args.task_name.startswith('car') and not self.args.task_name.endswith('car'):
            self.test_A = self.test_style_A
            self.test_B = read_images(self.test_style_B, None, self.args.image_size)

        if not self.args.task_name.startswith('car') and not self.args.task_name.endswith('car'):
            self.test_A = read_images(self.test_style_A, None, self.args.image_size)
            self.test_B = read_images(self.test_style_B, None, self.args.image_size)

    def get_images(self):
        if self.args.task_name.startswith('car') and self.args.task_name.endswith('car'):
            A = self.A_path
            B = self.B_path

        if self.args.task_name.startswith('car') and not self.args.task_name.endswith('car'):
            A = self.A_path
            B = read_images(self.B_path, None, self.args.image_size)

        if not self.args.task_name.startswith('car') and not self.args.task_name.endswith('car'):
            A = read_images(self.A_path, None, self.args.image_size)
            B = read_images(self.B_path, None, self.args.image_size)

        return A, B


    def __init__(self):
        options = AnglePairingOptions()
        options.initialize()
        self.args = options.parser.parse_args()


if __name__ == "__main__":
    model = DistanceGANAnglePairing()
    model.run()