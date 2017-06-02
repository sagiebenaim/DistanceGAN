# horse to zebra
python train.py --dataroot ./datasets/horse2zebra --name horse2zebra_distancegan --model distance_gan --pool_size 50

# mnist to svhn
python ./cyclegan_based_models/mnist_to_svhn/main.py --use_distance_loss=True --use_reconst_loss=False

# edges to shoes
python ./discogan_based_models/distance_gan_model.py --task_name=edges2shoes --model_arch=distancegan --num_layers=3

# shoes to handbags
python ./discogan_based_models/distance_gan_model.py --task_name=handbags2shoes --model_arch=distancegan
