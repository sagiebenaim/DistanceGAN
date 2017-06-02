
def create_model(opt, dataset=None):

    print(opt.model)

    if opt.model == 'gan':
        from .gan_model import GANModel
        model = GANModel()
    elif opt.model == 'cycle_gan':
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == "distance_gan":
        from .distance_gan_model import DistanceGANModel
        model = DistanceGANModel(dataset)
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)

    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
