
def CreateDataLoader(opt):
    if opt.align_data > 0:
        from cyclegan_arch.data.aligned_data_loader import AlignedDataLoader
        data_loader = AlignedDataLoader()
    else:
        from unaligned_data_loader import UnalignedDataLoader
        data_loader = UnalignedDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader
