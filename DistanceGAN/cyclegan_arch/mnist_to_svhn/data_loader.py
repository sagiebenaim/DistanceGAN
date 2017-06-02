import torch
from torchvision import datasets
from torchvision import transforms

def get_loader(config):
    """Builds and returns Dataloader for MNIST and SVHN dataset."""
    
    transform = transforms.Compose([
                    transforms.Scale(config.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    svhn = datasets.SVHN(root=config.svhn_path, download=True, transform=transform, split='train')
    mnist = datasets.MNIST(root=config.mnist_path, download=True, transform=transform, train=True)

    svhn_test = datasets.SVHN(root=config.svhn_path, download=True, transform=transform, split='test')
    mnist_test = datasets.MNIST(root=config.mnist_path, download=True, transform=transform, train=False)

    svhn_loader = torch.utils.data.DataLoader(dataset=svhn,
                                              batch_size=config.batch_size,
                                              shuffle=True,
                                              num_workers=config.num_workers)

    mnist_loader = torch.utils.data.DataLoader(dataset=mnist,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.num_workers)


    svhn_test_loader = torch.utils.data.DataLoader(dataset=svhn_test,
                                              batch_size=config.batch_size,
                                              shuffle=False,
                                              num_workers=config.num_workers)

    mnist_test_loader = torch.utils.data.DataLoader(dataset=mnist_test,
                                               batch_size=config.batch_size,
                                               shuffle=False,
                                               num_workers=config.num_workers)

    return svhn_loader, mnist_loader, svhn_test_loader, mnist_test_loader