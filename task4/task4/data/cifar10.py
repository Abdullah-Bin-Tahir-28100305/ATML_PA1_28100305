
from torchvision import transforms
from torchvision.datasets import CIFAR10


def build_transforms(cfg, train: bool, randaugment: bool = False):
   
    size = cfg["image"]["size"]
    pad = cfg["image"]["crop_padding"]
    mean = cfg["image"]["norm_mean"]
    std = cfg["image"]["norm_std"]

    if train:
        ops = [
            transforms.RandomCrop(size, padding=pad),   # random 32x32 crop, 4px pad
            transforms.RandomHorizontalFlip(),          # 50% flip
        ]
        if randaugment:
            
            ra = cfg["method"]["randaugment"]
            ops.append(transforms.RandAugment(num_ops=ra["num_ops"],
                                              magnitude=ra["magnitude"]))
        ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
        return transforms.Compose(ops)
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def load_cifar10_train(cfg, train_transform: bool, randaugment: bool = False):
   
    tfm = build_transforms(cfg, train=train_transform, randaugment=randaugment)
    return CIFAR10(cfg["known"]["root"], train=True, download=True, transform=tfm)


def load_cifar10_test(cfg):
    tfm = build_transforms(cfg, train=False)
    return CIFAR10(cfg["known"]["root"], train=False, download=True, transform=tfm)


CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
