import torchvision.transforms as T

_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_transforms(train: bool) -> T.Compose:
    if train:
        return T.Compose(
            [
                T.Resize(224),
                T.RandomCrop(224, padding=16, padding_mode="reflect"),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
            ]
        )
    return T.Compose(
        [
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
        ]
    )
