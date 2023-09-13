#Augmentation 코드

from torchvision import transforms
from torchvision.transforms import RandomRotation, RandomResizedCrop, ColorJitter, RandomPerspective


train_transform = transforms.Compose([
    transforms.RandomApply([RandomRotation(degrees=15)], p=0.5),
    transforms.RandomApply([ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)], p=0.5),
    # RandomHorizontalFlip(p=0.5),
    # RandomVerticalFlip(p=0.5),
    # RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333)),
    # RandomPerspective(0.3),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((512,512), antialias=True),
    transforms.ToTensor()
])