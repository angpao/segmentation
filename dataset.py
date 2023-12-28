import os

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Image 0 = black , 1 = red , 2 = green


class KittiSegmentMini:
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)  # how many images in folder and image name

    def __len__(self):  # lenght of image
        return len(self.images)

    def __getitem__(self, index):
        # get image path
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        # load image
        image = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32)
        mask = np.array(
            Image.open(mask_path).convert("L"), dtype=np.float32
        )  # L = mode black and white

        if self.transform is not None:
            augmentations = self.transform(
                image=image, mask=mask
            )  # rotate both image and mask
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


if __name__ == "__main__":
    image_dir = "data/train/image/"
    mask_dir = "data/train/mask"

    train_transform = A.Compose([A.Resize(height=94, width=301), ToTensorV2()])
    train_data_object = KittiSegmentMini(image_dir, mask_dir)
    train_loader = DataLoader(train_data_object, batch_size=4)

    x, y = next(iter(train_loader))
    print(x.shape)
