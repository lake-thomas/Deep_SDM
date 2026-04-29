import os
import torch
import rasterio
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF
import random


class RandomAugment4Band:
    def __init__(self, rotation_degrees=30, hflip_prob=0.5, vflip_prob=0.5):
        self.rotation_degrees = rotation_degrees
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob

    def __call__(self, img):
        angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
        img = TF.rotate(img, angle)

        if random.random() < self.hflip_prob:
            img = TF.hflip(img)

        if random.random() < self.vflip_prob:
            img = TF.vflip(img)

        return img


def load_naip_tensor(img_path):
    with rasterio.open(img_path) as src:
        img = src.read()  # (4, H, W)
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img)  # (4, H, W)


def show_naip_tensor(tensor_img, title=None):
    # Plot RGB only (bands 1, 2, 3)
    np_img = tensor_img[:3].numpy().transpose(1, 2, 0)
    np_img = np.clip(np_img, 0, 1)
    plt.imshow(np_img)
    if title:
        plt.title(title)
    plt.axis('off')


def main():
    
    img_path = r'D:\Ailanthus_NAIP_Classification\NAIP_NC_4Band\m_3608161_sw_17_060_20221004_20221207.tif'  # replace with your image path
    assert os.path.exists(img_path), f"File not found: {img_path}"

    img_tensor = load_naip_tensor(img_path)
    transform = RandomAugment4Band(rotation_degrees=45, hflip_prob=1.0, vflip_prob=1.0)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    show_naip_tensor(img_tensor, title="Original")

    aug_img = transform(img_tensor)
    plt.subplot(1, 2, 2)
    show_naip_tensor(aug_img, title="Transformed")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
