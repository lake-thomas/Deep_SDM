# Image Transformations for NAIP Host Model

import torchvision.transforms.functional as TF
import random

class RandomAugment4Band:
    """Randomly apply rotation, horizontal flip, and vertical flip to a 4-band tensor."""
    def __init__(self, rotation_degrees=30, hflip_prob=0.5, vflip_prob=0.5):
        self.rotation_degrees = rotation_degrees
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob

    def __call__(self, img):
        # Random Rotation
        angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
        img = TF.rotate(img, angle)

        # Random Horizontal Flip
        if random.random() < self.hflip_prob:
            img = TF.hflip(img)

        # Random Vertical Flip
        if random.random() < self.vflip_prob:
            img = TF.vflip(img)

        return img
