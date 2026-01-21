"""Image preprocessing for data preparation."""

import numpy as np
from typing import Optional, Tuple
from PIL import Image
import torchvision.transforms as transforms


class ImagePreprocessor:
    """Preprocessor for image data."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        self.transforms = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Preprocess single image."""
        return self.transforms(image)
    
    def batch_preprocess(self, images: list) -> np.ndarray:
        """Preprocess batch of images."""
        processed = [self.preprocess(img) for img in images]
        return np.stack(processed)
    
    def augment(self, image: Image.Image) -> Image.Image:
        """Apply data augmentation to image."""
        augment_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])
        return augment_transforms(image)
