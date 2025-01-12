import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import sys
from pathlib import Path
from typing import Tuple
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'third_party', 'image-matching-models'))
from matching.im_models.base_matcher import BaseMatcher



class ImagePairDataset(Dataset):
    def __init__(self, cfg, transform=None, save_memory=True):
        """
        Args:
            txt_file (string): Path to the text file with image pairs.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = cfg.image_dir
        self.transform = transform
        self.image_pairs = []
        self.label = []
        self.save_memory = save_memory
        self.cached = {}
        self.image_size = (cfg.image_height, cfg.image_width)
        
        # Read the file and extract image paths
        with open(cfg.pairs_info, 'r') as file:
            lines = file.readlines()
            for line in lines:
                image1_path, image2_path, gt = line.strip().split()
                self.image_pairs.append((image1_path, image2_path))
                self.label.append(int(gt))

    def __len__(self):
        return len(self.image_pairs)

    # wrapper of image-matching-models BaseMatcher's load image
    def load_image(self, path: str | Path, resize: int | Tuple = None, rot_angle: float = 0) -> torch.Tensor:
            return BaseMatcher.load_image(path, resize, rot_angle)
    
    def __getitem__(self, idx):
        
        if idx in self.cached:
            return self.cached[idx]
        else:
            img0_path, img1_path = self.image_pairs[idx]
            label = self.label[idx]
            # img1 = Image.open(os.path.join(self.root_dir, img1_path)).convert('RGB')
            # img2 = Image.open(os.path.join(self.root_dir, img2_path)).convert('RGB')

            img0_path = os.path.join(self.root_dir, img0_path)
            img1_path = os.path.join(self.root_dir, img1_path)

            if self.transform:
                raise NotImplementedError("Transform not implemented")
                # img0 = self.transform(img0)
                # img1 = self.transform(img1)
                
            data = {
                'img0': self.load_image(img0_path,resize=self.image_size),
                'img1': self.load_image(img1_path,resize=self.image_size),
                'label': label
            }
        
            if not self.save_memory:
                self.cached[idx] = data
            
            return data