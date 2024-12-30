import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class ImagePairDataset(Dataset):
    def __init__(self, cfg, transform=None):
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
        
        # Read the file and extract image paths
        with open(cfg.pairs_info, 'r') as file:
            lines = file.readlines()
            for line in lines:
                image1_path, image2_path, gt = line.strip().split()
                self.image_pairs.append((image1_path, image2_path))
                self.label.append(int(gt))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img1_path, img2_path = self.image_pairs[idx]
        label = self.label[idx]
        # img1 = Image.open(os.path.join(self.root_dir, img1_path)).convert('RGB')
        # img2 = Image.open(os.path.join(self.root_dir, img2_path)).convert('RGB')

        img1_path = os.path.join(self.root_dir, img1_path)
        img2_path = os.path.join(self.root_dir, img2_path)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1_path, img2_path, label
