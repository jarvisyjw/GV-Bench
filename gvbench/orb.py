import cv2
import torch
from types import SimpleNamespace
import glob
from pathlib import Path
from . import logger



class ImageDataset(torch.utils.data.Dataset):
    default_conf = {
        'globs': ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG'],
        'grayscale': False,
        'resize_max': None,
        'resize_force': False,
        'interpolation': 'cv2_area',  # pil_linear is more accurate but slower
    }

    def __init__(self, root, conf, paths=None):
        self.conf = conf = SimpleNamespace(**{**self.default_conf, **conf})
        self.root = root

        if paths is None:
            paths = []
            for g in conf.globs:
                paths += glob.glob(
                    (Path(root) / '**' / g).as_posix(), recursive=True)
            if len(paths) == 0:
                raise ValueError(f'Could not find any image in root: {root}.')
            paths = sorted(set(paths))
            self.names = [Path(p).relative_to(root).as_posix() for p in paths]
            logger.info(f'Found {len(self.names)} images in root {root}.')
        else:
            if isinstance(paths, (Path, str)):
                self.names = parse_image_lists(paths)
            elif isinstance(paths, collections.Iterable):
                self.names = [p.as_posix() if isinstance(p, Path) else p
                              for p in paths]
            else:
                raise ValueError(f'Unknown format for path argument {paths}.')

            for name in self.names:
                if not (root / name).exists():
                    raise ValueError(
                        f'Image {name} does not exists in root: {root}.')

    def __getitem__(self, idx):
        name = self.names[idx]
        image = read_image(self.root / name, self.conf.grayscale)
        image = image.astype(np.float32)
        size = image.shape[:2][::-1]

        if self.conf.resize_max and (self.conf.resize_force
                                     or max(size) > self.conf.resize_max):
            scale = self.conf.resize_max / max(size)
            size_new = tuple(int(round(x*scale)) for x in size)
            image = resize_image(image, size_new, self.conf.interpolation)

        if self.conf.grayscale:
            image = image[None]
        else:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = image / 255.

        data = {
            'image': image,
            'original_size': np.array(size),
        }
        return data

    def __len__(self):
        return len(self.names)


orb = cv2.ORB_create(nfeatures=1000, scoreType=cv2.ORB_FAST_SCORE)

queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw,None) 
trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw,None) 



from ..utils.base_model import BaseModel
import cv2
import torch

EPS = 1e-6


class DoG(BaseModel):
    default_conf = {
        'options': {
            'first_octave': 0,
            'peak_threshold': 0.01,
        },
        'max_keypoints': -1
    }
    required_inputs = ['image']
    detection_noise = 1.0
    max_batch_size = 1024

    def _init(self, conf):
        self.orb =  cv2.ORB_create() 

    def _forward(self, data):
        image = data['image']
        image_np = image.cpu().numpy()[0, 0]
        assert image.shape[1] == 1
        assert image_np.min() >= -EPS and image_np.max() <= 1 + EPS
        
        keypoints, descriptors = self.orb.detectAndCompute(image_np)
        scales = keypoints[:, 2]
      #   oris = np.rad2deg(keypoints[:, 3])

        if self.conf['descriptor'] in ['sift', 'rootsift']:
            # We still renormalize because COLMAP does not normalize well,
            # maybe due to numerical errors
            if self.conf['descriptor'] == 'rootsift':
                descriptors = sift_to_rootsift(descriptors)
            descriptors = torch.from_numpy(descriptors)
        elif self.conf['descriptor'] in ('sosnet', 'hardnet'):
            center = keypoints[:, :2] + 0.5
            laf_scale = scales * self.conf['mr_size'] / 2
            laf_ori = -oris
            lafs = laf_from_center_scale_ori(
                torch.from_numpy(center)[None],
                torch.from_numpy(laf_scale)[None, :, None, None],
                torch.from_numpy(laf_ori)[None, :, None]).to(image.device)
            patches = extract_patches_from_pyramid(
                    image, lafs, PS=self.conf['patch_size'])[0]
            descriptors = patches.new_zeros((len(patches), 128))
            if len(patches) > 0:
                for start_idx in range(0, len(patches), self.max_batch_size):
                    end_idx = min(len(patches), start_idx+self.max_batch_size)
                    descriptors[start_idx:end_idx] = self.describe(
                        patches[start_idx:end_idx])
        else:
            raise ValueError(f'Unknown descriptor: {self.conf["descriptor"]}')

        keypoints = torch.from_numpy(keypoints[:, :2])  # keep only x, y
        scales = torch.from_numpy(scales)
        oris = torch.from_numpy(oris)
        scores = torch.from_numpy(scores)

        if self.conf['max_keypoints'] != -1:
            # TODO: check that the scores from PyCOLMAP are 100% correct,
            # follow https://github.com/mihaidusmanu/pycolmap/issues/8
            indices = torch.topk(scores, self.conf['max_keypoints'])
            keypoints = keypoints[indices]
            scales = scales[indices]
            oris = oris[indices]
            scores = scores[indices]
            descriptors = descriptors[indices]

        return {
            'keypoints': keypoints[None],
            'scales': scales[None],
            'oris': oris[None],
            'scores': scores[None],
            'descriptors': descriptors.T[None],
        }