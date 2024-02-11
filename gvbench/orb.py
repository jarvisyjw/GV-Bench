from hloc.utils.base_model import BaseModel
import cv2
import torch
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import pprint
from tqdm import tqdm
import h5py


from . import logger
from . import data_loader
from hloc.utils.io import list_h5_names


class ORB(BaseModel):
    
    required_inputs = ['image']

    def _init(self, conf):
        self.orb = cv2.ORB_create(conf['options']['nfeatures'])
        
    def _forward(self, data):
        image = data['image']
      #   image_np = image.cpu().numpy()[0, 0]
        # plt.imshow(image_np)
        # plt.show()
        # print(image_np.shape)
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        print(descriptors)
        
        kpts = [kpt.pt for kpt in keypoints]
        print(kpts)
        scores = [kpt.response for kpt in keypoints]
        
        keypoints = np.array(kpts)  # keep only x, y
        scores = np.array(scores)

        return {
            'keypoints': keypoints,
            'scores': scores,
            'descriptors': descriptors.T,
        }


def main(conf, image_dir: Path, 
         export_dir: Path, 
         feature_path: Path = None, 
         overwrite: bool = False):
      
      logger.info('Extracting local features with configuration:'
                f'\n{pprint.pformat(conf)}')

      dataset = data_loader(image_dir, grayscal=True, resize_max=1600)
      if feature_path is None:
            feature_path = Path(export_dir, conf['output']+'.h5')
      feature_path.parent.mkdir(exist_ok=True, parents=True)
      skip_names = set(list_h5_names(feature_path)
                        if feature_path.exists() and not overwrite else ())
      dataset.names = [n for n in dataset.names if n not in skip_names]
      if len(dataset.names) == 0:
            logger.info('Skipping the extraction.')
            return feature_path
      
      default_conf = {
        'options': {
            'nfeatures': 1000,
            'scaleFactor': '1.2f',
            'nlevels': 8,
            'edgeThreshold': 31,
            'firstLevel': 0,
            'WTA_K': 2,
            'scoreType': cv2.ORB_HARRIS_SCORE,
            'patchSize': 31,
            'fastThreshold': 20,
        }
    }
      
      extractor = ORB(default_conf)
      
    #   loader = torch.utils.data.dataloader(
    #         dataset, num_workers=1, shuffle=False, pin_memory=True)
    
    loader = 
    
      
      for idx, data in enumerate(tqdm(loader)):
            name = dataset.names[idx]
            print(data['image'])
            pred = extractor({'image': data['image']})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            logger.debug(f'pred: {pred}')

            pred['image_size'] = original_size = data['original_size'][0].numpy()
            
            if 'keypoints' in pred:
                  size = np.array(data['image'].shape[-2:][::-1])
                  scales = (original_size / size).astype(np.float32)
                  pred['keypoints'] = (pred['keypoints'] + .5) * scales[None] - .5
                  # add keypoint uncertainties scaled to the original resolution
            
            with h5py.File(str(feature_path), 'a', libver='latest') as fd:
                  try:
                        if name in fd:
                              del fd[name]
                        grp = fd.create_group(name)
                        for k, v in pred.items():
                              grp.create_dataset(k, data=v)
                  except OSError as error:
                        if 'No space left on device' in error.args[0]:
                              logger.error(
                                    'Out of disk space: storing features on disk can take '
                                    'significant space, did you enable the as_half flag?')
                              del grp, fd[name]
                  raise error
      
            del pred
      
      logger.info(f'Extracted {conf["output"]} to {feature_path}. DONE!')


if __name__ == "__main__":
      conf = {
        'output': 'feats-orb',
        'model': {
            'name': 'orb'
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1600,
            'normalization': True,}}
      main(conf, Path('dataset/robotcar/images/'), Path('dataset/robotcar/features/'), Path('dataset/robotcar/features/orb.h5'))