from utils import read_image, plot_sequence
from triangulation import SuperPointDetector, FrameByFrameMatcher
from triangulation.tools import plot_keypoints, plot_matches
from pathlib import Path
import cv2

if __name__ == '__main__':
      # image path
      root = Path('dataset/images_seq')
      images_t = [1418134682928102, 1418134683428029, 1418134683990456, 1418134684615372, 1418134685115302]
      images_path = [f'day0/{str(t)}.jpg' for t in images_t]
      images = [read_image(Path(root , image)) for image in images_path]

      # plot images
      plot_sequence([images], label='Test sequence', show=True)

      # triangulate points
      # create Detector
      detector = SuperPointDetector()
      # create Matcher
      matcher = FrameByFrameMatcher({"type": "FLANN"})
      # detect keypoints
      kptdescs = [detector(img) for img in images]
      # match keypoints
      imgs = {}
      kptdescs = {}
      
      for i, image in enumerate(images):
          imgs["cur"] = image
          kptdescs["cur"] = detector(image)
          if i >= 1:
            matches = matcher(kptdescs)
            img = plot_matches(imgs['ref'], imgs['cur'],
                               matches['ref_keypoints'][0:200], matches['cur_keypoints'][0:200],
                               matches['match_score'][0:200], layout='lr')
            cv2.imshow("track", img)
            if cv2.waitKey() == 27:
                break

          kptdescs["ref"], imgs["ref"] = kptdescs["cur"], imgs["cur"]
          
      
      

    
      
      
    #   for image in images_path:
    #       image = cv2.imread(str(Path(root, image)))
    #       kpts = detector(image)
    #       img = plot_keypoints(image, kpts["keypoints"], kpts["scores"])
    #       cv2.imshow("SuperPoint", img)
    #       cv2.waitKey()
      
    #   for img in images:
    #         detector = SuperPointDetector()
    #         kpts = detector(img)
            
    #         img = plot_keypoints(img, kpts["keypoints"], kpts["scores"])
    #         cv2.imshow("SuperPoint", img)
    #         cv2.waitKey()