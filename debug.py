from hloc.utils.io import list_h5_names
import h5py
import pdb
from gvbench.utils import parse_pairs
from pathlib import Path
from gvbench.evaluation import get_matches
from gvbench.utils import read_image
import cv2

def list_h5_names(path):
    names = []
    with h5py.File(str(path), 'r', libver='latest') as fd:
        def visit_fn(_, obj):
            if isinstance(obj, h5py.Dataset):
                names.append(obj.parent.name.strip('/'))
        fd.visititems(visit_fn)
    return list(set(names))

def names_to_pair(name0, name1, separator='/'):
    return separator.join((name0.replace('/', '-'), name1.replace('/', '-')))

def read_h5(path):
      with h5py.File(str(path), 'r', libver='latest') as fd:
            keys = list(fd.keys())
            for key in keys:
                  pdb.set_trace()
                  key = key.strip('/')
                  print(key)
                  new_key = 'Night_mini_val/' + key
                  print(new_key)
                  fd.copy(key, new_key)
                  # del fd[key]
                  # names_to_pair(key, new_key)
                  # print()
                  # print('Night_mini_val' + key)
                  # print(fd[key])
                  # fd[]
            # dset = fd['superpoint']['keypoints']
            # p = dset.__array__()
            # uncertainty = dset.attrs.get('uncertainty')

if __name__ == '__main__':
    # file = 'dataset/robotcar/matches/robotcar_qAutumn_dbNight/matches-NN-mutual-ratio.8.h5'
    # file_path = Path(file)
    # print(file_path.stem)
    
    # # Found 2 matches between Autumn_mini_val/1418133267245762.jpg and Night_mini_val/1418757223104943.jpg.
    query = 'dataset/robotcar/images/Autumn_mini_val/1418133267245762.jpg'
    reference = 'dataset/robotcar/images/Night_mini_val/1418757223104943.jpg'
    orb = cv2.ORB_create(nfeatures=1000)
    
    kpt0, des0 = orb.detectAndCompute(read_image(query, True), None) 
    kpt1, des1 = orb.detectAndCompute(read_image(reference, True), None) 
    
    print(f'kpt0: {kpt0}')
    print(f'kpt1: {kpt1}')
    print(f'des0: {des0}')
    print(f'des1: {des1}')
        
    # matches, scores = get_matches(file, query, reference)
    # print(matches, scores)
    
    # pairs_loader = parse_pairs(file, allow_label=True)
    # pairs = [(q, r, int(l)) for q, r, l in pairs_loader]
    # print(pairs[0])
    # print(pairs[:][2])
    # labels = [pair[2] for pair in pairs]
    # print(labels)
    
    #   # read_h5(file)
    # names = list_h5_names(file)
    # print(names)
    # image0 = load_image(images / "DSC_0411.JPG")
    # image1 = load_image(images / "DSC_0410.JPG")

    # feats0 = extractor.extract(image0.to(device))
    # feats1 = extractor.extract(image1.to(device))
    # matches01 = matcher({"image0": feats0, "image1": feats1})
    # feats0, feats1, matches01 = [
    #     rbd(x) for x in [feats0, feats1, matches01]
    # ]  # remove batch dimension

    # kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    # m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    # axes = viz2d.plot_images([image0, image1])
    # viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    # viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)

    # kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
    # viz2d.plot_images([image0, image1])
    # viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
