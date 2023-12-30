from hloc.utils.io import list_h5_names
import h5py
import pdb

# print(len(names))


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
      file = 'features/sift.h5'
      # read_h5(file)
      names = list_h5_names(file)
      print(names)
