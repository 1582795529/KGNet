import glob
import re

import os.path as osp

from .bases import BaseImageDataset


class VeRi(BaseImageDataset):
    """
       VeRi-776
       Reference:
       Liu, Xinchen, et al. "Large-scale vehicle re-identification in urban surveillance videos." ICME 2016.

       URL:https://vehiclereid.github.io/VeRi/

       Dataset statistics:
       # identities: 776
       # images: 37778 (train) + 1678 (query) + 11579 (gallery)
       # cameras: 20
       """

    dataset_dir = 'VeRi'

    def __init__(self, root='/home/fei' ,verbose=True, **kwargs):
        super(VeRi, self).__init__()

        info_train = './data/VehicleKeyPointData/keypoint_train.txt'
        info_query = './data/VehicleKeyPointData/keypoint_query.txt'
        info_test = './data/VehicleKeyPointData/keypoint_test.txt'
        info_query_gallary = './data/VehicleKeyPointData/keypoint_query+gallary.txt'

        train = self._process_dir(root,info_train, relabel=True)
        query = self._process_dir(root,info_query, relabel=False)
        gallery = self._process_dir(root,info_test, relabel=False)
        query_gallary = self._process_dir(root,info_query_gallary, relabel=False)

        if verbose:
            print("=> VeRi-776 loaded")
            self.print_dataset_statistics(train, query, gallery,query_gallary)

        self.train = train
        self.query = query
        self.gallery = gallery
        self.query_gallary = query_gallary

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
        self.num_query_gallary_pids, self.num_query_gallary_imgs, self.num_quary_gallery_cams = self.get_imagedata_info(self.query_gallary)


    def _process_dir(self, dir_path,info_path, relabel=False):
        img_paths,landmark_path = self._get_TrainVeriinfo(dir_path,info_path)
        pattern = re.compile(r'([-\d]+)_c(\d+)')

        pid_container = set()
        for img_path in img_paths:
            pid, r = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 776  # pid == 0 means background
            assert 1 <= camid <= 20
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, landmark_path,pid, camid))

        return dataset

    def _get_TrainVeriinfo(self,root_path,info_path):
        path_list = []
        landmark_list = []
        anno = [line.rstrip('\n') for line in open(info_path)]

        # remove missing data files
        no_data = []
        for line in anno:
            if not osp.isfile(osp.join(root_path, line.split(' ')[0])):
                no_data.append(line)
        for line in no_data:
            anno.remove(line)

        for line in anno:
            path_list.append(osp.join(root_path, line.split(' ')[0]))
            landmark_list.append(line)

        return path_list,landmark_list

