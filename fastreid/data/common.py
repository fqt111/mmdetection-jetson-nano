# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import Dataset

from .data_utils import read_image


class CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel

        pid_set = set()
        cam_set = set()
        frame_set=set()
        person_set = set()
        for i in img_items:
            pid_set.add(i[1])
            cam_set.add(i[2])
            frame_set.add(i[3])
            person_set.add(i[4])

        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))
        self.frames = sorted(list(frame_set))
        self.persons = sorted(list(person_set))
        if relabel:
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])
            self.frame_dict = dict([(p, i) for i, p in enumerate(self.frames)])
            self.person_dict = dict([(p, i) for i, p in enumerate(self.persons)])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_item = self.img_items[index]
        img_path = img_item[0]
        pid = img_item[1]
        camid = img_item[2]
        frameid=img_item[3]
        personid = img_item[4]
        img = read_image(img_path)
        if self.transform is not None: img = self.transform(img)
        if self.relabel:
            pid = self.pid_dict[pid]
            camid = self.cam_dict[camid]
            frameid=self.frame_dict[frameid]
            personid=self.person_dict[personid]
        return {
            "images": img,
            "targets": pid,
            "camids": camid,
            "img_paths": img_path,
            "frameids":frameid,
            "person_num":personid
        }

    @property
    def num_classes(self):
        return len(self.pids)

    @property
    def num_cameras(self):
        return len(self.cams)
