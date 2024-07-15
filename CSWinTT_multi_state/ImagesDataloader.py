from __future__ import annotations

import os
from PIL import Image
import numpy as np
import cv2
import torch


class Load2DFolder(torch.utils.data.Dataset):
    def __init__(self, root, main_camera=0, limit=None):
        self.root = root
        self.main_camera = main_camera
        self.limit = limit
        self.paths = [os.path.join(self.root, i) for i in os.listdir(self.root)]
        self.paths.sort()
        self.files: list[list[str]] = []

        for i, path in enumerate(self.paths):
            # h_000_000.jpg
            a = path.split("\\")
            path = a[-1]
            _, frame, camera = path.split("_")
            camera = camera.split(".")[0]
            if int(camera) == 0:
                self.files.append([])
            self.files[int(frame)].append(os.path.join(self.root, path))

    def __getitem__(self, idx):
        limit = len(self.files[idx]) if self.limit is None else self.limit - 1
        print(self.files[idx][self.main_camera])
        images = [cv2.imread(self.files[idx][i], cv2.COLOR_BGR2RGB) for i in range(len(self.files[idx]))]
        # cameras = [i for i in range(limit)]
        # images.insert(0, cv2.imread(self.files[idx][self.main_camera], cv2.COLOR_BGR2RGB))
        # cameras.insert(0, self.main_camera)
        return images

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    dataloader = Load2DFolder(root='E:\\code\\SuperGluePretrainedNetwork-master\\unfold_images_color_1@', main_camera=7)
    print(np.array(dataloader.files).shape)
    img = dataloader[0]
    a = len(img)
    print(a)
    print(img)

