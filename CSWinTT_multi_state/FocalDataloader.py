import os
from PIL import Image
import numpy as np
import cv2


class LoadFocalFolder:
    def __init__(self, root, type, frame_range=None, focal_range=None):
        self.root = root
        # self.frames = os.listdir(root)
        # self.frames = [str(i) for i in range(frame_range[0], frame_range[1]+1)]  #  newvideo1: "images/007.png", 65 start frame
        # print('self.frames',self.frames)
        # self.frames = [str(i).zfill(3) for i in range(frame_range[0], frame_range[1] + 1)] #  newvideo1: "images/007.png", 065 start frame
        self.frames = [str(i) for i in range(frame_range[0], frame_range[1] + 1)]
        print('self.frames', self.frames)
        # self.frames = [str(i).zfill(3) for i in range(frame_range[0], frame_range[1]+1)]  #  Nonvideo3: "images/005.png", 33 start
        self.frame_range = frame_range
        self.type = type # focal or images
        self.focal_images_path = []
        self.focal_range = focal_range
        self.images_2d_path = []

        # frame range
        # if self.frame_range is None:
        #     pass
        # else:
        #     self.frames = self.frames[frame_range[0]-int(self.frames[0]): frame_range[1]-int(self.frames[0]) + 1]

        # 2D images setting
        for frame in self.frames:
            type_path = os.path.join(self.root, frame, 'images')
            # image = os.listdir(type_path)[7]

            # print(os.listdir(type_path))
            # image_path = os.path.join(type_path, '005.png')  #  Nonvideo3
            image_path = os.path.join(type_path, '007.png')  #  newvideo1
            self.images_2d_path.append(image_path)

        # focal images setting
        for frame in self.frames:
            type_path = os.path.join(root, frame, self.type)
            focal_images_name = [f'{i:03d}.png' for i in range(100)] # '000', '001',,,
            focal_planes = []

            for focal_image in focal_images_name:
                focal_image_path = os.path.join(type_path, focal_image)
                focal_planes.append(focal_image_path)  # 'D:/dataset/NonVideo3_tiny\\000\\focal\\020.png',,

            # focal range
            if self.focal_range is None:
                pass
            else:
                focal_planes = focal_planes[self.focal_range[0]: self.focal_range[1] + 1]

            self.focal_images_path.append(focal_planes)
            # self.images_path.append(os.path.join(root, frame, self.type))

    def __getitem__(self, idx):
        path = self.images_2d_path[idx]
        print(path)
        # img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.imread(path)
        # img = cv2.resize(img, (240, 135))
        focal_plane = self.focal_images_path[idx]

        # for focal_image_path in focal_plane:
        #     focal_image = cv2.imread(focal_image_path)
        #     focal_image_resize = focal_image_resize[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        #     focal_image_resize = np.ascontiguousarray(focal_image_resize)
        #     focal_planes_resize.append(focal_image_resize)

        return img, focal_plane

        # 2D Images setting

    # def set_images_path(self):
    #     images_path = []
    #     for frame in self.frames:
    #         type_path = os.path.join(self.root, frame, 'images')
    #         image = os.listdir(type_path)[5]
    #         image_path = os.path.join(type_path, image)
    #         images_path.append(image_path)
    #     return images_path


if __name__ == '__main__':
    # dataloader = LoadFocalFolder(root='/ssd2/vot_data/newvideo1/', type='focal', frame_range=(66, 300), focal_range=(0, 100))
    dataloader = LoadFocalFolder(root='E:\\tracker과제\\newvideo1_001_300', type='focal', frame_range=(66, 300), focal_range=(0, 100))
    print(np.array(dataloader.focal_images_path).shape)
    img = dataloader[0]
    print(img)
