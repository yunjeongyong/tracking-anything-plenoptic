import copy
import torch

# def cocopy(img):
#     img_ = copy.deepcopy(img)
#     return img_
#
# list = []
# img1 = torch.randn(1, 3, 224, 224)
# img2 = torch.randn(1, 3, 224, 224)
# list.append(img1)
# list.append(img2)

# for i in list:
#     oo = cocopy(i)
#     print(oo.shape)


def denormalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # 3, H, W, B
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

img = torch.randn(1, 3, 224, 224)
print(img)
a = denormalize(img)
print(a)
