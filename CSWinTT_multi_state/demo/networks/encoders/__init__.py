import pdb
from CSWinTT_multi_state.demo.networks.encoders.intern_image import INTERN_T,INTERN_H,INTERN_XL,INTERN_B
from CSWinTT_multi_state.demo.networks.layers.normalization import FrozenBatchNorm2d
from torch import nn


def build_encoder(name, frozen_bn=True, freeze_at=-1):
    if frozen_bn:
        BatchNorm = FrozenBatchNorm2d
    else:
        BatchNorm = nn.BatchNorm2d

    if 'intern_t' in name:
        print("ttttttttttttttt")
        return INTERN_T()
    elif 'intern_h' in name:
        print("hhhhhhhhhhhhhhhhhhh")
        return INTERN_H()
    elif 'intern_xl' in name:
        print("xxxxxxxxxxxxxxxxxxx")
        return INTERN_XL()
    elif 'intern_b' in name:
        print("bbbbbbbbbbbbbbbbbbb")
        return INTERN_B()
    else:
        print("NotImplementedError에러")
        raise NotImplementedError
