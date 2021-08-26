import torch.nn as nn

from .caml import CAML
from .kim_cnn import KimCNN
from .xml_cnn import XMLCNN
from .xml_cnn_v0 import XMLCNNv0
from .xml_cnn_liu import XMLCNNLiu


def get_init_weight_func(init_weight):
    def init_weight_func(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            getattr(nn.init, init_weight+ '_')(m.weight)
    return init_weight_func
