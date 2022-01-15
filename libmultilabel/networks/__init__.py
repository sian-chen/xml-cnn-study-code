import torch.nn as nn

from .caml import CAML
from .kim_cnn import KimCNN
from .kim_cnn_v2 import KimCNNv2
from .xml_cnn import XMLCNN
from .xml_cnn_nh import XMLCNNnh
from .xml_cnn_liu import XMLCNNLiu
from .xml_cnn_liu_nopool import XMLCNNLiuNP
from .xml_cnn_liu_maxpool import XMLCNNLiuMP
from .xml_mlp import XMLMLP
from .xml_cnn_liu_diff import XMLCNNLiuDiff


def get_init_weight_func(init_weight):
    def init_weight_func(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            getattr(nn.init, init_weight+ '_')(m.weight)
    return init_weight_func
