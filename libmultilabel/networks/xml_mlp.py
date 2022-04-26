import math
import torch
import torch.nn as nn

from ..networks.base import BaseModel


def out_size(l_in, kernel_size, padding=0, dilation=1, stride=1):
    # refer to https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    a = l_in + 2 * padding - dilation * (kernel_size - 1) - 1
    l_out = math.floor(a / stride + 1)
    return l_out


class XMLMLP(BaseModel):
    """XML-CNN author's implementation

        Args:
            embed_vecs (FloatTensor): The pre-trained word vectors of shape (vocab_size, embed_dim).
            num_classes (int): Total number of classes.
            dropout (float): The dropout rate of the word embedding. Defaults to 0.2.
            dropout2 (float): Optional specification of the second dropout. Defaults to 0.2.
            hidden_dim (int): Dimension of the hidden layer. Defaults to 512.
            activation (str): Activation function to be used. Defaults to 'relu'.
        """
    def __init__(
        self,
        embed_vecs,
        num_classes,
        dropout2=0.2,
        hidden_dim=512,
        num_filter_per_size=256,
        max_seq_length=500,
        **kwargs
    ):
        super(XMLMLP, self).__init__(embed_vecs, **kwargs)
        emb_dim = embed_vecs.shape[1]
        total_output_size  = emb_dim * max_seq_length

        self.dropout2 = nn.Dropout(dropout2)
        self.linear1 = nn.Linear(total_output_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, text):
        h = self.embedding(text) # (batch_size, length, embed_dim)
        h = self.embed_drop(h)
        h = h.view(h.shape[0], -1)

        # linear output
        h = self.activation(self.linear1(h))
        h = self.dropout2(h)
        h = self.linear2(h)
        return {'logits': h}
