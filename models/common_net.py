import torch
import torch.nn as nn

##########################################################
# Adaptive Instance Normalization
##########################################################
class AdaIN(nn.Module):
    def __init__(self, in_channel, num_classes, eps=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.l1 = nn.Linear(num_classes, in_channel * 4, bias=True)  # bias is good :)
        self.emb = nn.Embedding(num_classes, num_classes)

    def c_norm(self, x, bs, ch, eps=1e-7):
        assert isinstance(x, torch.cuda.FloatTensor)
        x_var = x.var(dim=-1) + eps
        x_std = x_var.sqrt().view(bs, ch, 1, 1)
        x_mean = x.mean(dim=-1).view(bs, ch, 1, 1)
        return x_std, x_mean

    def forward(self, x, y):
        assert x.size(0) == y.size(0)
        size = x.size()
        bs, ch = size[:2]
        x_ = x.view(bs, ch, -1)
        y_ = self.l1(self.emb(y.to(torch.int64))).view(bs, ch, -1)
        x_std, x_mean = self.c_norm(x_, bs, ch, eps=self.eps)
        y_std, y_mean = self.c_norm(y_, bs, ch, eps=self.eps)
        out = ((x - x_mean.expand(size)) / x_std.expand(size)) * y_std.expand(
            size
        ) + y_mean.expand(size)
        return out
