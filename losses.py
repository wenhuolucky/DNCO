import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy(outputs, smooth_labels):
    loss = torch.nn.KLDivLoss(reduction="batchmean")
    return loss(F.log_softmax(outputs, dim=1), smooth_labels)


def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method
    """
    device = true_labels.device
    true_labels = torch.nn.functional.one_hot(true_labels, classes).detach().cpu()
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        _, index = torch.max(true_labels, 1)

        true_dist.scatter_(1, torch.LongTensor(index.unsqueeze(1)), confidence)
    return true_dist.to(device)


def div_loss(outpus):
    softmax_o_S = F.softmax(outpus, dim=1).mean(dim=0)
    loss_div = (softmax_o_S * torch.log10(softmax_o_S)).sum()
    return loss_div

def class_loss(outpus):
    # 对输出进行 softmax 操作，将其转换为概率分布
    probabilities = F.softmax(outputs, dim=1)
    # 找出每个样本中概率最大的类的索引
    _, max_classes = torch.max(probabilities, dim=1)
    # 计算每个类别的出现频率
    class_counts = torch.bincount(max_classes, minlength=outputs.size(1)).float()
    class_frequencies = class_counts / class_counts.sum()
    # 避免对数运算时出现零值，添加一个极小的常数
    epsilon = 1e-10
    class_frequencies = torch.clamp(class_frequencies, min=epsilon)
    # 计算信息熵，信息熵越大表示多样性越高
    entropy = -torch.sum(class_frequencies * torch.log(class_frequencies))
    # 取负号，使得多样性越高损失越小
    loss = -entropy
    return loss_class

class Entropy_Loss(nn.Module):
    def __init__(self, reduction="mean"):
        super(Entropy_Loss, self).__init__()
        self.reduction = reduction

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(dim=1)
        if self.reduction == "mean":
            return b.mean()
        elif self.reduction == "sum":
            return b.sum()
        elif self.reduction == "none":
            return b
