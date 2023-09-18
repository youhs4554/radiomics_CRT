import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        # weight parameter will act as the alpha parameter to balance class weights
        self.weight = weight

    def forward(self, input, target):

        ce_loss = F.cross_entropy(
            input, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"

    def __init__(self, alpha=.25, gamma=2, smoothing=0.0):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, inputs, labels):
        with torch.no_grad():
            # label smoothing!
            targets = torch.ones_like(labels).mul(self.smoothing)
            targets.masked_fill_(labels.eq(1), 1.0-self.smoothing)

        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none')

        at = self.alpha.gather(0, labels.long().data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()


class Balanced_FocalLoss(nn.Module):

    def __init__(self, focusing_param=2, balance_param=1.0):
        super(Balanced_FocalLoss, self).__init__()

        self.focusing_param = focusing_param
        self.balance_param = balance_param
        self.num_of_classes = 2
        self.beta = 0.9999

    def balanced_weight(self, output, target, samples_per_current_batch, accum_neg, accum_pos):

        samples_per_cls = [0, 0]
        samples_per_cls[0] = samples_per_current_batch[0] + accum_neg
        samples_per_cls[1] = samples_per_current_batch[1] + accum_pos

        effective_num = 1.0 - np.power(self.beta, samples_per_cls) + 0.000001
        weights = (1.0 - self.beta) / np.array(effective_num)

        weights = torch.tensor(weights).float().to(target.device)
        weights = weights.unsqueeze(0)
        weights = weights.repeat(target.shape[0], 1) * target
        weights = weights.sum(1)

        return weights, samples_per_cls

    def forward(self, output, target, accum_neg, accum_pos):

        cross_entropy = F.cross_entropy(output, target, reduction='none')
        cross_entropy_log = torch.log(cross_entropy)
        logpt = - F.cross_entropy(output, target, reduction='none')
        pt = torch.exp(logpt)

        focal_loss = -((1 - pt) ** self.focusing_param) * logpt

        labels_one_hot = F.one_hot(target, self.num_of_classes).float()

        target_list = target.tolist()
        sample_num = []
        positive = 0.0
        negative = 0.0
        for label in target_list:
            if label == 0:
                negative += 1
            else:
                positive += 1
        sample_num.append(negative)
        sample_num.append(positive)

        weight, samples_per_cls = self.balanced_weight(
            output, labels_one_hot, sample_num, accum_neg, accum_pos)

        balanced_focal_loss = weight * focal_loss
        balanced_focal_loss = torch.sum(balanced_focal_loss)
        balanced_focal_loss /= torch.sum(labels_one_hot)
        #import ipdb; ipdb.set_trace()
        return balanced_focal_loss, samples_per_cls


def test_focal_loss():
    loss = FocalLoss()

    input = Variable(torch.randn(3, 5), requires_grad=True)
    target = Variable(torch.LongTensor(3).random_(5))

    print(input)
    print(target)

    output = loss(input, target)
    print(output)
    output.backward()
