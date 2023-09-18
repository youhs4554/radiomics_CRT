import torch
import torchvision
from . import utils


class I3ResNet(torch.nn.Module):
    def __init__(self, resnet2d, frame_nb=16, class_nb=1000, conv_class=False, freeze_bottom=False, clinical_hidden=None, use_bottleneck=True):
        """
        Args:
            conv_class: Whether to use convolutional layer as classifier to
                adapt to various number of frames
        """
        super(I3ResNet, self).__init__()
        self.conv_class = conv_class
        self.freeze_bottom = freeze_bottom

        self.stem = torch.nn.Sequential(
            utils.inflate_conv(resnet2d.conv1, time_dim=3,
                               time_padding=1, center=True),
            utils.inflate_batch_norm(resnet2d.bn1),
            torch.nn.ReLU(inplace=True),
            utils.inflate_pool(
                resnet2d.maxpool, time_dim=3, time_padding=1, time_stride=2
            ),
        )

        self.layer1 = inflate_reslayer(
            resnet2d.layer1, bottleneck=use_bottleneck)
        self.layer2 = inflate_reslayer(
            resnet2d.layer2, bottleneck=use_bottleneck)
        self.layer3 = inflate_reslayer(
            resnet2d.layer3, bottleneck=use_bottleneck)
        self.layer4 = inflate_reslayer(
            resnet2d.layer4, bottleneck=use_bottleneck)

        if conv_class:
            self.classifier = torch.nn.Conv3d(
                in_channels=2048,
                out_channels=class_nb,
                kernel_size=(1, 1, 1),
                bias=True,
            )
        else:
            self.fc = utils.inflate_linear(resnet2d.fc, 1)

    def train(self, mode=True):
        """
        Override the default train() to freeze the layers except for the top layers
        :return:
        """
        super(I3ResNet, self).train(mode)
        if self.freeze_bottom:
            for name, m in self.named_modules():
                # if not("layer3" in name or "layer4" in name or "fc" in name):
                if "fc" not in name:
                    module_name = m.__class__.__name__
                    if "Conv" in module_name or "BatchNorm" in module_name:
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        if m.bias is not None:
                            m.bias.requires_grad = False

    def get_last_feats(self):
        return self.last_feats

    def forward(self, x, clinical_factor=None):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.conv_class:
            x = x.mean((2, 3, 4))  # spatio-temporal average pooling
            x = self.classifier(x)
            x = x.squeeze(3)
            x = x.squeeze(3)
            x = x.mean(2)
        else:
            # x = self.avgpool(x)
            x = x.mean((2, 3, 4))  # spatio-temporal average pooling
            self.last_feats = x
            x_reshape = x.view(x.size(0), -1)
            if getattr(self, "clinical_mlp", None) is not None:
                clinical_feats = self.clinical_mlp(clinical_factor)
                x_reshape = torch.cat((x_reshape, clinical_feats), 1)

            x = self.fc(x_reshape)

        return x


def inflate_reslayer(reslayer2d, bottleneck=True):
    __block_class = Bottleneck3d if bottleneck else Basicblock3d
    reslayers3d = []
    for layer2d in reslayer2d:
        layer3d = __block_class(layer2d)
        reslayers3d.append(layer3d)
    return torch.nn.Sequential(*reslayers3d)


class Basicblock3d(torch.nn.Module):
    def __init__(self, basicblock2d):
        super(Basicblock3d, self).__init__()

        spatial_stride = basicblock2d.conv1.stride[0]

        self.conv1 = utils.inflate_conv(
            basicblock2d.conv1, time_dim=1, center=True)
        self.bn1 = utils.inflate_batch_norm(basicblock2d.bn1)

        self.conv2 = utils.inflate_conv(
            basicblock2d.conv2,
            time_dim=3,
            time_padding=1,
            time_stride=spatial_stride,
            center=True
        )
        self.bn2 = utils.inflate_batch_norm(basicblock2d.bn2)

        self.relu = torch.nn.ReLU(inplace=True)

        if basicblock2d.downsample is not None:
            self.downsample = inflate_downsample(
                basicblock2d.downsample, time_stride=spatial_stride
            )
        else:
            self.downsample = None
        self.stride = basicblock2d.stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck3d(torch.nn.Module):
    def __init__(self, bottleneck2d):
        super(Bottleneck3d, self).__init__()

        spatial_stride = bottleneck2d.conv2.stride[0]

        self.conv1 = utils.inflate_conv(
            bottleneck2d.conv1, time_dim=1, center=True)
        self.bn1 = utils.inflate_batch_norm(bottleneck2d.bn1)

        self.conv2 = utils.inflate_conv(
            bottleneck2d.conv2,
            time_dim=3,
            time_padding=1,
            time_stride=spatial_stride,
            center=True,
        )
        self.bn2 = utils.inflate_batch_norm(bottleneck2d.bn2)

        self.conv3 = utils.inflate_conv(
            bottleneck2d.conv3, time_dim=1, center=True)
        self.bn3 = utils.inflate_batch_norm(bottleneck2d.bn3)

        self.relu = torch.nn.ReLU(inplace=True)

        if bottleneck2d.downsample is not None:
            self.downsample = inflate_downsample(
                bottleneck2d.downsample, time_stride=spatial_stride
            )
        else:
            self.downsample = None

        self.stride = bottleneck2d.stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def inflate_downsample(downsample2d, time_stride=1):
    downsample3d = torch.nn.Sequential(
        utils.inflate_conv(
            downsample2d[0], time_dim=1, time_stride=time_stride, center=True
        ),
        utils.inflate_batch_norm(downsample2d[1]),
    )
    return downsample3d


def inflated_resnet(arch, frame_nb=16, clinical_hidden=0):

    if arch == "R18":
        use_bottleneck = False
        resnet = torchvision.models.resnet18(pretrained=True, progress=False)
    elif arch == "R34":
        use_bottleneck = False
        resnet = torchvision.models.resnet34(pretrained=True, progress=False)
    elif arch == "R50":
        use_bottleneck = True
        resnet = torchvision.models.resnet50(pretrained=True)
    elif arch == "R101":
        use_bottleneck = True
        resnet = torchvision.models.resnet101(pretrained=True)
    elif arch == "R152":
        use_bottleneck = True
        resnet = torchvision.models.resnet152(pretrained=True)

    freeze_bottom = True

    inflated_resnetnet = I3ResNet(
        resnet, frame_nb=frame_nb, clinical_hidden=clinical_hidden, freeze_bottom=freeze_bottom, use_bottleneck=use_bottleneck)
    return inflated_resnetnet
