import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.densenet import densenet121, densenet201
from torchvision.models.resnet import resnet34, resnet50, resnet101, resnet152


def get_model(m):
    '''Get model and the indicator of whether it is pre-trained.'''
    C = None
    for k, v in globals().items():
        if k.lower() == m.lower():
            C = v
            break
    else:
        raise ValueError("invalid model name '{0}'".format(m))
    return C, m[0].lower() in ('d', 'r')


class DenseNet121(nn.Module):
    def __init__(self, num_classes=10):
        super(DenseNet121, self).__init__()
        orig_model = densenet121(pretrained=True)
        self.features = nn.Sequential(*list(orig_model.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.PReLU(),
            nn.Linear(256, num_classes)
        )
        for layer in self.classifier.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform(layer.weight.data)

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7).view(x.size(0), -1)
        x = self.classifier(x)
        return x


class DenseNet201(nn.Module):
    def __init__(self, num_classes=10):
        super(DenseNet201, self).__init__()
        orig_model = densenet201(pretrained=True)
        self.features = nn.Sequential(*list(orig_model.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(1920, 256),
            nn.PReLU(),
            nn.Linear(256, num_classes)
        )
        for layer in self.classifier.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform(layer.weight.data)

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=5).view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNet34(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet34, self).__init__()
        orig_model = resnet34(pretrained=True)
        # self.features = nn.Sequential(*list(orig_model.children())[:-1])
        self.features = nn.Sequential(*list(orig_model.children())[:-2])
        # 2048-256-10
        self.classifier = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        for layer in self.classifier.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform(layer.weight.data)

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=2)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__()
        orig_model = resnet50(pretrained=True)
        self.features = nn.Sequential(*list(orig_model.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.PReLU(),
            nn.Linear(512, num_classes)
        )
        for layer in self.classifier.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform(layer.weight.data)

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=2).view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNet101(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet101, self).__init__()
        orig_model = resnet101(pretrained=True)
        self.features = nn.Sequential(*list(orig_model.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(2048, 256),
            nn.PReLU(),
            nn.Linear(256, num_classes)
        )
        for layer in self.classifier.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform(layer.weight.data)

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=2).view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNet152(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet152, self).__init__()
        orig_model = resnet152(pretrained=True)
        self.features = nn.Sequential(*list(orig_model.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(2048, 256),
            nn.PReLU(),
            nn.Linear(256, num_classes)
        )
        for layer in self.classifier.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform(layer.weight.data)

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=2).view(x.size(0), -1)
        x = self.classifier(x)
        return x


class CNN2(nn.Module):
    def __init__(self, num_classes=10, input_size=256):
        super(CNN2, self).__init__()

        self.input_size = input_size
        assert self.input_size in [128, 256]

        features = [
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features=32),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features=48),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features=128),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        if self.input_size == 256:
            features += [
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1),
                nn.BatchNorm2d(num_features=256),
                nn.PReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
        elif self.input_size == 128:
            features += [
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
                nn.BatchNorm2d(num_features=256),
                nn.PReLU()
            ]
        self.features = nn.Sequential(*features)

        n_units = [4096, 256] if self.input_size == 256 else [1024, 128]
        self.classifier = nn.Sequential(
            nn.Linear(n_units[0], n_units[1]),
            nn.PReLU(),
            nn.Linear(n_units[1], num_classes),
        )
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform(layer.weight.data)
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform(layer.weight.data)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class CNN3(nn.Module):
    def __init__(self, num_classes=10, input_size=256):
        super(CNN3, self).__init__()

        self.input_size = input_size
        assert self.input_size in [128, 256]

        features = [
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=7, stride=1),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
        ]
        if self.input_size == 256:
            features += [
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1),
                nn.BatchNorm2d(num_features=1024),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
        self.features = nn.Sequential(*features)

        n_units = [1024, 256] if self.input_size == 256 else [512, 128]
        self.classifier = nn.Sequential(
            nn.Linear(n_units[0], n_units[1]),
            nn.ReLU(inplace=True),
            nn.Linear(n_units[1], num_classes),
        )
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform(layer.weight.data)
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform(layer.weight.data)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
