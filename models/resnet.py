from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from torch.hub import load_state_dict_from_url

# Source: https://github.com/c0nn3r/RetinaNet/blob/master/resnet_features.py 

class BasicBlockFeatures(BasicBlock):
    '''
    BasicBlock that returns its last conv layer features.
    Source: https://github.com/c0nn3r/RetinaNet/blob/master/resnet_features.py  
    '''

    def forward(self, x):

        if isinstance(x, tuple):
            x = x[0]

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        conv2_rep = out
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out, conv2_rep

class BottleneckFeatures(Bottleneck):
    '''
    Bottleneck that returns its last conv layer features.
    Source: https://github.com/c0nn3r/RetinaNet/blob/master/resnet_features.py  
    '''

    def forward(self, x):

        if isinstance(x, tuple):
            x = x[0]

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        conv3_rep = out
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out, conv3_rep



class ResNetFeatures(ResNet):
    '''
    A ResNet that returns features instead of classification.
    Source: https://github.com/c0nn3r/RetinaNet/blob/master/resnet_features.py  
    '''

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x, c2 = self.layer1(x)
        x, c3 = self.layer2(x)
        x, c4 = self.layer3(x)
        x, c5 = self.layer4(x)

        return c2, c3, c4, c5


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def resnet18_features(pretrained=False, **kwargs):
    '''Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    '''
    model = ResNetFeatures(BasicBlockFeatures, [2, 2, 2, 2], **kwargs)

    if pretrained:
        checkpoint = model_urls['resnet18']
        state_dict = load_state_dict_from_url(checkpoint, progress=True, check_hash=True)
        model.load_state_dict(state_dict)

    return model


def resnet34_features(pretrained=False, **kwargs):
    '''Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    '''
    model = ResNetFeatures(BasicBlockFeatures, [3, 4, 6, 3], **kwargs)

    if pretrained:
        checkpoint = model_urls['resnet34']
        state_dict = load_state_dict_from_url(checkpoint, progress=True, check_hash=True)
        model.load_state_dict(state_dict)   

    return model


def resnet50_features(pretrained=False, **kwargs):
    '''Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    '''
    model = ResNetFeatures(BottleneckFeatures, [3, 4, 6, 3], **kwargs)

    if pretrained:
        checkpoint = model_urls['resnet50']
        state_dict = load_state_dict_from_url(checkpoint, progress=True, check_hash=True)
        model.load_state_dict(state_dict)
    
    return model


def resnet101_features(pretrained=False, **kwargs):
    '''Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    '''
    model = ResNetFeatures(BottleneckFeatures, [3, 4, 23, 3], **kwargs)

    if pretrained:
        checkpoint = model_urls['resnet101']
        state_dict = load_state_dict_from_url(checkpoint, progress=True, check_hash=True)
        model.load_state_dict(state_dict)
        
    return model


def resnet152_features(pretrained=False, **kwargs):
    '''Constructs a ResNet-152 model with number of classes being 2 (benign or malignant).
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    '''
    model = ResNetFeatures(BottleneckFeatures, [3, 8, 36, 3], num_classes=2, **kwargs)

    if pretrained:
        checkpoint = model_urls['resnet152']
        state_dict = load_state_dict_from_url(checkpoint, progress=True, check_hash=True)
        model.load_state_dict(state_dict)

    return model