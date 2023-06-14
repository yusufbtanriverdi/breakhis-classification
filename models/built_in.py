import torch
import torchvision.models as models
from torchvision import datasets, transforms as T
import os, sys

parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
# Add the parent directory to the Python path
sys.path.append(parent_dir)
from tools import BreaKHis

# from torch hub...

def init_weights_default(num_channels):
    # Define a simple convolutional layer
    conv = torch.nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1)
    return conv.weight.data

def normalize_data_for_builtin(root='D:\\BreaKHis_v1\\', mf='40X', mode='binary'):

    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    dataset = BreaKHis(root=root, mf=mf, mode=mode, transform=transform)

    means = []
    stds = []
    # TODO: Test here.
    for img in dataset:
        means.append(torch.mean(img))
        stds.append(torch.std(img))

    mean = torch.mean(torch.tensor(means))
    std = torch.mean(torch.tensor(stds))

    normalize = T.Normalize(mean=mean, std=std)

    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])
    dataset = BreaKHis(transform=transform)

    return dataset

def GhostNet():
    """All pre-trained models expect input images normalized in the same way, 
    i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H 
    and W are expected to be at least 224. The images have to be loaded in 
    to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] 
    and std = [0.229, 0.224, 0.225].
    Here’s a sample execution."""

    """# Download an example image from the pytorch website
    import urllib
    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    try: urllib.URLopener().retrieve(url, filename)
    except: urllib.request.urlretrieve(url, filename)
    # sample execution (requires torchvision)
    from PIL import Image
    from torchvision import transforms
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    print(output[0])
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    print(probabilities)"""

    ghostnet = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=False)

    return ghostnet 


def call_builtin_models(pretrained=True, num_classes=2, num_channels=3):
    """
    Returns a dictionary of built-in models. 

    Some models use modules which have different training and evaluation behavior, 
    such as batch normalization. To switch between these modes, use model.train() 
    or model.eval() as appropriate. See train() or eval() for details.

    All pre-trained models expect input images normalized in the same way,
    i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where 
    H and W are expected to be at least 224. The images have to be loaded 
    in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] 
    and std = [0.229, 0.224, 0.225]. You can use the following transform to normalize:

    # To normalize imagenet.
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]) 

    # To normalize custom dataset.                             
    import torch
    from torchvision import datasets, transforms as T

    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    dataset = datasets.ImageNet(".", split="train", transform=transform)

    means = []
    stds = []
    for img in subset(dataset):
        means.append(torch.mean(img))
        stds.append(torch.std(img))

    mean = torch.mean(torch.tensor(means))
    std = torch.mean(torch.tensor(stds))
    """

    # model_dict = {
    # 'resnet18': models.resnet18(pretrained=pretrained),
    # 'alexnet' : models.alexnet(pretrained=pretrained),
    # 'vgg16_bn': models.vgg16_bn(pretrained=pretrained),
    # 'vgg16' : models.vgg16(pretrained=pretrained),
    # 'vgg19_bn': models.vgg19_bn(pretrained=pretrained),
    # 'vgg19': models.vgg19(pretrained=pretrained),
    # 'squeezenet' : models.squeezenet1_0(pretrained=pretrained),
    # 'densenet' : models.densenet161(pretrained=pretrained),
    # 'inception_v3' : models.inception_v3(pretrained=pretrained),
    # 'googlenet' : models.googlenet(pretrained=pretrained),
    # 'shufflenet' : models.shufflenet_v2_x1_0(pretrained=pretrained),
    # 'mobilenet' : models.mobilenet_v2(pretrained=pretrained),
    # 'resnext50_32x4d' : models.resnext50_32x4d(pretrained=pretrained),
    # 'wide_resnet50_2' : models.wide_resnet50_2(pretrained=pretrained),
    # 'mnasnet' : models.mnasnet1_0(pretrained=pretrained),
    # }

    model_names = {
    'resnet18': 'fc',
    'alexnet': 'classifier',
    'vgg16_bn': 'classifier',
    'vgg16': 'classifier',
    'vgg19_bn': 'classifier',
    'vgg19': 'classifier',
    'squeezenet': 'classifier',
    'densenet': 'classifier',
    'inception_v3': 'fc',
    'googlenet': 'fc',
    'shufflenet': 'fc',
    'mobilenet': 'classifier',
    'resnext50_32x4d': 'fc',
    'wide_resnet50_2': 'fc',
    'mnasnet': 'classifier'
    }

    if pretrained:
        model_dict = {
            'resnet18': models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
            'alexnet' : models.alexnet(weights=models.AlexNet_Weights.DEFAULT),
            'vgg16_bn': models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT),
            'vgg16' : models.vgg16(weights=models.VGG16_Weights.DEFAULT),
            'vgg19_bn': models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT),
            'vgg19': models.vgg19(weights=models.VGG19_Weights.DEFAULT),
            'squeezenet' : models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.DEFAULT),
            'densenet' : models.densenet161(weights=models.DenseNet161_Weights.DEFAULT),
            'inception_v3' : models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT),
            'googlenet' : models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT),
            'shufflenet' : models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT),
            'mobilenet' : models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT),
            'resnext50_32x4d' : models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT),
            'wide_resnet50_2' : models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.DEFAULT),
            'mnasnet' : models.mnasnet1_0(weights=models.MNASNet1_0_Weights.DEFAULT),
        }
    else:
        model_dict = {
            'resnet18': models.resnet18(weights=None), # ++ 
            'alexnet' : models.alexnet(weights=None), # Error
            'vgg16_bn': models.vgg16_bn(weights=None), # Memory
            'vgg16' : models.vgg16(weights=None), 
            'vgg19_bn': models.vgg19_bn(weights=None), # Memory
            'vgg19': models.vgg19(weights=None), 
            'squeezenet' : models.squeezenet1_0(weights=None), # 
            'densenet' : models.densenet161(weights=None), # 
            'inception_v3' : models.inception_v3(weights=None), # 
            'googlenet' : models.googlenet(weights=None), # 
            'shufflenet' : models.shufflenet_v2_x1_0(weights=None), # 
            'mobilenet' : models.mobilenet_v2(weights=None), # ++
            'resnext50_32x4d' : models.resnext50_32x4d(weights=None), # 
            'wide_resnet50_2' : models.wide_resnet50_2(weights=None), # ++ (32bs = memory error)
            'mnasnet' : models.mnasnet1_0(weights=None), # ++
        }
            

    if num_channels != 3:
        for model_name, model in model_dict.items():
            if model_name in ['resnet18', 'resnext50_32x4d', 'wide_resnet50_2']:
                model.conv1.in_channels = num_channels
                model.conv1.weight.data = init_weights_default(num_channels=num_channels)
            elif model_name in ['alexnet', 'vgg16_bn', 'vgg19_bn', 'vgg19', 'vgg16', 'densenet', 'squeezenet']:
                model.features[0].in_channels = num_channels
                model.features[0].weight.data = init_weights_default(num_channels=num_channels)
            elif model_name in ['mnasnet']:
                model.layers[0].in_channels = num_channels
                model.layers[0].weight.data = init_weights_default(num_channels=num_channels)
            elif model_name == 'shufflenet':
                model.conv1[0].in_channels = 4
                model.conv1[0].weight.data = init_weights_default(num_channels=num_channels)
            elif model_name == 'inception_v3':
                model.Conv2d_1a_3x3.conv.in_channels = 4
                model.Conv2d_1a_3x3.conv.weight.data = init_weights_default(num_channels=num_channels)
            elif model_name == 'googlenet':
                model.conv1.conv.in_channels = 4
                model.conv1.conv.weight.data = init_weights_default(num_channels=num_channels)

        # Update the dictionary.
        model_dict[model_name] = model

    for model_name, model in model_dict.items():
        # Get last layer.
        last_layer_attr = model_names[model_name]
        last_layer = getattr(model, last_layer_attr)

        if isinstance(last_layer, torch.nn.Linear):
            num_ftrs = last_layer.in_features
            setattr(model, last_layer_attr, torch.nn.Linear(num_ftrs, num_classes))
        elif isinstance(last_layer, torch.nn.Sequential):
            if model_name != 'squeezenet':
                last_layer_ = last_layer[-1]
                num_ftrs = last_layer_.in_features
                model.classifier[-1] = torch.nn.Linear(num_ftrs, num_classes)
            else:
                last_layer = model.classifier
                last_layer_ = last_layer[1]
                num_ftrs = last_layer_.in_channels
                kernel_size = last_layer_.kernel_size
                model.classifier[1] = torch.nn.Conv2d(num_ftrs, num_classes, kernel_size)
        # Update the last layer.
        model_dict[model_name] = model

    return model_dict