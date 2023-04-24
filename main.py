from models import resnet
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

if __name__ == "__main__":
    print("Hello world!")
    modelFeatures = resnet.resnet18_features()
    print("Features to extract from ResNet18:")
    print(modelFeatures)

    print("--------------------------------------------------------------------------------------------- \n",
          "ResNet18 model originally:")

    modelWhole = resnet18()
    print(modelWhole)