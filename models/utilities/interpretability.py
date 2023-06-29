import torch
from lucent.optvis import render, param, transform, objectives
from lucent.modelzoo import inceptionv1, resnet18
from lucent.modelzoo.util import get_model_layers
import torchvision.models as models
from IPython.display import display, HTML
import matplotlib.pyplot as plt

def alter_num_classes(model, num_classes=2):
    last_layer_attr = 'fc'
    last_layer = getattr(model, last_layer_attr)

    num_ftrs = last_layer.in_features
    setattr(model, last_layer_attr, torch.nn.Linear(num_ftrs, num_classes))
    return model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pretrained_model = resnet18(weights=models.ResNet18_Weights.DEFAULT)
pretrained_model = alter_num_classes(pretrained_model, 2)
pretrained_model.load_state_dict(torch.load('models/results/40X/weights/40X_on-air-aug_std_none_pre-resnet18_sgde-2e-4_bcew_32bs-strf_100ep_2023-06-23.pth', weights_only=True))

pretrained_model.to(device).eval()

print(get_model_layers(pretrained_model))

# Try to activate the strawberry label using CPPN parameterization
cppn_param_f = lambda: param.cppn(224)
cppn_opt = lambda params: torch.optim.SGD(params, 1e-2)

html_object = render.render_vis(pretrained_model, "labels:0", cppn_param_f, cppn_opt, show_inline=False)

# display(html_object)
render.render_vis(pretrained_model, "labels:0")