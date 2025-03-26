from torchvision import models, transforms
from PIL import Image
import torch
import torchvision.models as models
import torch.nn as nn

def load_resnet18(path, output_dim=None, device="cpu"):
    model_state_dict = torch.load(path, map_location=device)
    model_state_dict = remove_prefix(model_state_dict, prefix="resnet.")
    model = models.resnet18(weights=None)
    if output_dim:   model.fc = nn.Linear(512, output_dim)
    model.load_state_dict(model_state_dict)
    model.eval()

    return model


def remove_prefix(state_dict, prefix="resnet."):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_state_dict[k[len(prefix):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def image_preprocessor(image, device="cpu"):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    # 图像预处理  
    transform = transforms.Compose([
    transforms.Resize((224, 224)),            
    transforms.Grayscale(num_output_channels=3), 
    transforms.ToTensor(),                    
    transforms.Normalize(                   
        mean=[0.485, 0.456, 0.406],           
        std=[0.229, 0.224, 0.225]
    )
    ])
    return transform(image).unsqueeze(0).to(device)
