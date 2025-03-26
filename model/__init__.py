import torch

from .cnn_classify_road import QBotCNN as CNN5ClassifyRoad
from .cnn_classify_road import image_preprocessor as load_5_cls_data

from .cnn_6_cls_road import QBotCNN as CNN6ClassifyRoad
from .cnn_6_cls_road import image_preprocessor as load_6_cls_data

from .cnn_3_cls_road import ImageClassifier3Class as CNN3ClassifyRoad
from .cnn_3_cls_road import image_preprocessor as load_3_cls_data

from .cnn_line_following import QBotCNN as CNNLineFollower
from .cnn_line_following import image_preprocessor as load_cnn_data

from .cnn_line_following2 import LineFollowerNet as CNNLineFollower2
from .cnn_line_following2 import image_preprocessor as load_cnn_data2

from .rnn_line_following import CNNRNNLineFollower
from .rnn_line_following import image_preprocessor as load_rnn_data

from .resnet import load_resnet18
from .resnet import image_preprocessor as load_resnet_data

from .cnn_turn_speed import TurnSpeedCNN
from .cnn_turn_speed import image_preprocessor as load_turnspd_data


def load_model(model_class, path, device="cpu"):
    model = model_class().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def preprocess_all(image, device):
    return {
        "3_cls": load_3_cls_data(image, device),
        "5_cls": load_5_cls_data(image, device),
        "6_cls": load_6_cls_data(image, device),
        "cnn":   load_cnn_data(image, device),
        "cnn2":  load_cnn_data2(image, device),
        "rnn":   load_rnn_data(image, device),
        "resnet": load_resnet_data(image, device),
        "tspd":  load_turnspd_data(image, device)
    }