from .cnn_classify_road import QBotCNN as CNNClassifyRoad
from .cnn_classify_road import image_preprocessor as load_classify_data

from .cnn_line_following import QBotCNN as CNNLineFollower
from .cnn_line_following import image_preprocessor as load_cnn_data

from .cnn_line_following2 import LineFollowerNet as CNNLineFollower2
from .cnn_line_following2 import image_preprocessor as load_cnn_data2

from .rnn_line_following import CNNRNNLineFollower
from .rnn_line_following import image_preprocessor as load_rnn_data