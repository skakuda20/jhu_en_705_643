import torch.nn as nn
from torchvision import models

# utility for replacing the original fc layer in ResNet backbone
# reference: https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class LRCN(nn.Module):
    def __init__(self, hidden_size, n_layers, dropout_rate, n_classes, pretrained=True, cnn_model='resnet34'):
        super(LRCN, self).__init__()

        # set up ResNet backbone as 2D CNN feature extractors
        if cnn_model=='resnet18':
            base_cnn = models.resnet18(pretrained=pretrained)
        elif cnn_model=='resnet34':
            base_cnn = models.resnet34(pretrained=pretrained)
        elif cnn_model=='resnet50':
            base_cnn = models.resnet50(pretrained=pretrained)
        elif cnn_model=='resnet101':
            base_cnn = models.resnet101(pretrained=pretrained)
        elif cnn_model=='resnet152':
            base_cnn = models.resnet34(pretrained=pretrained)
        else:
            raise ValueError('The input CNN backbone is not supported, please choose ResNet....')

        num_features = base_cnn.fc.in_features
        base_cnn.fc = Identity()
        self.base_model = base_cnn
        self.rnn = nn.LSTM(num_features, hidden_size, n_layers)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        bs, ts, c, h, w = x.shape                       # batch_size, time_steps, channel, height, width
        idx = 0
        y = self.base_model((x[:, idx]))
        _, (hn, cn) = self.rnn(y.unsqueeze(1))
        for idx in range(1, ts):
            y = self.base_model((x[:, idx]))
            out, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))
        out = self.dropout(out[:, -1])
        out = self.fc(out)
        return out