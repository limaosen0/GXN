from __future__ import print_function

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class MLPRegression(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLPRegression, self).__init__()

        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, 1)

    def forward(self, x, y = None):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)

        pred = self.h2_weights(h1)

        if y is not None:
            y = Variable(y)
            mse = F.mse_loss(pred, y)
            mae = F.l1_loss(pred, y)
            return pred, mae, mse
        else:
            return pred

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, with_dropout=False):
        super(MLPClassifier, self).__init__()

        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, num_class)
        self.with_dropout = with_dropout

    def forward(self, x, y = None):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)
        if self.with_dropout:
            h1 = F.dropout(h1, training=self.training)

        logits = self.h2_weights(h1)
        logits = F.log_softmax(logits, dim=1)

        if y is not None:
            loss = F.nll_loss(logits, y)

            pred = logits.data.max(1, keepdim=True)[1]
            acc = pred.eq(y.data.view_as(pred)).cpu().sum().item() / float(y.size()[0])
            return logits, loss, acc
        else:
            return logits
