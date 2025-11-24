import torch
import torch.nn as nn


# memo 音素とmspがあれば，f0出るはず．LSTMかGRUか1DConv
# input_size: The number of expected features in the input `x`
# hidden_size: The number of features in the hidden state `h`
# num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
#     would mean stacking two LSTMs together to form a `stacked LSTM`,
#     with the second LSTM taking in outputs of the first LSTM and
#     computing the final results. Default: 1
# bias: If ``False``, then the layer does not use bias weights
# `b_ih` and `b_hh`.
#     Default: ``True``
# batch_first: If ``True``, then the input and output tensors are provided
#     as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
#     Note that this does not apply to hidden or cell states. See the
#     Inputs/Outputs sections below for details.  Default: ``False``
# dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
#     LSTM layer except the last layer, with dropout probability equal to
#     :attr:`dropout`. Default: 0
# bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``
# proj_size: If ``> 0``, will use LSTM with projections of corresponding
# size. Default: 0


class LSTM_net(nn.Module):
    def __init__(
        self,
        n_inputs: int = 80,
        n_outputs: int = 80,
        n_layers: int = 2,
        hidden_size: int = 128,
        fc_size: int = 4096,
        dropout: float = 0.2,
        bidirectional: bool = False,
        l2softmax: bool = False,
        continuous: bool = False
    ):
        """ LSTMネットワーク

        Args:
            n_inputs (int, optional): 入力次元. Defaults to 80.
            n_outputs (int, optional): 出力次元. Defaults to 80.
            n_layers (int, optional): レイヤー数. Defaults to 2.
            hidden_size (int, optional): 隠れ層のサイズ. Defaults to 128.
            fc_size (int, optional): 全結合層のサイズ. Defaults to 4096.
            dropout (float, optional): ドロップアウト率. Defaults to 0.2.
            bidirectional (bool, optional): 双方向LSTMかどうか. Defaults to False.
            l2softmax (bool, optional): L2正規化付きソフトマックスかどうか. Defaults to False.
            continuous (bool, optional): 連続入力かどうか. Defaults to False.
        """
        self.alpha = 20.0
        self.input_size = n_inputs
        self.output_size = n_outputs
        self.hidden_size = hidden_size
        self.num_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.l2softmax = l2softmax
        self.hidden_state = None
        self.continuous = continuous

        super(LSTM_net, self).__init__()
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        )
        self.fc1 = nn.Linear(
            self.hidden_size*(1+self.bidirectional*1),
            fc_size
        )
        self.fc2 = nn.Linear(fc_size, self.output_size)
        self.mish = nn.Mish()
        # Xavierの初期化
        nn.init.xavier_normal_(self.lstm.weight_ih_l0)
        nn.init.xavier_normal_(self.lstm.weight_hh_l0)
        # Heの初期化
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x, mask=None):
        if not self.continuous:  # マイクで音声をひろうときよう 過去の記憶を保持した状態かどうかに設定
            h, self.hidden_state = self.lstm(x)
        else:
            h, self.hidden_state = self.lstm(x, self.hidden_state)
        if mask is None:
            mask = torch.ones_like(x)
        h = self.fc1(h.reshape(h.shape[0]*h.shape[1], h.shape[2]))
        h = self.mish(h)
        h = self.fc2(h)
        h = h.view(x.shape[0], x.shape[1], self.output_size)
        h = h

        if self.l2softmax:
            l2 = torch.sqrt((h**2).sum(dim=2))
            h = self.alpha * (h.T / l2.T).T

        return h


class GRU_net(nn.Module):
    def __init__(
        self,
        n_mels=80,
        n_phoneme=0,
        n_layers=2,
        hidden_size=128,
        fc_size=4096,
        dropout=0.2
    ):
        self.input_size = n_mels + n_phoneme
        self.hidden_size = hidden_size
        self.num_layers = n_layers
        self.output_size = n_phoneme
        self.dropout = dropout

        super(GRU_net, self).__init__()
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout
        )
        self.fc1 = nn.Linear(self.hidden_size, fc_size)
        self.fc2 = nn.Linear(fc_size, self.output_size)
        self.mish = nn.Mish()

    def forward(self, x):
        # x.shape == (Batch,Time,Mel)
        h, _ = self.gru(x)
        # h.shape == (Batch,Time,hidden_size)
        h = self.fc1(h.view(h.shape[0]*h.shape[1], h.shape[2]))
        h = self.mish(h)
        # h.shape == (Batch*Time, 4096)
        h = self.fc2(h)
        # h.shape == (Batch*Time, n_mels)
        return h.view(x.shape[0], x.shape[1], self.output_size)
