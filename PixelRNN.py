import numpy as np
np.random.seed(123)
import torch
import torch.nn as nn
import torch.nn.functional as F
# import time
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class MaskedConv2d(nn.Conv2d):
    """
    Masked convolution. input-to-state for Row LSTM and stete-to-state for Diagonal BiLSTM.
    """
    def __init__(self, *args, mask_type, data_channels, **kwargs):

        super(MaskedConv2d, self).__init__(*args, **kwargs)

        assert mask_type in ['A', 'B'], 'Invalid mask type.'

        out_channels, in_channels, height, width = self.weight.size()
        yc, xc = height // 2, width // 2

        mask = np.zeros(self.weight.size(), dtype=np.float32)
        mask[:, :, :yc, :] = 1 # until last row
        mask[:, :, yc, :xc + 1] = 1 # until this column, current row

        def cmask(out_c, in_c):
            # channel mask
            a = (np.arange(out_channels) % data_channels == out_c)[:, None]
            b = (np.arange(in_channels) % data_channels == in_c)[None, :]
            return a * b

        for o in range(data_channels):
            for i in range(o + 1, data_channels):
                mask[cmask(o, i), yc, xc] = 0

        if mask_type == 'A':
            for c in range(data_channels):
                mask[cmask(c, c), yc, xc] = 0

        mask = torch.from_numpy(mask).float()

        self.register_buffer('mask', mask)

    def forward(self, x):
        self.weight.data *= self.mask
        x = super(MaskedConv2d, self).forward(x)
        return x


class RowLSTM(nn.Module):
    """
    One Layer of Row LSTM.
    Residual Blocks:
    i-s: 3 x 1 mask B
    s-s: 3 x 1 no mask
    """
    def __init__(self, in_channels, out_channels, data_channels):
        super(RowLSTM, self).__init__()
        self.split_size = out_channels
        self.input_to_state = MaskedConv2d(in_channels, out_channels, kernel_size = (1, 3),
                                 data_channels=data_channels, mask_type= 'B', padding = (0, 3//2))
        self.rnn = nn.LSTM(out_channels, hidden_size = out_channels, batch_first = True, )
        self.state_to_state = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding = (0, 3//2))

    def step_fn(self, x, h, c):
        h_ = h.unsqueeze(2).contiguous().permute(0, 3, 2, 1) # N x C x H(1) x W
        h_ = self.state_to_state(h_)
        h_ = h_.squeeze(2).contiguous().permute(0, 2, 1) # N x W x C
        h_ = h_.view(-1, 1, self.split_size)
        curr_input_to_state = x + h_ # K^ss \conv h_{t-1} + K^is \conv x_t


        # LSTM
        # o_f_i = F.sigmoid(curr_input_to_state[:, :, :3*self.split_size])
        # o, f, i = o_f_i.chunk(3, dim = -1) # output, forget, input gates
        # g = F.tanh(curr_input_to_state[:, :, 3*self.split_size:]) # cell gate
        # c = f * c + i * g
        # h  = o * F.tanh(c)
        out, (h, c) = self.rnn(curr_input_to_state, (torch.zeros_like(h), c))

        return h, c


    def forward(self, x):
        batch_size, channels, height, width = x.shape
        x = self.input_to_state(x)

        h = torch.zeros(1, batch_size * width, self.split_size, device=x.device)
        c = torch.zeros(1, batch_size * width, self.split_size, device=x.device)
        outputs = []

        for i in range(height):
            x_ = x[:, :, i, :]
            x_ = x_.permute(0, 2, 1).contiguous().view(-1, 1, self.split_size)
            h, c = self.step_fn(x_, h, c)  # b x w x c

            outputs += [h.contiguous().view(batch_size, self.split_size, 1, width)]

        return torch.cat(outputs, dim = 2) # if layers


class DiagonalBiLSTM(nn.Module):
    """
    One Layer of Diagonal BiLSTM.
    Residual Blocks:
    i-s: 1 x 1 mask B
    s-2: 1 x 2 no mask
    """
    def __init__(self, in_channels, out_channels, data_channels):
        super(DiagonalBiLSTM, self).__init__()
        self.split_size = out_channels
        self.input_to_state = MaskedConv2d(in_channels, out_channels, kernel_size = (1, 1),
                                 data_channels=data_channels, mask_type= 'B')
        self.rnn = nn.LSTM(out_channels, hidden_size = out_channels, batch_first = True,)
        self.state_to_state = nn.Conv2d(out_channels, out_channels, kernel_size=(2, 1), padding = 'same')

    def shift(self, x):
        batch_size, channels, height, width = x.shape
        temp = torch.zeros((batch_size, channels, height, 2*width - 1), device=x.device)
        for i in range(height):
            temp[:, :, i, i:i+width] = x[:, :, i, :]
        return temp

    def unshift(self, x):
        batch_size, channels, height, width  = x.shape
        width_ = (width + 1)//2
        temp = torch.stack([x[:, :, i, i:i+width_] for i in range(height)], dim = 2)
        return temp


    def step_fn(self, x, h, c):
        # column-wise
        h_ = h.unsqueeze(3).contiguous().permute(0, 2, 1, 3) # N x C x H x W(1)
        h_ = self.state_to_state(h_)

        h_ = h_.squeeze(3).contiguous().permute(0, 2, 1) # N x H x C
        h_ = h_.view(-1, 1, self.split_size) # N*H x 1 x C
        curr_input_to_state = x + h_ # K^ss \conv h_{t-1} + K^is \conv x_t


        # LSTM
        # o_f_i = F.sigmoid(curr_input_to_state[:, :, :3*self.split_size])
        # o, f, i = o_f_i.chunk(3, dim = -1) # output, forget, input gates
        # g = F.tanh(curr_input_to_state[:, :, 3*self.split_size:]) # cell gate
        # c = f * c + i * g
        # h  = o * F.tanh(c)
        out, (h, c) = self.rnn(curr_input_to_state, (torch.zeros_like(h), c))
        return h, c


    def DiagonalLSTM(self, x):
        batch_size, channels, height, width = x.shape
        x = self.input_to_state(self.shift(x))
        width_ = 2 * width - 1
        h = torch.zeros(1, batch_size * height, self.split_size, device=x.device)
        c = torch.zeros(1, batch_size * height, self.split_size, device=x.device)
        outputs = []

        for i in range(width_):
            x_ = x[:, :, :, i]
            x_ = x_.permute(0, 2, 1).contiguous().view(-1, 1, self.split_size)
            h, c = self.step_fn(x_, h, c)

            outputs += [h.contiguous().view(batch_size, self.split_size, height, 1)]
        outputs = torch.cat(outputs, dim = 3)
        return self.unshift(outputs)

    def forward(self, x):
        left = self.DiagonalLSTM(x)
        right = self.DiagonalLSTM(x.flip(3)).flip(3)
        right_ = torch.cat([torch.zeros((right.shape[0], right.shape[1],
                                         1, right.shape[-1]), device=right.device),
                            right[:, :, :-1, :]], dim = 2)
        return left + right_

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, data_channels):
        super(ResidualBlock, self).__init__()
        self.row_lstm = RowLSTM(in_channels, out_channels, data_channels)
        self.diagonal_bilstm = DiagonalBiLSTM(in_channels, out_channels, data_channels)
        self.upsample = nn.Conv2d(out_channels, out_channels*2, kernel_size = 1)
        # self.pre_upsample = nn.Conv2d(in_channels, in_channels*2, kernel_size = 1)

    def forward(self, x):
        # x = self.pre_upsample(x)
        x_row, x_diag = torch.split(x, x.shape[1]//2, dim = 1)
        out = self.row_lstm(x_row) + self.diagonal_bilstm(x_diag)
        out = self.upsample(out)
        return x + out

class PixelRNN(nn.Module):
    def __init__(self, in_channels, out_channels, data_channels, num_layers = 12,):
        super(PixelRNN, self).__init__()
        """
        first layer: 7x7 masked conv2d mask A
        upsample
        residual blocks: residual conneciton with rowLSTM and diagonalBiLSTM
        final layer: ReLU -> 1x1 masked conv2d mask B -> 256-way Softmax per RGB channel
        """
        self.data_channels = data_channels
        self.first_layer = nn.Sequential(MaskedConv2d(data_channels, in_channels, data_channels = data_channels,
                                                      kernel_size = 7, padding = 3, mask_type='A'),
                                         nn.Conv2d(in_channels, data_channels*out_channels*2, kernel_size = 1))

        self.residual_blocks = nn.ModuleList([ResidualBlock(data_channels * out_channels, data_channels * out_channels,
                                                            data_channels)
                                              for _ in range(num_layers)])
        self.final_layer = nn.Sequential(nn.ReLU(), MaskedConv2d(2 * out_channels * data_channels,
                                                                 256 * data_channels, kernel_size = (1, 1),
                                 data_channels=data_channels, mask_type= 'B'),)
        self.softmax = nn.Softmax(dim = 2)


    def forward(self, x):
      out = self.first_layer(x)
      for block in self.residual_blocks:
        out = block(out)
      out = self.final_layer(out)
      out = torch.stack([out[:, i::self.data_channels, :, :] for i in range(self.data_channels)], dim = 2)

      return self.softmax(out)


if __name__ == '__main__':
    # get some random training images

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'\t{classes[labels[j]]:5s}\t' for j in range(batch_size)))


    model = PixelRNN(in_channels = 3, out_channels = 128, data_channels = 3, num_layers = 12)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
