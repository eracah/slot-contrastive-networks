# some of this code adapted from https://github.com/mila-iqia/atari-representation-learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Conv2dSame(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            Conv2dSame(in_channels, out_channels, 3),
            nn.ReLU(),
            Conv2dSame(in_channels, out_channels, 3)
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = F.relu(out)
        return out




class NatureCNN(nn.Module):
    def __init__(self, input_channels, args):
        super().__init__()
        self.feature_size = args.feature_size
        self.downsample = not args.no_downsample
        self.input_channels = input_channels
        self.end_with_relu = args.end_with_relu
        self.args = args

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))
        self.flatten = Flatten()

        if self.downsample:
            self.final_conv_size = 32 * 7 * 7
            self.final_conv_shape = (32, 7, 7)
            self.main = nn.Sequential(
                init_(nn.Conv2d(input_channels, 32, 8, stride=4)),
                nn.ReLU(),
                init_(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                init_(nn.Conv2d(64, 32, 3, stride=1)),
                nn.ReLU(),
                Flatten(),
                init_(nn.Linear(self.final_conv_size, self.feature_size)),
                #nn.ReLU()
            )
        else:
            self.final_conv_size = 64 * 9 * 6
            self.final_conv_shape = (64, 9, 6)
            self.main = nn.Sequential(
                init_(nn.Conv2d(input_channels, 32, 8, stride=4)),
                nn.ReLU(),
                init_(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                init_(nn.Conv2d(64, 128, 4, stride=2)),
                nn.ReLU(),
                init_(nn.Conv2d(128, 64, 3, stride=1)),
                nn.ReLU(),
                Flatten(),
                init_(nn.Linear(self.final_conv_size, self.feature_size)),
                #nn.ReLU()
            )
        self.train()

    def forward(self, inputs):
        fmaps = self.get_fmaps(inputs, fmaps=True)['f7']
        return fmaps


    def get_fmaps(self, inputs, fmaps=False):
        f5 = self.main[:6](inputs)
        f7 = self.main[6:8](f5)
        out = self.main[8:](f7)
        if self.end_with_relu:
            assert self.args.method != "vae", "can't end with relu and use vae!"
            out = F.relu(out)
        if fmaps:
            return {
                'f5': f5,
                'f7': f7,
                'out': out
            }
        return out

class SlotAddOn(nn.Module):
    def __init__(self, inp_shape, num_slots, slot_len):
        super().__init__()
        self.num_slots = num_slots
        self.slot_len = slot_len
        num_inp_channels, h, w = inp_shape
        self.slot_conv = nn.Conv2d(num_inp_channels, num_slots, 1)
        self.slot_mlp = nn.Sequential(nn.Linear(h*w, slot_len),
                                     nn.ReLU(),
                                     nn.Linear(slot_len, slot_len) )

    def forward(self,inp):
        slot_fmaps = self.slot_conv(inp)
        slots = []
        for i in range(self.num_slots):
            slot_fmap = slot_fmaps[:, i]
            slot = self.slot_mlp(Flatten()(slot_fmap))
            slots.append(slot)
        slots = torch.stack(slots, dim=1)
        #slots = torch.cat([self.slot_fc(slot_fmap) for slot_fmap in slot_fmaps])
        return slots

#class ReverseAttentionAddOn(nn.Module):


class SlotEncoder(nn.Module):
    def __init__(self,input_channels, args):
        super().__init__()
        self.slot_len = args.slot_len
        self.num_slots = args.num_slots
        self.base_encoder = NatureCNN(input_channels, args)
        inp_shape = self.base_encoder.final_conv_shape
        self.slot_addon = SlotAddOn(inp_shape, self.num_slots, self.slot_len)

    def forward(self, x):
        fmaps = self.base_encoder(x)
        slots = self.slot_addon(fmaps)
        return slots


class SlotIWrapper(nn.Module):
    def __init__(self, slot_encoder, i):
        super().__init__()
        self.slot_encoder = slot_encoder
        self.i = i

    def forward(self,x):
        slots = self.slot_encoder(x)
        slot_i = slots[:,self.i]
        return slot_i

class ConcatenateWrapper(nn.Module):
    def __init__(self, slot_encoder):
        super().__init__()
        self.slot_encoder = slot_encoder

    def forward(self,x):
        slots = self.slot_encoder(x)
        return Flatten()(slots)

