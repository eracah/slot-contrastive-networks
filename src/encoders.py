# some of this code adapted from https://github.com/mila-iqia/atari-representation-learning and https://github.com/tkipf/c-swm
import torch
import torch.nn as nn
import torch.nn.functional as F
import src.cswm_utils as utils

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

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

    def get_fmaps(self, inp):
        slot_fmaps = self.slot_conv(inp)
        return slot_fmaps


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
    def __init__(self,input_channels, slot_len, num_slots, args):
        super().__init__()
        self.slot_len = slot_len
        self.num_slots = num_slots
        self.base_encoder = NatureCNN(input_channels, args)
        inp_shape = self.base_encoder.final_conv_shape
        self.slot_addon = SlotAddOn(inp_shape, self.num_slots, self.slot_len)


    def get_fmaps(self, x):
        fmaps = self.base_encoder(x)
        slot_fmaps = self.slot_addon.get_fmaps(fmaps)
        return fmaps, slot_fmaps

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


class EncoderCNNSmall(nn.Module):
    """CNN encoder, maps observation to obj-specific feature maps."""

    def __init__(self, input_dim, hidden_dim, num_objects, act_fn='sigmoid',
                 act_fn_hid='relu'):
        super(EncoderCNNSmall, self).__init__()
        self.cnn1 = nn.Conv2d(
            input_dim, hidden_dim, (10, 10), stride=10)
        self.cnn2 = nn.Conv2d(hidden_dim, num_objects, (1, 1), stride=1)
        self.ln1 = nn.BatchNorm2d(hidden_dim)
        self.act1 = utils.get_act_fn(act_fn_hid)
        self.act2 = utils.get_act_fn(act_fn)

    def forward(self, obs):
        h = self.act1(self.ln1(self.cnn1(obs)))
        return self.act2(self.cnn2(h))


class EncoderCNNMedium(nn.Module):
    """CNN encoder, maps observation to obj-specific feature maps."""

    def __init__(self, input_dim, hidden_dim, num_objects, act_fn='sigmoid',
                 act_fn_hid='leaky_relu'):
        super(EncoderCNNMedium, self).__init__()

        self.cnn1 = nn.Conv2d(
            input_dim, hidden_dim, (9, 9), padding=4)
        self.act1 = utils.get_act_fn(act_fn_hid)
        self.ln1 = nn.BatchNorm2d(hidden_dim)

        self.cnn2 = nn.Conv2d(
            hidden_dim, num_objects, (5, 5), stride=5)
        self.act2 = utils.get_act_fn(act_fn)

    def forward(self, obs):
        h = self.act1(self.ln1(self.cnn1(obs)))
        h = self.act2(self.cnn2(h))
        return h


class EncoderCNNLarge(nn.Module):
    """CNN encoder, maps observation to obj-specific feature maps."""

    def __init__(self, input_dim, hidden_dim, num_objects, act_fn='sigmoid',
                 act_fn_hid='relu'):
        super(EncoderCNNLarge, self).__init__()

        self.cnn1 = nn.Conv2d(input_dim, hidden_dim, (3, 3), padding=1)
        self.act1 = utils.get_act_fn(act_fn_hid)
        self.ln1 = nn.BatchNorm2d(hidden_dim)

        self.cnn2 = nn.Conv2d(hidden_dim, hidden_dim, (3, 3), padding=1)
        self.act2 = utils.get_act_fn(act_fn_hid)
        self.ln2 = nn.BatchNorm2d(hidden_dim)

        self.cnn3 = nn.Conv2d(hidden_dim, hidden_dim, (3, 3), padding=1)
        self.act3 = utils.get_act_fn(act_fn_hid)
        self.ln3 = nn.BatchNorm2d(hidden_dim)

        self.cnn4 = nn.Conv2d(hidden_dim, num_objects, (3, 3), padding=1)
        self.act4 = utils.get_act_fn(act_fn)

    def forward(self, obs):
        h = self.act1(self.ln1(self.cnn1(obs)))
        h = self.act2(self.ln2(self.cnn2(h)))
        h = self.act3(self.ln3(self.cnn3(h)))
        return self.act4(self.cnn4(h))


class EncoderMLP(nn.Module):
    """MLP encoder, maps observation to latent state."""

    def __init__(self, input_dim, output_dim, hidden_dim, num_objects,
                 act_fn='relu'):
        super(EncoderMLP, self).__init__()

        self.num_objects = num_objects
        self.input_dim = input_dim

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.ln = nn.LayerNorm(hidden_dim)

        self.act1 = utils.get_act_fn(act_fn)
        self.act2 = utils.get_act_fn(act_fn)

    def forward(self, ins):
        h_flat = ins.view(-1, self.num_objects, self.input_dim)
        h = self.act1(self.fc1(h_flat))
        h = self.act2(self.ln(self.fc2(h)))
        return self.fc3(h)

