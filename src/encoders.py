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
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)


class SlotFlatten(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.view(x.size(0), x.size(1), -1)


class ConcatenateSlots(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.flatten = Flatten()
    def forward(self, x):
        return self.flatten(self.encoder(x))


init_ = lambda m: init(m,
       nn.init.orthogonal_,
       lambda x: nn.init.constant_(x, 0),
       nn.init.calculate_gain('relu'))

class NatureFMapEncoder(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        self.input_channels = input_channels

        self.flatten = Flatten()


        self.encode_to_f5 = nn.Sequential(
            init_(nn.Conv2d(input_channels, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 128, 4, stride=2)),
            nn.ReLU()
        )
        self.train()


    @property
    def local_layer_depth(self):
        return self.encode_to_f5[4].out_channels

    def forward(self, inputs):
        f5 = self.encode_to_f5(inputs)
        return f5

class NatureCNN(nn.Module):
    def __init__(self, input_channels, embedding_dim):
        super().__init__()
        self.fmap_encoder = NatureFMapEncoder(input_channels)
        self.embedding_dim = embedding_dim
        self.final_conv_size = 64 * 9 * 6
        self.final_conv_shape = (64, 9, 6)

        self.conv_to_f7 = nn.Sequential(
            init_(nn.Conv2d(128, 64, 3, stride=1)),
            nn.ReLU()
        )

        self.f7_to_global_vector = nn.Sequential(
            Flatten(),
            init_(nn.Linear(self.final_conv_size, self.embedding_dim))
        )
    def forward(self, x):
        fmaps = self.fmap_encoder(x)
        global_vec = self.f7_to_global_vector(self.conv_to_f7(fmaps))
        return global_vec


class NatureObjFMapEncoder(nn.Module):
    def __init__(self, num_slots):
        super().__init__()
        self.num_slots = num_slots
        self.conv_to_f7 = nn.Sequential(
            init_(nn.Conv2d(128, 64, 3, stride=1)),
            nn.ReLU()
        )
        self.slot_conv = nn.Sequential(
                            nn.Conv2d(64, num_slots, 1),
                            nn.ReLU()
        )

    def forward(self, fmaps):
        slot_fmaps = self.slot_conv(self.conv_to_f7(fmaps))
        return slot_fmaps

class NaureCNNObjExtractor(nn.Module):
    def __init__(self, input_dim, num_slots):
        super().__init__()
        self.cnn = NatureFMapEncoder(input_dim)
        self.fmap2objfmaps = NatureObjFMapEncoder(num_slots)
        self.num_slots = num_slots
        self.final_fmap_shape = (9, 6)

    def forward(self, input):
        slot_fmaps = self.fmap2objfmaps(self.cnn(input))
        return slot_fmaps


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


class CSWMSlotEncoder(nn.Module):
    def __init__(self, base_cnn, input_dim, output_dim, hidden_dim, num_objects, act_fn='relu' ):
        super().__init__()
        self.base_cnn = base_cnn
        self.mlp = EncoderMLP(input_dim, output_dim, hidden_dim, num_objects, act_fn)

    def forward(self, x):
        self.mlp(self.base_cnn(x))



