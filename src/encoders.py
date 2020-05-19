# some of this code adapted from https://github.com/mila-iqia/atari-representation-learning and https://github.com/tkipf/c-swm
import torch
import torch.nn as nn
import torch.nn.functional as F
import src.cswm_utils as utils
import numpy as np

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


class STDIMEncoder(nn.Module):
    def __init__(self, input_channels, global_vector_len):
        super().__init__()
        self.global_vector_len = global_vector_len
        self.final_conv_size = 64 * 9 * 6
        self.final_conv_shape = (64, 9, 6)



        self.layers = nn.Sequential(
            init_(nn.Conv2d(input_channels, 32, 8, stride=4)), # f1
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), # f3
            nn.ReLU(),
            init_(nn.Conv2d(64, 128, 4, stride=2)), # f5
            nn.ReLU(),
            init_(nn.Conv2d(128, 64, 3, stride=1)), #f7
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(self.final_conv_size, self.global_vector_len)
                  )
        )

    @property
    def local_vector_len(self):
        return self.layers[4].out_channels

    def get_f5(self, x):
        return self.layers[:5](x)

    def get_f7(self, x):
        return self.layers[:7](x)

    def f5_to_f7(self, f5):
        return self.layers[5:7](f5)

    def f5_to_global_vec(self, f5):
        return self.layers[5:](f5)

    def f7_to_global_vec(self, f7):
        return self.layers[7:](f7)

    def forward(self, x):
        global_vec = self.layers(x)
        return global_vec


class SlotSTDIMEncoder(STDIMEncoder):
    def __init__(self, input_channels, ablations=[], num_slots=8, slot_len=32):
        global_vector_len = 1 # for the last layer of st-dim which we don't use
        super().__init__(input_channels,
                         global_vector_len=global_vector_len)
        self.num_slots = num_slots
        self.slot_len = slot_len
        self.feat_maps_per_slot_map = super().local_vector_len // num_slots
        self.final_conv_shape = [ self.feat_maps_per_slot_map // 2, *self.final_conv_shape[1:]]
        self.final_conv_size = np.prod(self.final_conv_shape)

        self.slot_layers = nn.Sequential(
            nn.Conv2d(self.feat_maps_per_slot_map, self.feat_maps_per_slot_map // 2, 3, stride=1),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(self.final_conv_size, self.slot_len)
                  )
        )



    def forward(self, x):
        slot_maps = self.get_slot_maps(x)
        slots = self.slot_maps_to_slots(slot_maps)
        return slots

    def get_slot_maps(self, x):
        f5 = super().get_f5(x)
        bs, num_channels, h, w = f5.shape
        # in order for num_channels to evenly divide by num_slots
        # we subtract the remainder
        chopped_num_channels = num_channels - num_channels % self.num_slots
        reduced_f5 = f5[:, :chopped_num_channels]
        slot_maps = reduced_f5.reshape(bs, self.num_slots, -1, h, w)
        return slot_maps

    def slot_maps_to_slots(self, slot_maps):
        bs, num_slots, feat_maps_per_slot_map, h, w = slot_maps.shape

        # we resize so that the batch is essentially all slot maps for all examples in the batch
        # this helps us broadcast the next layers
        slot_map_4d = slot_maps.reshape(bs*num_slots, self.feat_maps_per_slot_map, h, w)

        # for each slot_map apply same conv layer + flatten + fc
        all_slots = self.slot_layers(slot_map_4d)

        # now we have number of examples by number of slots by slot_len
        slots = all_slots.reshape(bs, num_slots, -1)
        return slots


class CSWMEncoder(nn.Module):
    def __init__(self,input_dim, width_height, output_dim, hidden_dim, num_objects, act_fn='relu'):
        super().__init__()
        self.base_cnn = EncoderCNNMedium(input_dim, hidden_dim // 16, num_objects)
        self.mlp = EncoderMLP(np.prod(width_height // 5), output_dim, hidden_dim, num_objects, act_fn)

    def forward(self, x):
        return self.mlp(self.base_cnn(x))

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






