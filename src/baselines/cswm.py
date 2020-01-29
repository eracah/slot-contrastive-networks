import utils

import numpy as np

import torch
from torch import nn
from src.encoders import EncoderCNNSmall, EncoderCNNMedium, EncoderCNNLarge, EncoderMLP

class ContrastiveSWM(nn.Module):
    """Main module for a Contrastively-trained Structured World Model (C-SWM).

    Args:
        embedding_dim: Dimensionality of abstract state space.
        input_dims: Shape of input observation.
        hidden_dim: Number of hidden units in encoder and transition model.
        action_dim: Dimensionality of action space.
        num_objects: Number of object slots.
    """

    def __init__(self, embedding_dim, input_dims, hidden_dim, action_dim,
                 num_objects, hinge=1., sigma=0.5, encoder='large',
                 ignore_action=False, copy_action=False):
        super(ContrastiveSWM, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.num_objects = num_objects
        self.hinge = hinge
        self.sigma = sigma
        self.ignore_action = ignore_action
        self.copy_action = copy_action

        self.pos_loss = 0
        self.neg_loss = 0

        num_channels = input_dims[0]
        width_height = input_dims[1:]

        if encoder == 'small':
            self.obj_extractor = EncoderCNNSmall(
                input_dim=num_channels,
                hidden_dim=hidden_dim // 16,
                num_objects=num_objects)
            # CNN image size changes
            width_height = np.array(width_height)
            width_height = width_height // 10
        elif encoder == 'medium':
            self.obj_extractor = EncoderCNNMedium(
                input_dim=num_channels,
                hidden_dim=hidden_dim // 16,
                num_objects=num_objects)
            # CNN image size changes
            width_height = np.array(width_height)
            width_height = width_height // 5
        elif encoder == 'large':
            self.obj_extractor = EncoderCNNLarge(
                input_dim=num_channels,
                hidden_dim=hidden_dim // 16,
                num_objects=num_objects)

        self.obj_encoder = EncoderMLP(
            input_dim=np.prod(width_height),
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_objects=num_objects)

        self.transition_model = TransitionGNN(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            num_objects=num_objects,
            ignore_action=ignore_action,
            copy_action=copy_action)

        self.width = width_height[0]
        self.height = width_height[1]

    def energy(self, state, action, next_state, no_trans=False):
        """Energy function based on normalized squared L2 norm."""

        norm = 0.5 / (self.sigma ** 2)

        if no_trans:
            diff = state - next_state
        else:
            pred_trans = self.transition_model(state, action)
            diff = state + pred_trans - next_state

        return norm * diff.pow(2).sum(2).mean(1)

    def transition_loss(self, state, action, next_state):
        return self.energy(state, action, next_state).mean()

    def contrastive_loss(self, obs, action, next_obs):

        objs = self.obj_extractor(obs)
        next_objs = self.obj_extractor(next_obs)

        state = self.obj_encoder(objs)
        next_state = self.obj_encoder(next_objs)

        # Sample negative state across episodes at random
        batch_size = state.size(0)
        perm = np.random.permutation(batch_size)
        neg_state = state[perm]

        self.pos_loss = self.energy(state, action, next_state)
        zeros = torch.zeros_like(self.pos_loss)

        self.pos_loss = self.pos_loss.mean()
        self.neg_loss = torch.max(
            zeros, self.hinge - self.energy(
                state, action, neg_state, no_trans=True)).mean()

        loss = self.pos_loss + self.neg_loss

        return loss

    def forward(self, obs):
        return self.obj_encoder(self.obj_extractor(obs))


class TransitionGNN(torch.nn.Module):
    """GNN-based transition function."""

    def __init__(self, input_dim, hidden_dim, action_dim, num_objects,
                 ignore_action=False, copy_action=False, act_fn='relu'):
        super(TransitionGNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_objects = num_objects
        self.ignore_action = ignore_action
        self.copy_action = copy_action

        if self.ignore_action:
            self.action_dim = 0
        else:
            self.action_dim = action_dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            utils.get_act_fn(act_fn),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            utils.get_act_fn(act_fn),
            nn.Linear(hidden_dim, hidden_dim))

        node_input_dim = hidden_dim + input_dim + self.action_dim

        self.node_mlp = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            utils.get_act_fn(act_fn),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            utils.get_act_fn(act_fn),
            nn.Linear(hidden_dim, input_dim))

        self.edge_list = None
        self.batch_size = 0

    def _edge_model(self, source, target, edge_attr):
        del edge_attr  # Unused.
        out = torch.cat([source, target], dim=1)
        return self.edge_mlp(out)

    def _node_model(self, node_attr, edge_index, edge_attr):
        if edge_attr is not None:
            row, col = edge_index
            agg = utils.unsorted_segment_sum(
                edge_attr, row, num_segments=node_attr.size(0))
            out = torch.cat([node_attr, agg], dim=1)
        else:
            out = node_attr
        return self.node_mlp(out)

    def _get_edge_list_fully_connected(self, batch_size, num_objects, cuda):
        # Only re-evaluate if necessary (e.g. if batch size changed).
        if self.edge_list is None or self.batch_size != batch_size:
            self.batch_size = batch_size

            # Create fully-connected adjacency matrix for single sample.
            adj_full = torch.ones(num_objects, num_objects)

            # Remove diagonal.
            adj_full -= torch.eye(num_objects)
            self.edge_list = adj_full.nonzero()

            # Copy `batch_size` times and add offset.
            self.edge_list = self.edge_list.repeat(batch_size, 1)
            offset = torch.arange(
                0, batch_size * num_objects, num_objects).unsqueeze(-1)
            offset = offset.expand(batch_size, num_objects * (num_objects - 1))
            offset = offset.contiguous().view(-1)
            self.edge_list += offset.unsqueeze(-1)

            # Transpose to COO format -> Shape: [2, num_edges].
            self.edge_list = self.edge_list.transpose(0, 1)

            if cuda:
                self.edge_list = self.edge_list.cuda()

        return self.edge_list

    def forward(self, states, action):

        cuda = states.is_cuda
        batch_size = states.size(0)
        num_nodes = states.size(1)

        # states: [batch_size (B), num_objects, embedding_dim]
        # node_attr: Flatten states tensor to [B * num_objects, embedding_dim]
        node_attr = states.view(-1, self.input_dim)

        edge_attr = None
        edge_index = None

        if num_nodes > 1:
            # edge_index: [B * (num_objects*[num_objects-1]), 2] edge list
            edge_index = self._get_edge_list_fully_connected(
                batch_size, num_nodes, cuda)

            row, col = edge_index
            edge_attr = self._edge_model(
                node_attr[row], node_attr[col], edge_attr)

        if not self.ignore_action:

            if self.copy_action:
                action_vec = utils.to_one_hot(
                    action, self.action_dim).repeat(1, self.num_objects)
                action_vec = action_vec.view(-1, self.action_dim)
            else:
                action_vec = utils.to_one_hot(
                    action, self.action_dim * num_nodes)
                action_vec = action_vec.view(-1, self.action_dim)

            # Attach action to each state
            node_attr = torch.cat([node_attr, action_vec], dim=-1)

        node_attr = self._node_model(
            node_attr, edge_index, edge_attr)

        # [batch_size, num_nodes, hidden_dim]
        return node_attr.view(batch_size, num_nodes, -1)


def train():
    obs = train_loader.__iter__().next()[0]
    input_shape = obs[0].size()

    model = modules.ContrastiveSWM(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        action_dim=args.action_dim,
        input_dims=input_shape,
        num_objects=args.num_objects,
        sigma=args.sigma,
        hinge=args.hinge,
        ignore_action=args.ignore_action,
        copy_action=args.copy_action,
        encoder=args.encoder).to(device)

    model.apply(utils.weights_init)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate)

    if args.decoder:
        if args.encoder == 'large':
            decoder = modules.DecoderCNNLarge(
                input_dim=args.embedding_dim,
                num_objects=args.num_objects,
                hidden_dim=args.hidden_dim // 16,
                output_size=input_shape).to(device)
        elif args.encoder == 'medium':
            decoder = modules.DecoderCNNMedium(
                input_dim=args.embedding_dim,
                num_objects=args.num_objects,
                hidden_dim=args.hidden_dim // 16,
                output_size=input_shape).to(device)
        elif args.encoder == 'small':
            decoder = modules.DecoderCNNSmall(
                input_dim=args.embedding_dim,
                num_objects=args.num_objects,
                hidden_dim=args.hidden_dim // 16,
                output_size=input_shape).to(device)
        decoder.apply(utils.weights_init)
        optimizer_dec = torch.optim.Adam(
            decoder.parameters(),
            lr=args.learning_rate)

    # Train model.
    print('Starting model training...')
    step = 0
    best_loss = 1e9

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0

        for batch_idx, data_batch in enumerate(train_loader):
            data_batch = [tensor.to(device) for tensor in data_batch]
            optimizer.zero_grad()

            if args.decoder:
                optimizer_dec.zero_grad()
                obs, action, next_obs = data_batch
                objs = model.obj_extractor(obs)
                state = model.obj_encoder(objs)

                rec = torch.sigmoid(decoder(state))
                loss = F.binary_cross_entropy(
                    rec, obs, reduction='sum') / obs.size(0)

                next_state_pred = state + model.transition_model(state, action)
                next_rec = torch.sigmoid(decoder(next_state_pred))
                next_loss = F.binary_cross_entropy(
                    next_rec, next_obs,
                    reduction='sum') / obs.size(0)
                loss += next_loss
            else:
                loss = model.contrastive_loss(*data_batch)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if args.decoder:
                optimizer_dec.step()

            if batch_idx % args.log_interval == 0:
                print(
                    'Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data_batch[0]),
                        len(train_loader.dataset),
                               100. * batch_idx / len(train_loader),
                               loss.item() / len(data_batch[0])))

            step += 1

        avg_loss = train_loss / len(train_loader.dataset)
        print('====> Epoch: {} Average loss: {:.6f}'.format(
            epoch, avg_loss))

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_file)