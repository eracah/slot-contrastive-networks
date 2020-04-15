import torch.nn as nn
import torch
from .stdim import STDIMModel

class SlotSTDIMModel(STDIMModel):
    def __init__(self, encoder, args, global_vector_len, device=torch.device('cpu'), wandb=None):
        super().__init__(encoder, args, global_vector_len, device, wandb)
        self.encoder = encoder
        self.local_len = self.encoder.feat_maps_per_slot_map
        self.global_len = self.encoder.slot_len
        self.project_to_slot_len = nn.Linear(self.local_len, self.global_len)
        self.slot_len_to_slot_len = nn.Linear(self.local_len, self.local_len)

    def calc_loss(self, xt, a, xtp1):
        """ Compute loss slot-stdim loss and log it.

        Arguments:
            xt (torch.FloatTensor) -- batch of random frames
                                    size: (batch_size, 3, height, width)
            a (torch.LongTensor) -- 1-D tensor of actions taken at time t (not used except for C-SWM;
                                    size (batch_size,)
            xtp1 (torch.FloatTensor) -- batch of frames one time step later (t+1) than each frame in xt
                                        size: (batch_size, 3, height, width)

        Returns:
            loss (torch.Float) -- the overall loss for current batch of xt and xtp1
        """
        sv_t, sv_tp1 = self.encoder(xt), self.encoder(xtp1)
        sm_t, sm_tp1 = self.encoder.get_slot_maps(xt), self.encoder.get_slot_maps(xtp1)

        loss_gl, acc_gl = self.calc_slot_global_to_local_loss(sv_t, sm_tp1)
        loss_ll, acc_ll = self.calc_slot_local_to_local_loss(sm_t, sm_tp1)

        loss = loss_gl + loss_ll

        for k, v in dict(loss1=loss_gl, loss2=loss_ll, acc1=acc_gl, acc2=acc_ll).items():
            super().log(k, v)

        return loss

    def calc_slot_global_to_local_loss(self, slot_vectors, slot_maps):
        """ Compute slot-based global to local loss.

        Arguments:
        slot_vectors (torch.FloatTensor) --  a batch of outputs from the slot encoder
                                             size: (batch_size, num_slots, slot_len)
                                             it's a batch of sets of slot vectors

        slot_maps (torch.FloatTensor) -- a batch of intermediate layer feature maps from the encoder
                                        (segregated into num_slots groups of num_feat_maps_per_slot)
                                        size: (batch_size, num_slots, num_feat_maps_per_slot,
                                               height_feat_map, width_feat_map)

        Returns:
                loss (torch.float): the loss
                acc (torch.float): the contrastive accuracy
        """
        # N = batch_size
        N, num_slots, slot_len = slot_vectors.shape
        N, num_slots, num_feat_maps_per_slot, h, w = slot_maps.shape  # h,w are height and width of each feature map

        # permute tensor to make num_slots b/c logically the outermost loop is over slots
        slot_vectors = slot_vectors.transpose(1, 0)  # (num_slots, N, slot_len)
        slot_maps = slot_maps.transpose(1, 0)  # (num_slots, N, num_feat_maps_per_slot, h, w)

        # permute slot_map tensor to be num_slots,h,w,N,num_feat_maps_per_slot because
        # the next outermost loop is spatial dims of feature maps
        # also we want num_feat_maps_per_slot dimension to be outermost dimension for projection
        slot_maps = slot_maps.permute(0, 3, 4, 1, 2)  # (num_slots, h, w, N, num_feat_maps_per_slot)

        # projects the depth at each spatial dimension of the slot_maps to be the same length as a global slot vector
        slot_maps = self.project_to_slot_len(slot_maps)  # (num_slots, h, w, N, slot_len)

        ## Prep for the big matrix multiplication

        # insert dummy dimensions at dimensions 1 and 2 so we can broadcast the height and width from the slot_maps
        slot_vectors = slot_vectors.unsqueeze(1).unsqueeze(2)  # (num_slots, 1, 1, N, slot_len)

        # transpose so batch_size dimension is outermost one and slot_len one is second-outermost to facilitate a
        # batch matrix multiply
        slot_vectors = slot_vectors.transpose(3, 4)  # (num_slots, 1, 1, slot_len, N)

        ## The BIG Matrix Multiplication
        # (num_slots, h, w, N, slot_len) @ (num_slots, 1, 1, slot_len, N) -> (num_slots, h, w, N, N)
        #
        # pseudocode:
        # ------------
        #
        #   scores = torch.zeros(num_slots, h, w, N, N)
        #   for slot in range(num_slots):
        #       for i in range(h):
        #           for j in range(w):
        #               for sm_ind in range(N):
        #                   for sv_index in range(N):
        #                       score = torch.dot(slot_maps[slot, i, j, sm_ind, :], slot_vector[slot, 0, 0, :, sv_ind])
        #                       scores[slot, i, j, sm_ind, sv_ind] = score
        #
        # for every slot, for every (h,w) spatial dimension of feature maps, for every local depth vector
        # of slot_map at that spatial index in batch compute the dot product with every slot_vector in the batch
        # this gives a sort-of score of every local slot_map vector with every global slot_vector
        # output shape is (num_slots, h, w, N, N) because it's dot-product of each local slot_map vectors in the batch
        # with every slot_vector in the batch
        scores = torch.matmul(slot_maps, slot_vectors)

        # the scores tensor represents unnormalized logtits of num_slots * h * w * N  N-way classification problems
        # so we can actually just flatten this tensor to be (num_slots * h * w * N, N)
        inp = scores.reshape(-1, N)

        # we can also look at "scores" as num_slots * h * w NxN score matrices
        # the diagonals of each of these little score matrices represent the dot product of a
        # slot_map vector with a slot vector from the same batch index (this means they either come from the same frame or
        # a temporally local frame (e.g. the frame before or after)
        # the off-diagonals are the dot product between a slot_map vector and a slot vector from random examples
        # so in a temporally contrastive tasks, the "correct" or positive answer is the temporally close pair
        # and the incorrect or "negative: answer is a random pair of examples,
        # so diagonals = correct and off-diagonal = incorrect
        # so the target is just the index of the diagonal
        # so if we each NxN matrix is N N-way classification problems
        # the the correct class to classification problem number 0 is 0, to problem number 1 answer is 1,
        # problem number N-1 is N-1
        # so the overall target is just torch.range(N) repeated num_slots * h * w times
        target = torch.arange(N).repeat(num_slots * h * w).to(self.device)

        # the loss
        loss = nn.CrossEntropyLoss()(inp, target)

        # guesses
        # to compute contrastive accuracy, we can just get the argmax of each score and compare that with the target
        guesses = torch.argmax(inp.detach(), dim=1)
        is_correct = torch.eq(guesses, target)
        acc = 100 * torch.mean(is_correct.to(torch.float)).item()

        return loss, acc

    def calc_slot_local_to_local_loss(self, slot_maps1, slot_maps2):
        """

        computes slot-based local to local loss
        Arguments:
            slot_maps1 (torch.FloatTensor) -- a batch of intermediate layer feature maps from the encoder
                                             (segregated into num_slots groups of num_feat_maps_per_slot)
                                              size: (batch_size, num_slots, num_feat_maps_per_slot,
                                                    height_feat_map, width_feat_map)

           slot_maps2 (torch.FloatTensor) -- a batch of intermediate layer feature maps from the encoder
                                             (segregated into num_slots groups of num_feat_maps_per_slot)
                                              size: (batch_size, num_slots, num_feat_maps_per_slot,
                                                    height_feat_map, width_feat_map)

        Returns:
                loss (Torch.float): the loss
                acc (Torch.float): the contrastive accuracy
        """
        N, num_slots, num_feat_maps_per_slot, h, w = slot_maps1.shape
        assert slot_maps1.shape == slot_maps2.shape

        slot_maps1 = slot_maps1.transpose(1, 0)  # (num_slots, N, num_feat_maps_per_slot, h, w)
        slot_maps2 = slot_maps2.transpose(1, 0)  # (num_slots, N, num_feat_maps_per_slot, h, w)

        # permute slot_map tensor to be num_slots,h,w,N,num_feat_maps_per_slot because
        # the next outermost loop is spatial dims of feature maps
        # also we want num_feat_maps_per_slot dimension to be outermost dimension for projection
        slot_maps1 = slot_maps1.permute(0, 3, 4, 1, 2)  # (num_slots, h, w, N, num_feat_maps_per_slot)
        slot_maps2 = slot_maps2.permute(0, 3, 4, 1, 2)  # (num_slots, h, w, N, num_feat_maps_per_slot)

        slot_maps1 = self.slot_len_to_slot_len(slot_maps1)  # (num_slots, h, w, N, num_feat_maps_per_slot)

        # prep for matmul
        slot_maps2 = slot_maps2.transpose(3, 4)  # (num_slots, h, w,  num_feat_maps_per_slot, N)

        scores = torch.matmul(slot_maps1, slot_maps2)  # (num_slots, h, w, N, N)

        inp = scores.reshape(-1, N)

        target = torch.arange(N).repeat(num_slots * h * w).to(self.device)

        # the loss
        loss = nn.CrossEntropyLoss()(inp, target)

        # guesses
        # to compute contrastive accuracy, we can just get the argmax of each score and compare that with the target
        guesses = torch.argmax(inp.detach(), dim=1)
        is_correct = torch.eq(guesses, target)
        acc = 100 * torch.mean(is_correct.to(torch.float)).item()

        return loss, acc
