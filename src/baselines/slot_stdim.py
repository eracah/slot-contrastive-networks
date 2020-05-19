import torch.nn as nn
import torch

class SlotSTDIMModel(nn.Module):
    def __init__(self, encoder, args, device, wandb=None):
        super().__init__()
        self.args = args
        self.losses_to_use = self.args.losses
        self.encoder = encoder
        self.device = device
        self.wandb = wandb
        self.local_len = self.encoder.feat_maps_per_slot_map
        self.global_len = self.encoder.slot_len
        self.project_to_slot_len = nn.Linear(self.local_len, self.global_len)
        self.project_local_len_to_local_len = nn.Linear(self.local_len, self.local_len)
        self.project_slot_len_slot_len = nn.Linear(self.global_len, self.global_len)


    def log(self, name, scalar):
        if self.training:
            self.wandb.log({"tr_"+ name: scalar})
        else:
            self.wandb.log({"val_"+ name: scalar})

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

        loss = 0.0
        if "hcn" in self.losses_to_use:
            loss_gl, acc_gl = self.calc_slot_global_to_local_loss(sv_t, sm_tp1)
            loss += loss_gl

            for k, v in dict(loss_gl=loss_gl, acc_gl=acc_gl).items():
                self.log(k, v)

        if "smcn" in self.losses_to_use:
            loss_ll, acc_ll = self.calc_slot_local_to_local_loss(sm_t, sm_tp1)
            loss += loss_ll

            for k, v in dict(loss_ll=loss_ll, acc_ll=acc_ll).items():
                self.log(k, v)

        if "scn" in self.losses_to_use:
            loss_sv, acc_sv = self.calc_slot_global_to_global_loss(sv_t, sv_tp1)
            loss += loss_sv

            for k, v in dict(loss_sv=loss_sv, acc_sv=acc_sv).items():
                self.log(k, v)

        if "sdl" in self.losses_to_use:
            loss_ss, acc_ss = self.calc_slot_diversity_loss_in_slot_space(sv_t, sv_tp1)
            loss += loss_ss

            for k, v in dict(loss_ss=loss_ss, acc_ss=acc_ss).items():
                self.log(k, v)

        if "smdl" in self.losses_to_use:
            loss_sm, acc_sm = self.calc_slot_diversity_loss_in_local_fmap_space(sm_t, sm_tp1)
            loss += loss_sm

            for k, v in dict(loss_sm=loss_sm, acc_ss=acc_sm).items():
                self.log(k, v)


        return loss


    def calc_slot_global_to_global_loss(self, slot_vectors1, slot_vectors2):
        """ oss 1: Does a pair of slot vectors from the same slot
                   come from consecutive (or within a small window) time steps or not?

            Arguments:
                slot_vectors1 (torch.FloatTensor) --  a batch of outputs from the slot encoder
                                             size: (batch_size, num_slots, slot_len)
                                             it's a batch of sets of slot vectors
                slot_vectors2 (torch.FloatTensor) --  a batch of outputs from the slot encoder 1 time step later
                                             size: (batch_size, num_slots, slot_len)
                                             it's a batch of sets of slot vectors
                          """

        batch_size, num_slots, slot_len = slot_vectors1.shape

        # logits: batch_size x num_slots x num_slots
        #        for each example (set of 8 slots), for each slot, dot product with every other slot at next time step


        slot_vectors1 = self.project_slot_len_slot_len(slot_vectors1) # (batch_size, num_slots, slot_len)
        slot_vectors1 = slot_vectors1.transpose(1, 0) # (num_slots, batch_size, slot_len)
        slot_vectors2 = slot_vectors2.permute(1, 2, 0) # (num_slots, slot_len, batch_size) preps for batched mat mul


        # for every set of slots in batch, for each slot in set, compute dot product of that slot with every other slot in
        # set of slots of same batch index but one time step later
        #   logits = torch.zeros(num_slots, batch_size, batch_size)
        #   for slot_index in range(num_slots):
        #       for batch_index1 in range(batch_size):
        #           for batch_index2 in range(batch_size):
        #               logit = torch.dot(slot_vectors1[slot_index, batch1_index, :], slot_vectors2[slot_index, :, batch2_index])
        #               logits[slot_index, batch1_index, batch2_index] = logit
        logits = torch.matmul(slot_vectors1, slot_vectors2) # (num_slots, batch_size, batch_size)

        # logits represents num_slot sets of batch_size different batch_size-way classification problems
        # which is represented by num_slot different batch_size x batch_size matrices
        # the correct logit for each row of the matrix is the diagonal of the matrix
        # aka the the correct logit in the ith row is the ith element, so we represent that
        # with a target variable that is a set of num_slot vectors each of value torch.arange(batch_size)
        # [0, 1, 2, ... nbatch_size-1], which represents which element in each row of the matrix is the correct answer
        target = torch.stack([torch.arange(batch_size) for i in range(num_slots)]).to(self.device)

        # flatten logits to be a large set of size batch_size logits
        # aka we now have batch_size * num_slots different batch_size-way classification problems
        inp = logits.reshape(batch_size * num_slots, -1)
        # do the same for the target now for each of the batch_size*num_slots different batch_size-way classification problems
        # we have an int label
        target = target.reshape(batch_size*num_slots,)


        loss = nn.CrossEntropyLoss()(inp, target)

        guesses = torch.argmax(inp.detach(), dim=1)
        is_correct = torch.eq(guesses, target)
        acc = 100 * torch.mean(is_correct.to(torch.float)).item()

        return loss, acc


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

        slot_maps1 = self.project_local_len_to_local_len(slot_maps1)  # (num_slots, h, w, N, num_feat_maps_per_slot)

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


    def calc_slot_diversity_loss_in_slot_space(self, slot_vectors1, slots_vectors2):
        """Loss 2:  Does a pair of vectors close in time come from the
                          same slot of different slots

            Arguments:
                slot_vectors1 (torch.FloatTensor) --  a batch of outputs from the slot encoder
                                             size: (batch_size, num_slots, slot_len)
                                             it's a batch of sets of slot vectors
                slot_vectors2 (torch.FloatTensor) --  a batch of outputs from the slot encoder 1 time step later
                                             size: (batch_size, num_slots, slot_len)
                                             it's a batch of sets of slot vectors
                          """

        batch_size, num_slots, slot_len = slot_vectors1.shape

        # logits: batch_size x num_slots x num_slots
        #        for each example (set of 8 slots), for each slot, dot product with every other slot at next time step


        slot_vectors1 = self.project_slot_len_slot_len(slot_vectors1) # (batch_size, num_slots, slot_len)
        slot_vectors2 = slots_vectors2.transpose(2, 1) # (batch_size, slot_len, num_slots) preps for batched mat mul


        # for every set of slots in batch, for each slot in set, compute dot product of that slot with every other slot in
        # set of slots of same batch index but one time step later
        #   logits = torch.zeros(batch_size, num_slots, num_slots)
        #   for batch_index in range(batch_size):
        #       for slot1_index in range(num_slots):
        #           for slot2_index in range(num_slots):
        #               logit = torch.dot(slot_vectors1[batch_index, slot1_index, :], slot_vectors2[batch_index, :, slot2_index])
        #               logits[batch_index, slot1_index, slot2_index] = logit
        logits = torch.matmul(slot_vectors1, slot_vectors2) # (batch_size, num_slots, num_slots)

        # logits represents batch_size sets of num_slot different num_slot-way classification problems
        # which is represented by batch_size different num_slot x num_slot matrices
        # the correct logit for each row of the matrix is the diagonal of the matrix
        # aka the the correct logit in the ith row is the ith element, so we represent that
        # with a target variable that is a set of batch_size vectors each of value torch.arange(num_slots)
        # [0, 1, 2, ... num_slots-1], which represents which element in each row of the matrix is the correct answer
        target = torch.stack([torch.arange(num_slots) for i in range(batch_size)]).to(self.device)

        # flatten logits to be a large set of size num_slot logits
        # aka we now have batch_size * num_slots different num_slot-way classification problems
        inp = logits.reshape(batch_size * num_slots, -1)
        # do the same for the target now for each of the batch_size*num_slots different num_slot-way classification problems
        # we have an int label
        target = target.reshape(batch_size*num_slots,)


        loss = nn.CrossEntropyLoss()(inp, target)

        guesses = torch.argmax(inp.detach(), dim=1)
        is_correct = torch.eq(guesses, target)
        acc = 100 * torch.mean(is_correct.to(torch.float)).item()

        return loss, acc


    def calc_slot_diversity_loss_in_local_fmap_space(self, slot_maps1, slot_maps2):
        """computes slot-based slot diversity local to local loss
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


                Pseudocode (non-vectorized):
                    scores = torch.zeros(N, h, w, num_slots, num_slots)
                    for batch_index in range(N):
                      for i in range(h):
                          for j in range(w):
                              for slot1_index in range(num_slots):
                                  for slot2_index in range(num_slots):
                                      score = torch.dot(slot_maps1[batch_index, i, j, slot1_index, :],
                                                        slot_maps2[batch_index, i, j, :, slot2_index])
                                      scores[batch_index, i, j, slot1_index, slot2_index] = score

                """
        N, num_slots, num_feat_maps_per_slot, h, w = slot_maps1.shape
        assert slot_maps1.shape == slot_maps2.shape

        slot_maps1 = slot_maps1.permute(0, 3, 4, 1, 2) # (N, h, w, num_slots, num_feat_maps_per_slot)
        slot_maps2 = slot_maps2.permute(0, 3, 4, 2, 1)# (N, h, w, num_slots, num_feat_maps_per_slot)

        slot_maps1 = self.project_local_len_to_local_len(slot_maps1) # shape remains (N, h, w, num_slots,
                                                                                    # num_feat_maps_per_slot)

        # for every example in batch, for every spatial location, for every set of slotmaps in slot_maps1
        # dot product with every other
        # corresponding spatial location for the same batch index in slot_maps2
        # creates N * h * w  (num_slots, num_slots) matrices, where each matrix represents num_slots num_slots-way
        # classification problems, where the correct logit for each row is the element on the diagonal
        logits = torch.matmul(slot_maps1, slot_maps2) # N, h, w, num_slots, num_slots)
        inp=logits.reshape(N * h * w * num_slots, num_slots)

        # ground truth is the index diagonal of all the N * h * w little (num_slot,num_slot) matrices
        # aka torch.arange(num_slots) repeated N * h * w * num_slots times
        target = torch.arange(num_slots).repeat(N * h * w).to(self.device)


        loss = nn.CrossEntropyLoss()(inp, target)

        guesses = torch.argmax(inp.detach(), dim=1)
        is_correct = torch.eq(guesses, target)
        acc = 100 * torch.mean(is_correct.to(torch.float)).item()

        return loss, acc




