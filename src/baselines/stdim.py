import torch.nn as nn
import torch.nn.functional as F
import torch
from src.utils import calculate_accuracy

class STDIMModel(nn.Module):
    def __init__(self, encoder, args, global_vector_len, device=torch.device('cpu'), wandb=None):
        super().__init__()
        self.encoder = encoder
        self.fmap_encoder = encoder.get_f5
        self.global_vector_len = global_vector_len
        self.wandb = wandb
        self.args = args

        self.score_fxn1 = nn.Linear(self.global_vector_len, self.encoder.local_vector_len) # x1 = global, x2=patch, n_channels = 32
        self.score_fxn2 = nn.Linear(self.encoder.local_vector_len, self.encoder.local_vector_len)
        self.device = device


    def calc_global_to_local(self, global_t, local_tp1):
        N, sy, sx, d = local_tp1.shape
        # Loss 1: Global at time t, f5 patches at time t+1
        glob_score = self.score_fxn1(global_t)
        local_flattened = local_tp1.reshape(-1, d)
        # [N*sy*sx, d] @  [d, N] = [N*sy*sx, N ] -> dot product of every global vector in batch with local voxel at all spatial locations for all examples in the batch
        # then reshape to sy*sx, N, N then to sy*sx*N, N
        logits1 = torch.matmul(local_flattened, glob_score.t()).reshape(N, sy * sx, -1).transpose(1, 0).reshape(-1, N)
        # we now have sy*sx N x N matrices where the diagonals correspond to dot product between pairs consecutive in time at the same bagtch index
        # aka the correct answer. So the correct logit index is the diagonal sx*sy times
        target1 = torch.arange(N).repeat(sx * sy).to(self.device)
        loss1 = nn.CrossEntropyLoss()(logits1, target1)
        acc1 = calculate_accuracy(logits1.detach().cpu().numpy(), target1.detach().cpu().numpy())
        return loss1, acc1


    def calc_local_to_local(self, local_t, local_tp1):
        N, sy, sx, d = local_tp1.shape
        # Loss 2: f5 patches at time t, with f5 patches at time t+1
        local_t_score = self.score_fxn2(local_t.reshape(-1, d))
        transformed_local_t = local_t_score.reshape(N, sy*sx,d).transpose(0,1)
        local_tp1 = local_tp1.reshape(N, sy * sx, d).transpose(0, 1)
        logits2 = torch.matmul(transformed_local_t, local_tp1.transpose(1, 2)).reshape(-1, N)
        target2 = torch.arange(N).repeat(sx * sy).to(self.device)
        loss2 = nn.CrossEntropyLoss()(logits2, target2)
        acc2 = calculate_accuracy(logits2.detach().cpu().numpy(), target2.detach().cpu().numpy())
        return loss2, acc2

    def calc_loss(self, xt, a, xtp1):
        f_t = self.encoder(xt)
        f_tp1 = self.encoder(xtp1)
        fmap_t = self.fmap_encoder(xt).permute(0, 2, 3, 1)
        fmap_tp1 = self.fmap_encoder(xtp1).permute(0, 2, 3, 1)


        loss1, acc1 = self.calc_global_to_local(f_t, fmap_tp1)
        loss2, acc2 = self.calc_local_to_local(fmap_t, fmap_tp1)

        loss = loss1 + loss2
        if self.training:
            self.wandb.log({"tr_acc1": acc1})
            self.wandb.log({"tr_acc2": acc2})
            self.wandb.log({"tr_loss1": loss1.item()})
            self.wandb.log({"tr_loss2": loss2.item()})
        else:
            self.wandb.log({"val_acc1": acc1})
            self.wandb.log({"val_acc2": acc2})
            self.wandb.log({"val_loss1": loss1.item()})
            self.wandb.log({"val_loss2": loss2.item()})
        return loss


class SlotSTDIMModel(STDIMModel):
    def __init__(self,encoder, args, global_vector_len, device=torch.device('cpu'), wandb=None):
        super().__init__(encoder, args, global_vector_len, device, wandb)
        self.encoder = encoder
        self.local_len = self.encoder.feat_maps_per_slot_map
        self.global_len = self.encoder.slot_len
        self.project_to_slot_len = nn.Linear(self.local_len, self.global_len)
        self.slot_len_to_slot_len = nn.Linear(self.local_len, self.local_len)
        
        
    def calc_loss(self,xt, a, xtp1):
        sv_t, sv_tp1 = self.encoder(xt), self.encoder(xtp1)
        sm_t, sm_tp1 = self.encoder.get_slot_maps(xt), self.encoder.get_slot_maps(xtp1)

        loss_gl, acc_gl = self.slot_global_to_local(sv_t, sm_tp1)
        loss_ll,acc_ll = self.slot_local_to_local(sm_t, sm_tp1)

        loss = loss_gl + loss_ll

        return loss


    def slot_global_to_local(self, slot_vectors, slot_maps):
        """

        computes slot-based global to local loss
        Args:
        * slot_vectors (tensor):
                    size: (batch_size, num_slots, slot_len)
                    a batch of outputs from the slot encoder
                    it's a batch of sets of slot vectors

        * slot_maps (tensor):
                    size: (batch_size, num_slots, num_feat_maps_per_slot, height_feat_map, width_feat_map)

                    a batch of intermediate layer feature maps (segregated into num_slots groups of num_feat_maps_per_slot)
                    from the encoder

        :return:
                loss (float): the loss
                acc (float): the contrastive accuracy
        """
        # N = batch_size
        N, num_slots, slot_len = slot_vectors.shape
        N, num_slots, num_feat_maps_per_slot, h,w = slot_maps.shape # h,w are height and width of each feature map

        # permute tensor to make num_slots b/c logically the outermost loop is over slots
        slot_vectors = slot_vectors.transpose(1,0) # (num_slots, N, slot_len)
        slot_maps = slot_maps.transpose(1,0) #  (num_slots, N, num_feat_maps_per_slot, h, w)

        # permute slot_map tensor to be num_slots,h,w,N,num_feat_maps_per_slot because
        # the next outermost loop is spatial dims of feature maps
        # also we want num_feat_maps_per_slot dimension to be outermost dimension for projection
        slot_maps = slot_maps.permute(0, 3, 4, 1, 2) # (num_slots, h, w, N, num_feat_maps_per_slot)

        # projects the depth at each spatial dimension of the slot_maps to be the same length as a global slot vector
        slot_maps = self.project_to_slot_len(slot_maps) # (num_slots, h, w, N, slot_len)

        ## Prep for the big matrix multiplication

        # insert dummy dimensions at dimensions 1 and 2 so we can broadcast the height and width from the slot_maps
        slot_vectors = slot_vectors.unsqueeze(1).unsqueeze(2) # (num_slots, 1, 1, N, slot_len)

        # transpose so batch_size dimension is outermost one and slot_len one is second-outermost to facilitate a
        # batch matrix multiply
        slot_vectors = slot_vectors.transpose(3,4) # (num_slots, 1, 1, slot_len, N)

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
        target = torch.arange(N).repeat(num_slots * h * w)

        # the loss
        loss = nn.CrossEntropyLoss()(inp, target)

        # guesses
        # to compute contrastive accuracy, we can just get the argmax of each score and compare that with the target
        guesses = torch.argmax(inp.detach(), dim=1)
        is_correct = torch.eq(guesses, target)
        acc = 100*torch.mean(is_correct.to(torch.float)).item()


        return loss, acc

    def slot_local_to_local(self, slot_maps1, slot_maps2):
        """

        computes slot-based local to local loss
        Args:
            * slot_maps1 (tensor):
                size: (batch_size, num_slots, num_feat_maps_per_slot, height_feat_map, width_feat_map)

                a batch of intermediate layer feature maps (segregated into num_slots groups of num_feat_maps_per_slot)
                from the encoder

          * slot_maps2 (tensor):
                size: (batch_size, num_slots, num_feat_maps_per_slot, height_feat_map, width_feat_map)

                a batch of intermediate layer feature maps (segregated into num_slots groups of num_feat_maps_per_slot)
                from the encoder

        :return:
                loss (float): the loss
                acc (float): the contrastive accuracy
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

        slot_maps1 = self.slot_len_to_slot_len(slot_maps1) # (num_slots, h, w, N, num_feat_maps_per_slot)

        # prep for matmul
        slot_maps2 = slot_maps2.transpose(3,4)  # (num_slots, h, w,  num_feat_maps_per_slot, N)

        scores = torch.matmul(slot_maps1, slot_maps2)  # (num_slots, h, w, N, N)

        inp = scores.reshape(-1, N)

        target = torch.arange(N).repeat(num_slots * h * w)


        # the loss
        loss = nn.CrossEntropyLoss()(inp, target)

        # guesses
        # to compute contrastive accuracy, we can just get the argmax of each score and compare that with the target
        guesses = torch.argmax(inp.detach(), dim=1)
        is_correct = torch.eq(guesses, target)
        acc = 100*torch.mean(is_correct.to(torch.float)).item()


        return loss, acc















class STDIMStructureLossModel(STDIMModel):
    def __init__(self,encoder, args, global_vector_len, device=torch.device('cpu'), wandb=None):
        super().__init__(encoder, args, global_vector_len, device, wandb)
        num_slots = self.args.num_slots
        assert self.global_vector_len % num_slots == 0, "Embedding dim (%i) must be divisible by num_slots (%i)!!!"%(self.global_vector_len, num_slots)
        slot_len = self.global_vector_len // num_slots
        self.score_fxn_3 = nn.Linear(slot_len, slot_len)


    def calc_loss(self,xt, a, xtp1):
        loss = super().calc_loss(xt, a, xtp1)
        f_t = self.encoder(xt)
        f_tp1 = self.encoder(xtp1)


        loss3, acc3 = self.calc_slot_structure_loss(f_t, f_tp1)
        loss += loss3
        if self.training:
            self.wandb.log({"tr_acc_struc": acc3})
            self.wandb.log({"tr_loss_struc": loss3.item()})
        else:
            self.wandb.log({"val_acc_struc": acc3})
            self.wandb.log({"val_loss_struc": loss3.item()})

        return loss


    def calc_slot_structure_loss(self, f_t, f_tp1):
        batch_size, global_vec_len = f_t.shape
        num_slots = self.args.num_slots
        try:
            # cut up vector into slots
            slots_t = f_t.reshape(batch_size, num_slots, -1)
            slots_tp1 = f_tp1.reshape(batch_size, num_slots, -1)
        except:
            assert False, "Embedding dim must be divisible by num_slots!!!"

        batch_size, num_slots, slot_len = slots_t.shape
        # logits: batch_size x num_slots x num_slots
        #        for each example (set of 8 slots), for each slot, dot product with every other slot at next time step
        logits = torch.matmul(self.score_fxn_3(slots_t),
                              slots_tp1.transpose(2,1))
        inp = logits.reshape(batch_size * num_slots, -1)
        target = torch.cat([torch.arange(num_slots) for i in range(batch_size)]).to(self.device)
        loss3 = nn.CrossEntropyLoss()(inp, target)
        acc3 = calculate_accuracy(inp.detach().cpu().numpy(), target.detach().cpu().numpy())
        return loss3, acc3




