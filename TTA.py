
'''
Code Reference: 
1. `Tent: Fully Test-Time Adaptation by Entropy Minimization` (CVPR'20)
- https://github.com/DequanWang/tent
2. 'Contrastive Test-Time Adaptation' (CVPR22)
- https://github.com/DianCh/AdaContrast
'''
from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import math
from utils.utils import *
from info_nce import InfoNCE
import time




def NormalizeData_torch(data):
    return (data - torch.min(data)) / ((torch.max(data) - torch.min(data)) + 1e-08)

class TTA(nn.Module):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        # model separate
        self.FE = nn.Sequential(model.layer0, model.layer1 ,model.layer2, model.layer3)
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.Cls = model.classifier
        self.optimizer = optimizer
        
        #create memory bank
        self.K = 60
        self.N = 5 # neighbor
        self.register_buffer("live_bank", torch.zeros(self.K,256,16,16))  
        self.register_buffer("spoof_bank", torch.zeros(self.K,256,16,16))
        self.register_buffer("live_queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("spoof_queue_ptr", torch.zeros(1, dtype=torch.long))
        self.full = [0,0]

        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"


    @torch.no_grad()
    def _dequeue_and_enqueue(self, memory_bank, queue_ptr, keys, store_size, type):
        # gather keys before updating queue
        
        ptr = int(queue_ptr)
        # assert self.K % store_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        if ptr + store_size > (self.K -1):
            memory_bank[ptr : self.K, :, :, :] = keys[0:self.K-ptr,:]
            memory_bank[: store_size-self.K+ptr,: ,: ,:] = keys[self.K-ptr:,:]
            ptr = (ptr + store_size) % self.K  # move pointer
            if type == "live":
                self.full[0] = 1
            else: 
                self.full[1] = 1
            
        else:
            memory_bank[ptr : ptr + store_size, :, :, :] = keys
            ptr = (ptr + store_size) % self.K  # move pointer
        queue_ptr[0] = ptr

    @torch.enable_grad()
    def forward(self, x, live_cam=None, spoof_cam=None, adapt=False,select_type="attention", batch=None): 
        if adapt:
            for _ in range(self.steps):
                self.optimizer.zero_grad()
                device_id = 'cuda:0'
                criterionCls =  nn.CrossEntropyLoss().to(device_id)  #
                criterionCosine = nn.CosineSimilarity().to(device_id) 
                # ---Select data (score + region mask attention)----#
                #  forward
                
                x = self.FE(x)
                features = x 
               
                x = self.layer4(x)
                l4_map = x.mean(dim=1)
                
                x = self.avgpool(x)
                x = torch.flatten(x, 1) 
                batch_f = x
                
                
                outputs = self.Cls(x)
                score = outputs.softmax(1)[:, 1]
                pesudo_label = torch.ones(outputs.size(0), dtype = torch.int64).to(device_id)
                diff = torch.zeros(outputs.size(0)).to(device_id)

                loss = 0
            

                if  select_type =="attention":
                    # regional mask attention
                    for i in range(score.size(0)):
                        # # similarity 
                        live_simi = torch.mean(criterionCosine(NormalizeData_torch(l4_map[i]),NormalizeData_torch(live_cam[i]))).detach() 
                        spoof_simi = torch.mean(criterionCosine(NormalizeData_torch(l4_map[i]), NormalizeData_torch(spoof_cam[i]))).detach()  

                        diff[i] = live_simi-spoof_simi
                        if diff[i] < 0:
                            pesudo_label[i] = 0
                    
                    
                    score_threshold = 0.8
                    attention_threshold = 0.2
                    loss = 0
                    
                    ce_loss = criterionCls(outputs, pesudo_label)
                    
                    
                    live_f = features[torch.where(diff>=0)]
                    spoof_f = features[torch.where(diff<0)]

                    pos = self.live_queue_ptr
                    neg = self.spoof_queue_ptr
                    if self.full[0]==1:
                        pos = self.K
                    if self.full[1]==1:
                        neg = self.K

                    live_bank_f = (self.live_bank.clone().detach())[:pos, :, :, :]
                    spoof_bank_f = (self.spoof_bank.clone().detach())[:neg, :, :, :]
                    bank_f = torch.cat((live_bank_f, spoof_bank_f), dim=0)
                    bank_f = self.layer4(bank_f)
                    bank_f = self.avgpool(bank_f)
                    bank_f = torch.flatten(bank_f, 1) 
                    bank_outputs = self.Cls(bank_f)
                    bank_label = torch.cat((torch.ones(pos, dtype = torch.int64),
                                            torch.zeros(neg, dtype = torch.int64)),dim=0).to(device_id)

                    bank_ce_loss = criterionCls(bank_outputs, bank_label)
                    

                    

                    
                    # # reliable : have same prediction between score and attention difference
                    # # select reliable live data
                    filter_live = torch.where((score>score_threshold) &(diff > attention_threshold)) #
                    reliable_live = features[filter_live]
                    self._dequeue_and_enqueue(self.live_bank, self.live_queue_ptr, reliable_live, reliable_live.size(0), "live")
                    
                    # # select reliable spoof data
                    filter_spoof = torch.where((score<(1-score_threshold)) & (diff < -attention_threshold)) #
                    reliable_spoof = features[filter_spoof]
                    self._dequeue_and_enqueue(self.spoof_bank, self.spoof_queue_ptr, reliable_spoof, reliable_spoof.size(0), "spoof")

                   
                    live_batch_f = batch_f[torch.where(diff>=0)]
                    spoof_batch_f = batch_f[torch.where(diff<0)]

                    pos = self.live_queue_ptr
                    neg = self.spoof_queue_ptr
                    if self.full[0]==1:
                        pos = self.K
                    if self.full[1]==1:
                        neg = self.K
                   
                    #  put bank features into current layer4
                    live_bank_f = (self.live_bank.clone().detach())[:pos, :, :, :]
                    live_bank_f = self.layer4(live_bank_f)
                    live_bank_f = self.avgpool(live_bank_f)
                    live_bank_f = torch.flatten(live_bank_f, 1) 
                    

                    spoof_bank_f = (self.spoof_bank.clone().detach())[:neg, :, :, :]
                    spoof_bank_f = self.layer4(spoof_bank_f)
                    spoof_bank_f = self.avgpool(spoof_bank_f)
                    spoof_bank_f = torch.flatten(spoof_bank_f, 1)
                    

                    
                    
                  

                    # Asymmetric prototype contrastive learning 
                    criterionContrastive = InfoNCE(negative_mode='paired')
                    # for live features 
                    # positive pair
                    live_prototype = live_bank_f.mean(dim=0)
                    live_pos_keys = live_prototype.unsqueeze(0).expand(live_batch_f.size(0),-1)
                
                    # negative pairs
                    neighbor_idx = get_nearest_neighbor(live_batch_f, spoof_bank_f, self.N) 
                    live_neg_keys = spoof_bank_f[neighbor_idx, :]
    
                    live_loss = criterionContrastive(live_batch_f, live_pos_keys, live_neg_keys)

                    # for spoof features 
                    # find neighbor of spoof data as positive pair  (pos of spoof)
                    neighbor_idx = get_nearest_neighbor(spoof_batch_f, spoof_bank_f, self.N+1) 
                    spoof_pos_neighbor = spoof_bank_f[neighbor_idx[:,1:], :]
                    spoof_pos_prototype = spoof_pos_neighbor.mean(dim=1)
    
                    # negative pairs
                    negative_keys = live_bank_f.unsqueeze(0)
                    spoof_neg_keys = negative_keys.expand(spoof_f.size(0),-1,-1)
                    spoof_loss = criterionContrastive(spoof_batch_f, spoof_pos_prototype, spoof_neg_keys) 

                    contrastive_loss = live_loss + spoof_loss
                    
                    loss = 3*contrastive_loss +2* ce_loss + 1*bank_ce_loss
                    
                    

                if(torch.isnan(loss).any()):
                    return score
                else:
            
                    loss.backward()
                    self.optimizer.step()
                    
                    return score
                return score
        else:
            outputs = self.model(x)
        return outputs




        

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_parameters():
        if 'classifier' in nm:
            params.append(m)
            names.append(nm)
        if 'layer4' in nm:
            params.append(m)
            names.append(nm)
        

    return params, names




def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics

    for nm, m in model.named_parameters():
        if 'classifier' in nm:
            m.requires_grad_(True)
        if 'layer4' in nm:
            m.requires_grad_(True)


    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"

def close_gradient(model):
    """Configure model for use with tent."""
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisic
    for nm, m in model.named_parameters():
        if 'classifier' in nm:
            m.requires_grad_(True)
        if 'layer4' in nm:
            m.requires_grad_(True)
  

    return model


