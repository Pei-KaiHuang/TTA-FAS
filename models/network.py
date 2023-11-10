'''
Reference: `Single-side domain generalization for face anti-spoofing` (CVPR'20)
- https://arxiv.org/abs/2004.14043
'''
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch
from torch import Tensor
import torch.nn as nn 
from typing import Type, Any, Callable, Union, List, Optional
import torch.utils.model_zoo as model_zoo
from models.my_resnet import *


class TTA_Net(nn.Module):
    def __init__(self):
        super(TTA_Net, self).__init__()
        backbone = resnet18()
        backbone.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.classifier = Classifier()
        self.layer0 = backbone.layer0
        self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool, self.fc= \
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4, backbone.avgpool, backbone.fc

    def forward(self, input):
        x = self.layer0(input)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        classifier_out = self.classifier(x)
        
        return classifier_out 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        backbone = resnet18()
        backbone.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.classifier = Classifier()
        self.layer0 = backbone.layer0
        self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool, self.fc= \
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4, backbone.avgpool, backbone.fc

    def forward(self, input):
        x = self.layer0(input)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
       
        x = self.layer4(x)
        # l4_f = x
        l4_f = x.mean(dim=1)
        

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        classifier_out = self.classifier(x)
        
        # return classifier_out, l4_f
        return classifier_out 

 


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier_layer = nn.Linear(512, 128)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)
        self.classifier_layer2 = nn.Linear(128, 64)
        self.classifier_layer2.weight.data.normal_(0, 0.01)
        self.classifier_layer2.bias.data.fill_(0.0)
        self.classifier_layer3 = nn.Linear(64, 2)
        self.classifier_layer3.weight.data.normal_(0, 0.01)
        self.classifier_layer3.bias.data.fill_(0.0)
      

    def forward(self, input):
        classifier_out = self.classifier_layer(input)
        classifier_out = self.classifier_layer2(classifier_out)
        classifier_out = self.classifier_layer3(classifier_out)
        return classifier_out





























# import torch
# import torch.nn as nn
# import torch.nn.functional as F 
# import torch
# from torch import Tensor
# import torch.nn as nn 
# from typing import Type, Any, Callable, Union, List, Optional
# import torch.utils.model_zoo as model_zoo
# from models.my_resnet import *

# '''
# Reference: `Single-side domain generalization for face anti-spoofing` (CVPR'20)
# - https://arxiv.org/abs/2004.14043
# '''

# class Tent_Net(nn.Module):
#     def __init__(self):
#         super(Tent_Net, self).__init__()
#         backbone = resnet18()
#         backbone.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
#         self.classifier = Classifier()
#         self.layer0 = backbone.layer0
#         self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool, self.fc= \
#             backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4, backbone.avgpool, backbone.fc

#     def forward(self, input):
#         x = self.layer0(input)


#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
        

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         classifier_out = self.classifier(x)
        
#         # return classifier_out , features
#         return classifier_out 

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         backbone = resnet18()
#         backbone.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
#         self.classifier = Classifier()
#         self.layer0 = backbone.layer0
#         self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool, self.fc= \
#             backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4, backbone.avgpool, backbone.fc

#     def forward(self, input):
#         x = self.layer0(input)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         features = x.mean(dim=1)
        

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         classifier_out = self.classifier(x)
        
#         return classifier_out, features




# class Classifier(nn.Module):
#     def __init__(self):
#         super(Classifier, self).__init__()
#         self.classifier_layer = nn.Linear(512, 128)
#         self.classifier_layer.weight.data.normal_(0, 0.01)
#         self.classifier_layer.bias.data.fill_(0.0)
#         self.classifier_layer2 = nn.Linear(128, 64)
#         self.classifier_layer2.weight.data.normal_(0, 0.01)
#         self.classifier_layer2.bias.data.fill_(0.0)
#         self.classifier_layer3 = nn.Linear(64, 2)
#         self.classifier_layer3.weight.data.normal_(0, 0.01)
#         self.classifier_layer3.bias.data.fill_(0.0)
      
   

#     def forward(self, input):
#         classifier_out = self.classifier_layer(input)
#         classifier_out = self.classifier_layer2(classifier_out)
#         classifier_out = self.classifier_layer3(classifier_out)
#         return classifier_out

