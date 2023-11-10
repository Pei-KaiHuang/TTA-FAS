"""
Copyright to 3A-TTA Authors
built upon on SAR(Towards Stable Test-Time Adaptation in Dynamic Wild World) code.
-SAR:https://github.com/mr-eggplant/SAR
"""
import os
import time
import argparse
import json
import random
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
import logging
from pytz import timezone
from datetime import datetime

from utils.utils import *
from models.my_resnet import *
from torch.nn import functional as F
import torch
from torch.utils.data import ConcatDataset

from models.network import *
import TTA
from TTA import close_gradient


import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, EigenGradCAM,XGradCAM, EigenCAM, FullGrad,LayerCAM
import torchvision.transforms as T




def validate_gradcam(data_loader, model, criterion,select_type):
    target_layers = [model.model.layer4[-1]]
    T_transform_resize = T.Resize(8)
    model.eval()


    model.requires_grad_(True)
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    score_list = [] 
    label_list = []
  
    TP = 0.0000001
    TN = 0.0000001
    FP = 0.0000001
    FN = 0.0000001 


    for i, data in enumerate(data_loader, 0):
        images, labels = data
        images = NormalizeData_torch(images)
        images = images.cuda()

        # generate live and spoof activation maps
        model.requires_grad_(True)
        grayscale_cam_r = cam(input_tensor=images, target_category=1)      
        grayscale_cam_f = cam(input_tensor=images, target_category=0)
        

        grayscale_cam_r =  T_transform_resize(torch.from_numpy(grayscale_cam_r)).cuda()
        grayscale_cam_f = T_transform_resize(torch.from_numpy(grayscale_cam_f)).cuda()
        model = close_gradient(model)

        
        score = model(images, grayscale_cam_r, grayscale_cam_f, 
                        adapt=True,select_type=select_type)
        score = score.cpu().data.numpy() 

        for j in range(images.size(0)):
            score_list.append(score[j])
            label_list.append(labels[j]) 

    threshold_cs = 0.5
    for i in range(len(score_list)):
        score = score_list[i]

        if (score >= threshold_cs and label_list[i] == 1):
            TP += 1
        elif (score < threshold_cs and label_list[i] == 0):
            TN += 1
        elif (score >= threshold_cs and label_list[i] == 0):
            FP += 1
        elif (score < threshold_cs and label_list[i] == 1):
            FN += 1
   
    APCER = FP / (TN + FP)
    NPCER = FN / (FN + TP) 
    ACER = np.round((APCER + NPCER) / 2, 4)
    AUC = roc_auc_score(label_list, score_list) 
    return ACER, AUC


def validate(data_loader, model, criterion):
    '''
    evalute no adaptation model
    '''
    model.eval() 
    score_list = []
    label_list = []
    TP = 0.0000001
    TN = 0.0000001
    FP = 0.0000001
    FN = 0.0000001

    
    for i, data in enumerate(data_loader, 0):
        images, labels = data
        images = NormalizeData_torch(images) 
        label_pred = model(images.cuda())

        score = F.softmax(label_pred, dim=1).cpu().data.numpy()[:, 1]
        
        for j in range(images.size(0)):
            score_list.append(score[j]) 
            label_list.append(labels[j])


    threshold_cs = 0.5
    for i in range(len(score_list)):
        score = score_list[i]
        if (score >= threshold_cs and label_list[i] == 1):
            TP += 1
        elif (score < threshold_cs and label_list[i] == 0):
            TN += 1
        elif (score >= threshold_cs and label_list[i] == 0):
            FP += 1
        elif (score < threshold_cs and label_list[i] == 1):
            FN += 1
   
    APCER = FP / (TN + FP)
    NPCER = FN / (FN + TP) 
    ACER = np.round((APCER + NPCER) / 2, 4)
    AUC = roc_auc_score(label_list, score_list) 
    return ACER, AUC 


def get_args():

    parser = argparse.ArgumentParser(description='TTA for FAS')
    parser.add_argument('--trained_model', default='OMR', help='OMR/OCR/OCM/CMR')
    parser.add_argument('--protocol', default='unseen_attack', help='seen_attack / unseen_attack')
    parser.add_argument('--val_dataset', default='3DMAD', help='OCMR_3D: a list consists of 3DMAD, casia_3d, HKBUv1+; 3D_OCMR: a list consists of Oulu, casia, MSU, replay')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4) 
    parser.add_argument('--seed', type=int, default=666) 
    parser.add_argument('--method', default='no_adapt', help='no_adapt / ours') 
    
    return parser.parse_args()

def get_data(args) :
    if args.protocol=='unseen_attack':
        #------3dmask------#
        dataset = args.val_dataset
        shared_data_path = '/shared/test_time_da/FAS/3Dmask/'
        live_path = shared_data_path + dataset + '_images_live.npy'
        spoof_path = shared_data_path + dataset + '_images_spoof.npy'
        live_data = np.load(live_path)
        spoof_data = np.load(spoof_path)
        live_label = np.ones(len(live_data), dtype=np.int64)
        spoof_label = np.zeros(len(spoof_data), dtype=np.int64) 
        total_data = np.concatenate((live_data, spoof_data), axis=0)
        total_label = np.concatenate((live_label, spoof_label), axis=0) 
        trainset = torch.utils.data.TensorDataset(torch.tensor(np.transpose(total_data, (0, 3, 1, 2))),
                                                torch.tensor(total_label))
        data_loader = torch.utils.data.DataLoader(trainset, batch_size= 10, shuffle=True)

        
    elif args.protocol=='seen_attack':
        #-----PA-----#
        dataset = args.val_dataset
        shared_data_path = '/shared/test_time_da/FAS/OCMR/'

        live_path = shared_data_path + dataset + '_images_live.npy'
        print_path = shared_data_path + dataset + '_print_images.npy'
        replay_path = shared_data_path + dataset + '_replay_images.npy'

        live_data = np.load(live_path)
        live_label = np.ones(len(live_data), dtype=np.int64)
        print_data = np.load(print_path)
        replay_data = np.load(replay_path)
        spoof_data = np.concatenate((print_data, replay_data), axis=0)
        spoof_label = np.zeros(len(spoof_data), dtype=np.int64)
        total_data = np.concatenate((live_data, spoof_data), axis=0)
        total_label = np.concatenate((live_label, spoof_label), axis=0) 
        trainset = torch.utils.data.TensorDataset(torch.tensor(np.transpose(total_data, (0, 3, 1, 2))),
                                                torch.tensor(total_label))

        data_loader = torch.utils.data.DataLoader(trainset, batch_size= 10, shuffle=True)

    else: # args.protocol=='mix_unseen'
        dataset_O = "Oulu"
        dataset_C = "casia"
        dataset_M = "MSU"
        dataset_I = "replay"

        dataset_D = '3DMAD'
        dataset_H = 'HKBUv1+'

        shared_data_path1 = '/shared/test_time_da/FAS/OCMR/'
        shared_data_path2 = '/shared/test_time_da/FAS/3Dmask/'

        if args.val_dataset=='mix_print_DH':
            dataset = args.val_dataset
            # live from CDH 
            live_path1 = shared_data_path1 + dataset_C + '_images_live.npy'
            live_path2 = shared_data_path2 + dataset_D + '_images_live.npy'
            live_path3 = shared_data_path2 + dataset_H + '_images_live.npy'
            data1_real = np.load(live_path1)
            data2_real = np.load(live_path2)
            data3_real = np.load(live_path3)

            live_data = np.concatenate((data1_real, data2_real, data3_real), axis=0)
            live_label = np.ones(len(live_data), dtype=np.int64)

            # print from OCMI
            print_path1 = shared_data_path1 + dataset_O + '_print_images.npy'
            print_path2 = shared_data_path1 + dataset_C + '_print_images.npy'
            print_path3 = shared_data_path1 + dataset_M + '_print_images.npy'
            print_path4 = shared_data_path1 + dataset_I + '_print_images.npy'
            # 3d mask from DH
            mask_path1 = shared_data_path2 + dataset_D + '_images_spoof.npy'
            mask_path2 = shared_data_path2 + dataset_H + '_images_spoof.npy'

            data1_fake = np.load(print_path1)
            data2_fake = np.load(print_path2)
            data3_fake = np.load(print_path3)
            data4_fake = np.load(print_path4)
            data5_fake = np.load(mask_path1)
            data6_fake = np.load(mask_path2)

            spoof_data = np.concatenate((data1_fake, data2_fake, data3_fake, data4_fake, data5_fake, data6_fake), axis=0)
            spoof_label = np.zeros(len(spoof_data), dtype=np.int64)
            total_data = np.concatenate((live_data, spoof_data), axis=0)
            total_label = np.concatenate((live_label, spoof_label), axis=0)

            mixed_dataset = torch.utils.data.TensorDataset(torch.tensor(np.transpose(total_data, (0, 3, 1, 2))),
                                                    torch.tensor(total_label))
                                    
            data_loader = torch.utils.data.DataLoader(mixed_dataset, batch_size=10, shuffle=True) 
             
        elif args.val_dataset=='mix_replay_DH':
            dataset = args.val_dataset
            # live from CDH 
            live_path1 = shared_data_path1 + dataset_C + '_images_live.npy'
            live_path2 = shared_data_path2 + dataset_D + '_images_live.npy'
            live_path3 = shared_data_path2 + dataset_H + '_images_live.npy'
            data1_real = np.load(live_path1)
            data2_real = np.load(live_path2)
            data3_real = np.load(live_path3)

            live_data = np.concatenate((data1_real, data2_real, data3_real), axis=0)
            live_label = np.ones(len(live_data), dtype=np.int64)

            # replay from OCMI
            replay_path1 = shared_data_path1 + dataset_O + '_replay_images.npy'
            replay_path2 = shared_data_path1 + dataset_C + '_replay_images.npy'
            replay_path3 = shared_data_path1 + dataset_M + '_replay_images.npy'
            replay_path4 = shared_data_path1 + dataset_I + '_replay_images.npy'
            # 3d mask from DH
            mask_path1 = shared_data_path2 + dataset_D + '_images_spoof.npy'
            mask_path2 = shared_data_path2 + dataset_H + '_images_spoof.npy'

            data1_fake = np.load(replay_path1)
            data2_fake = np.load(replay_path2)
            data3_fake = np.load(replay_path3)
            data4_fake = np.load(replay_path4)
            data5_fake = np.load(mask_path1)
            data6_fake = np.load(mask_path2)

            spoof_data = np.concatenate((data1_fake, data2_fake, data3_fake, data4_fake, data5_fake, data6_fake), axis=0)
            spoof_label = np.zeros(len(spoof_data), dtype=np.int64)
            total_data = np.concatenate((live_data, spoof_data), axis=0)
            total_label = np.concatenate((live_label, spoof_label), axis=0)

            mixed_dataset = torch.utils.data.TensorDataset(torch.tensor(np.transpose(total_data, (0, 3, 1, 2))),
                                                    torch.tensor(total_label))
                                    
            data_loader = torch.utils.data.DataLoader(mixed_dataset, batch_size=10, shuffle=True)
          
        else: #args.val_dataset=="mix_pa"
            live_path1 = shared_data_path1 + dataset_O + '_images_live.npy'
            live_path2 = shared_data_path1 + dataset_M + '_images_live.npy'
            live_path3 = shared_data_path1 + dataset_I + '_images_live.npy'
            data1_real = np.load(live_path1)
            data2_real = np.load(live_path2)
            data3_real = np.load(live_path3)

            live_data = np.concatenate((data1_real, data2_real, data3_real), axis=0)
            live_label = np.ones(len(live_data), dtype=np.int64)

            # replay of OCMI
            print_path1 = shared_data_path1 + dataset_O + '_print_images.npy'
            print_path2 = shared_data_path1 + dataset_M + '_print_images.npy'
            print_path3 = shared_data_path1 + dataset_I + '_print_images.npy'
            print_path4 = shared_data_path1 + dataset_C + '_print_images.npy'

            replay_path1 = shared_data_path1 + dataset_O + '_replay_images.npy'
            replay_path2 = shared_data_path1 + dataset_M + '_replay_images.npy'
            replay_path3 = shared_data_path1 + dataset_I + '_replay_images.npy'
            replay_path4 = shared_data_path1 + dataset_C + '_replay_images.npy'


            data1_fake = np.load(print_path1)
            data2_fake = np.load(print_path2)
            data3_fake = np.load(print_path3)
            data4_fake = np.load(print_path4)
            data5_fake = np.load(replay_path1)
            data6_fake = np.load(replay_path2)
            data7_fake = np.load(replay_path3)
            data8_fake = np.load(replay_path4)


            spoof_data = np.concatenate((data1_fake, data2_fake, data3_fake, data4_fake, 
                                        data5_fake, data6_fake, data7_fake, data8_fake), axis=0)
            spoof_label = np.zeros(len(spoof_data), dtype=np.int64)
            total_data = np.concatenate((live_data, spoof_data), axis=0)
            total_label = np.concatenate((live_label, spoof_label), axis=0)

            mixed_dataset = torch.utils.data.TensorDataset(torch.tensor(np.transpose(total_data, (0, 3, 1, 2))),
                                                    torch.tensor(total_label))
                                    
            data_loader = torch.utils.data.DataLoader(mixed_dataset, batch_size=10, shuffle=True)

    return data_loader

if __name__ == '__main__':
    args = get_args()
    seed = seed_init(args)
  
    loggerpath = './logger/'+args.trained_model+'/'+args.val_dataset+'/'
    logger(loggerpath, args.protocol, train=False)

    
    net = Net().cuda()
    tta_net = TTA_Net().cuda()
    
    logging.info(f'testing on {args.val_dataset}') 
    logging.info(f"protocol: {args.trained_model}==>>{args.val_dataset}.")
    

    #unseen model 
    if args.trained_model=='OMR':  
        MataNet_path = './logs/OMR/6_8.tar'
    elif args.trained_model=='OCR':
        MataNet_path = './logs/OCR/12_15.tar'
    elif args.trained_model=='OCM': 
        MataNet_path = './logs/OCM/2_11.tar'
    elif args.trained_model=='CMR':
        MataNet_path = './logs/CMR/resnet18-235.tar'
    elif args.trained_model=='OMR_replay': 
        MataNet_path = './logs/OMR_replay/5.tar'
    elif args.trained_model=='OMR_print': 
        MataNet_path = './logs/OMR_print/22.tar'
    else:
        MataNet_path = './logs/CDH/40_24.tar'
        
    
    tta_net.load_state_dict(torch.load(MataNet_path))        
    net.load_state_dict(torch.load(MataNet_path))
    
    
            
    
    if args.protocol=="seen_attack" or args.protocol=="unseen_attack": 
        # no adaptation
        data_loader = get_data(args)
        net.load_state_dict(torch.load(MataNet_path))
        net = net
        ACER, AUC = validate(data_loader, net, None)
        logging.info(f"Result under source model(no adaptation)  is ACER: {ACER:.5f} and AUC: {AUC:.5f}")

        # 3A-TTA
        ACER_avg = 0
        AUC_avg = 0
        for i in range(10): 
            #  reset model
            tta_net.load_state_dict(torch.load(MataNet_path))
            tta_net = TTA.configure_model(tta_net)
            params, param_names = TTA.collect_params(tta_net)
            optimizer = torch.optim.Adam(params, lr=1e-4, betas=(0.9, 0.999))
            ttaed_model = TTA.TTA(tta_net, optimizer)
            ACER_TT, AUC_TT = validate_gradcam(data_loader, ttaed_model, None, "attention")
            ACER_avg += ACER_TT
            AUC_avg += AUC_TT   
        ACER_avg /= 10
        AUC_avg /= 10
        logging.info(f"Result under 3A-TTA model.  averge  is ACER {ACER_avg:.5f} and AUC: {AUC_avg:.5f}")
    
    # adapt on extended protocol (mix unseen with leave one attack type out strategy)
    else:
        # no adaptation
        seed = seed_init(args)
        data_loader = get_data(args)
        net.load_state_dict(torch.load(MataNet_path))
        net = net
        ACER, AUC = validate(data_loader, net, None)
        logging.info(f"Result under source model(no adaptation)  is ACER: {ACER:.5f} and AUC: {AUC:.5f}")

        #3A-TTA
        seed = seed_init(args)
        data_loader = get_data(args)
        tta_net.load_state_dict(torch.load(MataNet_path))
        tta_net = TTA.configure_model(tta_net)
        params, param_names = TTA.collect_params(tta_net)
        optimizer = torch.optim.Adam(params, lr=1e-4, betas=(0.9, 0.999))
        ttaed_model = TTA.TTA(tta_net, optimizer)
        ACER_TT, AUC_TT = validate_gradcam(data_loader, ttaed_model, None, "attention")
        logging.info(f"Result under 3A-TTA model  is ACER {ACER_TT:.5f} and AUC: {AUC_TT:.5f}")
    


