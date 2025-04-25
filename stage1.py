from __future__ import print_function
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import argparse
import numpy as np
from PreResNet import *
from resnet import SupCEResNet
from sklearn.mixture import GaussianMixture
import dataloader_cifar as dataloader
import pdb
import io
import PIL
from torchvision import transforms
import seaborn as sns
from sklearn.metrics import precision_score, recall_score
import pickle
import json
import pandas as pd
import time
from pathlib import Path
from utils_plot import plot_histogram_pred, plot_histogram_metric, plot_histogram_metric2
from utils_plot import plot_curve_accuracy_test, plot_curve_accuracy, plot_curve_loss, plot_hist_curve_loss_test, plot_curve_loss_train
from edl import get_device, one_hot_embedding, softplus_evidence, exp_evidence, relu_evidence, edl_log_loss, m_edl_log_loss, edl_mse_loss, edl_digamma_loss, compute_dirichlet_metrics
#psscl
import robust_loss, Contrastive_loss, Contrastive_loss2

sns.set()


contrastive_criterion2 = Contrastive_loss2.SupConLoss()

def conv_p(logits):
    # 10/n_class
    alpha_t = softplus_evidence(logits)+10./args.num_class
    total_alpha_t = torch.sum(alpha_t, dim=1, keepdim=True)
    expected_p = alpha_t / total_alpha_t
    return expected_p

def consistency_loss(output1, output2):            
    preds1 = conv_p(output1).detach()
    preds2 = torch.log(conv_p(output2))
    loss_kldiv = F.kl_div(preds2, preds1, reduction='none')
    loss_kldiv = torch.sum(loss_kldiv, dim=1)
    return loss_kldiv

                      
def train(epoch,net,net2,optimizer,labeled_trainloader, unlabeled_trainloader, all_train_loss, all_train_loss_x, all_train_loss_u, all_train_loss_contrastive, op, savelog=False):
    net.train()
    net2.eval() #fix one network and train the other

    train_loss = train_loss_lx = train_loss_u = train_loss_penalty = contrastive_loss = 0

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1

    max_iters = ((len(labeled_trainloader.dataset)+len(unlabeled_trainloader.dataset))//args.batch_size)+1
    cont_iters = 0 

    topK = 3
    if epoch >= 40:
        topK = 2
    elif epoch >= 70:
        topK = 1

    print("number of iterations: ", num_iter, max_iters)

    while(cont_iters<max_iters): #longmix 
        for batch_idx, (inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x, indices_x) in enumerate(labeled_trainloader):
            try:
                inputs_u, inputs_u2, inputs_u3, inputs_u4, labels_u, indices_u = unlabeled_train_iter.__next__()
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                inputs_u, inputs_u2, inputs_u3, inputs_u4, labels_u, indices_u = unlabeled_train_iter.__next__()
            batch_size = inputs_x.size(0)
            if inputs_u.size(0) <=1 or batch_size <= 1:
                continue

            # Transform label to one-hot
            labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)
            w_x = w_x.view(-1,1).type(torch.FloatTensor)

            inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), inputs_x3.cuda(), inputs_x4.cuda(), labels_x.cuda(), w_x.cuda()
            inputs_u, inputs_u2, inputs_u3, inputs_u4 = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda(), inputs_u4.cuda()

            with torch.no_grad():
            # label co-guessing of unlabeled samples
                outputs_u11 = net(inputs_u)
                outputs_u12 = net(inputs_u2)
                outputs_u21 = net2(inputs_u)
                outputs_u22 = net2(inputs_u2)
                # label refinement of labeled samples
                outputs_x = net(inputs_x)
                outputs_x2 = net(inputs_x2)

                if args.uncertainty:
                    evidence_u11 = softplus_evidence(outputs_u11)
                    evidence_u12 = softplus_evidence(outputs_u12)
                    evidence_u21 = softplus_evidence(outputs_u21)
                    evidence_u22 = softplus_evidence(outputs_u22)
                    alpha_u = evidence_u11/4 + evidence_u12/4 + evidence_u21/4 + evidence_u22/4 + args.evidence_factor
                    if args.uncertainty_class:
                        alpha_e = alpha_u[:, :args.num_class]  # Extract the first `num_classes` columns
                        alpha_u2 = torch.full((alpha_u.shape[0], 1), 10 + args.evidence_factor, dtype=torch.float32, device=device)
                        alpha_u = torch.cat([alpha_e, alpha_u2], dim=1)
                    S_u = torch.sum(alpha_u, dim=1, keepdim=True)
                    pu = alpha_u / S_u
                    ptu = pu**(1/args.T)

                    # label refinement of labeled samples
                    evidence_x = softplus_evidence(outputs_x)
                    evidence_x2 = softplus_evidence(outputs_x2)
                    alpha_x = evidence_x/2 + evidence_x2/2 + args.evidence_factor
                    if args.uncertainty_class:
                        alpha_e = alpha_x[:, :args.num_class]  # Extract the first `num_classes` columns
                        alpha_u = torch.full((alpha_x.shape[0], 1), 10 + args.evidence_factor, dtype=torch.float32, device=device)
                        alpha_x = torch.cat([alpha_e, alpha_u], dim=1)
                        extra_class = torch.zeros((labels_x.shape[0], 1), dtype=labels_x.dtype, device=labels_x.device)
                        labels_x = torch.cat([labels_x, extra_class], dim=1)

                    S_x = torch.sum(alpha_x, dim=1, keepdim=True)
                    px = alpha_x / S_x
                    px = w_x*labels_x + (1-w_x)*px
                    ptx = px**(1/args.T) # temparature sharpening

                else:
                    # label co-guessing of unlabeled samples
                    pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4
                    ptu = pu**(1/args.T)
                    # label refinement of labeled samples
                    px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
                    px = w_x*labels_x + (1-w_x)*px
                    ptx = px**(1/args.T) # temparature sharpening


                targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
                targets_u = targets_u.detach()
                targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize
                targets_x = targets_x.detach()


            
            # mixmatch
            all_inputs_aug = torch.cat([inputs_x3, inputs_x4, inputs_u3, inputs_u4], dim=0)
            all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
            all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            input_a_aug, input_b_aug = all_inputs_aug, all_inputs_aug[idx]
            target_a, target_b = all_targets, all_targets[idx]

            l = np.random.beta(args.alpha, args.alpha, size=(all_inputs.size(0), 1, 1, 1))  # Ensure broadcasting shape
            l = np.maximum(l, 1 - l)
            l = torch.from_numpy(l).float().cuda()

            mixed_input = l * input_a + (1 - l) * input_b       
            mixed_input_aug = l * input_a_aug + (1 - l) * input_b_aug   

            l_targets = l.view(all_inputs.size(0), 1)
            mixed_target = l_targets * target_a + (1 - l_targets) * target_b

            #model outputs
            inputs = torch.cat([mixed_input_aug, all_inputs, mixed_input], dim=0)
            all_logits, all_features = net(inputs, forward_pass='cls_proj')

            aug_mixed_logits = all_logits[:mixed_input.shape[0]]
            aug_mixed_features = all_features[:mixed_input.shape[0]]

            mixed_logits = all_logits[2*mixed_input.shape[0]:]
            mixed_features = all_features[2*mixed_input.shape[0]:]

            logits = all_logits[mixed_input.shape[0]:2*mixed_input.shape[0]]
            features = all_features[mixed_input.shape[0]:2*mixed_input.shape[0]]

            loss_plr = loss_mix_plr = loss_simCLR = loss_mixCLR = 0

            if args.plr_loss:
                indices = torch.cat((indices_x, indices_u), dim=0)
                eval_outputs = op[indices, :]
                labels_u = labels_u.cuda()
                labels_x = torch.argmax(labels_x, dim=1)
                labels = torch.cat((labels_x, labels_u), dim=0)
                
                contrastive_mask = Contrastive_loss.build_mask_step(eval_outputs, topK, labels, device) 
                
                fx3, fx4, fu3, fu4 = torch.chunk(features, 4, dim=0)
                f1, f2 = torch.cat([fx3, fu3], dim=0), torch.cat([fx4, fu4], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                loss_plr = plr_loss(features, mask=contrastive_mask) #unsupervised loss: positive examples are 2 augmentations of the same sample only. Negative examples are samples that do no share the same class between the topk predicted classes
                

            """if args.sim_clr:
                fx3, fx4, fu3, fu4 = torch.chunk(features, 4, dim=0) # same indices

                fu, fx = torch.cat([fu3.unsqueeze(1), fu4.unsqueeze(1)], dim=1), torch.cat([fx3.unsqueeze(1), fx4.unsqueeze(1)], dim=1)
                loss_simCLR_u = contrastive_criterion2(fu)#unsupervised simclr loss
                loss_simCLR = loss_simCLR_u

                if args.sim_supervised_clr:
                    labels_x = torch.argmax(labels_x, dim=1)
                    labels_x = labels_x.view(-1, 1)
                    loss_simCLR_x = contrastive_criterion2(fx, labels_x)#supervised simclr loss
                    loss_simCLR = loss_simCLR_u + loss_simCLR_x
                else:
                    loss_simCLR = loss_simCLR_u"""
                

            """if args.mix_clr:      
                fx3, fx4, fu3, fu4 = torch.chunk(mixclr_features, 4, dim=0)
                fu, fx = torch.cat([fu3.unsqueeze(1), fu4.unsqueeze(1)], dim=1), torch.cat([fx3.unsqueeze(1), fx4.unsqueeze(1)], dim=1)
                f_all = torch.cat([fu, fx], dim=0) 
                loss_mixCLR = contrastive_criterion2(f_all)#unsupervised mixclr loss"""


            if args.uncertainty:
                evidence = softplus_evidence(aug_mixed_logits)
                alpha = evidence + args.evidence_factor
                if args.uncertainty_class:
                    alpha_e = alpha[:, :args.num_class]  # Extract the first `num_classes` columns
                    alpha_u = torch.full((alpha.shape[0], 1), 10 + args.evidence_factor, dtype=torch.float32, device=device)
                    alpha = torch.cat([alpha_e, alpha_u], dim=1)
                S = torch.sum(alpha, dim=1, keepdim=True)
                probs = alpha / S
                pred_mean = probs.mean(0)
                outputs_x = aug_mixed_logits[:batch_size*2]
                probs_u = probs[batch_size*2:]

                Lx, Lu, lamb = criterion(outputs_x, mixed_target[:batch_size*2], probs_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up, target_a[:batch_size*2], target_b[:batch_size*2], l_targets)

            else:
                pred_mean = torch.softmax(aug_mixed_logits, dim=1).mean(0)
                logits_x = aug_mixed_logits[:batch_size*2]
                logits_u = aug_mixed_logits[batch_size*2:]
                Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)

            
            if args.consistency_loss:
                if args.dataset == "cifar10":
                    Con = 3 * torch.mean(consistency_loss(mixed_logits, aug_mixed_logits))
                else:
                    Con = 1 * torch.mean(consistency_loss(mixed_logits, aug_mixed_logits))

            #regularization
            if args.uncertainty_class:
                prior = torch.ones(args.num_class+1)/(args.num_class+1)
            else:
                prior = torch.ones(args.num_class)/args.num_class

            prior = prior.cuda()
            penalty = torch.sum(prior*torch.log(prior/pred_mean))

            cl = args.lambda_plr*(loss_plr + 0.2*loss_mix_plr) + args.lambda_c*(loss_simCLR + 0.2*loss_mixCLR) + Con
            contrastive_loss += cl
            loss = Lx + lamb * Lu + penalty + cl
            train_loss += loss
            train_loss_lx += Lx
            train_loss_u += Lu
            train_loss_penalty += penalty

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            """sys.stdout.write('\r')
            sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                             % (args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx + 1, num_iter,
                                Lx.item(), Lu.item()))
            sys.stdout.flush()"""

            cont_iters = cont_iters + 1
            if cont_iters == max_iters:
                break

    all_train_loss.append(train_loss)
    all_train_loss_x.append(train_loss_lx)
    all_train_loss_u.append(train_loss_u)
    all_train_loss_contrastive.append(contrastive_loss)
    print("contrastive loss", contrastive_loss, "loss ", train_loss_lx.item())

    if savelog:
        train_loss /= len(labeled_trainloader.dataset)
        train_loss_lx /= len(labeled_trainloader.dataset)
        train_loss_u /= len(labeled_trainloader.dataset)
        train_loss_penalty /= len(labeled_trainloader.dataset)
        contrastive_loss /= len(labeled_trainloader.dataset)

    return all_train_loss, all_train_loss_x, all_train_loss_u, all_train_loss_contrastive



def warmup(epoch,net,optimizer,dataloader,savelog=False):
    net.train()
    wm_loss = 0
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, inputs_aug1, inputs_aug2, labels, path) in enumerate(dataloader): 
        batch_size = inputs.size(0)
        y = torch.zeros(batch_size, args.num_class).scatter_(1, labels.view(-1,1), 1)    
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs) 
                      
        if args.uncertainty:
            if args.uncertainty_class:
                loss, _ = m_edl_loss(outputs, y.float(), epoch_num=epoch, num_classes = args.num_class, annealing_step= args.ann_step, activation = args.edl_activation, evidence_factor = args.evidence_factor)
            
            else:    
                loss, _ = edl_loss(outputs, y.float(), epoch_num=epoch, num_classes = args.num_class, annealing_step= args.ann_step, activation = args.edl_activation, evidence_factor = args.evidence_factor)
        else:
            loss = ce_loss(outputs, labels)
        if args.noise_mode=='asym':  # penalize confident prediction for asymmetric noise
            penalty = conf_penalty(outputs)
            L = loss + penalty      
        elif args.noise_mode=='sym':   
            L = loss

        wm_loss += L
        L.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Avg Iter Loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, wm_loss.item()/num_iter))
        sys.stdout.flush()


def test(epoch,net1,net2, acc_hist, loss_hist):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    test_loss = 0
    all_losses = torch.zeros(len(test_loader.dataset))

    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(test_loader):
            batch_size = inputs.size(0)
            y = torch.zeros(batch_size, args.num_class).scatter_(1, targets.view(-1,1), 1)
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)           

            if args.uncertainty:
                loss, losses = edl_loss((outputs1+outputs2)/2, y.float(), epoch_num=epoch, num_classes = args.num_class, annealing_step= args.ann_step, activation = args.edl_activation, evidence_factor = args.evidence_factor)

                evidence_1 = softplus_evidence(outputs1)
                alpha_1 = evidence_1 + args.evidence_factor
                predicted1 = alpha_1 / torch.sum(alpha_1, dim=1, keepdim=True)
                evidence_2 = softplus_evidence(outputs2)
                alpha_2 = evidence_2 + args.evidence_factor
                predicted2 = alpha_2 / torch.sum(alpha_2, dim=1, keepdim=True)
                
                outputs = (predicted1+predicted2)/2
                _, predicted = torch.max(outputs, 1) 

            else:
                outputs = (outputs1+outputs2)/2
                _, predicted = torch.max(outputs, 1) 
                loss = ce_loss(outputs, targets)
                losses = ce_loss_sample(outputs, targets)

            for b in range(inputs.size(0)):
                all_losses[index[b]]=losses[b]

            test_loss += loss                      
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                                
    acc = 100.*correct/total
    
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
    test_log.flush()  

    acc_hist.append(acc)
    loss_hist.append(all_losses)
    return acc_hist, loss_hist


def eval_train(model, all_loss, all_preds, all_hist, all_margins_labels, eval_loss_hist, eval_acc_hist, clean_labels, net_idx, savelog=False):
    model.eval()
    correct_indices = []
    noisy_correct_indices = []

    losses = torch.zeros(len(eval_loader.dataset))

    margins_labels = torch.zeros(len(eval_loader.dataset))
    margin_true_label = torch.zeros(len(eval_loader.dataset))
    evidences =  torch.zeros(len(eval_loader.dataset))

    vacuity = torch.zeros(len(eval_loader.dataset))
    dissonance = torch.zeros(len(eval_loader.dataset))
    entropy = torch.zeros(len(eval_loader.dataset))
    uncertainty_class = torch.zeros(len(eval_loader.dataset))
    mutual_info = torch.zeros(len(eval_loader.dataset))
    
    preds = torch.zeros(len(eval_loader.dataset))
    preds_classes = torch.zeros(len(eval_loader.dataset), args.num_class)
    eval_loss = train_acc = acc_clean = acc_noisy = 0

    #plr
    losses_proto = torch.zeros(len(eval_loader.dataset))
    pl = torch.zeros(len(eval_loader.dataset), dtype=torch.long, device=device)
    op = torch.zeros(len(eval_loader.dataset), args.num_class, dtype=torch.float, device=device)
    pt = torch.zeros(len(eval_loader.dataset), args.num_class, dtype=torch.float, device=device)
    ft = torch.zeros(len(eval_loader.dataset), 128, dtype=torch.float, device=device)

    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            batch_size = inputs.size(0)
            y = torch.zeros(batch_size, args.num_class).scatter_(1, targets.view(-1,1), 1)    
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, logits_proto, features = model(inputs, forward_pass='all')

            if args.uncertainty:
                if args.uncertainty_class:
                    loss, loss_per_sample = m_edl_loss(outputs, y.float(), epoch_num=epoch, num_classes = args.num_class, annealing_step= args.ann_step, activation = args.edl_activation, evidence_factor = args.evidence_factor)
                else:
                    loss, loss_per_sample = edl_loss(outputs, y.float(), epoch_num=epoch, num_classes = args.num_class, annealing_step= args.ann_step, activation = args.edl_activation, evidence_factor = args.evidence_factor)
                
                #compute prototype loss with
                if args.plr_loss:
                    if args.loss_proto == "edl":
                        _, loss_proto = edl_loss(logits_proto, y.float(), epoch_num=epoch, num_classes = args.num_class, annealing_step= args.ann_step, activation = args.edl_activation, evidence_factor = args.evidence_factor)
                    else:
                        loss_proto = ce_loss_sample(logits_proto, targets)
                
                evidence = softplus_evidence(outputs)
                alpha = evidence + args.evidence_factor
                
                eval_preds = alpha / torch.sum(alpha, dim=1, keepdim=True)
                eval_loss += loss
            else:
                loss = ce_loss(outputs, targets)
                loss_per_sample = ce_loss_sample(outputs, targets)
                eval_preds = F.softmax(outputs, -1).cpu().data
                eval_loss += loss


            _, pred = torch.max(outputs.data, -1)
            vac, entr, diss, margin, evide, uncert, mi = compute_dirichlet_metrics(outputs, args.num_class, args.edl_activation, args.evidence_factor)
            
            #marc: compute accuracy of clean and noisy samples with ground true labels and noisy labels
            inds_clean_set = set(map(int, inds_clean))  # Ensure all values are Python ints
            inds_noisy_set = set(map(int, inds_noisy))
            clean_mask = torch.tensor([int(i) in inds_clean_set for i in index], dtype=torch.bool, device=inputs.device)
            noisy_mask = torch.tensor([int(i) in inds_noisy_set for i in index], dtype=torch.bool, device=inputs.device)
            clean_labels_tensor = torch.tensor(clean_labels, dtype=torch.long, device=inputs.device)
            noisy_labels_tensor = torch.tensor(noisy_labels, dtype=torch.long, device=inputs.device)
            batch_clean_labels = clean_labels_tensor[index]
            if clean_mask.any().item():  
                clean_pred = pred[clean_mask]  
                clean_labels_batch = batch_clean_labels[clean_mask]  # Slice clean_labels for the batch
                acc_clean += float((clean_pred == clean_labels_batch).sum().item())  # Correct accumulation
            if noisy_mask.any().item():  
                noisy_pred = pred[noisy_mask]  
                noisy_labels_batch = batch_clean_labels[noisy_mask]  # Slice clean_labels for the batch
                acc_noisy += float((noisy_pred == noisy_labels_batch).sum().item())
            acc = float((pred==batch_clean_labels).sum().item())
            train_acc += acc
            correct_predictions = (pred == batch_clean_labels)
            correct_indices.extend(index.cpu()[correct_predictions.cpu()].tolist())
            noisy_correct_predictions = (pred == noisy_labels_tensor[index])
            noisy_correct_indices.extend(index.cpu()[noisy_correct_predictions.cpu()].tolist())

            for b in range(inputs.size(0)):
                losses[index[b]]=loss_per_sample[b]
                preds[index[b]] = eval_preds[b][targets[b]]
                preds_classes[index[b]] =  eval_preds[b]

            if args.uncertainty:
                for b in range(inputs.size(0)):
                    margins_labels[index[b]] =  margin[b]
                    vacuity[index[b]]=vac[b]
                    dissonance[index[b]]=diss[b]
                    entropy[index[b]]=entr[b]
                    evidences[index[b]] = evide[b]
                    uncertainty_class[index[b]] = uncert[b]
                    mutual_info[index[b]]=mi[b]

                    #compute margins of true labels
                    evidence_pos = outputs[b,targets[b]]
                    copy_outputs = outputs[b].clone()
                    copy_outputs[targets[b]] = float('-inf') 
                    evidence_neg = copy_outputs.max()
                    margin_true_label[index[b]]=evidence_pos-evidence_neg

                    if args.plr_loss:
                        losses_proto[index[b]] = loss_proto[b]
                        pl[index[b]] = pred[b]
                        op[index[b]] = outputs[b]
                        pt[index[b]] = logits_proto[b]
                        ft[index[b]] = features[b]
                    

    losses = (losses-losses.min())/(losses.max() - losses.min())
    losses_proto = (losses_proto - losses_proto.min()) / (losses_proto.max() - losses_proto.min())
    #margins_labels = (margins_labels - margins_labels.min())/(margins_labels.max()- margins_labels.min())
    margin_true_label = (margin_true_label - margin_true_label.min())/(margin_true_label.max() - margin_true_label.min())
    #vacuity = (vacuity - vacuity.min())/(vacuity.max() - vacuity.min())
    #dissonance = (dissonance - dissonance.min())/(dissonance.max() - dissonance.min())
    #entropy = (entropy - entropy.min())/(entropy.max() - entropy.min())
    #evidences =  (evidences - evidences.min())/(evidences.max() - evidences.min())


    eval_loss_hist.append(losses)
    all_preds.append(preds)
    all_loss.append(losses)
    all_proto_loss[net_idx].append(losses_proto)
    all_hist.append(preds_classes)
    all_margins_labels.append(margins_labels)
    eval_acc_hist.append([train_acc/len(eval_loader.dataset), acc_clean/len(inds_clean), acc_noisy/len(inds_noisy)])
    
    all_evidence[net_idx].append(evidences)
    all_vacuity[net_idx].append(vacuity)
    all_dissonance[net_idx].append(dissonance)
    all_entropy[net_idx].append(entropy)
    all_margin_true_label[net_idx].append(margin_true_label)
    all_uncertainty_class[net_idx].append(uncertainty_class)
    all_mutual_info[net_idx].append(mutual_info)

    
    if args.use_loss:
        if args.r==0.9: # average loss over last 5 epochs to improve convergence stability
            history = torch.stack(all_loss)
            input_loss = history[-5:].mean(0)
            input_loss = input_loss.reshape(-1,1)
            input_loss_proto = losses_proto.reshape(-1, 1)
        else:
            input_loss = losses.reshape(-1,1)
            input_loss_proto = losses_proto.reshape(-1, 1)

        # fit a two-component GMM to the loss

        if args.plr_loss:
            input_loss = input_loss.cpu().numpy()
            input_loss_proto = input_loss_proto.cpu().numpy()
            gmm_input = np.column_stack((input_loss, input_loss_proto))
            gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
            gmm.fit(gmm_input)
            mean_square_dists = np.array([np.sum(np.square(gmm.means_[i])) for i in range(2)])
            argmin, argmax = mean_square_dists.argmin(), mean_square_dists.argmax()
            prob = gmm.predict_proba(gmm_input)
            prob = prob[:, argmin]

        else:
            gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
            gmm.fit(input_loss)
            prob = gmm.predict_proba(input_loss)
            prob = prob[:,gmm.means_.argmin()]

    else:#use margins
        if args.r==0.9: # average loss over last 5 epochs to improve convergence stability
            history = torch.stack(all_loss)
            input_margin = history[-5:].mean(0)
            input_margin = input_margin.reshape(-1,1)
        else:
            input_margin = margin_true_label.reshape(-1,1)
        # fit a two-component GMM to the margins
        gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
        gmm.fit(input_margin)
        prob = gmm.predict_proba(input_margin) 
        prob = prob[:,gmm.means_.argmax()]

    return prob, all_loss, all_preds, all_hist, all_margins_labels, eval_loss_hist, eval_acc_hist, correct_indices, noisy_correct_indices, {'op': op, 'pl': pl, 'pt': pt, 'ft': ft}

def get_clean(prob, net_idx):
    pred = (prob > args.p_threshold)      
    idx_view_labeled = (pred).nonzero()[0]
    idx_view_unlabeled = (1-pred).nonzero()[0]
    all_idx_view_labeled[net_idx].append(idx_view_labeled)
    all_idx_view_unlabeled[net_idx].append(idx_view_unlabeled)

    thr_labeled, thr_unlabeled = get_thresholds(all_margins_labels[net_idx][-1], idx_view_labeled, idx_view_unlabeled)

    pred_labeled = (all_margins_labels[net_idx][-1] < thr_unlabeled)   
    pred_labeled = pred_labeled.cpu().numpy()
    idx_remove_labeled = pred_labeled.nonzero()[0]
    all_idx_view_remove_labeled[net_idx].append(idx_remove_labeled)

    pred_unlabeled = (all_margins_labels[net_idx][-1] > thr_labeled)  
    pred_unlabeled = pred_unlabeled.cpu().numpy()
    idx_relabel_unlabeled = pred_unlabeled.nonzero()[0]
    all_idx_view_relabel_unlabeled[net_idx].append(idx_relabel_unlabeled)

    pred_loss = np.array([True if p in idx_view_labeled else False for p in range(len(pred))])

    threshold = []
    closest_index = (np.abs(np.array(prob) - 0.5)).argmin()
    threshold.append(closest_index)

    return pred_loss, threshold, thr_labeled, thr_unlabeled


def get_thresholds(all_margins_labels, idx_view_labeled, idx_view_unlabeled):
    margins_labeled = all_margins_labels[idx_view_labeled]
    thr_labeled = margins_labeled.mean()
    margins_unlabeled = all_margins_labels[idx_view_unlabeled]
    thr_unlabeled = margins_unlabeled.mean()

    return thr_labeled, thr_unlabeled


def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up, targets_x1=None, targets_x2=None, l=None):
      if not args.uncertainty:
          probs_u = torch.softmax(outputs_u, dim=1)
          Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
          Lu = torch.mean((probs_u - targets_u)**2)
          return Lx, Lu, linear_rampup(epoch,warm_up)

      elif args.uncertainty:
          probs_u = outputs_u
          Lu = torch.mean((probs_u - targets_u)**2)
          if args.uncertainty_class:
            Lx, _ = m_edl_loss(outputs_x, targets_x.float(), epoch_num=epoch, num_classes = args.num_class, annealing_step= args.ann_step, activation = args.edl_activation, evidence_factor = args.evidence_factor)
          if args.two_edl:
            Lx11, Lx12 = edl_loss(outputs_x, targets_x1.float(), epoch_num=epoch, num_classes = args.num_class, annealing_step= args.ann_step, activation = args.edl_activation, evidence_factor = args.evidence_factor)
            Lx21, Lx22 = edl_loss(outputs_x, targets_x2.float(), epoch_num=epoch, num_classes = args.num_class, annealing_step= args.ann_step, activation = args.edl_activation, evidence_factor = args.evidence_factor)
            Lx = (l*Lx11 + (1-l)*Lx21).mean() + (l*Lx12 + (1-l)*Lx22).mean()
          else:  
            Lx, _ = edl_loss(outputs_x, targets_x.float(), epoch_num=epoch, num_classes = args.num_class, annealing_step= args.ann_step, activation = args.edl_activation, evidence_factor = args.evidence_factor)
          return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model():
    model = ResNet18(num_classes=args.num_class)
    if args.use_pretrained:
        chekpoint = torch.load('pretrained/ckpt_{}_resnet18.pth'.format(args.dataset))
        sd = {}
        for ke in chekpoint['model']:
            nk = ke.replace('module.', '')
            sd[nk] = chekpoint['model'][ke]
        
        model.load_state_dict(sd, strict=False)

    model = model.cuda()
    return model

def create_model_resnet():
    model = SupCEResNet('resnet18', num_classes=args.num_class)
    chekpoint = torch.load('pretrained/ckpt_{}_resnet18.pth'.format(args.dataset))
    if args.use_pretrained:
        chekpoint = torch.load('pretrained/ckpt_{}_resnet18.pth'.format(args.dataset))
        sd = {}
        for ke in chekpoint['model']:
            nk = ke.replace('module.', '')
            sd[nk] = chekpoint['model'][ke]
        missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        
        model.load_state_dict(sd, strict=False)

    model = model.cuda()
    return model

def guess_unlabeled(net1, net2, unlabeled_trainloader):
    net1.eval()
    net2.eval()

    guessedPred_unlabeled  = []
    for batch_idx, (inputs_u, inputs_u2) in enumerate(unlabeled_trainloader): 

        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net1(inputs_u)
            outputs_u12 = net1(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)            
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()  

            _, guessed_u = torch.max(targets_u, dim=-1)
            guessedPred_unlabeled.append(guessed_u) 

    return torch.cat(guessedPred_unlabeled)

def save_models(save_path):
    state = ({
                    'epoch'     : epoch,
                    'state_dict1'     : net1.state_dict(),
                    'optimizer1'      : optimizer1.state_dict(),
                    'state_dict2'     : net2.state_dict(),
                    'optimizer2'      : optimizer2.state_dict(),
                    'all_loss': all_loss,
                    'all_proto_loss': all_proto_loss,
                    'all_preds': all_preds,
                    'hist_preds': hist_preds,

                    'all_idx_view_labeled': all_idx_view_labeled,
                    'all_idx_view_unlabeled': all_idx_view_unlabeled,
                    'all_idx_view_remove_labeled': all_idx_view_remove_labeled,
                    'all_idx_view_relabel_unlabeled': all_idx_view_relabel_unlabeled,
                    'all_idx_superclean': all_idx_superclean,


                    'acc_hist': acc_hist,
                    'all_margins_labels': all_margins_labels,

                    'eval_acc_hist': eval_acc_hist,
                    'eval_loss_hist': eval_loss_hist,
                    'test_acc_hist': test_acc_hist,
                    'test_losses_hist': test_losses_hist,
                    'all_margins_labels': all_margins_labels,
                    'loss_train': loss_train,
                    'loss_train_x': loss_train_x,
                    'loss_train_u': loss_train_u,
                    'train_loss_contrastive' : train_loss_contrastive,

                    'relabel_idx_1': relabel_idx_1,
                    'relabel_idx_2': relabel_idx_2,

                    'new_labels_1': new_labels_1,
                    'new_labels_2': new_labels_2,

                    'all_vacuity': all_vacuity,
                    'all_dissonance': all_dissonance,
                    'all_entropy': all_entropy,
                    'all_evidence': all_evidence,
                    'all_margin_true_label': all_margin_true_label,
                    'all_uncertainty_class': all_uncertainty_class,
                    'all_mutual_info': all_mutual_info,

                    'clean_metrics': clean_metrics,
                    'correct_metrics': correct_metrics,
                    })


    if epoch%1==0:
        fn2 = os.path.join(save_path, 'model_ckpt.pth.tar')
        torch.save(state, fn2)
        if not os.path.exists('hcs'):
            os.makedirs('hcs')

    if epoch == 29:
        fn2 = os.path.join(save_path, f'model_ckpt_epoch{epoch}.pth.tar')
        torch.save(state, fn2)

    if epoch == 79:
        fn2 = os.path.join(save_path, f'model_ckpt_epoch{epoch}.pth.tar')
        torch.save(state, fn2)
        

def write_log(idx, predicted, log_file, threshold = None):
    acc, prec, recall = get_metrics(predicted, clean_labels, idx, clean_indices=inds_clean)

    total_instances = len(eval_loader.dataset)
    percentage = (len(idx) / total_instances) * 100
    true_clean_indices = set(inds_clean)

    correct_superclean = len(set(idx) & true_clean_indices)
    clean_accuracy = (correct_superclean / len(idx)) * 100 if len(idx) > 0 else 0.0
    
    log_file.write(f"Epoch: {epoch}\n")
    log_file.write(f"Number of superclean instances: {len(idx)}\n")
    log_file.write(f"Percentage of superclean instances: {percentage:.2f}%\n")
    log_file.write(f"Accuracy of superclean instances (clean vs noisy): {acc:.2f}%  \t {prec:.2f}% \t {recall:.2f}%\n")
    log_file.write(f"Accuracy - precision - recall (with respect to clean indices): {clean_accuracy:.2f}%\n")
    log_file.write("-" * 50 + "\n")
    log_file.flush()
    clean_metrics.append([len(idx), percentage, clean_accuracy, prec, recall, acc])

def write_log2(idx, predicted, log_file, threshold = None):
    acc, prec, recall = get_metrics(predicted, clean_labels, idx, clean_indices=correct_indices1)

    total_instances = len(eval_loader.dataset)
    percentage = (len(idx) / total_instances) * 100
    
    
    log_file.write(f"Epoch: {epoch}\n")
    log_file.write(f"Number of superclean instances: {len(idx)}\n")
    log_file.write(f"Percentage of superclean instances: {percentage:.2f}%\n")
    log_file.write(f"Accuracy - precision - recall (with respect to correct indices): {acc:.2f}%  \t {prec:.2f}% \t {recall:.2f}%\n")
    log_file.write("-" * 50 + "\n")
    log_file.flush()
    correct_metrics.append([len(idx), percentage, acc, prec, recall])

def test_superclean(net1, net2):
    net1.eval()
    net2.eval()
    all_predicted = torch.zeros(len(eval_loader.dataset))
    all_predicted1 = torch.zeros(len(eval_loader.dataset))
    all_predicted2 = torch.zeros(len(eval_loader.dataset))

    with torch.no_grad():
        for inputs, targets, index in eval_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1, outputs2 = net1(inputs), net2(inputs)

            if args.uncertainty:
                evidence_1 = softplus_evidence(outputs1)
                alpha_1 = evidence_1 + args.evidence_factor
                predicted1 = alpha_1 / torch.sum(alpha_1, dim=1, keepdim=True)

                evidence_2 = softplus_evidence(outputs2)
                alpha_2 = evidence_2 + args.evidence_factor
                predicted2 = alpha_2 / torch.sum(alpha_2, dim=1, keepdim=True)

                outputs = (predicted1 + predicted2) / 2
            else:
                outputs = (outputs1 + outputs2) / 2

            _, pred = torch.max(outputs, 1)
            _, pred_1 = torch.max(outputs1, 1)
            _, pred_2 = torch.max(outputs2, 1)

            for b in range(inputs.size(0)):
                all_predicted[index[b]]=pred[b]
                all_predicted1[index[b]]=pred_1[b]
                all_predicted2[index[b]]=pred_2[b]
    
    return all_predicted, all_predicted1, all_predicted2


def get_metrics(predicted, clean_labels, idx, clean_indices):
    # Ensure torch tensors
    predicted = torch.tensor(predicted) if not torch.is_tensor(predicted) else predicted
    clean_labels_tensor = torch.tensor(clean_labels, dtype=torch.long)
    idx = torch.tensor(idx) if not torch.is_tensor(idx) else idx
    clean_indices = torch.tensor(clean_indices) if not torch.is_tensor(clean_indices) else clean_indices

    if len(idx) == 0:
        return 0.0, 0.0, 0.0  # Avoid divide-by-zero

    # Accuracy: compare predictions vs. true labels on selected indices
    y_pred = predicted[idx]
    y_true = clean_labels_tensor[idx]
    acc = (y_pred == y_true).sum().item() / len(idx)

    # Precision and Recall
    idx_set = set(idx.tolist())
    clean_set = set(clean_indices.tolist())

    true_positives = len(idx_set & clean_set)
    false_positives = len(idx_set - clean_set)
    false_negatives = len(clean_set - idx_set)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0.0

    return 100 * acc, 100 * precision, 100 * recall





class_number = 10
dataset_name = 'cifar10'
dataset_path = './cifar-10-batches-py'
number_epochs = 30

"""argparse.Namespace(
    batch_size=64,
    lr=0.02,
    noise_mode='sym',
    alpha=4,
    lambda_u=0,
    p_threshold=0.5,
    T=0.5,
    num_epochs=number_epochs,
    r=0.2,
    id='',
    seed=123,
    gpuid=0,
    run=0,
    num_class=class_number,
    data_path= dataset_path,
    dataset='cifar100',
    gce_loss = False,
    use_pretrained = False,

    uncertainty = True,
    edl_loss = edl_log_loss,
    edl_activation = softplus_evidence,
    uncertainty_class = False,
    ann_step = 0.01, 
    evidence_factor = 1/10,

    epoch_relabel = 300,
    use_loss = True,
    name_exp = "0.2_edl_0.01_bptplr",

    lambda_plr = 0.5,
    plr_loss = True,
    lambda_c = 0.025, #unicon #psscl,
    sim_clr = False,
    mix_clr = False,

    bpt=True,
    ),
    
    argparse.Namespace(
    batch_size=64,
    lr=0.02,
    noise_mode='sym',
    alpha=4,
    lambda_u=0,
    p_threshold=0.5,
    T=0.5,
    num_epochs=number_epochs,
    r=0.2,
    id='',
    seed=123,
    gpuid=0,
    run=0,
    num_class=class_number,
    data_path= dataset_path,
    dataset='cifar100',
    gce_loss = False,
    use_pretrained = False,

    uncertainty = True,
    edl_loss = edl_log_loss,
    edl_activation = softplus_evidence,
    uncertainty_class = False,
    ann_step = 0.01, 
    evidence_factor = 1/10,

    epoch_relabel = 300,
    use_loss = True,
    name_exp = "0.2_edl_0.01",

    lambda_plr = 0.5,
    plr_loss = False,
    lambda_c = 0.025, #unicon #psscl,
    sim_clr = False,
    mix_clr = False,

    bpt=False,
    ),

    argparse.Namespace(
    batch_size=64,
    lr=0.02,
    noise_mode='sym',
    alpha=4,
    lambda_u=0,
    p_threshold=0.5,
    T=0.5,
    num_epochs=number_epochs,
    r=0.2,
    id='',
    seed=123,
    gpuid=0,
    run=0,
    num_class=class_number,
    data_path= dataset_path,
    dataset='cifar100',
    gce_loss = False,
    use_pretrained = False,

    uncertainty = True,
    edl_loss = edl_log_loss,
    edl_activation = softplus_evidence,
    uncertainty_class = False,
    ann_step = 0.1, 
    evidence_factor = 1/10,

    epoch_relabel = 300,
    use_loss = True,
    name_exp = "0.2_edl_0.1",

    lambda_plr = 0.5,
    plr_loss = False,
    lambda_c = 0.025, #unicon #psscl,
    sim_clr = False,
    mix_clr = False,

    bpt=False,
    ),

    argparse.Namespace(
    batch_size=64,
    lr=0.02,
    noise_mode='sym',
    alpha=4,
    lambda_u=0,
    p_threshold=0.5,
    T=0.5,
    num_epochs=number_epochs,
    r=0.2,
    id='',
    seed=123,
    gpuid=0,
    run=0,
    num_class=class_number,
    data_path= dataset_path,
    dataset='cifar100',
    gce_loss = False,
    use_pretrained = False,

    uncertainty = True,
    edl_loss = edl_log_loss,
    edl_activation = softplus_evidence,
    uncertainty_class = False,
    ann_step = 0.5, 
    evidence_factor = 1/10,

    epoch_relabel = 300,
    use_loss = True,
    name_exp = "0.2_edl_0.5",

    lambda_plr = 0.5,
    plr_loss = False,
    lambda_c = 0.025, #unicon #psscl,
    sim_clr = False,
    mix_clr = False,

    bpt=False,
    ),


    argparse.Namespace(
    batch_size=64,
    lr=0.02,
    noise_mode='sym',
    alpha=4,
    lambda_u=0,
    p_threshold=0.5,
    T=0.5,
    num_epochs=number_epochs,
    r=0.8,
    id='',
    seed=123,
    gpuid=0,
    run=0,
    num_class=class_number,
    data_path= dataset_path,
    dataset='cifar100',
    gce_loss = False,
    use_pretrained = False,

    uncertainty = False,
    edl_loss = edl_log_loss,
    edl_activation = softplus_evidence,
    uncertainty_class = False,
    ann_step = 0.01, 
    evidence_factor = 1/10,

    epoch_relabel = 300,
    use_loss = True,
    name_exp = "0.8_baseline_2",

    lambda_plr = 0.5,
    plr_loss = False,
    lambda_c = 0.025, #unicon #psscl,
    sim_clr = False,
    mix_clr = False,

    bpt=False,
    )

        argparse.Namespace(
    batch_size=64,
    lr=0.02,
    noise_mode='sym',
    alpha=4,
    lambda_u=0,
    p_threshold=0.5,
    T=0.5,
    num_epochs=number_epochs,
    r=0.2,
    id='',
    seed=123,
    gpuid=0,
    run=0,
    num_class=class_number,
    data_path= dataset_path,
    dataset='cifar100',
    gce_loss = False,
    use_pretrained = False,

    uncertainty = True,
    edl_loss = edl_log_loss,
    edl_activation = softplus_evidence,
    uncertainty_class = False,
    ann_step = 0.001, 
    evidence_factor = 1/10,

    epoch_relabel = 300,
    use_loss = True,
    name_exp = "0.2_edl_0.001",

    lambda_plr = 0,
    plr_loss = False,
    lambda_c = 0.025, #unicon #psscl,
    sim_clr = False,
    mix_clr = False,

    bpt=False,
    ),    
    
    """
a = [
    argparse.Namespace(
    batch_size=64,
    lr=0.02,
    noise_mode='sym',
    alpha=4,
    lambda_u=0,
    p_threshold=0.5,
    T=0.5,
    num_epochs=number_epochs,
    r=0.2,
    id='',
    seed=123,
    gpuid=0,
    run=0,
    num_class=class_number,
    data_path= dataset_path,
    dataset='cifar100',
    use_pretrained = False,

    uncertainty = True,
    edl_loss = edl_log_loss,
    edl_activation = softplus_evidence,
    uncertainty_class = False,
    ann_step = 0.01, 
    evidence_factor = 1/10,

    epoch_relabel = 300,
    use_loss = True,
    name_exp = "0.2_edl_0.01_consistency",

    lambda_plr = 1,
    lambda_c = 0.025, #unicon #psscl,
    sim_clr = False,
    mix_clr = False,
    consistency_loss = True,

    plr_loss = False,
    bpt=False,
    loss_proto = "",
    two_edl = False,
    ),


    argparse.Namespace(
    batch_size=64,
    lr=0.02,
    noise_mode='sym',
    alpha=4,
    lambda_u=0,
    p_threshold=0.5,
    T=0.5,
    num_epochs=number_epochs,
    r=0.2,
    id='',
    seed=123,
    gpuid=0,
    run=0,
    num_class=class_number,
    data_path= dataset_path,
    dataset='cifar100',
    use_pretrained = False,

    uncertainty = True,
    edl_loss = edl_log_loss,
    edl_activation = softplus_evidence,
    uncertainty_class = False,
    ann_step = 0.1, 
    evidence_factor = 1/10,

    epoch_relabel = 300,
    use_loss = True,
    name_exp = "0.8_edl_0.1_consistency",

    lambda_plr = 1,
    plr_loss = False,
    lambda_c = 0.025, #unicon #psscl,
    sim_clr = False,
    mix_clr = False,
    consistency_loss = True,

    bpt=False,
    loss_proto = "",

    two_edl = False,
    ),



    argparse.Namespace(
    batch_size=64,
    lr=0.02,
    noise_mode='sym',
    alpha=4,
    lambda_u=0,
    p_threshold=0.5,
    T=0.5,
    num_epochs=number_epochs,
    r=0.2,
    id='',
    seed=123,
    gpuid=0,
    run=0,
    num_class=class_number,
    data_path= dataset_path,
    dataset='cifar100',
    use_pretrained = False,

    uncertainty = True,
    edl_loss = edl_log_loss,
    edl_activation = softplus_evidence,
    uncertainty_class = False,
    ann_step = 0.01, 
    evidence_factor = 1/10,

    epoch_relabel = 300,
    use_loss = True,
    name_exp = "0.2_edl_0.01_bptplr_ce",

    lambda_c = 0.025, #unicon #psscl,
    sim_clr = False,
    mix_clr = False,
    consistency_loss = False,

    lambda_plr = 1,
    plr_loss = True,
    bpt=True,
    loss_proto = "ce",

    two_edl = False,
    ),


    argparse.Namespace(
    batch_size=64,
    lr=0.02,
    noise_mode='sym',
    alpha=4,
    lambda_u=0,
    p_threshold=0.5,
    T=0.5,
    num_epochs=number_epochs,
    r=0.2,
    id='',
    seed=123,
    gpuid=0,
    run=0,
    num_class=class_number,
    data_path= dataset_path,
    dataset='cifar100',
    use_pretrained = False,

    uncertainty = True,
    edl_loss = edl_log_loss,
    edl_activation = softplus_evidence,
    uncertainty_class = False,
    ann_step = 0.1, 
    evidence_factor = 1/10,

    epoch_relabel = 300,
    use_loss = True,
    name_exp = "0.8_edl_0.1_bptplr_ce",

    lambda_c = 0.025, #unicon #psscl,
    sim_clr = False,
    mix_clr = False,
    consistency_loss = False,

    lambda_plr = 1,
    plr_loss = True,
    bpt=True,
    loss_proto = "ce",

    two_edl = False,
    ),

]



for args in a:
    print(args.name_exp)
    if args.dataset == 'cifar100':
        args.num_class=100
        args.data_path= './cifar-100'
        args.num_epochs = 100

    elif args.dataset == 'cifar10':
        args.num_class=10
        args.data_path= './cifar-10-batches-py'
        args.num_epochs = 200


    
    if args.bpt == False:
        args.loss_proto = "ce"

    torch.cuda.set_device(args.gpuid)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    flat=False
    info_nce_loss = Contrastive_loss.InfoNCELoss(temperature=0.1,
                                    batch_size=args.batch_size * 2,
                                    flat=flat,
                                    n_views=2)
    plr_loss = Contrastive_loss.PLRLoss(flat=flat)
    device = torch.device("cuda")


    #adapt lambda_u parametr according to psscl
    if args.noise_mode == "sym":
        if args.r == 0.2:
            args.lambda_u = 0
        if args.r == 0.5:
            args.lambda_u = 25            
        if args.r == 0.8:
            args.lambda_u = 25
        if args.r == 0.9:
            args.lambda_u = 50
    else:
        args.lambda_u = 0
    
    #criterion
    edl_loss = args.edl_loss
    m_edl_loss = m_edl_log_loss
    ce_loss_sample = nn.CrossEntropyLoss(reduction='none')
    ce_loss = nn.CrossEntropyLoss()
    if args.noise_mode=='asym':
        conf_penalty = NegEntropy()
    gce_loss = robust_loss.GCELoss(args.num_class, gpu='0') #only used in warmup
    contrastive_criterion = Contrastive_loss.SupConLoss()
    
    exp_str = f"{args.name_exp}"

    if args.run >0:
        exp_str = exp_str + '_run%d'%args.run
    path_exp='./checkpoint/' + exp_str

    path_plot = os.path.join(path_exp, 'plots')

    Path(path_exp).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(path_exp, 'savedDicts')).mkdir(parents=True, exist_ok=True)
    Path(path_plot).mkdir(parents=True, exist_ok=True)

    
    model_path = "./checkpoint/%s/model_ckpt.pth.tar"%(exp_str)
    incomplete = os.path.exists(model_path)
    
    if incomplete == False:
        model_path = f"./checkpoint/model_ckpt_epoch29_edl{args.r}_{args.ann_step}.pth.tar"
        incomplete = os.path.exists(model_path)

    print('Incomplete...', incomplete)

    if incomplete == False:
        log_mode = 'w'
    else:
        log_mode = 'a'
    stats_log=open('./checkpoint/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_stats.txt',log_mode) 
    test_log=open('./checkpoint/%s/%s_%.2f_%s'%(exp_str,args.dataset,args.r,args.noise_mode)+'_acc.txt',log_mode) 
    time_log=open('./checkpoint/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_time.txt',log_mode) 
    superclean_log= open('./checkpoint/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_superclean.txt',log_mode)

    if args.dataset=='cifar10':
        warm_up = 10
    elif args.dataset=='cifar100':
        warm_up = 30

    loader = dataloader.cifar_dataloader(args.dataset,r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=5,\
        root_dir=args.data_path,log=stats_log,noise_file='noise/%s/%.2f_%s.json'%(args.dataset,args.r,args.noise_mode))

    warmup_trainloader = loader.run('warmup')

    print('| Building net')
    if incomplete:
        args.use_pretrained = False
    net1 = create_model()
    net2 = create_model()
    cudnn.benchmark = True

    criterion = SemiLoss()
    optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    resume_epoch = 0
    if incomplete == True:
        print('loading Model...\n')
        load_path = model_path
        ckpt = torch.load(load_path)
        resume_epoch = ckpt['epoch']+1
        print('resume_epoch....', resume_epoch)
        net1.load_state_dict(ckpt['state_dict1'])
        net2.load_state_dict(ckpt['state_dict2'])
        optimizer1.load_state_dict(ckpt['optimizer1'])
        optimizer2.load_state_dict(ckpt['optimizer2'])

        all_idx_view_labeled = ckpt['all_idx_view_labeled']
        all_idx_view_unlabeled = ckpt['all_idx_view_unlabeled']
        all_idx_view_remove_labeled = ckpt['all_idx_view_remove_labeled']
        all_idx_view_relabel_unlabeled = ckpt['all_idx_view_relabel_unlabeled']
        all_idx_superclean = ckpt['all_idx_superclean']

        all_preds = ckpt['all_preds']
        hist_preds = ckpt['hist_preds']
        acc_hist = ckpt['acc_hist']
        all_loss = ckpt['all_loss']
        all_margins_labels = ckpt['all_margins_labels']

        eval_loss_hist = ckpt['eval_loss_hist']
        eval_acc_hist = ckpt['eval_acc_hist']
        test_acc_hist = ckpt['test_acc_hist']
        test_losses_hist = ckpt['test_losses_hist']
        loss_train = ckpt.get("loss_train", [[],[]])
        loss_train_x = ckpt.get("loss_train_x", [[],[]])
        loss_train_u = ckpt.get("loss_train_u", [[],[]])
        train_loss_contrastive = ckpt.get("train_loss_contrastive", [[],[]])

        new_labels_1 = ckpt['new_labels_1']
        new_labels_2 = ckpt['new_labels_2']
        relabel_idx_1 = ckpt['relabel_idx_1']
        relabel_idx_2 = ckpt['relabel_idx_2']

        all_vacuity = ckpt['all_vacuity']
        all_dissonance = ckpt['all_dissonance']
        all_entropy = ckpt['all_entropy']
        all_evidence = ckpt['all_evidence']
        all_margin_true_label = ckpt['all_margin_true_label']
        all_uncertainty_class = ckpt.get("all_uncertainty_class", [[],[]])
        all_mutual_info = ckpt.get("all_mutual_info", [[],[]])

        all_proto_loss = ckpt.get("all_proto_loss", [[],[]])

        clean_metrics= ckpt['clean_metrics']
        correct_metrics = ckpt['correct_metrics']


    else:
        all_idx_view_labeled = [[],[]]
        all_idx_view_unlabeled = [[], []]
        all_idx_view_remove_labeled = [[], []]
        all_idx_view_relabel_unlabeled = [[], []]
        all_idx_superclean = [[], []]

        all_preds = [[], []] # save the history of preds for two networks
        hist_preds = [[],[]]
        acc_hist = []
        all_loss = [[],[]] # save the history of losses from two networks
        all_proto_loss = [[],[]]
        all_margins_labels = [[],[]]

        eval_loss_hist = [[], []]
        eval_acc_hist = [[], []]
        test_acc_hist = []
        test_losses_hist = []
        loss_train = [[],[]]
        loss_train_x = [[],[]]
        loss_train_u = [[],[]]
        train_loss_contrastive = [[],[]]

        new_labels_1 = []
        new_labels_2 = []
        relabel_idx_1 = []
        relabel_idx_2 = []

        all_vacuity = [[],[]]
        all_dissonance = [[],[]]
        all_entropy = [[],[]]
        all_evidence = [[],[]]
        all_margin_true_label = [[],[]]
        all_uncertainty_class = [[],[]]
        all_mutual_info = [[],[]]

        clean_metrics= []
        correct_metrics = []

    second_ind = True
    test_loader = loader.run('test', second_ind=second_ind)
    eval_loader = loader.run('eval_train', second_ind=second_ind)
    noisy_labels = eval_loader.dataset.noise_label
    clean_labels = eval_loader.dataset.train_label 
    inds_noisy = np.asarray([ind for ind in range(len(noisy_labels)) if noisy_labels[ind] != clean_labels[ind]])
    inds_clean = np.delete(np.arange(len(noisy_labels)), inds_noisy)

    total_time =  0
    warmup_time = 0

    for epoch in range(resume_epoch, args.num_epochs+1):   
        lr=args.lr
        if 160 > epoch >= 80:
            lr /= 10
        elif epoch >= 160:
            lr /= 100      
        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr       
        for param_group in optimizer2.param_groups:
            param_group['lr'] = lr          
            
        if epoch<warm_up:       
            warmup_trainloader = loader.run('warmup')
            start_time = time.time()
            print('Warmup Net1')
            warmup(epoch,net1,optimizer1,warmup_trainloader, savelog=True)    
            print('\nWarmup Net2')
            warmup(epoch,net2,optimizer2,warmup_trainloader, savelog=False) 
            end_time = round(time.time() - start_time)
            total_time+= end_time
            warmup_time+= end_time

            if args.plr_loss:
                print("INITIALIZE PROTOTYPES...")
                Contrastive_loss.init_prototypes(net1, eval_loader, device)
                Contrastive_loss.init_prototypes(net2, eval_loader, device)

            prob_loss1, all_loss[0], all_preds[0], hist_preds[0], all_margins_labels[0], eval_loss_hist[0], eval_acc_hist[0], correct_indices1, noisy_correct_indices1, _ = eval_train(net1, all_loss[0], all_preds[0], hist_preds[0], all_margins_labels[0], eval_loss_hist[0], eval_acc_hist[0], clean_labels, net_idx = 0)
            prob_loss2, all_loss[1], all_preds[1], hist_preds[1], all_margins_labels[1], eval_loss_hist[1], eval_acc_hist[1], correct_indices2, noisy_correct_indices2, _ = eval_train(net2, all_loss[1], all_preds[1], hist_preds[1], all_margins_labels[1], eval_loss_hist[1], eval_acc_hist[1], clean_labels, net_idx = 1) 
            pred_loss1, threshold_loss_1, thr1_labeled, thr1_unlabeled = get_clean(prob_loss1, net_idx = 0)
            pred_loss2, threshold_loss_2, thr2_labeled, thr2_unlabeled = get_clean(prob_loss2, net_idx = 1)
            predicted_labels1 = torch.argmax(hist_preds[0][-1], dim=1)
            predicted_labels2 = torch.argmax(hist_preds[1][-1], dim=1)
            
            if epoch==(warm_up-1):
                time_log.write('Warmup: %f \n'%(warmup_time))
                time_log.flush()

            write_log((pred_loss1).nonzero()[0], predicted_labels1, log_file=superclean_log)
            write_log2((pred_loss1).nonzero()[0], predicted_labels1, log_file=superclean_log)
            
        else:       
            print("training epoch ", epoch)       
            start_time = time.time()               
            if epoch < args.epoch_relabel:
                prob_loss1, all_loss[0], all_preds[0], hist_preds[0], all_margins_labels[0], eval_loss_hist[0], eval_acc_hist[0], correct_indices1, noisy_correct_indices1, op1 = eval_train(net1, all_loss[0], all_preds[0], hist_preds[0], all_margins_labels[0], eval_loss_hist[0], eval_acc_hist[0], clean_labels, net_idx = 0)
                prob_loss2, all_loss[1], all_preds[1], hist_preds[1], all_margins_labels[1], eval_loss_hist[1], eval_acc_hist[1], correct_indices2, noisy_correct_indices2, op2 = eval_train(net2, all_loss[1], all_preds[1], hist_preds[1], all_margins_labels[1], eval_loss_hist[1], eval_acc_hist[1], clean_labels, net_idx = 1) 

            else:
                eval_loader = loader.run('eval_train', relabel_inds = relabel_idx_1, new_labels= new_labels_1)
                prob_loss1, all_loss[0], all_preds[0], hist_preds[0], all_margins_labels[0], eval_loss_hist[0], eval_acc_hist[0], correct_indices1, noisy_correct_indices1, op1 = eval_train(net1, all_loss[0], all_preds[0], hist_preds[0], all_margins_labels[0], eval_loss_hist[0], eval_acc_hist[0], clean_labels, net_idx = 0)
                eval_loader = loader.run('eval_train', relabel_inds = relabel_idx_2, new_labels= new_labels_2)
                prob_loss2, all_loss[1], all_preds[1], hist_preds[1], all_margins_labels[1], eval_loss_hist[1], eval_acc_hist[1], correct_indices2, noisy_correct_indices2, op2 = eval_train(net2, all_loss[1], all_preds[1], hist_preds[1], all_margins_labels[1], eval_loss_hist[1], eval_acc_hist[1], clean_labels, net_idx = 1) 

            pred_loss1, threshold_loss_1, thr1_labeled, thr1_unlabeled = get_clean(prob_loss1, net_idx = 0)
            pred_loss2, threshold_loss_2, thr2_labeled, thr2_unlabeled = get_clean(prob_loss2, net_idx = 1)

            predicted_labels1 = torch.argmax(hist_preds[0][-1], dim=1)
            predicted_labels2 = torch.argmax(hist_preds[1][-1], dim=1)
            
            if epoch >= args.epoch_relabel-1:
                print("relabeling...")
                relabel_idx_1 = list(set(relabel_idx_1) | (set(all_idx_view_relabel_unlabeled[0][-1]) - set(all_idx_superclean[0][-1])))
                new_labels_1 = predicted_labels1[relabel_idx_1].to(torch.int64)
                relabel_idx_2 = list(set(relabel_idx_2) | set(all_idx_view_relabel_unlabeled[1][-1]) - set(all_idx_superclean[1][-1]))
                new_labels_2 = predicted_labels2[relabel_idx_2].to(torch.int64)

            if args.bpt:
                print("BPT...")
                sample_rate1 = len((pred_loss1).nonzero()[0]) / len(eval_loader.dataset)
                sample_rate2 = len((pred_loss2).nonzero()[0]) / len(eval_loader.dataset)
                pred1_new = np.zeros(len(eval_loader.dataset)).astype(np.bool_)
                pred2_new = np.zeros(len(eval_loader.dataset)).astype(np.bool_)
                class_len1 = int(sample_rate1 * len(eval_loader.dataset) / args.num_class)
                class_len2 = int(sample_rate2 * len(eval_loader.dataset) / args.num_class)
                for i in range(args.num_class):
                    class_indices = np.where(np.array(eval_loader.dataset.noise_label) == i)[0]
                    size1 = len(class_indices)
                    class_len_temp1 = min(size1, class_len1)
                    class_len_temp2 = min(size1, class_len2)

                    prob = np.argsort(-prob_loss1[class_indices])
                    select_idx = class_indices[prob[:class_len_temp1]]
                    pred1_new[select_idx] = True

                    prob = np.argsort(-prob_loss2[class_indices])
                    select_idx = class_indices[prob[:class_len_temp2]]
                    pred2_new[select_idx] = True
                pred_loss1 = pred1_new
                pred_loss2 = pred2_new

            #write log superclean
            write_log((pred_loss1).nonzero()[0], predicted_labels1, log_file=superclean_log)
            write_log2((pred_loss1).nonzero()[0], predicted_labels1, log_file=superclean_log)

            end_time = round(time.time() - start_time)
            total_time+= end_time
            start_time = time.time()
            
            if epoch < args.epoch_relabel:
                print('Train Net1')
                labeled_trainloader, unlabeled_trainloader, _ = loader.run('train',pred_loss2,prob_loss2, second_ind=second_ind) # co-divide
                loss_train[0], loss_train_x[0], loss_train_u[0], train_loss_contrastive[0] = train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader, loss_train[0], loss_train_x[0], loss_train_u[0], train_loss_contrastive[0], op2['op'], savelog=True) # train net1  
                        
                print('\nTrain Net2')
                labeled_trainloader, unlabeled_trainloader, u_map_trainloader = loader.run('train',pred_loss1,prob_loss1, second_ind=second_ind) # co-divide
                loss_train[1], loss_train_x[1], loss_train_u[1], train_loss_contrastive[1] = train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader, loss_train[1], loss_train_x[1], loss_train_u[1], train_loss_contrastive[1], op1['op'], savelog=False) # train net2         
            else:
                print('Train Net1')
                labeled_trainloader, unlabeled_trainloader, _ = loader.run('train',pred_loss2,prob_loss2, relabel_inds = relabel_idx_2, new_labels= new_labels_2) # co-divide
                loss_train[0], loss_train_x[0], loss_train_u[0] = train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader, loss_train[0], loss_train_x[0], loss_train_u[0], savelog=True) # train net1  
                        
                print('\nTrain Net2')
                labeled_trainloader, unlabeled_trainloader, u_map_trainloader = loader.run('train',pred_loss1,prob_loss1, relabel_inds = relabel_idx_1, new_labels= new_labels_1) # co-divide
                loss_train[1], loss_train_x[1], loss_train_u[1] = train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader, loss_train[1], loss_train_x[1], loss_train_u[1], savelog=False) # train net2         


            if args.plr_loss:
                print("PLR CORRECTION AND PROTOTYPES...")
                all_indices_x = torch.tensor(pred_loss2.nonzero()[0])
                clean_labels_x = Contrastive_loss.noise_correction(op2['pt'][all_indices_x, :],
                                                                op2['op'][all_indices_x, :],
                                                                op2['pl'][all_indices_x],
                                                                all_indices_x, device)
                all_indices_u = torch.tensor((1 - pred_loss2).nonzero()[0])
                clean_labels_u = Contrastive_loss.noise_correction(op2['pt'][all_indices_u, :],
                                                                op2['op'][all_indices_u, :],
                                                                op2['pl'][all_indices_u],
                                                                all_indices_u, device)
                # update class prototypes
                features = op2['ft'].to(device)
                labels = op2['pl'].to(device)
                labels[all_indices_x] = clean_labels_x
                labels[all_indices_u] = clean_labels_u
                net1.update_prototypes(features, labels)

                all_indices_x = torch.tensor(pred_loss1.nonzero()[0])
                clean_labels_x = Contrastive_loss.noise_correction(op1['pt'][all_indices_x, :],
                                                                op1['op'][all_indices_x, :],
                                                                op1['pl'][all_indices_x],
                                                                all_indices_x, device)
                all_indices_u = torch.tensor((1 - pred_loss1).nonzero()[0])
                clean_labels_u = Contrastive_loss.noise_correction(op1['pt'][all_indices_u, :],
                                                                op1['op'][all_indices_u, :],
                                                                op1['pl'][all_indices_u],
                                                                all_indices_u, device)
                # update class prototypes
                features = op1['ft'].to(device)
                labels = op1['pl'].to(device)
                labels[all_indices_x] = clean_labels_x
                labels[all_indices_u] = clean_labels_u
                net2.update_prototypes(features, labels)


            end_time = round(time.time() - start_time)
            total_time+= end_time

        test_acc_hist, test_losses_hist = test(epoch,net1,net2, test_acc_hist, test_losses_hist)
        if (epoch%5==0 and epoch !=0):
            plot_hist_curve_loss_test(data_hist= test_losses_hist, path=path_plot, epoch=epoch )
            if epoch>=warm_up:
                plot_curve_loss_train(data_hist=[loss_train[0], loss_train_x[0], loss_train_u[0], train_loss_contrastive[0]], path=path_plot)

            print("Plots...")
            plot_curve_loss(data_hist= eval_loss_hist[0], inds_clean=inds_clean, inds_noisy=inds_noisy, path=path_plot, epoch=epoch )
            plot_curve_accuracy(data_hist= eval_acc_hist[0], inds_clean=inds_clean, inds_noisy=inds_noisy, path=path_plot, epoch=epoch )
            if args.use_loss:
                plot_histogram_metric(data_hist=all_loss[0], inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= all_idx_view_labeled[0][-1], inds_relabeled = relabel_idx_1, thresholds = threshold_loss_1, path=path_plot, epoch=epoch, metric = "Loss"  )
            else:
                plot_histogram_metric(data_hist=all_margin_true_label[0], inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= all_idx_view_labeled[0][-1], inds_relabeled = relabel_idx_1, thresholds = threshold_loss_1, path=path_plot, epoch=epoch, metric = "Margins"  )

            if args.plr_loss:
                plot_histogram_metric(data_hist=all_proto_loss[0], inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= all_idx_view_labeled[0][-1], inds_relabeled = relabel_idx_1, thresholds = threshold_loss_1, path=path_plot, epoch=epoch, metric = "Contrastive_loss"  )
            if args.uncertainty:
                    mean_uncertainty = ( (1 - np.array(all_margins_labels[0][-1])) + np.array(all_vacuity[0][-1]) 
                        + np.array(all_entropy[0][-1]) + np.array(all_dissonance[0][-1]) ) / 4
                    plot_histogram_metric2(data_hist=mean_uncertainty, inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= all_idx_view_labeled[0][-1], thresholds = [], path=path_plot, epoch=epoch, data_hist_2= all_loss[0], data_hist_3 = all_margins_labels[0], data_hist_4 = all_margin_true_label[0], metric = "Mean_uncertainty"  )
                    plot_histogram_metric2(data_hist=all_margins_labels[0], inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= all_idx_view_labeled[0][-1], thresholds = [], path=path_plot, epoch=epoch, data_hist_2= all_loss[0], data_hist_3 = all_margins_labels[0], data_hist_4 = all_margin_true_label[0], metric = "Margins"  )
                    plot_histogram_metric2(data_hist=all_vacuity[0], inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= all_idx_view_labeled[0][-1], thresholds = [], path=path_plot, epoch=epoch, data_hist_2= all_loss[0], data_hist_3 = all_margins_labels[0], data_hist_4 = all_margin_true_label[0], metric = "Vacuity"  )
                    plot_histogram_metric2(data_hist=all_entropy[0], inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= all_idx_view_labeled[0][-1], thresholds = [], path=path_plot, epoch=epoch, data_hist_2= all_loss[0], data_hist_3 = all_margins_labels[0], data_hist_4 = all_margin_true_label[0], metric = "Entropy"  )
                    plot_histogram_metric2(data_hist=all_dissonance[0], inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= all_idx_view_labeled[0][-1],thresholds = [], path=path_plot, epoch=epoch, data_hist_2= all_loss[0], data_hist_3 = all_margins_labels[0], data_hist_4 = all_margin_true_label[0], metric = "Dissonance"  )
                    plot_histogram_metric2(data_hist=all_uncertainty_class[0], inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= all_idx_view_labeled[0][-1], thresholds = [], path=path_plot, epoch=epoch, data_hist_2= all_loss[0], data_hist_3 = all_margins_labels[0], data_hist_4 = all_margin_true_label[0], metric = "uncert_class"  )
                    plot_histogram_metric2(data_hist=all_mutual_info[0], inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= all_idx_view_labeled[0][-1], thresholds = [], path=path_plot, epoch=epoch, data_hist_2= all_loss[0], data_hist_3 = all_margins_labels[0], data_hist_4 = all_margin_true_label[0], metric = "mutual_information"  )
            print("Plots finished")

        save_models(path_exp)
        

    test_log.write('\nBest:%.2f  avgLast10: %.2f\n'%(max(test_acc_hist),sum(test_acc_hist[-10:])/10.0))
    test_log.close() 
    test_log.close() 

    time_log.write('SSL Time: %f \n'%(total_time-warmup_time))
    time_log.write('Total Time: %f \n'%(total_time))
    time_log.close()

    superclean_log.close()








