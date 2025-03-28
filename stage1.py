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
from sklearn.mixture import GaussianMixture
import dataloader_cifar as dataloader
import pdb
import io
import PIL
from torchvision import transforms
import seaborn as sns
import sklearn.metrics as metrics
import pickle
import json
import pandas as pd
import time
from pathlib import Path
from utils_plot import plot_histogram_pred, plot_histogram_metric, plot_histogram_metric2
from utils_plot import plot_curve_accuracy_test, plot_curve_accuracy, plot_curve_loss, plot_hist_curve_loss_test, plot_curve_loss_train
from edl import get_device, one_hot_embedding, softplus_evidence, exp_evidence, relu_evidence, edl_log_loss, edl_mse_loss, edl_digamma_loss, compute_dirichlet_metrics
#psscl
import robust_loss, Contrastive_loss

sns.set()

def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader, all_train_loss, all_train_loss_x, all_train_loss_u, savelog=False):
    net.train()
    net2.eval() #fix one network and train the other

    train_loss = train_loss_lx = train_loss_u = train_loss_penalty = train_loss_simclr = train_loss_mixclr = 0

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x, _) in enumerate(labeled_trainloader):
        try:
            inputs_u, inputs_u2, inputs_u3, inputs_u4, _ = unlabeled_train_iter.__next__()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, inputs_u3, inputs_u4, _ = unlabeled_train_iter.__next__()
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
                S_u = torch.sum(alpha_u, dim=1, keepdim=True)
                pu = alpha_u / S_u
                ptu = pu**(1/args.T)

                # label refinement of labeled samples
                evidence_x = softplus_evidence(outputs_x)
                evidence_x2 = softplus_evidence(outputs_x2)
                alpha_x = evidence_x/2 + evidence_x2/2 + args.evidence_factor
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

        ## Unsupervised Contrastive Loss
        if args.sim_clr:
            f1, _ = net(inputs_u3, mode = "encoder")
            f2, _ = net(inputs_u4, mode = "encoder")
            f1 = F.normalize(f1, dim=1)
            f2 = F.normalize(f2, dim=1)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss_simCLR = contrastive_criterion(features)
        else:
            loss_simCLR = 0
        
        # mixmatch
        all_inputs = torch.cat([inputs_x3, inputs_x4, inputs_u3, inputs_u4], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)
        idx = torch.randperm(all_inputs.size(0))
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        l = np.random.beta(args.alpha, args.alpha, size=(all_inputs.size(0), 1, 1, 1))  # Ensure broadcasting shape
        l = np.maximum(l, 1 - l)
        l = torch.from_numpy(l).float().cuda()
        mixed_input = l * input_a + (1 - l) * input_b
        l_targets = l.view(all_inputs.size(0), 1)
        mixed_target = l_targets * target_a + (1 - l_targets) * target_b
        logits = net(mixed_input)

        #mixclr
        if args.mix_clr:
            all_inputs_1 = torch.cat([inputs_x3, inputs_u3], dim=0)
            all_inputs_2 = torch.cat([inputs_x4, inputs_u4], dim=0)
            idx = torch.randperm(all_inputs_1.size(0))
            input_a1, input_b1 = all_inputs_1, all_inputs_1[idx]
            input_a2, input_b2 = all_inputs_2, all_inputs_2[idx]

            l = np.random.beta(args.alpha, args.alpha, size=(all_inputs_1.size(0), 1, 1, 1))  # Ensure broadcasting shape
            l = np.maximum(l, 1 - l)
            l = torch.from_numpy(l).float().cuda()
            mixed_input_1 = l * input_a1 + (1 - l) * input_b1
            mixed_input_2 = l * input_a2 + (1 - l) * input_b2

            f1, _ = net(mixed_input_1, mode = "encoder")
            f2, _ = net(mixed_input_2, mode = "encoder")
            f1 = F.normalize(f1, dim=1)
            f2 = F.normalize(f2, dim=1)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss_mixCLR = contrastive_criterion(features)
        else:
            loss_mixCLR = 0

        if args.uncertainty:
            evidence = softplus_evidence(logits)
            alpha = evidence + args.evidence_factor
            S = torch.sum(alpha, dim=1, keepdim=True)
            probs = alpha / S
            pred_mean = probs.mean(0)
            outputs_x = logits[:batch_size*2]
            probs_u = probs[batch_size*2:]
            Lx, Lu, lamb = criterion(outputs_x, mixed_target[:batch_size*2], probs_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)

        else:
            pred_mean = torch.softmax(logits, dim=1).mean(0)
            logits_x = logits[:batch_size*2]
            logits_u = logits[batch_size*2:]
            Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)


        #regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        loss = Lx + lamb * Lu + penalty + args.lambda_c*(loss_simCLR + 0.2*loss_mixCLR)
        train_loss += loss
        train_loss_lx += Lx
        train_loss_u += Lu
        train_loss_penalty += penalty
        train_loss_simclr += loss_simCLR
        train_loss_mixclr += loss_mixCLR

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if args.sim_clr:
        sys.stdout.write(f'\r{args.dataset}: {args.r:.1f}-{args.noise_mode} | Epoch [{epoch:3d}/{args.num_epochs}] Iter[{batch_idx+1:3d}/{num_iter:3d}]\t Labeled loss: {train_loss_lx.item()/num_iter:.2f}  Unlabeled loss: {train_loss_u.item()/num_iter:.2f}   SimCLR loss: {train_loss_simclr.item()/num_iter:.2f}')
        sys.stdout.flush()
    else: 
        sys.stdout.write(f'\r{args.dataset}: {args.r:.1f}-{args.noise_mode} | Epoch [{epoch:3d}/{args.num_epochs}] Iter[{batch_idx+1:3d}/{num_iter:3d}]\t Labeled loss: {train_loss_lx.item()/num_iter:.2f}  Unlabeled loss: {train_loss_u.item()/num_iter:.2f}')
        sys.stdout.flush()

    all_train_loss.append(train_loss)
    all_train_loss_x.append(train_loss_lx)
    all_train_loss_u.append(train_loss_u)

    if savelog:
        train_loss /= len(labeled_trainloader.dataset)
        train_loss_lx /= len(labeled_trainloader.dataset)
        train_loss_u /= len(labeled_trainloader.dataset)
        train_loss_penalty /= len(labeled_trainloader.dataset)

    return all_train_loss, all_train_loss_x, all_train_loss_u



def warmup(epoch,net,optimizer,dataloader,savelog=False):
    net.train()
    wm_loss = 0
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, _) in enumerate(dataloader):  
        batch_size = inputs.size(0)
        y = torch.zeros(batch_size, args.num_class).scatter_(1, labels.view(-1,1), 1)    
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs) 
                      
        if args.uncertainty:    
            loss, _ = edl_loss(outputs, y.float(), epoch_num=epoch, num_classes = args.num_class, annealing_step= args.ann_step, activation = args.edl_activation, evidence_factor = args.evidence_factor)
        elif args.gce_loss:
            loss = gce_loss(outputs, labels)
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


def eval_train(model, all_loss, all_preds, all_hist, all_margin_true_label, all_margins_labels, all_evidences, eval_loss_hist, eval_acc_hist, clean_labels, savelog=False):
    model.eval()
    correct_indices = []
    noisy_correct_indices = []

    losses = torch.zeros(len(eval_loader.dataset))
    margins_labels = torch.zeros(len(eval_loader.dataset))
    margin_true_label = torch.zeros(len(eval_loader.dataset))
    evidences = torch.zeros(len(eval_loader.dataset))
    
    preds = torch.zeros(len(eval_loader.dataset))
    preds_classes = torch.zeros(len(eval_loader.dataset), args.num_class)
    eval_loss = train_acc = acc_clean = acc_noisy = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            batch_size = inputs.size(0)
            y = torch.zeros(batch_size, args.num_class).scatter_(1, targets.view(-1,1), 1)    
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)

            if args.uncertainty:
                loss, loss_per_sample = edl_loss(outputs, y.float(), epoch_num=epoch, num_classes = args.num_class, annealing_step= args.ann_step, activation = args.edl_activation, evidence_factor = args.evidence_factor)
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
            _, _, _, margin, top_evidence = compute_dirichlet_metrics(outputs, args.num_class, args.edl_activation, args.evidence_factor)

            
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
                    evidences[index[b]] =  top_evidence[b]

                    evidence_pos = outputs[b,targets[b]]
                    copy_outputs = outputs[b].clone()
                    copy_outputs[targets[b]] = -1e5
                    evidence_neg = copy_outputs.max()
                    margin_true_label[index[b]]=evidence_pos-evidence_neg

    losses = (losses-losses.min())/(losses.max() - losses.min())
    margins_labels = (margins_labels - margins_labels.min())/(margins_labels.max()- margins_labels.min())
    margin_true_label = (margin_true_label - margin_true_label.min())/(margin_true_label.max() - margin_true_label.min())
    evidences =  (evidences - evidences.min())/(evidences.max()- evidences.min())

    eval_loss_hist.append(losses)
    all_loss.append(losses)
    all_preds.append(preds)
    all_hist.append(preds_classes)
    all_margin_true_label.append(margin_true_label)
    all_margins_labels.append(margins_labels)
    all_evidences.append(evidences)
    eval_acc_hist.append([train_acc/len(eval_loader.dataset), acc_clean/len(inds_clean), acc_noisy/len(inds_noisy)])

    if args.r==0.9: # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1,1)
    else:
        input_loss = losses.reshape(-1,1)

    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob_loss = prob[:,gmm.means_.argmin()]

    # fit a two-component GMM to the margins
    input_margin = margin_true_label.reshape(-1,1)
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_margin)
    prob = gmm.predict_proba(input_margin) 
    prob_margin = prob[:,gmm.means_.argmax()]
    return prob_loss, prob_margin, all_loss, all_preds, all_hist, all_margin_true_label, all_margins_labels, all_evidences, eval_loss_hist, eval_acc_hist, correct_indices, noisy_correct_indices


def get_superclean_extra(prob1, prob2, all_superclean, all_labeled, all_unlabeled, threshold):
    if threshold < 0.5:
        pred1 = (threshold < prob1) & (prob1 <= 0.5)    
        pred2 = (threshold < prob2) & (prob2 <= 0.5)
    else:
        pred1 = (threshold > prob1) & (prob1 >= 0.5)    
        pred2 = (threshold > prob2) & (prob2 >= 0.5)
    idx_view_labeled = (pred1).nonzero()[0]
    idx_view_unlabeled = (1-pred1).nonzero()[0]
    all_labeled[0].append(idx_view_labeled)
    all_labeled[1].append((pred2).nonzero()[0])
    all_unlabeled[0].append(idx_view_unlabeled)
    all_unlabeled[1].append((1-pred2).nonzero()[0])

    #check hist of predclean net 1
    superclean = []
    nclean = args.num_clean
    for ii in range(len(eval_loader.dataset)):
        clean_lastn = True
        for h_ep in all_labeled[0][-nclean:]:   #check last nclean epochs
            if ii not in h_ep:
                clean_lastn = False
                break
        if clean_lastn:
            superclean.append(ii)
    all_superclean[0].append(superclean)
    pred1 = np.array([True if p in superclean else False for p in range(len(pred1))])

    #check hist of predclean net 2
    superclean = []
    nclean = args.num_clean
    for ii in range(len(eval_loader.dataset)):
        clean_lastn = True
        for h_ep in all_labeled[1][-nclean:]:   #check last nclean epochs
            if ii not in h_ep:
                clean_lastn = False
                break
        if clean_lastn:
            superclean.append(ii)
    all_superclean[1].append(superclean)
    pred2 = np.array([True if p in superclean else False for p in range(len(pred2))])

    return all_superclean, pred1, pred2, all_labeled, all_unlabeled


def get_superclean_relabeled(prob1, prob2, all_superclean, all_labeled, all_unlabeled, threshold):
    """pred1 = (prob1 > threshold)      
    pred2 = (prob2 > threshold)
    idx_view_labeled = pred1.nonzero(as_tuple=True)[0]
    idx_view_unlabeled = (~pred1).nonzero(as_tuple=True)[0]
    all_labeled[0].append(idx_view_labeled)
    all_labeled[1].append(pred2.nonzero(as_tuple=True)[0])
    all_unlabeled[0].append(idx_view_unlabeled)
    all_unlabeled[1].append((~pred2).nonzero(as_tuple=True)[0])  # Fix applied here"""
    pred1 = (prob1 > threshold)   
    pred2 = (prob2 > threshold)
    pred1 = pred1.cpu().numpy()
    pred2 = pred2.cpu().numpy()
    idx_view_labeled = pred1.nonzero()[0]
    idx_view_unlabeled = (1-pred1).nonzero()[0]
    all_labeled[0].append(idx_view_labeled)
    all_labeled[1].append(pred2.nonzero()[0])
    all_unlabeled[0].append(idx_view_unlabeled)
    all_unlabeled[1].append((1-pred2).nonzero()[0])

    """#check hist of predclean net 1
    superclean = []
    nclean = args.num_clean
    for ii in range(len(eval_loader.dataset)):
        clean_lastn = True
        for h_ep in all_labeled[0][-nclean:]:   #check last nclean epochs
            if ii not in h_ep:
                clean_lastn = False
                break
        if clean_lastn:
            superclean.append(ii)
    all_superclean[0].append(superclean)
    pred1 = np.array([True if p in superclean else False for p in range(len(pred1))])

    #check hist of predclean net 2
    superclean = []
    nclean = args.num_clean
    for ii in range(len(eval_loader.dataset)):
        clean_lastn = True
        for h_ep in all_labeled[1][-nclean:]:   #check last nclean epochs
            if ii not in h_ep:
                clean_lastn = False
                break
        if clean_lastn:
            superclean.append(ii)
    all_superclean[1].append(superclean)
    pred2 = np.array([True if p in superclean else False for p in range(len(pred2))])"""

    return all_superclean, pred1, pred2, all_labeled, all_unlabeled
    
def get_superclean(prob1, prob2, all_superclean, all_labeled, all_unlabeled):
    pred1 = (prob1 > args.p_threshold)      
    pred2 = (prob2 > args.p_threshold)
    idx_view_labeled = (pred1).nonzero()[0]
    idx_view_unlabeled = (1-pred1).nonzero()[0]
    all_labeled[0].append(idx_view_labeled)
    all_labeled[1].append((pred2).nonzero()[0])
    all_unlabeled[0].append(idx_view_unlabeled)
    all_unlabeled[1].append((1-pred2).nonzero()[0])

    #check hist of predclean net 1
    superclean = []
    nclean = args.num_clean
    for ii in range(len(eval_loader.dataset)):
        clean_lastn = True
        for h_ep in all_labeled[0][-nclean:]:   #check last nclean epochs
            if ii not in h_ep:
                clean_lastn = False
                break
        if clean_lastn:
            superclean.append(ii)
    all_superclean[0].append(superclean)
    pred1 = np.array([True if p in superclean else False for p in range(len(pred1))])

    #check hist of predclean net 2
    superclean = []
    nclean = args.num_clean
    for ii in range(len(eval_loader.dataset)):
        clean_lastn = True
        for h_ep in all_labeled[1][-nclean:]:   #check last nclean epochs
            if ii not in h_ep:
                clean_lastn = False
                break
        if clean_lastn:
            superclean.append(ii)
    all_superclean[1].append(superclean)
    pred2 = np.array([True if p in superclean else False for p in range(len(pred2))])

    values_idx =[]
    for target in np.arange(0.1, 1.0, 0.1):
        closest_index = (np.abs(np.array(prob1) - target)).argmin()
        values_idx.append(closest_index)

    return all_superclean, pred1, pred2, all_labeled, all_unlabeled, values_idx



def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
      if not args.uncertainty:
          probs_u = torch.softmax(outputs_u, dim=1)
          Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
          Lu = torch.mean((probs_u - targets_u)**2)
          return Lx, Lu, linear_rampup(epoch,warm_up)

      elif args.uncertainty:
          probs_u = outputs_u
          Lu = torch.mean((probs_u - targets_u)**2)
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
                    'all_preds': all_preds,
                    'hist_preds': hist_preds,

                    'all_idx_view_labeled': all_idx_view_labeled,
                    'all_idx_view_unlabeled': all_idx_view_unlabeled,
                    'all_idx_view_labeled_margin': all_idx_view_labeled_margin,
                    'all_idx_view_unlabeled_margin': all_idx_view_unlabeled_margin,
                    'all_idx_view_labeled_relabeled': all_idx_view_labeled_relabeled,
                    'all_idx_view_unlabeled_relabeled': all_idx_view_unlabeled_relabeled,
                    'all_idx_view_labeled_extra': all_idx_view_labeled_extra,
                    'all_idx_view_unlabeled_extra': all_idx_view_unlabeled_extra,

                    'all_superclean': all_superclean,
                    'all_superclean_margin': all_superclean_margin,
                    'all_superclean_relabeled': all_superclean_relabeled,
                    'all_superclean_extra': all_superclean_extra,

                    'acc_hist': acc_hist,
                    'all_evidences': all_evidences,
                    'all_margins_labels': all_margins_labels,
                    'all_margin_true_label': all_margin_true_label,

                    'eval_acc_hist': eval_acc_hist,
                    'eval_loss_hist': eval_loss_hist,
                    'test_acc_hist': test_acc_hist,
                    'test_losses_hist': test_losses_hist,
                    'all_margins_labels': all_margins_labels,
                    'all_margin_true_label': all_margin_true_label,
                    'loss_train': loss_train,
                    'loss_train_x': loss_train_x,
                    'loss_train_u': loss_train_u,
                    })
    state3 = ({
                'all_superclean': all_superclean,
                'all_superclean_margin': all_superclean_margin,
                })


    if epoch%1==0:
        fn2 = os.path.join(save_path, 'model_ckpt.pth.tar')
        torch.save(state, fn2)
        if not os.path.exists('hcs'):
            os.makedirs('hcs')
        fn3 = os.path.join('hcs/', 'hcs_%s_%.2f_%s_cn%d_run%d.pth.tar'%(args.dataset, args.r, args.noise_mode,args.num_clean, args.run))
        torch.save(state3, fn3)

def write_log_superclean2(all_superclean, predicted1, predicted2, log_file):
    acc_union = get_metrics_superclean2(predicted2, clean_labels, list(set(all_superclean[0]) | set(all_superclean[1])))
    acc_intersection =get_metrics_superclean2(predicted2, clean_labels, list(set(all_superclean[0]) & set(all_superclean[1])))
    acc_union_noisy = get_metrics_superclean2(predicted2, clean_labels, list((set(all_superclean[0]) | set(all_superclean[1])) & set(inds_clean)))
    acc_intersection_noisy =get_metrics_superclean2(predicted2, clean_labels, list((set(all_superclean[0]) & set(all_superclean[1])) & set(inds_clean)))

    # Define superclean sets
    union_superclean = set(all_superclean[0]) | set(all_superclean[1])
    intersection_superclean = set(all_superclean[0]) & set(all_superclean[1])
    
    union_num_superclean = len(union_superclean)
    intersection_num_superclean = len(intersection_superclean)
    total_instances = len(eval_loader.dataset)
    
    # Compute statistics for union
    percentage_superclean = (union_num_superclean / total_instances) * 100
    true_clean_indices = set(inds_clean)
    correct_superclean = len(union_superclean & true_clean_indices)
    superclean_accuracy = (correct_superclean / union_num_superclean) * 100 if union_num_superclean > 0 else 0.0

    # Write results for union

    log_file.write(f"Epoch: {epoch}\n")

    log_file.write(f"Number of superclean instances: {union_num_superclean}\n")
    log_file.write(f"Percentage of superclean instances: {percentage_superclean:.2f}%\n")
    log_file.write(f"Accuracy of superclean instances (clean vs noisy): {superclean_accuracy:.2f}%\n")
    log_file.write(f"Accuracy of the superclean instances (true class): {acc_union:.2f}%\n")
    log_file.write(f"Accuracy of the superclean instances (clean instances): {acc_union_noisy:.2f}%\n")
    log_file.write("-" * 50 + "\n")
    log_file.flush()

    # Compute statistics for intersection
    percentage_superclean = (intersection_num_superclean / total_instances) * 100
    correct_superclean = len(intersection_superclean & true_clean_indices)
    superclean_accuracy = (correct_superclean / intersection_num_superclean) * 100 if intersection_num_superclean > 0 else 0.0

    # Write results for intersection
    log_file.write(f"Number of superclean instances (intersection): {intersection_num_superclean}\n")
    log_file.write(f"Percentage of superclean instances (intersection): {percentage_superclean:.2f}%\n")
    log_file.write(f"Accuracy of superclean instances (intersection) (clean vs noisy): {superclean_accuracy:.2f}%\n")
    log_file.write(f"Accuracy of the superclean instances (intersection) (true class): {acc_intersection:.2f}%\n")
    log_file.write(f"Accuracy of the superclean instances (clean instances): {acc_intersection_noisy:.2f}%\n")
    log_file.write("-" * 50 + "\n")
    log_file.flush()


def write_log_superclean(all_superclean, predicted, log_file, threshold = None):
    acc_union, acc_intersection, acc_union_comp, acc_intersection_comp = get_metrics_superclean(predicted, clean_labels, all_superclean)

    # Define superclean sets
    union_superclean = set(all_superclean[0]) | set(all_superclean[1])
    intersection_superclean = set(all_superclean[0]) & set(all_superclean[1])
    
    union_num_superclean = len(union_superclean)
    intersection_num_superclean = len(intersection_superclean)
    total_instances = len(eval_loader.dataset)
    
    # Compute statistics for union
    percentage_superclean = (union_num_superclean / total_instances) * 100
    true_clean_indices = set(inds_clean)
    correct_superclean = len(union_superclean & true_clean_indices)
    superclean_accuracy = (correct_superclean / union_num_superclean) * 100 if union_num_superclean > 0 else 0.0

    # Write results for union
    if threshold:
        log_file.write(f"Epoch: {epoch}, threshold {threshold}\n")
    else:
        log_file.write(f"Epoch: {epoch}\n")

    log_file.write(f"Number of superclean instances: {union_num_superclean}\n")
    log_file.write(f"Percentage of superclean instances: {percentage_superclean:.2f}%\n")
    log_file.write(f"Accuracy of superclean instances (clean vs noisy): {superclean_accuracy:.2f}%\n")
    log_file.write(f"Accuracy of the superclean instances (true class): {acc_union:.2f}%\n")
    log_file.write(f"Accuracy of the NOT superclean instances (true class): {acc_union_comp:.2f}%\n")
    log_file.write("-" * 50 + "\n")
    log_file.flush()

    # Compute statistics for intersection
    percentage_superclean = (intersection_num_superclean / total_instances) * 100
    correct_superclean = len(intersection_superclean & true_clean_indices)
    superclean_accuracy = (correct_superclean / intersection_num_superclean) * 100 if intersection_num_superclean > 0 else 0.0

    # Write results for intersection
    log_file.write(f"Number of superclean instances (intersection): {intersection_num_superclean}\n")
    log_file.write(f"Percentage of superclean instances (intersection): {percentage_superclean:.2f}%\n")
    log_file.write(f"Accuracy of superclean instances (intersection) (clean vs noisy): {superclean_accuracy:.2f}%\n")
    log_file.write(f"Accuracy of the superclean instances (intersection) (true class): {acc_intersection:.2f}%\n")
    log_file.write(f"Accuracy of the NOT superclean instances (intersection) (true class): {acc_intersection_comp:.2f}%\n")
    log_file.write("-" * 50 + "\n")
    log_file.flush()

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

def get_metrics_superclean2(predicted, clean_labels, all_superclean):
    # Initialize accuracy counters
    acc = 0
    # Convert clean labels to tensor
    clean_labels_tensor = torch.tensor(clean_labels, dtype=torch.long)

    # Compute accuracy for union superclean
    if len(all_superclean) > 0:
        correct_union = (predicted[all_superclean] == clean_labels_tensor[all_superclean]).sum().item()
        acc = correct_union / len(all_superclean)
    return 100*acc


def get_metrics_superclean(predicted, clean_labels, all_superclean):
    # Initialize accuracy counters
    acc_union = acc_intersection = 0
    acc_union_comp = acc_intersection_comp = 0

    # Define superclean sets
    union_superclean = set(all_superclean[0]) | set(all_superclean[1])  # Union of both sets
    intersection_superclean = set(all_superclean[0]) & set(all_superclean[1])  # Intersection
    total_samples = len(clean_labels)  # Assuming 50,000 samples, use dynamic length

    # Convert clean labels to tensor
    clean_labels_tensor = torch.tensor(clean_labels, dtype=torch.long)

    # Get indices of union and intersection sets
    union_indices = torch.tensor(list(union_superclean))
    intersection_indices = torch.tensor(list(intersection_superclean))

    # Compute accuracy for union superclean
    if len(union_indices) > 0:
        correct_union = (predicted[union_indices] == clean_labels_tensor[union_indices]).sum().item()
        acc_union = correct_union / len(union_indices)

    # Compute accuracy for intersection superclean
    if len(intersection_indices) > 0:
        correct_intersection = (predicted[intersection_indices] == clean_labels_tensor[intersection_indices]).sum().item()
        acc_intersection = correct_intersection / len(intersection_indices)

    # Compute accuracy for the complement of union
    union_comp_indices = torch.tensor([i for i in range(total_samples) if i not in union_superclean])
    if len(union_comp_indices) > 0:
        correct_union_comp = (predicted[union_comp_indices] == clean_labels_tensor[union_comp_indices]).sum().item()
        acc_union_comp = correct_union_comp / len(union_comp_indices)

    # Compute accuracy for the complement of intersection
    intersection_comp_indices = torch.tensor([i for i in range(total_samples) if i not in intersection_superclean])
    if len(intersection_comp_indices) > 0:
        correct_intersection_comp = (predicted[intersection_comp_indices] == clean_labels_tensor[intersection_comp_indices]).sum().item()
        acc_intersection_comp = correct_intersection_comp / len(intersection_comp_indices)

    return 100*acc_union, 100*acc_intersection, 100*acc_union_comp, 100*acc_intersection_comp



"""class_number = 100
dataset_name = 'cifar100'
dataset_path = './cifar-100'
number_epochs = 300"""

class_number = 10
dataset_name = 'cifar10'
dataset_path = './cifar-10-batches-py'
number_epochs = 30





argum=[
    argparse.Namespace(
    batch_size=64,
    lr=0.02,
    noise_mode='sym',
    alpha=4,
    lambda_u=0,
    lambda_c = 0.025,
    p_threshold=0.5,
    T=0.5,
    num_epochs=number_epochs,
    num_clean=5,
    r=0.2,
    id='',
    seed=123,
    gpuid=0,
    run=0,
    num_class=class_number,
    data_path= dataset_path,
    dataset='cifar100',
    ann_step = 0.5,
    uncertainty = True, 
    evidence_factor = 1/10,
    gce_loss = False,
    sim_clr = False,
    mix_clr = False,
    use_pretrained = False,
    edl_loss = edl_log_loss,
    edl_activation = softplus_evidence,
    use_loss = False,
    name_exp = "0.2_ctnt"),

    argparse.Namespace(
    batch_size=64,
    lr=0.02,
    noise_mode='sym',
    alpha=4,
    lambda_u=0,
    lambda_c = 0.025,
    p_threshold=0.5,
    T=0.5,
    num_epochs=number_epochs,
    num_clean=5,
    r=0.2,
    id='',
    seed=123,
    gpuid=0,
    run=0,
    num_class=class_number,
    data_path= dataset_path,
    dataset='cifar100',
    ann_step = 0,
    uncertainty = True, 
    evidence_factor = 1/10,
    gce_loss = False,
    sim_clr = False,
    mix_clr = False,
    use_pretrained = False,
    edl_loss = edl_log_loss,
    edl_activation = softplus_evidence,
    use_loss = True,
    name_exp = "0.2_0"),

    argparse.Namespace(
    batch_size=64,
    lr=0.02,
    noise_mode='sym',
    alpha=4,
    lambda_u=0,
    lambda_c = 0.025,
    p_threshold=0.5,
    T=0.5,
    num_epochs=number_epochs,
    num_clean=5,
    r=0.8,
    id='',
    seed=123,
    gpuid=0,
    run=0,
    num_class=class_number,
    data_path= dataset_path,
    dataset='cifar100',
    ann_step = 0.5,
    uncertainty = True, 
    evidence_factor = 1/10,
    gce_loss = False,
    sim_clr = False,
    mix_clr = False,
    use_pretrained = False,
    edl_loss = edl_log_loss,
    edl_activation = softplus_evidence,
    use_loss = False,
    name_exp = "0.8_ctnt"),

    argparse.Namespace(
    batch_size=64,
    lr=0.02,
    noise_mode='sym',
    alpha=4,
    lambda_u=0,
    lambda_c = 0.025,
    p_threshold=0.5,
    T=0.5,
    num_epochs=number_epochs,
    num_clean=5,
    r=0.8,
    id='',
    seed=123,
    gpuid=0,
    run=0,
    num_class=class_number,
    data_path= dataset_path,
    dataset='cifar100',
    ann_step = 0,
    uncertainty = True, 
    evidence_factor = 1/10,
    gce_loss = False,
    sim_clr = False,
    mix_clr = False,
    use_pretrained = False,
    edl_loss = edl_log_loss,
    edl_activation = softplus_evidence,
    use_loss = True,
    name_exp = "0.8_0"),

    argparse.Namespace(
    batch_size=64,
    lr=0.02,
    noise_mode='sym',
    alpha=4,
    lambda_u=0,
    lambda_c = 0.025,
    p_threshold=0.5,
    T=0.5,
    num_epochs=number_epochs,
    num_clean=5,
    r=0.2,
    id='',
    seed=123,
    gpuid=0,
    run=0,
    num_class=class_number,
    data_path= dataset_path,
    dataset='cifar100',
    ann_step = 0.01,
    uncertainty = True, 
    evidence_factor = 1/10,
    gce_loss = False,
    sim_clr = False,
    mix_clr = False,
    use_pretrained = False,
    edl_loss = edl_log_loss,
    edl_activation = softplus_evidence,
    use_loss = True,
    name_exp = "0.2_0.01"),

    argparse.Namespace(
    batch_size=64,
    lr=0.02,
    noise_mode='sym',
    alpha=4,
    lambda_u=0,
    lambda_c = 0.025,
    p_threshold=0.5,
    T=0.5,
    num_epochs=number_epochs,
    num_clean=5,
    r=0.8,
    id='',
    seed=123,
    gpuid=0,
    run=0,
    num_class=class_number,
    data_path= dataset_path,
    dataset='cifar100',
    ann_step = 0.01,
    uncertainty = True, 
    evidence_factor = 1/10,
    gce_loss = False,
    sim_clr = False,
    mix_clr = False,
    use_pretrained = False,
    edl_loss = edl_log_loss,
    edl_activation = softplus_evidence,
    use_loss = True,
    name_exp = "0.8_0.01"),

    argparse.Namespace(
    batch_size=64,
    lr=0.02,
    noise_mode='sym',
    alpha=4,
    lambda_u=0,
    lambda_c = 0.025,
    p_threshold=0.5,
    T=0.5,
    num_epochs=number_epochs,
    num_clean=5,
    r=0.2,
    id='',
    seed=123,
    gpuid=0,
    run=0,
    num_class=class_number,
    data_path= dataset_path,
    dataset='cifar100',
    ann_step = 0.1,
    uncertainty = True, 
    evidence_factor = 1/10,
    gce_loss = False,
    sim_clr = False,
    mix_clr = False,
    use_pretrained = False,
    edl_loss = edl_log_loss,
    edl_activation = softplus_evidence,
    use_loss = True,
    name_exp = "0.2_0.1")]

a = [
    argparse.Namespace(
    batch_size=64,
    lr=0.02,
    noise_mode='sym',
    alpha=4,
    lambda_u=0,
    lambda_c = 0.025,
    p_threshold=0.5,
    T=0.5,
    num_epochs=number_epochs,
    num_clean=5,
    r=0.8,
    id='',
    seed=123,
    gpuid=0,
    run=0,
    num_class=class_number,
    data_path= dataset_path,
    dataset='cifar100',
    ann_step = 0.1,
    uncertainty = True, 
    evidence_factor = 1/10,
    gce_loss = False,
    sim_clr = False,
    mix_clr = False,
    use_pretrained = False,
    edl_loss = edl_log_loss,
    edl_activation = softplus_evidence,
    use_loss = True,
    name_exp = "0.8_0.1"),

    argparse.Namespace(
    batch_size=64,
    lr=0.02,
    noise_mode='sym',
    alpha=4,
    lambda_u=0,
    lambda_c = 0.025,
    p_threshold=0.5,
    T=0.5,
    num_epochs=number_epochs,
    num_clean=5,
    r=0.2,
    id='',
    seed=123,
    gpuid=0,
    run=0,
    num_class=class_number,
    data_path= dataset_path,
    dataset='cifar100',
    ann_step = 0.05,
    uncertainty = True, 
    evidence_factor = 1/10,
    gce_loss = False,
    sim_clr = False,
    mix_clr = False,
    use_pretrained = False,
    edl_loss = edl_log_loss,
    edl_activation = softplus_evidence,
    use_loss = True,
    name_exp = "0.2_0.05"),

    argparse.Namespace(
    batch_size=64,
    lr=0.02,
    noise_mode='sym',
    alpha=4,
    lambda_u=0,
    lambda_c = 0.025,
    p_threshold=0.5,
    T=0.5,
    num_epochs=number_epochs,
    num_clean=5,
    r=0.8,
    id='',
    seed=123,
    gpuid=0,
    run=0,
    num_class=class_number,
    data_path= dataset_path,
    dataset='cifar100',
    ann_step = 0.05,
    uncertainty = True, 
    evidence_factor = 1/10,
    gce_loss = False,
    sim_clr = False,
    mix_clr = False,
    use_pretrained = False,
    edl_loss = edl_log_loss,
    edl_activation = softplus_evidence,
    use_loss = True,
    name_exp = "0.8_0.05"),
]

for args in a:
    if args.dataset == 'cifar100':
        args.num_class=100
        args.data_path= './cifar-100'
        args.num_epochs = 200

    elif args.dataset == 'cifar10':
        args.num_class=10
        args.data_path= './cifar-10-batches-py'
        args.num_epochs = 10
    


    torch.cuda.set_device(args.gpuid)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

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
    ce_loss_sample = nn.CrossEntropyLoss(reduction='none')
    ce_loss = nn.CrossEntropyLoss()
    if args.noise_mode=='asym':
        conf_penalty = NegEntropy()
    gce_loss = robust_loss.GCELoss(args.num_class, gpu='0') #only used in warmup
    contrastive_criterion = Contrastive_loss.SupConLoss()
    
    exp_str = f"test_{args.name_exp}"

    if args.run >0:
        exp_str = exp_str + '_run%d'%args.run
    path_exp='./checkpoint/' + exp_str

    path_plot = os.path.join(path_exp, 'plots')

    Path(path_exp).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(path_exp, 'savedDicts')).mkdir(parents=True, exist_ok=True)
    Path(path_plot).mkdir(parents=True, exist_ok=True)


    incomplete = os.path.exists("./checkpoint/%s/model_ckpt.pth.tar"%(exp_str))
    print('Incomplete...', incomplete)

    if incomplete == False:
        stats_log=open('./checkpoint/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_stats.txt','w') 
        test_log=open('./checkpoint/%s/%s_%.2f_%s'%(exp_str,args.dataset,args.r,args.noise_mode)+'_acc.txt','w') 
        time_log=open('./checkpoint/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_time.txt','w') 
        superclean_log= open('./checkpoint/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_superclean.txt','w')

        relabeled_log= open('./checkpoint/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_relabeled.txt','w')
        superclean_relabeled_log= open('./checkpoint/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_superclean_relabeled.txt','w')

        extra_log= open('./checkpoint/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_extra.txt','w')
        superclean_extra_log= open('./checkpoint/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_superclean_extra.txt','w')
    else:    
        stats_log=open('./checkpoint/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_stats.txt','a') 
        test_log=open('./checkpoint/%s/%s_%.2f_%s'%(exp_str,args.dataset,args.r,args.noise_mode)+'_acc.txt','a') 
        time_log=open('./checkpoint/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_time.txt','a') 
        superclean_log= open('./checkpoint/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_superclean.txt','a')

        relabeled_log= open('./checkpoint/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_relabeled.txt','a')
        superclean_relabeled_log= open('./checkpoint/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_superclean_relabeled.txt','a')

        extra_log= open('./checkpoint/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_extra.txt','a')
        superclean_extra_log= open('./checkpoint/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_superclean_extra.txt','a')

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
        load_path = 'checkpoint/%s/model_ckpt.pth.tar'%(exp_str)
        ckpt = torch.load(load_path)
        resume_epoch = ckpt['epoch']+1
        print('resume_epoch....', resume_epoch)
        net1.load_state_dict(ckpt['state_dict1'])
        net2.load_state_dict(ckpt['state_dict2'])
        optimizer1.load_state_dict(ckpt['optimizer1'])
        optimizer2.load_state_dict(ckpt['optimizer2'])

        all_idx_view_labeled = ckpt['all_idx_view_labeled']
        all_idx_view_unlabeled = ckpt['all_idx_view_unlabeled']
        all_idx_view_labeled_margin = ckpt['all_idx_view_labeled_margin']
        all_idx_view_unlabeled_margin = ckpt['all_idx_view_unlabeled_margin']
        all_idx_view_labeled_extra = ckpt['all_idx_view_labeled_extra']
        all_idx_view_unlabeled_extra = ckpt['all_idx_view_unlabeled_extra']
        all_idx_view_labeled_relabeled = ckpt['all_idx_view_labeled_relabeled']
        all_idx_view_unlabeled_relabeled = ckpt['all_idx_view_unlabeled_relabeled']

        all_preds = ckpt['all_preds']
        hist_preds = ckpt['hist_preds']
        acc_hist = ckpt['acc_hist']
        all_loss = ckpt['all_loss']
        all_evidences = ckpt['all_evidences']
        all_margin_true_label = ckpt['all_margin_true_label']
        all_margins_labels = ckpt['all_margins_labels']

        eval_loss_hist = ckpt['eval_loss_hist']
        eval_acc_hist = ckpt['eval_acc_hist']
        test_acc_hist = ckpt['test_acc_hist']
        test_losses_hist = ckpt['test_losses_hist']
        loss_train = ckpt.get("loss_train", [[],[]])
        loss_train_x = ckpt.get("loss_train_x", [[],[]])
        loss_train_u = ckpt.get("loss_train_u", [[],[]])
        all_superclean_relabeled = ckpt['all_superclean_relabeled']
        all_superclean_extra = ckpt["all_superclean_extra"]

        superclean_path =os.path.join('hcs/', 'hcs_%s_%.2f_%s_cn%d_run%d.pth.tar'%(args.dataset, args.r, args.noise_mode,args.num_clean, args.run))
        ckpt = torch.load(superclean_path)
        all_superclean = ckpt['all_superclean']
        all_superclean_margin = ckpt["all_superclean_margin"]

    else:
        all_superclean = [[],[]]
        all_superclean_margin = [[],[]]
        all_superclean_relabeled = [[],[]]
        all_superclean_extra = [[],[]]

        all_idx_view_labeled = [[],[]]
        all_idx_view_unlabeled = [[], []]
        all_idx_view_labeled_margin = [[], []]
        all_idx_view_unlabeled_margin = [[], []]
        all_idx_view_labeled_extra = [[],[]]
        all_idx_view_unlabeled_extra = [[], []]
        all_idx_view_labeled_relabeled = [[], []]
        all_idx_view_unlabeled_relabeled = [[], []]

        all_preds = [[], []] # save the history of preds for two networks
        hist_preds = [[],[]]
        acc_hist = []
        all_loss = [[],[]] # save the history of losses from two networks
        all_evidences = [[],[]]
        all_margin_true_label = [[],[]]
        all_margins_labels = [[],[]]

        eval_loss_hist = [[], []]
        eval_acc_hist = [[], []]
        test_acc_hist = []
        test_losses_hist = []
        loss_train = [[],[]]
        loss_train_x = [[],[]]
        loss_train_u = [[],[]]


    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train') 
    noisy_labels = eval_loader.dataset.noise_label
    clean_labels = eval_loader.dataset.train_label 
    inds_noisy = np.asarray([ind for ind in range(len(noisy_labels)) if noisy_labels[ind] != clean_labels[ind]])
    inds_clean = np.delete(np.arange(len(noisy_labels)), inds_noisy)


    total_time =  0
    warmup_time = 0

    if resume_epoch == 201:
        all_superclean_relabeled
        all_idx_view_labeled_relabeled


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

            prob_loss1, prob_margin1, all_loss[0], all_preds[0], hist_preds[0], all_margin_true_label[0], all_margins_labels[0], all_evidences[0], eval_loss_hist[0], eval_acc_hist[0], correct_indices1, noisy_correct_indices1 = eval_train(net1, all_loss[0], all_preds[0], hist_preds[0], all_margin_true_label[0], all_margins_labels[0], all_evidences[0], eval_loss_hist[0], eval_acc_hist[0], clean_labels)
            prob_loss2, prob_margin2, all_loss[1], all_preds[1], hist_preds[1], all_margin_true_label[1], all_margins_labels[1], all_evidences[1], eval_loss_hist[1], eval_acc_hist[1], correct_indices2, noisy_correct_indices2 = eval_train(net2, all_loss[1], all_preds[1], hist_preds[1], all_margin_true_label[1], all_margins_labels[1], all_evidences[1], eval_loss_hist[1], eval_acc_hist[1], clean_labels) 


            all_superclean, pred_loss1, pred_loss2, all_idx_view_labeled, all_idx_view_unlabeled, indices_values = get_superclean(prob_loss1, prob_loss2, all_superclean, all_idx_view_labeled, all_idx_view_unlabeled)
            all_superclean_margin, pred_margin1, pred_margin2, all_idx_view_labeled_margin, all_idx_view_unlabeled_margin, indices_values_margin = get_superclean(prob_margin1, prob_margin2, all_superclean_margin, all_idx_view_labeled_margin, all_idx_view_unlabeled_margin)
            all_superclean_extra, _, _, all_idx_view_labeled_extra, all_idx_view_unlabeled_extra = get_superclean_extra(prob_loss1, prob_loss2, all_superclean_extra, all_idx_view_labeled_extra, all_idx_view_unlabeled_extra, threshold = 0.5)   


            if epoch==(warm_up-1):
                time_log.write('Warmup: %f \n'%(warmup_time))
                time_log.flush()  
            
        else:       
            print("training epoch ", epoch)
            start_time = time.time()
            prob_loss1, prob_margin1, all_loss[0], all_preds[0], hist_preds[0], all_margin_true_label[0], all_margins_labels[0], all_evidences[0], eval_loss_hist[0], eval_acc_hist[0], correct_indices1, noisy_correct_indices1 = eval_train(net1, all_loss[0], all_preds[0], hist_preds[0], all_margin_true_label[0], all_margins_labels[0], all_evidences[0], eval_loss_hist[0], eval_acc_hist[0], clean_labels)
            prob_loss2, prob_margin2, all_loss[1], all_preds[1], hist_preds[1], all_margin_true_label[1], all_margins_labels[1], all_evidences[1], eval_loss_hist[1], eval_acc_hist[1], correct_indices2, noisy_correct_indices2 = eval_train(net2, all_loss[1], all_preds[1], hist_preds[1], all_margin_true_label[1], all_margins_labels[1], all_evidences[1], eval_loss_hist[1], eval_acc_hist[1], clean_labels) 
            all_superclean, pred_loss1, pred_loss2, all_idx_view_labeled, all_idx_view_unlabeled, indices_values = get_superclean(prob_loss1, prob_loss2, all_superclean, all_idx_view_labeled, all_idx_view_unlabeled)
            all_superclean_margin, pred_margin1, pred_margin2, all_idx_view_labeled_margin, all_idx_view_unlabeled_margin, indices_values_margin = get_superclean(prob_margin1, prob_margin2, all_superclean_margin, all_idx_view_labeled_margin, all_idx_view_unlabeled_margin)

            end_time = round(time.time() - start_time)
            total_time+= end_time


            predicted_labels, predicted_labels1, predicted_labels2 = test_superclean(net1, net2)

            #write log superclean
            all_last_elements = [sublist[-1] for sublist in all_superclean]
            write_log_superclean(all_last_elements, predicted_labels, log_file=superclean_log)
            all_last_elements_margin = [sublist[-1] for sublist in all_superclean_margin]
            write_log_superclean(all_last_elements_margin, predicted_labels, log_file=superclean_log)

            #log superclean extra:
            threshold_extra = 1 - len(all_idx_view_labeled[0][-1])/50000
            all_superclean_extra, _, _, all_idx_view_labeled_extra, all_idx_view_unlabeled_extra = get_superclean_extra(prob_loss1, prob_loss2, all_superclean_extra, all_idx_view_labeled_extra, all_idx_view_unlabeled_extra, threshold = threshold_extra)
            filtered_elements = [all_idx_view_labeled_extra[0][-1], all_idx_view_labeled_extra[0][-1]]
            write_log_superclean(filtered_elements, predicted_labels1, log_file = extra_log, threshold = threshold_extra)
            
            filtered_elements = [all_idx_view_labeled_extra[0][-1], all_idx_view_labeled_extra[1][-1]]
            write_log_superclean(filtered_elements, predicted_labels, log_file = superclean_extra_log, threshold = threshold_extra)

            #write log superclean relabeled
            for threshold in [0.2, 0.4]:
                all_superclean_relabeled, _, _, all_idx_view_labeled_relabeled, all_idx_view_unlabeled_relabeled = get_superclean_relabeled(all_margins_labels[0][-1], all_margins_labels[1][-1], all_superclean_relabeled, all_idx_view_labeled_relabeled, all_idx_view_unlabeled_relabeled, threshold = threshold)  
                all_last_elements_relabel = [sublist[-1] for sublist in all_idx_view_labeled_relabeled]
                filtered_elements = [[index for index in all_last_elements_relabel[0] if index not in all_idx_view_labeled[0][-1]], [index for index in all_last_elements_relabel[0] if index not in all_idx_view_labeled[0][-1]]]
                write_log_superclean(filtered_elements, predicted_labels1, log_file= relabeled_log, threshold=threshold)
                filtered_elements = [[index for index in all_last_elements_relabel[0] if index not in all_idx_view_labeled[0][-1]], [index for index in all_last_elements_relabel[1] if index not in all_idx_view_labeled[1][-1]]]
                write_log_superclean(filtered_elements, predicted_labels, log_file= superclean_relabeled_log, threshold=threshold)

            start_time = time.time()
            if args.use_loss:
                print('Train Net1')
                labeled_trainloader, unlabeled_trainloader, _ = loader.run('train',pred_loss2,prob_loss2) # co-divide
                loss_train[0], loss_train_x[0], loss_train_u[0] = train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader, loss_train[0], loss_train_x[0], loss_train_u[0], savelog=True) # train net1  
                
                print('\nTrain Net2')
                labeled_trainloader, unlabeled_trainloader, u_map_trainloader = loader.run('train',pred_loss1,prob_loss1) # co-divide
                loss_train[1], loss_train_x[1], loss_train_u[1] = train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader, loss_train[1], loss_train_x[1], loss_train_u[1], savelog=False) # train net2         
            else:
                print('Train Net1')
                labeled_trainloader, unlabeled_trainloader, _ = loader.run('train',pred2, prob1) # co-divide
                loss_train[0], loss_train_x[0], loss_train_u[0] = train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader, loss_train[0], loss_train_x[0], loss_train_u[0], savelog=True) # train net1  
                
                print('\nTrain Net2')
                labeled_trainloader, unlabeled_trainloader, u_map_trainloader = loader.run('train',pred_margin1,prob_margin1) # co-divide
                loss_train[1], loss_train_x[1], loss_train_u[1] = train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader, loss_train[1], loss_train_x[1], loss_train_u[1], savelog=False) # train net2                    
            
            end_time = round(time.time() - start_time)
            total_time+= end_time

        test_acc_hist, test_losses_hist = test(epoch,net1,net2, test_acc_hist, test_losses_hist)
        if (epoch == warm_up -1) or (epoch%5==0 and epoch !=0):
            plot_hist_curve_loss_test(data_hist= test_losses_hist, path=path_plot, epoch=epoch )
            if epoch>=warm_up:
                plot_curve_loss_train(data_hist=[loss_train[0], loss_train_x[0], loss_train_u[0]], path=path_plot)

        if (epoch == warm_up -1) or (epoch%5==0 and epoch !=0):
                if args.use_loss:
                    print("Plots...")
                    plot_curve_loss(data_hist= eval_loss_hist[0], inds_clean=inds_clean, inds_noisy=inds_noisy, path=path_plot, epoch=epoch )
                    plot_curve_accuracy(data_hist= eval_acc_hist[0], inds_clean=inds_clean, inds_noisy=inds_noisy, path=path_plot, epoch=epoch )
                    plot_histogram_metric(data_hist=all_loss[0], inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= all_idx_view_labeled[0][-1], thresholds = indices_values, path=path_plot, epoch=epoch, metric = "Loss"  )
                    if args.uncertainty:
                        plot_histogram_metric(data_hist=all_margin_true_label[0], inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= all_idx_view_labeled_margin[0][-1], thresholds = indices_values_margin, path=path_plot, epoch=epoch, metric = "Margins_true"  )
                        plot_histogram_metric2(data_hist=all_evidences[0], inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= all_idx_view_labeled[0][-1], thresholds = [], path=path_plot,  epoch=epoch, metric = "evidence"  )
                        plot_histogram_metric2(data_hist=all_margins_labels[0], inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= all_idx_view_labeled[0][-1], thresholds = [], path=path_plot, epoch=epoch, metric = "Margins"  )
                    print("Plots finished")
                else:
                    print("Plots...")
                    plot_curve_loss(data_hist= eval_loss_hist[0], inds_clean=inds_clean, inds_noisy=inds_noisy, path=path_plot, epoch=epoch )
                    plot_curve_accuracy(data_hist= eval_acc_hist[0], inds_clean=inds_clean, inds_noisy=inds_noisy, path=path_plot, epoch=epoch )
                    plot_histogram_metric(data_hist=all_loss[0], inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= all_idx_view_labeled[0][-1], thresholds = indices_values, path=path_plot, epoch=epoch, metric = "Loss"  )
                    if args.uncertainty:
                        plot_histogram_metric(data_hist=all_margin_true_label[0], inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= all_idx_view_labeled_margin[0][-1], thresholds = indices_values_margin, path=path_plot, epoch=epoch, metric = "Margins_true"  )
                        plot_histogram_metric2(data_hist=all_evidences[0], inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= all_idx_view_labeled_margin[0][-1], thresholds = [], path=path_plot,  epoch=epoch, metric = "evidence"  )
                        plot_histogram_metric2(data_hist=all_margins_labels[0], inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= all_idx_view_labeled_margin[0][-1], thresholds = [], path=path_plot, epoch=epoch, metric = "Margins"  )
                    print("Plots finished")
        
        save_models(path_exp)
        

    #test_log.write('\nBest:%.2f  avgLast10: %.2f\n'%(max(test_acc_hist),sum(test_acc_hist[-10:])/10.0))
    test_log.close() 

    time_log.write('SSL Time: %f \n'%(total_time-warmup_time))
    time_log.write('Total Time: %f \n'%(total_time))
    time_log.close()

    superclean_log.close()
    superclean_relabeled_log.close()
    relabeled_log.close()
    extra_log.close()
    superclean_extra_log.close()

    #check hist of predclean net 1

    relabeled_log2= open('./checkpoint/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_relabeled_2.txt','w')
    all = [[], []]
    all_lab_idx = [[], []]
    all_unlab_idx = [[], []]


    for epoch_count in range(1, 171):
        epoch = 200 - epoch_count
        idx = all_idx_view_labeled[0][-epoch_count] 
        margins_labeled1 = all_margins_labels[0][-epoch_count][idx]

        idx = all_idx_view_unlabeled[0][-epoch_count] 
        margins_unlabeled1 = all_margins_labels[0][-epoch_count][idx]

        idx = all_idx_view_labeled[1][-epoch_count] 
        margins_labeled2 = all_margins_labels[1][-epoch_count][idx]

        idx = all_idx_view_unlabeled[1][-epoch_count] 
        margins_unlabeled2 = all_margins_labels[1][-epoch_count][idx]

        # Fixed string formatting issues
        relabeled_log2.write(f'epoch count -: \n{epoch_count}\n')
        relabeled_log2.write(f'mean margins labeled : \n{margins_labeled1.mean()}\n')
        relabeled_log2.write(f'mean margins unlabeled : \n{margins_unlabeled1.mean()}\n')

        predicted_labels1 = hist_preds[0][-epoch_count]
        predicted_labels2 = hist_preds[1][-epoch_count]
        predicted_labels = (predicted_labels1+predicted_labels2).argmax(dim=1)
        predicted_labels1 = predicted_labels1.argmax(dim=1)
        predicted_labels2 = predicted_labels2.argmax(dim=1)

        diff_indices = np.where(predicted_labels1 != predicted_labels2)[0]

        _, _, _, indices, _ = get_superclean_relabeled(
            all_margins_labels[0][-epoch_count], 
            all_margins_labels[1][-epoch_count], 
            all, all_lab_idx, all_unlab_idx, 
            threshold=margins_labeled1.mean()
        )

        filtered_elements = [
            list(set(indices[0][-1]) - (set(all_idx_view_labeled[0][-epoch_count]) - set(diff_indices))),
            list(set(indices[1][-1]) - (set(all_idx_view_labeled[1][-epoch_count]) - set(diff_indices)))
        ]


        write_log_superclean2(filtered_elements, predicted_labels1, predicted_labels2, log_file=relabeled_log2)
        _, _, _, indices, _ = get_superclean_relabeled(
            (all_margins_labels[0][-epoch_count]+all_margins_labels[1][-epoch_count])/2, 
            (all_margins_labels[0][-epoch_count]+all_margins_labels[1][-epoch_count])/2, 
            all, all_lab_idx, all_unlab_idx, 
            threshold=margins_labeled1.mean()
        )
        filtered_elements = [
            list(set(indices[0][-1]) - (set(all_idx_view_labeled[0][-epoch_count]) - set(diff_indices))),
            list(set(indices[0][-1]) - (set(all_idx_view_labeled[1][-epoch_count]) - set(diff_indices)))
        ]
        write_log_superclean(filtered_elements, predicted_labels, log_file = relabeled_log2)
    relabeled_log2.close()




    log_intersection= open('./checkpoint/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_superclean_intersection.txt','w')

    for epoch_count in range(1, 171):
        epoch = 200 - epoch_count
        loss_superclean1 = all_superclean[0][-epoch_count] 
        loss_superclean2 = all_superclean[1][-epoch_count]
        margin_superclean1 = all_superclean_margin[0][-epoch_count] 
        margin_superclean2 = all_superclean_margin[1][-epoch_count] 

        loss_intersection = list(set(loss_superclean1) & set(loss_superclean2))
        margin_intersection = list(set(margin_superclean1) & set(margin_superclean2))

        predicted_labels1 = hist_preds[0][-epoch_count]
        predicted_labels2 = hist_preds[1][-epoch_count]
        predicted_labels = (predicted_labels1+predicted_labels2).argmax(dim=1)

        filtered_elements = [loss_intersection, margin_intersection]

        write_log_superclean(filtered_elements, predicted_labels, log_file=log_intersection)

    log_intersection.close()



