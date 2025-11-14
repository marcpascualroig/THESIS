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
from PreResNet_tiny import *
from sklearn.mixture import GaussianMixture
from dataloader_tiny2 import tinyImagenet_dataloader as dataloader
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

import Contrastive_loss, Contrastive_loss2

contrastive_criterion = Contrastive_loss2.SupConLoss()
sns.set()
def percentile_normalize(tensor, lower=1.0, upper=99.0):
    # Convert percentiles to thresholds
    lower_val = torch.quantile(tensor, lower / 100.0)
    upper_val = torch.quantile(tensor, upper / 100.0)

    # Clamp values to [lower_val, upper_val]
    clipped = tensor.clamp(min=lower_val.item(), max=upper_val.item())

    # Normalize to [0, 1]
    norm = (clipped - clipped.min()) / (clipped.max() - clipped.min() + 1e-8)
    return norm

def conv_p(logits):
    # 10/n_class
    alpha_t = args.edl_activation(logits)+args.evidence_factor
    total_alpha_t = torch.sum(alpha_t, dim=1, keepdim=True)
    expected_p = alpha_t / total_alpha_t
    return expected_p

def consistency_loss(output1, output2):            
    preds1 = conv_p(output1).detach()
    preds2 = torch.log(conv_p(output2))
    loss_kldiv = F.kl_div(preds2, preds1, reduction='none')
    loss_kldiv = torch.sum(loss_kldiv, dim=1)
    return loss_kldiv

def train(epoch,net,net2,optimizer,labeled_trainloader, unlabeled_trainloader, all_train_loss, all_train_loss_x, all_train_loss_u, all_train_loss_contrastive, op, vacuity, margin, dissonance, savelog=False):
    net.train()
    net2.eval() #fix one network and train the other

    #VACUITY WEIGHTS IN MIXMATCH
    weights = vacuity
    print("weights mean: ", weights.mean(), weights.min(), weights.max())

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

                #EDL
                evidence_u11 = args.edl_activation(outputs_u11)
                evidence_u12 = args.edl_activation(outputs_u12)
                evidence_u21 = args.edl_activation(outputs_u21)
                evidence_u22 = args.edl_activation(outputs_u22)
                alpha_u = evidence_u11/4 + evidence_u12/4 + evidence_u21/4 + evidence_u22/4 + args.evidence_factor
                S_u = torch.sum(alpha_u, dim=1, keepdim=True)
                pu = alpha_u / S_u
                ptu = pu**(1/args.T)

                # label refinement of labeled samples
                evidence_x = args.edl_activation(outputs_x)
                evidence_x2 = args.edl_activation(outputs_x2)
                alpha_x = evidence_x/2 + evidence_x2/2 + args.evidence_factor
                S_x = torch.sum(alpha_x, dim=1, keepdim=True)
                px = alpha_x / S_x
                device = pu.device
                px = w_x*labels_x + (1-w_x)*px
                ptx = px**(1/args.T) # temparature sharpening

                targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
                targets_u = targets_u.detach()
                targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize
                targets_x = targets_x.detach()
            
            # mixmatch
            all_inputs_aug = torch.cat([inputs_x3, inputs_x4, inputs_u3, inputs_u4], dim=0)
            all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)
            idx = torch.randperm(all_inputs_aug.size(0))
            input_a, input_b = all_inputs_aug, all_inputs_aug[idx]
            target_a, target_b = all_targets, all_targets[idx]
            l = np.random.beta(args.alpha, args.alpha, size=(all_inputs_aug.size(0), 1, 1, 1))  # Ensure broadcasting shape
            l = np.maximum(l, 1-l)
            #l = torch.ones((all_inputs_aug.size(0), 1, 1, 1), device=all_inputs_aug.device)
            l = torch.from_numpy(l).float().cuda()
            #l = l.float() 

            weights_all = torch.cat([weights[indices_x], weights[indices_x], weights[indices_u], weights[indices_u]], dim=0)
            weights_all = weights_all.to(l.device)
            weights_all = weights_all.view(-1, 1, 1, 1).to(l.device)
            l = 0.5*(1+l*weights_all)
            l_targets = l.view(all_inputs_aug.size(0), 1)

            mixed_input_aug = l * input_a + (1 - l) * input_b
            mixed_target = l_targets * target_a + (1 - l_targets) * target_b

            logits, aug_features = net(mixed_input_aug, forward_pass='cls_proj')

            #SUPERVISED AND UNSUPERVISED LOSS
            evidence = args.edl_activation(logits)
            alpha = evidence + args.evidence_factor
            S = torch.sum(alpha, dim=1, keepdim=True)
            probs = alpha / S
            pred_mean = probs.mean(0)
            outputs_x = logits[:batch_size*2]
            probs_u = probs[batch_size*2:]
            outputs_u = logits[batch_size*2:]
            Lx, Lu, lamb = criterion(outputs_x, mixed_target[:batch_size*2], probs_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up, outputs_u)

            #PLR LOSS
            if args.plr_loss:
                bs_x = inputs_x.size(0)
                bs_u = inputs_u.size(0)
                all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
                all_features = net(all_inputs, forward_pass='proj')
                if all_features.size(0) != 2*bs_x + 2*bs_u:
                    print("Error: Mismatch in total feature size!")
                    exit(1)

                indices = torch.cat((indices_x, indices_u), dim=0)
                eval_outputs = op[indices, :]
                labels_u = labels_u.cuda()
                labels_x = torch.argmax(labels_x, dim=1)
                labels = torch.cat((labels_x, labels_u), dim=0)
                    
                contrastive_mask = Contrastive_loss.build_mask_step(eval_outputs, topK, labels, device) 
                    
                start = 0
                fx3 = all_features[start:start + bs_x]
                start += bs_x
                fx4 = all_features[start:start + bs_x]
                start += bs_x
                fu3 = all_features[start:start + bs_u]
                start += bs_u
                fu4 = all_features[start:start + bs_u]
                f1, f2 = torch.cat([fx3, fu3], dim=0), torch.cat([fx4, fu4], dim=0)
                all_features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                weights_all =None
                loss_plr = plr_loss(all_features, mask=contrastive_mask, sample_weights = weights_all) #unsupervised loss: positive examples are 2 augmentations of the same sample only. Negative examples are samples that do no share the same class between the topk predicted classes
            else:
                loss_plr=0

            #regularization
            prior = torch.ones(args.num_class)/args.num_class
            prior = prior.cuda()
            penalty = torch.sum(prior*torch.log(prior/pred_mean))

            loss_plr=0
            penalty=0

            cl = loss_plr
            contrastive_loss += cl
            if cl == 0:
                loss = Lx + lamb * Lu + penalty
            else:
                loss = Lx + lamb * Lu + penalty + cl
            train_loss += loss
            train_loss_lx += Lx
            train_loss_u += Lu*lamb
            train_loss_penalty += penalty



            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cont_iters = cont_iters + 1
            if cont_iters == max_iters:
                break

    all_train_loss.append(train_loss)
    all_train_loss_x.append(train_loss_lx)
    all_train_loss_u.append(train_loss_u)
    all_train_loss_contrastive.append(contrastive_loss)
    print("loss ", train_loss_lx, contrastive_loss, train_loss_u, train_loss_penalty)

    return all_train_loss, all_train_loss_x, all_train_loss_u, all_train_loss_contrastive



def warmup(epoch,net,optimizer,dataloader,savelog=False):
    net.train()
    wm_loss = 0
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader): 
        batch_size = inputs.size(0)
        y = torch.zeros(batch_size, args.num_class).scatter_(1, labels.view(-1,1), 1)    
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs) 
                      
        loss, _ = edl_loss(outputs, y.float(), epoch_num=epoch, num_classes = args.num_class, annealing_step= args.ann_step, activation = args.edl_activation, evidence_factor = args.evidence_factor)

        if args.noise_mode=='asym':  # penalize confident prediction for asymmetric noise
            penalty = conf_penalty(outputs)
            L = loss + penalty      
        elif args.noise_mode=='sym':   
            L = loss
        else:
            L=loss

        wm_loss += L
        L.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Avg Iter Loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, wm_loss.item()/num_iter))
        sys.stdout.flush()
    return wm_loss


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

            loss, losses = edl_loss((outputs1+outputs2)/2, y.float(), epoch_num=epoch, num_classes = args.num_class, annealing_step= args.ann_step, activation = args.edl_activation, evidence_factor = args.evidence_factor)
            evidence_1 = args.edl_activation(outputs1)
            alpha_1 = evidence_1 + args.evidence_factor
            predicted1 = alpha_1 / torch.sum(alpha_1, dim=1, keepdim=True)
            evidence_2 = args.edl_activation(outputs2)
            alpha_2 = evidence_2 + args.evidence_factor
            predicted2 = alpha_2 / torch.sum(alpha_2, dim=1, keepdim=True)
            outputs = (predicted1+predicted2)/2
            _, predicted = torch.max(outputs, 1) 

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


def compute_entropy(outputs, act):
    evidence = act(outputs)
    alpha = evidence + args.evidence_factor
    S = torch.sum(alpha, dim=1, keepdim=True)
    prob_mass = alpha / S
    belief_mass = evidence / S 


    # Compute entropy
    entropy = -torch.sum(
        prob_mass * (torch.digamma(alpha + 1) - torch.digamma(S + 1)),
        dim=1
    )
    entropy = (entropy - entropy.min())/(entropy.max() - entropy.min())

    #margin
    top2_vals, _ = torch.topk(outputs, k=2, dim=1)
    margin = (top2_vals[:, 0] - top2_vals[:, 1])
    margin = (margin - margin.min())/(margin.max() - margin.min())
    margin = 1-margin
        
    #vacuity
    vacuity = args.num_class / S
    vacuity = percentile_normalize(vacuity, lower=1.0, upper=99.0)

    # dissonance
    dissonance = torch.zeros_like(entropy)  # Initialize with zeros
    for c in range(args.num_class):
        bc = belief_mass[:, c]  # b_c (belief for class c)
        other_b = torch.cat((belief_mass[:, :c], belief_mass[:, c+1:]), dim=1)  # b_i (other classes)
        balance = 1 - torch.abs(other_b - bc.unsqueeze(1)) / (other_b + bc.unsqueeze(1) + 1e-10)
        balance[other_b == 0] = 0  # If b_i or b_c is zero, balance is 0
        # Sum over all other classes
        dissonance += bc * torch.sum(other_b * balance, dim=1) / (torch.sum(other_b, dim=1) + 1e-10)
    dissonance = dissonance.flatten()

    dissonance = percentile_normalize(dissonance, lower=1.0, upper=99.0)

    return entropy, vacuity, margin, dissonance


def eval_train(model, clean_labels, net_idx, save_tensors = True, savelog=False):
    model.eval()
    correct_indices = []
    noisy_correct_indices = []

    losses = torch.zeros(len(eval_loader.dataset))
    
    preds = torch.zeros(len(eval_loader.dataset))
    preds_classes = torch.zeros(len(eval_loader.dataset), args.num_class)
    outputs_tensor = torch.zeros(len(eval_loader.dataset), args.num_class)
    eval_loss = train_acc = acc_clean = acc_noisy = 0

    losses_proto = torch.zeros(len(eval_loader.dataset))
    margins = torch.zeros(len(eval_loader.dataset))
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

            #EDL
            loss, loss_per_sample = edl_loss(outputs, y.float(), epoch_num=epoch, num_classes = args.num_class, annealing_step= args.ann_step, activation = args.edl_activation, evidence_factor = args.evidence_factor)
            evidence = args.edl_activation(outputs)
            alpha = evidence + args.evidence_factor  
            eval_preds = alpha / torch.sum(alpha, dim=1, keepdim=True)
            eval_loss += loss

            #prototype loss
            if args.bpt:
                loss_proto = ce_loss_sample(logits_proto, targets)
            
            _, pred = torch.max(outputs.data, -1)

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
                outputs_tensor[index[b]] = outputs[b]
                evidence_pos = outputs[b,targets[b]]
                copy_outputs = outputs[b].clone()
                copy_outputs[targets[b]] = -1e5
                evidence_neg = copy_outputs.max()
                margins[index[b]]=evidence_pos-evidence_neg

                if args.bpt:
                    losses_proto[index[b]] = loss_proto[b]
                    pl[index[b]] = pred[b]
                    op[index[b]] = outputs[b]
                    pt[index[b]] = logits_proto[b]
                    ft[index[b]] = features[b]
                    losses_proto = (losses_proto - losses_proto.min()) / (losses_proto.max() - losses_proto.min())
        

    losses = (margins-margins.min())/(margins.max() - margins.min())
    losses = 1 - losses

    if save_tensors:
        eval_loss_hist[net_idx].append(losses)
        all_preds[net_idx].append(preds)
        all_loss[net_idx].append(losses)
        hist_preds[net_idx].append(preds_classes)
        eval_acc_hist[net_idx].append([train_acc/len(eval_loader.dataset), acc_clean/len(inds_clean)])
        all_outputs[net_idx].append(outputs_tensor)
        all_proto_loss[net_idx].append(losses_proto)   

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
    prob = prob[:,gmm.means_.argmin()]

    entropy, vacuity, margin, dissonance = compute_entropy(outputs_tensor, args.edl_activation)
    return prob, correct_indices, {'op': op, 'pl': pl, 'pt': pt, 'ft': ft}, entropy, vacuity,margin, dissonance


def get_clean(prob, loss, net_idx, threshold=0.5):
    pred = (prob > threshold)      
    idx_view_labeled = (pred).nonzero()[0]
    idx_view_unlabeled = (1-pred).nonzero()[0]
    all_idx_view_labeled[net_idx].append(idx_view_labeled)
    all_idx_view_unlabeled[net_idx].append(idx_view_unlabeled)

    pred_loss = np.array([True if p in idx_view_labeled else False for p in range(len(pred))])

    loss_labeled = loss[idx_view_labeled]
    max_loss_labeled = loss_labeled.max() if len(loss_labeled) > 0 else 0
    return pred_loss, max_loss_labeled


def relabel_indices(metric, pred_loss, threshold, net_idx):
    mask = (metric.cpu().numpy() < threshold)  # move to CPU and convert to NumPy
    indices = np.where(mask)[0]
    filtered_indices = [i for i in indices if not pred_loss[i]]
    print(len(indices), len(filtered_indices))
    return np.array(filtered_indices)



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
    def __call__(self, outputs_x, targets_x, probs_u, targets_u, epoch, warm_up, outputs_u):
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
        #checkpoint = torch.load(f'../tiny/pretrained/all4one-tiny-preresnet18.ckpt')
        checkpoint = torch.load(
            f'../tiny/pretrained/all4one-tiny-preresnet18.ckpt',
            map_location=lambda storage, loc: storage.cuda(0)
        )
        if "state_dict" in checkpoint:       # Lightning ckpt format
            checkpoint_state = checkpoint["state_dict"]
        elif "model" in checkpoint:          # some methods wrap it here
            checkpoint_state = checkpoint["model"]
        else:                                # pure state_dict (.pth)
            checkpoint_state = checkpoint
        checkpoint_state = {k.replace("module.", ""): v for k, v in checkpoint_state.items()}
        model_state = model.state_dict()

        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"❌ Bad values in {name}: NaN/Inf detected")

        for name, buf in model.named_buffers():
            if torch.isnan(buf).any() or torch.isinf(buf).any():
                print(f"⚠️ Bad values in buffer {name}: NaN/Inf detected")


        # Cleaned versions of keys for fuzzy matching
        def clean_key(k: str) -> str:
            k = k.replace("module.", "")
            k = k.replace("encoder.", "")
            #k = k.replace("projector.", "")
            k = k.replace("backbone.", "")  
            k = k.replace("downsample", "shortcut") 
            return k
        
        excluded_layers = {
            "prototypes",
            "linear.weight",
            "linear.bias",
            "projector.0.weight",
            "projector.0.bias",
            "projector.2.weight",
            "projector.2.bias",
        }


        # Reverse lookup maps
        cleaned_ckpt = {clean_key(k): k for k in checkpoint_state}
        cleaned_model = {clean_key(k): k for k in model_state}

        matched, fuzzy_matched, shape_mismatched, only_in_ckpt, only_in_model = [], [], [], [], []

        # First try exact matches by clean name
        for ckpt_clean, ckpt_raw in cleaned_ckpt.items():
            if ckpt_clean in cleaned_model:
                model_raw = cleaned_model[ckpt_clean]
                if model_state[model_raw].shape == checkpoint_state[ckpt_raw].shape:
                    matched.append((ckpt_clean, model_raw, ckpt_raw))
                else:
                    shape_mismatched.append((ckpt_clean, model_state[model_raw].shape, checkpoint_state[ckpt_raw].shape))

            else:
                only_in_ckpt.append(ckpt_raw)

        for model_clean, model_raw in cleaned_model.items():
            if model_clean not in cleaned_ckpt:
                only_in_model.append(model_raw)

        # Save to file
        with open("comparison.txt", "w") as f:
            f.write("🔍 Fuzzy Layer Comparison Report (SimCLR)\n\n")

            f.write("✅ Perfectly Matched Layers (by logical name and shape):\n")
            for k_clean, model_k, ckpt_k in matched:
                if k_clean not in excluded_layers:
                    f.write(f"  - {k_clean} ← model: {model_k} == checkpoint: {ckpt_k}\n")

            f.write("\n⚠️ Same logical layer name but different shape (not loaded):\n")
            for k_clean, model_shape, ckpt_shape in shape_mismatched:
                f.write(f"  - {k_clean}: model {model_shape} vs checkpoint {ckpt_shape}\n")

            f.write("\n❌ Layers only in checkpoint (no matching in model):\n")
            for k in only_in_ckpt:
                ckpt_shape = tuple(checkpoint_state[k].shape)
                f.write(f"  - {k} : {ckpt_shape}\n")


            f.write("\n🚫 Excluded Layers (matched by name/shape but skipped):\n")
            for k_clean, model_k, ckpt_k in matched:
                if k_clean in excluded_layers:
                    model_shape = tuple(model_state[model_k].shape)
                    ckpt_shape = tuple(checkpoint_state[ckpt_k].shape)
                    f.write(
                        f"  - {k_clean} ← model: {model_k} {model_shape} "
                        f"vs checkpoint: {ckpt_k} {ckpt_shape} (❌ EXCLUDED)\n"
                    )

            f.write("\n❌ Layers only in model (not found in checkpoint):\n")
            for k in only_in_model:
                model_shape = tuple(model_state[k].shape)
                f.write(f"  - {k} : {model_shape}\n")

                


        checkpoint_to_load = {
            model_k: checkpoint_state[ckpt_k]
            for k_clean, model_k, ckpt_k in matched
            if k_clean not in excluded_layers
        }

        # Actually load
        model.load_state_dict(checkpoint_to_load, strict=False)

    model = model.cuda()
    return model


def write_log(idx, predicted, log_file, threshold = None):
    acc, prec, recall = get_metrics(predicted, clean_labels, idx, clean_indices=inds_clean)

    total_instances = len(eval_loader.dataset)
    percentage = (len(idx) / total_instances) * 100
    true_clean_indices = set(inds_clean)

    correct_superclean = len(set(idx) & true_clean_indices)
    clean_accuracy = (correct_superclean / len(idx)) * 100 if len(idx) > 0 else 0.0

    idx_unlabeled = torch.tensor(np.setdiff1d(np.arange(len(pred_loss1)), idx)).long()

    if len(idx_unlabeled) > 0:
        correct_unlabeled = (predicted[idx_unlabeled] == clean_labels[idx_unlabeled]).sum().item()
        acc_unlabeled = 100 * correct_unlabeled / len(idx_unlabeled)
    else:
        acc_unlabeled = 0.0

    if len(idx_unlabeled) > 0:
        correct_unlabeled = (predicted[idx_unlabeled] == clean_labels[idx_unlabeled]).sum().item()
        acc_unlabeled = 100 * correct_unlabeled / len(idx_unlabeled)
    else:
        acc_unlabeled = 0.0

    print("precision of labeled: ", prec, acc)
    print("accuracy unlabeled: ", acc_unlabeled)

    all_loss = loss_train[0][-1]
    sup_loss = loss_train_x[0][-1]
    unsupervised_loss = loss_train_u[0][-1]
    contrastive_loss = train_loss_contrastive[0][-1]
    
    log_file.write(f"Epoch: {epoch}\n")
    log_file.write(f"Number of superclean instances: {len(idx)}\n")
    log_file.write(f"Percentage of superclean instances: {percentage:.2f}%\n")
    log_file.write(f"Accuracy - precision - recall (with respect to clean indices): {acc:.2f}%  \t {prec:.2f}% \t {recall:.2f}%\n")
    log_file.write(f"Accuracy unlabeeld: {acc_unlabeled:.2f}%\n")

    log_file.write(f"Total Loss: {all_loss:.6f}\n")
    log_file.write(f"Supervised Loss: {sup_loss:.6f}\n")
    log_file.write(f"Unsupervised Loss: {unsupervised_loss:.6f}\n")
    log_file.write(f"Contrastive Loss: {contrastive_loss:.6f}\n")

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


                    'eval_acc_hist': eval_acc_hist,
                    'eval_loss_hist': eval_loss_hist,
                    'test_acc_hist': test_acc_hist,
                    'test_losses_hist': test_losses_hist,

                    'loss_train': loss_train,
                    'loss_train_x': loss_train_x,
                    'loss_train_u': loss_train_u,
                    'train_loss_contrastive' : train_loss_contrastive,

                    'relabel_idx_1': relabel_idx_1,
                    'relabel_idx_2': relabel_idx_2,

                    'new_labels_1': new_labels_1,
                    'new_labels_2': new_labels_2,

                    'clean_metrics': clean_metrics,
                    'correct_metrics': correct_metrics,
                    'all_outputs': all_outputs,

                    'threshold_loss': threshold_loss
                    })


    if epoch%1==0 or epoch ==4 or epoch ==29:
        fn2 = os.path.join(save_path, 'model_ckpt.pth.tar')
        torch.save(state, fn2)
        if not os.path.exists('hcs'):
            os.makedirs('hcs')
    if epoch ==4:
        fn2 = os.path.join(save_path, f'model_ckpt_epoch{epoch}.pth.tar')
        torch.save(state, fn2)

    if epoch == 29:
        fn2 = os.path.join(save_path, f'model_ckpt_epoch{epoch}.pth.tar')
        torch.save(state, fn2)

    if epoch == 79:
        fn2 = os.path.join(save_path, f'model_ckpt_epoch{epoch}.pth.tar')
        torch.save(state, fn2)
        
    if epoch == 120:
        fn2 = os.path.join(save_path, f'model_ckpt_epoch{epoch}.pth.tar')
        torch.save(state, fn2)

    if epoch == 180:
        fn2 = os.path.join(save_path, f'model_ckpt_epoch{epoch}.pth.tar')
        torch.save(state, fn2)



a = [
    argparse.Namespace(
    batch_size=32,
    lr=0.005,
    noise_mode='sym',
    alpha=0.5,
    p_threshold=0.5,
    T=0.5,
    num_epochs=200,
    r=0.2,
    id='',
    seed=123,
    gpuid=0,
    run=0,
    dataset='tiny-imagenet',
    ann_step = 0.005,
    evidence_factor = 10/200,
    use_pretrained = True,
    edl_loss = edl_log_loss,
    edl_activation = exp_evidence,
    lambda_u = 25,
    name_exp = "tiny_0.2_sym_alpha0.5_bs32_0.005",
    bpt=True,
    plr_loss = True
    ),
]


for args in a:
    args.num_class=200
    args.data_path= './data/tiny-imagenet-200'
    args.num_epochs = 200

    print(args.name_exp)
    print("epochs ", args.num_epochs)

    torch.cuda.set_device(args.gpuid)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    flat=False
    info_nce_loss = Contrastive_loss.InfoNCELoss(temperature=0.1,
                                    batch_size=args.batch_size * 2,
                                    flat=flat,
                                    n_views=2)
    plr_loss = Contrastive_loss.PLRLoss(flat=flat)
    
    #criterion
    edl_loss = args.edl_loss
    ce_loss_sample = nn.CrossEntropyLoss(reduction='none')
    ce_loss = nn.CrossEntropyLoss()
    if args.noise_mode=='asym':
        conf_penalty = NegEntropy()
    
    exp_str = f"{args.name_exp}"

    if args.run >0:
        exp_str = exp_str + '_run%d'%args.run
    path_exp='./checkpoint_tiny/' + exp_str

    path_plot = os.path.join(path_exp, 'plots')

    Path(path_exp).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(path_exp, 'savedDicts')).mkdir(parents=True, exist_ok=True)
    Path(path_plot).mkdir(parents=True, exist_ok=True)

    model_path = "./checkpoint_tiny/%s/model_ckpt.pth.tar"%(exp_str)
    incomplete = os.path.exists(model_path)
    print('Incomplete...', incomplete)

    if incomplete == False:
        log_mode = 'w'
    else:
        log_mode = 'a'
    stats_log=open('./checkpoint_tiny/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_stats.txt',log_mode) 
    test_log=open('./checkpoint_tiny/%s/%s_%.2f_%s'%(exp_str,args.dataset,args.r,args.noise_mode)+'_acc.txt',log_mode) 
    time_log=open('./checkpoint_tiny/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_time.txt',log_mode) 
    superclean_log= open('./checkpoint_tiny/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_superclean.txt',log_mode)


    warm_up = 30
    if args.use_pretrained:
        warm_up=5
    
    loader = dataloader(root=args.data_path, batch_size=args.batch_size, num_workers=4, log = stats_log, r = args.r, noise_mode = args.noise_mode, noise_file='%s/clean_%.2f_%s.npz'%(args.data_path,args.r, args.noise_mode))

    warmup_trainloader = loader.run('warmup')

    print('| Building net')
    if incomplete:
        args.use_pretrained = False
    net1 = create_model()
    net2 = create_model()
    cudnn.benchmark = True

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

        clean_metrics= ckpt['clean_metrics']
        correct_metrics = ckpt['correct_metrics']
        all_outputs = ckpt["all_outputs"]
        all_proto_loss = ckpt.get("all_proto_loss", [[],[]])

        threshold_loss = ckpt.get("threshold_loss", [[0.5],[0.5]])

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

        clean_metrics= []
        correct_metrics = []

        all_outputs = [[],[]]
        all_proto_loss = [[],[]]

        threshold_loss=[[0.5],[0.5]]


    criterion = SemiLoss()
    test_loader = loader.run('val')
    eval_loader = loader.run('eval_train') 

    noisy_labels = eval_loader.dataset.noise_label
    clean_labels = eval_loader.dataset.train_label 

    #inds_noisy = np.asarray([ind for ind in range(len(noisy_labels)) if noisy_labels[ind] != clean_labels[ind]])

    inds_noisy = np.asarray(
        [ind for ind in range(len(noisy_labels)) if noisy_labels[ind] != clean_labels[ind]],
        dtype=int
    )
    
    inds_clean = np.delete(np.arange(len(noisy_labels)), inds_noisy)
    total_time =  0
    warmup_time = 0
    wm_loss = []

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
            wm_loss1 = warmup(epoch,net1,optimizer1,warmup_trainloader, savelog=True)   
            wm_loss.append(wm_loss1) 
            print('\nWarmup Net2')
            wm_loss2 = warmup(epoch,net2,optimizer2,warmup_trainloader, savelog=False) 
            end_time = round(time.time() - start_time)
            total_time+= end_time
            warmup_time+= end_time

            if epoch==(warm_up-1):
                time_log.write('Warmup: %f \n'%(warmup_time))
                time_log.flush()

            
        else:       
            if epoch == warm_up and args.bpt:
                print("INITIALIZE PROTOTYPES...")
                Contrastive_loss.init_prototypes(net1, eval_loader, device)
                Contrastive_loss.init_prototypes(net2, eval_loader, device)

            print("training epoch ", epoch)       
            start_time = time.time()  

            prob_loss1, correct_indices1, op1, entr1, vac1, mar1, diss1 = eval_train(net1, clean_labels, net_idx = 0)
            prob_loss2, correct_indices2, op2, entr2, vac2, mar2, diss2 = eval_train(net2, clean_labels, net_idx = 1) 

            pred_loss1, threshold_loss_1 = get_clean(prob_loss1, all_loss[0][-1], net_idx = 0, threshold=0.5)
            pred_loss2, threshold_loss_2 = get_clean(prob_loss2, all_loss[1][-1], net_idx = 1, threshold=0.5)

            predicted_labels1 = torch.argmax(hist_preds[0][-1], dim=1)
            predicted_labels2 = torch.argmax(hist_preds[1][-1], dim=1)

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
            #write_log((pred_loss1).nonzero()[0], predicted_labels1, log_file=superclean_log)
            #write_log((pred_loss2).nonzero()[0], predicted_labels2, log_file=superclean_log)
            #write_log2((pred_loss1).nonzero()[0], predicted_labels1, log_file=superclean_log)

            end_time = round(time.time() - start_time)
            total_time+= end_time
            start_time = time.time()
              
            print('Train Net1')
            labeled_trainloader, unlabeled_trainloader = loader.run('train',pred_loss2,prob_loss2) # co-divide
            loss_train[0], loss_train_x[0], loss_train_u[0], train_loss_contrastive[0] = train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader, loss_train[0], loss_train_x[0], loss_train_u[0], train_loss_contrastive[0], op2['op'], vac2, mar2, diss2, savelog=True) # train net1  
                        
            print('\nTrain Net2')
            labeled_trainloader, unlabeled_trainloader = loader.run('train',pred_loss1,prob_loss1) # co-divide
            loss_train[1], loss_train_x[1], loss_train_u[1], train_loss_contrastive[1] = train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader, loss_train[1], loss_train_x[1], loss_train_u[1], train_loss_contrastive[1], op1['op'], vac1, mar1, diss1, savelog=False) # train net2         

            write_log((pred_loss1).nonzero()[0], predicted_labels1, log_file=superclean_log)
            
            if args.bpt:
                all_indices_x = torch.tensor(pred_loss2.nonzero()[0])

                print("PLR CORRECTION AND PROTOTYPES...")
                all_indices_x = torch.tensor(pred_loss2.nonzero()[0])
                clean_labels_x = Contrastive_loss.noise_correction(op2['pt'][all_indices_x, :],
                                                                op2['op'][all_indices_x, :],
                                                                op2['pl'][all_indices_x],
                                                                all_indices_x, device, args.edl_activation, args.evidence_factor)
                all_indices_u = torch.tensor((1 - pred_loss2).nonzero()[0])
                clean_labels_u = Contrastive_loss.noise_correction(op2['pt'][all_indices_u, :],
                                                                op2['op'][all_indices_u, :],
                                                                op2['pl'][all_indices_u],
                                                                all_indices_u, device, args.edl_activation, args.evidence_factor)
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
                                                                all_indices_x, device, args.edl_activation, args.evidence_factor)
                all_indices_u = torch.tensor((1 - pred_loss1).nonzero()[0])
                clean_labels_u = Contrastive_loss.noise_correction(op1['pt'][all_indices_u, :],
                                                                op1['op'][all_indices_u, :],
                                                                op1['pl'][all_indices_u],
                                                                all_indices_u, device, args.edl_activation, args.evidence_factor)
                # update class prototypes
                features = op1['ft'].to(device)
                labels = op1['pl'].to(device)
                labels[all_indices_x] = clean_labels_x
                labels[all_indices_u] = clean_labels_u
                net2.update_prototypes(features, labels)

            end_time = round(time.time() - start_time)
            total_time+= end_time
        
        test_acc_hist, test_losses_hist = test(epoch,net1,net2, test_acc_hist, test_losses_hist)

        save_models(path_exp)
        

    #save_models(path_exp)
    test_log.write('\nBest:%.2f  avgLast10: %.2f\n'%(max(test_acc_hist),sum(test_acc_hist[-10:])/10.0))
    test_log.close() 
    test_log.close() 

    time_log.write('SSL Time: %f \n'%(total_time-warmup_time))
    time_log.write('Total Time: %f \n'%(total_time))
    time_log.close()

    superclean_log.close()








