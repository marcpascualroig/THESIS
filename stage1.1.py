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
from utils_plot import plot_histogram_pred, plot_histogram_loss, plot_histogram_metric, plot_curve_accuracy_test, plot_curve_accuracy, plot_curve_loss, plot_hist_curve_loss_test, plot_curve_loss_train
from edl import get_device, one_hot_embedding, softplus_evidence, edl_log_loss, compute_dirichlet_uncert_entropy
#psscl
import robust_loss, Contrastive_loss

sns.set()

def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader, all_train_loss, all_train_loss_x, all_train_loss_u, savelog=False):
    net.train()
    net2.eval() #fix one network and train the other

    uncert = torch.zeros(len(eval_loader.dataset))

    train_loss = train_loss_lx = train_loss_u = train_loss_penalty = train_loss_simclr = train_loss_mixclr = 0

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    max_iters = ((len(labeled_trainloader.dataset)+len(unlabeled_trainloader.dataset))//args.batch_size)+1
    cont_iters = 0

    while(cont_iters<max_iters):
        for batch_idx, (inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x, indices_x) in enumerate(labeled_trainloader):
            try:
                inputs_u, inputs_u2, inputs_u3, inputs_u4, indices_u = unlabeled_train_iter.__next__()
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                inputs_u, inputs_u2, inputs_u3, inputs_u4, indices_u = unlabeled_train_iter.__next__()
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
                    alpha_u = (evidence_u11 + evidence_u12 + evidence_u21 + evidence_u22) / 4 + args.evidence_factor
                    S_u = torch.sum(alpha_u, dim=1, keepdim=True)
                    pu = alpha_u / S_u
                    ptu = pu**(1/args.T)

                    uncert_u = args.num_class / torch.sum(alpha_u, dim=1, keepdim=True)
                    # label refinement of labeled samples
                    evidence_x = softplus_evidence(outputs_x)
                    evidence_x2 = softplus_evidence(outputs_x2)
                    alpha_x = (evidence_x + evidence_x2)/2 + args.evidence_factor
                    S_x = torch.sum(alpha_x, dim=1, keepdim=True)
                    px = alpha_x / S_x
                    px = w_x*labels_x + (1-w_x)*px
                    ptx = px**(1/args.T) # temparature sharpening

                    uncert_x = args.num_class / torch.sum(alpha_x, dim=1, keepdim=True)

                    for b in range(inputs_u.size(0)):
                        uncert[indices_u[b]] = uncert_u[b]  # Use indices_u for unlabeled samples
                    for b in range(inputs_x.size(0)):
                        uncert[indices_x[b]] = uncert_x[b]  # Use indices_x for labeled samples


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


            loss = Lx + lamb * Lu + penalty+ args.lambda_c*(loss_simCLR + 0.2*loss_mixCLR)
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

            cont_iters = cont_iters + 1
            if cont_iters == max_iters:
                break

    sys.stdout.write(f'\r{args.dataset}: {args.r:.1f}-{args.noise_mode} | Epoch [{epoch:3d}/{args.num_epochs}] Iter[{batch_idx+1:3d}/{num_iter:3d}]\t Labeled loss: {train_loss_lx.item()/num_iter:.2f}  Unlabeled loss: {train_loss_u.item()/num_iter:.2f}   SimCLR loss: {train_loss_simclr.item()/num_iter:.2f}')
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




def test(epoch,net1,net2, acc_hist, loss_hist):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    test_loss = 0
    all_losses = torch.zeros(len(test_loader.dataset))

    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)           
            
            if args.uncertainty:
                y = one_hot_embedding(targets, args.num_class)
                device = get_device()
                y = y.to(device)

                evidence_1 = softplus_evidence(outputs1)
                alpha_1 = evidence_1 + args.evidence_factor
                predicted1 = alpha_1 / torch.sum(alpha_1, dim=1, keepdim=True)
                evidence_2 = softplus_evidence(outputs2)
                alpha_2 = evidence_2 + args.evidence_factor
                predicted2 = alpha_2 / torch.sum(alpha_2, dim=1, keepdim=True)
                
                outputs = (predicted1+predicted2)/2
                _, predicted = torch.max(outputs, 1) 
                loss, losses = edl_loss(outputs, y.float(), epoch_num=epoch, num_classes = args.num_class, annealing_step= args.ann_step)

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


def test_superclean(net1, net2, all_superclean):
    net1.eval()
    net2.eval()

    # Initialize accuracy counters
    acc_union = acc_union_noisy = acc_intersection = acc_intersection_noisy = 0
    acc_union_comp = acc_union_comp_noisy = acc_intersection_comp = acc_intersection_comp_noisy = 0

    # Define superclean sets
    union_superclean = set(all_superclean[0][-1]) | set(all_superclean[1][-1])
    intersection_superclean = set(all_superclean[0][-1]) & set(all_superclean[1][-1])
    total_samples = 50000  # Assuming CIFAR-10 dataset size

    # Convert clean and noisy labels to tensors (moved to correct device inside loop)
    clean_labels_tensor = torch.tensor(clean_labels, dtype=torch.long)
    noisy_labels_tensor = torch.tensor(noisy_labels, dtype=torch.long)

    with torch.no_grad():
        for inputs, targets, index in eval_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1, outputs2 = net1(inputs), net2(inputs)

            if args.uncertainty:
                device = inputs.device
                y = one_hot_embedding(targets, args.num_class).to(device)

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

            # Convert index list to check membership in Python sets
            index_list = index.tolist()
            union_mask = torch.tensor([i in union_superclean for i in index_list], dtype=torch.bool, device=inputs.device)
            intersection_mask = torch.tensor([i in intersection_superclean for i in index_list], dtype=torch.bool, device=inputs.device)

            # Complementary masks (automatically inferred)
            union_mask_comp = ~union_mask
            intersection_mask_comp = ~intersection_mask

            # Get clean and noisy labels
            batch_clean_labels = clean_labels_tensor[index].to(inputs.device)
            batch_noisy_labels = noisy_labels_tensor[index].to(inputs.device)

            # Compute accuracy for union (clean & noisy)
            if union_mask.any():
                union_pred = pred[union_mask]
                acc_union += (union_pred == batch_clean_labels[union_mask]).sum().item()
                acc_union_noisy += (union_pred == batch_noisy_labels[union_mask]).sum().item()

            # Compute accuracy for intersection (clean & noisy)
            if intersection_mask.any():
                intersection_pred = pred[intersection_mask]
                acc_intersection += (intersection_pred == batch_clean_labels[intersection_mask]).sum().item()
                acc_intersection_noisy += (intersection_pred == batch_noisy_labels[intersection_mask]).sum().item()

            # Compute accuracy for union complement (clean & noisy)
            if union_mask_comp.any():
                union_pred_comp = pred[union_mask_comp]
                acc_union_comp += (union_pred_comp == batch_clean_labels[union_mask_comp]).sum().item()
                acc_union_comp_noisy += (union_pred_comp == batch_noisy_labels[union_mask_comp]).sum().item()

            # Compute accuracy for intersection complement (clean & noisy)
            if intersection_mask_comp.any():
                intersection_pred_comp = pred[intersection_mask_comp]
                acc_intersection_comp += (intersection_pred_comp == batch_clean_labels[intersection_mask_comp]).sum().item()
                acc_intersection_comp_noisy += (intersection_pred_comp == batch_noisy_labels[intersection_mask_comp]).sum().item()

    # Compute accuracies safely (avoid division by zero)
    num_union = len(union_superclean)
    num_intersection = len(intersection_superclean)
    num_union_comp = total_samples - num_union
    num_intersection_comp = total_samples - num_intersection

    acc_union = 100 * acc_union / num_union if num_union > 0 else 0
    acc_union_noisy = 100 * acc_union_noisy / num_union if num_union > 0 else 0

    acc_intersection = 100 * acc_intersection / num_intersection if num_intersection > 0 else 0
    acc_intersection_noisy = 100 * acc_intersection_noisy / num_intersection if num_intersection > 0 else 0

    acc_union_comp = 100 * acc_union_comp / num_union_comp if num_union_comp > 0 else 0
    acc_union_comp_noisy = 100 * acc_union_comp_noisy / num_union_comp if num_union_comp > 0 else 0

    acc_intersection_comp = 100 * acc_intersection_comp / num_intersection_comp if num_intersection_comp > 0 else 0
    acc_intersection_comp_noisy = 100 * acc_intersection_comp_noisy / num_intersection_comp if num_intersection_comp > 0 else 0

    # Return clean & noisy accuracies for both normal and complementary sets
    return acc_union, acc_union_noisy, acc_intersection, acc_intersection_noisy, acc_union_comp, acc_union_comp_noisy, acc_intersection_comp, acc_intersection_comp_noisy




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
          Lx, _ = edl_loss(outputs_x, targets_x.float(), epoch_num=epoch, num_classes = args.num_class, annealing_step= args.ann_step)
          Lu = torch.mean((probs_u - targets_u)**2)
          return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model():
    model = ResNet18(num_classes=args.num_class)
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

def save_models(epoch, net1, optimizer1, net2, optimizer2, save_path, all_loss,
                 all_preds, hist_preds, inds_clean, inds_noisy, clean_labels, noisy_labels,
                 all_idx_view_labeled, all_idx_view_unlabeled, all_superclean, acc_hist,
                all_vacuity, all_entropy, all_dissonance, all_vagueness, eval_acc_hist, eval_loss_hist, test_acc_hist, test_losses_hist,
                loss_train, loss_train_x, loss_train_u):
    state = ({
                    'epoch'     : epoch,
                    'state_dict1'     : net1.state_dict(),
                    'optimizer1'      : optimizer1.state_dict(),
                    'state_dict2'     : net2.state_dict(),
                    'optimizer2'      : optimizer2.state_dict(),
                    'all_loss': all_loss,
                    'all_preds': all_preds,
                    'hist_preds': hist_preds,
                    'inds_clean': inds_clean,
                    'inds_noisy': inds_noisy,
                    'clean_labels': clean_labels,
                    'noisy_labels': noisy_labels,
                    'all_idx_view_labeled': all_idx_view_labeled,
                    'all_idx_view_unlabeled': all_idx_view_unlabeled,
                    'all_superclean': all_superclean,
                    'acc_hist': acc_hist,
                    'all_vacuity': all_vacuity,
                    'all_entropy': all_entropy,
                    'eval_acc_hist': eval_acc_hist,
                    'eval_loss_hist': eval_loss_hist,
                    'test_acc_hist': test_acc_hist,
                    'test_losses_hist': test_losses_hist,
                    'all_vagueness': all_vagueness,
                    'all_dissonance': all_dissonance,
                    'loss_train': loss_train,
                    'loss_train_x': loss_train_x,
                    'loss_train_u': loss_train_u,
                    })
    state3 = ({
                'all_superclean': all_superclean
                })


    if epoch%1==0:
        fn2 = os.path.join(save_path, 'model_ckpt.pth.tar')
        torch.save(state, fn2)
        if not os.path.exists('hcs'):
            os.makedirs('hcs')
        fn3 = os.path.join('hcs/', 'hcs_%s_%.2f_%s_cn%d_run%d.pth.tar'%(args.dataset, args.r, args.noise_mode,args.num_clean, args.run))
        torch.save(state3, fn3)


def write_log_superclean(net1, net2, all_superclean):
    # Run the test_superclean function to get accuracies
    acc_union, acc_union_noisy, acc_intersection, acc_intersection_noisy, acc_union_comp, acc_union_comp_noisy, acc_intersection_comp, acc_intersection_comp_noisy = test_superclean(net1, net2, all_superclean)

    # Define superclean sets
    union_superclean = set(all_superclean[0][-1]) | set(all_superclean[1][-1])
    intersection_superclean = set(all_superclean[0][-1]) & set(all_superclean[1][-1])
    
    union_num_superclean = len(union_superclean)
    intersection_num_superclean = len(intersection_superclean)
    total_instances = len(eval_loader.dataset)
    
    # Compute statistics for union
    percentage_superclean = (union_num_superclean / total_instances) * 100
    true_clean_indices = set(inds_clean)
    correct_superclean = len(union_superclean & true_clean_indices)
    superclean_accuracy = (correct_superclean / union_num_superclean) * 100 if union_num_superclean > 0 else 0.0

    # Write results for union
    superclean_log.write(f"Epoch: {epoch}\n")
    superclean_log.write(f"Number of superclean instances: {union_num_superclean}\n")
    superclean_log.write(f"Percentage of superclean instances: {percentage_superclean:.2f}%\n")
    superclean_log.write(f"Accuracy of superclean instances (clean vs noisy): {superclean_accuracy:.2f}%\n")
    superclean_log.write(f"Accuracy of the superclean instances (true class): {acc_union:.2f}%\n")
    superclean_log.write(f"Accuracy of the superclean instances (noisy class): {acc_union_noisy:.2f}%\n")
    superclean_log.write(f"Accuracy of the NOT superclean instances (true class): {acc_union_comp:.2f}%\n")
    superclean_log.write(f"Accuracy of the NOT superclean instances (noisy class): {acc_union_comp_noisy:.2f}%\n")
    superclean_log.write("-" * 50 + "\n")
    superclean_log.flush()

    # Compute statistics for intersection
    percentage_superclean = (intersection_num_superclean / total_instances) * 100
    correct_superclean = len(intersection_superclean & true_clean_indices)
    superclean_accuracy = (correct_superclean / intersection_num_superclean) * 100 if intersection_num_superclean > 0 else 0.0

    # Write results for intersection
    superclean_log.write(f"Number of superclean instances (intersection): {intersection_num_superclean}\n")
    superclean_log.write(f"Percentage of superclean instances (intersection): {percentage_superclean:.2f}%\n")
    superclean_log.write(f"Accuracy of superclean instances (intersection) (clean vs noisy): {superclean_accuracy:.2f}%\n")
    superclean_log.write(f"Accuracy of the superclean instances (intersection) (true class): {acc_intersection:.2f}%\n")
    superclean_log.write(f"Accuracy of the superclean instances (intersection) (noisy class): {acc_intersection_noisy:.2f}%\n")
    superclean_log.write(f"Accuracy of the NOT superclean instances (intersection) (true class): {acc_intersection_comp:.2f}%\n")
    superclean_log.write(f"Accuracy of the NOT superclean instances (intersection) (noisy class): {acc_intersection_comp_noisy:.2f}%\n")
    superclean_log.write("-" * 50 + "\n")
    superclean_log.flush()



"""class_number = 100
dataset_name = 'cifar100'
dataset_path = './cifar-100'
number_epochs = 300"""

class_number = 10
dataset_name = 'cifar10'
dataset_path = './cifar-10-batches-py'
number_epochs = 300

arguments = [argparse.Namespace(
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
    dataset='cifar10',
    ann_step = 300, #tuning??
    uncertainty = False, #either uncertainty or gce loss??
    evidence_factor = 1,
    gce_loss = True,
    sim_clr = True,
    mix_clr = False),

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
    ann_step = 300, #tuning??
    uncertainty = False, #either uncertainty or gce loss??
    evidence_factor = 1,
    gce_loss = True,
    sim_clr = True,
    mix_clr = False)
    ]
    

arguments_2 = arguments
for args in arguments_2:
    if args.dataset == 'cifar100':
        args.num_class=100
        args.data_path= './cifar-100'


    elif args.dataset == 'cifar10':
        args.num_class=10
        args.data_path= './cifar-10-batches-py'


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
    edl_loss = edl_log_loss
    ce_loss_sample = nn.CrossEntropyLoss(reduction='none')
    ce_loss = nn.CrossEntropyLoss()
    if args.noise_mode=='asym':
        conf_penalty = NegEntropy()
    gce_loss = robust_loss.GCELoss(args.num_class, gpu='0') #only used in warmup
    contrastive_criterion = Contrastive_loss.SupConLoss()

    exp_str = f"exp_{args.dataset}_{args.r}"

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
    else:    
        stats_log=open('./checkpoint/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_stats.txt','a') 
        test_log=open('./checkpoint/%s/%s_%.2f_%s'%(exp_str,args.dataset,args.r,args.noise_mode)+'_acc.txt','a') 
        time_log=open('./checkpoint/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_time.txt','a')
        superclean_log= open('./checkpoint/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_superclean.txt','a')

    # writer_tensorboard = SummaryWriter('tensor_runs/'+exp_str) 

    loader = dataloader.cifar_dataloader(args.dataset,r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=5,\
        root_dir=args.data_path,log=stats_log,noise_file='noise/%s/%.2f_%s.json'%(args.dataset,args.r,args.noise_mode))

    warmup_trainloader = loader.run('warmup')

    print('| Building net')
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
        all_preds = ckpt['all_preds']
        hist_preds = ckpt['hist_preds']
        acc_hist = ckpt['acc_hist']
        all_loss = ckpt['all_loss']
        all_vacuity = ckpt['all_vacuity']
        all_entropy = ckpt['all_entropy']
        all_dissonance = ckpt['all_dissonance']
        all_vagueness = ckpt['all_vagueness']
        eval_loss_hist = ckpt['eval_loss_hist']
        eval_acc_hist = ckpt['eval_acc_hist']
        test_acc_hist = ckpt['test_acc_hist']
        test_losses_hist = ckpt['test_losses_hist']
        try:
            loss_train = ckpt.get("loss_train", [[],[]])
            loss_train_x = ckpt.get("loss_train_x", [[],[]])
            loss_train_u = ckpt.get("loss_train_u", [[],[]])
        except Exception as e:
            loss_train, loss_train_x, loss_train_u = [[],[]], [[],[]], [[],[]]

    else:
        all_superclean = [[],[]]
        all_idx_view_labeled = [[],[]]
        all_idx_view_unlabeled = [[], []]
        all_preds = [[], []] # save the history of preds for two networks
        hist_preds = [[],[]]
        acc_hist = []
        all_loss = [[],[]] # save the history of losses from two networks
        all_vacuity = [[],[]]
        all_entropy = [[],[]]
        all_dissonance = [[],[]]
        all_vagueness = [[],[]]
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

    superclean_path =os.path.join('hcs/', 'hcs_%s_%.2f_%s_cn%d_run%d.pth.tar'%(args.dataset, args.r, args.noise_mode,args.num_clean, args.run))
    ckpt = torch.load(superclean_path)
    all_superclean = ckpt['all_superclean']

    maxsize = 0
    max_i = 0
    for i in range(1,151):
        size = len(all_superclean[0][-i])
        if size > maxsize:
            maxsize = size
            max_i = i
    print('net 1 max = %d, i=%d'%(maxsize,max_i))
    idx_superclean_1 = all_superclean[0][-max_i]
    maxsize = 0
    max_i = 0
    for i in range(1,151):
        size = len(all_superclean[1][-i])
        if size > maxsize:
            maxsize = size
            max_i = i
    print('net 2 max = %d, i=%d'%(maxsize,max_i))
    idx_superclean_2 = all_superclean[1][-max_i]
    intersection_clean = set(idx_superclean_1) & set(idx_superclean_2)
    union_clean = set(idx_superclean_1) | set(idx_superclean_2)
    print("Intersection:", len(intersection_clean))
    print("Union:", len(union_clean))
    test_log.write('net1 clean samples: %d \n'%(len(idx_superclean_1)))
    test_log.write('net2 clean samples: %d \n'%(len(idx_superclean_2)))
    test_log.write('Intersection of clean samples: %d \n'%(len(intersection_clean)))
    test_log.write('Union of clean samples: %d \n'%(len(union_clean)))
    union_clean_indices = np.array(list(set(idx_superclean_1) | set(idx_superclean_2)))

    total_time =  0
    warmup_time = 0

    for epoch in range(resume_epoch, args.num_epochs+1):   
        lr=args.lr
        if epoch >= 150:
            lr /= 10      
        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr       
        for param_group in optimizer2.param_groups:
            param_group['lr'] = lr          
            
            
        if True:         
            print("Training Epoch ",epoch)
            prob1 = torch.zeros(len(eval_loader.dataset))
            prob2 = torch.zeros(len(eval_loader.dataset))

            #Update probabilities
            prob1[union_clean_indices] = 1
            prob2[union_clean_indices] = 1

            pred1 = (prob1 > args.p_threshold)      
            pred2 = (prob2 > args.p_threshold)


            start_time = time.time()
            print('Train Net1')
            labeled_trainloader, unlabeled_trainloader, _ = loader.run('train',pred2,prob2) # co-divide
            loss_train[0], loss_train_x[0], loss_train_u[0] = train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader, loss_train[0], loss_train_x[0], loss_train_u[0], savelog=True) # train net1  
            
            print('\nTrain Net2')
            labeled_trainloader, unlabeled_trainloader, u_map_trainloader = loader.run('train',pred1,prob1) # co-divide
            loss_train[1], loss_train_x[1], loss_train_u[1] = train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader, loss_train[1], loss_train_x[1], loss_train_u[1], savelog=False) # train net2         
            end_time = round(time.time() - start_time)
            total_time+= end_time

        test_acc_hist, test_losses_hist = test(epoch,net1,net2, test_acc_hist, test_losses_hist)

        if epoch%5==0 and epoch !=0:
            print("Plots...")
            plot_curve_loss(data_hist= eval_loss_hist[0], inds_clean=inds_clean, inds_noisy=inds_noisy, path=path_plot, epoch=epoch )
            plot_curve_accuracy(data_hist= eval_acc_hist[0], inds_clean=inds_clean, inds_noisy=inds_noisy, path=path_plot, epoch=epoch )
            plot_curve_accuracy_test(data_hist= test_acc_hist, inds_clean=inds_clean, inds_noisy=inds_noisy, path=path_plot, epoch=epoch )

            plot_hist_curve_loss_test(data_hist= test_losses_hist, path=path_plot, epoch=epoch )

            plot_histogram_loss(data_hist=all_loss[0], inds_clean=inds_clean, inds_noisy=inds_noisy, path=path_plot, epoch=epoch )
            plot_histogram_pred(data=all_preds[0], entropy = all_entropy[0], inds_clean=inds_clean, inds_noisy=inds_noisy, path=path_plot, epoch=epoch )

            plot_histogram_metric(data_hist=all_vacuity[0], inds_clean=inds_clean, inds_noisy=inds_noisy, path=path_plot, epoch=epoch, metric = "Vacuity" )
            plot_histogram_metric(data_hist=all_entropy[0], inds_clean=inds_clean, inds_noisy=inds_noisy, path=path_plot, epoch=epoch, metric = "Entropy"  )
            plot_histogram_metric(data_hist=all_dissonance[0], inds_clean=inds_clean, inds_noisy=inds_noisy, path=path_plot, epoch=epoch, metric = "Dissonance" )
            #plot_histogram_metric(data_hist=all_vagueness[0], inds_clean=inds_clean, inds_noisy=inds_noisy, path=path_plot, epoch=epoch, metric = "Vagueness"  )
            print("Plots finished")
        
    
        save_models(epoch, net1, optimizer1, net2, optimizer2, path_exp, all_loss,
                            all_preds, hist_preds, inds_clean, inds_noisy, clean_labels, noisy_labels,
                            all_idx_view_labeled, all_idx_view_unlabeled, all_superclean, acc_hist,
                            all_vacuity, all_entropy, all_dissonance, all_vagueness, eval_acc_hist, eval_loss_hist, test_acc_hist, test_losses_hist,
                            loss_train, loss_train_x, loss_train_u)

    test_log.write('\nBest:%.2f  avgLast10: %.2f\n'%(max(test_acc_hist),sum(test_acc_hist[-10:])/10.0))
    test_log.close() 

    time_log.write('SSL Time: %f \n'%(total_time-warmup_time))
    time_log.write('Total Time: %f \n'%(total_time))
    time_log.close()

    superclean_log.close()

