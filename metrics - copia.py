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
from utils_plot import plot_histogram_pred, plot_histogram_metric, plot_histogram_metric2, scatter_plot_probs
from utils_plot import plot_curve_accuracy_test, plot_curve_accuracy, plot_curve_loss, plot_hist_curve_loss_test, plot_curve_loss_train
from edl import compute_metrics, get_device, one_hot_embedding, softplus_evidence, exp_evidence, relu_evidence, edl_log_loss, m_edl_log_loss, edl_mse_loss, edl_digamma_loss, compute_dirichlet_metrics
#psscl


sns.set()

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


def estimate_log_gaussian_prob(gmm, X):
    from scipy.linalg import cholesky, solve_triangular

    X = np.atleast_2d(X)
    n_samples, n_features = X.shape
    means = gmm.means_                            # (n_components, n_features)
    covariances = gmm.covariances_                # (n_components, n_features, n_features)
    n_components = gmm.n_components

    if gmm.covariance_type != 'full':
        raise NotImplementedError("This function currently only supports 'full' covariance_type.")

    log_prob = np.empty((n_samples, n_components))

    for k in range(n_components):
        mean = means[k]
        cov = covariances[k]

        # Cholesky decomposition for numerical stability
        try:
            cholesky_cov = cholesky(cov, lower=True)
        except np.linalg.LinAlgError:
            raise ValueError(f"Covariance matrix of component {k} is not positive definite.")

        # Center data
        diff = X - mean  # (n_samples, n_features)

        # Solve for Mahalanobis distance: (L^T)^(-1) * diff.T
        solved = solve_triangular(cholesky_cov, diff.T, lower=True)
        mahal = np.sum(solved**2, axis=0)  # shape (n_samples,)

        # Log determinant
        log_det = 2 * np.sum(np.log(np.diagonal(cholesky_cov)))

        # Log probability under Gaussian
        log_prob[:, k] = -0.5 * (n_features * np.log(2 * np.pi) + log_det + mahal)

    return log_prob



def get_clean(prob, net_idx, thres = 0.5):
    pred = (prob > thres)      
    idx_view_labeled = (pred).nonzero()[0]

    pred_loss = np.array([True if p in idx_view_labeled else False for p in range(len(pred))])

    threshold = []
    closest_index = (np.abs(np.array(prob) - thres)).argmin()
    threshold.append(closest_index)

    return pred_loss, threshold 

def get_clean_margin(prob):
    pred = (prob > 0)  # Boolean mask

    # Get indices where pred is True (prob > 0)
    idx_view_labeled = np.nonzero(pred)[0] if np.any(pred) else []

    # Count positives and non-positives
    num_positive = np.sum(pred)
    num_non_positive = len(pred) - num_positive

    print(f"Number of samples with prob > 0: {num_positive}")
    print(f"Number of samples with prob <= 0: {num_non_positive}")

    pred_loss = np.array([p in idx_view_labeled for p in range(len(pred))])

    return pred_loss

def margin_relabel_metrics(thresh=0.5, idx_view_labeled=None, margin=None, clean_labels=None, predicted_labels=None, larger=False):
    print("length idx view labeled:", len(idx_view_labeled))
    idx_view_labeled_np = idx_view_labeled.astype(int)

    total_indices = np.arange(len(margin))

    # Compute high-margin count without excluding labeled
    if larger:
        global_high_margin_mask = margin > thresh
    else:
        global_high_margin_mask = margin < thresh
    global_high_margin_indices = total_indices[global_high_margin_mask]
    print("High-margin (no exclusion):", len(global_high_margin_indices))

    # Now exclude labeled samples
    mask_unlabeled = np.ones(len(margin), dtype=bool)
    mask_unlabeled[idx_view_labeled_np] = False
    idx_unlabeled = total_indices[mask_unlabeled]

    if larger:
        high_margin_mask = margin[idx_unlabeled] > thresh
    else:
        high_margin_mask = margin[idx_unlabeled] < thresh
    high_margin_indices = idx_unlabeled[high_margin_mask]
    print("High-margin (excluding labeled):", len(high_margin_indices))

    num_high_margin = len(high_margin_indices)

    if num_high_margin > 0:
        correct = (clean_labels[high_margin_indices] == predicted_labels[high_margin_indices])
        accuracy = correct.sum() / num_high_margin
    else:
        accuracy = 0.0

    return num_high_margin, accuracy * 100




def get_clean_label_ranks(predicted_probs, clean_labels, idx_view_labeled, noisy_labels):
    idx_view_labeled = np.array(idx_view_labeled)
    all_indices = np.arange(len(clean_labels))
    idx_unlabeled = np.setdiff1d(all_indices, idx_view_labeled)
    #predicted_probs[torch.arange(len(noisy_labels)), noisy_labels] = -1

    def compute_ranks(indices):
        ranks = []
        for idx in indices:
            probs = predicted_probs[idx]
            label = clean_labels[idx]
            sorted_indices = np.argsort(-probs)
            rank = np.where(sorted_indices == label)[0][0] + 1
            ranks.append(rank)
        return np.array(ranks)

    labeled_ranks = compute_ranks(idx_view_labeled)
    unlabeled_ranks = compute_ranks(idx_unlabeled)

    return labeled_ranks, unlabeled_ranks


def get_clean_label_ranks_exclude_noisy(predicted_probs, clean_labels, idx_view_labeled, noisy_labels):
    idx_view_labeled = np.array(idx_view_labeled)
    all_indices = np.arange(len(clean_labels))
    idx_unlabeled = np.setdiff1d(all_indices, idx_view_labeled)
    predicted_probs[torch.arange(len(noisy_labels)), noisy_labels] = -1

    def compute_ranks(indices):
        ranks = []
        for idx in indices:
            probs = predicted_probs[idx]
            label = clean_labels[idx]
            sorted_indices = np.argsort(-probs)
            rank = np.where(sorted_indices == label)[0][0] + 1
            ranks.append(rank)
        return np.array(ranks)

    labeled_ranks = compute_ranks(idx_view_labeled)
    unlabeled_ranks = compute_ranks(idx_unlabeled)

    return labeled_ranks, unlabeled_ranks

def write_log_ranks(log_file, predicted_probs, clean_labels, idx_view_labeled, noisy_labels):
            #uncertainty
            labeled_rank, unlabeled_rank = get_clean_label_ranks(predicted_probs, clean_labels, idx_view_labeled, noisy_labels)
            total_unlabeled = len(unlabeled_rank)
            rank1_pct = np.sum(unlabeled_rank == 1) / total_unlabeled * 100
            rank2_pct = np.sum(unlabeled_rank == 2) / total_unlabeled * 100
            rank_other_pct = 100 - rank1_pct - rank2_pct
            # Logging
            log_file.write(f"Epoch: {epoch}\n")
            log_file.write(f"Rank=1: {rank1_pct:.2f}%\n")
            log_file.write(f"Rank=2: {rank2_pct:.2f}%\n")
            log_file.write(f"Rank>2: {rank_other_pct:.2f}%\n")
            log_file.write("-" * 50 + "\n")
            log_file.flush()

            #uncertainty
            labeled_rank, unlabeled_rank = get_clean_label_ranks_exclude_noisy(predicted_probs, clean_labels, idx_view_labeled, noisy_labels)
            total_unlabeled = len(unlabeled_rank)
            rank1_pct = np.sum(unlabeled_rank == 1) / total_unlabeled * 100
            rank2_pct = np.sum(unlabeled_rank == 2) / total_unlabeled * 100
            rank_other_pct = 100 - rank1_pct - rank2_pct
            # Logging
            log_file.write(f"Epoch: {epoch}\n")
            log_file.write(f"Rank=1: {rank1_pct:.2f}%\n")
            log_file.write(f"Rank=2: {rank2_pct:.2f}%\n")
            log_file.write(f"Rank>2: {rank_other_pct:.2f}%\n")
            log_file.write("-" * 50 + "\n")
            log_file.flush()


def write_log_acc(log_file, predicted_probs, clean_labels, idx_view_labeled, noisy_labels):
            all_indices = np.arange(len(clean_labels))
            idx_view_unlabeled = np.setdiff1d(all_indices, idx_view_labeled)
            
            predicted_labels_labeled = torch.argmax(predicted_probs[idx_view_labeled], dim=1)
            correct_indices_labeled = np.where(clean_labels[idx_view_labeled] == predicted_labels_labeled)[0]
            acc_labeled = len(correct_indices_labeled)/len(idx_view_labeled)*100
            
            #unlabeled metrics
            predicted_labels_unlabeled = torch.argmax(predicted_probs[idx_view_unlabeled], dim=1)
            correct_indices_unlabeled = np.where(clean_labels[idx_view_unlabeled] == predicted_labels_unlabeled)[0]
            acc_unlabeled1 = len(correct_indices_unlabeled)/len(idx_view_unlabeled)*100

            predicted_probs_2 = predicted_probs.clone()  # avoid modifying original
            predicted_probs_2[torch.arange(len(noisy_labels)), noisy_labels] = -1
            predicted_labels_unlabeled_2 = torch.argmax(predicted_probs_2[idx_view_unlabeled], dim=1)
            correct_indices_unlabeled_2 = np.where(clean_labels[idx_view_unlabeled] == predicted_labels_unlabeled_2)[0]
            acc_unlabeled2 = len(correct_indices_unlabeled_2)/len(idx_view_unlabeled)*100

            # Logging
            log_file.write(f"Epoch: {epoch}\n")
            log_file.write(f"Accuracy labeled: {acc_labeled:.2f}%\n")
            log_file.write(f"Accuracy unlabeled: {acc_unlabeled1:.2f}%\n")
            log_file.write(f"Accuracy unlabeled 2: {acc_unlabeled2:.2f}%\n")
            log_file.write("-" * 50 + "\n")
            log_file.flush()



class_number = 100
dataset_name = 'cifar100'
dataset_path = './cifar-10-batches-py'


a = [
    argparse.Namespace(
    name_exp = "0.8_edl_0.01_expact_bptplr_pretrained_u500",
    batch_size=64,
    noise_mode='sym',
    r=0.2,
    data_path= dataset_path,
    dataset='cifar100',
    ),  

]



for args in a:
    print(args.name_exp)
    if args.dataset == 'cifar100':
        args.num_class=100
        args.data_path= './cifar-100'
        args.num_epochs = 200

    elif args.dataset == 'cifar10':
        args.num_class=10
        args.data_path= './cifar-10-batches-py'
        args.num_epochs = 200


    torch.cuda.set_device(0)
    random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)


    device = torch.device("cuda")
    
    exp_str = f"{args.name_exp}"

    path_exp='./checkpoint/' + exp_str

    path_plot = os.path.join(path_exp, 'plots')

    Path(path_exp).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(path_exp, 'savedDicts')).mkdir(parents=True, exist_ok=True)
    Path(path_plot).mkdir(parents=True, exist_ok=True)


    
    model_path = "./checkpoint/%s/model_ckpt.pth.tar"%(exp_str)
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
    metrics_log= open('./checkpoint/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_metrics.txt','w')
    ranks_log= open('./checkpoint/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_ranks.txt','w')
    bm_log= open('./checkpoint/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_bm.txt','w')
    acc_unlabeled_log= open('./checkpoint/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_acc_unlabeled.txt','w')

    loader = dataloader.cifar_dataloader(args.dataset,r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=5,\
        root_dir=args.data_path,log=stats_log,noise_file='noise/%s/%.2f_%s.json'%(args.dataset,args.r,args.noise_mode))


    resume_epoch = 0
    if incomplete == True:
        print('loading Model...\n')
        load_path = model_path
        ckpt = torch.load(load_path)
        resume_epoch = ckpt['epoch']+1
        print('resume_epoch....', resume_epoch)

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



    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')
    noisy_labels = eval_loader.dataset.noise_label
    clean_labels = eval_loader.dataset.train_label 
    inds_noisy = np.asarray([ind for ind in range(len(noisy_labels)) if noisy_labels[ind] != clean_labels[ind]])
    inds_clean = np.delete(np.arange(len(noisy_labels)), inds_noisy)


    print("length all loss ", len(all_loss[0]))

    plr_loss = True
    bpt_margins = False
    bpt=True
    margins_modeling=False

    accuracy_relabel = []
    number = []

    #plot_curve_loss_train(data_hist=[loss_train[0], loss_train_x[0], loss_train_u[0], train_loss_contrastive[0]], path=path_plot)
    #plot_curve_loss(data_hist= eval_loss_hist[0], inds_clean=inds_clean, inds_noisy=inds_noisy, path=path_plot)
    #plot_curve_accuracy(data_hist= eval_acc_hist[0], inds_clean=inds_clean, inds_noisy=inds_noisy, path=path_plot)

    for i in range(5, 201, 5):
    #for i in [30, 80]:
        epoch = i
        epoch2 = i-5
        print("epoch", epoch2)
        losses = all_loss[0][epoch2]
        preds = all_preds[0][epoch2]
        predicted_probs = hist_preds[0][epoch2]  # Shape: [num_samples, num_classes]
        predicted_labels = torch.argmax(predicted_probs, dim=1)

        outputs = all_outputs[0][epoch2]

        #if (epoch%5==0 and epoch >=30):
        if True:
            vacuity, dissonance, dissonance2, entropy, mutual_information, margin, evi_noisy_class, evi_true_class, max_evi_excl_noisy, predicted_class_2, belief_mass, noisy_margins = compute_metrics(outputs, 100, noisy_labels, clean_labels, act = exp_evidence)
            correct_indices1 = np.where(clean_labels == predicted_labels)[0]
            #gmm:
            input_loss = losses.reshape(-1,1)
            if bpt:
                        losses_proto = all_proto_loss[0][epoch2]
                        input_loss_proto = losses_proto.reshape(-1, 1)

                        input_loss = input_loss.cpu().numpy()

                        input_loss_proto = input_loss_proto.cpu().numpy()
                        gmm_input = np.column_stack((input_loss, input_loss_proto))
                        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
                        gmm.fit(gmm_input)
                        mean_square_dists = np.array([np.sum(np.square(gmm.means_[i])) for i in range(2)])
                        argmin, argmax = mean_square_dists.argmin(), mean_square_dists.argmax()
                        prob = gmm.predict_proba(gmm_input)
                        prob = prob[:, argmin]
                        pred_loss1, threshold_loss_1 = get_clean(prob, net_idx = 0, thres = 0.5)
                        
                        
                        
                        means = gmm.means_  # shape: (2, 2)
                        covariances = gmm.covariances_  # shape: (2, 2, 2) if full covariance

                        # Identify low-loss and high-loss components
                        mean_square_dists = np.array([np.sum(np.square(means[i])) for i in range(2)])
                        argmin, argmax = mean_square_dists.argmin(), mean_square_dists.argmax()  # low-loss = argmin, high-loss = argmax

                        # Extract mean and covariance of both components
                        low_loss_mean = means[argmin]              # shape: (2,)
                        high_loss_mean = means[argmax]            # shape: (2,)

                        low_loss_cov = covariances[argmin]        # shape: (2, 2)
                        high_loss_cov = covariances[argmax]       # shape: (2, 2)

                        # Optionally, extract variances (diagonal of covariance matrix)
                        low_loss_variance = np.diag(low_loss_cov)     # shape: (2,)
                        high_loss_variance = np.diag(high_loss_cov)  

                        print("means: ", means)

            else:
                        gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
                        gmm.fit(input_loss)
                        prob = gmm.predict_proba(input_loss)
                        prob = prob[:,gmm.means_.argmin()]


                        pred_loss1, threshold_loss_1 = get_clean(prob, net_idx = 0, thres = 0.5)

            if bpt or bpt_margins:
                sample_rate1 = len((pred_loss1).nonzero()[0]) / len(eval_loader.dataset)
                pred1_new = np.zeros(len(eval_loader.dataset)).astype(np.bool_)
                class_len1 = int(sample_rate1 * len(eval_loader.dataset) / args.num_class)
                for i in range(args.num_class):
                    class_indices = np.where(np.array(eval_loader.dataset.noise_label) == i)[0]
                    size1 = len(class_indices)
                    class_len_temp1 = min(size1, class_len1)

                    prob2 = np.argsort(-prob[class_indices])
                    select_idx = class_indices[prob2[:class_len_temp1]]
                    pred1_new[select_idx] = True

                pred_loss1 = pred1_new


            idx_view_labeled = np.where(pred_loss1 == 1)[0]
            idx_view_unlabeled = np.setdiff1d(np.arange(len(pred_loss1)), idx_view_labeled)
            #print("length idx labeled", len(idx_view_labeled))
            acc = len(list(set(correct_indices1) & set(idx_view_labeled)))/len(list(set(idx_view_labeled)))
            #print("accuracy labeled is ", acc)
            acc_unlabeled = len(list(set(correct_indices1) & set(idx_view_unlabeled)))/len(list(set(idx_view_unlabeled)))

            acc, prec, recall = get_metrics(predicted_labels, clean_labels, (pred_loss1).nonzero()[0], clean_indices=inds_clean)
            print(acc, prec, recall)

            # Logging
            log_file=acc_unlabeled_log
            log_file.write(f"Epoch: {epoch}\n")
            log_file.write(f"Accuracy unlabeled: {acc_unlabeled:.2f}%\n")
            log_file.write(f"precision clean: {prec:.2f}%\n")
            log_file.write(f"recall: {recall:.2f}%\n")
            log_file.write("-" * 50 + "\n")
            log_file.flush()



            #plot_hist_curve_loss_test(data_hist= test_losses_hist, path=path_plot, epoch=epoch )
            plot_histogram_metric(data_hist=all_loss[0][epoch2], inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= idx_view_labeled, inds_relabeled = relabel_idx_1, thresholds = threshold_loss, path=path_plot, epoch=epoch, metric = "Loss2"  )
            #plot_histogram_metric(data_hist=margins_plot, inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= idx_view_labeled, inds_relabeled = relabel_idx_1, thresholds = [], path=path_plot, epoch=epoch, metric = "Margin_True"  )
            #if plr_loss:
                #plot_histogram_metric(data_hist=all_proto_loss[0][epoch2], inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= idx_view_labeled, inds_relabeled = relabel_idx_1, thresholds = threshold_loss, path=path_plot, epoch=epoch, metric = "Prototype_loss"  )
            #plot_histogram_metric2(data_hist=margin, inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= idx_view_labeled, thresholds = [], path=path_plot, epoch=epoch, data_hist_2= all_loss[0][epoch2], data_hist_3= preds, metric = "Margins_adjust"  )
            #plot_histogram_metric2(data_hist=noisy_margins, inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= idx_view_labeled, thresholds = [], path=path_plot, epoch=epoch, data_hist_2= all_loss[0][epoch2], data_hist_3= preds, metric = "True_margin"  )
            #plot_histogram_metric2(data_hist=vacuity, inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= idx_view_labeled, thresholds = [], path=path_plot, epoch=epoch, data_hist_2= all_loss[0][epoch2], data_hist_3= preds, metric = "Vacuity_adjust"  )
            #plot_histogram_metric2(data_hist=entropy, inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= idx_view_labeled, thresholds = [], path=path_plot, epoch=epoch, data_hist_2= all_loss[0][epoch2], data_hist_3= preds, metric = "Entropy"  )
            #plot_histogram_metric2(data_hist=dissonance, inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= idx_view_labeled,thresholds = [], path=path_plot, epoch=epoch, data_hist_2= all_loss[0][epoch2], data_hist_3= preds, metric = "Dissonance"  )
            #plot_histogram_metric2(data_hist=mutual_information, inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= idx_view_labeled, thresholds = [], path=path_plot, epoch=epoch, data_hist_2= all_loss[0][epoch2], data_hist_3= preds, metric = "Mutual_information"  )
            #scatter_plot_probs(prob_true_class, prob_noisy_class, best_prob, evi_noisy_class, evi_true_class, max_evi_excl_noisy, idx_view_labeled, correct_indices1, inds_clean, path=path_plot, epoch=epoch)


    def plot_curve(data_hist, data_hist_2, path, metric):
        print(data_hist)
        print(data_hist_2)
        # History plot with three curves
        metric_path = os.path.join(path, metric)
        data_hist_2 = np.array(data_hist_2) / 500
        os.makedirs(metric_path, exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(data_hist) + 1), data_hist, linestyle='-', color='red', label='accuracy')
        plt.plot(range(1, len(data_hist_2) + 1), data_hist_2, linestyle='-', color='blue', label='samples relabeled')

        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title('Relabeled samples Across Epochs')
        plt.legend()
        plt.savefig(f'{metric_path}/{metric}_relabel.png')
        plt.clf()

    #plot_curve(accuracy_relabel, number, path=path_plot, metric = 'Acc_relabeled')
    #plot_curve(number, path=path_plot, metric = 'Num_relabeled')
    
    test_log.close() 
    test_log.close() 
    time_log.close()
    superclean_log.close()
    ranks_log.close()
    metrics_log.close()
    bm_log.close()








