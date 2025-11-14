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
from utils_plot import plot_histogram_metric, plot_histogram_metric2, plot_histogram_metric3, plot_histogram_metric22
from utils_plot import plot_curve_accuracy_test, plot_curve_accuracy, plot_curve_loss, plot_hist_curve_loss_test, plot_curve_loss_train
from edl import compute_metrics, get_device, one_hot_embedding, softplus_evidence, exp_evidence, relu_evidence, edl_log_loss, m_edl_log_loss, edl_mse_loss, edl_digamma_loss, compute_dirichlet_metrics
#psscl
from collections import defaultdict


sns.set()



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
    name_exp = "0.8_noplr_vacuitypenalty",
    batch_size=64,
    noise_mode='sym',
    r=0.8,
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

    path_exp='./checkpoint3/' + exp_str

    path_plot = os.path.join(path_exp, 'plots')

    Path(path_exp).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(path_exp, 'savedDicts')).mkdir(parents=True, exist_ok=True)
    Path(path_plot).mkdir(parents=True, exist_ok=True)


    
    model_path = "./checkpoint3/%s/model_ckpt.pth.tar"%(exp_str)
    incomplete = os.path.exists(model_path)

    print('Incomplete...', incomplete)

    if incomplete == False:
        log_mode = 'w'
    else:
        log_mode = 'a'
    stats_log=open('./checkpoint3/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_stats.txt',log_mode) 
    test_log=open('./checkpoint3/%s/%s_%.2f_%s'%(exp_str,args.dataset,args.r,args.noise_mode)+'_acc.txt',log_mode) 
    time_log=open('./checkpoint3/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_time.txt',log_mode) 
    superclean_log= open('./checkpoint3/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_superclean.txt',log_mode)
    metrics_log= open('./checkpoint3/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_metrics.txt','w')
    ranks_log= open('./checkpoint3/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_ranks.txt','w')
    bm_log= open('./checkpoint3/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_bm.txt','w')
    acc_unlabeled_log= open('./checkpoint3/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_acc_unlabeled.txt','w')
    dissonance_log= open('./checkpoint3/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_dissonance.txt','w')

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
    print("length outputs", len(all_outputs[0]))
    print("length preds", len(all_preds[0]), len(hist_preds[0]))
    print("length labeled", len(all_idx_view_labeled[0]))      


    plr_loss = False
    bpt=True

    accuracy_relabel = []
    number = []

    plot_curve_loss_train(data_hist=[loss_train[0], loss_train_x[0], loss_train_u[0], train_loss_contrastive[0]], path=path_plot)
    plot_curve_loss(data_hist= eval_loss_hist[0], inds_clean=inds_clean, inds_noisy=inds_noisy, path=path_plot)
    plot_curve_accuracy(data_hist= eval_acc_hist[0], inds_clean=inds_clean, inds_noisy=inds_noisy, path=path_plot)

    for i in list(range(5, 20)) + list(range(20, 201, 5)):
        epoch = i
        epoch2 = i-5
        print("epoch", epoch2)
        losses = all_loss[0][epoch2]
        preds = all_preds[0][epoch2]
        predicted_probs = hist_preds[0][epoch2]  # Shape: [num_samples, num_classes]
        predicted_labels = torch.argmax(predicted_probs, dim=1)

        outputs = all_outputs[0][epoch2]
        evidence = outputs[torch.arange(outputs.size(0)), noisy_labels]
        copy_outputs = outputs.clone()
        copy_outputs[torch.arange(outputs.size(0)), noisy_labels] = -1e5
        evidence_neg = copy_outputs.max()
        margins=evidence-evidence_neg

        if True:
            vacuity, dissonance, dissonance2, entropy, mutual_information, margin, evi_noisy_class, evi_true_class, max_evi_excl_noisy, predicted_class_2, belief_mass, noisy_margins = compute_metrics(outputs, 100, noisy_labels, clean_labels, act = exp_evidence)
            print("epoch: ", epoch )
            #gmm:
            input_loss = losses.reshape(-1,1)
            if plr_loss:
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


            else:
                        gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
                        gmm.fit(input_loss)
                        prob = gmm.predict_proba(input_loss)
                        prob = prob[:,gmm.means_.argmin()]
                        pred_loss1, threshold_loss_1 = get_clean(prob, net_idx = 0, thres = 0.5)

            if bpt:
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
            correct_indices1 = np.where(clean_labels == predicted_labels)[0]
            acc = len(list(set(correct_indices1) & set(idx_view_labeled)))/len(list(set(idx_view_labeled)))
            #print("accuracy labeled is ", acc)
            acc_unlabeled = len(list(set(correct_indices1) & set(idx_view_unlabeled)))/len(list(set(idx_view_unlabeled)))
            #print("accuracy unlabeled: ",acc_unlabeled)

            selected_by_margin = np.where(margins > 0)[0]
            # Convert to sets for easier comparison
            selected_set = set(selected_by_margin)
            clean_set = set(inds_clean)
            # Compute true positives: selected samples that are truly clean
            true_positives = selected_set & clean_set
            # Precision: TP / (TP + FP) = TP / total selected
            if len(selected_set) > 0:
                precision_margin = len(true_positives) / len(selected_set)
            else:
                precision_margin = 0.0  # avoid division by zero
            #print("Precision of samples with margin > 0:", precision_margin)
            
            
            import numpy as np
            from collections import Counter



            # Flatten copies
            flat_clean = np.ravel(clean_labels)
            flat_pred = np.ravel(predicted_labels)

            def top_errors_with_confusion(idx_subset, name="labeled"):
                subset_clean = flat_clean[idx_subset]
                subset_pred = flat_pred[idx_subset]

                # Get incorrect predictions
                mask_wrong = subset_clean != subset_pred
                wrong_true = subset_clean[mask_wrong].astype(int)
                wrong_pred = subset_pred[mask_wrong].astype(int)

                # Count errors per true class
                error_counts = Counter(wrong_true)
                top_errors = error_counts.most_common(5)

                print(f"\nTop 5 classes with most errors ({name}):")
                for class_id, count in top_errors:
                    # Filter all predictions that were made instead of the correct class_id
                    wrong_preds_for_class = wrong_pred[wrong_true == class_id]
                    most_common_wrong_pred, num_misclassified_to_it = Counter(wrong_preds_for_class).most_common(1)[0]
                    print(f"Class {class_id}: {count} errors (→ misclassified most often as class {most_common_wrong_pred}, {num_misclassified_to_it} times)")

            # Labeled
            #top_errors_with_confusion(idx_view_labeled, name="labeled")

            # Unlabeled
            #top_errors_with_confusion(idx_view_unlabeled, name="unlabeled")


            dissonance2 = np.asarray(dissonance)
            dissonance_unlabeled = dissonance2[idx_view_unlabeled]
            thresholds = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]

            log_file = dissonance_log  # ✅ use your file handle

            log_file.write(f"[Epoch {epoch}] Accuracy on dissonance value thresholds (unlabeled data):\n")
            for t in thresholds:
                mask = dissonance_unlabeled < t
                selected_indices = idx_view_unlabeled[mask]
                num_selected = len(selected_indices)

                if num_selected == 0:
                    acc_low_dissonance = 0.0
                else:
                    acc_low_dissonance = len(set(correct_indices1) & set(selected_indices)) / num_selected

                log_file.write(
                    f"Dissonance < {t:.2f} - acc: {acc_low_dissonance:.4f} - samples: {num_selected}\n"
                )

            log_file.flush()

            
            
            """dissonance2 = np.array(dissonance)

            # Threshold for high/low dissonance2
            threshold = 0.2

            # Labeled data
            dissonance_labeled = dissonance2[idx_view_labeled]
            high_labeled_mask = dissonance_labeled >= threshold
            low_labeled_mask = dissonance_labeled < threshold

            high_labeled_indices = idx_view_labeled[high_labeled_mask]
            low_labeled_indices = idx_view_labeled[low_labeled_mask]

            # Unlabeled data
            dissonance_unlabeled = dissonance2[idx_view_unlabeled]
            high_unlabeled_mask = dissonance_unlabeled >= threshold
            low_unlabeled_mask = dissonance_unlabeled < threshold

            high_unlabeled_indices = idx_view_unlabeled[high_unlabeled_mask]
            low_unlabeled_indices = idx_view_unlabeled[low_unlabeled_mask]

            # Accuracy for labeled subsets: proportion of clean indices
            high_labeled_accuracy = np.isin(high_labeled_indices, inds_clean).mean()
            low_labeled_accuracy = np.isin(low_labeled_indices, inds_clean).mean()

            # Accuracy for unlabeled subsets: predicted_label == clean_label
            high_unlabeled_accuracy = (predicted_labels[high_unlabeled_indices] == clean_labels[high_unlabeled_indices]).float().mean()
            low_unlabeled_accuracy = (predicted_labels[low_unlabeled_indices] == clean_labels[low_unlabeled_indices]).float().mean()

            # Report
            print("Labeled - High Dissonance Accuracy (Clean Proportion):", high_labeled_accuracy)
            print("Labeled - Low Dissonance Accuracy (Clean Proportion):", low_labeled_accuracy)
            print("Unlabeled - High Dissonance Accuracy (Prediction Correctness):", high_unlabeled_accuracy)
            print("Unlabeled - Low Dissonance Accuracy (Prediction Correctness):", low_unlabeled_accuracy)
            print("dissonance mean: ", dissonance2.mean(), dissonance_labeled.mean(), dissonance_unlabeled.mean())"""




            

            def get_metrics(predicted, clean_labels, idx, clean_indices, predicted_probs):
                # Ensure torch tensors
                predicted = torch.tensor(predicted) if not torch.is_tensor(predicted) else predicted
                clean_labels_tensor = torch.tensor(clean_labels, dtype=torch.long)
                idx = torch.tensor(idx) if not torch.is_tensor(idx) else idx
                clean_indices = torch.tensor(clean_indices) if not torch.is_tensor(clean_indices) else clean_indices
                noisy_labels_tensor = torch.tensor(noisy_labels, dtype=torch.long)
                predicted_probs = torch.tensor(predicted_probs) if not torch.is_tensor(predicted_probs) else predicted_probs

                if len(idx) == 0:
                    return 0.0, 0.0, 0.0, {}  # Avoid divide-by-zero

                # Accuracy: compare predictions vs. true labels on selected indices
                y_pred = predicted[idx]
                y_true = clean_labels_tensor[idx]
                y_true_noisy = noisy_labels_tensor[idx]
                acc = (y_pred == y_true).sum().item() / len(idx)
                acc2 = (y_pred == y_true_noisy).sum().item() / len(idx)

                # Precision and Recall
                idx_set = set(idx.tolist())
                clean_set = set(clean_indices.tolist())

                true_positives = len(idx_set & clean_set)
                false_positives = len(idx_set - clean_set)
                false_negatives = len(clean_set - idx_set)

                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0.0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0.0

                transitions = defaultdict(lambda: [0, 0, 0])

                noisy = noisy_labels_tensor[idx]
                clean = clean_labels_tensor[idx]
                incorrect_mask = y_pred != noisy  # Only consider cases where prediction ≠ noisy label

                for pred_class, noisy_class, clean_class in zip(y_pred[incorrect_mask], noisy[incorrect_mask], clean[incorrect_mask]):
                    key = (noisy_class.item(), pred_class.item())
                    if noisy_class.item() == clean_class.item():
                        transitions[key][0] += 1  # correct transition
                    else:
                        transitions[key][1] += 1  # incorrect transition
                    transitions[key][2] += 1

                total_transitions = sum(value[2] for value in transitions.values())
                print(f"Total number of transitions: {total_transitions}", acc, acc2, len(idx))

                noisy_label_counts = Counter(noisy.tolist())
                transition_matrix_percent = {}
                for (noisy_lbl, pred_lbl), (_, _, total_transitions) in transitions.items():
                    total_noisy_count = noisy_label_counts[noisy_lbl]
                    percent = total_transitions / total_noisy_count if total_noisy_count > 0 else 0.0
                    transition_matrix_percent[(noisy_lbl, pred_lbl)] = percent


                """sorted_transitions = sorted(transitions.items(), key=lambda x: x[1][2], reverse=True)
                print("Top 5 most frequent transitions (noisy_label → predicted_label):")
                for (noisy, pred), (correct, incorrect, total) in sorted_transitions[:5]:
                    print(f"  {noisy} → {pred}: {total} times (Correct: {correct}, Incorrect: {incorrect})")"""
                
                prob_diff_matrix = defaultdict(list)
                for i in idx[incorrect_mask]:
                    i = i.item()
                    pred_class = predicted[i].item()
                    noisy_class = noisy_labels_tensor[i].item()

                    prob_pred = predicted_probs[i, pred_class].item()
                    prob_noisy = predicted_probs[i, noisy_class].item()
                    diff = prob_pred - prob_noisy

                    prob_diff_matrix[(noisy_class, pred_class)].append(diff)

                mean_prob_diff_matrix = {
                    key: sum(diffs) / len(diffs) if diffs else 0.0
                    for key, diffs in prob_diff_matrix.items()
                }


                return 100 * acc, 100 * precision, 100 * recall, transition_matrix_percent,  mean_prob_diff_matrix


            acc, prec, recall, transition_matrix_percent, mean_prob_diff_matrix = get_metrics(predicted_labels, clean_labels, (pred_loss1).nonzero()[0], clean_indices=inds_clean, predicted_probs=predicted_probs)
            new_predicted_labels = predicted_labels.clone()
            total_relabel = 0
            
            
            excluded_idx = []
            #excluded_idx = (pred_loss1).nonzero()[0]
            excluded_set = set(excluded_idx)
            # Apply relabeling using transition_matrix_percent
            """for pred_class in range(predicted_probs.shape[1]):
                # Find all transitions where the predicted class is the target
                for (noisy_label, predicted_label), percent in transition_matrix_percent.items():
                    if predicted_label != pred_class:
                        continue

                    # Indices where current predicted label is pred_class
                    pred_indices = (predicted_labels == pred_class).nonzero(as_tuple=True)[0]

                    if len(pred_indices) == 0:
                        continue

                    # Filter pred_indices to exclude high-loss samples
                    filtered_pred_indices = [i.item() for i in pred_indices if i.item() not in excluded_set]

                    if len(filtered_pred_indices) == 0:
                        continue

                    # Convert back to tensor for indexing
                    filtered_pred_indices = torch.tensor(filtered_pred_indices, dtype=torch.long)

                    # Get model's probability for noisy_label on these samples
                    noisy_probs = predicted_probs[filtered_pred_indices, noisy_label]

                    # Number of samples to relabel based on transition percent
                    factor = 1
                    num_to_relabel = int(percent * len(filtered_pred_indices)*factor)
                    total_relabel += num_to_relabel
                    if num_to_relabel == 0:
                        continue

                    # Get indices of top samples based on probability for the noisy label
                    topk = torch.topk(noisy_probs, k=num_to_relabel)
                    top_indices = filtered_pred_indices[topk.indices]

                    # Relabel those samples from pred_class to noisy_label
                    new_predicted_labels[top_indices] = noisy_label"""
            

            adjusted_probs = predicted_probs.clone()
            total_adjusted = 0

            for pred_class in range(predicted_probs.shape[1]):
                for (noisy_label, predicted_label), percent in transition_matrix_percent.items():
                    if predicted_label != pred_class:
                        continue

                    # Get indices where current predicted label is pred_class
                    pred_indices = (predicted_labels == pred_class).nonzero(as_tuple=True)[0]
                    if len(pred_indices) == 0:
                        continue

                    # Filter out excluded indices
                    filtered_pred_indices = [i.item() for i in pred_indices if i.item() not in excluded_set]
                    if len(filtered_pred_indices) == 0:
                        continue

                    filtered_pred_indices = torch.tensor(filtered_pred_indices, dtype=torch.long)

                    # Shift probability mass for ALL filtered samples (no top-k)
                    original_pred_probs = predicted_probs[filtered_pred_indices, pred_class]
                    factor = 1
                    prob_shift = factor*original_pred_probs * percent

                    adjusted_probs[filtered_pred_indices, noisy_label] += prob_shift
                    adjusted_probs[filtered_pred_indices, pred_class] -= prob_shift  

                    total_adjusted += len(filtered_pred_indices)


            new_predicted_labels = torch.argmax(adjusted_probs, dim=1)

            correct_indices1 = np.where(clean_labels == predicted_labels)[0]
            acc1 = len(list(set(correct_indices1)))

            correct_indices2 = np.where(clean_labels == new_predicted_labels)[0]
            acc2 = len(list(set(correct_indices2)))

            print("accuracies: ", acc1, acc2, acc1-acc2)


            #plot_hist_curve_loss_test(data_hist= test_losses_hist, path=path_plot, epoch=epoch )
            thres1 = max(all_loss[0][epoch2][all_idx_view_labeled[0][epoch2]])
            thres2 = min(all_loss[0][epoch2][all_idx_view_unlabeled[0][epoch2]])
            threshold_loss = [thres1.item()]

            #plot_histogram_metric(data_hist=all_loss[0][epoch2], inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= idx_view_labeled, inds_relabeled = relabel_idx_1, thresholds = threshold_loss, path=path_plot, epoch=epoch, metric = "Loss" , noisy_labels=noisy_labels)
            #plot_histogram_metric(data_hist=margins_plot, inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= idx_view_labeled, inds_relabeled = relabel_idx_1, thresholds = [], path=path_plot, epoch=epoch, metric = "Margin_True"  )
            #f plr_loss:
                #plot_histogram_metric3(data_hist=all_proto_loss[0][epoch2], inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= idx_view_labeled, inds_relabeled = relabel_idx_1, thresholds = threshold_loss, path=path_plot, epoch=epoch, metric = "Prototype_loss"  )
            #plot_histogram_metric2(data_hist=margin, inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= idx_view_labeled, thresholds = [], path=path_plot, epoch=epoch, data_hist_2= all_loss[0][epoch2], data_hist_3= preds, metric = "Margins_adjust"  )
            #plot_histogram_metric2(data_hist=noisy_margins, inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= idx_view_labeled, thresholds = [], path=path_plot, epoch=epoch, data_hist_2= all_loss[0][epoch2], data_hist_3= preds, metric = "True_margin"  )
            #plot_histogram_metric2(data_hist=vacuity, inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= idx_view_labeled, thresholds = [], path=path_plot, epoch=epoch, data_hist_2= all_loss[0][epoch2], data_hist_3= preds, metric = "Vacuity" , noisy_labels=noisy_labels)
            #plot_histogram_metric22(data_hist=vacuity, inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= idx_view_labeled, thresholds = [], path=path_plot, epoch=epoch, data_hist_2= all_loss[0][epoch2], data_hist_3= preds, metric = "Vacuity" ,   )
            #plot_histogram_metric2(data_hist=entropy, inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= idx_view_labeled, thresholds = [], path=path_plot, epoch=epoch, data_hist_2= all_loss[0][epoch2], data_hist_3= preds, metric = "Entropy"  )
            #plot_histogram_metric2(data_hist=dissonance, inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= idx_view_labeled,thresholds = [], path=path_plot, epoch=epoch, data_hist_2= all_loss[0][epoch2], data_hist_3= preds, metric = "Dissonance", noisy_labels=noisy_labels  )
            #plot_histogram_metric22(data_hist=dissonance, inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= idx_view_labeled,thresholds = [], path=path_plot, epoch=epoch, data_hist_2= all_loss[0][epoch2], data_hist_3= preds, metric = "Dissonance"  )
            
            #plot_histogram_metric22(data_hist=dissonance, inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= idx_view_labeled,thresholds = [], path=path_plot, epoch=epoch, data_hist_2= vacuity, data_hist_3= preds, metric = "Vac-Diss", divide = True, noisy_labels = noisy_labels  )
            #plot_histogram_metric2(data_hist=mutual_information, inds_clean=inds_clean, inds_correct=correct_indices1, inds_labeled= idx_view_labeled, thresholds = [], path=path_plot, epoch=epoch, data_hist_2= all_loss[0][epoch2], data_hist_3= preds, metric = "Mutual_information"  )
            #scatter_plot_probs(prob_true_class, prob_noisy_class, best_prob, evi_noisy_class, evi_true_class, max_evi_excl_noisy, idx_view_labeled, correct_indices1, inds_clean, path=path_plot, epoch=epoch)



    
    test_log.close()  
    time_log.close()
    superclean_log.close()
    ranks_log.close()
    metrics_log.close()
    dissonance_log.close()
    bm_log.close()








