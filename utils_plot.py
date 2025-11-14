import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import torch
import os


def compute_histogram_bins2(data, desired_bin_size):
    min_val = np.min(data)
    max_val = np.max(data)
    min_boundary = -1.0 * (min_val % desired_bin_size - min_val)
    max_boundary = max_val - max_val % desired_bin_size + desired_bin_size
    n_bins = int((max_boundary - min_boundary) / desired_bin_size) + 1
    bins = np.linspace(min_boundary, max_boundary, n_bins)
    return bins

def compute_histogram_bins(data, num_bins=100):
    min_val = np.min(data)
    max_val = np.max(data)
    bins = np.linspace(min_val, max_val, num_bins + 1)  # 101 edges → 100 bins
    return bins


def plot_histogram_pred(data, entropy, inds_clean, inds_noisy, path, epoch):
    data = data[-1].numpy()
    entropy = entropy[-1].numpy()
    bins = compute_histogram_bins(data)
    inds_clean = np.array(inds_clean, dtype=int)
    inds_noisy = np.array(inds_noisy, dtype=int)

    num_inds_clean = len(inds_clean)
    num_inds_noisy = len(inds_noisy)
    perc_clean = 100*num_inds_clean/float(num_inds_clean+num_inds_noisy)

    plt.hist(data[inds_clean],bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5, label='clean - %d (%.1f%%)'%(num_inds_clean,perc_clean))
    if len(inds_noisy) >0:
        plt.hist(data[inds_noisy], bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5, label='noisy- %d (%.1f%%)'%(num_inds_noisy,100-perc_clean))
    plt.xlabel('Prob');
    plt.ylabel('Number of data')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
       ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig('%s/Pred_Epoch%03d.png' % (path,epoch))
    plt.clf() 

    plt.figure(figsize=(8, 6))  # Adjust figure size for better visibility
    plt.scatter(data[inds_clean], entropy[inds_clean], color='green', alpha=0.5, label='Clean')
    plt.scatter(data[inds_noisy], entropy[inds_noisy], color='red', alpha=0.5, label='Noisy')
    
    plt.xlabel('Probability')
    plt.ylabel('Entropy')
    plt.title('Probability vs Entropy Scatter Plot')
    plt.legend()
    plt.savefig(f'{path}/Pred_vs_Entropy_Epoch{epoch:03d}.png')
    plt.clf()

    
def plot_curve_loss(data_hist, inds_clean, inds_noisy, path):
    # History plot with three curves
    all_loss_per_epoch = [np.mean(np.array(epoch_loss)) for epoch_loss in data_hist]
    clean_loss_per_epoch = [np.mean(np.array(epoch_loss)[inds_clean]) if len(inds_clean) > 0 else np.nan for epoch_loss in data_hist]
    noisy_loss_per_epoch = [np.mean(np.array(epoch_loss)[inds_noisy]) if len(inds_noisy) > 0 else np.nan for epoch_loss in data_hist]
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(data_hist) + 1), all_loss_per_epoch, linestyle='-', color='red', label='All Loss')
    plt.plot(range(1, len(data_hist) + 1), clean_loss_per_epoch, linestyle='--', color='green', label='Clean Loss')
    plt.plot(range(1, len(data_hist) + 1), noisy_loss_per_epoch, linestyle='-.', color='blue', label='Noisy Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Across Epochs')
    plt.legend()
    plt.savefig(f'{path}/WarmUp_Loss_curve.png')
    plt.clf()



def plot_curve_loss_train(data_hist, path):
    # Convert all to tensors if needed
    all_loss_per_epoch = torch.tensor(data_hist[0]) if isinstance(data_hist[0], list) else data_hist[0]
    clean_loss_per_epoch = torch.tensor(data_hist[1]) if isinstance(data_hist[1], list) else data_hist[1]
    noisy_loss_per_epoch = torch.tensor(data_hist[2]) if isinstance(data_hist[2], list) else data_hist[2]
    contrastive_loss_raw = data_hist[3]
    contrastive_loss_per_epoch = torch.tensor(contrastive_loss_raw) if isinstance(contrastive_loss_raw, list) else contrastive_loss_raw
    noisy_loss_per_epoch *= 500  # Apply scaling

    # Convert to NumPy arrays, move to CPU if necessary
    all_loss_per_epoch = all_loss_per_epoch.cpu().numpy() if isinstance(all_loss_per_epoch, torch.Tensor) else np.array(all_loss_per_epoch)
    clean_loss_per_epoch = clean_loss_per_epoch.cpu().numpy() if isinstance(clean_loss_per_epoch, torch.Tensor) else np.array(clean_loss_per_epoch)
    noisy_loss_per_epoch = noisy_loss_per_epoch.cpu().numpy() if isinstance(noisy_loss_per_epoch, torch.Tensor) else np.array(noisy_loss_per_epoch)
    contrastive_loss_per_epoch = contrastive_loss_per_epoch.cpu().numpy() if isinstance(contrastive_loss_per_epoch, torch.Tensor) else np.array(contrastive_loss_per_epoch)

    # Epochs start at 5
    num_epochs = len(all_loss_per_epoch)
    epoch_range = range(5, 5 + num_epochs)

    # Plot 1: All losses
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_range, clean_loss_per_epoch, linestyle='--', color='red', linewidth=3, label='Supervised Loss')
    plt.plot(epoch_range, noisy_loss_per_epoch, linestyle='-.', color='blue', linewidth=3, label='Unsupervised Loss')
    plt.plot(epoch_range, contrastive_loss_per_epoch, linestyle='--', color='green', linewidth=3, label='Contrastive Loss')

    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.title('Warmup Loss across Epochs', fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{path}/SSLoss_curve.png')
    plt.clf()

    plt.figure(figsize=(12, 6))  # Slightly wider

    plt.plot(epoch_range, clean_loss_per_epoch, linestyle='--', color='red', linewidth=3, label='Supervised')
    plt.plot(epoch_range, noisy_loss_per_epoch, linestyle='-.', color='blue', linewidth=3, label='Unsupervised')
    plt.plot(epoch_range, contrastive_loss_per_epoch, linestyle='--', color='green', linewidth=3, label='Contrastive')

    plt.xlabel('Epoch', fontsize=24)
    plt.ylabel('Loss', fontsize=24)
    plt.title('Loss across Epochs', fontsize=28)

    plt.xticks(fontsize=20)

    # Set custom y-ticks (e.g., 0 to max_loss rounded up, every 0.5)
    max_loss = max(max(clean_loss_per_epoch), max(noisy_loss_per_epoch), max(contrastive_loss_per_epoch))
    yticks = [round(y, 1) for y in plt.yticks()[0] if y % 0.5 == 0]
    max_loss = max(
    max(clean_loss_per_epoch),
    max(noisy_loss_per_epoch),
    max(contrastive_loss_per_epoch)
    )
    y_max = int(np.ceil(max_loss / 1000.0)) * 1000
    plt.yticks(np.arange(0, y_max + 1000, 1000), fontsize=20)


    plt.legend(
        fontsize=20,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.25),
        ncol=3,
        frameon=False
    )

    plt.tight_layout()
    plt.savefig(f'{path}/SSandContrastiveLoss_curve.png')
    plt.clf()


def plot_curve_accuracy(data_hist, inds_clean, inds_noisy, path):
    # History plot with three curves
    all_loss_per_epoch = [epoch_loss[0] for epoch_loss in data_hist]
    clean_loss_per_epoch = [epoch_loss[1] for epoch_loss in data_hist]
    noisy_loss_per_epoch = [epoch_loss[2] for epoch_loss in data_hist]

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(data_hist) + 1), all_loss_per_epoch, linestyle='-', color='red', label='All acc')
    plt.plot(range(1, len(data_hist) + 1), clean_loss_per_epoch, linestyle='--', color='green', label='Clean acc')
    plt.plot(range(1, len(data_hist) + 1), noisy_loss_per_epoch, linestyle='-.', color='blue', label='Noisy acc')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Across Epochs')
    plt.legend()
    plt.savefig(f'{path}/Train_Accuracy_curve.png')
    plt.clf()

def plot_curve_accuracy_test(data_hist, inds_clean, inds_noisy, path, epoch):
    # History plot with three curves
    all_loss_per_epoch = [epoch_loss for epoch_loss in data_hist]
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(data_hist) + 1), all_loss_per_epoch, linestyle='-', color='red', label='All acc')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Across Epochs')
    plt.legend()
    plt.savefig(f'{path}/Test_Accuracy_curve.png')
    plt.clf()

def plot_hist_curve_loss_test(data_hist, path, epoch):

    #histogram
    data = data_hist[-1].numpy()
    bins = compute_histogram_bins(data)

    plt.hist(data,bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5, label='all')
    plt.xlabel('Loss')
    plt.ylabel('Number of data')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
       ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig('%s/Test_Loss_histogram_epoch%03d.png' % (path,epoch))
    plt.clf()

    #curve
    all_metric_per_epoch = [np.mean(np.array(epoch_data)) for epoch_data in data_hist]
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(data_hist) + 1), all_metric_per_epoch, linestyle='-', color='red',
             label=f'All loss')
    plt.xlabel('Epoch')
    plt.ylabel(f'Mean loss')
    plt.title(f'Mean loss Across Epochs')
    plt.legend()
    plt.savefig(f'{path}/Test_loss_curve.png')
    plt.clf()

def plot_histogram_metric(data_hist, inds_clean, inds_correct, inds_labeled, inds_relabeled, thresholds, path, epoch, metric="Uncertainty", noisy_labels =None):

    # Convert data_hist to numpy array
    if isinstance(data_hist, torch.Tensor):
        data = data_hist.detach().cpu().numpy()
    elif isinstance(data_hist, (list, tuple)):
        try:
            data = torch.tensor(data_hist).numpy()
        except Exception as e:
            print(f"Failed to convert list to tensor: {e}")
            print(f"Data: {data_hist}")
            raise
    else:
        raise TypeError(f"Unsupported data type in data_hist: {type(data_hist)}")

    total_samples = len(data)

    # Prepare directories and indices
    metric_path = os.path.join(path, metric)
    os.makedirs(metric_path, exist_ok=True)
    inds_unlabeled = list(set(range(total_samples)) - set(inds_labeled))
    inds_noisy = list(set(range(total_samples)) - (set(inds_clean) | set(inds_relabeled)))
    inds_error = list(set(range(total_samples)) - set(inds_correct))

    bins = compute_histogram_bins(data)
    bins2 = compute_histogram_bins(data, num_bins=20)

    # Plot histogram: clean vs noisy (including relabeled)
    num_inds_clean = len(inds_clean)
    num_inds_noisy = len(inds_noisy)
    perc_clean = 100 * num_inds_clean / float(num_inds_clean + num_inds_noisy) if (num_inds_clean + num_inds_noisy) > 0 else 0

    # White figure background
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')

    ax.hist(data[inds_clean], bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5,
            label=f'Clean - {num_inds_clean} ({perc_clean:.1f}%)', color='blue')
    if len(inds_noisy) > 0:
        ax.hist(data[inds_noisy], bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5,
                label=f'Noisy - {num_inds_noisy} ({100 - perc_clean:.1f}%)', color='red')
    if len(inds_relabeled) > 0:
        ax.hist(data[inds_relabeled], bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5,
                label=f'Relabeled - {len(inds_relabeled)}', color='green')

    ax.set_ylim(0, 3500)
    ax.set_xlabel(metric, fontsize=24)
    ax.set_ylabel('Number of Data', fontsize=24)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand",
                  borderaxespad=0., fontsize=20)

    fig.tight_layout()
    fig.savefig(f'{metric_path}/{metric}_clean_epoch{epoch:03d}.png')
    plt.close(fig)

    if noisy_labels is not None and len(noisy_labels) > 0:
            noisy_labels_array = np.array(noisy_labels)
            labeled_noisy_indices = list(set(inds_noisy) & set(inds_labeled))
            noisy_classes_in_labeled = np.unique(noisy_labels_array[labeled_noisy_indices])
            selected_classes = np.random.choice(noisy_classes_in_labeled, size=min(5, len(noisy_classes_in_labeled)), replace=False)
            selected_classes = [5, 20, 37, 86, 93]

            for class_id in selected_classes:
                    # Plot histogram for clean vs. noisy (labeled only)
                    inds_clean_labeled = [i for i in inds_clean if i in inds_labeled and noisy_labels[i] == class_id]
                    inds_noisy_labeled = [i for i in inds_noisy if i in inds_labeled and noisy_labels[i] == class_id]

                    plt.hist(data[inds_clean_labeled], bins=bins2, range=(0., 1.), edgecolor='black', alpha=0.5,
                            label=f'clean - {len(inds_clean_labeled)}')
                    if len(inds_noisy_labeled) > 0:
                        plt.hist(data[inds_noisy_labeled], bins=bins2, range=(0., 1.), edgecolor='black', alpha=0.5,
                                label=f'Noisy - {len(inds_noisy_labeled)}')

                    plt.xlabel(metric, fontsize=14)
                    plt.ylabel('Number of Data', fontsize=14)
                    plt.xticks(fontsize=12)
                    plt.yticks(fontsize=12)
                    class_metric_path = os.path.join(path, f"{metric}_{class_id}")
                    os.makedirs(class_metric_path, exist_ok=True)
                    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
                    plt.savefig(f'{class_metric_path}/{metric}_clean_noisy_epoch{epoch:03d}.png')
                    plt.clf()
                    



def plot_histogram_metric3(data_hist, inds_clean, inds_correct, inds_labeled, inds_relabeled, thresholds, path, epoch, metric="Uncertainty"):

    # Convert data_hist to numpy array
    if isinstance(data_hist, torch.Tensor):
        data = data_hist.detach().cpu().numpy()
    elif isinstance(data_hist, (list, tuple)):
        try:
            data = torch.tensor(data_hist).numpy()
        except Exception as e:
            print(f"Failed to convert list to tensor: {e}")
            print(f"Data: {data_hist}")
            raise
    else:
        raise TypeError(f"Unsupported data type in data_hist: {type(data_hist)}")

    total_samples = len(data)

    # Prepare directories and indices
    metric_path = os.path.join(path, metric)
    os.makedirs(metric_path, exist_ok=True)
    inds_unlabeled = list(set(range(total_samples)) - set(inds_labeled))
    inds_noisy = list(set(range(total_samples)) - (set(inds_clean) | set(inds_relabeled)))
    inds_error = list(set(range(total_samples)) - set(inds_correct))

    bins = compute_histogram_bins(data)

    # Plot histogram: clean vs noisy (including relabeled)
    num_inds_clean = len(inds_clean)
    num_inds_noisy = len(inds_noisy)
    perc_clean = 100 * num_inds_clean / float(num_inds_clean + num_inds_noisy) if (num_inds_clean + num_inds_noisy) > 0 else 0

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')  # white background for the figure only

    ax.hist(data[inds_clean], bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5,
            label=f'Clean - {num_inds_clean} ({perc_clean:.1f}%)', color='blue')
    if len(inds_noisy) > 0:
        ax.hist(data[inds_noisy], bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5,
                label=f'Noisy - {num_inds_noisy} ({100 - perc_clean:.1f}%)', color='red')
    if len(inds_relabeled) > 0:
        ax.hist(data[inds_relabeled], bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5,
                label=f'Relabeled - {len(inds_relabeled)}', color='green')

    # Keep axes background default (transparent or grid background) by not changing ax.set_facecolor()

    ax.set_xlabel(metric, fontsize=24)
    ax.set_ylabel('Number of Data', fontsize=24)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand",
                  borderaxespad=0., fontsize=20)

    fig.tight_layout()
    fig.savefig(f'{metric_path}/{metric}_clean_epoch{epoch:03d}.png')
    plt.close(fig)




def plot_histogram_metric22(data_hist, inds_clean, inds_correct, inds_labeled, thresholds, path, epoch, data_hist_2, data_hist_3, metric="True_margin", divide = False, noisy_labels = None):
    if metric != "Mean_uncertainty":
        data = data_hist.numpy()
    else:
        data = data_hist

    total_samples = len(data)

    loss = data_hist_2.numpy()
    preds = data_hist_3.numpy()

    # Create metric-specific folder
    metric_path = os.path.join(path, metric)
    os.makedirs(metric_path, exist_ok=True)

    if isinstance(inds_labeled, torch.Tensor):
        inds_labeled = inds_labeled.numpy()

    mean_labeled = data[inds_labeled].mean()

    inds_error = list(set(range(total_samples)) - set(inds_correct))
    inds_noisy = list(set(range(total_samples)) - set(inds_clean))
    inds_unlabeled = list(set(range(total_samples)) - set(inds_labeled))

    # Grouping
    clean_correct = list(set(inds_clean) & set(inds_correct))
    clean_error = list(set(inds_clean) & set(inds_error))
    noisy_correct = list(set(inds_noisy) & set(inds_correct))
    noisy_error = list(set(inds_noisy) & set(inds_error))

    inds_labeled_set = set(inds_labeled)
    inds_unlabeled_set = set(inds_unlabeled)
    if divide:
        # Filter each group to keep only labeled indices
        clean_correct_labeled = list(set(clean_correct) & inds_labeled_set)
        clean_error_labeled   = list(set(clean_error) & inds_labeled_set)
        noisy_correct_labeled = list(set(noisy_correct) & inds_labeled_set)
        noisy_error_labeled   = list(set(noisy_error) & inds_labeled_set)
    # Filter each group to keep only labeled indices
        clean_correct_unlabeled = list(set(clean_correct) & inds_unlabeled_set)
        clean_error_unlabeled   = list(set(clean_error) & inds_unlabeled_set)
        noisy_correct_unlabeled = list(set(noisy_correct) & inds_unlabeled_set)
        noisy_error_unlabeled   = list(set(noisy_error) & inds_unlabeled_set)


    if not divide:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')  # Create one figure with white background
        ax.set_facecolor('white')  # Optional: also set axes background to white

        ax.scatter(loss[clean_correct], data[clean_correct], color='navy', alpha=0.5, s=3, label=f'Clean & Correct ({len(clean_correct)})')
        ax.scatter(loss[clean_error], data[clean_error], color='skyblue', alpha=0.5, s=3, label=f'Clean & Error ({len(clean_error)})')
        ax.scatter(loss[noisy_correct], data[noisy_correct], color='orange', alpha=0.5, s=3, label=f'Noisy & Correct ({len(noisy_correct)})')
        ax.scatter(loss[noisy_error], data[noisy_error], color='firebrick', alpha=0.5, s=3, label=f'Noisy & Error ({len(noisy_error)})')

        # Axis, title, etc.
        # Make axis spines visible and bold
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.5)
        ax.set_xlabel('Loss', fontsize=24)
        ax.set_ylabel(metric, fontsize=24)
        ax.set_title(f'{metric} vs Loss (Epoch {epoch})', fontsize=28)
        ax.tick_params(axis='both', labelsize=20)
        ax.grid(True)

        # Save
        fig.tight_layout()
        fig.savefig(f'{metric_path}/{metric}_epoch_{epoch:03d}.png', facecolor='white')
        plt.close(fig)

    if divide:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')  # Create one figure with white background
        ax.set_facecolor('white')  # Optional: also set axes background to white

        ax.scatter(loss[clean_correct_labeled], data[clean_correct_labeled], color='navy', alpha=0.5, s=3, label=f'Clean & Correct ({len(clean_correct_labeled)})')
        ax.scatter(loss[clean_error_labeled], data[clean_error_labeled], color='skyblue', alpha=0.5, s=3, label=f'Clean & Error ({len(clean_error_labeled)})')
        ax.scatter(loss[noisy_correct_labeled], data[noisy_correct_labeled], color='orange', alpha=0.5, s=3, label=f'Noisy & Correct ({len(noisy_correct_labeled)})')
        ax.scatter(loss[noisy_error_labeled], data[noisy_error_labeled], color='firebrick', alpha=0.5, s=3, label=f'Noisy & Error ({len(noisy_error_labeled)})')

        # Axis, title, etc.
        # Make axis spines visible and bold
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.5)
        ax.set_xlabel('Vacuity', fontsize=24)
        ax.set_ylabel('Diss', fontsize=24)
        ax.set_title(f'Vacuity - Dissonance (Epoch {epoch})', fontsize=28)
        ax.tick_params(axis='both', labelsize=20)
        ax.grid(True)
        plt.legend()

        # Save
        fig.tight_layout()
        fig.savefig(f'{metric_path}/{metric}_labeled_epoch_{epoch:03d}.png', facecolor='white')
        plt.close(fig)








        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')  # Create one figure with white background
        ax.set_facecolor('white')  # Optional: also set axes background to white

        ax.scatter(loss[clean_correct_unlabeled], data[clean_correct_unlabeled], color='navy', alpha=0.5, s=3, label=f'Clean & Correct ({len(clean_correct_unlabeled)})')
        ax.scatter(loss[clean_error_unlabeled], data[clean_error_unlabeled], color='skyblue', alpha=0.5, s=3, label=f'Clean & Error ({len(clean_error_unlabeled)})')
        ax.scatter(loss[noisy_correct_unlabeled], data[noisy_correct_unlabeled], color='orange', alpha=0.5, s=3, label=f'Noisy & Correct ({len(noisy_correct_unlabeled)})')
        ax.scatter(loss[noisy_error_unlabeled], data[noisy_error_unlabeled], color='firebrick', alpha=0.5, s=3, label=f'Noisy & Error ({len(noisy_error_unlabeled)})')

        # Axis, title, etc.
        # Make axis spines visible and bold
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.5)
        ax.set_xlabel('Vacuity', fontsize=24)
        ax.set_ylabel('Diss', fontsize=24)
        ax.set_title(f'Vacuity - Dissonance (Epoch {epoch})', fontsize=28)
        ax.tick_params(axis='both', labelsize=20)
        ax.grid(True)
        plt.legend()

        # Save
        fig.tight_layout()
        fig.savefig(f'{metric_path}/{metric}_unlabeled_epoch_{epoch:03d}.png', facecolor='white')
        plt.close(fig)

    
    
    
    
    fig_legend = plt.figure(figsize=(12, 2))  # Wider figure
    ax = fig_legend.add_subplot(111)

    # Plot invisible points to create legend handles
    scatter_handles = [
        plt.Line2D([0], [0], marker='o', color='w', 
                markerfacecolor='navy', markersize=20, alpha=0.5, 
                label='Clean & Correct'),
        plt.Line2D([0], [0], marker='o', color='w', 
                markerfacecolor='skyblue', markersize=20, alpha=0.5, 
                label='Clean & Incorrect'),
        plt.Line2D([0], [0], marker='o', color='w', 
                markerfacecolor='orange', markersize=20, alpha=0.5, 
                label='Noisy & Correct'),
        plt.Line2D([0], [0], marker='o', color='w',
                markerfacecolor='firebrick', markersize=20, alpha=0.5, 
                label='Noisy & Incorrect'),
    ]

    # Add legend to this new figure
    ax.legend(
        handles=scatter_handles,
        loc='center',
        ncol=2,  # Change to 1 or 2 columns to better fit
        fontsize=20,
        frameon=False,
        handletextpad=0.8,
        columnspacing=1.5,
    )

    ax.axis('off')  # Hide axes

    # Save the legend figure separately
    fig_legend.tight_layout()
    fig_legend.savefig(f'{metric_path}/{metric}_legend.png', facecolor='white')
    plt.close(fig_legend)


    if metric == "Vac-Diss":
        if noisy_labels is not None and len(noisy_labels) > 0:

            noisy_labels_array = np.array(noisy_labels)
            labeled_noisy_indices = list(set(inds_noisy) & set(inds_labeled))
            noisy_classes_in_labeled = np.unique(noisy_labels_array[labeled_noisy_indices])
            selected_classes = np.random.choice(noisy_classes_in_labeled, size=min(5, len(noisy_classes_in_labeled)), replace=False)
            selected_classes = [5, 20, 37, 86, 93]

            for class_id in selected_classes:
                    inds_clean_labeled = [i for i in inds_clean if i in inds_labeled and noisy_labels[i] == class_id]
                    inds_noisy_labeled = [i for i in inds_noisy if i in inds_labeled and noisy_labels[i] == class_id]
                    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
                    ax.set_facecolor('white')

                    ax.scatter(loss[inds_clean_labeled], data[inds_clean_labeled], color='navy', alpha=0.5, s=3, label=f'Clean ({len(inds_clean_labeled)})')
                    ax.scatter(loss[inds_noisy_labeled], data[inds_noisy_labeled], color='firebrick', alpha=0.5, s=3, label=f'Noisy ({len(inds_noisy_labeled)})')

                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_color('black')
                        spine.set_linewidth(1.5)

                    ax.set_xlabel('Vacuity', fontsize=24)
                    ax.set_ylabel('Dissonance', fontsize=24)
                    ax.set_title(f'Vacuity vs Dissonance - Class {class_id} (Epoch {epoch})', fontsize=26)
                    ax.tick_params(axis='both', labelsize=20)
                    ax.grid(True)

                    fig.tight_layout()
                    class_metric_path = os.path.join(path, f"{metric}_{class_id}")
                    os.makedirs(class_metric_path, exist_ok=True)
                    fig.savefig(f'{class_metric_path}/{metric}_scatter_epoch{epoch:03d}.png', facecolor='white')
                    plt.close(fig)




def plot_histogram_metric2(data_hist, inds_clean, inds_correct, inds_labeled, thresholds, path, epoch, data_hist_2, data_hist_3, metric="True_margin", noisy_labels=False):
    if metric != "Mean_uncertainty":
        data = data_hist.numpy()
    else:
        data = data_hist

    total_samples = len(data)

    loss = data_hist_2.numpy()
    preds = data_hist_3.numpy()

    # Create metric-specific folder
    metric_path = os.path.join(path, metric)
    os.makedirs(metric_path, exist_ok=True)
    if isinstance(inds_labeled, torch.Tensor):
        inds_labeled = inds_labeled.numpy()

    mean_labeled = data[inds_labeled].mean()

    inds_error = list(set(range(total_samples)) - set(inds_correct))
    inds_noisy = list(set(range(total_samples)) - set(inds_clean))
    inds_unlabeled = list(set(range(total_samples)) - set(inds_labeled))

    bins = compute_histogram_bins(data)
    bins2 = compute_histogram_bins(data, num_bins=20)



    # Filter only labeled indices for clean/noisy
    inds_clean_labeled = list(set(inds_clean) & set(inds_labeled))
    inds_noisy_labeled = list(set(inds_noisy) & set(inds_labeled))

    # Plot histogram for clean vs. noisy (labeled only)
    plt.hist(data[inds_clean_labeled], bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5,
            label=f'clean - {len(inds_clean_labeled)}')
    if len(inds_noisy_labeled) > 0:
        plt.hist(data[inds_noisy_labeled], bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5,
                label=f'Noisy - {len(inds_noisy_labeled)}')

    plt.xlabel(metric, fontsize=14)
    plt.ylabel('Number of Data', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig(f'{metric_path}/{metric}_clean_noisy_epoch{epoch:03d}.png')
    plt.clf()

    if noisy_labels is not None and len(noisy_labels) > 0:
            noisy_labels_array = np.array(noisy_labels)
            labeled_noisy_indices = list(set(inds_noisy) & set(inds_labeled))
            noisy_classes_in_labeled = np.unique(noisy_labels_array[labeled_noisy_indices])
            selected_classes = np.random.choice(noisy_classes_in_labeled, size=min(5, len(noisy_classes_in_labeled)), replace=False)
            selected_classes = [5, 20, 37, 86, 93]

            for class_id in selected_classes:
                    # Plot histogram for clean vs. noisy (labeled only)
                    inds_clean_labeled = [i for i in inds_clean if i in inds_labeled and noisy_labels[i] == class_id]
                    inds_noisy_labeled = [i for i in inds_noisy if i in inds_labeled and noisy_labels[i] == class_id]

                    plt.hist(data[inds_clean_labeled], bins=bins2, range=(0., 1.), edgecolor='black', alpha=0.5,
                            label=f'clean - {len(inds_clean_labeled)}')
                    if len(inds_noisy_labeled) > 0:
                        plt.hist(data[inds_noisy_labeled], bins=bins2, range=(0., 1.), edgecolor='black', alpha=0.5,
                                label=f'Noisy - {len(inds_noisy_labeled)}')

                    plt.xlabel(metric, fontsize=14)
                    plt.ylabel('Number of Data', fontsize=14)
                    plt.xticks(fontsize=12)
                    plt.yticks(fontsize=12)
                    class_metric_path = os.path.join(path, f"{metric}_{class_id}")
                    os.makedirs(class_metric_path, exist_ok=True)
                    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
                    plt.savefig(f'{class_metric_path}/{metric}_clean_noisy_epoch{epoch:03d}.png')
                    plt.clf()
                    

                    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
                    ax.set_facecolor('white')

                    ax.scatter(loss[inds_clean_labeled], data[inds_clean_labeled], color='navy', alpha=0.5, s=3, label=f'Clean ({len(inds_clean_labeled)})')
                    ax.scatter(loss[inds_noisy_labeled], data[inds_noisy_labeled], color='firebrick', alpha=0.5, s=3, label=f'Noisy ({len(inds_noisy_labeled)})')

                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_color('black')
                        spine.set_linewidth(1.5)

                    ax.set_xlabel('Loss', fontsize=24)
                    ax.set_ylabel(metric, fontsize=24)
                    ax.set_title(f'{metric} vs Loss - Class {class_id} (Epoch {epoch})', fontsize=26)
                    ax.tick_params(axis='both', labelsize=20)
                    ax.grid(True)

                    fig.tight_layout()
                    fig.savefig(f'{class_metric_path}/{metric}_scatter_epoch{epoch:03d}.png', facecolor='white')
                    plt.close(fig)



   # Plot histogram for correct vs. error
    # Filter only unlabeled indices for correct/error
    inds_correct_unlabeled = list(set(inds_correct) & set(inds_unlabeled))
    inds_error_unlabeled = list(set(inds_error) & set(inds_unlabeled))

    # Plot histogram for correct vs. error (unlabeled only)
    plt.hist(data[inds_correct_unlabeled], bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5,
            label=f'correct - {len(inds_correct_unlabeled)} ')
    if len(inds_error_unlabeled) > 0:
        plt.hist(data[inds_error_unlabeled], bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5,
                label=f'error - {len(inds_error_unlabeled)} ')
    for v in thresholds:
        plt.axvline(x=v, color='red', linestyle='dashed', linewidth=2)

    plt.xlabel(metric, fontsize=14)
    plt.ylabel('Number of Data', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig(f'{metric_path}/{metric}_correct_error_epoch{epoch:03d}.png')
    plt.clf()

    """# Scatter plot: labeled vs. unlabeled
    plt.scatter(loss[inds_labeled], data[inds_labeled], color='blue', alpha=0.5, s=3, label=f'labeled - {len(inds_labeled)}')
    if len(inds_unlabeled) > 0:
        plt.scatter(loss[inds_unlabeled], data[inds_unlabeled], color='red', alpha=0.5, s=3, label=f'unlabeled - {len(inds_unlabeled)}')

    plt.xlabel('Loss', fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{metric_path}/{metric}_loss_scatter_labeled_epoch{epoch:03d}.png')
    plt.clf()"""


    """
   # Scatter plot: correct vs. error
    plt.scatter(loss[inds_correct], data[inds_correct], color='blue', alpha=0.5, s=3, label=f'correct - {len(inds_correct)}')
    if len(inds_error) > 0:
        plt.scatter(loss[inds_error], data[inds_error], color='red', alpha=0.5, s=3, label=f'error - {len(inds_error)}')

    plt.xlabel('Loss', fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{metric_path}/{metric}_loss_scatter_correct_error_epoch{epoch:03d}.png')
    plt.clf()
    """

    """# Scatter plot: preds vs. clean/noisy
    plt.scatter(preds[inds_clean], data[inds_clean], color='blue', alpha=0.5, s=3, label=f'clean - {len(inds_clean)}')
    if len(inds_noisy) > 0:
        plt.scatter(preds[inds_noisy], data[inds_noisy], color='red', alpha=0.5, s=3, label=f'noisy - {len(inds_noisy)}')

    plt.xlabel('preds', fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{metric_path}/{metric}_preds_scatter_clean_noisy_epoch{epoch:03d}.png')
    plt.clf()

    # Scatter plot: preds vs. correct/error
    plt.scatter(preds[inds_correct], data[inds_correct], color='blue', alpha=0.5, s=3, label=f'correct - {len(inds_correct)}')
    if len(inds_error) > 0:
        plt.scatter(preds[inds_error], data[inds_error], color='red', alpha=0.5, s=3, label=f'error - {len(inds_error)}')

    plt.xlabel('preds', fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{metric_path}/{metric}_preds_scatter_correct_error_epoch{epoch:03d}.png')
    plt.clf()"""


    



def scatter_plot_probs(prob_true_class, prob_noisy_class, prob_best_class, evi_noisy_class, evi_true_class, max_evi_excl_noisy, idx_view_labeled, idx_correct, idx_clean, path, epoch):
    metric = "Probs"
    metric_path = os.path.join(path, metric)
    os.makedirs(metric_path, exist_ok=True)
    prob_true_class = np.array(prob_true_class)
    prob_noisy_class = np.array(prob_noisy_class)
    idx_view_labeled = np.array(idx_view_labeled)
    
    num_samples = len(prob_true_class)
    all_indices = np.arange(num_samples)
    idx_unlabeled = np.setdiff1d(all_indices, idx_view_labeled)


    # Scatter plot for data vs. preds
    plt.scatter(prob_true_class[idx_unlabeled], prob_noisy_class[idx_unlabeled], color='red', alpha=0.5, s=3, label=f'unlabeled - {len(idx_unlabeled)}')
    plt.scatter(prob_true_class[idx_view_labeled], prob_noisy_class[idx_view_labeled], color='blue', alpha=0.5, s=3, label=f'labeled - {len(idx_view_labeled)}')

    # Labels and legend
    plt.xlabel('True Class')
    plt.ylabel('Noisy Class')
    plt.legend(loc='upper right')
    plt.grid(True)
    # Save figure
    plt.savefig(f'{metric_path}/{metric}_probs_epoch{epoch:03d}.png')
    plt.clf()

    
    
    idx_error = np.setdiff1d(all_indices, idx_correct)
    # Scatter plot for data vs. preds
    plt.scatter(prob_best_class[idx_correct], prob_noisy_class[idx_correct], color='red', alpha=0.5, s=3, label=f'correct - {len(idx_correct)}')
    plt.scatter(prob_best_class[idx_error], prob_noisy_class[idx_error], color='blue', alpha=0.5, s=3, label=f'error - {len(idx_error)}')

    # Labels and legend
    plt.xlabel('True Class')
    plt.ylabel('Noisy Class')
    plt.legend(loc='upper right')
    plt.grid(True)
    # Save figure
    plt.savefig(f'{metric_path}/{metric}_probs_correct_epoch{epoch:03d}.png')
    plt.clf()






    # Scatter plot for data vs. preds
    plt.scatter(max_evi_excl_noisy[idx_correct], evi_noisy_class[idx_correct], color='red', alpha=0.5, s=3, label=f'correct - {len(idx_correct)}')
    plt.scatter(max_evi_excl_noisy[idx_error], evi_noisy_class[idx_error], color='blue', alpha=0.5, s=3, label=f'error - {len(idx_error)}')

    # Labels and legend
    plt.xlabel('Evidence Best Class')
    plt.ylabel('Evidence Noisy Class')
    plt.legend(loc='upper right')
    plt.grid(True)
    # Save figure
    plt.savefig(f'{metric_path}/{metric}_evidence_correct_epoch{epoch:03d}.png')
    plt.clf()


    idx_noisy = np.setdiff1d(all_indices, idx_clean)

    # Scatter plot for data vs. preds
    plt.scatter(max_evi_excl_noisy[idx_clean], evi_noisy_class[idx_clean], color='red', alpha=0.5, s=3, label=f'clean - {len(idx_clean)}')
    plt.scatter(max_evi_excl_noisy[idx_noisy], evi_noisy_class[idx_noisy], color='blue', alpha=0.5, s=3, label=f'noisy - {len(idx_noisy)}')

    # Labels and legend
    plt.xlabel('Evidence Best Class')
    plt.ylabel('Evidence Noisy Class')
    plt.legend(loc='upper right')
    plt.grid(True)
    # Save figure
    plt.savefig(f'{metric_path}/{metric}_evidence_clean_epoch{epoch:03d}.png')
    plt.clf()