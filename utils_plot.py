import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import torch
import os


def compute_histogram_bins(data, desired_bin_size):
    min_val = np.min(data)
    max_val = np.max(data)
    min_boundary = -1.0 * (min_val % desired_bin_size - min_val)
    max_boundary = max_val - max_val % desired_bin_size + desired_bin_size
    n_bins = int((max_boundary - min_boundary) / desired_bin_size) + 1
    bins = np.linspace(min_boundary, max_boundary, n_bins)
    return bins


def plot_histogram_pred(data, entropy, inds_clean, inds_noisy, path, epoch):
    data = data[-1].numpy()
    entropy = entropy[-1].numpy()
    bins = compute_histogram_bins(data, 0.01)
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

    
def plot_curve_loss(data_hist, inds_clean, inds_noisy, path, epoch):
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
    # Ensure each item in data_hist is a tensor before conversion
    all_loss_per_epoch = torch.tensor(data_hist[0]) if isinstance(data_hist[0], list) else data_hist[0]
    clean_loss_per_epoch = torch.tensor(data_hist[1]) if isinstance(data_hist[1], list) else data_hist[1]
    noisy_loss_per_epoch = torch.tensor(data_hist[2]) if isinstance(data_hist[2], list) else data_hist[2]

    # Convert to NumPy arrays, moving to CPU first if needed
    all_loss_per_epoch = all_loss_per_epoch.cpu().numpy() if isinstance(all_loss_per_epoch, torch.Tensor) else np.array(all_loss_per_epoch)
    clean_loss_per_epoch = clean_loss_per_epoch.cpu().numpy() if isinstance(clean_loss_per_epoch, torch.Tensor) else np.array(clean_loss_per_epoch)
    noisy_loss_per_epoch = noisy_loss_per_epoch.cpu().numpy() if isinstance(noisy_loss_per_epoch, torch.Tensor) else np.array(noisy_loss_per_epoch)

    # Ensure the data is correctly plotted, with length determined by the losses
    num_epochs = len(all_loss_per_epoch)  # Assuming all losses have the same length
    
    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), all_loss_per_epoch, linestyle='-', color='red', label='All Loss')
    plt.plot(range(1, num_epochs + 1), clean_loss_per_epoch, linestyle='--', color='green', label='Clean Loss')
    plt.plot(range(1, num_epochs + 1), noisy_loss_per_epoch, linestyle='-.', color='blue', label='Noisy Loss')

    # Adding labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Across Epochs')
    plt.legend()

    # Save the plot to the specified path
    plt.savefig(f'{path}/SSLoss_curve.png')
    plt.clf()  # Clear the figure to avoid overlapping with other plots


def plot_curve_accuracy(data_hist, inds_clean, inds_noisy, path, epoch):
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
    bins = compute_histogram_bins(data, 0.01)

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



def plot_histogram_metric(data_hist, inds_clean, inds_correct, inds_labeled, thresholds, path, epoch, metric="Uncertainty"):
    data = data_hist[-1].numpy()
    total_samples = len(data)

    # Create metric-specific folder
    metric_path = os.path.join(path, metric)
    os.makedirs(metric_path, exist_ok=True)
    inds_unlabeled = list(set(range(total_samples)) - set(inds_labeled))
    inds_noisy = list(set(range(total_samples)) - set(inds_clean))
    inds_error = list(set(range(total_samples)) - set(inds_correct))

    bins = compute_histogram_bins(data, 0.01)

    # 3. Plot histogram labeled vs unlabeled
    num_inds_labeled = len(inds_labeled)
    num_inds_unlabeled = len(inds_unlabeled)
    perc_labeled = 100 * num_inds_labeled / float(num_inds_labeled + num_inds_unlabeled)

    plt.hist(data[inds_labeled], bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5,
             label=f'Labeled - {num_inds_labeled} ({perc_labeled:.1f}%)')
    if len(inds_unlabeled) > 0:
        plt.hist(data[inds_unlabeled], bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5,
                 label=f'Unlabeled - {num_inds_unlabeled} ({100 - perc_labeled:.1f}%)')

    plt.xlabel(metric)
    plt.ylabel('Number of Data')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig(f'{metric_path}/{metric}_labeled_epoch{epoch:03d}.png')
    plt.clf()


    # 1. Plot histogram clean vs noisy
    num_inds_clean = len(inds_clean)
    num_inds_noisy = len(inds_noisy)
    perc_clean = 100 * num_inds_clean / float(num_inds_clean + num_inds_noisy)

    plt.hist(data[inds_clean], bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5,
             label=f'Clean - {num_inds_clean} ({perc_clean:.1f}%)')
    if len(inds_noisy) > 0:
        plt.hist(data[inds_noisy], bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5,
                 label=f'Noisy - {num_inds_noisy} ({100 - perc_clean:.1f}%)')
    for ind in thresholds:
        v = data[ind]
        plt.axvline(x=v, color='red', linestyle='dashed', linewidth=2)
    plt.xlabel(metric)
    plt.ylabel('Number of Data')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig(f'{metric_path}/{metric}_clean_epoch{epoch:03d}.png')
    plt.clf()

    # 2. Plot histogram correct vs incorrect
    num_inds_correct = len(inds_correct)
    num_inds_error = len(inds_error)
    perc_correct = 100 * num_inds_correct / float(num_inds_correct + num_inds_error)

    plt.hist(data[inds_correct], bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5,
             label=f'Correct - {num_inds_correct} ({perc_correct:.1f}%)')
    if len(inds_error) > 0:
        plt.hist(data[inds_error], bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5,
                 label=f'Error - {num_inds_error} ({100 - perc_correct:.1f}%)')
        
    for ind in thresholds:
        v = data[ind]
        plt.axvline(x=v, color='red', linestyle='dashed', linewidth=2)
    plt.xlabel(metric)
    plt.ylabel('Number of Data')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig(f'{metric_path}/{metric}_correct_epoch{epoch:03d}.png')
    plt.clf()




def plot_histogram_metric2(data_hist, inds_clean, inds_correct, inds_labeled, thresholds, path, epoch, metric="True_margin"):
    data = data_hist[-1].numpy()
    total_samples = len(data)

    # Create metric-specific folder
    metric_path = os.path.join(path, metric)
    os.makedirs(metric_path, exist_ok=True)
    if isinstance(inds_labeled, torch.Tensor):
        inds_labeled = inds_labeled.numpy()

    inds_error = list(set(range(total_samples)) - set(inds_correct))
    inds_noisy = list(set(range(total_samples)) - set(inds_clean))
    inds_unlabeled = list(set(range(total_samples)) - set(inds_labeled))

    bins = compute_histogram_bins(data, 0.01)
    # Filter only labeled indices
    inds_clean_labeled = list(set(inds_clean) & set(inds_labeled))
    inds_noisy_labeled = list(set(inds_noisy) & set(inds_labeled))

    num_inds_clean_labeled = len(inds_clean_labeled)
    num_inds_noisy_labeled = len(inds_noisy_labeled)
    perc_clean_labeled = 100 * num_inds_clean_labeled / float(num_inds_clean_labeled + num_inds_noisy_labeled)

    # Plot histogram for labeled clean vs. error
    plt.hist(data[inds_clean_labeled], bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5,
            label=f'clean - {num_inds_clean_labeled} ({perc_clean_labeled:.1f}%)')
    if len(inds_noisy_labeled) > 0:
        plt.hist(data[inds_noisy_labeled], bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5,
                label=f'Noisy - {num_inds_noisy_labeled} ({100 - perc_clean_labeled:.1f}%)')

    plt.xlabel(metric)
    plt.ylabel('Number of Data')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig(f'{metric_path}/{metric}_clean_labeled_epoch{epoch:03d}.png')
    plt.clf()

 
    #  Plot whole histogram
    plt.hist(data[inds_labeled], bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5,
            label=f'all labeled')
    for ind in thresholds:
        v = data[ind]
        plt.axvline(x=v, color='red', linestyle='dashed', linewidth=2)
    plt.xlabel(metric)
    plt.ylabel('Number of Data')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig(f'{metric_path}/{metric}_all_labeled_epoch{epoch:03d}.png')
    plt.clf()


    
    # Filter only unlabeled indices
    inds_correct_unlabeled = list(set(inds_correct) & set(inds_unlabeled))
    inds_error_unlabeled = list(set(inds_error) & set(inds_unlabeled))

    num_inds_correct_unlabeled = len(inds_correct_unlabeled)
    num_inds_error_unlabeled = len(inds_error_unlabeled)
    perc_correct_unlabeled = 100 * num_inds_correct_unlabeled / float(num_inds_correct_unlabeled + num_inds_error_unlabeled)

    # Plot histogram for unlabeled correct vs. error
    plt.hist(data[inds_correct_unlabeled], bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5,
            label=f'correct - {num_inds_correct_unlabeled} ({perc_correct_unlabeled:.1f}%)')

    if len(inds_error_unlabeled) > 0:
        plt.hist(data[inds_error_unlabeled], bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5,
                label=f'error - {num_inds_error_unlabeled} ({100 - perc_correct_unlabeled:.1f}%)')
    for ind in thresholds:
        v = data[ind]
        plt.axvline(x=v, color='red', linestyle='dashed', linewidth=2)
    plt.xlabel(metric)
    plt.ylabel('Number of Data')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig(f'{metric_path}/{metric}_correct_unlabeled_epoch{epoch:03d}.png')
    plt.clf()

    
    
    # Plot whole histogram
    plt.hist(data[inds_unlabeled], bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5,
            label=f'all unlabeled')
    for ind in thresholds:
        v = data[ind]
        plt.axvline(x=v, color='red', linestyle='dashed', linewidth=2)
    plt.xlabel(metric)
    plt.ylabel('Number of Data')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig(f'{metric_path}/{metric}_all_unlabeled_epoch{epoch:03d}.png')
    plt.clf()
