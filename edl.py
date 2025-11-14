import torch
import torch.nn.functional as F
import numpy as np


#https://github.com/dougbrion/pytorch-classification-uncertainty

def compute_metrics(vacuity, eval_preds, num_class, noisy_labels, clean_labels):
    #recover
    S = num_class/vacuity
    #print("Shape of S:", S.shape)
    #print("Shape of eval_preds:", eval_preds.shape)
    alpha = S.unsqueeze(1) * eval_preds
    #alpha = S*eval_preds
    evidence = alpha - 0.1

    # Compute vacuity (total uncertainty)
    vacuity = num_class / S

    # Compute belief mass
    belief_mass = evidence / S.unsqueeze(1)  # b_i = alpha_i / sum(alpha)
    prob_mass = alpha / S.unsqueeze(1)


    # Compute entropy
    entropy = -torch.sum(
        prob_mass * (torch.digamma(alpha + 1) - torch.digamma(S.unsqueeze(1) + 1)),
        dim=1
    )

    # dissonance
    dissonance = torch.zeros_like(entropy)  # Initialize with zeros
    for c in range(num_class):
        bc = belief_mass[:, c]  # b_c (belief for class c)
        other_b = torch.cat((belief_mass[:, :c], belief_mass[:, c+1:]), dim=1)  # b_i (other classes)
        balance = 1 - torch.abs(other_b - bc.unsqueeze(1)) / (other_b + bc.unsqueeze(1) + 1e-10)
        balance[other_b == 0] = 0  # If b_i or b_c is zero, balance is 0
        
        # Sum over all other classes
        dissonance += bc * torch.sum(other_b * balance, dim=1) / (torch.sum(other_b, dim=1) + 1e-10)

    # dissonance2
    dissonance2 = torch.zeros_like(entropy)  # Initialize with zeros
    for c in range(num_class):
        bc = prob_mass[:, c]  # b_c (belief for class c)
        other_b = torch.cat((prob_mass[:, :c], prob_mass[:, c+1:]), dim=1)  # b_i (other classes)
        balance = 1 - torch.abs(other_b - bc.unsqueeze(1)) / (other_b + bc.unsqueeze(1) + 1e-10)
        balance[other_b == 0] = 0  # If b_i or b_c is zero, balance is 0
        
        # Sum over all other classes
        dissonance2 += bc * torch.sum(other_b * balance, dim=1) / (torch.sum(other_b, dim=1) + 1e-10)
    #margin 2 top values
    top2_vals, _ = torch.topk(evidence, k=2, dim=1)
    margin = top2_vals[:, 0] - top2_vals[:, 1]
    
    
    #mutual info
    log_expected_probs = torch.log(prob_mass + 1e-10)
    entropy_of_expected = -torch.sum(prob_mass * log_expected_probs, dim=1)
    mutual_information = entropy_of_expected - entropy

    margin = (margin - margin.min())/(margin.max()- margin.min())
    vacuity = (vacuity - vacuity.min())/(vacuity.max() - vacuity.min())
    dissonance = (dissonance - dissonance.min())/(dissonance.max() - dissonance.min())
    entropy = (entropy - entropy.min())/(entropy.max() - entropy.min())
    mutual_information = (mutual_information - mutual_information.min())/(mutual_information.max() - mutual_information.min())

    prob_noisy_class = prob_mass[np.arange(len(noisy_labels)), noisy_labels]
    prob_true_class = prob_mass[np.arange(len(clean_labels)), clean_labels]
    prob_mass_excl_noisy = prob_mass.clone()
    prob_mass_excl_noisy[torch.arange(len(noisy_labels)), noisy_labels] = -1
    max_prob_excl_noisy = prob_mass_excl_noisy.max(dim=1).values

    evi_noisy_class = evidence[np.arange(len(noisy_labels)), noisy_labels]
    evi_true_class = evidence[np.arange(len(clean_labels)), clean_labels]
    evidence_excl_noisy = evidence.clone()
    evidence_excl_noisy[torch.arange(len(noisy_labels)), noisy_labels] = -1
    max_evi_excl_noisy, predicted_class_2 = evidence_excl_noisy.max(dim=1)


    return vacuity, dissonance, dissonance2, entropy, mutual_information, margin, prob_noisy_class, prob_true_class, max_prob_excl_noisy, evi_noisy_class, evi_true_class, max_evi_excl_noisy, predicted_class_2

  



def compute_dirichlet_metrics(outputs, num_class, activation, evidence_factor=1):
    evidence = activation(outputs) 
    alpha = evidence + evidence_factor
    alpha_0 = torch.sum(alpha, dim=1, keepdim=True)  # Dirichlet strength (S)

    # Compute vacuity (total uncertainty)
    vacuity = num_class / alpha_0

    # Compute belief mass
    belief_mass = evidence / alpha_0  # b_i = alpha_i / sum(alpha)
    prob_mass = alpha / alpha_0

    # Compute entropy of belief mass
    entropy = -torch.sum(
        prob_mass * (torch.digamma(alpha + 1) - torch.digamma(alpha_0 + 1)),
        dim=1
    )

    # Compute dissonance properly
    dissonance = torch.zeros_like(entropy)  # Initialize with zeros
    for c in range(num_class):
        bc = belief_mass[:, c]  # b_c (belief for class c)
        other_b = torch.cat((belief_mass[:, :c], belief_mass[:, c+1:]), dim=1)  # b_i (other classes)
        balance = 1 - torch.abs(other_b - bc.unsqueeze(1)) / (other_b + bc.unsqueeze(1) + 1e-10)
        balance[other_b == 0] = 0  # If b_i or b_c is zero, balance is 0
        
        # Sum over all other classes
        dissonance += bc * torch.sum(other_b * balance, dim=1) / (torch.sum(other_b, dim=1) + 1e-10)


    top2_vals, _ = torch.topk(outputs, k=2, dim=1)
    margin = top2_vals[:, 0] - top2_vals[:, 1]

    #mutual info
    # Entropy of expected class probabilities (total uncertainty)
    log_expected_probs = torch.log(prob_mass + 1e-10)
    entropy_of_expected = -torch.sum(prob_mass * log_expected_probs, dim=1)
    # Mutual Information = Entropy of expected - Expected entropy
    mutual_information = entropy_of_expected - entropy

    return vacuity.squeeze(1), entropy, dissonance, margin, mutual_information


def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device

def one_hot_embedding(labels, num_classes=10):
    y = torch.eye(num_classes, device=labels.device)
    return y[labels]

def relu_evidence(y):
    return F.relu(y)

def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))

def softplus_evidence(y):
    return F.softplus(y)

def kl_divergence(alpha, num_classes, device=None):
    if not device:
        device = get_device()
    num_classes = alpha.size(-1)
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl

def loglikelihood_loss(y, alpha, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, evidence_factor, device=None):
    y = y.to(device)
    alpha = alpha.to(device)

    S = torch.sum(alpha, dim=1, keepdim=True)
    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    if annealing_step == "None":
        annealing_coef = 0
    elif annealing_step == "progression":
        annealing_coef = torch.min(
            torch.tensor(1, dtype=torch.float32),
            torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
        )
    else:
        annealing_coef = annealing_step

    #L_variance = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    L_variance = 0

    kl_alpha = (alpha - evidence_factor) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)

    return A + L_variance + kl_div



def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, activation=softplus_evidence, evidence_factor = 1, device=None):
    if not device:
        device = get_device()
    evidence = activation(output)
    alpha = evidence + evidence_factor

    loss = mse_loss(
            torch.log, target, alpha, epoch_num, num_classes, annealing_step, evidence_factor, device
        )

    return torch.mean(loss), loss.squeeze(-1)



def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, activation=softplus_evidence, evidence_factor = 1, device=None):
    if not device:
        device = get_device()
    evidence = activation(output)
    alpha = evidence + evidence_factor

    loss = edl_loss(
            torch.log, target, alpha, epoch_num, num_classes, annealing_step, evidence_factor, device
        )

    return torch.mean(loss), loss.squeeze(-1)

def m_edl_log_loss(output, target, epoch_num, num_classes, annealing_step, activation=softplus_evidence, evidence_factor=1, device=None):
    if not device:
        device = get_device()
    
    target = target.to(device)  # ✅ Ensure target is on the right device
    
    evidence = activation(output)  # Convert output into evidence
    alpha = evidence + evidence_factor  # Compute Dirichlet parameters
    alpha_e = alpha[:, :num_classes]  # Extract the first `num_classes` columns
    alpha_u = torch.full((alpha.shape[0], 1), 10 + evidence_factor, dtype=torch.float32, device=device)
    alpha_p = torch.cat([alpha_e, alpha_u], dim=1)

    if target.shape[1] == alpha_p.shape[1]:
        target_p = target  # Use target as-is
    else:
        batch_size = target.shape[0]
        target_u = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
        target_p = torch.cat([target, target_u], dim=1)

    loss = edl_loss(torch.log, target_p, alpha_p, epoch_num, num_classes, annealing_step, evidence_factor, device)

    return torch.mean(loss), loss.squeeze(-1)



def edl_digamma_loss(
    output, target, epoch_num, num_classes, annealing_step, device=None
):
    if not device:
        device = get_device()
    evidence = softplus_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss, edl_loss(
            torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device
        ).squeeze(-1)


def mse_loss(
    output, target, epoch_num, num_classes, annealing_step, device=None
):
    if not device:
        device = get_device()
    evidence = softplus_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss, edl_loss(
            torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device
        ).squeeze(-1)

