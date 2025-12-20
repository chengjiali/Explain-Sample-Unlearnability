from trainer.unlearn.base import UnlearnTrainer

import torch
import torch.nn.functional as F


def cross_entropy_unlearning_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Implementation of Cross Entropy Unlearning Loss (CE-U).

    This function creates a modified target distribution by setting the logit corresponding to the true label to negative infinity, effectively forcing the model to assign zero probability to the correct answer. The loss then minimizes the KL divergence between this target distribution and the model's output.

    Args:
      logits: Model output logits with shape [batch_size, sequence_length, vocabulary_size]
      labels: Ground truth token indices with shape [batch_size, sequence_length]
      ignore_index: Token indices to ignore in the loss calculation (typically padding)

    Returns:
      A scalar tensor representing the mean unlearning loss across valid positions
    """
    batch_size, sequence_length, vocabulary_size = logits.shape
    # Extract valid logits and labels based on ignore_index.
    if ignore_index is not None:
        # Shape: [batch_size, sequence_length], boolean mask
        valid_mask = labels != ignore_index
        # Shape: [num_valid_positions, vocabulary_size]
        valid_logits = logits[valid_mask]
        # Shape: [num_valid_positions]
        valid_labels = labels[valid_mask]
    else:
        # Shape: [batch_size*sequence_length, vocabulary_size]
        valid_logits = logits.view(-1, vocabulary_size)
        # Shape: [batch_size*sequence_length]
        valid_labels = labels.view(-1)

    # Create a copy of valid_logits to generate the target distribution
    # Shape: [num_valid_positions, vocabulary_size]
    valid_target_logits = valid_logits.detach().clone()

    # Suppress the logits corresponding to the true token by setting them to -inf.
    # This ensures that the probability for the true token is effectively zero after softmax.
    valid_target_logits.scatter_(
        dim=-1,
        index=valid_labels.unsqueeze(-1),  # Shape: [num_valid_positions, 1]
        value=float("-inf"),
    )  # Result shape: [num_valid_positions, vocabulary_size]

    # Apply softmax to generate the target probability distribution
    # Shape: [num_valid_positions, vocabulary_size]
    valid_target_probabilities = F.softmax(valid_target_logits, dim=-1)

    # Compute the cross entropy loss between input logits and target probabilities
    # The loss is averaged over the valid positions and returns a scalar tensor
    return F.cross_entropy(
        input=valid_logits,
        target=valid_target_probabilities,
        reduction='none'
    )

def compute_batch_ceu_per_token(model, inputs, ignore_first_n_answer_tokens=1, return_shifted_labels=False):
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]

    # Implement the trick to ignore the first n answer tokens mentioned in the footnote in the Training Settings section of arXiv:2503.01224
    valid_mask = labels != -100
    update_mask = (
        valid_mask.cumsum(dim=-1) <= ignore_first_n_answer_tokens
    ) & valid_mask
    labels_without_first_n_answer_tokens = labels.masked_fill(update_mask, -100)

    shifted_labels = labels_without_first_n_answer_tokens[..., 1:].contiguous()
    shifted_logits = logits[..., :-1, :].contiguous()
    loss_per_token = cross_entropy_unlearning_loss(
        shifted_logits, shifted_labels, ignore_index=-100
    )
    if return_shifted_labels:
        return loss_per_token, outputs, shifted_labels
    else:
        return loss_per_token, outputs

def compute_batch_ceu(model, inputs, ignore_first_n_answer_tokens=1):
    loss, outputs = compute_batch_ceu_per_token(model, inputs, ignore_first_n_answer_tokens)
    return loss.mean(), outputs

def compute_batch_ceu_per_sample(model, inputs, ignore_first_n_answer_tokens=1):
    loss_per_token, outputs, shifted_labels = compute_batch_ceu_per_token(model, inputs, ignore_first_n_answer_tokens, return_shifted_labels=True)
    loss_per_sample = _token2sample_loss(loss_per_token, shifted_labels != -100)
    return loss_per_sample, outputs

def _token2sample_loss(
    loss_per_token: torch.Tensor,   # [num_valid_tokens]
    valid_mask: torch.Tensor,       # [batch_size, seq_len]  bool
):
    batch_size, seq_len = valid_mask.shape
    # 1. check which sample each token belongs to
    #    flatten (0,0)…(0,L-1),(1,0)… and align with loss_per_token
    sample_id_full = torch.arange(batch_size, device=loss_per_token.device).repeat_interleave(seq_len)
    sample_id_valid = sample_id_full[valid_mask.flatten()]         # [num_valid_tokens]

    # 2. scatter_add / bincount to accumulate loss_per_token to each sample
    per_sample_loss = torch.zeros(batch_size, device=loss_per_token.device)
    per_sample_loss.scatter_add_(0, sample_id_valid, loss_per_token)

    valid_counts = torch.zeros(batch_size, device=loss_per_token.device)
    valid_counts.scatter_add_(0, sample_id_valid,
                                torch.ones_like(loss_per_token))
    per_sample_loss = per_sample_loss / valid_counts.clamp(min=1)

    return per_sample_loss


class CEU(UnlearnTrainer):
    def __init__(self, ignore_first_n_answer_tokens=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_first_n_answer_tokens = ignore_first_n_answer_tokens

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        loss, outputs = compute_batch_ceu(
            model,
            forget_inputs,
            ignore_first_n_answer_tokens=self.ignore_first_n_answer_tokens,
        )
        return (loss, outputs) if return_outputs else loss
    
    def compute_loss_per_token_superloss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        loss_per_token, outputs = compute_batch_ceu_per_token(
            model,
            forget_inputs,
            ignore_first_n_answer_tokens=self.ignore_first_n_answer_tokens,
        )
        loss = self.calculate_superloss(loss_per_token)
        return (loss, outputs) if return_outputs else loss
    
    def compute_loss_per_sample_superloss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        loss_per_sample, outputs = compute_batch_ceu_per_sample(
            model,
            forget_inputs,
            ignore_first_n_answer_tokens=self.ignore_first_n_answer_tokens,
        )
        loss = self.calculate_superloss(loss_per_sample)
        return (loss, outputs) if return_outputs else loss
