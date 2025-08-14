"""
Utility functions for Echo State Network data preprocessing and manipulation.

Author: Stefano Nardo
"""
import torch


def prepare_target(target, seq_lengths, washout, batch_first=False):
    """Preprocess target sequences for offline ESN training methods.
    
    This function transforms batched target sequences into the flattened format
    required by offline training algorithms (SVD, Cholesky, matrix inversion).
    It removes washout timesteps and concatenates all valid target values into
    a single tensor, which is essential for setting up the ridge regression
    problem: X @ W = Y.
    
    Args:
        target (Tensor): Target sequences of shape:
            - (seq_len, batch, output_size) if batch_first=False
            - (batch, seq_len, output_size) if batch_first=True
        seq_lengths (list of int): Actual length of each sequence in the batch
            (before washout removal). Length must equal batch size.
        washout (list of int): Number of initial timesteps to remove per sequence.
            Must have same length as seq_lengths. These initial steps are ignored
            during training to allow reservoir dynamics to stabilize.
        batch_first (bool, optional): Input tensor format. Default: False
            
    Returns:
        Tensor: Flattened target tensor of shape (total_train_samples, output_size)
            where total_train_samples = sum(seq_lengths[i] - washout[i] for all i).
            Sequences are concatenated in batch order.
            
    Example:
        >>> # Prepare targets for 2 sequences with different lengths and washouts
        >>> target = torch.randn(20, 2, 3)  # max_len=20, batch=2, features=3
        >>> seq_lengths = [15, 18]          # actual sequence lengths
        >>> washout = [5, 3]                # washout per sequence
        >>> flat_target = prepare_target(target, seq_lengths, washout)
        >>> print(flat_target.shape)        # torch.Size([25, 3]) = (10+15, 3)
        
    Note:
        This function is typically used before calling model.fit() with offline
        training methods. The resulting tensor corresponds to the Y matrix in
        the ridge regression formulation: (X^T X + λI) W = X^T Y.
    """

    # Ensure time-first format for processing
    if batch_first:
        target = target.transpose(0, 1)
        
    # Extract dimensions
    n_sequences = target.size(1)
    target_dim = target.size(2)
    
    # Calculate total number of training samples after washout removal
    train_len = sum(torch.tensor(seq_lengths) - torch.tensor(washout)).item()

    # Allocate output tensor
    new_target = torch.zeros(train_len, target_dim, device=target.device)

    # Concatenate all valid (post-washout) target values
    idx = 0
    for s in range(n_sequences):
        batch_len = seq_lengths[s] - washout[s]
        # Extract valid timesteps: from washout[s] to seq_lengths[s]
        new_target[idx:idx + batch_len, :] = target[washout[s]:seq_lengths[s], s, :]
        idx += batch_len

    return new_target


def washout_tensor(tensor, washout, seq_lengths, bidirectional=False, batch_first=False):
    """Remove washout timesteps from tensor sequences and adjust sequence lengths.
    
    The washout period allows reservoir dynamics to stabilize by discarding initial
    timesteps where the internal state may still be transitioning. This function
    efficiently removes these timesteps while preserving the tensor structure
    and updating sequence length information.
    
    This operation is essential for ESN training because:
    1. Initial reservoir states are typically zero or random
    2. It takes time for input-driven dynamics to develop meaningful patterns
    3. Using unstabilized states can hurt learning performance
    4. The washout period acts as a "burn-in" for the reservoir
    
    Args:
        tensor (Tensor): Input sequences of shape:
            - (seq_len, batch, features) if batch_first=False  
            - (batch, seq_len, features) if batch_first=True
        washout (list of int): Number of timesteps to remove per sequence.
            Must have length equal to batch size. Can be different per sequence.
        seq_lengths (list of int or Tensor): Current length of each sequence.
            Will be modified in-place to reflect new lengths after washout.
        bidirectional (bool, optional): If True, removes washout from both ends
            of sequences (for bidirectional RNNs). Default: False
        batch_first (bool, optional): Input tensor format. Default: False
            
    Returns:
        tuple: (processed_tensor, updated_seq_lengths) where:
            - processed_tensor: Tensor with washout timesteps removed, trimmed
              to maximum remaining sequence length
            - updated_seq_lengths: Modified sequence lengths after washout removal
              
    Example:
        >>> # Remove 3 timesteps from start of each sequence  
        >>> x = torch.randn(10, 2, 5)  # 10 timesteps, 2 sequences, 5 features
        >>> seq_lens = [8, 10]         # actual lengths before padding
        >>> washout_lens = [3, 3]      # remove 3 timesteps from each
        >>> x_clean, new_lens = washout_tensor(x, washout_lens, seq_lens)
        >>> print(new_lens)            # [5, 7] = [8-3, 10-3]
        >>> print(x_clean.shape)       # torch.Size([7, 2, 5]) = max(new_lens)
        
    Note:
        This function modifies seq_lengths in-place and creates a new tensor
        for the processed sequences. The output tensor is trimmed to the maximum
        remaining sequence length to avoid unnecessary padding.
    """
    # Ensure time-first format and create working copies
    tensor = tensor.transpose(0, 1) if batch_first else tensor.clone()
    
    # Create working copies of seq_lengths to avoid modifying input
    if type(seq_lengths) == list:
        seq_lengths = seq_lengths.copy()
    if type(seq_lengths) == torch.Tensor:
        seq_lengths = seq_lengths.clone()

    # Process each sequence in the batch
    for b in range(tensor.size(1)):
        if washout[b] > 0:
            # Extract the valid portion (after washout) of the sequence
            tmp = tensor[washout[b]:seq_lengths[b], b].clone()
            
            # Shift valid data to the beginning of the sequence
            tensor[:seq_lengths[b] - washout[b], b] = tmp
            
            # Zero out the tail (where data was moved from)
            tensor[seq_lengths[b] - washout[b]:, b] = 0
            
            # Update sequence length
            seq_lengths[b] -= washout[b]

            # For bidirectional RNNs, remove washout from the end too
            if bidirectional:
                tensor[seq_lengths[b] - washout[b]:, b] = 0
                seq_lengths[b] -= washout[b]

    # Trim tensor to maximum remaining sequence length to save memory
    if type(seq_lengths) == list:
        max_len = max(seq_lengths)
    else:
        max_len = max(seq_lengths).item()

    return tensor[:max_len], seq_lengths
