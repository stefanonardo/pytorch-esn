import torch


def prepare_target(target, seq_lengths, washout, batch_first=False):
    """ Preprocess target for offline training.

    Args:
        target (seq_len, batch, output_size): tensor containing
            the features of the target sequence.
        seq_lengths: list of lengths of each sequence in the batch.
        washout: number of initial timesteps during which output of the
            reservoir is not forwarded to the readout. One value per sample.
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``

    Returns:
        tensor containing the features of the batch's sequences rolled out along
        one axis, minus the washouts and the padded values.
    """

    if batch_first:
        target = target.transpose(0, 1)
    n_sequences = target.size(1)
    target_dim = target.size(2)
    train_len = sum(torch.tensor(seq_lengths) - torch.tensor(washout)).item()

    new_target = torch.zeros(train_len, target_dim, device=target.device)

    idx = 0
    for s in range(n_sequences):
        batch_len = seq_lengths[s] - washout[s]
        new_target[idx:idx + batch_len, :] = target[washout[s]:seq_lengths[s], s, :]
        idx += batch_len

    return new_target


def washout_tensor(tensor, washout, seq_lengths, bidirectional=False, batch_first=False):
    tensor = tensor.transpose(0, 1) if batch_first else tensor.clone()
    if type(seq_lengths) == list:
        seq_lengths = seq_lengths.copy()
    if type(seq_lengths) == torch.Tensor:
        seq_lengths = seq_lengths.clone()

    for b in range(tensor.size(1)):
        if washout[b] > 0:
            tmp = tensor[washout[b]:seq_lengths[b], b]
            tensor[:seq_lengths[b] - washout[b], b] = tmp
            tensor[seq_lengths[b] - washout[b]:, b] = 0
            seq_lengths[b] -= washout[b]

            if bidirectional:
                tensor[seq_lengths[b] - washout[b]:, b] = 0
                seq_lengths[b] -= washout[b]

    if type(seq_lengths) == list:
        max_len = max(seq_lengths)
    else:
        max_len = max(seq_lengths).item()

    return tensor[:max_len], seq_lengths
