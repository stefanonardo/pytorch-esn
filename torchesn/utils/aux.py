import torch


def prepare_target(target, seq_lengths, washout, batch_first=False):
    """ Preprocess target for offline training.

    Args:
        target (seq_len, batch, output_size): tensor containing
            the features of the target sequence.
        seq_lengths: list of lengths of each sequence in the batch.
        washout: number of initial timesteps during which output of the
            reservoir is not forwarded to the readout.
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
    train_len = int(sum(seq_lengths) - washout * n_sequences)

    new_target = torch.zeros(train_len, target_dim)

    idx = 0
    for s in range(n_sequences):
        batch_len = seq_lengths[s] - washout
        new_target[idx:idx + batch_len, :] = target[washout:seq_lengths[s], s,
                                             :]
        idx += batch_len

    return new_target
