import torch


def ridge_regression(X, y, lambda_reg):
    """ Scikit-Learn SVD solver for ridge regression."""
    U, s, V = torch.svd(X)
    idx = s > 1e-15  # same default value as scipy.linalg.pinv
    s_nnz = s[idx][:, None]
    UTy = torch.mm(U.t(), y.data)
    d = torch.zeros(s.size(0), 1)
    d[idx] = s_nnz / (s_nnz ** 2 + lambda_reg)
    d_UT_y = d * UTy
    return torch.mm(V, d_UT_y)

