import torch


def PCA(X, latent_dim):
    # Step 1: center the data
    X_centered = X - X.mean(dim=0, keepdim=True)

    # Step 2: PCA
    U, S, V = torch.pca_lowrank(X_centered, q=latent_dim)

    # Step 3: obtain projected data
    Z = X_centered @ V[:, :latent_dim]


    # Step 4: report compressibility (shared formula with Autoencoder)
    X_reconstructed = Z @ V[:, :latent_dim].T + X.mean(dim=0, keepdim=True)
    from compression.autoencoder import Autoencoder
    c_mean, c_frob = Autoencoder._compressibility_metrics(X_reconstructed, X)
    print('compressibility (mean)=', c_mean, '  (frob)=', c_frob)
    return Z