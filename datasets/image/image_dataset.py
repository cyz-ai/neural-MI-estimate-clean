import numpy as np
import torch
import matplotlib.pyplot as plt



from .utils.dependent_norm import multivariate_normal_from_MI
from .utils.synthetic import *





min_delta = 2
img_width = 16
img_height = 16








'''
    Gaussian plots
''' 

def gaussian_mapping(X):
    """ Map Gaussian vector to the coordinates of the mode (center of the plot). """
    return normal_to_uniform(X)

def gaussian_2_gaussianImg(X):
    """ Map Gaussian vector to an image. """
    distribution = lambda X, Y, params : np.exp(-10.0 * ((X - params[:,0,None,None])**2 + (Y - params[:,1,None,None])**2))
    # distribution = lambda X, Y, params : np.exp(
    # -10.0 * (
    #     (1.0 + params[:,2,None,None])**2 * (X - params[:,0,None,None])**2 +
    #     (1.0 + params[:,3,None,None])**2 * (Y - params[:,1,None,None])**2 +
    #     (-1.0 + 2.0 * params[:,4,None,None])*(1.0 + params[:,2,None,None])*(1.0 + params[:,3,None,None]) * 
    #     (X - params[:,0,None,None])*(Y - params[:,1,None,None])
    # )
    # )
    return params_to_2d_distribution(X, distribution, img_width, img_height)




def generate_gaussian_plot(mi=3, n_samples=10000, device='cuda:0'):
    X_dimension = 2
    Y_dimension = 2
    latent_dimension = 2
    
    
    random_variable = multivariate_normal_from_MI(X_dimension, Y_dimension, mi)
    X_Y = random_variable.rvs(n_samples)
    X = X_Y[:, 0:X_dimension]
    Y = X_Y[:, X_dimension:X_dimension + Y_dimension]
    X, Y = normal_to_uniform(X), normal_to_uniform(Y)


    # Map to image spce
    X_map, Y_map = gaussian_2_gaussianImg, gaussian_2_gaussianImg
    X, Y = X_map(X), Y_map(Y)


    # To tensor
    X0, Y0 = torch.tensor(X).float().to(device), torch.tensor(Y).float().to(device)
    print('image size', X0.size(), Y0.size())


    # flattening
    X, Y = X0.view(len(X0), -1), Y0.view(len(Y0), -1)
    print('data size', X.size(), Y.size())
    return X, Y











'''
    Rectangle plots
''' 

def rectangles_mapping(X):
    """ Map Gaussian vector to the coordinates of the rectangle. """
    return normal_to_rectangle_coords(X, min_delta, img_width, min_delta, img_height)

def gaussian_2_rectangles(X):
    """ Map Gaussian vector to an image. """
    return rectangle_coords_to_rectangles(rectangles_mapping(X), img_width, img_height)



def generate_rectangle_plot(mi=3, n_samples=10000, device='cuda:0'):
    X_dimension = 4
    Y_dimension = 4
    latent_dimension = 4

    # Generation
    random_variable = multivariate_normal_from_MI(X_dimension, Y_dimension, mi)
    X_Y = random_variable.rvs(n_samples)
    X = X_Y[:, 0:X_dimension]
    Y = X_Y[:, X_dimension:X_dimension + Y_dimension]


    # Map to image spce
    X_map, Y_map = gaussian_2_rectangles, gaussian_2_rectangles
    X, Y = X_map(X), Y_map(Y)


    # To tensor
    X0, Y0 = torch.tensor(X).float().to(device), torch.tensor(Y).float().to(device)
    print('image size', X0.size(), Y0.size())


    # flattening
    X, Y = X0.view(len(X0), -1), Y0.view(len(Y0), -1)
    print('data size', X.size(), Y.size())
    return X, Y
   
    
    
    
    
# ----------------- Visualization facilities ----------------- #  
    
    
    
from torchvision.utils import make_grid


def visualize_images(X, Y, fn='figs'):
    plt.figure()
    fz = 14

    n = len(X)
    
    images_X = X.view(n, 1, img_width, img_height)
    images_Y = Y.view(n, 1, img_width, img_height)

    ax = plt.subplot(1, 2, 1)
    ax.set_title(r"$X$", fontsize=fz)
    ax.set_axis_off()
    grid_img = make_grid(images_X[0:16].cpu(), nrow=8)
    plt.imshow(grid_img.permute(1, 2, 0))

    ax = plt.subplot(1, 2, 2)
    ax.set_title(r"$Y$", fontsize=fz)
    ax.set_axis_off()
    grid_img = make_grid(images_Y[0:16].cpu(), nrow=8)
    
    plt.tight_layout()
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.savefig(fn, dpi=300)