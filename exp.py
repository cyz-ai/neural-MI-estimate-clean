import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd as autograd
import numpy as np
import time 
from utils import utils_os

from datasets import NonlinearGaussian, MoG, SwissRoll, Spiral
from datasets.Student_t import MultivariateStudentT

from estimators.MRE import MRE
from estimators.NAE import NAE
from estimators.VCE import VCE
from estimators.MINE import MINE
from estimators.SMILE import SMILE
from estimators.DoE import DoE
from estimators.InfoNCE import InfoNCE
from estimators.MIENF import MIENF
from estimators.MINDE import MINDE


# Hyperparams of MI estimators
class Hyperparams(object):
    def __init__(self): 
        self.critic = 'neural'                # ('neural', 'quadratic')
        self.lr = 5e-4
        self.bs = 500
        self.wd = 1e-5
        self.n_bridges = 4
        self.early_stop = True
        self.t_patience = 500
        self.importance_sampling = True
        self.max_iteration = 1000

hyperparams=Hyperparams()

# Archtecture of MI estimators
architecture_critic = [np.inf, 500, 500, 500, 1]
architecture_encode = None

# MI estimators considered:
estimator_names = ['Multinomial', 'MINE', 'DoE', 'MIENF', 'VCE', 'InfoNCE', 'MINDE']         
estimators = [MRE, MINE, DoE, MIENF, VCE, InfoNCE, MINDE]                                       
      

# Neural density estimator
estimator2nde = {'Adaptive':'VGC', 'MIENF':'FM', 'VCE':'FM', 'Adaptive-GC':'GC', 'Adaptive-FM':'FM'}
estimator2K_components = {'MIENF':1, 'VCE':5, 'VCE-8':8}            # <-- for student-t, VCE use K=8

# Data saving directory
DIR = 'results/synthetic/data_n10000'
n_data = 10000



def _evaluate_core(fn, X, Y, ground_truth_MI, n_exps, device):
    n, d = torch.cat([X, Y], dim=1).size()
    print('n data=', n, 'd=', d)

    # Create result file
    fn_full = fn+'.npy'
    if utils_os.is_file_exist(DIR, fn_full):
        print('result file already exist, will update file. \n')
        results = utils_os.load_object(DIR, fn_full).item()
    else:
        print('result file not exist, will create new file. \n')
        results = {}
        
    # MI estimate
    for i, name in enumerate(estimator_names):
        results[name] = []
        print('estimator:', name, 'GT MI:', ground_truth_MI)
        
        # set device
        hyperparams.device = device

        # choose nde type (for our estimator only)
        hyperparams.nde_type = estimator2nde.get(name)
        hyperparams.K_components = estimator2K_components.get(name)
        
        # record dimensionality
        hyperparams.dim = d//2
        
        for _ in range(n_exps):
            # consider each configurations
            estimator = estimators[i](architecture_encoder_x=architecture_encode, 
                            architecture_encoder_y=architecture_encode, 
                            architecture_critic= [d] + architecture_critic[1:], 
                            hyperparams=hyperparams)

            estimator.to(device)            
            estimator.learn(X, Y)
            estimator.eval()
            MI_est = estimator.MI(X, Y)
            results[name].append(MI_est)
        print('estimator=', name, 'results=', results[name], '\n')
    results['Truth'] = ground_truth_MI
    # save result
    utils_os.save_object(DIR, fn, results)

    return utils_os.load_object(DIR, fn_full).item()




def evaluate_nonlinear_gaussian(case, dim, rho, n_exps=3, device='cuda:0'):
    print('case', case, 'dim', dim, 'rho', rho)

    # Dataset preparation
    n, d = n_data, dim               
    true_rho = rho
    case = case

    dataset = NonlinearGaussian.NonlinearGaussian(n_samples=n, n_dims=d, rho=true_rho, mu=0, case=case)
    X0, Y0 = dataset.sample_data(n_samples = n)
    X, Y = dataset.transformation(X0, Y0)
    X, Y = X.to(device), Y.to(device)

    # File name to save
    fn = f'nonlinear_gaussian_{case}_{dim}_{rho}'

    return _evaluate_core(X=X, Y=Y, fn=fn, ground_truth_MI=dataset.true_mutual_info(), n_exps=n_exps, device=device)




def evaluate_MoG(case, dim, n_exps=3, device='cuda:0'):
    print('case', case, 'dim', dim)

    # Dataset preparation
    n, d = n_data, dim               
    case = case

    shifts = [
        [-0.4, -0.1, 0, 0.1, 0.4],
        [-0.2, -0.1, 0, 0.3, 0.4]
    ]
    rhos = [
        [0.5, 0.6, 0.7, 0.8, 0.9],
        [-0.3, 0.5, 0.2, 0.4, 0.9]
    ]

    dataset = MoG.MoG(n_samples=n, n_dims=d, K=5, shifts=shifts[case], rhos=rhos[case])
    X, Y = dataset.sample_data(n_samples = n)
    X, Y = X.to(device), Y.to(device)

    # File name to save
    fn = f'MoG_{case}_{dim}'

    return _evaluate_core(X=X, Y=Y, fn=fn, ground_truth_MI=dataset.empirical_mutual_info(), n_exps=n_exps, device=device)




def evaluate_swiss_roll(dim, rho, n_exps=3, device='cuda:0'):
    print('case', 'default', 'dim', dim, 'rho', rho)

    # Dataset preparation
    n, d = n_data, dim     
    true_rho = rho          
    case = 'default'

    dataset = SwissRoll.SwissRoll(n_samples=n, n_dims=d, rho=true_rho, mu=0)
    X, Y = dataset.sample_data(n_samples = n)
    X, Y = X.to(device), Y.to(device)

    assert 1!=2, 'this part is unfinished.'

    d = d*2                        # <-- needs not to handle
    hyperparams=Hyperparams()
    hyperparams.n_bridges = 0      # <-- needs some good way to handle


    # File name to save
    fn = f'swissroll_{case}_{dim}_{rho}'
    return _evaluate_core(X=X, Y=Y, fn=fn, ground_truth_MI=dataset.empirical_mutual_info(), n_exps=n_exps, device=device)




def evaluate_student_t_new(dim, rho=0.8, n_exps=3, v=3, device='cuda:0'):
    print('case', 'default', 'dim', dim, 'rho', rho)

    # Dataset preparation
    n, d = n_data, dim     
    true_rho = rho          
    case = 'default'

    d = dim
    dim_x = dim_y = d//2
    rho = rho
    df = v
    mean = np.zeros(dim_x + dim_y)
    V = np.eye(dim_x)*rho
    dispersion = np.eye(dim_x + dim_y)
    dispersion[0:dim_x, :][:, dim_x:] = V
    dispersion[dim_x:, :][:, 0:dim_x] = V

    dataset = MultivariateStudentT(
            dim_x=dim_x,
            dim_y=dim_y,
            mean=mean,
            dispersion=dispersion,
            df=df)

    X, Y = dataset.sample(n_data)
    X, Y = torch.Tensor(X).float().to(device), torch.Tensor(Y).float().to(device)

    # File name to save
    fn = f'student_t_new_v{df}_{dim}_{rho}'
    return _evaluate_core(X=X, Y=Y, fn=fn, ground_truth_MI=dataset.mutual_information(), n_exps=n_exps, device=device)

        




    
# def evaluate_spiral(dim, rho, n_exps=3, device='cuda:0'):
#     print('case', 'default', 'dim', dim, 'rho', rho)

#     # Dataset preparation
#     n, d = n_data, dim     
#     true_rho = rho          
#     case = 'default'

#     dataset = Spiral.Spiral(rho=true_rho, dim=d)
#     X, Y = dataset.sample(n=10000)
#     X, Y = X.to(device).clone().detach(), Y.to(device).clone().detach()

#     assert 1!=2, 'this part is unfinished.'

#     d = d*2                        # <-- needs not to handle
#     hyperparams=Hyperparams()
#     hyperparams.n_bridges = 0      # <-- needs some good way to handle

#     # File name to save
#     fn = f'spiral_{case}_{dim}_{rho}'
#     return _evaluate_core(X=X, Y=Y, fn=fn, ground_truth_MI=dataset.MI(), n_exps=n_exps, device=device)









