'''
In this file, we implement functions that handle restarting.
Specifically, we implement:
    - KKT error function
    - Getting our restart candidate
    - Restart criteria

    Our functions expect cupy arrays. CRUCIAL!
'''
import numpy as np
import cupy as cp
import torch

def KKT(z, G, A, K, h, c, b, lambd, omega):
    '''
    This functions calculates our KKT error given required inputs
    Takes as input (x,y) which is z, G and A matrices, c, b, lambd vectors, and our primal weight omega
    Every input is a pytorch tensor!
    
    Outputs: scalar (float) error
    '''
    # Unpack x and y from z
    x, y = z

    #Primal residual: [A x - b; h - G x]
    primal_residual = torch.cat([
        torch.matmul(A, x) - b,
        torch.relu(h - torch.matmul(G, x))
    ], dim=0)
    #Note relu sets negative entries to zero
    primal_dev = torch.linalg.norm(primal_residual)

    #Dual residual: c - K^T y - lambd
    dual_dev = torch.linalg.norm(c - torch.matmul(K.t(), y) - lambd)

    #Primal-dual gap: c - K^T y - lambd (same as dual residual in your code)
    prim_dual_gap = torch.linalg.norm(c - torch.matmul(K.t(), y) - lambd)

    #KKT error
    error = torch.sqrt(omega**2 * primal_dev + (1/(omega**2)) * dual_dev + prim_dual_gap)

    return error

def zip_kkt_vars(z,G,A,K,h,c,b,lambd,omega):
    '''Helper function that zips required variables for KKT function, outputs single variables
    This might be fucking useless, likely is so check in implementation and testing'''
    return (z,G,A,K,h,c,b,lambd,omega.zip())


def get_restart_candidate(z_next, z_bar_next, kkt_vars):
    '''
    This functions takes in z_next (z^{n, t+1}), z_bar_next (\bar{z}^{n,t+1})
    It also takes in KKT function, and KKT variables


    Outputs: z-vector restart candidate
    '''
    #Unzip kkt variables
    if (KKT(z_next, kkt_vars) < KKT(z_bar_next, kkt_vars)):
        return z_next 
    else:
        return z_bar_next

def checkRestart(betas, z_cand_next, z_cand_cur, z_n_0, t, k):
    '''
    Function that checks restart criteria. See Lu & Yang 2024 for more deets

    Takes as input beta vector of sensitivies,  z_c^{n,t+1},z_c^{n,t},z^{n,0},t,k
    Outputs: True iff any criteria met
    '''
    #TODO later

