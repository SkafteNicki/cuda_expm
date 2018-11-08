# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 06:41:30 2018

@author: nsde
"""
#%%
import torch

#%%
def _complex_case3x3(a,b,c,d,e,f,x,y):
    ''' Complex solution for the expm function for special form 3x3 matrices '''
    nom = 1/((-a**2-2*a*e-e**2-x**2)*x)
    sinx = (0.5*x).sin()
    cosx = (0.5*x).cos()
    expea = (0.5*(a+e)).exp()
    
    Ea = -(4*((a-e)*sinx+cosx*x))*(a*e-b*d)*expea * nom
    Eb = -8*b*(a*e-b*d)*sinx*expea * nom
    Ec = -4*(((-c*e**2+(a*c+b*f)*e+b*(a*f-2*c*d))*sinx-cosx*x*(b*f-c*e))*expea+(b*f-c*e)*x)*nom
    Ed = -8*d*(a*e-b*d)*sinx*expea * nom
    Ee = 4*((a-e)*sinx-cosx*x)*(a*e-b*d)*expea * nom
    Ef = 4*(((a**2*f+(-c*d-e*f)*a+d*(2*b*f-c*e))*sinx-x*cosx*(a*f-c*d))*expea+x*(a*f-c*d))*nom
    
    E = torch.stack([torch.stack([Ea, Eb, Ec], dim=1),
                     torch.stack([Ed, Ee, Ef], dim=1)], dim=1)
    return E
    
#%%
def _real_case3x3(a,b,c,d,e,f,x,y):
    ''' Real solution for the expm function for special form 3x3 matrices '''
    eap = a+e+x
    eam = a+e-x
    nom = 1/(x*eam*eap)
    expeap = (0.5*eap).exp()
    expeam = (0.5*eam).exp()
      
    Ea = -2*(a*e-b*d)*((a-e-x)*expeam-(a-e+x)*expeap)*nom
    Eb = -4*b*(expeam-expeap)*(a*e-b*d)*nom
    Ec = (((4*c*d-2*f*eap)*b-2*c*e*(a-e-x))*expeam +
         ((-4*c*d+2*f*eam)*b+2*c*e*(a-e+x))*expeap+4*(b*f-c*e)*x)*nom
    Ed = -4*d*(expeam-expeap)*(a*e-b*d)*nom
    Ee = 2*(a*e-b*d)*((a-e+x)*expeam-(a-e-x)*expeap)*nom
    Ef = ((2*a**2*f+(-2*c*d-2*f*(e-x))*a+4*d*(b*f-(1/2)*c*(e+x)))*expeam + 
         (-2*a**2*f+(2*c*d+2*f*(e+x))*a-4*(b*f-(1/2)*c*(e-x))*d)*expeap - 
         (4*(a*f-c*d))*x ) * nom

    E = torch.stack([torch.stack([Ea, Eb, Ec], dim=1),
                     torch.stack([Ed, Ee, Ef], dim=1)], dim=1)
    return E

#%%
def _limit_case3x3(a,b,c,d,e,f,x,y):
    """ Limit solution for the expm function for special form 3x3 matrices """
    ea2 = (a + e)**2
    expea = (0.5*(a + e)).exp()
    Ea = 2*(a - e + 2)*(a * e - b * d) * expea / ea2
    Eb = 4 * b * (a*e - b*d) * expea /ea2
    Ec = ((-2*c*e**2+(2*b*f+2*c*(a+2))*e+2*b*(-2*c*d+f*(a-2)))*expea+4*b*f-4*c*e)/ea2
    Ed = 4*d*(a*e - b*d) * expea / ea2
    Ee = -(2*(a-e-2))*(a*e-b*d) * expea /ea2
    Ef = ((-2*a**2*f+(2*c*d+2*f*(e+2))*a-4*d*(b*f-0.5*c*(e-2)))*expea-4*a*f+4*c*d)/ea2    

    E = torch.stack([torch.stack([Ea, Eb, Ec], dim=1),
                     torch.stack([Ed, Ee, Ef], dim=1)], dim=1)
    return E

#%%
def torch_expm3x3(A):
    """ Tensorflow implementation for finding the matrix exponential of a batch
        of 3x3 matrices that have special form (last row is zero).
    
    Arguments:
        A: 3D-`Tensor` [N,2,3]. Batch of input matrices.
        
    Output:
        expA: 3D-`Tensor` [N,2,3]. Matrix exponential for each matrix in input tensor A.
    """
    # Initilial computations
    a,b,c = A[:,0,0], A[:,0,1], A[:,0,2]
    d,e,f = A[:,1,0], A[:,1,1], A[:,1,2]
    y = a**2 - 2*a*e + 4*b*d + e**2
    x = y.abs().sqrt()
    
    # Calculate all cases and then choose according to the input
    real_res = _real_case3x3(a,b,c,d,e,f,x,y)
    complex_res = _complex_case3x3(a,b,c,d,e,f,x,y)

    expmA = torch.where(y[:,None,None] > 0, real_res, complex_res)
    return expmA

#%%
def torch_expm(A):
    """ """
    n_A = A.shape[0]
    A_fro = torch.sqrt(A.abs().pow(2).sum(dim=(1,2), keepdim=True))
    
    # Scaling step
    maxnorm = torch.Tensor([5.371920351148152]).type(A.dtype).to(A.device)
    zero = torch.Tensor([0.0]).type(A.dtype).to(A.device)
    n_squarings = torch.max(zero, torch.ceil(torch_log2(A_fro / maxnorm)))
    Ascaled = A / 2.0**n_squarings    
    n_squarings = n_squarings.flatten().type(torch.int32)
    
    # Pade 13 approximation
    U, V = torch_pade13(Ascaled)
    P = U + V
    Q = -U + V
    R, _ = torch.gesv(P, Q) # solve P = Q*R
    
    # Unsquaring step
    expmA = [ ]
    for i in range(n_A):
        l = [R[i]]
        for _ in range(n_squarings[i]):
            l.append(l[-1].mm(l[-1]))
        expmA.append(l[-1])
    
    return torch.stack(expmA)

#%%
def torch_log2(x):
    return torch.log(x) / torch.log(torch.Tensor([2.0])).type(x.dtype).to(x.device)

#%%    
def torch_pade13(A):
    b = torch.Tensor([64764752532480000., 32382376266240000., 7771770303897600.,
                      1187353796428800., 129060195264000., 10559470521600.,
                      670442572800., 33522128640., 1323241920., 40840800.,
                      960960., 16380., 182., 1.]).type(A.dtype).to(A.device)
        
    ident = torch.eye(A.shape[1], dtype=A.dtype).to(A.device)
    A2 = torch.matmul(A,A)
    A4 = torch.matmul(A2,A2)
    A6 = torch.matmul(A4,A2)
    U = torch.matmul(A, torch.matmul(A6, b[13]*A6 + b[11]*A4 + b[9]*A2) + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = torch.matmul(A6, b[12]*A6 + b[10]*A4 + b[8]*A2) + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
    return U, V

#%%
if __name__ == '__main__':
    from scipy.linalg import expm
    import numpy as np
    n = 10
    A = torch.randn(n,3,3)
    A[:,2,:] = 0
    
    expm_scipy = np.zeros_like(A)
    for i in range(n):
        expm_scipy[i] = expm(A[i].numpy())
    expm_torch = torch_expm(A)
    print('Difference: ', np.linalg.norm(expm_scipy - expm_torch))