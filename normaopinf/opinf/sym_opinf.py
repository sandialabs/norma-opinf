import numpy as np
import scipy.linalg as la


import abc
class NormaOpinfModel(abc.ABC):

    @abc.abstractmethod  
    def __init__(self):
        pass

    @abc.abstractmethod
    def fit(self):
        pass

    @abc.abstractmethod
    def set_solver(self):
        pass


class ShaneNonParametricOpinfModel:
    def __init__(self,model_string):
        self.model = opinf.models.ContinuousModel(model_string)
        self.solver = opinf.lstsq.L2Solver()

    def fit(self,states,ddts,inputs):
        self.model.solver = self.solver
        self.model.fit(states=states, ddts=ddts,inputs=inputs)

    def set_solver(self,reg_parameter):
        self.solver = opinf.lstsq.L2Solver(regularizer=reg_parameter)        

class AnthonyNonParametricOpinfModel:
    def __init__(self):
        self.regularization_parameter = 0 
        self.max_iterations = 100
        self.tolerance = 1e-8

    def fit(self,states,ddts,inputs=None):
        X = np.eye(states.shape[-1])
        self.A,self.B = aoi.alternating_minimization(X,
                                     states,
                                     inputs,
                                     ddts,
                                     reg=self.regularization_parameter,
                                     symmetric_A=self.symmetric_A,
                                     max_iter=self.max_iterations,
                                     tol = self.tolerance)
   
    def set_solver(self,reg_parameter):
        self.regularization_parameter = reg_parameter 

  
                                               
def alternating_minimization(Xs, Ys, Us, Zs, 
                             nus: np.array = np.array(1), 
                             reg: np.array = np.array(0), 
                             symmetric_A: bool = True,
                             max_iter = 100, tol = 1e-8):
    '''
    Routine which infers A, B in Z = XAY + BU with optional
    symmetry constraint on A.  Supports scalar regularization
    and affine parametric dependence in A,B.
    '''
    _, Us, _, _, _ = _fix_input_dims_(Xs, Us, Zs, nus)
    Xs, Ys, Zs, nus, case = _fix_input_dims_(Xs, Ys, Zs, nus)

    if symmetric_A:
        func_A = lambda nus, Xs, Ys, Zs: infer_Tbar_with_symmetry(nus, Xs, Ys,
                                                                  Zs, reg=reg, symmetric=True)
    else:
        func_A = lambda nus, Xs, Ys, Zs: infer_Tbar_with_lstsq(nus, Ys, Zs, reg=reg)
    
    func_B = infer_Tbar_with_lstsq
    
    A = func_A(nus, Xs, Ys, Zs)
    B = func_B(nus, Us, Zs, reg=reg)
    
    for i in range(max_iter):
        # Minimize A given B
        rhs_A = Zs - np.einsum('ibp,ps,bas->ias', B, nus, Us)
        A_new = func_A(nus, Xs, Ys, rhs_A)
        
        # Minimize B given A
        rhs_B = Zs - np.einsum('cds,dep,ps,eas->cas', Xs, A_new, nus, Ys)
        B_new = func_B(nus, Us, rhs_B, reg=reg)

        # Check for convergence
        if np.linalg.norm(A_new - A) < tol and np.linalg.norm(B_new - B) < tol:
            print(f"Converged in {i+1} iterations.")
            break
        
        A = A_new
        B = B_new
    
    return (A, B) if case==2 else (A.squeeze(), B.squeeze())


def symmetric_OpInf(Xs, Ys, Zs, nus: np.array = np.array(1),
                    reg: np.array = np.array(0), symmetric: bool = True):
    '''
    Routine which infers a symmetric/antisymmetric operator
    '''
    Xs, Ys, Zs, nus, case = _fix_input_dims_(Xs, Ys, Zs, nus)
    T = infer_Tbar_with_symmetry(nus, Xs, Ys, Zs, reg, symmetric)

    return T if case==2 else T.squeeze()


def unstructured_OpInf(Ys, Zs, nus: np.array = np.array(1), 
                       reg: np.array = np.array(0)):
    '''
    Routine which infers an unstructured operator 
    '''
    _, Ys, Zs, nus, case = _fix_input_dims_(Ys, Ys, Zs, nus)
    T = infer_Tbar_with_lstsq(nus, Ys, Zs, reg)

    return T if case==2 else T.squeeze()


def infer_Tbar_with_lstsq(nus, Ys, Zs, reg: np.array = np.array(0)):
    '''
    Infers an unstructured tensor operator.
    '''
    p, Ns = nus.shape
    b, Nt, Ns = Ys.shape
    r, Nt, Ns = Zs.shape 

    # Reshape regularization into rp x rp matrix
    if len(reg.shape) == 0:
        reg = reg * np.eye(b*p)
    elif len(reg.shape) == 1:
        reg = np.diag(reg)

    K  = np.einsum("ias,xs->iaxs", Ys, nus)
    Dt = K.transpose((0, 2, 1, 3)).reshape((b*p, Nt*Ns), order="F")
    R  = Zs.reshape((r, Nt*Ns), order="F")

    if reg.sum() != 0.:
        Dt = np.concatenate([Dt, reg], axis=1)
        R  = np.concatenate([R, np.zeros((r, b*p))], axis=1)

    Obar = np.linalg.lstsq(Dt.T, R.T)[0].T
    return Obar.reshape((r, b, p), order="F")


def infer_Tbar_with_symmetry(nus, Xs, Ys, Zs, reg: np.array = np.array(0),
                             symmetric: bool = True):    
    '''
    Infers a tensor operator with symmetry or antisymmetry
    '''
    p, Ns = nus.shape
    r, Nt, Ns = Ys.shape  # or Zs.shape

    # Reshape regularization into r^2p x r^2p matrix
    if len(reg.shape) == 0:
        reg = reg * np.eye(r*r*p)
    elif len(reg.shape) == 1:
        reg = np.diag(reg)

    XsTXs = np.einsum("kis,kjs->ijs", Xs, Xs)
    YsYsT = np.einsum("iks,jks->ijs", Ys, Ys)
    nusnusT = np.einsum("xs,ys->xys", nus, nus)

    Btsr = np.einsum("xys,ijs,kls->xyijkl", nusnusT, XsTXs, YsYsT)
    Btsr += Btsr.transpose(0, 1, 4, 5, 2, 3)  # tensorized Kronecker sum
    Bhat = Btsr.transpose(0, 2, 4, 1, 3, 5).reshape((r*r*p, r*r*p), order="C")

    if reg.sum() != 0.:
        Bhat += reg

    Ctsr = np.einsum("kis,kas,jas,xs->ijx", Xs, Zs, Ys, nus)
    Ctsr += (1 if symmetric else -1) * Ctsr.transpose(1, 0, 2)
    Chat = Ctsr.flatten(order="F")

    vecT = la.solve(Bhat, Chat, assume_a="sym")
    return vecT.reshape((r, r, p), order="F")


def _fix_input_dims_(Xs, Ys, Zs, nus):
    '''
    Helper function to fix dimensions of inputs to p-OpInf
    '''
    if len(nus.shape) == 0:
        # No parametric dependence
        # Expected nus = np.array(1)
        nus = nus[None,None]
        case = 0
        
        if len(Ys.shape) == 2:
            # Only samples in time dimension
            # Expected Ys,Zs have dim 2
            Xs = Xs[:,:,None]
            Ys = Ys[:,:,None]
            Zs = Zs[:,:,None]

    elif len(nus.shape) == 1:
        # Single parametric dependence
        # Expected nus = np.ones(Ns) or Ns parameter samples
        nus = nus[None,:]
        case = 1

    else:
        case = 2

    p, Ns = nus.shape
    b, Nt, Ns = Ys.shape 
    r, Nt, Ns = Zs.shape 

    if len(Xs.shape) == 2:
        # No sample dependence in X matrix
        Xs = Xs[:,:,None] * np.ones([r,r,Ns])

    return Xs, Ys, Zs, nus, case
