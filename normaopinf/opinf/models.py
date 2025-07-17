import numpy as np
import scipy.linalg as la
import opinf
import normaopinf.opinf
import normaopinf.opinf.sym_opinf as sym_opinf

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


    @abc.abstractmethod
    def get_stiffness_matrix(self):
        pass

    def get_exogenous_input_matrix(self):
        pass

class ShaneNonParametricOpInfModel(NormaOpinfModel):
    def __init__(self,model_string):
        print('OpInf Model Type:', model_string)

        self.model = opinf.models.ContinuousModel(model_string)
        self.solver = opinf.lstsq.L2Solver(0.)

    def fit(self,states,ddts,inputs):
        def reshaper(f):
            f_shp = f.shape
            f_reshape = np.reshape(f,(f_shp[0],f_shp[1]*f_shp[2]))
            return f_reshape
        self.model.solver = self.solver
        self.model.fit(states=reshaper(states), ddts=reshaper(ddts),inputs=reshaper(inputs))

    def set_solver(self,reg_parameter):
        self.solver = opinf.lstsq.L2Solver(regularizer=reg_parameter)        

    def get_stiffness_matrix(self):
        return -self.model.A_.entries[:]

    def get_exogenous_input_matrix(self):
        return self.model.B_.entries[:]
    

class ShaneNonParametricQuadraticOpInfModel(NormaOpinfModel):
    def __init__(self,model_string):
        print('OpInf Model Type:',model_string)

        self.model = opinf.models.ContinuousModel(model_string)
        self.solver = opinf.lstsq.L2Solver(0.)

    def fit(self,states,ddts,inputs):
        def reshaper(f):
            f_shp = f.shape
            f_reshape = np.reshape(f,(f_shp[0],f_shp[1]*f_shp[2]))
            return f_reshape
        self.model.solver = self.solver
        self.model.fit(states=reshaper(states), ddts=reshaper(ddts),inputs=reshaper(inputs))

    def set_solver(self,reg_parameter):
        self.solver = opinf.lstsq.L2Solver(regularizer=reg_parameter)        

    def get_stiffness_matrix(self):
        return -self.model.A_.entries[:]

    def get_quadratic_stiffness_matrix(self):
        return -self.model.H_.expand_entries(self.model.H_.entries)

    def get_exogenous_input_matrix(self):
        return self.model.B_.entries[:]


class ShaneNonParametricCubicOpInfModel(NormaOpinfModel):
    def __init__(self,model_string):
        print('OpInf Model Type:',model_string)
        self.model = opinf.models.ContinuousModel(model_string)
        self.solver = opinf.lstsq.L2Solver(0.)

    def fit(self,states,ddts,inputs):
        def reshaper(f):
            f_shp = f.shape
            f_reshape = np.reshape(f,(f_shp[0],f_shp[1]*f_shp[2]))
            return f_reshape
        self.model.solver = self.solver
        self.model.fit(states=reshaper(states), ddts=reshaper(ddts),inputs=reshaper(inputs))

    def set_solver(self,reg_parameter):
        self.solver = opinf.lstsq.L2Solver(regularizer=reg_parameter)        

    def get_stiffness_matrix(self):
        return -self.model.A_.entries[:]

    def get_quadratic_stiffness_matrix(self):
        return -self.model.H_.expand_entries(self.model.H_.entries)

    def get_cubic_stiffness_matrix(self):
        return -self.model.G_.expand_entries(self.model.G_.entries)

    def get_exogenous_input_matrix(self):
        return self.model.B_.entries[:]


class AnthonyNonParametricOpInfModel(NormaOpinfModel):
    def __init__(self):
        self.regularization_parameter = 0 
        self.max_iterations = 100
        self.tolerance = 1e-8

    def fit(self,states,ddts,inputs=None):
        X = np.eye(states.shape[-1])
        self.A,self.B = sym_opinf.alternating_minimization(X,
                                     states,
                                     inputs,
                                     ddts,
                                     reg=self.regularization_parameter,
                                     symmetric_A=True,
                                     max_iter=self.max_iterations,
                                     tol = self.tolerance)
   
    def set_solver(self,reg_parameter):
        self.regularization_parameter = reg_parameter 

    def get_stiffness_matrix(self):
        return -self.A.entries[:]

    def get_exogenous_input_matrix(self):
        return self.B.entries[:]








class QuadraticOpInfRom:
    def __init__(self,A,B,H,opinf_model):
      #N = Phi.shape[0]
      K = A.shape[0]
      self.opinf_model_ = opinf_model
      self.un_ = np.zeros(K)
      self.udotn_ = np.zeros(K)
      self.uddotn_ = np.zeros(K)
      self.unp1_ = np.zeros(K)
      self.unm1_ = np.zeros(K)
      self.udotnp1_ = np.zeros(K)
      self.uddotnp1_ = np.zeros(K)
      self.unm1_ = np.zeros(K)
      self.unp1_ = np.zeros(K)
      self.I_ = np.eye(K)
      #self.Phi_ = Phi
      self.M_ = np.eye(K)
      self.A_ = A
      self.K_ = -self.A_
      self.K_quadratic_ = -H
      self.u_history_ = self.un_[:,None]
      self.B_ = B
      #self.xR = fom.xR
      #self.xL = fom.xL


    def update_states_newmark(self):
        self.u_history_ = np.append(self.u_history_,self.unp1_[:,None],axis=1)
        self.un_[:] =  self.unp1_[:]
        self.udotn_[:] = self.udotnp1_[:]
        self.uddotn_[:] = self.uddotnp1_[:]

    def advance_newmark(self,dt,bcs):
        gamma = 0.5
        beta = 0.25*(gamma + 0.5)**2


        def jacobian(x):
          J =  self.M_/(dt**2 * beta) + self.K_
          x1 = np.kron(self.I_,x)
          x2 = np.kron(x,self.I_)
          J += self.K_quadratic_ @ x1.transpose() + self.K_quadratic_ @ x2.transpose()
          #J -= self.opinf_model_.H_.jacobian(unp1)
          return J

        counter = 0 
        def my_residual(x):
          LHS = (self.M_/(dt**2 * beta) + self.K_) @ x 
          x_sqr = np.kron(x,x)
          LHS += self.K_quadratic_ @ x_sqr
          RHS = 1./(dt**2*beta)*( self.M_ @ self.un_) + 1./(beta*dt)*(self.M_@ self.udotn_)  + 1./(2.*beta)*(1. - 2.*beta)*(self.M_ @  self.uddotn_)
          RHS_f = self.B_ @ bcs
          residual = LHS - (RHS + RHS_f)
          return residual
              
        def my_newton(x):
          #r(x) = 0
          #r(x0) + J dx = 0
          #J dx = -r(x0)
          r = my_residual(x)
          r0_norm = np.linalg.norm(r)
          iteration = 0
          max_its = 20
          while np.linalg.norm(r)/r0_norm > 1e-7 and iteration < max_its:
            J = jacobian(x)
            dx = np.linalg.solve(J,-r)
            #print(f'Relative residual norm: {np.linalg.norm(r)/r0_norm:.4f}, dx: {np.linalg.norm(dx):.4f}, iteration: {iteration}') 
            x = x + dx
            r = my_residual(x)
            iteration += 1
          if iteration == max_its:
            x /= 0. # return nan
          return x[:]
                
#        solution = scipy.optimize.newton_krylov(residual,self.un_*1)
#        solution = scipy.optimize.fsolve(residual,self.un_*1.,fprime=jacobian)
        inputs = self.un_*1.
        solution = my_newton(inputs)

        self.unp1_[:] = solution[:] 
        self.uddotnp1_[:] =  1./(dt**2*beta)*(self.unp1_ - self.un_) - 1./(beta*dt)*self.udotn_ - 1./(2*beta)*(1 - 2.*beta)*self.uddotn_
        self.udotnp1_[:] = self.udotn_ + (1. - gamma)*dt*self.uddotn_ + gamma*dt*self.uddotnp1_

    def advance_n_steps_newmark(self,un,udotn,uddotn,dt,n_steps,bc_hook):
        self.un_[:] = un[:]
        self.udotn_[:] = udotn[:]
        for i in range(0,n_steps):
          bcs = bc_hook(i)
          self.advance_newmark(dt,bcs)
          self.update_states_newmark()
        return self.u_history_

    def get_unp1(self):
        return self.unp1_




class LinearOpInfRom:
    def __init__(self,A,B):
      #N = Phi.shape[0]
      K = A.shape[0]

      self.un_ = np.zeros(K)
      self.udotn_ = np.zeros(K)
      self.uddotn_ = np.zeros(K)
      self.unp1_ = np.zeros(K)
      self.unm1_ = np.zeros(K)
      self.udotnp1_ = np.zeros(K)
      self.uddotnp1_ = np.zeros(K)
      self.unm1_ = np.zeros(K)
      self.unp1_ = np.zeros(K)

      #self.Phi_ = Phi
      self.M_ = np.eye(K)
      self.A_ = A
      self.K_ = -self.A_
      self.u_history_ = self.un_[:,None]
      self.B_ = B
      #self.xR = fom.xR
      #self.xL = fom.xL


    def update_states_newmark(self):
        self.u_history_ = np.append(self.u_history_,self.unp1_[:,None],axis=1)
        self.un_[:] =  self.unp1_[:]
        self.udotn_[:] = self.udotnp1_[:]
        self.uddotn_[:] = self.uddotnp1_[:]

    def advance_newmark(self,dt,bcs):
        gamma = 0.5
        beta = 0.25*(gamma + 0.5)**2

        ##M uddot + Ku = f
        LHS = self.M_/(dt**2 * beta) + self.K_
        RHS = 1./(dt**2*beta)*( self.M_ @ self.un_) + 1./(beta*dt)*(self.M_@ self.udotn_)  + 1./(2.*beta)*(1. - 2.*beta)*(self.M_ @  self.uddotn_)
        RHS_f = self.B_ @ bcs

        self.unp1_[:] = np.linalg.solve(LHS,RHS + RHS_f)
        self.uddotnp1_[:] =  1./(dt**2*beta)*(self.unp1_ - self.un_) - 1./(beta*dt)*self.udotn_ - 1./(2*beta)*(1 - 2.*beta)*self.uddotn_
        self.udotnp1_[:] = self.udotn_ + (1. - gamma)*dt*self.uddotn_ + gamma*dt*self.uddotnp1_


    def advance_n_steps_newmark(self,un,udotn,uddotn,dt,n_steps,bc_hook):
        self.un_[:] = un[:]
        self.udotn_[:] = udotn[:]
        self.uddotn_[:] = uddotn[:]
        for i in range(0,n_steps):
          bcs = bc_hook(i)
          self.advance_newmark(dt,bcs)
          self.update_states_newmark()
        return self.u_history_

    def get_unp1(self):
        return self.unp1_





