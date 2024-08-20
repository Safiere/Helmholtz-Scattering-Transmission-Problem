import numpy as np
import multiprocessing as mp
import pickle
import time
from math import floor
from scipy.stats import uniform, multivariate_normal


class Sequential_Monte_Carlo():
    def __init__(self, func, meas, var, J, **kwargs):
        self.forward_observation = func
        self.delta               = meas
        self.var                 = var # Variance of the noise on the measurments
        self.J                   = J   # 2J Number of coefficients in Fourier series
    
        self.M     = kwargs["M"]     if "M"     in kwargs else 1000                  # Number of particles
        self.loc   = kwargs["loc"]   if "loc"   in kwargs else np.full(2*self.J, -1) # For i.i.d uniform r.v. [-1,1]
        self.scale = kwargs["scale"] if "scale" in kwargs else np.full(2*self.J, 2)
        
        self.particles = np.array([uniform.rvs(loc=self.loc[i], scale=self.scale[i], size=self.M) for i in range(len(self.loc))]).reshape((len(self.loc), self.M)).T # [loc[i], loc[i]+scale[i]]
        self.weights   = np.full(self.M, 1/self.M)

        self.rho_ratio  = kwargs["rho_ratio"]  if "rho_ratio"  in kwargs else 1.1  # Effective sample size ratio for adaptive temperature choice
        self.max_iter   = kwargs["max_iter"]   if "max_iter"   in kwargs else 25   # Maximal number of iterations in calculation of adaptive temperature choice
        self.p_min      = kwargs["p_min"]      if "p_min"      in kwargs else 0.05 # Minimal increase in calculation of adaptive temperature choice
        self.m          = kwargs["m"]          if "m"          in kwargs else 1    # Global parameter adaptive number of MCMC moves
        self.MCMC_lower = kwargs["MCMC_lower"] if "MCMC_lower" in kwargs else 3    # Lower bound for number of MCMC moves
        self.MCMC_upper = kwargs["MCMC_upper"] if "MCMC_upper" in kwargs else 10   # Upper bound for number of MCMC moves 
        self.lambda_l   = kwargs["lambda_l"]   if "lambda_l"   in kwargs else 0.5  # Initial value global parameter adaptive variance RW MH
        
        self.alpha_l  = 0.2 # Initial acceptance ratio
        self.T        = [0] # Initial temperature


    @staticmethod
    def potential(self, Y):
        return -np.sum((self.delta-self.forward_observation(Y))**2)/(2*self.var)


    def vector_potential(self, *args):
        pool = mp.Pool(mp.cpu_count()) # For multiprocessing
        if len(args) == 0:
            potent = np.array(pool.map(self.potential,[self.particles[i] for i in range(len(self.particles))]))
        else:
            potent = np.array(pool.map(self.potential,[args[0][i] for i in range(len(args[0]))]))
        pool.close()
        return potent


    def reweight(self, potent):
        self.weights = np.exp(np.log(self.weights) + (self.T[-1]-self.T[-2])*potent) # Update is done in log−scale
        self.weights /= np.sum(self.weights) # normalize weights


    def effective_sample_size_after_reweight(self, potent, mid_tmp):
        weights_tmp  = np.exp(np.log(self.weights) + (mid_tmp-self.T[-1])*potent)
        weights_tmp /= np.sum(weights_tmp)
        return 1/np.dot(weights_tmp, weights_tmp)


    def adaptive_temperature(self, potent):
        if (1-self.T[-1]) <= self.p_min:
            T_new = 1
        else:
            # Bisection algorithm
            l      = self.T[-1] # Initial value left limit for bisection
            r      = 1     # Initial value right limit for bisection
            n_iter = 0     # Initial value number of iterations
            while n_iter < self.max_iter and (r-l) > self.p_min:
                mid      = (l+r)/2 # midpoint
                ess_tmp  = self.effective_sample_size_after_reweight(potent, mid)
                if ess_tmp > self.M / self.rho_ratio: # mid can be larger: take half interval on the right
                    l = mid
                else: # mid has to be smaller: take half interval on the left
                    r = mid
                n_iter = n_iter + 1
            T_new = mid
            if (1-T_new) <= self.p_min:
                T_new = 1
        self.T.append(T_new)


    def resample(self):
        self.particles = self.particles[np.random.choice(np.arange(self.M), size=self.M, p=self.weights, replace=True)]
        self.weights   = np.full(self.M, 1/self.M)


    def adaptive_RW_MH(self):
        if self.alpha_l > 0.3:
            self.lambda_l = 2*self.lambda_l
        elif self.alpha_l < 0.15:
            self.lambda_l = 0.5*self.lambda_l
  
        var_RW = self.lambda_l**2 * np.var(self.particles,axis=0)
        M_l = floor(self.m/self.lambda_l**2)
        if M_l < self.MCMC_lower:
            M_l = self.MCMC_lower
        elif M_l> self.MCMC_upper:
            M_l = self.MCMC_upper
        return var_RW, M_l


    def random_walk(self, x, var_RW):
        proposals = np.zeros(x.shape)
        for i in range(len(x)):
            proposals[i,:] = multivariate_normal(mean=np.real(x[i]),cov=np.diag(var_RW)).rvs()
        return proposals


    def MCMC_moves(self, potent):
        var_RW, M_l = self.adaptive_RW_MH()
        total_accepted = 0

        for i in range(M_l):
            proposals       = self.random_walk(var_RW)
            proposal_potent = self.vector_potential(proposals)
            
            potent_ratio = (proposal_potent-potent)*self.T[-1]
            potent_ratio[potent_ratio>0] = 0 # Pobability is maximal 1 (so 0 in log-scale)
            acceptance_prob = np.exp(potent_ratio)*np.logical_and(-1<=proposals, proposals<=1).all(axis=1)

            # Randomly accept the transitions based on the acceptance probability
            accepted = np.random.uniform(size=self.M) < acceptance_prob
            self.particles[accepted] = proposals[accepted]
            potent[accepted] = proposal_potent[accepted]
            total_accepted += np.sum(accepted)

        self.alpha_l = total_accepted/(M_l*self.M)
        return potent


    def SMC_update(self, potent):
        self.reweight(potent)
        self.resample() # Resample every iteration
        potent = self.MCMC_moves(potent)
        return potent


    def SMC_algorithm(self):
        with open(time.strftime("%Y%m%d-%H%M%S") + "_Prior.pickle", "wb") as file:
            pickle.dump(self.particles, file)
            pickle.dump(self.weights, file)  
        
        potent = self.vector_potential()
        while self.T[-1] != 1:
            self.adaptive_temperature(potent)
            potent = self.smc_update(potent)

            if self.T[-1] == 1:
                print("T = {0} is finished".format(self.T[-1]))
                with open(time.strftime("%Y%m%d-%H%M%S") + "_Posterior.pickle", "wb") as file:
                    pickle.dump(self.particles, file)
                    pickle.dump(self.weights, file)
            else:
                print("T = {0:.3} is finished".format(self.T[-1]))
                with open(time.strftime("%Y%m%d-%H%M%S") + "_T={:.3}.pickle".format(self.T[-1]), "wb") as file:
                    pickle.dump(self.particles, file)
                    pickle.dump(self.weights, file)
            print('Average Acceptance Rate:')
            print(self.alpha_l)

        print('Used Temperatures:')
        print(self.T)