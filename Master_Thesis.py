import numpy as np
import pickle
import time
from scipy.stats import uniform, multivariate_normal
from Helmholtz import *
from Sequential_Monte_Carlo import *


def main():
    kwargs    = {"freq":10**9, "char_len":True, "s":0.001, "K":100}
    helm      = Helmholtz(kwargs)
    loc       = np.full(2*helm.J, -1) 
    scale     = np.full(2*helm.J, 2)
    Y         = np.array([uniform.rvs(loc=loc[i], scale=scale[i]) for i in range(len(loc))])[0]
    forward_Y = helm.forward_observation(Y)
    var       = np.mean(forward_Y)*0.1
    eta       = multivariate_normal(mean=np.zeros(helm.K), cov=var*np.eye(helm.K)).rvs()
    delta     = forward_Y + eta

    with open(time.strftime("%Y%m%d-%H%M%S") + "Parameters_Simulation.pickle", "wb") as file:
            pickle.dump(Y, file)
            pickle.dump(eta, file)

    smc = Sequential_Monte_Carlo(helm.forward_observation, delta, var, helm.J)
    smc.SMC_algorithm()

if __name__ == "__main__":
    main()


## Docker
# docker run --rm -ti -v c:\Users\safie\OneDrive\Documenten\Master_Thesis\Code:/opt/project wvharten/dolfinx:0.5.1
# singularity run -B Master_Thesis:/opt/project -B scratch:/scratch Singularity/dolfinx_complex.sif

#cd /opt/project
#python3 script.py