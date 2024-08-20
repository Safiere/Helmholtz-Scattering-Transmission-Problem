import numpy as np
import ufl
from scipy.special import zeta
from mpmath import nsum, inf
from dolfinx import fem, geometry
from petsc4py import PETSc
from Generate_Mesh import *

'''
This code is made for DOLFINx version 0.5.1.
'''

class Helmholtz():
    def __init__(self, **kwargs):
        self.alpha_in  = kwargs["alpha_in"]  if "alpha_in"  in kwargs else 1   # Material constant inside scatterer
        self.alpha_out = kwargs["alpha_out"] if "alpha_out" in kwargs else 1   # Material constant outside scatterer
        self.n_in      = kwargs["n_in"]      if "n_in"      in kwargs else 0.9 # Refractive index inside scatterer
        self.n_out     = kwargs["n_out"]     if "n_out"     in kwargs else 1   # Refractive index ouside scatterer

        self.dir  = kwargs["dir"]  if "dir"  in kwargs else np.array([1.0,0.0]) # Direction of propagation, norm should be 1
        self.c    = kwargs["c"]    if "c"    in kwargs else 3*10**10            # Lightspeed in cm
        self.freq = kwargs["freq"] if "freq" in kwargs else 10**9               # Frequency of incoming wave in dm
        self.kappa_0 = 2*np.pi*self.freq/self.c

        self.r0        = kwargs["r0"]        if "r0"        in kwargs else 1            # Radius of reference configuration in cm (scaling because of numerical underflow)
        self.r1        = kwargs["r1"]        if "r1"        in kwargs else 6            # Radius of measured points in physical domain in dm, must be greater than 1.5*r0, smaller than R
        self.R         = kwargs["R"]         if "R"         in kwargs else 7            # Radius of coordinate transformation domain D_R in cm
        self.R_tilde   = kwargs["R_tilde"]   if "R_tilde"   in kwargs else 7.5          # Outer radius PML in cm
        self.R_PML     = kwargs["R_PML"]     if "R_PML"     in kwargs else 11           # Outer radius PML in cm
        self.sigma_PML = kwargs["sigma_PML"] if "sigma_PML" in kwargs else 10000        # Global demping parameter of PML layer
        self.gdim      = kwargs["gdim"]      if "gdim"      in kwargs else 2            # Geometric dimension of the mesh
        self.h         = kwargs["h"]         if "h"         in kwargs else self.r0/2**3 # Characteristic length of mesh elements

        create_domain = Generate_Mesh(kwargs)
        self.domain, ct, ft = create_domain()

        self.V  = fem.FunctionSpace(self.domain, ("CG", 1)) # Solution space
        self.Q  = fem.FunctionSpace(self.domain, ("DG", 0)) # For discontinuous expressions

        self.alpha      = fem.Function(self.Q)
        self.kappa_sqrd = fem.Function(self.Q)
        material_tags = np.unique(self.ct.values)
        for tag in material_tags:
            cells = ct.find(tag)
            if tag == 1 or tag == 2 or tag == 3:
                alpha_ = self.alpha_out
                kappa_sqrd_ = self.kappa_0**2*self.n_out
            elif tag == 4 or tag == 5:
                alpha_ = self.alpha_in
                kappa_sqrd_ = self.kappa_0**2*self.n_in
            self.alpha.x.array[cells] = np.full_like(cells, alpha_, dtype=PETSc.ScalarType)
            self.kappa_sqrd.x.array[cells] = np.full_like(cells, kappa_sqrd_, dtype=PETSc.ScalarType)

        self.bc  = fem.dirichletbc(fem.Constant(self.domain, PETSc.ScalarType(0)), fem.locate_dofs_topological(self.V, self.gdim-1, ft.find(6)), self.V) # Set zero Dirichlet boundary condition at R_PML
        self.u_i_boundary  = fem.Function(self.V)
        dof = self.V.tabulate_dof_coordinates()[:, 0:2]
        dofs_boundary    = fem.locate_dofs_topological(self.V, self.gdim-1, ft.find(8)) # Degrees of freedom at R
        coords_boundary  = self.domain.geometry.x[dofs_boundary]
        values_boundary  = self.u_i(dof[dofs_boundary].transpose())
        with self.u_i_boundary.vector.localForm() as loc:
            loc.setValues(dofs_boundary, values_boundary)
        self.u_i_n = fem.Function(self.V)
        self.u_i_n.interpolate(self.u_in)

        self.dx_inner = ufl.Measure('dx', domain=self.domain, subdomain_data=ct, subdomain_id=3) # Integration on medium domain
        self.dS       = ufl.Measure('dS', domain=self.domain, subdomain_data=ft, subdomain_id=8) # Surface integration at R

        self.A_matrix, self.dd_bar = self.build_PML(self.Q, self.V)

        self.u = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)

        self.solver = PETSc.KSP().create(self.domain.comm)
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)

        L = self.alpha('+')*ufl.inner(self.u_i_n, self.v)('+')*self.dS - self.alpha*ufl.inner(ufl.grad(self.u_i_boundary), ufl.grad(self.v))*self.dx_inner + self.kappa_sqrd*ufl.inner(self.u_i_boundary, self.v)*self.dx_inner
        self.b = fem.petsc.assemble_vector(fem.form(L))
        self.b.assemble()
        fem.petsc.set_bc(self.b, [self.bc])

        self.epsilon  = kwargs["epsilon"]  if "epsilon"  in kwargs else 0.001   # Small number greater than zero for convergence of radius expansion
        self.char_len = kwargs["char_len"] if "char_len" in kwargs else False   # Determines type of expansion

        if self.char_len == True:
            self.s = kwargs["s"] if "s" in kwargs else 0.001 # Scaled version of correlation length
            self.sum = nsum(lambda k: 1/(1 + self.s*k**(2 + self.epsilon), [1, inf]))

            var_sum = nsum(lambda k: 1/((1 + self.s*k**(2 + self.epsilon)**2), [1, inf]))
            sum_j, j = 0, 0
            while sum_j < 0.95*var_sum:
                j += 1
                sum_j += 1/((1 + self.s*j**(2 + self.epsilon))**2)
            self.J = j
        
        else:
            var_sum = zeta(2*(2 + self.epsilon))
            sum_j, j = 0, 0
            while sum_j < 0.95*var_sum:
                j += 1
                sum_j += 1/(j**(2*(2 + self.epsilon)))
            self.J = j

        self.K = kwargs["K"] if "K" in kwargs else 100  # Number of measured points
        self.angles_meas = np.array([i for i in range(self.K)])/self.K*2*np.pi


    @staticmethod
    def u_i(self, x): # Amplitude of incoming wave
        return np.e**(complex(0,1)*self.kappa_0*np.sqrt(self.n_out/self.alpha_out)*(dir[0]*x[0] + dir[1]*x[1]))


    def u_in(self, x): # Radial normal derivative of u_i
        return complex(0,1)*self.kappa_0*np.sqrt(self.n_out/self.alpha_out)*(x[0]*self.dir[0] + x[1]*self.dir[1])*np.e**(complex(0,1)*self.kappa_0*np.sqrt(self.n_out/self.alpha_out)*(self.dir[0]*x[0] + self.dir[1]*x[1]))/np.sqrt(x[0]**2+x[1]**2)
    

    def sigma(self, rho):
            return self.sigma_PML*np.minimum(np.maximum((rho - self.R_tilde)/(self.R_PML - self.R_tilde), 0), 1)


    def sigma_bar(self, rho):
        return self.sigma_PML*(rho - self.R_tilde)**2/(2*rho*(self.R_PML - self.R_tilde))*(self.R_tilde <= rho)*(rho <= self.R_PML) + self.sigma_PML*(-(self.R_PML + self.R_tilde)/(2*rho) + 1)*(self.R_PML < rho)


    def d(self, rho):
        return 1 + complex(0,1)*self.sigma(rho)/(2*np.pi*self.freq)


    def d_bar(self, rho):
        return 1 + complex(0,1)*self.sigma_bar(rho)/(2*np.pi*self.freq)


    def Axx(self, x):
        rho, phi = np.sqrt(x[0]**2 + x[1]**2), np.arctan2(x[1], x[0])
        d_rho, d_bar_rho = self.d(rho), self.d_bar(rho)
        return d_bar_rho/d_rho*np.cos(phi)**2 + d_rho/d_bar_rho*np.sin(phi)**2


    def Axy(self, x):
        rho, phi = np.sqrt(x[0]**2 + x[1]**2), np.arctan2(x[1], x[0])
        d_rho, d_bar_rho = self.d(rho), self.d_bar(rho)
        return (d_bar_rho/d_rho - d_rho/d_bar_rho)*np.cos(phi)*np.sin(phi)


    def Ayy(self, x):
        rho, phi = np.sqrt(x[0]**2 + x[1]**2), np.arctan2(x[1], x[0])
        d_rho, d_bar_rho = self.d(rho), self.d_bar(rho)
        return d_rho/d_bar_rho*np.cos(phi)**2 + d_bar_rho/d_rho*np.sin(phi)**2
    

    def build_PML(self, Q, V): # Perfectly Matching Layer
        a11    = fem.Function(Q)
        a12    = fem.Function(Q)
        a22    = fem.Function(Q)
        dd_bar = fem.Function(V)
        a11.interpolate(self.Axx)
        a12.interpolate(self.Axy)
        a22.interpolate(self.Ayy)
        dd_bar.interpolate(lambda x: self.d(np.sqrt(x[0]**2 + x[1]**2))*self.d_bar(np.sqrt(x[0]**2 + x[1]**2)))
        return ufl.as_matrix([[a11,a12], [a12,a22]]), dd_bar
    

    def radial(self, Y, phi):
        return np.sum(np.array([(Y[2*j-2]*np.cos(j*phi) + Y[2*j-1]*np.sin(j*phi))/(1 + self.s*j**(2 + self.epsilon)) for j in range(1, self.J+1)]))*self.r0/(4*sum)


    def der_radial_x(self, Y, x, rho, phi):
        if self.char_len == True:
            return np.sum(np.array([(Y[2*j-2]*np.sin(j*phi) - Y[2*j-1]*np.cos(j*phi))*j/(1 + self.s*j**(2 + self.epsilon)) for j in range(1, self.J+1)]))*x[1]/rho**2*self.r0/(4*self.sum)
        else:
            return np.sum(np.array([(Y[2*j-2]*np.sin(j*phi) - Y[2*j-1]*np.cos(j*phi))*j/(j**(2 + self.epsilon)) for j in range(1, self.J+1)]))*x[1]/rho**2*self.r0/(4*zeta(2 + self.epsilon))


    def der_radial_y(self, Y, x, rho, phi):
        if self.char_len == True:
            return np.sum(np.array([(Y[2*j-2]*np.sin(j*phi) - Y[2*j-1]*np.cos(j*phi))*j/(1 + self.s*j**(2 + self.epsilon)) for j in range(1, self.J+1)]))*-x[0]/rho**2*self.r0/(4*self.sum)
        else:
            return np.sum(np.array([(Y[2*j-2]*np.sin(j*phi) - Y[2*j-1]*np.cos(j*phi))*j/(j**(2 + self.epsilon)) for j in range(1, self.J+1)]))*-x[0]/rho**2*self.r0/(4*zeta(2 + self.epsilon))


    def mollifier_1(self, rho): # Mollifier for r0/4 \leq \rho \leq r0
        return (4*rho - self.r0)/(3*self.r0)


    def mollifier_2(self, rho): # Mollifier for r0 \leq \rho \leq R
        return (self.R - rho)/(self.R - self.r0)


    def Jacxx(self, x, rho, radial_Y, der_radial_x_Y):
        return 1 + (4*x[0]/(3*self.r0*rho)*x[0]/rho*radial_Y + self.mollifier_1(rho)*x[1]**2/rho**3*radial_Y + self.mollifier_1(rho)*x[0]/rho*der_radial_x_Y)*(self.r0/4 < rho)*(rho <= self.r0) + (-x[0]/((self.R - self.r0)*rho)*x[0]/rho*radial_Y + self.mollifier_2(rho)*x[1]**2/rho**3*radial_Y + self.mollifier_2(rho)*x[0]/rho*der_radial_x_Y)*(self.r0 < rho)*(rho <= self.R)


    def Jacxy(self, x, rho, radial_Y, der_radial_y_Y):
        return (4*x[1]/(3*self.r0*rho)*x[0]/rho*radial_Y - self.mollifier_1(rho)*x[0]*x[1]/rho**3*radial_Y + self.mollifier_1(rho)*x[0]/rho*der_radial_y_Y)*(self.r0/4 < rho)*(rho <= self.r0) + (-x[1]/((self.R - self.r0)*rho)*x[0]/rho*radial_Y - self.mollifier_2(rho)*x[0]*x[1]/rho**3*radial_Y + self.mollifier_2(rho)*x[0]/rho*der_radial_y_Y)*(self.r0 < rho)*(rho <= self.R)


    def Jacyx(self, x, rho, radial_Y, der_radial_x_Y):
        return (4*x[0]/(3*self.r0*rho)*x[1]/rho*radial_Y - self.mollifier_1(rho)*x[0]*x[1]/rho**3*radial_Y + self.mollifier_1(rho)*x[1]/rho*der_radial_x_Y)*(self.r0/4 < rho)*(rho <= self.r0) + (-x[0]/((self.R - self.r0)*rho)*x[1]/rho*radial_Y - self.mollifier_2(rho)*x[0]*x[1]/rho**3*radial_Y + self.mollifier_2(rho)*x[1]/rho*der_radial_x_Y)*(self.r0 < rho)*(rho <= self.R)


    def Jacyy(self, x, rho, radial_Y, der_radial_y_Y):
        return 1 + (4*x[1]/(3*self.r0*rho)*x[1]/rho*radial_Y + self.mollifier_1(rho)*x[0]**2/rho**3*radial_Y + self.mollifier_1(rho)*x[1]/rho*der_radial_y_Y)*(self.r0/4 < rho)*(rho <= self.r0) + (-x[1]/((self.R - self.r0)*rho)*x[1]/rho*radial_Y + self.mollifier_2(rho)*x[0]**2/rho**3*radial_Y + self.mollifier_2(rho)*x[1]/rho*der_radial_y_Y)*(self.r0 < rho)*(rho <= self.R)


    def alpha_hatxx(self, Y, x):
        rho, phi = np.sqrt(x[0]**2 + x[1]**2), np.arctan2(x[1], x[0])
        radial_Y, der_radial_x_Y, der_radial_y_Y = self.radial(Y, phi), self.der_radial_x(Y, x, rho, phi), self.der_radial_y(Y, x, rho, phi)
        Jac00, Jac01, Jac10, Jac11 = self.Jacxx(x, rho, radial_Y,der_radial_x_Y), self.Jacxy(x, rho, radial_Y, der_radial_y_Y), self.Jacyx(x, rho, radial_Y, der_radial_x_Y), self.Jacyy(x, rho, radial_Y, der_radial_y_Y)
        return (Jac01**2 + Jac11**2)/(Jac00*Jac11 - Jac01*Jac10)


    def alpha_hatxy(self, Y, x):
        rho, phi = np.sqrt(x[0]**2 + x[1]**2), np.arctan2(x[1], x[0])
        radial_Y, der_radial_x_Y, der_radial_y_Y = self.radial(Y, phi), self.der_radial_x(Y, x, rho, phi), self.der_radial_y(Y, x, rho, phi)
        Jac00,Jac01,Jac10,Jac11 = self.Jacxx(x, rho, radial_Y, der_radial_x_Y), self.Jacxy(x, rho, radial_Y, der_radial_y_Y), self.Jacyx(x, rho, radial_Y, der_radial_x_Y),self.Jacyy(x, rho, radial_Y, der_radial_y_Y)
        return (-(Jac00*Jac01 + Jac10*Jac11))/(Jac00*Jac11 - Jac01*Jac10)


    def alpha_hatyy(self, Y, x):
        rho, phi = np.sqrt(x[0]**2 + x[1]**2), np.arctan2(x[1], x[0])
        radial_Y, der_radial_x_Y, der_radial_y_Y = self.radial(Y, phi), self.der_radial_x(Y, x, rho, phi), self.der_radial_y(Y, x, rho, phi)
        Jac00, Jac01, Jac10, Jac11 = self.Jacxx(x, rho, radial_Y, der_radial_x_Y), self.Jacxy(x, rho, radial_Y, der_radial_y_Y),self.Jacyx(x, rho, radial_Y, der_radial_x_Y),self.Jacyy(x, rho, radial_Y, der_radial_y_Y)
        return (Jac00**2 + Jac10**2)/(Jac00*Jac11 - Jac01*Jac10)


    def kappa_sqrd_trans(self, Y, x):
        rho, phi = np.sqrt(x[0]**2 + x[1]**2), np.arctan2(x[1], x[0])
        radial_Y, der_radial_x_Y, der_radial_y_Y = self.radial(Y, phi), self.der_radial_x(Y, x, rho, phi), self.der_radial_y(Y, x, rho, phi)
        Jac00, Jac01, Jac10, Jac11 = self.Jacxx(x, rho, radial_Y, der_radial_x_Y), self.Jacxy(x, rho, radial_Y, der_radial_y_Y), self.Jacyx(x, rho, radial_Y, der_radial_x_Y),self.Jacyy(x, rho, radial_Y, der_radial_y_Y)
        return Jac00*Jac11 - Jac01*Jac10


    def build_mapping(self, Y): # Coordinate mapping
        alpha_hat00    = fem.Function(self.Q)
        alpha_hat01    = fem.Function(self.Q)
        alpha_hat11    = fem.Function(self.Q)
        kappa_sqrd_hat = fem.Function(self.Q)
        alpha_hat00.interpolate(lambda x: self.alpha_hatxx(Y, x))
        alpha_hat01.interpolate(lambda x: self.alpha_hatxy(Y, x))
        alpha_hat11.interpolate(lambda x: self.alpha_hatyy(Y, x))
        kappa_sqrd_hat.interpolate(lambda x: self.kappa_sqrd_trans(Y, x))
        return ufl.as_matrix([[alpha_hat00,alpha_hat01], [alpha_hat01, alpha_hat11]]), kappa_sqrd_hat


    def chi(self, rho):
        return np.piecewise(rho, [rho <= self.r0/4, (self.r0/4 < rho) & (rho <= self.r0), (self.r0 < rho) & (rho <= self.R), self.R < rho], [lambda rho: 0, lambda rho: (4*rho - self.r0)/(3*self.r0), lambda rho: (self.R - rho)/(self.R - self.r0), lambda rho: 0])
    

    def f(self, r_hat, rad_factor):
        return r_hat + rad_factor*self.chi(r_hat) - self.r1


    def find_nearest(self, array, value):
        return (np.abs(np.asarray(array) - value)).argmin()


    def observation_radius_reference(self, Y):
        radius_meas = np.zeros(len(self.angles_meas))
        r_hat = np.linspace(self.r0/4, self.R, 50000)
        for i in range(len(self.angles_meas)):
            if self.char_len == True:
                rad_factor = np.sum(np.array([(Y[2*j-2]*np.cos(j*self.angles_meas[i]) + Y[2*j-1]*np.sin(j*self.angles_meas[i]))/(1 + self.s*j**(2 + self.epsilon)) for j in range(1, self.J+1)]))*self.r0/(4*self.sum)
            else:
                rad_factor = np.sum(np.array([(Y[2*j-2]*np.cos(j*self.angles_meas[i]) + Y[2*j-1]*np.sin(j*self.angles_meas[i]))/(j**(2 + self.epsilon)) for j in range(1, self.J+1)]))*self.r0/(4*zeta(2 + self.epsilon))
            r_hat_result = self.f(r_hat,rad_factor)
            radius_meas[i] = r_hat[self.find_nearest(r_hat_result, 0)]
        return radius_meas


    def forward_observation(self, Y):
        uh = fem.Function(self.V)
        alpha_hat, kappa_sqrd_hat = self.build_mapping(Y)

        a = ufl.inner(self.alpha*alpha_hat*self.A_matrix*ufl.grad(self.u), ufl.grad(self.v))*ufl.dx - ufl.inner(self.kappa_sqrd*kappa_sqrd_hat*self.dd_bar*self.u, self.v)*ufl.dx
        bilinear_form = fem.form(a)
        A = fem.petsc.assemble_matrix(bilinear_form, bcs=[self.bc])
        A.assemble()

        self.solver.setOperators(A)
        self.solver.solve(self.b, uh.vector)

        # Observation operator
        radius_meas = self.observation_radius_reference(Y)
        x_meas = radius_meas*np.cos(self.angles_meas)
        y_meas = radius_meas*np.sin(self.angles_meas)

        measurement_points = np.zeros((3, self.K))
        measurement_points[0] = x_meas
        measurement_points[1] = y_meas

        bb_tree = geometry.BoundingBoxTree(self.domain, self.gdim)
        cells = []
        points_on_proc = []
        # Find cells whose bounding-box collide with the points
        cell_candidates = geometry.compute_collisions(bb_tree, measurement_points.T)
        # Choose one of the cells that contains the point
        colliding_cells = geometry.compute_colliding_cells(self.domain, cell_candidates, measurement_points.T)
        for i, point in enumerate(measurement_points.T):
            if len(colliding_cells.links(i)) > 0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])
        points_on_proc = np.array(points_on_proc, dtype=np.float64)
        uh_vector = np.array(uh.eval(points_on_proc, cells))
        uh_vector.reshape(len(self.angles_meas))
        return np.real(uh_vector)