import numpy as np
import gmsh
from dolfinx import io
from dolfinx.io.gmshio import model_to_mesh
from mpi4py import MPI

'''
This code is made for DOLFINx version 0.5.1.
'''

class Generate_Mesh():
    def __init__(self, **kwargs):
        self.r0      = kwargs["r0"]      if "r0"      in kwargs else 1            # Radius of reference configuration in cm (scaling because of numerical underflow)
        self.R       = kwargs["R"]       if "R"       in kwargs else 7            # Radius of coordinate transformation domain D_R in cm
        self.R_tilde = kwargs["R_tilde"] if "R_tilde" in kwargs else 7.5          # Radius where PML absorption starts in cm
        self.R_PML   = kwargs["R_PML"]   if "R_PML"   in kwargs else 11           # Outer radius PML in cm
        self.gdim    = kwargs["gdim"]    if "gdim"    in kwargs else 2            # Geometric dimension of the mesh
        self.h       = kwargs["h"]       if "h"       in kwargs else self.r0/2**3 # Characteristic length of mesh elements
        self.quad    = kwargs["quad"]    if "quad"    in kwargs else False        # If False, triangular mesh, if True quadrilateral


    def __call__(self):
        gmsh.initialize()

        # Define geometry of outer domain
        gmsh.model.occ.addCircle(0, 0, 0, self.R_PML, tag=1)
        gmsh.model.occ.addCircle(0, 0, 0, self.R_tilde, tag=2)
        gmsh.model.occ.addCurveLoop([1], tag=1)
        gmsh.model.occ.addCurveLoop([2], tag=2)
        PML_1 = gmsh.model.occ.addPlaneSurface([1, 2])
        gmsh.model.occ.synchronize()

        gmsh.model.occ.addCircle(0, 0, 0, self.R, tag=3)
        gmsh.model.occ.addCurveLoop([3], tag=3)
        PML_2 = gmsh.model.occ.addPlaneSurface([2, 3])
        gmsh.model.occ.synchronize()

        gmsh.model.occ.addCircle(0, 0, 0, self.r0, tag=4)
        gmsh.model.occ.addCurveLoop([4], tag=4)
        Medium = gmsh.model.occ.addPlaneSurface([3, 4])
        gmsh.model.occ.synchronize()

        # Define geometry of inner domain
        gmsh.model.occ.addCircle(0, 0, 0, self.r0/4, tag=5) # Radius at which coordinate transformation starts
        gmsh.model.occ.addCurveLoop([5], tag=5)
        Object_1 = gmsh.model.occ.addPlaneSurface([4, 5])
        gmsh.model.occ.synchronize()

        Object_2 = gmsh.model.occ.addDisk(0, 0, 0, self.r0/4, self.r0/4)
        gmsh.model.occ.synchronize()

        # Resolve all boundaries
        whole_domain = gmsh.model.occ.fragment([(self.gdim, Object_1)],[(self.gdim, Object_2),(self.gdim, Medium),(self.gdim, PML_2),(self.gdim, PML_1)])
        gmsh.model.occ.synchronize()

        # We use the following markers for the domains:
        # PML_1:    1
        # PML_2:    2
        # Medium:   3
        # Object_1: 4
        # Object_2: 5

        # We use the following markers for the boundaries:
        # Boundary at R_PML:   6
        # Boundary at R_tilde: 7
        # Boundary at R:       8
        # Boundary at r0:      9
        # Boundary at r0/4:    10

        visited_boundaries = []
        for domain in whole_domain[0]:
            mass = gmsh.model.occ.getMass(domain[0], domain[1])
            # Identify PML_1, PML_2, Medium, Object_1 and Object_2 by their masses
            if np.isclose(mass, np.pi*self.R_PML**2 - np.pi*self.R_tilde**2):
                gmsh.model.addPhysicalGroup(domain[0], [domain[1]], tag=1)
            elif np.isclose(mass, np.pi*self.R_tilde**2 - np.pi*self.R**2):
                gmsh.model.addPhysicalGroup(domain[0], [domain[1]], tag=2)
            elif np.isclose(mass, np.pi*self.R**2 - np.pi*self.r0**2):
                gmsh.model.addPhysicalGroup(domain[0], [domain[1]], tag=3)
            elif np.isclose(mass, np.pi*self.r0**2 - np.pi*(self.r0/4)**2):
                gmsh.model.addPhysicalGroup(domain[0], [domain[1]], tag=4)
            elif np.isclose(mass, np.pi*(self.r0/4)**2):
                gmsh.model.addPhysicalGroup(domain[0], [domain[1]], tag=5)

            boundaries = gmsh.model.getBoundary([domain], oriented=False)
            for boundary in boundaries:
                if boundary not in visited_boundaries:
                    mass_boundary = gmsh.model.occ.getMass(boundary[0], boundary[1])
                    if np.isclose(mass_boundary, 2*np.pi*self.R_PML):
                        gmsh.model.addPhysicalGroup(boundary[0], [boundary[1]], tag=6)
                    elif np.isclose(mass_boundary, 2*np.pi*self.R_tilde):
                        gmsh.model.addPhysicalGroup(boundary[0], [boundary[1]], tag=7)
                    elif np.isclose(mass_boundary, 2*np.pi*self.R):
                        gmsh.model.addPhysicalGroup(boundary[0], [boundary[1]], tag=8)
                    elif np.isclose(mass_boundary, 2*np.pi*self.r0):
                        gmsh.model.addPhysicalGroup(boundary[0], [boundary[1]], tag=9)
                    elif np.isclose(mass_boundary, 2*np.pi*self.r0/4):
                        gmsh.model.addPhysicalGroup(boundary[0], [boundary[1]], tag=10)
                visited_boundaries.append(boundary)

        # Set characteristic length of mesh elements
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", self.h)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.h)
        if self.quad == True:
            gmsh.option.setNumber('Mesh.RecombineAll', 1)

        # Generate the mesh
        gmsh.model.mesh.generate(self.gdim)

        # Create dolfinx mesh saving cell and facet tags
        domain,ct,ft = io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=self.gdim)
        gmsh.finalize()
        self.domain = domain
        self.ct     = ct
        self.ft     = ft

if __name__ == "__main__":
    gen = Generate_Mesh()
    gen()
    with io.XDMFFile(MPI.COMM_SELF, "mesh.XDMF", "w") as xdmf:
        xdmf.write_mesh(gen.domain)