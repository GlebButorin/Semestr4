import meshio
import numpy as np

mesh = meshio.read("build/Mountain.msh")
meshio.write("Mountain.xdmf", mesh)