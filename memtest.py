import underworld as uw
mesh = uw.mesh.FeMesh_Cartesian(elementRes = (16, 16))
while True: uw.utils.Integral(mesh = mesh, fn = 1.)#.evaluate()
