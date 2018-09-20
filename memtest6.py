import underworld as uw
import glucifer
import time
mesh = uw.mesh.FeMesh_Cartesian(elementRes = (16, 16))
meshVar = uw.mesh.MeshVariable(mesh=mesh, nodeDofCount=1)
meshHandle = mesh.save("mesh.h5")
fig = glucifer.Figure()
fig.append(glucifer.objects.Surface(mesh, meshVar))
step = 0
while True:
    step += 1
    t = time.clock()
    if step % 10 == 0 and uw.rank() == 0:
        print step, t
    meshVar.save("meshVar.h5", meshHandle)
    fig.save("fig.png")

