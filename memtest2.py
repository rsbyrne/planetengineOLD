import underworld as uw
from underworld import function as fn
import math
import glucifer

mesh = uw.mesh.FeMesh_Cartesian(elementRes = (256, 256))

velocityField = uw.mesh.MeshVariable(mesh=mesh, nodeDofCount=2)
pressureField = uw.mesh.MeshVariable(mesh=mesh.subMesh, nodeDofCount=1)
temperatureField = uw.mesh.MeshVariable(mesh=mesh, nodeDofCount=1)
temperatureDotField = uw.mesh.MeshVariable(mesh=mesh, nodeDofCount=1)

velocityField.data[:] = [0.,0.]
pressureField.data[:] = 0.

pertStrength = 0.2
tempMax = 1.
tempMin = 0.
deltaTemp = tempMax - tempMin
boxHeight = 1.0
for index, coord in enumerate(mesh.data):
    pertCoeff = math.cos(math.pi * coord[0]) * math.sin(math.pi * coord[1])
    temperatureField.data[index] = tempMin + deltaTemp*(boxHeight - coord[1]) + pertStrength * pertCoeff
    temperatureField.data[index] = max(tempMin, min(tempMax, temperatureField.data[index]))
for index in mesh.specialSets["MinJ_VertexSet"]:
    temperatureField.data[index] = tempMax
for index in mesh.specialSets["MaxJ_VertexSet"]:
    temperatureField.data[index] = tempMin

iWalls = mesh.specialSets["MinI_VertexSet"] + mesh.specialSets["MaxI_VertexSet"]
jWalls = mesh.specialSets["MinJ_VertexSet"] + mesh.specialSets["MaxJ_VertexSet"]
freeslipBC = uw.conditions.DirichletCondition(variable = velocityField, indexSetsPerDof = (iWalls,jWalls))
tempBC = uw.conditions.DirichletCondition(variable = temperatureField, indexSetsPerDof = (jWalls,))

buoyancyFn = (0.0, 1.0) * fn.misc.constant(1e9) * temperatureField

stokes = uw.systems.Stokes(
    velocityField = velocityField,
    pressureField = pressureField,
    conditions = [freeslipBC,],
    fn_viscosity = fn.misc.constant(1.),
    fn_bodyforce = buoyancyFn
    )

solver = uw.systems.Solver(stokes)

advDiff = uw.systems.AdvectionDiffusion(
    phiField = temperatureField,
    phiDotField = temperatureDotField,
    velocityField = velocityField,
    fn_diffusivity = fn.misc.constant(1.),
    conditions = [tempBC,]
    )

figtemp = glucifer.Figure()
figtemp.append(glucifer.objects.Surface(mesh, temperatureField))

for step in range(int(1e5)):
    solver.solve(nonLinearIterate=True)
    dt = advDiff.get_max_dt()
    advDiff.integrate(dt)
    if step % 10 == 0:
        figtemp.save("figtemp.png")

meshHandle = mesh.save('mesh256.h5')
temperatureField.save('isoviscousRa1e9res256', meshHandle)
