import underworld as uw
from underworld import function as fn
import math
import numpy as np
import glucifer

#mesh = uw.mesh.FeMesh_Cartesian(elementRes = (16, 16))
mesh = uw.mesh.FeMesh_Cartesian()

velocityField = uw.mesh.MeshVariable(mesh=mesh, nodeDofCount=2)
pressureField = uw.mesh.MeshVariable(mesh=mesh.subMesh, nodeDofCount=1)
temperatureField = uw.mesh.MeshVariable(mesh=mesh, nodeDofCount=1)
temperatureDotField = uw.mesh.MeshVariable(mesh=mesh, nodeDofCount=1)

velocityField.data[:] = [0.,0.]
pressureField.data[:] = 0.

def initialTempFn(temperatureField, mesh, tempRange = (0.,1.), pertStrength = 0.2):
    temperatureField.data[:] = 0.
    tempMin, tempMax = tempRange
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
    if uw.rank() == 0: print "Sinusoidal initial temperature function applied."

initialTempFn(temperatureField, mesh)
#temperatureField.load("isoviscousRa1e7res256x128.h5")

iWalls = mesh.specialSets["MinI_VertexSet"] + mesh.specialSets["MaxI_VertexSet"]
jWalls = mesh.specialSets["MinJ_VertexSet"] + mesh.specialSets["MaxJ_VertexSet"]
freeslipBC = uw.conditions.DirichletCondition(variable = velocityField, indexSetsPerDof = (iWalls,jWalls))
tempBC = uw.conditions.DirichletCondition(variable = temperatureField, indexSetsPerDof = (jWalls,))

buoyancyFn = (0.0, 1.0) * fn.misc.constant(1e1) * temperatureField
viscosityFn = fn.misc.constant(1.)

stokes = uw.systems.Stokes(
    velocityField = velocityField,
    pressureField = pressureField,
    conditions = [freeslipBC,],
    fn_viscosity = viscosityFn,
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

#meshHandle = mesh.save("mesh16.h5")

#figTemp = glucifer.Figure()
#figTemp.append(glucifer.objects.Surface(mesh, temperatureField))

#t, step = 0., 0

while True:

    solver.solve(nonLinearIterate=False)
    dt = advDiff.get_max_dt()
    advDiff.integrate(dt)
    #step += 1
    #t += dt

    #if step % 1000 == 0:
        #print "-" * 20
        #print str(step), str(t)
        #print "-" * 20
        #figTemp.save("figtemp.png")
        #temperatureField.save("tempfield.h5", meshHandle)

