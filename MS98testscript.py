
# coding: utf-8

# In[ ]:

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import math
import underworld as uw
from underworld import function as fn
import h5py
import glucifer
import csv
import mpi4py
import os
import time
import planetengine
from planetengine import utilities
from planetengine import physics

comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#%matplotlib inline
CoordFn = uw.function.input()
depthFn = 1. - CoordFn[1]


# In[ ]:

MODEL = utilities.Collection()
MODEL.SetVals({
    'OPTIONS': utilities.Collection(),
    'PARAMETERS': utilities.Collection(),
    'MESHES': utilities.Collection(),
    'SWARMS': utilities.Collection(),
    'FUNCTIONS': utilities.Collection(),
    'SYSTEMS': utilities.Collection(),
    'DATA': utilities.Collection(),
    'MISC': utilities.Collection(),
    'LOG': utilities.Collection()
    })


# In[ ]:

# For ease of assignment,
# aliases are set

OPTIONS = MODEL.OPTIONS
PARAMETERS = MODEL.PARAMETERS
MESHES = MODEL.MESHES
SWARMS = MODEL.SWARMS
FUNCTIONS = MODEL.FUNCTIONS
SYSTEMS = MODEL.SYSTEMS
DATA = MODEL.DATA
MISC = MODEL.MISC


# In[ ]:

OPTIONS.SetVals({
    'projectname': 'Testing', #'MS98Ra1e7res64eta3e4tauone1e7taunought3e5', #'ParaMS98Ra1e7res32',

    'showfigquality': 4,
    'savefigquality': 8,
    
    'numpyAnalysis': False,
    'dataRes': 100,
    
    'analyseFromLoadedState': False
    })

#OPTIONS.SetVal('modelRunCondition', utilities.RuntimeCondition.TimeInterval(0.05, False))
OPTIONS.SetVal('modelRunCondition', utilities.RuntimeCondition.AfterStep(1000, False))
OPTIONS.SetVal('updateDataCondition', utilities.RuntimeCondition.StepInterval(10, True))
OPTIONS.SetVal('printDataCondition', utilities.RuntimeCondition.ConstantBool(True))
OPTIONS.SetVal('saveDataCondition', utilities.RuntimeCondition.StepInterval(100, True))
OPTIONS.SetVal('saveFigsCondition', utilities.RuntimeCondition.ConstantBool(False))
OPTIONS.SetVal('showFigsCondition', utilities.RuntimeCondition.ConstantBool(False))
OPTIONS.SetVal('saveStateCondition', utilities.RuntimeCondition.CombinedCondition('any',
    ((utilities.RuntimeCondition.StepInterval(1000, True), utilities.RuntimeCondition.UponCompletion(True)))))

# In[ ]:

PARAMETERS.SetVals({
    'Ra':1e7,
    'initialH':0.,
    'initialWatts':0.,
    'timeDilation':1.,
    'HDecay':0.,
    'rho0':1.,
    'rho0cont':1.,
    'diffusivity':1.,
    'lengthScale':2900e3,

    'z_hat':(0., 1.),

    'eta0':1.,
    'viscRatio':1.,
    'minMantleVisc':1.,
    'minContVisc':1.,
    'surfEta':3e4,
    'yielding':True,
    'refYieldStress':fn.misc.constant(4e5),
    'contStressRatio':1e1,
    'frictionCoefficient':1.,
    'isoviscous':False,

    'tau0': 15e5,
    'tau1': 1e7,

    'maxTemp':1.,
    'minTemp':0.,
    'refTemp':1.,

    'contY':0.9,
    'cont':False,

    # Non-physicsy stuff

    'aspect':1,
    'res':64,
    'particlesPerCell':12,
    'presolve':False,
    'randomSeed':1066,

    'boxdimension':1.,

    'deformMesh':False
    })

# Self-referencing parameters:
PARAMETERS.SetVals({
    'rho0cont': PARAMETERS.rho0 * 1e1,
    'alpha': PARAMETERS.Ra,
    'alphaCont': PARAMETERS.Ra * 1e1, 
    'surfTemp': PARAMETERS.minTemp,
    'baseTemp': PARAMETERS.maxTemp,
    'boundaryLayerThickness': 1.,
    'minX': 0.,
    'maxX': PARAMETERS.aspect * PARAMETERS.boxdimension,
    'minY': 0.,
    'maxY': PARAMETERS.boxdimension
    })

PARAMETERS.SetVals({
    'deltaT': PARAMETERS.baseTemp - PARAMETERS.surfTemp
    })


# In[ ]:

MESHES.SetVals({
    'mesh': uw.mesh.FeMesh_Cartesian(
        elementType = ("Q1/dQ0"), 
        elementRes  = (int(PARAMETERS.res*PARAMETERS.aspect), PARAMETERS.res), 
        minCoord    = (PARAMETERS.minX, PARAMETERS.minY), 
        maxCoord    = (PARAMETERS.maxX, PARAMETERS.maxY),
        #periodic    = [True, False]
        )
    })

if PARAMETERS.deformMesh:
    with MESHES.mesh.deform_mesh():
        for index, coord in enumerate(MESHES.mesh.data):
            if 0. < MESHES.mesh.data[index][1] < 1.0:
                MESHES.mesh.data[index][1] = MESHES.mesh.data[index][1]**(0.5**(coord[1]-0.1))

MESHES.SetVals({
    'temperatureField': uw.mesh.MeshVariable(
        mesh = MESHES.mesh,
        nodeDofCount = 1
        ),
    'temperatureDotField': uw.mesh.MeshVariable(
        mesh = MESHES.mesh,
        nodeDofCount = 1
        ),
    'pressureField': uw.mesh.MeshVariable(
        mesh = MESHES.mesh.subMesh,
        nodeDofCount = 1
        ),
    'velocityField': uw.mesh.MeshVariable(
        mesh = MESHES.mesh,
        nodeDofCount = 2
        ),
    'HField': uw.mesh.MeshVariable(
        mesh = MESHES.mesh,
        nodeDofCount = 1
        ),
    'devStressField': uw.mesh.MeshVariable(
        mesh = MESHES.mesh,
        nodeDofCount = 3
        )
    })


# In[ ]:

# Setting up boundary conditions

BottomWall = MESHES.mesh.specialSets["MinJ_VertexSet"] 
TopWall = MESHES.mesh.specialSets["MaxJ_VertexSet"] 
LeftWall = MESHES.mesh.specialSets["MinI_VertexSet"] 
RightWall = MESHES.mesh.specialSets["MaxI_VertexSet"]
IWalls = LeftWall + RightWall
JWalls = TopWall + BottomWall
AllWalls = IWalls + JWalls

tempBC = uw.conditions.DirichletCondition(
    variable = MESHES.temperatureField, 
    indexSetsPerDof = (JWalls)
    )

#periodicBC = uw.conditions.DirichletCondition(
    #variable = MESHES.velocityField, 
    #indexSetsPerDof = (BottomWall, JWalls)
    #)

freeslipBC = uw.conditions.DirichletCondition(
    variable = MESHES.velocityField, 
    indexSetsPerDof = (IWalls, JWalls)
    )


# In[ ]:

# Setting up advector

SWARMS.SetVal('swarm', uw.swarm.Swarm(mesh = MESHES.mesh))

SWARMS.swarm.populate_using_layout(
    layout = uw.swarm.layouts.GlobalSpaceFillerLayout(
        swarm = SWARMS.swarm,
        particlesPerCell = PARAMETERS.particlesPerCell
        )
    )

SWARMS.SetVal('materialVar', SWARMS.swarm.add_variable(dataType = "int", count = 1))
SWARMS.materialVar.data[:] = 0

FUNCTIONS.SetVal('strainRateFn', fn.tensor.symmetric(MESHES.velocityField.fn_gradient))
FUNCTIONS.SetVal('secInv', fn.tensor.second_invariant(FUNCTIONS.strainRateFn))

FUNCTIONS.SetVals({
    'yieldStressFn': PARAMETERS.tau0 + (PARAMETERS.tau1 * utilities.depthFn),
    'secInv': fn.tensor.second_invariant(fn.tensor.symmetric(MESHES.velocityField.fn_gradient))
    })

FUNCTIONS.SetVals({
    'initialTempFn': utilities.InitialConditions.NoisyGradient(
        MESHES.temperatureField,
        gradient = 10.,
        smoothness = 10,
        randomSeed = PARAMETERS.randomSeed,
        range = (PARAMETERS.minTemp, PARAMETERS.maxTemp)
        ),
    'initialHFn': PARAMETERS.initialH,
    'densityFn': PARAMETERS.Ra * MESHES.temperatureField,
    'creepViscFn': fn.math.exp(-1. * np.log(PARAMETERS.surfEta) * (MESHES.temperatureField - 1.)), # was FKtemp1
    #'creepViscFn': PARAMETERS.eta0 * fn.math.pow(PARAMETERS.surfEta, 1. - MESHES.temperatureField),
    'plasticViscFn': FUNCTIONS.yieldStressFn / (2. * FUNCTIONS.secInv + 1e-18), # was plasticvisc1
    'timescale': physics.Build_TimescaleGyFn_1(PARAMETERS.lengthScale, PARAMETERS.Ra, PARAMETERS.diffusivity)
    })

FUNCTIONS.SetVals({
    'viscosityFn': fn.branching.map(
        fn_key = SWARMS.materialVar,
        mapping = {
            0: utilities.CapValue(
                fn.misc.min(FUNCTIONS.creepViscFn, FUNCTIONS.plasticViscFn),
                (PARAMETERS.eta0, PARAMETERS.eta0 * PARAMETERS.surfEta)
                )
            }
        ),
    'yieldFn': uw.function.branching.conditional([
        (FUNCTIONS.creepViscFn < FUNCTIONS.plasticViscFn, 0.),
        (True, 1.)
        ])
    })

FUNCTIONS.SetVal('stressFn', 2. * FUNCTIONS.viscosityFn * FUNCTIONS.strainRateFn)
FUNCTIONS.SetVal('devStressFn', fn.tensor.deviatoric(FUNCTIONS.stressFn))
FUNCTIONS.SetVal('devStress2ndInv', fn.tensor.second_invariant(FUNCTIONS.devStressFn))

MESHES.temperatureField.load("VERYIMPORTANT64.h5", interpolate=True)

SYSTEMS.SetVals({
    'population_control': uw.swarm.PopulationControl(
        SWARMS.swarm,
        aggressive = True,
        splitThreshold = 0.15,
        maxDeletions = 2,
        maxSplits = 10,
        particlesPerCell = PARAMETERS.particlesPerCell
        ),
    'advDiff':  uw.systems.AdvectionDiffusion( 
        MESHES.temperatureField, 
        MESHES.temperatureDotField, 
        MESHES.velocityField, 
        fn_diffusivity = PARAMETERS.diffusivity,
        fn_sourceTerm = MESHES.HField,
        conditions = [tempBC]
        ),
    'advector': uw.systems.SwarmAdvector(
        swarm = SWARMS.swarm,
        velocityField = MESHES.velocityField,
        order = 2
        ),
    'stokes': uw.systems.Stokes(
        velocityField = MESHES.velocityField, 
        pressureField = MESHES.pressureField,
        voronoi_swarm = SWARMS.swarm, #(not sure if this is necessary or not)
        conditions = [freeslipBC,], #[periodicBC,],
        fn_viscosity = FUNCTIONS.viscosityFn, 
        fn_bodyforce = FUNCTIONS.densityFn * PARAMETERS.z_hat
        ),
    })

SYSTEMS.SetVal('solver', uw.systems.Solver(SYSTEMS.stokes))

utilities.Run(MODEL, startStep = 0)

#utilities.MakeDataFromLoadState(MODEL, (1, 80000), figures = False)

#for integer in range(1, 8):
    #index = integer*10000
    #utilities.MakeDataFromLoadState(MODEL, (index, index+1), data = False)

#for integer in range(1, 70):
    #index = integer*1000
    #utilities.MakeDataFromLoadState(MODEL, (index, index+1), data = False)

#for integer in range(300, 400):
    #index = integer*100
    #utilities.MakeDataFromLoadState(MODEL, (index, index+1), data = False)


