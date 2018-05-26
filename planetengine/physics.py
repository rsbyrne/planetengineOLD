# physics.py
# part of the planetengine package

# this is a test of git commits

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

import utilities

comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

CoordFn = uw.function.input()
depthFn = 1. - CoordFn[1]

def FindSurfaceFlux(mesh, whichwall, scalarField):
    wall = utilities.WhichWall(whichwall)
    flux = uw.utils.Integral(
        fn = scalarField.fn_gradient[1],
        mesh = mesh,
        integrationType = "surface",
        surfaceIndexSet = wall
        )
    return flux.evaluate()[0]

def FindFieldTotal(mesh, scalarField):
    total = uw.utils.Integral(
        fn = scalarField,
        mesh = mesh,
        integrationType = "volume"
        )
    return total.evaluate()[0]

def FindFieldAverage(mesh, scalarField):
    average = FindFieldTotal(mesh, scalarField) / utilities.FindVolume(mesh)
    return average

def FindVRMS(velocityField, mesh):
    # **RMS velocity**
    #
    # The root mean squared velocity is defined by intergrating over the entire simulation domain via
    #
    # \\[
    # \begin{aligned}
    # v_{rms}  =  \sqrt{ \frac{ \int_V (\mathbf{v}.\mathbf{v}) dV } {\int_V dV} }
    # \end{aligned}
    # \\]
    #
    # where $V$ denotes the volume of the box.

    # In[12]:

    intVdotV = uw.utils.Integral(
        fn.math.dot(velocityField, velocityField),
        mesh
        )
    vrms = math.sqrt(intVdotV.evaluate()[0]) / utilities.FindVolume(mesh)
    return vrms

def FindNusseltNumber(temperatureField, mesh):
    # The Nusselt number is the ratio between
    # convective and conductive heat transfer.
    # It is defined as
    # the top vertical surface gradient
    # divided by the basal temperature.

    topTempSurfGradIntegral = FindSurfaceFlux(
        mesh, 'Top', temperatureField
        )
    basalTempIntegral = uw.utils.Integral(
        fn = temperatureField,
        mesh = mesh,
        integrationType = 'surface',
        surfaceIndexSet = utilities.WhichWall('Bottom')
        )
    NuFn = topTempSurfGradIntegral / basalTempIntegral
    Nu = NuFn.evaluate_global()[0]
    return Nu

def FindSurfVRMS(mesh, vectorField, whichwall):
    wall = utilities.WhichWall(whichwall)
    vectorFieldSqIntegral = uw.utils.Integral(
        fn = fn.math.dot(vectorField, vectorField),
        mesh = mesh,
        integrationType = 'surface',
        surfaceIndexSet = wall
        )
    meshIntegral = uw.utils.Integral(
        fn = 1.,
        mesh = mesh,
        integrationType = 'surface',
        surfaceIndexSet = wall
        )
    rms = math.sqrt(vectorFieldSqIntegral.evaluate()[0] / meshIntegral.evaluate()[0])
    return rms

def FindSurfaceStresses(devStressField, mesh, whichwall):
    horizStresses, vertStresses = devStressField[0], devStressField[1]
    wall = utilities.WhichWall(whichwall)
    hSt = utilities.EvaluateScalarMeshVarOnSurface(
        mesh, vertStresses, whichwall, 1000
        )
    vSt = utilities.EvaluateScalarMeshVarOnSurface(
        mesh, vertStresses, whichwall, 1000
        )
    hStRMS = FindSurfVRMS(mesh, horizStresses, whichwall)
    vStRMS = FindSurfVRMS(mesh, vertStresses, whichwall)
    hStMax = math.max(hSt)
    vStMax = math.max(vSt)
    outTuple = (hSt, vSt, hStRMS, vStRMS, hStMax, vStMax)
    return outTuple

def npFindNusseltNumber(maxVertCoord, npTemperatureField):
    TopInt = sum(np.gradient(npTemperatureField)[1][0])
    BottomInt = sum(npTemperatureField[-1])
    Nu = - maxVertCoord * TopInt / BottomInt
    return Nu

def GetStressStrain(viscosityFn, velocityField):

    # **Calculate stress values for benchmark comparison**
    #
    #
    # Determine stress field for whole box in dimensionless units (King 2009)
    # \begin{equation}
    # \tau_{ij} = \eta \frac{1}{2} \left[ \frac{\partial v_j}{\partial x_i} + \frac{\partial v_i}{\partial x_j}\right]
    # \end{equation}
    # which for vertical normal stress becomes
    # \begin{equation}
    # \tau_{zz} = \eta \frac{1}{2} \left[ \frac{\partial v_z}{\partial z} + \frac{\partial v_z}{\partial z}\right] = \eta \frac{\partial v_z}{\partial z}
    # \end{equation}
    # which is implemented for the whole box in the functions defined below.

    # ### Deviatoric stress
    #
    # The deviatoric stress is computed from the constitutive law based on the viscosity that
    # results from the solution to the non-linear Stokes equation.
    #
    # **Note:** the deviatoric stress is defined in terms of functions we have defined already
    # but the value will be meaningless if the viscosityFn is modified in any way after the
    # solve is complete because evaluation is made only when the values at particular points are needed.

    # In[14]:

    strainRateFn = fn.tensor.symmetric(velocityField.fn_gradient)
    secInv = fn.tensor.second_invariant(strainRateFn)
    stressFn = 2. * viscosityFn * strainRateFn
    devStressFn = fn.tensor.deviatoric(stressFn)
    devStress2ndInv = fn.tensor.second_invariant(devStressFn)

    return strainRateFn, secInv, stressFn, devStressFn, devStress2ndInv

def FindSurfaceStresses(velocityField, viscosityFn, mesh, swarm, devStressField):

    # TESTING!!!
    # 'stressField' input currently being ignored

    strainRateFn, secInv, stressFn, devStressFn, devStress2ndInv = GetStressStrain(viscosityFn, velocityField)

    projector = uw.utils.MeshVariable_Projection(devStressField, stressFn, type=0)
    projector.solve()

    horizStressTop = utilities.EvaluateScalarMeshVarOnSurface(
        mesh,
        devStressField[0],
        'Top',
        100
        )

    vertStressTop = utilities.EvaluateScalarMeshVarOnSurface(
        mesh,
        devStressField[1],
        'Top',
        100
        )

    return horizStressTop, vertStressTop

def FindSurfaceHorizontalVelocity(mesh, velocityField, n):
    surfVel = utilities.EvaluateScalarMeshVarOnSurface(mesh, velocityField[0], 'Top', n)
    return surfVel

def Build_FKTempVisc_1 (temperatureField, PARAMETERS):
    #Frank-Kamenetskii Temperature-Dependent Rheology

    eta0 = PARAMETERS.eta0
    surfEta = PARAMETERS.surfEta
    refTemp = PARAMETERS.refTemp

    tempViscosityFn = eta0 * uw.function.math.exp(
        np.log(surfEta) * (refTemp - temperatureField) / (refTemp**2)
        )

    return tempViscosityFn

def Build_FKTempVisc_2 (temperatureField, PARAMETERS):
    # (equivalent to MS98)

    eta0 = PARAMETERS.eta0
    surfEta = PARAMETERS.surfEta
    refTemp = PARAMETERS.refTemp

    tempViscosityFn = eta0 * surfEta * uw.function.math.exp(
        -1. * np.log(surfEta) * temperatureField
        )

    return tempViscosityFn

def Build_BlankenVisc_1 (eta0, refTemp, temperatureField):
    tempViscosityFn = eta0 * fn.math.exp(- 6.9077/(refTemp - temperatureField))
    return tempViscosityFn

def Build_PlasticVisc_1 (velocityField, PARAMETERS):
    # This is based off Lenardic et al 2008 - A climate induced transition.
    # It outputs an effective viscosity for a plastically deforming material.

	# ### Viscoplastic rheology
    #
    # Lenardic et al 2008 - A climate induced transition
    #
    # Stress is nondimensionalised with $\frac{d^2}{\kappa \mu_0} $ where d is the mantle depth, $ \kappa $ is the mantle thermal diffusivity, and $\mu_0$ is the reference viscosity (defined at the top surface of the mantle).
    #
    # Nondimensional yield stress is:
    #
    # $ \tau_y = \tau_{y0} + \tau_{yz}Z $
    #
    # $\tau_y$ is the depth-dependent yield stress, $\tau_{y0}$ is a prescribed surface value, and Z is non-dimensional depth.
    #
    # $\tau_{yz}$ is the depth-dependent term. It is non-dimensionalised by $\frac{f_cRa_0}{\alpha\Delta T} $ where $Ra_0$ is the mantle Rayleigh number at the reference viscosity, $\Delta T $ is the temperature drop across the mantle, $f_c$ is a friction coefficient, and $\alpha$ is the coefficient of thermal expansion.

    # In[ ]:

    Ra = PARAMETERS.Ra
    frictionCoefficient = PARAMETERS.frictionCoefficient
    boundaryLayerThickness = PARAMETERS.boundaryLayerThickness
    diffusivity = PARAMETERS.diffusivity
    eta0 = PARAMETERS.eta0
    alpha = PARAMETERS.alpha
    deltaT = PARAMETERS.deltaT
    refYieldStress = PARAMETERS.refYieldStress

    strainRateFn, secInv, stressFn, devStressFn, devStress2ndInv = GetStressStrain(1., velocityField)

    stressNonDimFactor = boundaryLayerThickness**2 / diffusivity / eta0
    depthyieldNonDimFactor = frictionCoefficient * Ra / alpha / deltaT

    yieldStress = refYieldStress / stressNonDimFactor
    depthYieldStress = yieldStress / depthyieldNonDimFactor

    yieldStressFn = refYieldStress + depthYieldStress * utilities.depthFn
    yieldViscosityFn = yieldStressFn / (secInv + 1.0e-18)

    plasticViscFn = yieldViscosityFn

    return plasticViscFn

def Build_PlasticVisc_2(tau0, tau1, velocityField):
    # as used in MS98
    strainRateFn, secInv, stressFn, devStressFn, devStress2ndInv = GetStressStrain(1., velocityField)
    yieldStressFn = tau0 + (tau1 * utilities.depthFn)
    plasticViscFn = yieldStressFn / (secInv + 1.0e-18)
    return plasticViscFn

def Build_IsoviscousFn_1 (refVisc):
    return refVisc

def Build_LinearInitialTempFn_1(PARAMETERS):

    smoothness = 1
    randomSeed = PARAMETERS.randomSeed
    depthFn = utilities.depthFn
    tempGradient = PARAMETERS.tempGradient
    baseTemp = PARAMETERS.baseTemp
    surfTemp = PARAMETERS.surfTemp
    minTemp = PARAMETERS.minTemp
    maxTemp = PARAMETERS.maxTemp

    tempGradFn = utilities.randomise(smoothness, randomSeed) * (depthFn * tempGradient * (baseTemp - surfTemp) + surfTemp)
    initialTempFn = uw.function.branching.conditional([
        (depthFn < 0.05, surfTemp),
        (depthFn > 0.95, baseTemp),
        (tempGradFn < minTemp, minTemp), # if this, that
        (tempGradFn > maxTemp , maxTemp), # otherwise, if this, this other thing
        (True, tempGradFn) # otherwise, this one
        ])

    return initialTempFn

def Build_BlankenInitialTempFn_1(mesh, temperatureField, PARAMETERS):

    pertStrength = PARAMETERS.pertStrength
    minTemp = PARAMETERS.minTemp
    maxTemp = PARAMETERS.maxTemp
    deltaTemp = PARAMETERS.deltaT
    coord = utilities.CoordFn
    maxX = PARAMETERS.maxX

    temperatureField.data[:] = 0.

    for index, coord in enumerate(mesh.data):
        pertCoeff = math.cos( math.pi * coord[0]/maxX ) * math.sin( math.pi * coord[1]/maxX )
        temperatureField.data[index] = minTemp + deltaTemp*(maxX - coord[1]) + pertStrength * pertCoeff
        temperatureField.data[index] = max(minTemp, min(maxTemp, temperatureField.data[index]))

def Build_TimescaleGyFn_1(lengthScale, Ra, diffusivity):

    t_seconds = lengthScale**2. * Ra * diffusivity
    seconds_per_Gy = 31556926000000000.
    t_Gy = t_seconds / seconds_per_Gy

    return t_Gy
