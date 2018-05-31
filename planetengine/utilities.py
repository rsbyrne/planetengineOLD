# utilities.py

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
import shutil

import physics

# This is a test to learn the ropes of git commits!
# this is another test

comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

CoordFn = uw.function.input()
depthFn = 1. - CoordFn[1]

class Collection():
    def __init__(self):
        self.selfdict = {}
    def __bunch__(self, adict):
        self.__dict__.update(adict)
    def SetVal(self, key, val):
        self.selfdict[key] = val
        self.__bunch__(self.selfdict)
    def SetVals(self, dict):
        for key in dict.keys():
            self.SetVal(key, dict[key])
    def ClearAttr(self):
        for key in self.selfdict.keys():
            delattr(self, key)
        self.selfdict = {}
    def SetDict(self, dict):
        self.ClearAttr()
        self.SetVals(dict)
    def Out(self):
        outstring = ""
        for key in self.selfdict.keys():
            thing = self.selfdict[key]
            if isinstance(thing, self):
                thing.Out()
            else:
                outstring += key + ": " + thing
        return outstring

class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

class RuntimeCondition():

    class CombinedCondition():
        def __init__(self, arg, tupleOfFunctions):
            self.tupleOfFunctions = tupleOfFunctions
            self.arg = arg
        def evaluate(self, MODEL):
            boolList = [function.evaluate(MODEL) for function in self.tupleOfFunctions]
            antiBoolList = [not x for x in boolList]
            if self.arg == 'all':
                return all(boolList)
            elif self.arg == 'none':
                return all(antiBoolList)
            elif self.arg == 'any':
                return not all(antiBoolList)
            else:
                print "CombinedCondition argument not recognised"

    class UponCompletion():
        def __init__(self, TrueOrFalse):
            self.TrueOrFalse = TrueOrFalse
        def evaluate(self, MODEL):
            if MODEL.MISC.modelrunComplete:
                return self.TrueOrFalse
            else:
                return not self.TrueOrFalse

    class ConstantBool():
        def __init__(self, arg):
            self.arg = arg
        def evaluate(self, MODEL):
            return self.arg

    class StepInterval():
        def __init__(self, stepInterval, TrueOrFalse):
            self.stepInterval = stepInterval
            self.TrueOrFalse = TrueOrFalse
        def evaluate(self, MODEL):
            currentStep = MODEL.MISC.currentStep
            if currentStep % self.stepInterval == 0:
                return self.TrueOrFalse
            else:
                return not self.TrueOrFalse

    class TimeInterval():
        def __init__(self, timeInterval, TrueOrFalse):
            self.timeInterval = timeInterval
            self.lastTimeOut = 0.
            self.TrueOrFalse = TrueOrFalse
        def evaluate(self, MODEL):
            currentTime = MODEL.MISC.currentTime
            if currentTime > self.lastTimeOut + self.timeInterval:
                self.lastTimeOut += self.timeInterval
                return self.TrueOrFalse
            else:
                return not self.TrueOrFalse

    class EpochTimeInterval():
        def __init__(self, timeInterval):
            self.timeInterval = timeInterval
            self.lastTimeOut = 0.
        def evaluate(self, MODEL):
            if MODEL.MISC.runningEpochTime > self.lastTimeOut + self.timeInterval:
                self.lastTimeOut += self.timeInterval
                return True
            else:
                return False

    class AfterStep():
        def __init__(self, targetStep, TrueOrFalse):
            self.targetStep = targetStep
            self.TrueOrFalse = TrueOrFalse
        def evaluate(self, MODEL):
            currentStep = MODEL.MISC.currentStep
            if currentStep == self.targetStep:
                return self.TrueOrFalse
            elif currentStep < self.targetStep:
                return not self.TrueOrFalse
            else:
                return self.TrueOrFalse

    class AfterTime():
        def __init__(self, targetTime, TrueOrFalse):
            self.targetTime = targetTime
            self.TrueOrFalse = TrueOrFalse
        def evaluate(self, MODEL):
            currentTime = MODEL.MISC.currentTime
            if currentTime > self.targetTime:
                return self.TrueOrFalse
            else:
                return not self.TrueOrFalse

    class AfterEpochTimeDuration():
        def __init__(self, timeCheck, TrueOrFalse):
            self.timeCheck = timeCheck
            self.TrueOrFalse = TrueOrFalse
        def evaluate(self, MODEL):
            if MODEL.MISC.runningEpochTime < self.timeCheck:
                return not self.TrueOrFalse
            else:
                return self.TrueOrFalse

    class SteadyStateCriterion_1():
        def __init__(self, keytuple, timeHorizon, threshold, TrueOrFalse):
            self.keytuple = keytuple
            self.timeHorizon = timeHorizon
            self.threshold = threshold
            self.TrueOrFalse = TrueOrFalse
        def evaluate(self, MODEL):
            if MODEL.MISC.freshData:
                print "Fresh data is available - checking steady state criterion..."
                isSteady = CheckSteadyState_1(MODEL.DATA, self.keytuple, self.timeHorizon, self.threshold)
                if isSteady:
                    return self.TrueOrFalse
                else:
                    return not self.TrueOrFalse
            else:
                return not self.TrueOrFalse

    class SteadyStateCriterion_2():
        def __init__(self, key, timeHorizon, threshold, TrueOrFalse):
            self.key = key
            self.timeHorizon = timeHorizon
            self.threshold = threshold
            self.TrueOrFalse = TrueOrFalse
        def evaluate(self, MODEL):
            if MODEL.MISC.freshData:
                print "Fresh data is available - checking steady state criterion..."
                isSteady = CheckSteadyState_2(MODEL.DATA, self.key, self.timeHorizon, self.threshold)
                if isSteady:
                    print "Steady state achieved!"
                    return self.TrueOrFalse
                else:
                    print "Steady state not yet achieved."
                    return not self.TrueOrFalse
            else:
                print "No fresh data available."
                return not self.TrueOrFalse

class InitialConditions():

    class NoisyGradient():
        def __init__(self, field, gradient = 1., smoothness = 1, randomSeed = 1066, range = (0., 1.)):
            self.field = field
            self.gradient = gradient
            self.smoothness = smoothness
            self.range = range
            self.randomSeed = randomSeed
        def evaluate(self, mesh):
            #tempGradFn = depthFn * self.gradient * (self.range[1] - self.range[0]) + self.range[0]
            #field.data[:] = CapValue(randomise(self.smoothness, self.randomSeed) * tempGradFn.evaluate(mesh), self.range)
            tempGradFn = depthFn * self.gradient * (self.range[1] - self.range[0]) + self.range[0]
            initialTempFn = uw.function.branching.conditional([
                (depthFn == 0., self.range[0]),
                (depthFn == 1., self.range[1]),
                (tempGradFn < self.range[0], self.range[0]), # if this, that
                (tempGradFn > self.range[1] , self.range[1]), # otherwise, if this, this other thing
                (True, tempGradFn) # otherwise, this one
                ])

            self.field.data[:] = initialTempFn.evaluate(mesh)

            # Introduce some random noise
            np.random.seed(self.randomSeed)
            for i in range(len(mesh.data)):
                yCoord = mesh.data[i][1]
                if 0 < yCoord < 1.:
                    randnum = 0.
                    smoothness = self.smoothness
                    for number in range(smoothness):
                        randnum += 2 * np.random.rand() / smoothness
                    randTemp = self.field.data[i] * randnum
                    if self.range[0] < randTemp < self.range[1]:
                        self.field.data[i] = randTemp

def FindManualDerivative(inputTuple):
    timeList, dataList = inputTuple
    derivativeList = []
    for index in range(len(dataList)):
        if index > 0:
            deltaVal = dataList[index] - dataList[index - 1]
            deltaTime = timeList[index] - timeList[index - 1]
            derivativeVal = deltaVal/deltaTime
            derivativeList.append(derivativeVal)
    return (timeList[1:], derivativeList)

def CheckSteadyState_1(DATA, keytuple, timeHorizon, threshold):
    allSteps = [str(x) for x in sorted([int(y) for y in DATA.selfdict.keys()])]
    if len(allSteps) > 1:
        key1, key2 = keytuple
        lastStep = allSteps[-1]
        currentTime = DATA.selfdict[lastStep]['modeltime']
        currentData1 = DATA.selfdict[lastStep][key1]
        currentData2 = DATA.selfdict[lastStep][key2]
        timeAtHorizon = currentTime - timeHorizon
        criterion = True
        prevTime, prevData1, prevData2 = currentTime, currentData1, currentData2
        checksperformed = 0
        iterable = iter(allSteps[::-1])
        iterable.next()
        for step in iterable:
            stepTime = DATA.selfdict[step]['modeltime']
            stepData1 = DATA.selfdict[step][key1]
            stepData2 = DATA.selfdict[step][key2]
            if criterion == True and stepTime > timeAtHorizon > 0.:
                dev = abs((prevData2 - stepData2) / (prevData1 - stepData1))
                print "Derivative: " , dev
                checksperformed += 1
                criterion = threshold > dev
                prevTime, prevData1, prevData2 = stepTime, stepData1, stepData2
            elif checksperformed >= 3:
                return criterion
            else:
                return False
    else:
        return False

def CheckSteadyState_2(DATA, key, timeHorizon, threshold):
    allSteps = [str(x) for x in sorted([int(y) for y in DATA.selfdict.keys()])]
    if len(allSteps) > 1:
        lastStep = allSteps[-1]
        currentTime = DATA.selfdict[lastStep]['modeltime']
        currentData = DATA.selfdict[lastStep][key]
        timeAtHorizon = currentTime - timeHorizon
        criterion = True
        checksperformed = 0
        iterable = iter(allSteps[::-1])
        iterable.next()
        for step in iterable:
            stepTime = DATA.selfdict[step]['modeltime']
            stepData = DATA.selfdict[step][key]
            if criterion == True and stepTime > timeAtHorizon > 0.:
                checksperformed += 1
                variance = abs(currentData - stepData) / currentData
                print "Variance = ", variance, ", threshold is: ", threshold, " , checks performed: ", checksperformed
                criterion = variance < threshold
            elif checksperformed >= 3:
                return criterion
            else:
                return False
    else:
        return False

def FindVolume(mesh):
    #blah
    volumeIntegral = uw.utils.Integral(
        mesh = mesh,
        fn = 1.
        )
    return volumeIntegral.evaluate()[0]

def randomise(smoothness, randomSeed):
    randnum = 0.
    for number in range(smoothness):
        randnum += 2. * np.random.rand() / smoothness
    return randnum

def UpdateField(field, function):
    projectorStress = uw.utils.MeshVariable_Projection(field, function, type=0)
    projectorStress.solve()

def WhichWall(mesh, whichwall):
    if whichwall == 'Top':
        wall = mesh.specialSets["MaxJ_VertexSet"]
    elif whichwall == 'Bottom':
        wall = mesh.specialSets["MinJ_VertexSet"]
    elif whichwall == 'Left':
        wall = mesh.specialSets["MinI_VertexSet"]
    elif whichwall == 'Right':
        wall = mesh.specialSets["MaxI_VertexSet"]
    return wall

def Numpify(arrRes, arrAspect, uwMesh, uwVar):
    # Takes as input a uw variable (mesh or swarm)
    # and outputs an evenly spaced numpy array
    # of the same aspect ratio
    # but different resolution
    # equally spaced (no mesh deformation).
    #xCoords, yCoords = zip(*uwMesh.data)
    shape = (arrRes, arrAspect * arrRes)
    xRange = np.linspace(uwMesh.minCoord[0], uwMesh. maxCoord[0], arrRes * arrAspect)
    yRange = np.linspace(uwMesh.minCoord[1], uwMesh. maxCoord[1], arrRes)
    coordList = []
    for y in reversed(yRange):
        for x in xRange:
            coordList.append((x,y))
    if isinstance(uwVar, uw.mesh.MeshVariable):
        dataList = []
        for coord in coordList:
            try:
                dataList.append(temperatureField.evaluate(coord)[0][0])
            except:
                dataList.append(1.)
        outArray = np.reshape(np.array(dataList), shape)
        return outArray
    elif isinstance(uwVar, uw.swarm.SwarmVariable):
        print "Can't handle swarmvars yet."
        return None
    else:
        print "uwVar is not of recognised type."
        return None

def EvaluateScalarMeshVarOnSurface(mesh, field, whichwall, n):
    # Take a scalar field and interpolate
    # an n-length numpy array across a specified surface

    if whichwall == 'Top' or whichwall == 'Bottom':
        line = np.linspace(
            mesh.minCoord[0],
            mesh.maxCoord[0],
            n
            )
    elif whichwall == 'Left' or whichwall == 'Right':
        line = np.linspace(
            mesh.minCoord[1],
            mesh.maxCoord[1],
            n
            )

    surfVariable = np.zeros(n)
    for i in range(n):
        if whichwall == 'Top' or whichwall == 'Right':
            surfVariable[i] = field.evaluate_global(
                (line[i], mesh.maxCoord[1])
                )
        elif whichwall == 'Bottom' or whichwall == 'Left':
            surfVariable[i] = field.evaluate_global(
                (line[i], mesh.maxCoord[0])
                )

    return surfVariable

def EasyData(zerodDataTuple, step):
    dataHeader, dataList = zerodDataTuple
    outstring = "step: " + str(step) + "; "
    for label, value in zip(dataHeader, dataList):
        outstring += label + ": " + str(value) + "; "
    return outstring

def MakeFigs(MODEL, key):

    if key == 'showFigs':
        showFigs = True
        saveFigs = False
        print "Showing figures."
    elif key == 'saveFigs':
        showFigs = False
        saveFigs = True
        print "Saving figures."
    else:
        showFigs = False
        saveFigs = False
        print "MakeFigs did not recognise argument."

    MESHES = MODEL.MESHES
    SWARMS = MODEL.SWARMS
    FUNCTIONS = MODEL.FUNCTIONS
    PARAMETERS = MODEL.PARAMETERS
    OPTIONS = MODEL.OPTIONS
    SYSTEMS = MODEL.SYSTEMS

    m = MESHES
    s = SWARMS
    p = PARAMETERS
    f = FUNCTIONS
    o = OPTIONS
    sys = SYSTEMS

    stokes = sys.stokes
    z_hat = p.z_hat
    mesh = m.mesh
    swarm = s.swarm
    materialVar = s.materialVar
    temperatureField = m.temperatureField
    pressureField = m.pressureField
    velocityField = m.velocityField
    devStressField = m.devStressField

    densityFn = f.densityFn
    viscosityFn = f.viscosityFn
    yieldFn = f.yieldFn

    #strainRateFn, secInv, stressFn, devStressFn, devStress2ndInv = physics.GetStressStrain(viscosityFn, velocityField)

    outputPath = o.projectname + '_Output/' + o.projectname

    aspect = p.aspect

    step = MODEL.MISC.currentStep
    modeltime = MODEL.MISC.currentTime

    showfigheight = (OPTIONS.showfigquality * 100) + 100
    showfigwidth = OPTIONS.showfigquality * 100 * int(aspect)
    savefigheight = (OPTIONS.savefigquality * 100) + 100
    savefigwidth = OPTIONS.savefigquality * 100 * int(aspect)

    showParticles = True
    showTemp = True
    showDensity = True
    showVisc = True
    showPressure = False
    showYield = True
    showVelocity = True

    if showParticles:
        if showFigs and size == 1:
            fig = glucifer.Figure( figsize=(showfigwidth,showfigheight), title="Particle Type Distribution: step " + str(step) )
            fig.append( glucifer.objects.Points(swarm,fn_colour=materialVar,fn_size=4. ))
            fig.show()
        if saveFigs and rank == 0:
            fig = glucifer.Figure( figsize=(savefigwidth,savefigheight), title="Particle Type Distribution: step " + str(step) )
            fig.append( glucifer.objects.Points(swarm,fn_colour=materialVar,fn_size=4. ))
            fig.save_image(outputPath + "_particles_" + str(step))

    if showTemp:
        #fig.append( glucifer.objects.VectorArrows(mesh, 1e2/Ra*velocityField) )
        if showFigs and size == 1:
            fig = glucifer.Figure( figsize=(showfigwidth,showfigheight), title = "Temperature: step " + str(step) )
            fig.append( glucifer.objects.Surface(mesh, temperatureField) )
            fig.show()
        if saveFigs and rank == 0:
            fig = glucifer.Figure( figsize=(savefigwidth,savefigheight), title = "Temperature: step " + str(step) )
            fig.append( glucifer.objects.Surface(mesh, temperatureField) )
            fig.save_image(outputPath + "_temperature_" + str(step))

    if showDensity:
        if showFigs and size == 1:
            fig = glucifer.Figure( figsize=(showfigwidth,showfigheight), title="Density: step " + str(step) )
            fig.append( glucifer.objects.Points(swarm,fn_colour=densityFn,fn_size=4.))
            fig.show()
        if saveFigs and rank == 0:
            fig = glucifer.Figure( figsize=(savefigwidth,savefigheight), title="Density: step " + str(step) )
            fig.append( glucifer.objects.Points(swarm,fn_colour=densityFn,fn_size=4.))
            fig.save_image(outputPath + "_density_" + str(step))

    if showVisc:
        if showFigs and size == 1:
            fig = glucifer.Figure( figsize = (showfigwidth,showfigheight), title = "Viscosity Log 10: step " + str(step) )
            fig.append( glucifer.objects.Points(swarm,fn_colour=fn.math.log10(viscosityFn ),fn_size=4.))
            fig.show()
        if saveFigs and rank == 0:
            fig = glucifer.Figure( figsize = (savefigwidth,savefigheight), title = "Viscosity Log 10: step " + str(step) )
            fig.append( glucifer.objects.Points(swarm,fn_colour=fn.math.log10(viscosityFn ),fn_size=4.))
            fig.save_image(outputPath + "_viscosity_" + str(step))

    if showPressure:
        if showFigs and size == 1:
            fig = glucifer.Figure( figsize=(showfigwidth,showfigheight), title="Pressure: step " + str(step) )
            fig.append( glucifer.objects.Surface(mesh, pressureField) )
            fig.show()
        if saveFigs and rank == 0:
            fig = glucifer.Figure( figsize=(savefigwidth,savefigheight), title="Pressure: step " + str(step) )
            fig.append( glucifer.objects.Surface(mesh, pressureField) )
            fig.save_image(outputPath + "_pressure_" + str(step))

    if showYield:
        if showFigs and size == 1:
            fig = glucifer.Figure( figsize = (showfigwidth,showfigheight), title = "Is it yielding?: "+ str(step) )
            fig.append( glucifer.objects.Points(swarm,fn_colour= yieldFn,fn_size=4.))
            fig.show()
        if saveFigs and rank == 0:
            fig = glucifer.Figure( figsize = (savefigwidth,savefigheight), title = "Is it yielding?: "+ str(step) )
            fig.append( glucifer.objects.Points(swarm,fn_colour= yieldFn,fn_size=4.))
            fig.save_image(outputPath + "_yielding_" + str(step))

    if showVelocity:
        if showFigs and size == 1:
            fig = glucifer.Figure( figsize=(showfigwidth,showfigheight), title="Velocity: step " + str(step) )
            velmagfield = uw.function.math.sqrt( uw.function.math.dot(velocityField,velocityField) )
            fig.append( glucifer.objects.VectorArrows(mesh, velocityField/(velmagfield), arrowHead=0.3, scaling=0.07) )
            fig.append( glucifer.objects.Surface(mesh, velmagfield) )
            fig.show()
        if saveFigs and rank == 0:
            fig = glucifer.Figure( figsize=(savefigwidth,savefigheight), title="Velocity: step " + str(step) )
            velmagfield = uw.function.math.sqrt( uw.function.math.dot(velocityField,velocityField) )
            fig.append( glucifer.objects.VectorArrows(mesh, velocityField/(velmagfield), arrowHead=0.3, scaling=0.07) )
            fig.append( glucifer.objects.Surface(mesh, velmagfield) )
            fig.save_image(outputPath + "_velocity_" + str(step))

def MakeData(MODEL):

    MESHES = MODEL.MESHES
    SWARMS = MODEL.SWARMS
    FUNCTIONS = MODEL.FUNCTIONS
    PARAMETERS = MODEL.PARAMETERS
    SYSTEMS = MODEL.SYSTEMS
    OPTIONS = MODEL.OPTIONS

    m = MESHES
    s = SWARMS
    p = PARAMETERS
    f = FUNCTIONS
    sys = SYSTEMS

    stokes = sys.stokes
    z_hat = p.z_hat
    mesh = m.mesh
    swarm = s.swarm
    materialVar = s.materialVar
    temperatureField = m.temperatureField
    pressureField = m.pressureField
    velocityField = m.velocityField
    devStressField = m.devStressField

    densityFn = f.densityFn
    viscosityFn = f.viscosityFn
    yieldFn = f.yieldFn

    maxX = p.maxX
    maxY = p.maxY
    modeltime = MODEL.MISC.currentTime

    hStTop, vStTop, hStRMSTop, vStRMSTop, hStMaxTop, vStMaxTop = physics.FindSurfaceStresses(
        MESHES.devStressField, MESHES.mesh, 'Top')

    # Data points
    dataDict = {
        'step': MODEL.MISC.currentStep,
        'Nu': physics.FindNusseltNumber(temperatureField, mesh),
        'VRMS':physics.FindVRMS(velocityField, mesh),
        #'avSurfVel': np.mean(surfVel),
        'surfVRMS': physics.FindSurfVRMS(mesh, velocityField, 'Top'),
        'avTemp': physics.FindFieldAverage(mesh, temperatureField),
        'avYield': physics.FindFieldAverage(mesh, yieldFn),
        'avVisc': physics.FindFieldAverage(mesh, viscosityFn),
        'hStRMSTop': hStRMSTop,
        'vStRMSTop': vStRMSTop,
        'hStMaxTop': hStMaxTop,
        'vStMaxTop': vStMaxTop,
        'modeltime': modeltime,
        'timeGa': modeltime * f.timescale
        }

    return dataDict

def MakeDataTuples(dataDict):
    keyList = sorted(dataDict.keys(), key=str.lower)
    zerodDataList = []
    zerodHeaderList = []
    onedDataList = []
    onedHeaderList = []
    for key in keyList:
        data = dataDict[key]
        if not type(data) == np.ndarray:
            zerodHeaderList.append(key)
            zerodDataList.append(dataDict[key])
        else:
            onedHeaderList.append(key)
            onedDataList.append(dataDict[key])

    zerodDataTuple = (zerodHeaderList, zerodDataList)
    onedDataTuple = (onedHeaderList, onedDataList)

    return zerodDataTuple, onedDataTuple

def PrintData(DATA, step, modeltime):
    dataDict = DATA.selfdict[str(step)]
    zerodDataTuple, onedDataTuple = MakeDataTuples(dataDict)
    easyData = EasyData(zerodDataTuple, step)
    print easyData

def GatherData(MODEL):
    # Pulls all necessary raw data onto Proc 0
    # so it can be analysed.
    # Returns a 'workingDict' to Proc 0
    # containing numpy arrays
    # (all other procs get None)

    # (DOESN'T DO ANYTHING YET)

    workingDict = {}
    return workingDict

def UpdateData(MODEL):
    #workingDict = GatherData(MODEL)
    dataDict = MakeData(MODEL)
    MODEL.DATA.SetVal(str(MODEL.MISC.currentStep), dataDict)

def Analyse(MODEL):

    if rank == 0:
        print "Starting analysis now"

    m = MODEL.MESHES
    s = MODEL.SWARMS
    f = MODEL.FUNCTIONS
    p = MODEL.PARAMETERS
    o = MODEL.OPTIONS
    d = MODEL.DATA
    misc = MODEL.MISC
    sys = MODEL.SYSTEMS

    modeltime = MODEL.MISC.currentTime
    step = MODEL.MISC.currentStep
    projectname = MODEL.OPTIONS.projectname

    if o.updateDataCondition.evaluate(MODEL):
        UpdateData(MODEL)
        MODEL.MISC.SetVal('freshData', True)
        MODEL.MISC.updateDataBuffer.append(step)

    if rank == 0:
        if o.printDataCondition.evaluate(MODEL) and MODEL.MISC.freshData:
            PrintData(MODEL.DATA, step, modeltime)
        if o.saveDataCondition.evaluate(MODEL) and not MODEL.MISC.updateDataBuffer == []:
            SaveData(MODEL.DATA, projectname, MODEL.MISC.updateDataBuffer)
            MODEL.DATA.selfdict.clear()
            MODEL.MISC.SetVal('updateDataBuffer', [])
        if o.showFigsCondition.evaluate(MODEL):
            MakeFigs(MODEL, 'showFigs')
        if o.saveFigsCondition.evaluate(MODEL):
            MakeFigs(MODEL, 'saveFigs')

def CopyMODEL(MODEL):
    MODELcopy = Collection()
    MODELcopy.SetVals({
        'OPTIONS': Collection(),
        'PARAMETERS': Collection(),
        'MESHES': Collection(),
        'SWARMS': Collection(),
        'FUNCTIONS': Collection(),
        'SYSTEMS': Collection(),
        'DATA': Collection(),
        'MISC': Collection()
        })
    MODELcopy.OPTIONS.SetDict(MODEL.OPTIONS.selfdict)
    MODELcopy.PARAMETERS.SetDict(MODEL.PARAMETERS.selfdict)
    MODELcopy.DATA.SetDict(MODEL.DATA.selfdict)
    MODELcopy.MISC.SetDict(MODEL.MISC.selfdict)
    MODELcopy.SYSTEMS.SetDict(MODEL.SYSTEMS.selfdict)
    MODELcopy.FUNCTIONS.SetDict(MODEL.FUNCTIONS.selfdict)

    MODELcopy.MESHES.SetVals({
        'mesh': MODEL.MESHES.mesh,
        'temperatureField': MODEL.MESHES.temperatureField.copy(deepcopy = True),
        'temperatureDotField': MODEL.MESHES.temperatureDotField.copy(deepcopy = True),
        'pressureField': MODEL.MESHES.pressureField.copy(deepcopy = True),
        'velocityField': MODEL.MESHES.velocityField.copy(deepcopy = True),
        'HField': MODEL.MESHES.HField.copy(deepcopy = True),
        'devStressField': MODEL.MESHES.devStressField.copy(deepcopy = True)
        })
    #blahmesh = uw.mesh.FeMesh_Cartesian( elementType='Q1/dQ0', elementRes=(16,16), minCoord=(0.,0.), maxCoord=(1.,1.) )
    #MODELcopy.SetVal('mesh', MODELcopy.MESHES.temperatureField.mesh)

    MODELcopy.SWARMS.SetVal('swarm', uw.swarm.Swarm(MODELcopy.MESHES.mesh))
    MODELcopy.SWARMS.SetVal('materialVar', MODELcopy.SWARMS.swarm.add_variable(dataType='int', count=1))

    return MODELcopy

def LoadState(MODEL, loadStep):
    step = loadStep
    filename = MODEL.OPTIONS.projectname + '_Output/' + MODEL.OPTIONS.projectname + '_'

    MODEL.MESHES.mesh.load(filename + 'mesh_' + str(step) + '.h5')
    MODEL.MESHES.velocityField.load(filename + 'velocityField_' + str(step) + '.h5')
    MODEL.MESHES.temperatureField.load(filename + 'temperature_' + str(step) + '.h5')
    MODEL.MESHES.pressureField.load(filename + 'pressureField_' + str(step) + '.h5')

    #loadSwarm = uw.swarm.Swarm(mesh = MODEL.MESHES.mesh)
    #loadSwarm.load(filename + 'swarm_' + str(step) + '.h5')
    #for index, coords in enumerate(MODEL.SWARMS.swarm.particleCoordinates.data):
        #MODEL.SWARMS.swarm.particleCoordinates.data[index] = [999999, 999999]
    #print MODEL.SWARMS.swarm.particleCoordinates.data
    #MODEL.SWARMS.swarm.add_particles_with_coordinates(loadSwarm.particleCoordinates.data)
    #MODEL.SWARMS.materialVar.load(filename + 'materialVar_' + str(step) + '.h5')

    #loadSwarm = uw.swarm.Swarm(mesh = MODEL.MESHES.mesh)
    #loadSwarm.load(filename + 'swarm_' + str(step) + '.h5')
    #for index, coords in enumerate(loadSwarm.particleCoordinates.data):
        #with MODEL.SWARMS.swarm.deform_swarm():
            #MODEL.SWARMS.swarm.particleCoordinates.data[index] = coords

    #MODEL.SWARMS.SetVal('swarm', uw.swarm.Swarm(mesh = MODEL.MESHES.mesh))
    #MODEL.SWARMS.SetVal('materialVar', MODEL.SWARMS.swarm.add_variable(dataType = "int", count = 1))
    #MODEL.SWARMS.swarm.load(filename + 'swarm_' + str(step) + '.h5')
    #MODEL.SWARMS.materialVar.load(filename + 'materialVar_' + str(step) + '.h5')

    if rank == 0:
        print "Loaded step ", step

def GetMiscLog(LOG, projectname):
    loadedMiscLog = ReadCSV(projectname, 'miscLog')
    miscLogHeader = loadedMiscLog[0]
    miscLogData = loadedMiscLog[1:]
    for miscLogRow in miscLogData:
        miscDict = {}
        for key, itemString in zip(miscLogHeader, miscLogRow):
            for item in miscLogRow:
                if itemString == 'True' or itemString == 'False':
                    if itemString == 'True':
                        item = True
                    else:
                        item = False
                else:
                    try:
                        item = int(itemString)
                    except:
                        try:
                            item = float(itemString)
                        except:
                            item = itemString
                miscDict[key] = item
        LOG.SetVal(miscDict['currentStep'], miscDict)

def SaveState(MODEL):

    m = MODEL.MESHES
    s = MODEL.SWARMS
    o = MODEL.OPTIONS

    step = MODEL.MISC.currentStep

    mesh = m.mesh
    swarm = s.swarm
    materialVar = s.materialVar
    temperatureField = m.temperatureField
    pressureField = m.pressureField
    velocityField = m.velocityField
    filename = o.projectname + '_Output/' + o.projectname + '_'

    swarm.save(filename + 'swarm_' + str(step) + '.h5')
    materialVar.save(filename + 'materialVar_' + str(step) + '.h5')
    mesh.save(filename + 'mesh_' + str(step) + '.h5')
    velocityField.save(filename + 'velocityField_' + str(step) + '.h5')
    temperatureField.save(filename + 'temperature_' + str(step) + '.h5')
    pressureField.save(filename + 'pressureField_' + str(step) + '.h5')

    MODEL.MISC.SetVal('saveState', True)
    if rank == 0:
        SaveLog(MODEL.LOG, MODEL.MISC, MODEL.MISC.unsavedSteps, o.projectname)
    MODEL.MISC.SetVal('unsavedSteps', [])

    MODEL.LOG.selfdict.clear()

def SaveLog(LOG, MISC, stepsToSave, projectname):
    keyList = sorted(MISC.selfdict.keys(), key = str.lower)
    dataLoL = []
    for step in stepsToSave:
        dataDict = LOG.selfdict[step]
        row = []
        for key in keyList:
            row.append(dataDict[key])
        dataLoL.append(row)
    MakeCSV(projectname, "miscLog", keyList, dataLoL)

def MakeDataFromLoadState(MODEL, stepTuple, figures = True, data = True):
    print "Making data from loaded state."
    if rank == 0:
        startStep, stopStep = stepTuple
        step = startStep
        projectname = MODEL.OPTIONS.projectname
        outputDir = projectname + '_Output/'
        #loadedMiscLog = ReadCSV(projectname, 'miscLog')
        #miscLogHeader = loadedMiscLog[0]
        filename = MODEL.OPTIONS.projectname + '_Output/' + MODEL.OPTIONS.projectname + '_'
        GetMiscLog(MODEL.LOG, projectname)
        while step < stopStep:
            print "Checking step: ", step
            if os.path.exists(filename + 'mesh_' + str(step) + '.h5'):
                print "Saved state found!"
                MODEL.MISC.SetDict(MODEL.LOG.selfdict[step])
                LoadState(MODEL, step)
                if data:
		            UpdateData(MODEL)
		            SaveData(MODEL.DATA, projectname, [step])
                if figures:
                    MakeFigs(MODEL, 'saveFigs')
            step += 1

def UpdateEpochTime():
    if rank == 0:
        epochTime = time.time()
    else:
        epochTime = None
    epochTime = comm.bcast(epochTime, root=0)
    return epochTime

def SynchroniseMeshVars(MESHES):
    for key in MESHES.selfdict.keys():
        item = MESHES.selfdict[key]
        if isinstance(item, uw.mesh.MeshVariable):
            item.syncronise()

def RunLoop(MODEL, startTime):

    if rank == 0:
        print "Doing the run loop."

    MODEL.SYSTEMS.solver.solve(nonLinearIterate = MODEL.PARAMETERS.yielding)
    dt = np.min([MODEL.SYSTEMS.advDiff.get_max_dt(), MODEL.SYSTEMS.advector.get_max_dt()])
    MODEL.SYSTEMS.advDiff.integrate(dt)
    MODEL.SYSTEMS.advector.integrate(dt)

    MODEL.SYSTEMS.population_control.repopulate()

    projector = uw.utils.MeshVariable_Projection(
        MODEL.MESHES.devStressField,
        MODEL.FUNCTIONS.stressFn,
        type = 0
        )
    projector.solve()

    if rank == 0:
        print "All the meaty stuff is done."

    timecheck = (UpdateEpochTime(), time.clock())
    stopwatch = np.subtract(timecheck, startTime)

    MODEL.MISC.SetVals({
        'currentTime': MODEL.MISC.currentTime + dt,
        'currentStep': MODEL.MISC.currentStep + 1,
        'runningEpochTime': stopwatch[0],
        'runningClockTime': stopwatch[1],
        'freshData': False,
        'saveState': False
        })
    MODEL.MISC.SetVal('modelrunComplete', not MODEL.OPTIONS.modelRunCondition.evaluate(MODEL))
    unsavedSteps = MODEL.MISC.unsavedSteps
    unsavedSteps.append(MODEL.MISC.currentStep)
    MODEL.MISC.SetVal('unsavedSteps', unsavedSteps)

    Analyse(MODEL)

    #UpdateMiscLog(MODEL.MISC, MODEL.LOG)
    if rank == 0:
        print MODEL.MISC.selfdict
    MODEL.LOG.SetVal(MODEL.MISC.currentStep, MODEL.MISC.selfdict.copy())
    if MODEL.OPTIONS.saveStateCondition.evaluate(MODEL):
        SaveState(MODEL)

    if rank == 0:
        print '-' * 10 + ' Finished step ' + str(MODEL.MISC.currentStep) + ' ' + '-' * 10

def Run(MODEL, startStep = 0):

    projectname = MODEL.OPTIONS.projectname

    if rank == 0:
        outputDir = projectname + '_Output/'
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)
        #shutil.copy(__file__, outputDir + projectname + '.py')

    if startStep == 0:
        if rank == 0:
            print "Starting from step zero."

        # Bit of house cleaning to make sure the output folder is suitable:
        #if rank == 0:
            #if os.path.exists(outputDir + 'miscLog.csv'):
                #os.rename(outputDir + 'miscLog.csv', outputDir + 'miscLog' + str(int(time.time())) + '.csv')

        # Make sure all procs use the same clock:
        MODEL.MISC.SetVals({
            'currentTime': 0.,
            'currentStep': 0,
            'runningEpochTime': 0.,
            'runningClockTime': 0.,
            'updateDataBuffer': [],
            'freshData': False,
            'runningStatus': False,
            'modelrunComplete': False,
            'savedState': False,
            'unsavedSteps': []
            })

        # Initialise meshvars
        #SWARMS.materialVar.data[:] = FUNCTIONS.initialMaterialFn
        MODEL.FUNCTIONS.initialTempFn.evaluate(MODEL.MESHES.mesh)
        MODEL.MESHES.HField.data[:] = MODEL.FUNCTIONS.initialHFn

        projector = uw.utils.MeshVariable_Projection(
            MODEL.MESHES.devStressField,
            MODEL.FUNCTIONS.stressFn,
            type = 0
            )
        projector.solve()

        if MODEL.OPTIONS.saveStateCondition.evaluate(MODEL):
            SaveState(MODEL)
        Analyse(MODEL)
        MODEL.LOG.SetVal(MODEL.MISC.currentStep, MODEL.MISC.selfdict.copy())

    else:
        if rank == 0:
            print "Starting from loaded state."
        GetMiscLog(MODEL.LOG, projectname)
        MODEL.MISC.SetDict(MODEL.LOG.selfdict[startStep])
        #if rank == 0:
            #if os.path.exists(outputDir + 'miscLog.csv'):
                #os.rename(outputDir + 'miscLog.csv', outputDir + 'miscLog' + str(int(time.time())) + '.csv')
        #MODEL.MISC.SetVal('unsavedSteps', [step for step in range(startStep + 1)])
        LoadState(MODEL, startStep)
        MODEL.MISC.SetVal('modelrunComplete', False)
        MODEL.MISC.SetVal('unsavedSteps', [])
        MODEL.MISC.SetVal('updateDataBuffer', []) # Will this cause problems???

    # RUNNING!
    MODEL.MISC.SetVal('runningStatus', True)
    startTime = (UpdateEpochTime(), time.clock())
    while not MODEL.MISC.modelrunComplete:
        RunLoop(MODEL, startTime) # <<<<<< this is where the magic happens
    MODEL.MISC.SetVal('runningStatus', False)
    # FINISHED!

    if rank == 0:
        print "Model successfully ran to specified end state."

def SaveData(DATA, projectname, updateDataBuffer):

    allDataDict = DATA.selfdict
    zerodAllData = []
    onedAllData = []
    for stepNo in updateDataBuffer:
        dataDict = allDataDict[str(stepNo)]
        zerodDataTuple, onedDataTuple = MakeDataTuples(dataDict)
        zerodDataHeader, zerodDataList = zerodDataTuple
        onedDataHeader, onedDataLoL = onedDataTuple
        zerodAllData.append(zerodDataList)
        onedAllData.append(onedDataLoL)

    MakeCSV(projectname, 'zerodData', zerodDataHeader, zerodAllData)

    for dataindex in range(len(onedDataHeader)):
        dataLoL = []
        title = onedDataHeader[dataindex]
        for index in range(len(onedAllData)):
            dataList = onedAllData[index][dataindex]
            dataLoL.append(dataList)
        MakeCSV(projectname, title, [title], dataLoL)

def MakeCSV(projectname, title, header, data):
    # Turns a list of lists into a CSV file

    outputDir = projectname + '_Output/'
    outputPath = outputDir + title + '.csv'
    csvFilename = outputPath

    if not os.path.exists(csvFilename):
        mode = 'w'
        writeHeader = True
    else:
        mode = 'a'
        writeHeader = False

    with open(csvFilename, mode) as csvFile:
        csvWriter = csv.writer(
            csvFile,
            delimiter=',',
            quotechar='|',
            quoting=csv.QUOTE_MINIMAL
            )
        if writeHeader:
            csvWriter.writerow(header)
        for dataList in data:
            csvWriter.writerow(dataList)
    csvFile.close()

def ReadCSV(projectname, title):
    outputDir = projectname + '_Output/'
    outputPath = outputDir + title + '.csv'
    csvFilename = outputPath
    output = []
    if not os.path.exists(csvFilename):
        print "No file found."
        return None
    else:
        with open(csvFilename, 'rb') as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                output.append(row)
    csvFile.close()
    return output

def CapValue(value, rangeTuple):
    lowercap, uppercap = rangeTuple
    cappedValue = fn.misc.max(
        fn.misc.min(
            value,
            fn.misc.constant(uppercap)
            ),
        fn.misc.constant(lowercap)
        )
    return cappedValue

def CombinedFn(option, fnTuple = (1.,1.), range = "N/A"):
    if option == 'min':
        outFn = fn.misc.min(fnTuple[0], fnTuple[1])
    elif option == 'max':
        outFn = fn.misc.max(fnTuple[0], fnTuple[1])
    elif option == 'prod':
        outFn = np.prod(fnTuple)
    if not range == "N/A":
        cappedFn = CapValue(outFn, range)
        return cappedFn
    else:
        return outFn
