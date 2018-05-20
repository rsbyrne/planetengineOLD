# templates.py
# part of the planetengine package

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

import physics
import utilities

comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

CoordFn = uw.function.input()
depthFn = 1. - CoordFn[1]
