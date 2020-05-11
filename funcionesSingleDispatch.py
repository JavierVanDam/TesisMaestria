import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from functools import singledispatch

@singledispatch
def ploteaBarrasModeloBinario(y_real, m, interactivo=True):
    print("y_real es {} y modelo es {}".format(type(y_real), type(m)))
    raise NotImplementedError("SOLO SE SOPORTAN OTROS DATOS")


@ploteaBarrasModeloBinario.register(np.ndarray)
@ploteaBarrasModeloBinario.register(BaseEstimator)
def _(y_real, m, interactivo=True):
    print("ARRAY / BASE ESTIMATOR")
    print("y_real es {} y modelo es {}".format(type(y_real) , type(m)))


@ploteaBarrasModeloBinario.register(np.ndarray)
@ploteaBarrasModeloBinario.register(np.ndarray)
def _(y_real, m, interactivo=True):
    print("NP ARRAY / NP ARRAY")
    print("y_real es {} y prob_y_modelo es {}".format(type(y_real) , type(m)))
