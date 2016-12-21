"""  some temproty script """

import numpy as np

x = np.random.rand(10,10)
y = np.random.rand(1000,1000)

np.savez('tempsotre.npz', x=x,y=y)

""" add some updates """


