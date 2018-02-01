import spynnaker7.pyNN as sim
from vision import ConvolutionPopulation
import numpy as np

sim.setup(timestep=1.)

width = 32
height = 24
polarity_bits = 1
n_pre = 1 << int(np.ceil(np.log2(width)) + np.ceil(np.log2(height)) + polarity_bits)

sample_step = 2
kernel = np.ones((3, 3))/2.
pre = sim.Population(n_pre, sim.SpikeSourcePoisson,
                     {'rate': 100}, label='image source')
post = sim.Population(1, ConvolutionPopulation,
                      {'src_width': width, 'src_height': height,
                       'src_polarity_bits': polarity_bits,
                       'sample_step_width': sample_step,
                       'sample_step_height': sample_step,
                       'kernel': kernel,
                       'time_window': 100,},
                      label='convolution core average')
proj = sim.Projection(pre, post, sim.FromListConnector([(0, 0, 1, 1)]))
sim.run(5000)