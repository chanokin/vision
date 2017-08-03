import numpy as np
import pylab as plt
import time
import sys
import pickle

from pyNN import spiNNaker as sim
# from pyNN import nest as sim
# import spynnaker_extra_pynn_models as q
# sim = None

def remove_ticks(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])


test_exc = True if 1 else False
neurons_per_core = 255

ms = 1000.
time_step = 1.
tstep_ms = int(ms/time_step)

debug         = True if 1 else False
izk_neurons   = True if 1 else False

num_neurons = 1
num_exc = 1

# w2s   = 4.376069 # for 10 ms tau_m
w2s   = 3.94125 # for 20 ms tau_m
w_max = w2s*1.


hours   = 0
minutes = 0
seconds = 5.
real_time = int( np.ceil( tstep_ms*(hours*60*60 + minutes*60 + seconds) ) )
sim_time = int( real_time*1.5 )

########################################################################
# S P I K E    T I M E S    G E N E R A T I O N
########################################################################
spike_times = [10]


#########################################################################
# S I M U L A T O R    S E T U P
#########################################################################

cell = sim.IF_curr_exp

exc_params = {  'cm': 0.35,  # nF
                'i_offset': 0.0,
                'tau_m': 20.0,
                'tau_refrac': 2.0,
                'tau_syn_E': 1.,
                'tau_syn_I': 1.,
                'v_reset': -70.0,
                'v_rest': -65.0,
                'v_thresh': -55.4
            }

inh_params = {  'cm': 0.35,  # nF
                'i_offset': 0.0,
                'tau_m': 20.0,
                'tau_refrac': 1.0,
                'tau_syn_E': 1.,
                'tau_syn_I': 1.,
                'v_reset': -70.0,
                'v_rest': -65.0,
                'v_thresh': -58.
            }

if sim.__name__ == 'pyNN.spiNNaker':
    sim.set_number_of_neurons_per_core(cell, neurons_per_core)

sim.setup(timestep=time_step, min_delay=1., max_delay=144.)

########################################################################
# P O P U L A T I O N S
########################################################################

source = sim.Population(num_exc, sim.SpikeSourceArray, 
                        {'spike_times': spike_times}, label='Source (EXC)')


if test_exc:
    target = sim.Population(num_exc, cell, exc_params, label='Target (EXC)')
else:
    target = sim.Population(num_exc, cell, inh_params, label='Target (INH)')

source.record()
target.record()

########################################################################
# P R O J E C T I O N S
########################################################################


src_to_tgt = sim.Projection(source, target,
                            sim.OneToOneConnector(weights=w2s),
                            target='excitatory')


########################################################################
# R U N   S I M U L A T I O N
########################################################################


sim.run(sim_time)
spikes = {'target': target.getSpikes(compatible_output=True),
          'source': source.getSpikes(compatible_output=True),
         }

sim.end()

print("Total number of spikes %d"%(len(spikes['target'])))
########################################################################
# P L O T    R E S U L T S
########################################################################

# Spikes
fig = plt.figure()
ax = plt.subplot(1, 1, 1)
nids  = [nid for (nid, spkt) in spikes['source']]
spkts = [spkt for (nid, spkt) in spikes['source']]
plt.plot(spkts, nids, '^g', markersize=4)

nids  = [nid for (nid, spkt) in spikes['target']]
spkts = [spkt for (nid, spkt) in spikes['target']]
plt.plot(spkts, nids, 'ob', markersize=4)
plt.ylim(-1, num_neurons+1)
plt.draw()
plt.margins(0.05, 0.05)
plt.savefig("characterize_spike_activity.png", dpi=300)
# plt.show()
plt.close(fig)




