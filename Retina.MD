To instantiate a retina object:

```python
import copy
from vision.retina import Retina, dvs_modes, MERGED
### MERGED means neuron id = [x|y|p]
from vision import default_config
from pyNN import spiNNaker as sim
from vision.sim_tools.connectors.mapping_funcs import row_col_to_input_breakout

### increase max_delay for motion sensing
sim.setup(timestep=1., max_delay=144., min_delay=1.)

### retina configuration
ret_conf = copy.deepcopy(default_config.defaults_retina)
### (optional) to disable motion sensing
if 'direction' in ret_conf: ret_conf['direction'] = False
### (optional) to disable orientation sensing
if 'gabor' in ret_conf: ret_conf['gabor'] = False

### use special input to population mapping for live game
ret_conf['input_mapping_func'] = row_col_to_input_breakout
ret_conf['row_bits'] = 8

mode = dvs_modes[MERGED]
retina = Retina(sim, breakout_pop, X_RESOLUTION, Y_RESOLUTION,
                mode, cfg=ret_conf)
```

Output populations of Retina can be accessed through

```python
for channel in retina.channels:
    for pop_key in retina.get_output_keys():
        retina.pops[channel][pop_key]['ganglion']
```

There are convenience functions to access the shapes of populations:

```python
retina.pop_width(pop_key)
retina.pop_height(pop_key)
retina.pop_size(pop_key)
```

Neuron IDs in "ganglion" populations are row-major mapped. That is

```
[  0     1   2  ...    W-1   ]
[  W    W+1 W+2 ...   2W-1   ]
   :     :   :         :
[(H-1)W    ...     (H-1)(W-1)]
```

Mapping to and from neuron id 

```python
nID = row*W + col
row = row//W #integer division
col = row%W  #modulo operator
```
