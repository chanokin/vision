from sim_tools.common import *
from sim_tools.kernels import center_surround as krn_gen, gabor as krn_gbr
from sim_tools.connectors import kernel_connectors as krn_con, \
                                 standard_connectors as conn_std, \
                                 mapping_funcs as mapf
from scipy.signal import convolve2d, correlate2d

from default_config import defaults_lgn as defaults


class LGN():
    def __init__(self, simulator, retina, cfg=defaults):
        
        print("Building LGN...")
        
        for k in defaults.keys():
            if k not in cfg.keys():
                cfg[k] = defaults[k]
        
        self.cfg      = cfg
        self.sim      = simulator
        self.retina   = retina
        self.rcfg     = retina.cfg
        self.rcs      = retina.cs
        self.channels = retina.channels
        self.shapes   = retina.shapes
        self.width    = retina.width
        self.height   = retina.height
        self.css      = retina.css

        print("\tBuilding kernels...")
        self.build_kernels()
        print("\t\tdone!")
        
        print("\tBuilding connectors...")
        self.build_connectors()
        print("\t\tdone!")
        
        print("\tBuilding populations...")
        self.build_populations()
        print("\t\tdone!")
        
        print("\tBuilding projections...")
        self.build_projections()
        print("\t\tdone!")

    def _right_key(self, key):
        if 'gabor' in key:
            return 'gabor'
        elif 'dir' in key:
            return 'dir'
        else:
            return key
        
    def pop_size(self, key):
        return self.shapes[self._right_key(key)]['size']
    
    def pop_width(self, key):
        return self.shapes[self._right_key(key)]['width']
    
    def pop_height(self, key):
        return self.shapes[self._right_key(key)]['height']

    def sample_step(self, key):
        return self.shapes[self._right_key(key)]['step']
    
    def sample_start(self, key):
        return self.shapes[self._right_key(key)]['start']

    def output_keys(self):
        return self.pops[self.channels[0]].keys()
        
    def build_kernels(self):
        rcfg = self.rcfg
        cfg  = self.cfg
        cs = {}
        ccss = 'cs'
        cs[ccss] = krn_gen.split_center_surround_kernel(rcfg[ccss]['width'],
                                                        rcfg[ccss]['std_dev'], 
                                                        rcfg[ccss]['sd_mult'])
        for i in range(len(cs[ccss])): 
            # print( cs[ccss][i] )

            inv_sum_pos = 1./(np.sum(cs[ccss][i]))
            cs[ccss][i] *= inv_sum_pos * cfg['w2s'] * rcfg[ccss]['w2s_mult']
            print( cs[ccss][i] )
        
        # sys.exit(0)
        self.split_cs = cs


    def build_connectors(self):
        sim = self.sim
        cfg = self.cfg
        conns = {}
        scs = self.split_cs
        for ccss in self.retina.get_output_keys():

            pre_shape = (self.pop_height(ccss), self.pop_width(ccss))
            pre_start = (self.sample_start(ccss), self.sample_start(ccss))
            pre_steps = (self.sample_step(ccss), self.sample_step(ccss))

            conns[ccss] = {}
            
            if self.sim.__name__ == 'pyNN.spiNNaker':

                exc = sim.KernelConnector(pre_shape, pre_shape,
                                          weights=scs['cs'][EXC], 
                                          delays=cfg['kernel_exc_delay'], 
                                          generate_on_machine=True)

                inh = sim.KernelConnector(pre_shape, pre_shape,
                                          weights=scs['cs'][INH], 
                                          delays=cfg['kernel_inh_delay'], 
                                          generate_on_machine=True)
            else:
                ex, _ = krn_con.full_kernel_connector(self.pop_width(ccss), 
                                                      self.pop_height(ccss),
                                                      scs['cs'][EXC], 
                                                      cfg['kernel_exc_delay'], 
                                                      cfg['kernel_inh_delay'],
                                                      map_to_src=mapf.row_major,
                                                      pop_width=self.pop_width(ccss)
                                                     )

                ih, _ = krn_con.full_kernel_connector(self.pop_width(ccss), 
                                                      self.pop_height(ccss),
                                                      scs['cs'][INH], 
                                                      cfg['kernel_exc_delay'], 
                                                      cfg['kernel_inh_delay'],
                                                      map_to_src=mapf.row_major,
                                                      pop_width=self.pop_width(ccss)
                                                     )

                exc = sim.FromListConnector(ex)
                inh = sim.FromListConnector(ih)

            conns[ccss] = {EXC: exc, INH: inh}

        self.conns = conns


    def build_populations(self):
        sim = self.sim
        cfg = self.cfg
        exc_cell = getattr(sim, cfg['exc_cell']['cell'], None)
        exc_parm = cfg['exc_cell']['params']
        inh_cell = getattr(sim, cfg['inh_cell']['cell'], None)
        inh_parm = cfg['inh_cell']['params']
        
        pops = {}

        for c in self.channels:
            c = self.channels[c]

            pops[c] = {}
            for k in self.retina.get_output_keys():
                popsize = self.pop_size(k)

                pops[c][k] = {}
                pops[c][k]['inter']    = sim.Population(popsize,
                                               inh_cell, inh_parm,
                                               label='LGN inter %s %s'%(c, k))
                pops[c][k]['ganglion'] = sim.Population(popsize,
                                               exc_cell, exc_parm,
                                               label='LGN output %s %s'%(c, k))

                if cfg['record']['voltages']:
                    pops[c][k]['inter'].record_v()
                    pops[c][k]['ganglion'].record_v()

                if cfg['record']['spikes']:
                    pops[c][k]['inter'].record()
                    pops[c][k]['ganglion'].record()
            
        self.pops = pops


    def build_projections(self):
        sim = self.sim
        cfg = self.cfg
        projs = {}
        for c in self.channels:
            c = self.channels[c]
            projs[c] = {}
            for k in self.retina.get_output_keys():
                # print('lgn - projections - key: %s'%k)
                
                projs[c][k] = {}
                o2o = sim.OneToOneConnector(weights=cfg['w2s'],
                                            delays=cfg['kernel_inh_delay'])
                # print("src size: %d"%self.retina.pops['off'][k]['ganglion'].size) 
                # print("dst size: %d"%self.pops[k]['inter'].size)
                projs[c][k]['inter'] = sim.Projection(self.retina.pops[c][k]['ganglion'],
                                                      self.pops[c][k]['inter'], o2o,
                                                      target='excitatory')

                split = self.conns[k]
                projs[c][k]['exc'] = sim.Projection(self.retina.pops[c][k]['ganglion'], 
                                                    self.pops[c][k]['ganglion'], 
                                                    split[EXC],
                                                    target='excitatory')

                cntr_c = 'off' if c == 'on' else 'on'
                projs[c][k]['inh'] = sim.Projection(self.pops[cntr_c][k]['inter'], 
                                                    self.pops[c][k]['ganglion'], 
                                                    split[INH],
                                                    target='inhibitory')

            self.projs = projs

        
