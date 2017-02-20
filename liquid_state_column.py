from base_column import *

class V1MultiColumn(BaseColumn):
    
    def __init__(self, sim,  lgn, width, height, location, learning_on, 
                 group_size, cfg):
        # print("\t\tBuilding MultiColumn ... ")
        BaseColumn.__init__(self, lgn, width, height, location, 
                            learning_on, cfg)

        self.sim = sim
        self.group_size = group_size

        self.pix_key   = 'cs'
        self.feat_keys = [k for k in lgn.pops.keys() if k != 'cs']
        self.num_in_ctx = len(self.feat_keys)

        # print("\t\t\tbuilding input indices...")
        self.build_input_indices_weights()
        # print("\t\t\tdone!")
        
        # print("\t\t\tbuilding connectors...")
        self.build_connectors()
        # print("\t\t\tdone!")
        
        # print("\t\t\tbuilding populations...")
        self.build_populations()
        # print("\t\t\tdone!")
        
        # print("\t\t\tbuilding projections...")
        self.build_projections()
        # print("\t\t\tdone!")

    def update_weights(self, new_weights):
        pass
        
    
    def build_connectors(self):
        conns = {}
        cfg = self.cfg
        size = self.group_size
        sipl_idx = [i for i in range(size)]
        
        for k in self.in_indices:
            in_idx = self.in_indices[k]
            in_ws  = self.in_weights[k]
            conns[k] = conn_std.list_all2all(in_idx, sipl_idx, 
                                             weight=cfg['pix_in_weight'], 
                                             delay=2., sd=0.1,
                                             in_weight_scaling=in_ws)

        conns['sipl2intr'] = conn_std.list_wta_interneuron(sipl_idx, sipl_idx, 
                                                           ff_weight=cfg['w2s'], 
                                                           fb_weight=-cfg['w2s'], 
                                                           delay=1.)
        
        self.conns = conns


    def build_populations(self):
        def loc2lbl(loc, pop):
            return "column (%d, %d) - %s"%(loc[ROW], loc[COL], pop)

        sim = self.sim
        cfg = self.cfg
        exc_cell = getattr(sim, cfg['exc_cell']['cell'], None)
        exc_parm = cfg['exc_cell']['params']
        inh_cell = getattr(sim, cfg['wta_inh_cell']['cell'], None)
        inh_parm = cfg['wta_inh_cell']['params']
        
        pops = {}
        pops['simple'] = sim.Population(self.group_size,
                                        exc_cell, exc_parm,
                                        label=loc2lbl(self.in_location, \
                                                     'simple') )

        pops['wta_inh'] = sim.Population(self.group_size,
                                         inh_cell, inh_parm,
                                         label=loc2lbl(self.in_location, 'wta') )
                                                
        
        if cfg['record']['spikes']:
            pops['simple'].record()
            pops['wta_inh'].record()

        if cfg['record']['voltages']:
            pops['simple'].record_v()
            pops['wta_inh'].record_v()
        
        self.pops = pops
    
    
    def build_projections(self):
        sim = self.sim
        cfg = self.cfg
        projs = {}
        
        for k in self.conns:
            if k == 'sipl2intr':
                continue
            
            #both channels land on the same neuron group
            
            in_pop = self.lgn.pops['on'][k]['output']
            conn = sim.FromListConnector(self.conns[k])
            syn_dyn = self.get_synapse_dynamics()
            projs['in2sipl'] = sim.Projection(in_pop, self.pops['simple'],
                                              conn, synapse_dynamics=syn_dyn,
                                              label='input to simple')

            in_pop = self.lgn.pops['off'][k]['output']
            projs['in2sipl'] = sim.Projection(in_pop, self.pops['simple'],
                                              conn, synapse_dynamics=syn_dyn,
                                              label='input to simple')
            
        
        conn = sim.FromListConnector(self.conns['sipl2intr'][0])
        projs['sipl2intr'] = sim.Projection(self.pops['simple'],
                                            self.pops['wta_inh'],
                                            conn, 
                                            label='simple to inter')

        conn = sim.FromListConnector(self.conns['sipl2intr'][1])
        projs['intr2sipl'] = sim.Projection(self.pops['wta_inh'],
                                            self.pops['simple'],
                                            conn, 
                                            label='inter to simple')
            
        self.projs = projs





    def get_weights_input(self):
        sp = self.projs['in2sipl']
        all_ws = sp.getWeights(format='array', gather=False)
        # print(all_ws[self.in_indices, 0].shape)
        weights = [ all_ws[self.in_indices, i] for i in range(self.group_size) ]
        
        return weights
        
        
        
    def get_weights_pictures(self, weights):
        w = weights
        recept_shape = (self.in_receptive_width, self.in_receptive_width)
        
        imgs = [ np.array(w[i]).reshape(recept_shape) for i in range(len(w)) ]
        
        return imgs
        
        
        
