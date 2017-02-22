from base_column import *


class V1FourToOneColumn(BaseColumn):
    
    def __init__(self, sim, lgn, width, height, location, learning_on, cfg):
        # print("Building Four-to-One column (%d, %d)"%(row, col))
        BaseColumn.__init__(self, lgn, width, height, location, 
                            learning_on, cfg)
        self.sim = sim
        self.neuron_count     = cfg['neurons_in_column']
        self.input_conn_prob  = cfg['input_conn_prob']
        self.column_conn_prob = cfg['column_conn_prob']
        self.column_conn_wgt  = cfg['column_conn_wgt']
        self.pop_ratio        = cfg['pop_ratio']

        # print("\t\t\tbuilding population sizes...")
        self.build_pop_sizes()
        # print("\t\t\tdone!")
        
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


    def build_pop_sizes(self):
        sizes = {}
        for lyr in self.column_conn_prob:
            sizes[lyr] = {}
            exc_rat = self.pop_ratio[lyr]['exc']
            sizes[lyr]['exc'] = int(self.neuron_count[lyr]*exc_rat)
            sizes[lyr]['inh'] = int(self.neuron_count[lyr] - sizes[lyr]['exc'])
        self.pop_sizes = sizes


    def decode_conn_key(self, layer, key):
        # print_debug(key)
        key_split = key.split('2')
        src = key_split[0]
        if len(key_split) == 3:
            dst_lyr = 'l2'
            dst = 'exc' if key_split[-1] == 'e' else 'inh'
        else:
            if 'l' in key_split[-1]:
                dst_lyr = key_split[-1][0:2]
                dst = 'exc' if key_split[-1][-1] == 'e' else 'inh'
            else:
                dst_lyr = layer
                dst = key_split[-1]
                
        return src, dst, dst_lyr


    def decode_in_conn_key(self, key):
        dst = 'exc' if key[2] == 'e' else 'inh'
        return key[0:2], dst
        
    def build_intra_conns(self):
        cfg = self.cfg
        sim = self.sim
        prob_conn = conn_std.probability_connector
        inner_conns = {}
        for lyr in self.column_conn_prob:
            inner_conns[lyr] = {}
            for conn in self.column_conn_prob[lyr]:
                if '2' not in conn:
                    continue
                
                src, dst, dst_lyr = self.decode_conn_key(lyr, conn)
                max_weight = self.column_conn_wgt[lyr][conn]
                min_weight = self.min_weight
                prob = self.column_conn_prob[lyr][conn]
                delay = cfg['min_delay']
                rng = sim.NumpyRNG(seed=None)
                w_dist = sim.RandomDistribution('uniform', 
                                                [min_weight, max_weight],
                                                rng=rng)
                conns = sim.FixedProbabilityConnector(prob, weights=w_dist,
                                                      delays=delay)
                # conns = prob_conn(self.pop_sizes[lyr][src],
                                  # self.pop_sizes[dst_lyr][dst],
                                  # prob, weight, delay,
                                  # weight_std_dev=1.)

                inner_conns[lyr][conn] = conns
                
        return inner_conns


    def build_input_conns(self):
        cfg = self.cfg
        sim = self.sim
        list_prob_conn = conn_std.list_probability_connector
        key_to_pop = {'l4e': self.lgn.css, 'l2e': ['gabor', 'dir'],
                      'l4i': self.lgn.css, 'l5e': []}
        input_conns = {}
        dst_indices = []
        delay = cfg['input_delay']
        
        for ch in self.lgn.channels:
            input_conns[ch] = {}
            for pop in self.lgn.output_keys():
                input_conns[ch][pop] = {}
                for conn_key in self.input_conn_prob['main']:
                    if pop in key_to_pop[conn_key]:
                        dst_lyr, dst_pop = self.decode_in_conn_key(conn_key)

                        sign = 1 if dst_pop == 'exc' else -1
                        weight = sign*self.in_weights[pop]
                        dst_size = self.pop_sizes[dst_lyr][dst_pop]
                        dst_indices[:] = [i for i in range(dst_size)]
                        prob = self.input_conn_prob['main'][conn_key]
                        conns = list_prob_conn(self.in_indices[pop],
                                               dst_indices, 
                                               prob, weight, delay)

        return input_conns
        

    def build_connectors(self):
        self.intra_conns = self.build_intra_conns()
        # centre-surround to layer 4
        # direction to layer 2
        # gabor to layer 2
        self.input_conns = self.build_input_conns()


    def build_populations(self):
        def label(lyr, pop_type):
            return "(%d, %d) - (%s, %s)"%(self.location[ROW], \
                                          self.location[COL], \
                                          lyr, pop_type)
        sim = self.sim
        cfg = self.cfg
        exc_cell = getattr(sim, cfg['exc_cell']['cell'], None)
        exc_parm = cfg['exc_cell']['params']
        inh_cell = getattr(sim, cfg['inh_cell']['cell'], None)
        inh_parm = cfg['inh_cell']['params']
        pops = {}
        for lyr in self.pop_ratio:
            pops[lyr] = {}
            for pop_type in self.pop_ratio[lyr]:
                lbl  = label(lyr, pop_type)
                parm = exc_parm if pop_type == 'exc' else inh_parm
                cell = exc_cell if pop_type == 'exc' else inh_cell
                size = self.pop_sizes[lyr][pop_type]
                pops[lyr][pop_type] = sim.Population(size, cell, parm,
                                                     label=lbl)
        
        self.pops = pops
        self.input_pop    = self.pops['l4']['exc']
        self.output_pop   = self.pops['l2']['exc']
        self.feedback_pop = self.pops['l5']['exc']
        self.interconnect_pop = self.pops['l2']['exc']

        if cfg['record']['voltages']:
            self.input_pop.record_v()
            self.output_pop.record_v()
            self.feedback_pop.record_v()

        if cfg['record']['spikes']:
            self.input_pop.record()
            self.output_pop.record()
            self.feedback_pop.record()


    def build_projections(self):
        sim = self.sim
        cfg = self.cfg
        intra_projs = {}
        for lyr in self.intra_conns:
            intra_projs[lyr] = {}
            for conn in self.intra_conns[lyr]:
                src, dst, dst_lyr = self.decode_conn_key(lyr, conn)
                tgt = 'excitatory' if src == 'exc' else 'inhibitory'
                # flc = sim.FromListConnector(self.intra_conns[lyr][conn])
                flc = self.intra_conns[lyr][conn]
                
                if src == 'inh' or dst == 'inh':
                    # print('inhibitory projection %s, %s'%(lyr, conn))
                    syn_dyn = None
                else:
                    syn_dyn = self.get_synapse_dynamics()

                proj = sim.Projection(self.pops[lyr][src], 
                                      self.pops[dst_lyr][dst],
                                      flc, target=tgt,
                                      synapse_dynamics = syn_dyn)
                intra_projs[lyr][conn] = proj
        self.intra_projs = intra_projs
        
        input_projs = {}
        for ch in self.input_conns:
            input_projs[ch] = {}
            for pop in self.input_conns[ch]:
                input_projs[ch][pop] = {}
                for conn_key in self.input_conns[ch][pop]:
                    dst_lyr, dst_pop = self.decode_in_conn_key(conn_key)
                    conn = self.input_conns[ch][pop][conn_key]
                    flc = sim.FromListConnector(conn)

                    proj = sim.Projection(self.lgn.pops[ch][pop]['output'],
                                          self.pops[dst_lyr][dst_pop],
                                          flc, target='excitatory')

                    input_projs[ch][pop][conn_key] = proj

        self.input_projs = input_projs
    
    
    
    def get_initial_input_weights(self):
        ws = {}
        for ch in self.input_conns:
            ws[ch] = {}
            for pop in self.input_conns[ch]:
                ws[ch][pop] = {}

                for cn_k in self.input_conns[ch][pop]:
                    dst_lyr, dst_pop = self.decode_in_conn_key(cn_k)
                    pre_size  = self.lgn.pop_size(pop)
                    post_size = self.pop_sizes[dst_lyr][dst_pop]
                    conn = self.input_conns[ch][pop][cn_k]
                    
                    ws[ch][pop][cn_k] = self.conn_list_to_array(conn, 
                                                                pre_size, 
                                                                post_size)
        return ws


    def get_input_weights(self, get_initial=False):
        if get_initial:
            return self.get_initial_input_weights()
        else:
            ws = {}
            for ch in self.input_projs:
                ws[ch] = {}
                for pop in self.input_projs[ch]:
                    ws[ch][pop] = {}

                    for cn_k in self.input_projs[ch][pop]:
                        w = self.input_projs[ch][pop][cn_k].\
                                              getWeights(format='array')
                        ws[ch][pop][cn_k] = np.array(w)
            return ws


    def connect_unit_to(self, unit):
        sim = self.sim
        cfg = self.cfg
        
        key = "%s"%unit
        w_min = self.min_weight
        w_max = cfg['inter_conn_weight']
        self.inter_projs[key] = self.get_fixed_prob_proj(self.output_pop, 
                                                         unit, 
                                                         cfg['inter_conn_prob'], 
                                                         w_min, w_max)
        


