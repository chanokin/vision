from base_column import *


class V1SimpleColumn(BaseColumn):
    
    def __init__(self, sim, lgn, width, height, location, learning_on, cfg,
                 input_conns):
        # print("Building Four-to-One column (%d, %d)"%(row, col))
        BaseColumn.__init__(self, lgn, width, height, location, 
                            learning_on, cfg, input_conns)
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
        # if input_weights is None:
        #     self.build_input_indices_weights()
        # print("\t\t\tdone!")

        # print("\t\t\tbuilding connectors...")
        self.build_connectors(input_conns)
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
        minw, conw = 0., 0.
        for lyr in self.column_conn_prob:
            inner_conns[lyr] = {}
            for conn in self.column_conn_prob[lyr]:
                if '2' not in conn:
                    continue
                
                src, dst, dst_lyr = self.decode_conn_key(lyr, conn)
                conw = self.column_conn_wgt[lyr][conn]
                minw = self.min_weight if conw > 0 else -self.min_weight
                max_weight = max(minw, conw)
                min_weight = min(minw, conw)

                prob = self.column_conn_prob[lyr][conn]
                delay = cfg['min_delay']
                conns = prob_conn(self.pop_sizes[lyr][src],
                                  self.pop_sizes[dst_lyr][dst],
                                  prob, conw, delay,
                                  weight_std_dev=1.)
                # print(lyr, conn, conns)
                if len(conns):
                    inner_conns[lyr][conn] = sim.FromListConnector(conns)
                
        return inner_conns


    def build_input_conns(self, in_conns=None):
        cfg = self.cfg
        sim = self.sim
        list_prob_conn = conn_std.list_probability_connector
        key_to_pop = {'l0e': self.lgn.css}
        input_conns = {}
        dst_indices = []
        delay = cfg['input_delay']
        nid0, nidN = 0, 0
        for ch in self.lgn.channels:
            ch = self.lgn.channels[ch]
            input_conns[ch] = {}
            for pop in self.lgn.output_keys():
                input_conns[ch][pop] = {}
                for conn_key in self.input_conn_prob['main']:
                    if pop in key_to_pop[conn_key]:
                        if in_conns is None:
                            dst_lyr, dst_pop = self.decode_in_conn_key(conn_key)

                            sign   = 1 if dst_pop == 'exc' else -1
                            weight = sign*self.in_weights[pop]
                            dst_size = self.pop_sizes[dst_lyr][dst_pop]

                            nid0, nidN = 0, dst_size

                            dst_indices[:] = [i for i in range(nid0, nidN)]
                            prob = self.input_conn_prob['main'][conn_key]
                            conns = list_prob_conn(self.in_indices[pop],
                                                   dst_indices,
                                                   prob, weight, delay)
                        else:
                            conns = in_conns[ch][pop][conn_key]
                        # print("\n\n************** input conn %s, %s, "
                        #       "%s %s **************"%(ch, pop, conn_key, weight))
                        # print(conns)
                        if len(conns):
                            input_conns[ch][pop][conn_key] = \
                                                    sim.FromListConnector(conns)

        return input_conns
        

    def build_connectors(self, input_conns):
        self.intra_conns = self.build_intra_conns()
        # centre-surround to layer 4
        # direction to layer 2
        # gabor to layer 2
        self.input_conns = self.build_input_conns(input_conns)


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
        self.in_key  = 'l0'
        self.out_key = 'l0'
        self.fb_key  = 'l0'
        self.inter_key = 'l0'

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
        if key_is_true('noise_per_unit', cfg):
            lbl = 'noise population at %s, %s'%(self.location[ROW], self.location[COL])
            # print("Noise pop size = %d"%self.pop_sizes[self.in_key]['exc'])
            self.noise_pop = sim.Population(self.pop_sizes[self.in_key]['exc'],
                                            sim.SpikeSourcePoisson,
                                            {'rate': cfg['noise_rate'], 'start': 0,
                                             'duration': 100000000},
                                            label=lbl)

        self.pops = pops
        self.input_pop    = self.pops[self.in_key]['exc']
        self.output_pop   = self.pops[self.out_key]['exc']
        self.feedback_pop = self.pops[self.fb_key]['exc']
        self.interconnect_pop = self.pops[self.inter_key]['exc']

        if cfg['record']['voltages']:
            self.input_pop.record_v()
            self.output_pop.record_v()
            self.feedback_pop.record_v()

        if cfg['record']['spikes']:
            self.input_pop.record()
            self.output_pop.record()
            self.feedback_pop.record()
            self.noise_pop.record()


    def build_projections(self):
        sim = self.sim
        cfg = self.cfg
        intra_projs = {}
        for lyr in self.intra_conns:
            intra_projs[lyr] = {}
            for conn in self.intra_conns[lyr]:
                src, dst, dst_lyr = self.decode_conn_key(lyr, conn)
                tgt = 'excitatory' if src == 'exc' else 'inhibitory'
                flc = self.intra_conns[lyr][conn]
                
                syn_dyn = self.get_synapse_dynamics(src, dst)
                lbl = 'simple v1 intra %s-%s to %s-%s' % \
                                        (lyr, src, dst_lyr, dst)
                proj = sim.Projection(self.pops[lyr][src], 
                                      self.pops[dst_lyr][dst],
                                      flc, target=tgt,
                                      synapse_dynamics = syn_dyn,
                                      label=lbl)
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
                    # print("---- proj %s %s to %s %s -----------\n"%
                    #       (ch, pop, dst_lyr, dst_pop))
                    #only exc populations output in lgn
                    syn_dyn = self.get_synapse_dynamics('exc', dst_pop)
                    lbl = 'simple v1 input %s-%s to %s-%s' % \
                            (ch, pop, dst_lyr, dst_pop)

                    proj = sim.Projection(self.lgn.pops[ch][pop]['relay'],
                                          self.pops[dst_lyr][dst_pop],
                                          conn, target='excitatory',
                                          synapse_dynamics = syn_dyn,
                                          label=lbl)

                    input_projs[ch][pop][conn_key] = proj

        if key_is_true('noise_per_unit', cfg):
            # print("Noise in ")

            lbl = 'noise proj %s, %s' % \
                    (self.location[ROW], self.location[COL])
            conn = sim.OneToOneConnector(weights=cfg['noise_weight'],
                                         # generate_on_machine=True
                                         )
            self.noise_proj = sim.Projection(self.noise_pop,
                                     self.input_pop, conn,
                                     target='excitatory', label=lbl)

        self.input_projs = input_projs
    
    
    def get_initial_input_lists(self):
        in_lists = {}
        for ch in self.input_conns:
            in_lists[ch] = {}
            for pop in self.input_conns[ch]:
                in_lists[ch][pop] = {}

                for cn_k in self.input_conns[ch][pop]:
                    dst_lyr, dst_pop = self.decode_in_conn_key(cn_k)
                    pre_size  = self.lgn.pop_size(pop)
                    post_size = self.pop_sizes[dst_lyr][dst_pop]
                    conn = self.input_conns[ch][pop][cn_k]
                    in_lists[ch][pop][cn_k] = conn._conn_list

        return in_lists

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
        list_prob_conn = conn_std.list_probability_connector

        src_size = self.pop_sizes[self.inter_key]['exc']
        dst_size = unit.pop_sizes[unit.inter_key]['exc']

        key = "%s"%unit
        conn_w = cfg['inter_conn_weight']
        delay = cfg['inter_conn_weight']
        syn_dyn = self.get_synapse_dynamics('exc', 'exc')
        prob = cfg['inter_conn_prob']
        conns = list_prob_conn(range(src_size), range(dst_size),
                               prob, conn_w, delay)
        src_pop = self.pops[self.inter_key]['exc']
        dst_pop = unit.pops[unit.inter_key]['exc']

        self.inter_conns[key] = conns
        self.inter_projs[key] = sim.Projection(src_pop, dst_pop,
                                               sim.FromListConnector(conns),
                                               synapse_dynamics=syn_dyn,
                                               target='excitatory',
                                               label='%s to %s'%(self, unit))




