from sim_tools.common import *
from sim_tools.connectors import kernel_connectors as conn_krn, \
                                 standard_connectors as conn_std
from default_config import defaults_v1 as defaults

import abc


class BaseColumn(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, lgn, width, height, location, learning_on, cfg):

        for k in defaults.keys():
            if k not in cfg.keys():
                cfg[k] = defaults[k]

        self.cfg = cfg
        self._v1_width  = width
        self._v1_height = height

        self.learn_on = learning_on
        self.lgn     = lgn
        self.retina  = lgn.retina
        self.width   = lgn.width 
        self.height  = lgn.height
        self.in_weight_func = cfg['col_weight_func']
        self.location = location
        self.recpt_width = cfg['in_receptive_width']
        
        ### these 3 should be pointers to self.pops
        self.input_pop    = None
        self.output_pop   = None
        self.feedback_pop = None


    def get_map_params(self, k):
        if k == 'cs4':
            kk = 'cs_quart'
        elif k == 'cs2':
            kk = 'cs_half'
        else:
            kk = k
        
        width  = self.lgn.pop_width(k)
        height = self.lgn.pop_height(k)
        step   = self.lgn.sample_step(k)
        start  = self.lgn.sample_start(k)
        
        if 'dir' not in kk:
            krn_width = self.retina.cfg[kk]['width']        
        else:
            krn_width = 0

        return step, start, width, height, krn_width

    def get_row_col_limits(self, half_krn_width):
        #location is in highest resolution scale (i.e. 'ctr_srr')
        hkw = half_krn_width
        fr_r = max(0, self.location[ROW] - hkw)
        to_r = min(self.height, self.location[ROW] + hkw + 1)
        fr_c = max(0, self.location[COL] - hkw)
        to_c = min(self.width,  self.location[COL] + hkw + 1)
        
        return {ROW:fr_r, COL:fr_c}, {ROW:to_r, COL:to_c}

    def build_input_indices_weights(self):
        cfg = self.cfg
        indices = {k: [] for k in self.lgn.output_keys()}
        weights = {k: [] for k in self.lgn.output_keys()}
        step    = 1; start   = 0; width   = 1; height  = 1
        sanity = {}; frm = {}; to = {}
        ssmp_r = 0; ssmp_c = 0
        my_r = self.location[ROW] #in full resolution space
        my_c = self.location[COL]
        
        to_delete = []
        half_rec_w = self.recpt_width//2
        for k in self.lgn.output_keys():
            # print("----------- KEY %s ---------------"%k)
            step, start, width, height, krn_width = self.get_map_params(k)
            half_krn_w = max(krn_width//2, half_rec_w)
            frm, to = self.get_row_col_limits(half_krn_w)
            sanity.clear()
            for r in range(frm[ROW], to[ROW]): #r in full resolution space
                ssmp_r = subsamp_size(start, r, step)
                if ssmp_r not in sanity:
                    sanity[ssmp_r] = []
                    
                for c in range(frm[COL], to[COL]): #c in full resolution space
                    ssmp_c = subsamp_size(start, c, step)
                    # print(ssmp_r, ssmp_c)
                    if ssmp_c in sanity[ssmp_r]:
                        continue
                    
                    src = int(ssmp_r*width + ssmp_c) # in subsample space (lgn pop)
                    print("%d*%d + %d = %d"%(ssmp_r, width, ssmp_c, src))
                    d = np.sqrt( (my_r - r)**2 + (my_c - c)**2 )#in full resolution
                    w = self.in_weight_func(d)

                    if w < cfg['min_scale_weight']:
                        # print("dist %s, weight %s"%(d, w))
                        continue
                    
                    sanity[ssmp_r].append(ssmp_c)                        
                    indices[k].append( src )
                    weights[k].append( np.abs(w) )
            
            if len(indices[k]) == 0:
                to_delete.append(k)

        for k in to_delete:
            # print("(%d, %d) %s"%(my_r, my_c, k))
            del indices[k]
            del weights[k]

        self.in_indices = {k: np.array(indices[k]) for k in indices}
        self.in_weights = {k: np.array(weights[k]) for k in weights}
    
    def get_synapse_dynamics(self):
        if not self.learn_on:
            return None
            
        cfg = self.cfg['stdp']
        sim = self.sim
        
        stdp_model = sim.STDPMechanism(
            timing_dependence = sim.SpikePairRule(tau_plus=cfg['tau_plus'], 
                                                  tau_minus=cfg['tau_minus']),
            weight_dependence = sim.AdditiveWeightDependence(w_min=cfg['w_min'], 
                                                             w_max=cfg['w_max'], 
                                                             A_plus=cfg['a_plus'], 
                                                             A_minus=cfg['a_minus']),
        )
        syn_dyn = sim.SynapseDynamics(slow=stdp_model)
        
        return syn_dyn
    
    @staticmethod
    def conn_list_to_array(conns, pre_size, post_size):
        ws = np.inf*np.ones((pre_size, post_size))
        print_debug( (pre_size, post_size) )
        for c in conns:
            print_debug(c)
            # ws[c[0], c[1]] = c[2]
        return ws

    @abc.abstractmethod
    def build_connectors(self):
        """create connector lists
           input (lgn) to column go in
           self.input_conns
           
           internal column connections go in
           self.intra_conns
        """
        pass
        
    @abc.abstractmethod
    def build_populations(self):
        """required PyNN Populations per column
           all populations should go into self.pops (dictionary)
           
           input, output and feedback populations MUST either exist or be
           aliased (e.g. pops['input'] = pops['my_input_pop'])
        """
        pass

    @abc.abstractmethod
    def build_projections(self):
        """create PyNN Projections acording to self.input_/intra_conns,
           these should be stored in
           self.input_projs and self.intra_conns
           respectively
        """
        pass

    @abc.abstractmethod
    def connect_unit_to(self, unit):
        """how to connect a unit (column) with other units(columns)
           should return a PyNN Projection
        """
        
        pass
