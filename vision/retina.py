from sim_tools.common import *
from sim_tools.kernels import center_surround as krn_cs, gabor as krn_gbr
from sim_tools.connectors import kernel_connectors as conn_krn, \
                                 standard_connectors as conn_std, \
                                 direction_connectors as dir_conn, \
                                 mapping_funcs as mapf

from spike_tools.vis import plot_kernel

from scipy.signal import convolve2d, correlate2d

from default_config import defaults_retina as defaults

import sys


class Retina():
    
    def __init__(self, simulator, camera_pop, width, height, dvs_mode, 
                cfg=defaults):
        
        print("Building Retina (%d x %d)"%(width, height))
        
        for k in defaults.keys():
            if k not in cfg.keys():
                cfg[k] = defaults[k]

        self.sim = simulator
        self.cfg = cfg

        self.width = width
        self.height = height

        self.dvs_mode = dvs_mode
        if self.dvs_mode == dvs_modes[0]:
            self.on_idx = 0
            self.off_idx = width*height
        else:
            self.on_idx = 0
            self.off_idx = 0

        self.shapes = {'orig': (height, width)}

        # self.css = ['cs', 'cs2', 'cs4']
        self.css = [ k for k in cfg.keys() if 'cs' in k ]

        for ccss in self.css:
            self.shapes[ccss]  = self.gen_shape(width, height, 
                                                cfg[ccss]['step'],
                                                cfg[ccss]['start'])

        if 'direction' in cfg and cfg['direction']:
            self.shapes['dir'] = self.gen_shape(width, height, 
                                                cfg['direction']['step'],
                                                cfg['direction']['start'])

        if 'gabor' in cfg and cfg['gabor']:
            self.shapes['gabor'] = self.gen_shape(width, height, 
                                                 cfg['gabor']['step'],
                                                 cfg['gabor']['start'])

            self.ang_div = deg2rad(180./cfg['gabor']['num_divs'])
            self.angles = [i*self.ang_div for i in range(cfg['gabor']['num_divs'])]

        # print(self.shapes)

        self.channels = {ON: 'on', OFF: 'off'}

        if dvs_mode == dvs_modes[MERGED]:
            self.cam = self.split_cam(camera_pop, width, height, 
                                        cfg['input_mapping_func'])
        else:
            self.cam = {self.channels[ON]:  camera_pop[ON],
                        self.channels[OFF]: camera_pop[OFF], }

        self.mapping_f = cfg['input_mapping_func']

        print("\tBuilding kernels...")
        self.build_kernels()
        print("\t\tdone!")

        print("\tBuilding connectors...")
        self.build_connectors()
        print("\t\tdone!")

        print("\tBuilding populations...")
        self.build_populations()
        # import pprint
        # pprint.pprint(self.pops)
        print("\t\tdone!")

        print("\tBuilding projections...")
        self.build_projections()
        # import pprint
        # pprint.pprint(self.projs)
        print("\t\tdone!")


    def split_cam(self, in_cam, in_width, in_height, map_to_src):
        sim = self.sim
        cfg = self.cfg
        exc_cell = getattr(sim, cfg['exc_cell']['cell'], None)
        exc_parm = cfg['exc_cell']['params']
        on_pop  = sim.Population(in_width*in_height,
                                 exc_cell, exc_parm, 'Cam: On Channel')
        on_pop.record()
        off_pop = sim.Population(in_width*in_height,
                                 exc_cell, exc_parm, 'Cam: Off Channel')
        off_pop.record()

        row_start = 0
        if 'split_cam_off_arg' in cfg:
            if cfg['split_cam_off_arg'] == 'height':
                row_start = in_height

        on_conns  = []
        off_conns = []
        for r in range(in_height):
            for c in range(in_width):
                on_input = True
                pre_idx = map_to_src(r, c, on_input, cfg['row_bits'])
                post_idx = mapf.row_major(r, c, on_input, in_width)
                on_conns.append((pre_idx, post_idx, 
                                 cfg['w2s'], cfg['kernel_inh_delay']))

                on_input = False
                pre_idx = map_to_src(r, c, on_input, cfg['row_bits'], 
                                     row_start)
                post_idx = mapf.row_major(r, c, on_input, in_width)
                off_conns.append((pre_idx, post_idx, cfg['w2s'], 
                                  cfg['kernel_inh_delay']))
        
        sim.Projection(in_cam, on_pop, 
                        sim.FromListConnector(on_conns))
        sim.Projection(in_cam, off_pop, 
                        sim.FromListConnector(off_conns))

        return {self.channels[ON]:  on_pop, 
                self.channels[OFF]: off_pop}


    def gen_shape(self, width, height, step, start):
        w = subsamp_size(start, width,  step)
        h = subsamp_size(start, height,  step)
        sh = {  'width': w, 'height': h, 'size': w*h,
                'start': start, 'step': step}
        return sh
    
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


    def get_output_keys(self):
        return [k for k in self.pops['on'] if k is not 'cam_inter']


    def build_kernels(self):
        def a2k(a):
            return 'gabor_%d'%( int( a ) )
        if os.path.isfile("kernel_correlation_cache.pickle"):
            d = pickle.load(open("kernel_correlation_cache.pickle", 'rb'))
            self.cs = d['ctr_srr']
            self.kernels = d['ctr_srr']
            self.corr = d['cs_corr']
            self.gab  = d['gab']
            return 

        krns = {}
        corr = {}
        cfg = self.cfg
        for ccss in self.css:
            krns[ccss] = krn_cs.center_surround_kernel(cfg[ccss]['width'],
                                                       cfg[ccss]['std_dev'], 
                                                       cfg[ccss]['sd_mult'])
            
            sum_pos = np.sum(krns[ccss][ krns[ccss] > 0 ])
            inv_sum_pos = 1./sum_pos

            krns[ccss] *= inv_sum_pos * cfg['w2s'] * cfg[ccss]['w2s_mult']

            if 'plot_kernels' in cfg and cfg['plot_kernels']:
                plot_kernel(krns[ccss], ccss)

        self.cs = krns

        gab = None
        if 'gabor' in cfg and cfg['gabor']:
            angles = self.angles
            gab = krn_gbr.multi_gabor(cfg['gabor']['width'], angles, 
                                      cfg['gabor']['std_dev'], 
                                      cfg['gabor']['freq'])

            self.gab = {a2k(k): gab[k]*cfg['w2s'] 
                                        for k in gab.keys()}
            
            gab_corr = {}
            for g0 in self.gab:
                for g1 in self.gab:
                    gab_corr[g0][g1] = convolve2d(self.gab[g0], self.gab[g1], 
                                                  mode='same')

        # if cfg['lateral_competition']:
        for cs0 in self.css:
            corr[cs0] = {}
            for cs1 in self.css:
                corr[cs0][cs1]  = correlate2d(krns[cs0], krns[cs1], mode='same')
                corr[cs0][cs1][:] = sum2zero(corr[cs0][cs1])
                corr[cs0][cs1][:], _ = conv2one(corr[cs0][cs1])
                corr[cs0][cs1] *= cfg['w2s']*cfg['corr_w2s_mult']

                if 'plot_kernels' in cfg and cfg['plot_kernels']:
                    plot_kernel(corr[cs0][cs1], "corr_%s_to_%s"%(cs0, cs1))
        
        pickle.dump({'ctr_srr': krns, 'cs_corr': corr, 'gab': gab},
                    open("kernel_correlation_cache.pickle", 'wb'))

        self.kernels = krns
        self.corr = corr

    def build_connectors(self):
        cfg = self.cfg
        sim = self.sim
        shapes = self.shapes
        self.conns = {self.channels[ch]: {} for ch in self.channels}
        mapping_f = mapf.row_major
        css = self.css
        krn_conn = conn_krn.full_kernel_connector
        for c in self.conns:
            for k in css:
                print("\t\tcentre-surround %s: %s"%(c, k))
                on_path = (c == 'on')
                step = self.sample_step(k)
                start = self.sample_start(k)
                post_shape = (shapes[k]['height'], shapes[k]['width'])
                krn = self.kernels[k]

                if self.sim.__name__ == 'pyNN.spiNNaker':

                    exc = sim.KernelConnector(self.shapes['orig'], post_shape,
                                              krn.shape,
                                              post_sample_steps=(step, step), 
                                              post_start_coords=(start, start),
                                              weights=krn*(krn > 0), 
                                              delays=cfg['kernel_exc_delay'], 
                                              generate_on_machine=True)

                    # inh = sim.KernelConnector(self.shapes['orig'], post_shape,
                    #                         (step, step), (start, start),
                    #                         krn*(krn < 0), 
                    #                         cfg['kernel_inh_delay'], 
                    #                         generate_on_machine=True)
                    inh = False
                    conn = { EXC: exc, 
                             INH: inh 
                           }
                else:
                    conns = krn_conn(self.width, self.height, 
                                     self.kernels[k],
                                     exc_delay=cfg['kernel_exc_delay'],
                                     inh_delay=cfg['kernel_inh_delay'],
                                     col_step=step, row_step=step,
                                     col_start=start, row_start=start, 
                                     map_to_src=mapping_f,
                                     pop_width=self.width,
                                     on_path=on_path)
                    conn = { EXC: sim.FromListConnector(conns[EXC]),
                            #  INH: sim.FromListConnector(conns[INH]),
                             INH: False,
                           }

                self.conns[c][k] = conn

        if 'gabor' in cfg and cfg['gabor']:
            for c in self.conns:
                for k in self.gab.keys():
                    print("\t\tgabor %s: %s"%(c, k))
                    krn = self.gab[k]
                    on_path = (c == 'on')
                    step = self.sample_step(k)
                    start = self.sample_start(k)

                    if self.sim.__name__ == 'pyNN.spiNNaker':
                        pre_shape = self.shapes['orig']
                        post_shape = (shapes[k]['height'], shapes[k]['width'])
                        exc =  sim.KernelConnector(pre_shape, post_shape,
                                                   (step, step), 
                                                   (start, start),
                                                   krn*(krn > 0),
                                                   cfg['kernel_exc_delay'], 
                                                   generate_on_machine=True)
                        inh =  sim.KernelConnector(pre_shape, post_shape,
                                                   (step, step), 
                                                   (start, start),
                                                   krn*(krn < 0),
                                                   cfg['kernel_exc_delay'], 
                                                   generate_on_machine=True)
                        conn = { EXC: exc, 
                                #  INH: inh 
                               }
                    else:
                        conns = krn_conn(self.width, self.height, krn,
                                         cfg['kernel_exc_delay'],
                                         cfg['kernel_inh_delay'],
                                         col_step=step, row_step=step,
                                         col_start=start, row_start=start, 
                                         map_to_src=mapping_f,
                                         row_bits=cfg['row_bits'],
                                         on_path=on_path) 

                        conn ={ EXC: sim.FromListConnector(conns[EXC]),
                                # INH: sim.FromListConnector(conns[INH]) 
                              }

                    self.conns[c][k] = conn

        if 'direction' in cfg and cfg['direction']:
            for dk in cfg['direction']['keys']:
                print("\t\tdirection: %s"%(dk))
                k = "%s_dir"%dk
                step = self.sample_step(k)
                start = self.sample_start(k)
                if '2' in dk  or  dk.isupper():
                    dir_cn = dir_conn.direction_connection_angle
                    conns = dir_cn(dk, 
                                   cfg['direction']['angle'],
                                   cfg['direction']['dist'], 
                                   self.width, self.height, 
                                   mapping_f,
                                   start=start, step=step,
                                   exc_delay=cfg['kernel_exc_delay'],
                                   inh_delay=cfg['kernel_inh_delay'],
                                   delay_func=cfg['direction']['delay_func'], 
                                   weight=cfg['direction']['weight'],
                                   row_bits=cfg['row_bits'],)
                else:
                    dir_cn = dir_conn.direction_connection
                    conns = dir_cn(dk, \
                                   self.width, self.height,
                                   cfg['direction']['div'],
                                   cfg['direction']['delays'],
                                   cfg['direction']['weight'],
                                   mapping_f)
                # print(conns)
                self.conns['on'][k], self.conns['off'][k] = conns

######################################

        # for c in self.conns['on'][dk]:
            # print(c)

        self.extra_conns = {}
        #cam to inh-version of cam
        # if self.dvs_mode = dvs_modes[MERGED]:
        if sim.__name__ == 'pyNN.spiNNaker':
            conns = sim.OneToOneConnector(weights=cfg['inhw'], 
                                      delays=cfg['kernel_inh_delay'],
                                      generate_on_machine=True)
        else:
            conns = sim.OneToOneConnector(weights=cfg['inhw'], 
                                      delays=cfg['kernel_inh_delay'])
        self.extra_conns['o2o'] = conns
        
        #bipolar to interneuron 
        self.extra_conns['inter'] = {}
        for k in css:
            if sim.__name__ == 'pyNN.spiNNaker':
                conns = sim.OneToOneConnector(weights=cfg['inhw'], 
                                        delays=cfg['kernel_inh_delay'],
                                        generate_on_machine=True)
            else:
                conns = sim.OneToOneConnector(weights=cfg['inhw'], 
                                        delays=cfg['kernel_inh_delay'])
            self.extra_conns['inter'][k] = conns
        
        if 'gabor' in cfg and cfg['gabor']:
            size = self.pop_size('gabor')
            if sim.__name__ == 'pyNN.spiNNaker':
                conns = sim.OneToOneConnector(weights=cfg['inhw'], 
                                        delays=cfg['kernel_inh_delay'],
                                        generate_on_machine=True)
            else:
                conns = sim.OneToOneConnector(weights=cfg['inhw'], 
                                    delays=cfg['kernel_inh_delay'])            
                                    
            self.extra_conns['inter']['gabor'] = conns
        
        if 'direction' in cfg and cfg['direction']:
            # print_debug('attempting to build direction EXTRA connectors')
            size = self.pop_size('dir')
            if sim.__name__ == 'pyNN.spiNNaker':
                conns = sim.OneToOneConnector(weights=cfg['inhw'], 
                                        delays=cfg['kernel_inh_delay'],
                                        generate_on_machine=True)
            else:
                conns = sim.OneToOneConnector(weights=cfg['inhw'], 
                                        delays=cfg['kernel_inh_delay'])

            self.extra_conns['inter']['dir'] = conns
        
        #competition between same kind (size) of cells
        self.extra_conns['ganglion'] = {}
        for k in css:

            #correlation between two of the smallest (3x3) centre-surround
            krn = self.corr['cs']['cs']
            print("---------------------------------------------------------")
            print(krn)
            print("---------------------------------------------------------")
            if self.sim.__name__ == 'pyNN.spiNNaker':
                post_shape = (shapes[k]['height'], shapes[k]['width'])

                exc = sim.KernelConnector(post_shape, post_shape, krn.shape,
                                          weights=krn*(krn > 0), 
                                          delays=cfg['kernel_exc_delay'], 
                                          generate_on_machine=True)

                inh = sim.KernelConnector(post_shape, post_shape, krn.shape,
                                          weights=-krn*(krn < 0), 
                                          delays=cfg['kernel_inh_delay'], 
                                          generate_on_machine=True)

                conn = { EXC: exc, INH: inh }
            else:
                conns = krn_conn(shapes[k]['width'], shapes[k]['height'], 
                                 krn,
                                 exc_delay=cfg['kernel_exc_delay'],
                                 inh_delay=cfg['kernel_inh_delay'],
                                 map_to_src=mapf.row_major,
                                 pop_width=shapes[k]['width'])
                conn ={ EXC: sim.FromListConnector(conns[EXC]),
                        INH: sim.FromListConnector(conns[INH]) }

            self.extra_conns['ganglion'][k] = conn


        if 'gabor' in cfg and cfg['gabor']:
            w, h = self.pop_width('gabor'), self.pop_height('gabor')
            conns = conn_krn.full_kernel_connector(w, h,
                                                    self.corr['cs'],
                                                    cfg['kernel_exc_delay'],
                                                    cfg['kernel_inh_delay'],
                                                    map_to_src=mapf.row_major,
                                                    pop_width=w)
            
            conn = {EXC: sim.FromListConnector(conns[EXC]),
                    INH: sim.FromListConnector(conns[INH]),}
            
            self.extra_conns['ganglion']['gabor'] = conn
        
        if 'direction' in cfg and cfg['direction']:
            w, h = self.pop_width('dir'), self.pop_height('dir')
            conns = conn_krn.full_kernel_connector(w, h,
                                                    self.corr['cs'],
                                                    cfg['kernel_exc_delay'],
                                                    cfg['kernel_inh_delay'],
                                                    map_to_src=mapf.row_major,
                                                    pop_width=w)

            conn = {EXC: sim.FromListConnector(conns[EXC]),
                    INH: sim.FromListConnector(conns[INH]),}

            self.extra_conns['ganglion']['dir'] = conn

        #bipolar/interneuron to ganglion (use row-major mapping)
        self.lat_conns = {}
        if 'lateral_competition' in cfg and cfg['lateral_competition']:
            for cs0 in self.css:
                self.lat_conns[cs0] = {}
                for cs1 in self.css:
                    if cs0 == cs1:
                        continue

                    print("\t\tlateral competition: %s -> %s"%(cs0, cs1))

                    w0, h0 = self.pop_width(cs0), self.pop_height(cs0)
                    w1, h1 = self.pop_width(cs1), self.pop_height(cs1)
                    pre_shape  = (h0, w0)
                    post_shape = (h1, w1)
                    common_shape = (self.height, self.width)
                    pre_start = (self.sample_start(cs0), self.sample_start(cs0))
                    pre_step  = (self.sample_step(cs0), self.sample_step(cs0))
                    post_start = (self.sample_start(cs1), self.sample_start(cs1))
                    post_step  = (self.sample_step(cs1), self.sample_step(cs1))
                    krn = self.corr[cs0][cs1]
                    if self.sim.__name__ == 'pyNN.spiNNaker':

                        conns = {}
                        conns[EXC] = sim.KernelConnector(pre_shape, post_shape,
                                                 krn.shape,
                                                 shape_common=common_shape,
                                                 pre_sample_steps=pre_step,
                                                 pre_start_coords=pre_start,
                                                 post_sample_steps=post_step,
                                                 post_start_coords=post_start,
                                                 weights=krn*(krn > 0), 
                                                 delays=cfg['kernel_exc_delay'], 
                                                 generate_on_machine=True )

                        conns[INH] = sim.KernelConnector(pre_shape, post_shape,
                                                 krn.shape,
                                                 shape_common=common_shape,
                                                 pre_sample_steps=pre_step,
                                                 pre_start_coords=pre_start,
                                                 post_sample_steps=post_step,
                                                 post_start_coords=post_start,
                                                 weights=np.abs(krn*(krn < 0)), 
                                                 delays=cfg['kernel_inh_delay'], 
                                                 generate_on_machine=True )

                    else:
                        conn = conn_krn.competition_connector(self.width, self.height,
                                                    pre_shape, pre_start, pre_step,
                                                    post_shape, post_start, post_step,
                                                    krn, cfg['kernel_exc_delay'], 
                                                    cfg['kernel_inh_delay'])

                        
                        # print(conn)
                        # sys.exit(0)

                        if len(conn[EXC]) == 0 or len(conn[INH]) == 0:
                            continue

                        conns = {EXC: sim.FromListConnector(conn[EXC]),
                                INH: sim.FromListConnector(conn[INH])} 
                    
                    self.lat_conns[cs0][cs1] = conns


    def build_populations(self):
        self.pops = {}
        sim = self.sim
        cfg = self.cfg
        exc_cell = getattr(sim, cfg['exc_cell']['cell'], None)
        exc_parm = cfg['exc_cell']['params']
        inh_cell = getattr(sim, cfg['inh_cell']['cell'], None)
        inh_parm = cfg['inh_cell']['params']

        for k in self.conns.keys(): 
            # print("populations for channel %s"%k)
            self.pops[k] = {}
            self.pops[k]['cam_inter'] = sim.Population(self.width*self.height,
                                                        inh_cell, inh_parm,
                                                        label='cam_inter_%s'%k)
            if cfg['record']['voltages']:
                self.pops[k]['cam_inter'].record_v()

            if cfg['record']['spikes']:
                self.pops[k]['cam_inter'].record()
        

        for k in self.conns.keys():
            for p in self.conns[k].keys():
                filter_size = self.pop_size(p)
                self.pops[k][p] = {'bipolar': sim.Population(
                                            filter_size, exc_cell, exc_parm,
                                            label='Retina: bipolar_%s_%s'%(k, p)),

                                    'inter': sim.Population(
                                            filter_size, inh_cell, inh_parm,
                                            label='Retina: inter_%s_%s'%(k, p)),

                                    'ganglion': sim.Population(
                                            filter_size, exc_cell, exc_parm,
                                            label='Retina: ganglion_%s_%s'%(k, p)),
                                } 
                if cfg['record']['voltages']:
                    self.pops[k][p]['bipolar'].record_v()
                    self.pops[k][p]['inter'].record_v()
                    self.pops[k][p]['ganglion'].record_v()

                if cfg['record']['spikes']:
                    self.pops[k][p]['bipolar'].record()
                    self.pops[k][p]['inter'].record()
                    self.pops[k][p]['ganglion'].record()

    def build_projections(self):
        self.projs = {}
        cfg = self.cfg
        sim = self.sim
        
        #on/off photoreceptors interneuron projections (for inhibition)
        for k in self.channels:
            k = self.channels[k]
            self.projs[k] = {}
            self.projs[k]['cam_inter'] = {}
            pre  = self.cam[k]
            post = self.pops[k]['cam_inter']
            conn = self.extra_conns['o2o']
            exc = sim.Projection(pre, post, conn, target='excitatory')            
            self.projs[k]['cam_inter']['cam2intr'] = [exc]

        # photo to bipolar, 
        # bipolar to interneurons and 
        # bipolar/inter to ganglions
        for k in self.conns.keys():
            for p in self.conns[k].keys():
                print("\t\t%s channel - %s filter"%(k, p))
                self.projs[k][p] = {}
                exc_src = self.cam[k]
                conn = self.conns[k][p][EXC] 
                exc = sim.Projection(exc_src, self.pops[k][p]['bipolar'],
                                     conn, target='excitatory')

                if self.conns[k][p][INH]:
                    inh = sim.Projection(self.pops[k]['cam_inter'], 
                                         self.pops[k][p]['bipolar'],
                                         conn, target='inhibitory')

                    self.projs[k][p]['cam2bip'] = [exc, inh]
                else:
                    self.projs[k][p]['cam2bip'] = [exc]
                
        
                if 'cs' in p:
                    inter  = self.extra_conns['inter'][p]
                    cs_exc = self.extra_conns['ganglion'][p][EXC]
                    cs_inh = self.extra_conns['ganglion'][p][INH]
                if 'gabor' in p:
                    inter  = self.extra_conns['inter']['gabor']
                    cs_exc = self.extra_conns['ganglion']['gabor'][EXC]
                    cs_inh = self.extra_conns['ganglion']['gabor'][INH]
                elif 'dir' in p:
                    inter  = self.extra_conns['inter']['dir']
                    cs_exc = self.extra_conns['ganglion']['dir'][EXC]
                    cs_inh = self.extra_conns['ganglion']['dir'][INH]
                    
                exc = sim.Projection(self.pops[k][p]['bipolar'], 
                                     self.pops[k][p]['inter'],
                                     inter,
                                     target='excitatory')
                
                self.projs[k][p]['bip2intr'] = [exc]

                exc = sim.Projection(self.pops[k][p]['bipolar'], 
                                     self.pops[k][p]['ganglion'],
                                     cs_exc,
                                     target='excitatory')

                inh = sim.Projection(self.pops[k][p]['inter'], 
                                     self.pops[k][p]['ganglion'],
                                     cs_inh,
                                     target='inhibitory')
                # inh = None
                self.projs[k][p]['bip2gang'] = [exc, inh]
        
        # competition between centre-surround ganglion cells
        lat = {}
        for k in self.channels:
            ch = self.channels[k]
            lat[ch] = {}
            for cs0 in self.lat_conns:
                lat[ch][cs0] = {}
                for cs1 in self.lat_conns[cs0]:
                    print("\t\tlateral competition %s: %s -> %s"%
                          (ch, cs0, cs1))
                    prjs = {}
                    pre = self.pops[ch][cs0]['ganglion']
                    post = self.pops[ch][cs1]['ganglion']
                    conn = self.lat_conns[cs0][cs1][EXC] 

                    prjs[EXC] = sim.Projection(pre, post, conn,
                                        target='excitatory',
                                        label='lateral exc %s -> %s'%(cs0, cs1))
                    
                    conn = self.lat_conns[cs0][cs1][INH] 
                    prjs[INH] = sim.Projection(pre, post, conn,
                                        target='inhibitory',
                                        label='lateral inh %s -> %s'%(cs0, cs1))

                    lat[ch][cs0][cs1] = prjs

        self.projs['lateral'] = lat

    
    
