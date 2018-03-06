from sim_tools.common import *
from sim_tools.kernels import center_surround as krn_cs, \
                              gabor as krn_gbr, \
                              direction_detection as krn_dir

from sim_tools.connectors import kernel_connectors as conn_krn, \
                                 standard_connectors as conn_std, \
                                 direction_connectors as dir_conn, \
                                 mapping_funcs as mapf
from spynnaker7.pyNN.external_devices import \
    SpynnakerExternalDevicePluginManager as ex


from spike_tools.vis import plot_kernel

from scipy.signal import convolve2d, correlate2d

from default_config import defaults_retina as defaults

import sys


class Retina():
    _DEBUG = True
    def _dir_lyr(self, lyr):
        return "direction_%s"%(lyr)

    def _dir_key(self, ang):
        return "direction_%s"%(ang)

    def _orient_key(self, ang):
        return "orientation_%03d"%int(ang)

    def __init__(self, simulator, camera_pop, width, height, dvs_mode, 
                cfg=defaults):
        
        print("Building Retina (%d x %d)"%(width, height))

        for k in defaults.keys():
            if k not in cfg.keys() and ('cs' not in k) and ('dir' not in k):
            # if k not in cfg.keys():
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

        self.shapes = {'orig': (height, width),
                       'cam': {'width': width, 'height': height,
                               'size': width*height,
                               'step': 1, 'start': 0, }}

        # self.css = ['cs', 'cs2', 'cs4']
        self.css = [ k for k in cfg.keys() if 'cs' in k ]

        for ccss in self.css:
            key = self._right_key(ccss)
            self.shapes[key]  = self.gen_shape(width, height, 
                                               cfg[ccss]['step'], 
                                               cfg[ccss]['start'])

        if not key_is_false('direction', cfg):

            frm = cfg['direction']['sample_from']
            key = self._right_key('direction')

            for dir_str in cfg['direction']['keys']:
                c_step_w = 0
                c_step_h = 0
                c_start_w = 0
                c_start_h = 0
                if dir_str.upper() == 'N' or dir_str.upper() == 'S' \
                    or dir_str.upper() == 'NORTH' or dir_str.upper() == 'SOUTH':
                    step_w = cfg['direction']['bipolars']['step']
                    start_w = cfg['direction']['bipolars']['start']
                    step_h = 1
                    start_h = 0
                    c_step_w = step_w
                    c_start_w = start_w
                else:
                    step_h = cfg['direction']['bipolars']['step']
                    start_h = cfg['direction']['bipolars']['start']
                    step_w = 1
                    start_w = 0
                    c_step_h = step_h
                    c_start_h = start_h

                self.shapes[self._dir_key(dir_str)] = \
                    self.gen_shape(self.shapes[frm]['width'],
                        self.shapes[frm]['height'], step_w, start_w, step_h, start_h)

                if dir_str.upper() == 'N' or dir_str.upper() == 'S' \
                    or dir_str.upper() == 'NORTH' or dir_str.upper() == 'SOUTH':
                    step_h = cfg['direction']['ganglion']['step']
                    start_h = cfg['direction']['ganglion']['start']
                    step_w = 1
                    start_w = 0
                    c_step_h = step_h
                    c_start_h = start_h
                else:
                    step_w = cfg['direction']['ganglion']['step']
                    start_w = cfg['direction']['ganglion']['start']
                    step_h = 1
                    start_h = 0
                    c_step_w = step_w
                    c_start_w = start_w

                pre_key = self._dir_key(dir_str)
                self.shapes[self._dir_key(dir_str+'G')] = \
                    self.gen_shape(self.shapes[pre_key]['width'],
                        self.shapes[pre_key]['height'], step_w, start_w, step_h, start_h)

                self.shapes[self._dir_key(dir_str+'GC')] = {
                    'width': self.shapes[self._dir_key(dir_str + 'G')]['width'],
                    'height': self.shapes[self._dir_key(dir_str + 'G')]['height'],
                    'step': c_step_w, 'start': c_start_w,
                    'step_h': c_step_h, 'start_h': c_start_h,
                }

        if not key_is_false('orientation', cfg):
            frm = cfg['orientation']['sample_from']
            key = self._right_key('orientation')
            self.shapes[key] = self.gen_shape(self.shapes[frm]['width'], 
                                              self.shapes[frm]['height'], 
                                              cfg['orientation']['step'],
                                              cfg['orientation']['start'])

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
        # sys.exit()

        print(self.shapes)

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


    def gen_shape(self, width, height, step, start, stp_h=None, stt_h=None):
        if stp_h is None:
            stp_h = step
        if stt_h is None:
            stt_h = start
        w = subsamp_size(start, width,  step)
        h = subsamp_size(stt_h, height,  stp_h)
        sh = {  'width': w, 'height': h, 'size': w*h,
                'start': start, 'step': step, 'step_h': stp_h, 'start_h': stt_h}
        return sh


    def split_cam(self, in_cam, in_width, in_height, map_to_src):
        sim = self.sim
        cfg = self.cfg
        exc_cell = getattr(sim, cfg['exc_cell']['cell'], None)
        exc_parm = cfg['exc_cell']['params']
        on_pop  = sim.Population(in_width*in_height,
                                 exc_cell, exc_parm, label='Cam Split - On Channel')

        off_pop = sim.Population(in_width*in_height,
                                 exc_cell, exc_parm, label='Cam Split - Off Channel')
        if key_is_true('spikes', cfg['record']):
            on_pop.record()
            off_pop.record()

        row_start = 0
        if 'split_cam_off_arg' in cfg:
            if cfg['split_cam_off_arg'] == 'height':
                row_start = in_height

        if is_spinnaker(sim):
            print("\tMappingConnector")
            on_conn  = sim.MappingConnector(in_width, in_height, ON, cfg['height_bits'],
                                            channel_bits=cfg['channel_bits'],
                                            event_bits=cfg['event_bits'],
                                            weights=cfg['w2s'],
                                            generate_on_machine=True)

            off_conn = sim.MappingConnector(in_width, in_height, OFF, cfg['height_bits'],
                                            channel_bits=cfg['channel_bits'],
                                            event_bits=cfg['event_bits'],
                                            weights=cfg['w2s'],
                                            generate_on_machine=True)
        else:
            on_conns  = []
            off_conns = []
            for r in range(in_height):
                for c in range(in_width):
                    on_input = True
                    pre_idx = map_to_src(r, c, on_input, cfg['height_bits'])
                    post_idx = mapf.row_major(r, c, on_input, in_width)
                    on_conns.append((pre_idx, post_idx,
                                     cfg['w2s'], cfg['kernel_inh_delay']))

                    on_input = False
                    pre_idx = map_to_src(r, c, on_input, cfg['height_bits'],
                                         row_start)
                    post_idx = mapf.row_major(r, c, on_input, in_width)
                    off_conns.append((pre_idx, post_idx, cfg['w2s'],
                                      cfg['kernel_inh_delay']))

            on_conn  = sim.FromListConnector(on_conns)
            off_conn = sim.FromListConnector(off_conns)

        in2on = sim.Projection(in_cam, on_pop, on_conn, target='excitatory',
                               label='cam mapping - on')
        in2off = sim.Projection(in_cam, off_pop, off_conn, target='excitatory',
                                label='cam mapping - off')
        self._cam_map_projs = {self.channels[ON]: in2on,
                               self.channels[OFF]: in2off}
        return {self.channels[ON]:  on_pop, 
                self.channels[OFF]: off_pop}



    
    def _right_key(self, key, direction=None):
        if 'orientation' in key or 'orient' in key:
            return 'orientation'
        elif 'direction' in key or 'dir' in key:
            return key#'direction'
        else:
            return key
        
    def pop_size(self, key, dir=None):
        return self.shapes[self._right_key(key, dir)]['size']
    
    def pop_width(self, key, dir=None):
        return self.shapes[self._right_key(key, dir)]['width']
    
    def pop_height(self, key, dir=None):
        return self.shapes[self._right_key(key, dir)]['height']

    def sample_step(self, key, dir=None):
        return self.shapes[self._right_key(key, dir)]['step']
    
    def sample_start(self, key, dir=None):
        return self.shapes[self._right_key(key, dir)]['start']

    def pop_shape(self, key, dir=None):
        return (self.pop_height(key, dir), self.pop_width(key, dir))

    def steps(self, key, dir=None):
        return (self.shapes[self._right_key(key, dir)]['step_h'],
                self.shapes[self._right_key(key, dir)]['step'])

    def starts(self, key, dir=None):
        return (self.shapes[self._right_key(key, dir)]['start_h'],
                self.shapes[self._right_key(key, dir)]['start'])

    def get_output_keys(self):
        return [k for k in self.pops['on'] if ('cam' not in k)]


    def build_kernels(self):
        def cap_vals(k, mw):
            return k*(k > mw)

        # if os.path.isfile("kernel_correlation_cache.pickle.bz2"):
        #     d = load_compressed("kernel_correlation_cache.pickle")
        #     self.cs = d['ctr_srr']
        #     self.kernels = d['ctr_srr']
        #     self.corr = d['cs_corr']
        #     self.gab = d['gab']
        #     self.direction_kernels = d['direction_kernels']
        #     return

        krns = {}
        corr = {}
        cfg = self.cfg
        for ccss in self.css:
            krns[ccss] = krn_cs.gaussian2D(cfg[ccss]['width'],
                                           cfg[ccss]['std_dev'])
            
            # sum_pos = np.sum(krns[ccss][ krns[ccss] > 0 ])
            # inv_sum_pos = 1./sum_pos

            # krns[ccss] *= inv_sum_pos * cfg['w2s'] * cfg[ccss]['w2s_mult']
            # krns[ccss] *= inv_sum_pos * cfg[ccss]['w2s_mult']
            krns[ccss] = sum2one(krns[ccss])
            if key_is_true('flatten_centre-surround', cfg[ccss]):
                thr = cfg[ccss]['flatten_threshold']
                rows, cols = np.where(krns[ccss]>=thr)
                if key_is_true('square_centre-surround', cfg):
                    rmin = np.min(rows)
                    rmax = np.max(rows) + 1
                    cmin = np.min(cols)
                    cmax = np.max(cols) + 1
                    krns[ccss][rmin:rmax, cmin:cmax] = 1.
                else:
                    krns[ccss][rows, cols] = 1.
            # else:
            #     krns[ccss] /= np.sqrt(np.sum(krns[ccss]**2))

            krns[ccss] *= cfg['w2s'] * cfg[ccss]['w2s_mult']
            krns[ccss][:] = cap_vals(krns[ccss], cfg['min_weight'])

            if True or key_is_true('plot_kernels', cfg):
                plot_kernel(krns[ccss], ccss, diagonal=False)

        self.cs = krns

        def o2k(a):
            return self._orient_key(a)

        orient = {}
        orient_corr = {}
        if not key_is_false('orientation', cfg):

            w = cfg['orientation']['width']
            sx = cfg['orientation']['std_dev']
            sy = cfg['orientation']['std_dev']/cfg['orientation']['std_dev_div']
            for angle in cfg['orientation']['angles']:
                k = o2k(angle)
                orient[k] = krn_cs.gaussian2D(w, sx, sy, angle)
                if key_is_true('plot_kernels', cfg):
                    plot_kernel(
                        orient[k]*cfg['w2s']*cfg['orientation']['w2s_mult'], k)

            self.orient = orient
            
            for k in self.orient:
                orient_corr[k] = correlate2d(self.orient[k], self.orient[k],
                                             mode='same')
                orient_corr[k] /= np.sum(orient_corr[k])
                orient_corr[k] *= cfg['inhw'] * cfg['orientation']['w2s_mult']
                self.orient[k] *= cfg['w2s'] * cfg['orientation']['w2s_mult']

                if key_is_true('plot_kernels', cfg):
                    plot_kernel(-orient_corr[k], "corr_%s"%(k))
                    plot_kernel(orient[k] - orient_corr[k], "cs_%s"%(k))

                # orient_corr[k][w//2, w//2] = 0
                orient_corr[k] *= cfg['orientation']['inh_mult']

            self.orient_corr = orient_corr

        # if cfg['lateral_competition']:
        for cs0 in self.css:
            corr[cs0] = {}
            for cs1 in self.css:
                if cs0 != cs1 and key_is_false('lateral_competition', cfg):
                    continue

                if Retina._DEBUG:
                    print("\t\tGenerating correlation %s <-> %s"%(cs0, cs1))

                corr[cs0][cs1]  = correlate2d(krns[cs0], krns[cs1], 'same')

                # if key_is_true('flatten_centre-surround', cfg):
                #     corr[cs0][cs1]  = correlate2d(krns[cs0], krns[cs1], 'same')
                # else:
                #     corr[cs0][cs1]  = krn_cs.correlationGaussian2D(
                #                             cfg[cs0]['width'],   cfg[cs1]['width'],
                #                             cfg[cs0]['std_dev'], cfg[cs1]['std_dev'])

                corr[cs0][cs1] = sum2one(corr[cs0][cs1])
                corr[cs0][cs1] *= cfg['inhw']*cfg['corr_w2s_mult']

                if True or key_is_true('plot_kernels', cfg):
                    tmp = np.zeros_like(corr[cs0][cs1])
                    pad = (tmp.shape[0] - krns[cs0].shape[0])//2
                    tmp[:] = -corr[cs0][cs1]
                    tmp[pad:pad+krns[cs0].shape[0], pad:pad+krns[cs0].shape[0]] += krns[cs0]
                    plot_kernel(-corr[cs0][cs1], "corr_%s_to_%s"%(cs0, cs1))
                    plot_kernel(tmp, "cs_%s_to_%s"%(cs0, cs1))
                
                # corr[cs0][cs1][cfg[cs0]['width']//2, \
                #                cfg[cs0]['width']//2] = 0
                
                # corr[cs0][cs1] = cap_vals(corr[cs0][cs1], cfg['min_weight'])

        
        # for cs0 in self.css:
        #     self.cs[cs0] = cap_vals(self.cs[cs0], cfg['min_weight'])


        def d2k(a):
            return self._dir_key(a)

        dir_krns = {}
        if not key_is_false('direction', cfg):
            dir_krns['bipolars'] = {}
            dir_krns['ganglion'] = {}
            dir_krns['slow'] = {}
            dir_krns['fast'] = {}

            dkrn_g = krn_dir.direction_kernel
            dkrn_b = krn_dir.direction_subsamp
            d2a = krn_dir.dir_to_ang
            for direction in cfg['direction']['keys']:
                if Retina._DEBUG:
                    print("\t\tGenerating direction kernel %s"%direction)

                dk = d2k(direction)

                dir_krns['bipolars'][dk] = dkrn_b(cfg['direction']['bipolars']['width'],
                                            d2a(direction),
                                            cfg['direction']['bipolars']['in_w'])

                fast_dist = int(round(cfg['direction']['ganglion']['width'] * 2./11.))
                fast_dist += (fast_dist%2 == 0) # if it's even we force it to be odd

                min_delay = 1
                dir_krns['slow'][dk] = \
                    dkrn_g(cfg['direction']['detector']['width'],
                           cfg['direction']['detector']['width'],
                           min_delay,
                           cfg['direction']['detector']['slow_w'], d2a(direction),
                           cfg['direction']['angle'], cfg['direction']['delay_func'],
                           cfg['direction']['weight_func'])

                hlf_w = cfg['direction']['detector']['width'] // 2
                frm = hlf_w - fast_dist//2
                to = hlf_w + fast_dist//2 + 1
                dir_krns['slow'][dk][WEIGHT][frm:to, frm:to] = 0
                dir_krns['slow'][dk][DELAY][frm:to, frm:to] = 0
                # dir_krns['slow'][dk][DELAY][:] = dir_krns['slow'][dk][DELAY] > 0

                dir_krns['fast'][dk] = \
                    dkrn_g(fast_dist, fast_dist, min_delay,
                           cfg['direction']['detector']['fast_w'], d2a(direction),
                           cfg['direction']['angle'], cfg['direction']['delay_func'],
                           cfg['direction']['weight_func'])
                # dir_krns['fast'][dk][DELAY][:] = dir_krns['fast'][dk][DELAY] > 0

                ### haaack
                DX = 0
                hlf_ws = dir_krns['slow'][dk][WEIGHT].shape[1] // 2
                hlf_hs = dir_krns['slow'][dk][WEIGHT].shape[0] // 2
                hlf_wf = dir_krns['fast'][dk][WEIGHT].shape[1] // 2
                hlf_hf = dir_krns['fast'][dk][WEIGHT].shape[0] // 2
                for dx in range(-DX, DX+1):
                    if 'E' in dk or 'W' in dk:
                        for speed in ['slow', 'fast']:
                            if dir_krns[speed][dk][WEIGHT].shape[0] == 1:
                                continue
                            row = (hlf_hs if speed == 'slow' else hlf_hf)
                            dir_krns[speed][dk][WEIGHT][row + dx, :] = \
                                                dir_krns[speed][dk][WEIGHT][row, :]
                            dir_krns[speed][dk][DELAY][row + dx, :] = \
                                                dir_krns[speed][dk][DELAY][row, :]

                    if 'N' in dk or 'S' in dk:
                        for speed in ['slow', 'fast']:
                            if dir_krns[speed][dk][WEIGHT].shape[0] == 1:
                                continue
                            col = (hlf_ws if speed == 'slow' else hlf_wf)

                            dir_krns[speed][dk][WEIGHT][:, col + dx] = \
                                                dir_krns[speed][dk][WEIGHT][:, col]
                            dir_krns[speed][dk][DELAY][:, col + dx] = \
                                                dir_krns[speed][dk][DELAY][:, col]

                if 'E' in dk or 'W' in dk:
                    # Remove zero rows to reduce transfer times
                    for speed in ['slow', 'fast']:
                        dir_krns[speed][dk][WEIGHT] = \
                            dir_krns[speed][dk][WEIGHT][\
                                ~np.all(dir_krns[speed][dk][WEIGHT] == 0, axis=1)]
                        dir_krns[speed][dk][DELAY] = \
                            dir_krns[speed][dk][DELAY][\
                                ~np.all(dir_krns[speed][dk][DELAY] == 0, axis=1)]


                if 'S' in dk or 'N' in dk:
                    # Remove zero columns to reduce transfer times
                    for speed in ['slow', 'fast']:
                        dir_krns[speed][dk][WEIGHT] = \
                            dir_krns[speed][dk][WEIGHT][\
                                ~np.all(dir_krns[speed][dk][WEIGHT] == 0, axis=0)]
                        dir_krns[speed][dk][DELAY] = \
                            dir_krns[speed][dk][DELAY][\
                                ~np.all(dir_krns[speed][dk][DELAY] == 0, axis=0)]


                if True or key_is_true('plot_kernels', cfg):
                    print('WEIGHTS', dir_krns['bipolars'][dk][WEIGHT])
                    plot_kernel(dir_krns['bipolars'][dk][WEIGHT],
                                "direction_kernel_bipolar_weight_%s"%(direction),
                                diagonal=False)
                    print('DELAYS', dir_krns['bipolars'][dk][DELAY])
                    plot_kernel(dir_krns['bipolars'][dk][DELAY],
                                "direction_kernel_bipolar_delay_%s"%(direction),
                                diagonal=False)

                    print('WEIGHTS', dir_krns['slow'][dk][WEIGHT])
                    plot_kernel(dir_krns['slow'][dk][WEIGHT],
                                "direction_kernel_weight_slow_%s"%(direction),
                                diagonal=False)
                    print('DELAYS', dir_krns['slow'][dk][DELAY])
                    plot_kernel(dir_krns['slow'][dk][DELAY],
                                "direction_kernel_delay_slow_%s"%(direction),
                                diagonal=False)

                    print('WEIGHTS', dir_krns['fast'][dk][WEIGHT])
                    plot_kernel(dir_krns['fast'][dk][WEIGHT],
                                "direction_kernel_weight_fast_%s"%(direction),
                                diagonal=False)
                    print('DELAYS', dir_krns['fast'][dk][DELAY])
                    plot_kernel(dir_krns['fast'][dk][DELAY],
                                "direction_kernel_delay_fast_%s"%(direction),
                                diagonal=False)

        # import sys
        # sys.exit(1)

        dump_compressed({'ctr_srr': krns, 'cs_corr': corr, 
                         'orientation': orient, 'orientation_corr': orient_corr,
                         'direction_kernels': dir_krns},
                         "kernel_correlation_cache.pickle")

        self.direction_kernels = dir_krns
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
                if Retina._DEBUG:
                    print("\t\tcentre-surround %s: %s"%(c, k))

                on_path = (c == 'on')
                step = self.sample_step(k)
                start = self.sample_start(k)
                post_shape = (shapes[k]['height'], shapes[k]['width'])
                krn = self.kernels[k]

                if is_spinnaker(self.sim):
                    exc = sim.KernelConnector(self.shapes['orig'], post_shape,
                                              krn.shape,
                                              post_sample_steps=(step, step), 
                                              post_start_coords=(start, start),
                                              weights=krn*(krn > 0), 
                                              delays=cfg['kernel_exc_delay'],
                                              generate_on_machine=True)
                    inh = False


                self.conns[c][k] = { EXC: exc, INH: inh }

        if not key_is_false('orientation', cfg):
            for c in self.conns:
                for k in self.orient.keys():
                    if Retina._DEBUG:
                        print("\t\torientation %s: %s"%(c, k))
                    rk = self._right_key(k)
                    krn = self.orient[k]
                    on_path = (c == 'on')
                    step = self.sample_step(self._right_key(k))
                    start = self.sample_start(self._right_key(k))

                    if is_spinnaker(self.sim):
                        sfrm = cfg['orientation']['sample_from']
                        pre_shape = (shapes[sfrm]['height'], shapes[sfrm]['width'])
                        post_shape = (shapes[rk]['height'], shapes[rk]['width'])
                        exc =  sim.KernelConnector(pre_shape, post_shape,
                                                   krn.shape,
                                                   post_sample_steps=(step, step),
                                                   post_start_coords=(start, start),
                                                   weights=krn*(krn > 0),
                                                   delays=cfg['kernel_exc_delay'],
                                                   generate_on_machine=True)
                        # exc =  sim.OneToOneConnector(weights=cfg['w2s'],
                        #                              delays=cfg['kernel_exc_delay'])
                        conn = { EXC: exc, 
                                 INH: None  }

                    self.conns[c][k] = conn

        if not key_is_false('direction', cfg):
            def d2k(a):
                return self._dir_key(a)
            for c in self.channels:
                ch = self.channels[c]
                for dk in cfg['direction']['keys']:
                    if Retina._DEBUG:
                        print("\t\tdirection: %s"%(dk))

                    k = d2k(dk)
                    frm = cfg['direction']['sample_from']
                    pre_key = self._right_key(frm)
                    pre_shape = self.pop_shape(pre_key)
                    post_key = self._right_key(k)
                    step = self.steps(post_key)
                    start = self.starts(post_key)
                    post_shape = self.pop_shape(post_key)

                    krn = self.direction_kernels['bipolars'][k][WEIGHT]
                    dly = 1 + cfg['kernel_exc_delay']

                    # BIPOLARS --- INTEGRATORS
                    if is_spinnaker(self.sim):
                        # print(krn.shape)
                        exc = sim.KernelConnector(pre_shape, post_shape, krn.shape,
                            post_sample_steps=step, post_start_coords=start,
                            weights=krn*(krn > 0), delays=dly,
                            generate_on_machine=True)

                        self.conns[ch][k] = {EXC: exc}


######################################

        # for c in self.conns['on'][dk]:
            # print(c)

        self.extra_conns = {}
        #cam to inh-version of cam
        # if self.dvs_mode = dvs_modes[MERGED]:
        if is_spinnaker(sim):
            conns = sim.OneToOneConnector(weights=cfg['w2s'],
                                      delays=cfg['kernel_inh_delay'],
                                      generate_on_machine=True)
        else:
            conns = sim.OneToOneConnector(weights=cfg['w2s'],
                                      delays=cfg['kernel_inh_delay'])
        self.extra_conns['o2o'] = conns
        
        #bipolar to interneuron 
        self.extra_conns['inter'] = {}
        for k in css:
            if is_spinnaker(sim):
                conns = sim.OneToOneConnector(weights=cfg['w2s'],
                                              delays=cfg['kernel_inh_delay'],
                                              generate_on_machine=True)
            else:
                conns = sim.OneToOneConnector(weights=cfg['w2s'],
                                              delays=cfg['kernel_inh_delay'])
            self.extra_conns['inter'][k] = conns
        
        if not key_is_false('orientation', cfg):
            if is_spinnaker(sim):
                conns = sim.OneToOneConnector(weights=cfg['w2s'],
                                              delays=cfg['kernel_inh_delay'],
                                              generate_on_machine=True)
            else:
                conns = sim.OneToOneConnector(weights=cfg['w2s'],
                                              delays=cfg['kernel_inh_delay'])            

            self.extra_conns['inter']['orientation'] = conns
        
        if not key_is_false('direction', cfg):
            # print_debug('attempting to build direction EXTRA connectors')
            if is_spinnaker(sim):
                conns = sim.OneToOneConnector(weights=cfg['w2s'],
                                              delays=cfg['kernel_inh_delay'],
                                              generate_on_machine=True)
            else:
                conns = sim.OneToOneConnector(weights=cfg['w2s'],
                                              delays=cfg['kernel_inh_delay'])

            self.extra_conns['inter']['dir'] = conns
        
        #competition between same kind (size) of cells
        self.extra_conns['ganglion'] = {}
        for k in css:

            #correlation between two of the smallest (3x3) centre-surround
            krn = self.corr[k][k]
            dly = np.ones_like(krn)*(krn > 0)*cfg['kernel_inh_delay']
            dly[dly.shape[0]//2, dly.shape[1]//2] += cfg['kernel_exc_delay']
            # print("---------------------------------------------------------")
            # print(krn)
            # print("---------------------------------------------------------")
            if is_spinnaker(self.sim):
                post_shape = (shapes[k]['height'], shapes[k]['width'])
                shape_common = self.shapes['orig']
                pre_step = (shapes[k]['step'], shapes[k]['step'])
                pre_start = (shapes[k]['start'], shapes[k]['start'])
                exc =  sim.OneToOneConnector(weights=cfg['w2s'],
                                             delays=cfg['kernel_exc_delay'],
                                             generate_on_machine=True)

                inh = sim.KernelConnector(post_shape, post_shape, krn.shape,
                                          shape_common=shape_common,
                                          pre_sample_steps=pre_step,
                                          pre_start_coords=pre_start,
                                          post_sample_steps=pre_step,
                                          post_start_coords=pre_start,
                                          weights=krn*(krn > 0), 
                                          delays=dly,
                                          generate_on_machine=True)

                conn = { EXC: exc, INH: inh }

                # conn ={ EXC: sim.FromListConnector(conns[EXC]),
                # conn ={ EXC: sim.OneToOneConnector(weights=cfg['w2s'],
                #                                    delays=cfg['kernel_exc_delay']),
                #         INH: sim.FromListConnector(conns[INH]) }

            self.extra_conns['ganglion'][k] = conn


        if not key_is_false('orientation', cfg):
            shape = (self.pop_width('orientation'), self.pop_height('orientation'))
            shape_common = self.shapes['orig']
            pre_step = (shapes['orientation']['step'], shapes['orientation']['step'])
            pre_start = (shapes['orientation']['start'], shapes['orientation']['start'])

            for k in self.orient_corr:
                krn = self.orient_corr[k]
                # print(krn)
                # print(krn*(krn > 0))
                if is_spinnaker(self.sim):

                    exc = sim.OneToOneConnector(weights=cfg['w2s'],
                                                delays=cfg['kernel_exc_delay'],
                                                generate_on_machine=True)
                    inh = sim.KernelConnector(shape, shape, krn.shape,
                                              shape_common=shape_common,
                                              pre_sample_steps=pre_step,
                                              pre_start_coords=pre_start,
                                              post_sample_steps=pre_step,
                                              post_start_coords=pre_start,
                                              weights=krn*(krn > 0),
                                              delays=cfg['kernel_inh_delay'],
                                              generate_on_machine=True)

                conn = {EXC: exc, INH: inh}

                self.extra_conns['ganglion'][k] = conn
        
        if not key_is_false('direction', cfg):

            for dir_str in cfg['direction']['keys']:
                _key = 'direction_%s'%(dir_str)
                self.extra_conns['ganglion'][_key] = {}

                post_key = _key+'G'
                pre_shape = self.pop_shape(_key)
                post_shape = self.pop_shape(post_key)
                start = self.starts(post_key)
                step = self.steps(post_key)

                # SLOW - NMDA-like
                krn = self.direction_kernels['slow'][_key][WEIGHT]
                dly = (self.direction_kernels['slow'][_key][DELAY] + \
                        cfg['kernel_exc_delay']) * (krn > 0)

                exc = sim.KernelConnector(pre_shape, post_shape, krn.shape,
                        post_sample_steps=step, post_start_coords=start,
                        weights=krn * (krn > 0), delays=dly,
                        generate_on_machine=True)

                self.extra_conns['ganglion'][_key]['slow'] = exc

                # FAST - standard excitatory

                krn = self.direction_kernels['fast'][_key][WEIGHT]
                dly = (self.direction_kernels['fast'][_key][DELAY] + \
                       cfg['kernel_exc_delay']) * (krn > 0)

                exc = sim.KernelConnector(pre_shape, post_shape, krn.shape,
                        post_sample_steps=step, post_start_coords=start,
                        weights=krn * (krn > 0), delays=dly,
                        generate_on_machine=True)

                self.extra_conns['ganglion'][_key]['fast'] = exc

                self.extra_conns['ganglion'][_key]['preferred'] = \
                                sim.OneToOneConnector(
                                    weights=cfg['direction']['ganglion']['in_w'],
                                    delays=1.,
                                    generate_on_machine=True)

                if cfg['direction']['inh_w_scale'] == 0:
                    self.extra_conns['ganglion'][_key]['opposite'] = None
                else:
                    krn = np.ones((3,3)) * cfg['w2s'] * cfg['direction']['inh_w_scale']
                    shp = self.pop_shape(_key)
                    step = (1, 1)
                    start = (0, 0)
                    self.extra_conns['ganglion'][_key]['opposite'] = \
                                    sim.KernelConnector(shp, shp, krn.shape,
                                        post_start_coords=start, post_sample_steps=step,
                                        weights=krn,
                                        delays=1., generate_on_machine=True)


        #bipolar/interneuron to ganglion (use row-major mapping)
        self.lat_conns = {}
        if not key_is_false('lateral_competition', cfg):
            for cs0 in self.css:
                self.lat_conns[cs0] = {}
                for cs1 in self.css:
                    if cs0 == cs1:
                        continue

                    if Retina._DEBUG:
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
                    dly = np.ones_like(krn) * (krn > 0) * cfg['kernel_inh_delay']
                    dly[dly.shape[0] // 2, dly.shape[1] // 2] += cfg['kernel_exc_delay']

                    if is_spinnaker(self.sim):

                        conns = {}

                        conns[INH] = sim.KernelConnector(pre_shape, post_shape,
                                                 krn.shape,
                                                 shape_common=common_shape,
                                                 pre_sample_steps=pre_step,
                                                 pre_start_coords=pre_start,
                                                 post_sample_steps=post_step,
                                                 post_start_coords=post_start,
                                                 weights=np.abs(krn*(krn > 0)), 
                                                 delays=dly,
                                                 generate_on_machine=True )


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
            # if key_is_true('voltages', cfg['record']):
            #     self.pops[k]['cam_inter'].record_v()
            #
            # if key_is_true('spikes', cfg['record']):
            #     self.pops[k]['cam_inter'].record()
        

        for k in self.conns.keys():
            for p in self.conns[k].keys():
                if 'cs' in p:
                    bip_size = inter_size = gang_size = self.pop_size(p)
                    params = cfg[p]['params']
                    g_params = exc_parm
                elif 'dir' in p:
                    bip_size = inter_size = self.pop_size(p)
                    gang_size = self.pop_size(p+'G')
                    # bip_size = inter_size = gang_size = self.pop_size(p)
                    params = cfg['direction']['bipolars']['params']
                    g_params = cfg['direction']['ganglion']['params']
                else:
                    bip_size = inter_size = gang_size = self.pop_size(p)
                    params = exc_parm
                    g_params = params

                self.pops[k][p] = {'bipolar': sim.Population(
                                            bip_size, exc_cell, params,
                                            label='Retina: bipolar_%s_%s'%(k, p)),

                                    'inter': sim.Population(
                                            inter_size, inh_cell, inh_parm,
                                            label='Retina: inter_%s_%s'%(k, p)),

                                    'ganglion': sim.Population(
                                            gang_size, exc_cell, g_params,
                                            label='Retina: ganglion_%s_%s'%(k, p)),
                                  }
                if 'dir' in p:
                    params = cfg['direction']['detector']['params']
                    self.pops[k][p]['detector'] = sim.Population(
                                                gang_size, sim.IF_curr_enabler, params,
                                                label='Retina: detector_%s_%s'%(k, p))

                if key_is_true('voltages', cfg['record']):
                    self.pops[k][p]['bipolar'].record_v()
                    self.pops[k][p]['inter'].record_v()
                    self.pops[k][p]['ganglion'].record_v()
                    try:
                        self.pops[k][p]['detector'].record_v()
                    except:
                        pass

                if key_is_true('spikes', cfg['record']):
                    self.pops[k][p]['bipolar'].record()
                    self.pops[k][p]['inter'].record()
                    self.pops[k][p]['ganglion'].record()
                    try:
                        self.pops[k][p]['detector'].record()
                    except:
                        pass


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
            # conn = self.extra_conns['o2o']
            conn = sim.OneToOneConnector(weights=cfg['w2s'],
                                         generate_on_machine=True)
            exc = sim.Projection(pre, post, conn, target='excitatory',
                                 label='cam to cam inter %s'%k)
            self.projs[k]['cam_inter']['cam2intr'] = [exc]

        # photo to bipolar, 
        # bipolar to interneurons and 
        # bipolar/inter to ganglions
        for ch in self.conns.keys():
            for p in self.conns[ch].keys():
                print("\t\t%s channel - %s filter"%(ch, p))
                self.projs[ch][p] = {}

                if 'cs' in p:
                    exc_src = self.cam[ch]

                elif 'orient' in p:
                    frm = cfg['orientation']['sample_from']
                    if 'cam' == frm:
                        exc_src = self.cam[ch]
                    else:
                        exc_src = self.pops[ch][frm]['ganglion']

                elif 'dir' in p:
                    frm = cfg['direction']['sample_from']
                    if 'cam' == frm:
                        exc_src = self.cam[ch]
                    else:
                        exc_src = self.pops[ch][frm]['ganglion']

                conn = self.conns[ch][p][EXC]
                exc = sim.Projection(exc_src, self.pops[ch][p]['bipolar'],
                                     conn, target='excitatory',
                                     label='input to bipolar EXC %s-%s'%(ch, p))

                if INH in self.conns[ch][p] and self.conns[ch][p][INH]:
                    if 'cs' in p:
                        frm = 'cam_inter'
                        pop = self.pops[ch][frm]
                    elif 'orient' in p:
                        frm = cfg['orientation']['sample_from']
                        pop = self.pops[ch][frm]['inter']
                    elif 'dir' in p:
                        frm = cfg['direction']['sample_from']
                        if 'cam' == frm:
                            pop = self.cam[ch]['cam_inter']
                        else:
                            pop = self.pops[ch][frm]['inter']
                        
                    inh = sim.Projection(pop, self.pops[ch][p]['bipolar'],
                                         conn, target='inhibitory',
                                         label='input to bipolar INH %s-%s' % (ch, p))

                    self.projs[ch][p]['cam2bip'] = [exc, inh]
                else:
                    self.projs[ch][p]['cam2bip'] = [exc]
                
        
                if 'cs' in p:
                    _pre = self.pops[ch][p]['bipolar']
                    _post = self.pops[ch][p]['ganglion']
                    _post_inh = self.pops[ch][p]['inter']
                    inter  = self.extra_conns['inter'][p]
                    _exc = self.extra_conns['ganglion'][p][EXC]
                    _inh = self.extra_conns['ganglion'][p][INH]
                elif 'orient' in p:
                    _pre = self.pops[ch][p]['bipolar']
                    _post = self.pops[ch][p]['ganglion']
                    _post_inh = self.pops[ch][p]['inter']
                    inter = self.extra_conns['inter']['orientation']
                    _exc = self.extra_conns['ganglion'][p][EXC]
                    _inh = self.extra_conns['ganglion'][p][INH]
                    # print(inter)
                    # print(_exc)
                    # print(_inh)
                elif 'dir' in p:
                    _pre = self.pops[ch][p]['detector']
                    _post = self.pops[ch][p]['ganglion']
                    _post_inh = self.pops[ch][p]['inter']
                    inter  = self.extra_conns['inter']['dir']
                    _exc = self.extra_conns['ganglion'][p]['preferred']

                exc = sim.Projection(_pre, _post_inh, inter, target='excitatory',
                                     label='bipolar to inter %s-%s' % (ch, p))
                
                self.projs[ch][p]['bip2intr'] = [exc]

                exc = sim.Projection(_pre, _post, _exc, target='excitatory',
                                     label='bipolar to ganglion %s-%s' % (ch, p))
                
                if _inh:
                    inh = sim.Projection(_post_inh, _post, _inh, target='inhibitory',
                                         label='inter to ganglion inh %s-%s' % (ch, p))
                else:
                    inh = None    

                self.projs[ch][p]['bip2gang'] = [exc, inh]

                if 'dir' in p:
                    _pre = self.pops[ch][p]['bipolar']
                    _post = self.pops[ch][p]['detector']
                    _conn = self.extra_conns['ganglion'][p]['slow']
                    nmda = sim.Projection(_pre, _post, _conn, target='enabler',
                                          label='nmda-like %s-%s'%(ch, p))

                    _conn = self.extra_conns['ganglion'][p]['fast']
                    exc = sim.Projection(_pre, _post, _conn, target='excitatory',
                                         label='EXCitatory %s-%s' % (ch, p))
                    self.projs[ch][p]['bip2detect'] = [exc, nmda]

                    if 'E' in p:
                        pre_key = p.replace('E', 'W')
                    elif 'W' in p:
                        pre_key = p.replace('W', 'E')
                    elif 'S' in p:
                        pre_key = p.replace('S', 'N')
                    elif 'N' in p:
                        pre_key = p.replace('N', 'S')

                    if pre_key in self.pops[ch].keys():
                        _pre = self.pops[ch][pre_key]['detector']
                        _post = self.pops[ch][p]['ganglion']
                        _conn = self.extra_conns['ganglion'][p]['opposite']
                        if _conn is None:
                            continue

                        inh = sim.Projection(_pre, _post, _conn, target='inhibitory',
                                             label='opposite dir inh - %s - %s'%(ch, p))
                        self.projs[ch][p]['opp2gang'] = [inh]


        # competition between centre-surround ganglion cells
        lat = {}
        if 'lateral_competition' in cfg and cfg['lateral_competition']:
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
                        # conn = self.lat_conns[cs0][cs1][EXC] 

                        # prjs[EXC] = sim.Projection(pre, post, conn,
                        #                     target='excitatory',
                        #                     label='lateral exc %s -> %s'%(cs0, cs1))
                        
                        conn = self.lat_conns[cs0][cs1][INH] 
                        prjs[INH] = sim.Projection(pre, post, conn,
                                            target='inhibitory',
                                            label='lateral inh %s -> %s'%(cs0, cs1))

                        lat[ch][cs0][cs1] = prjs

        self.projs['lateral'] = lat

        import os
        import pprint

        with open(os.path.join(os.getcwd(), "output_projections.txt"), "w+") as f:
            pp = pprint.PrettyPrinter(indent=4, stream=f)
            pp.pprint(self.projs)
    
    
    def retina_live_out_setup(self, ports):
        for ch in self.conns.keys():
            for pop in self.conns[ch].keys():
                ex.activate_live_output_for(self.pops[ch][pop]['ganglion'],
                                            host="0.0.0.0", port=ports[ch][pop])

    def get_spikes(self):
        spikes = {}
        if key_is_true('spikes', self.cfg['record']):

            for ch in self.conns.keys():
                spikes[ch] = {}
                spikes[ch]['cam'] = self.cam[ch].getSpikes(compatible_output=True)

                for pop in self.conns[ch].keys():

                    bipolar  = self.pops[ch][pop]['bipolar'].\
                                    getSpikes(compatible_output=True)
                    ganglion = self.pops[ch][pop]['ganglion'].\
                                    getSpikes(compatible_output=True)
                    inter    = self.pops[ch][pop]['inter'].\
                                    getSpikes(compatible_output=True)

                    spikes[ch][pop] = {'bipolar': bipolar, 'inter': inter,
                                       'ganglion': ganglion}

        return spikes
