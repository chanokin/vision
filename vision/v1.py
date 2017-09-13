from sim_tools.common import *
from column import *
from default_config import defaults_v1 as defaults
import sys



class V1():
    
    def __init__(self, sim, lgn, learning_on, cfg=defaults, input_conns=None):
        
        for k in defaults.keys():
            if k not in cfg.keys():
                cfg[k] = defaults[k]

        self.sim = sim
        self.cfg = cfg
        self.lgn = lgn
        self.retina = lgn.retina
        self.learn_on = learning_on
        self.width   = lgn.width  # full input resolution
        self.height  = lgn.height # full input resolution
        self.complex_recp_width = cfg['complex_recp_width']
        self.unit_type = cfg['unit_type']
        self.Unit = self.unit_object()
        
        # 3x3 -> [[o o o][o o(r,c) o][o o o]]
        print("Building V1...")
        self.build_units(input_conns)
        if cfg['inter_unit_connect']:
            self.connect_units()


    def unit_object(self):
        if self.unit_type == 'autoencoder':
            return V1AutoEncoderColumn
        elif self.unit_type == 'liquid_state':
            return V1MultiColumn
        elif self.unit_type == 'four-to-one':
            return V1FourToOneColumn
        else:
            return V1SimpleColumn


    def build_units(self, input_conns):
        cfg = self.cfg
        in_start = cfg['in_receptive_start']
        in_step  = cfg['in_receptive_step']
        cols = []
        total_cols = subsamp_size(in_start, self.width,  in_step)*\
                     subsamp_size(in_start, self.height, in_step)
        num_steps = 60
        cols_to_steps = float(num_steps)/total_cols
        
        print("\t%d Columns..."%(total_cols))
        ### COLUMNS (input interface)  ---------------------------------
        prev_step = 0
        curr_col = 0
        units = {}
        sys.stdout.write("\t\tInput/Output/Feedback layers \n")
        sys.stdout.flush()

        for r in range(in_start, self.height, in_step):
            ur = subsamp_size(in_start, r, in_step)
            units[ur] = {}
            for c in range(in_start, self.width, in_step):
                # print_debug(("column ", r, c))
                if input_conns is not None and \
                    r in input_conns and c in input_conns[r]:
                    in_conn_list = input_conns[r][c]
                else:
                    in_conn_list = None

                coords = {ROW: r, COL: c}

                unit = self.Unit(self.sim, self.lgn, 
                                 self.width, self.height, 
                                 coords, self.learn_on, cfg=cfg,
                                 input_conns=in_conn_list)
                uc = subsamp_size(in_start, c, in_step)
                units[ur][uc] = unit
                
                # curr_col += 1
                # curr_step = int(curr_col*cols_to_steps)
                # if curr_step > prev_step:
                #     prev_step = curr_step
                #     sys.stdout.write("\r%03d, %03d"%(r, c))
                #     sys.stdout.flush()
                sys.stdout.write("\r\t\t\t%03d, %03d" % (r, c))
                sys.stdout.flush()

        sys.stdout.write("\n\n")
        sys.stdout.flush()
        self.units = units

        # sys.exit(0)
        # self.connect_units()

    def connect_units(self):
        cfg = self.cfg
        num_steps = 60
        in_start = cfg['in_receptive_start']
        in_step  = cfg['in_receptive_step']
        total_cols = subsamp_size(in_start, self.width,  in_step)*\
                     subsamp_size(in_start, self.height, in_step)
        cols_to_steps = float(num_steps)/total_cols
        prev_step = 0
        curr_col = 0
        sys.stdout.write("\t\t\t Unit-to-unit")
        sys.stdout.flush()
        
        hw = cfg['inter_conn_width']//2
        for r in sorted(self.units.keys())[1:-2]:
            for c in sorted(self.units[r].keys())[1:-2]:
                for ir in range(r-hw, r+hw+1):
                    for ic in range(c-hw, c+hw+1):
                        self.units[r][c].connect_unit_to(self.units[ir][ic])

                curr_col += 1
                curr_step = int(curr_col*cols_to_steps)
                if curr_step > prev_step:
                    prev_step = curr_step
                    sys.stdout.write(".")
                    sys.stdout.flush()

        sys.stdout.write("\n")


    def get_out_spikes(self):
        out_spikes = {}
        for r in sorted(self.units.keys()):
            out_spikes[r] = {}
            for c in sorted(self.units[r].keys()):
                try:
                    out_spikes[r][c] = self.units[r][c].output_pop.\
                                            getSpikes(compatible_output=True)
                except:
                    print("\n\nUnable to get spikes for V1 (%d, %d) output" %
                          (r, c))

        return out_spikes


    def get_input_weights(self, get_initial=False):
        in_weights = {}
        for r in sorted(self.units.keys()):
            in_weights[r] = {}
            for c in sorted(self.units[r].keys()):
                try:
                    in_weights[r][c] = self.units[r][c].get_input_weights(get_initial)
                except:
                    print("\n\nUnable to get weights for V1 (%d, %d) output" %
                          (r, c))

        return in_weights

    def updated_in_conn_lists(self, end_weights):
        new_conn_lists = {}
        for r in sorted(self.units.keys()):
            new_conn_lists[r] = {}
            for c in sorted(self.units[r].keys()):
                new_conn_lists[r][c] = {}
                old_in_lists = self.units[r][c].get_initial_input_lists()

                for ch in old_in_lists:
                    new_conn_lists[r][c][ch] = {}
                    for pop in old_in_lists[ch]:
                        new_conn_lists[r][c][ch][pop] = {}
                        for conn_key in old_in_lists[ch][pop]:
                            new_conns = []
                            ws = end_weights[r][c][ch][pop][conn_key]
                            for src, tgt, w, d in  old_in_lists[ch][pop][conn_key]:
                                if src >= ws.shape[0] or tgt >= ws.shape[1]:
                                    continue

                                new_conns.append((src, tgt, ws[src, tgt], d))

                            new_conn_lists[r][c][ch][pop][conn_key] = new_conns

        return new_conn_lists