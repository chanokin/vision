from sim_tools.connectors.mapping_funcs import  row_col_to_input, \
                                                row_col_to_input_breakout
import numpy as np
frame_rate = 100
dir_delay = int(1000./frame_rate)

exc_cell = "IF_curr_exp"
exc_cell_params = { 'cm': 0.35,  # nF
                  'i_offset': 0.0,
                  'tau_m': 20.0,
                  'tau_refrac': 2.0,
                  'tau_syn_E': 1.,
                  'tau_syn_I': 1.,
                  'v_reset': -70.0,
                  'v_rest': -65.0,
                  'v_thresh': -55.4
                  }

inh_cell = "IF_curr_exp"
inh_cell_params = { 'cm': 0.35,  # nF
                  'i_offset': 0.0,
                  'tau_m': 20.0,
                  'tau_refrac': 1.0,
                  'tau_syn_E': 1.,
                  'tau_syn_I': 1.,
                  'v_reset': -70.0,
                  'v_rest': -65.0,
                  'v_thresh': -58.
                  }

wta_inh_cell = "IF_curr_exp"
wta_inh_cell_params = { 'cm': 0.3,  # nF
                        'i_offset': 0.0,
                        'tau_m': 4.0,
                        'tau_refrac': 2.0,
                        'tau_syn_E': 2.,
                        'tau_syn_I': 1.,
                        'v_reset': -70.0,
                        'v_rest': -65.0,
                        'v_thresh': -58.
                        }


g_w2s = 4.376069
inh_w2s = 4.376069
dir_w2s = 2.
ssamp_w2s = 4.376069

defaults_retina = {
                # 'kernel_width': 3,
                'kernel_exc_delay': 3.,
                'kernel_inh_delay': 1.,
                'corr_self_delay': 4.,
                'corr_w2s_mult': 1.,
                'min_weight': 0.01,
                'row_step': 1, 'col_step': 1,
                'start_row': 0, 'start_col': 0,
                # 'gabor': {'num_divs': 2., 'freq': 5., 'std_dev': 5., 'width': 7,
                            # 'step': 3, 'start': 0},

                'cs': {'std_dev': 0.8, 'sd_mult': 6.7, 'width': 3, 
                       'step': 1, 'start':1, 'w2s_mult':1.},
                'cs2': {'std_dev': 0.865492, 'sd_mult': 6.63, 'width': 7,
                        'step': 2, 'start': 2, 'w2s_mult': 1.},
                'cs4': {'std_dev': 1.353551, 'sd_mult': 6.18, 'width': 15,
                        'step': 5, 'start': 5, 'w2s_mult': 1.},
                'cs8': {'std_dev': 3.809901, 'sd_mult': 5.57, 'width': 31,
                        'step': 10, 'start': 15, 'w2s_mult': 1.},
                # #retina receives 1 spike per change, needs huge weights
                # 'cs': {'std_dev': 0.8, 'sd_mult': 6.7, 'width': 3, 
                #        'step': 1, 'start':0, 'w2s_mult':1.},
                # 'cs2': {'std_dev': 0.865492, 'sd_mult': 6.63, 'width': 7,
                #         'step': 1, 'start':0, 'w2s_mult': 2.5},
                # 'cs4': {'std_dev': 1.353551, 'sd_mult': 6.18, 'width': 15,
                #         'step': 2, 'start': 0, 'w2s_mult': 4.},
                # 'cs8': {'std_dev': 3.809901, 'sd_mult': 5.57, 'width': 31,
                #         'step': 2, 'start': 0, 'w2s_mult': 12.},
                'w2s': g_w2s, 
                'inhw': inh_w2s,
                'inh_cell': {'cell': inh_cell,
                            'params': inh_cell_params,
                            }, 
                'exc_cell': {'cell': exc_cell,
                            'params': exc_cell_params,
                            },
                'record': {'voltages': False, 
                            'spikes': False,
                            },

                'direction': {'keys': [
                                        'E', 
                                        # 'W',
                                        # 'N', 
                                        # 'S',
                                        #'NW', 'SW', 'NE', 'SE',
                                        #'east', 'south', 'west', 'north',
                                        #'south east', 'south west', 
                                        #'north east', 'north west'
                                      ],
                            'div': 4,#6,
                            'weight': dir_w2s,
                            'delays': [1, 4, 6, 8],#, 3, 4 ],
                            'subsamp': 1,#2,
                            'w2s': ssamp_w2s,
                            'angle': 11,
                            'dist': 4,
                            'delay_func': lambda dist: dir_delay*dist, 
                                            #20ms = 1000/framerate
                            'step': 3,
                            'start': 0,
                            },

                # 'input_mapping_func': row_col_to_input_breakout,
                'input_mapping_func': row_col_to_input,
                'row_bits': 6,
                'lateral_competition': True,
                'split_cam_off_arg': False, #'height',
                'plot_kernels': False,
                }


#######################################################################
####################         L G N          ###########################
#######################################################################

defaults_lgn = { 'kernel_width': 3,
                 'kernel_exc_delay': 2.,
                 'kernel_inh_delay': 1.,
                 'row_step': 1, 'col_step': 1,
                 'start_row': 0, 'start_col': 0,
                 'gabor': {'num_divs': 7., 'freq': 5., 'std_dev': 1.1},
                 'ctr_srr': {'std_dev': 0.8, 'sd_mult': 6.7} ,
                 'w2s': g_w2s*1.1,
                 'inh_cell': {'cell': inh_cell,
                              'params': inh_cell_params
                             }, 
                 'exc_cell': {'cell': exc_cell,
                              'params': exc_cell_params
                             },
                 'record': {'voltages': False, 
                              'spikes': True,
                           },
                 'lat_inh': False,
               }



#######################################################################
####################           V 1          ###########################
#######################################################################

# unit_type = 'autoencoder'
# unit_type = 'liquid_state'
unit_type = 'four_to_one'


#from A Statistical Analysis of Information-Processing Properties of 
#Lamina-SpecificCortical Microcircuit Models
pop_ratio = {'l2': {'inh': 0.2, 'exc': 0.8},
             'l4': {'inh': 0.2, 'exc': 0.8},
             'l5': {'inh': 0.2, 'exc': 0.8},
            }


column_conn_wgt = {'l2': {'exc2inh': 1.90, 'inh2exc': -0.65,
                          'exc2exc': 1.70, 'inh2inh': -1.35,
                          'exc2l5e': 1.40, 'inh2l5e': -5.20,
                          'exc2l4i': 1.60,
                         },
                   'l4': {'exc2inh': 3.70, 'inh2exc': -0.85,
                          'exc2exc': 1.10, 'inh2inh': -1.55,
                          'exc2l2e': 4.00, 'inh2l2e': -1.75,
                          'inh2l2i': -1.5,
                         },
                   'l5': {'exc2inh': 0.90, 'inh2exc': -1.20,
                          'exc2exc': 1.70, 'inh2inh': -1.20,
                          'exc2l2e': 0.30,
                         },
                  }
w_conv = (g_w2s*1.1)/5.2 #5.2 is abs max weight in dict
for l in column_conn_wgt:
    for c in column_conn_wgt[l]:
        column_conn_wgt[l][c] *= w_conv

column_conn_prob = {'l2': {'exc2inh': 0.21, 'inh2exc': 0.16,
                           'exc2exc': 0.26, 'inh2inh': 0.25,
                           'exc2l5e': 0.55, 'inh2l5e': 0.20,
                           'exc2l4i': 0.08,
                          },
                    'l4': {'exc2inh': 0.19, 'inh2exc': 0.10,
                           'exc2exc': 0.17, 'inh2inh': 0.50,
                           'exc2l2e': 0.28, 'inh2l2e': 0.50,
                           'inh2l2i': 0.20,
                          },
                    'l5': {'exc2inh': 0.10, 'inh2exc': 0.12,
                           'exc2exc': 0.09, 'inh2inh': 0.60,
                           'exc2l2e': 0.03,
                          },
                   }

input_conn_prob = {'main':  {'l2e': 0.20,
                             'l4e': 0.80, 'l4i': 0.50,
                             'l5e': 0.10,
                            },
                   'extra': {'l2e': 0.20,
                            }
                  }
neurons_in_column = {'l2': 40,
                     'l4': 100,
                     'l5': 40}
defaults_v1 = { 'unit_type': unit_type,
                'w2s': g_w2s,
                'pop_ratio': pop_ratio,
                'column_conn_prob': column_conn_prob,
                'column_conn_wgt': column_conn_wgt,
                'input_conn_prob': input_conn_prob,
                'neurons_in_column': neurons_in_column,
                'inter_conn_prob': input_conn_prob['extra']['l2e'],
                'inter_conn_weight': g_w2s*0.5,
                'inter_conn_width': 3,
                'input_delay': 1,
                'context_in_weight': 0.3,
                'context_to_context_weight': 0.5, 
                'context_to_simple_weight': 1., 
                'min_delay': 2.,
                'max_delay': 14.,
                'max_weight': 1.7,
                'wta_inh_cell': { 'cell': wta_inh_cell,
                                  'params': wta_inh_cell_params,
                                }, 
                'inh_cell': {'cell': inh_cell,
                             'params': inh_cell_params,
                            }, 
                'exc_cell': {'cell': exc_cell,
                             'params': exc_cell_params,
                            },
                'record': {'voltages': False, 
                           'spikes': False,
                          },
                'lat_inh': False,
                'stdp': {'tau_plus': 20,
                         'tau_minus': 20,
                         'w_max': 0.25,
                         'w_min': 0.,
                         'a_plus': 0.1,
                         'a_minus': 0.12,
                        },
                'in_receptive_width': 5,
                'in_receptive_step':  10,
                'in_receptive_start': 3,
                'min_scale_weight': 0.00001,
                'pix_in_weight': g_w2s*0.2,
                'readout_w': 0.5,
                'num_input_wta': 15,
                'num_liquid': 500,
                'num_output': 25,
                'in_to_liquid_exc_probability': 0.8,
                'in_to_liquid_inh_probability': 0.5,
                'col_weight_func': lambda dist: (g_w2s*1.)*np.exp(-dist/1.75),
                'build_complex': False,
                'build_readout': False,
                'complex_recp_width': 9,
               }



