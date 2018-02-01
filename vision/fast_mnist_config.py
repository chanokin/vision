from sim_tools.connectors.mapping_funcs import  row_col_to_input, \
                                                row_col_to_input_breakout
import numpy as np

exc_cell = "IF_curr_exp"
exc_cell_params = { 'cm': 0.25,  # nF
                  'i_offset': 0.0,
                  'tau_m': 10.0,
                  'tau_refrac': 3.0,
                  'tau_syn_E': 1., #2
                  'tau_syn_I': 1., #4
                  'v_reset': -80.0,
                  'v_rest': -65.0,
                  'v_thresh': -55.4
                  }
exc_big_cell_params = { 'cm': 0.25,  # nF
                        'i_offset': 0.0,
                        'tau_m': 10.0,
                        'tau_refrac': 3.0,
                        'tau_syn_E': 2., #2
                        'tau_syn_I': 1., #4
                        'v_reset': -80.0,
                        'v_rest': -65.0,
                        'v_thresh': -55.4
                  }
dir_cell_params = { 'cm': 0.25,  # nF
                  'i_offset': 0.0,
                  'tau_m': 10.0,
                  'tau_refrac': 3.0,
                  'tau_syn_E': 5., #2
                  'tau_syn_I': 2.5, #4
                  'v_reset': -70.0,
                  'v_rest': -65.0,
                  'v_thresh': -55.4
                  }

inh_cell = "IF_curr_exp"
inh_cell_params = { 'cm': 0.25,  # nF
                  'i_offset': 0.0,
                  'tau_m': 10.0,
                  'tau_refrac': 2.0,
                  'tau_syn_E': 2., #2
                  'tau_syn_I': 1., #2
                  'v_reset': -70.0,
                  'v_rest': -65.0,
                  'v_thresh': -57.
                  }


wta_inh_cell = "IF_curr_exp"
wta_inh_cell_params = {'cm': 0.3,  # nF
                       'i_offset': 0.0,
                       'tau_m': 4.0,
                       'tau_refrac': 2.0,
                       'tau_syn_E': 2.,
                       'tau_syn_I': 1.,
                       'v_reset': -70.0,
                       'v_rest': -65.0,
                       'v_thresh': -58.
                      }


# g_w2s = 4.376069
# inh_w2s = 4.376069
# dir_w2s = 2.1
# g_w2s = 1.78681
# inh_w2s = 1.78681
# dir_w2s = 1.
# ssamp_w2s = 4.376069
g_w2s = 4.8#78681
inh_w2s = 5.#78681

frame_rate = 90
pix_dist = 0.25
dir_delay = int((1000./frame_rate)*pix_dist)
dir_max_dist = 5
dir_width = dir_max_dist*2 + 1
dir_w2s = (g_w2s/(dir_max_dist-0.5))*(1.)
defaults_retina = {
                'channel_bits': 1,
                'event_bits': 0,
                # 'kernel_width': 3,
                'kernel_exc_delay': 4.,
                'kernel_inh_delay': 1.,
                'corr_self_delay': 4.,
                'corr_w2s_mult': 1.2,
                'min_weight': 0.001,
                'row_step': 1, 'col_step': 1,
                'start_row': 0, 'start_col': 0,
                # 'gabor': {'num_divs': 2., 'freq': 5., 'std_dev': 5., 'width': 7,
                            # 'step': 3, 'start': 0},
                'cs': {'std_dev': 0.57, 'sd_mult': 6.7, 'width': 3,
                       'step': 2, 'start':1, 'w2s_mult': 1.,
                       'params': exc_cell_params},
                'cs2': {'std_dev': 0.865492, 'sd_mult': 6.63, 'width': 7,
                        'step': 4, 'start': 3, 'w2s_mult': 1.,
                        'params': exc_cell_params},
                'cs3': {'std_dev': 1.653551, 'sd_mult': 6.18, 'width': 15,
                        'step': 6, 'start': 7, 'w2s_mult': 2.,
                        'params': exc_big_cell_params},
                # 'cs4': {'std_dev': 3.809901, 'sd_mult': 5.57, 'width': 31,
                #         'step': 10, 'start': 15, 'w2s_mult': 1.},
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
                # 'orientation':{'width': 7,
                                # 'std_dev': 6.,
                                # 'std_dev_div': 10.,
                                # 'angles': [0, 45, 90, 135],
                                # 'w2s_mult': 1.,
                                # 'inh_mult': 1.,
                                # 'sample_from': 'cam',
                                # 'start': 3, 'step': 3,
                               # },
                # 'direction': {'keys': [
                #                         'E',
                #                         'W',
                #                         'N',
                #                         'S',
                #                         #'NW', 'SW', 'NE', 'SE',
                #                         #'east', 'south', 'west', 'north',
                #                         #'south east', 'south west',
                #                         #'north east', 'north west'
                #                       ],
                #             'div': 4,#6,
                #             'weight': dir_w2s,
                #             'delays': [1, 4, 6, 8],#, 3, 4 ],
                #             'subsamp': 1,#2,
                #             'angle': 30.,
                #             'dist': dir_max_dist,
                #             'width': dir_width,
                #             'inh_w_scale': 1.,
                #             'delay_func': lambda dist: dir_delay*dist,
                #             'weight_func': lambda d,a,w: w/(1.+(0.01*d)+a),
                #             'step': 2,
                #             'start': 0,
                #             'sample_from': 'cs',
                #             'params': dir_cell_params,
                #             },

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

defaults_lgn = {
                 'kernel_exc_delay': 3.,
                 'kernel_inh_delay': 1.,
                 #default for training mnist!!!
#                 'cs': {'std_dev': 0.43, 'width': 3, 'wmult': 1.} ,
                 'cs': {'std_dev': 0.37, 'width': 3, 'wmult': 1.} ,
                 'w2s': g_w2s*1.0,
                 'inh_w2s': inh_w2s,
                 'inh_cell': {'cell': inh_cell,
                              'params': inh_cell_params
                             }, 
                 'exc_cell': {'cell': exc_cell,
                              'params': exc_cell_params
                             },
                 'record': {'voltages': False, 
                              'spikes': True,
                           },
                 'plot_kernels': True,

               }


#######################################################################
####################           V 1          ###########################
#######################################################################

# unit_type = 'autoencoder'
# unit_type = 'liquid_state'
unit_type = 'four-tp-one'
unit_type = 'simple'


#from A Statistical Analysis of Information-Processing Properties of 
#Lamina-SpecificCortical Microcircuit Models
# pop_ratio = {'l2': {'inh': 0.2, 'exc': 0.8},
#              'l4': {'inh': 0.2, 'exc': 0.8},
#              'l5': {'inh': 0.2, 'exc': 0.8},
#             }

pop_ratio = {'l0': {'inh': 0.2, 'exc': 0.8}}

# column_conn_wgt = {'l2': {'exc2inh': 1.90, 'inh2exc': -0.65,
#                           'exc2exc': 1.70, 'inh2inh': -1.35,
#                           'exc2l5e': 1.40, 'inh2l5e': -5.20,
#                           'exc2l4i': 1.60,
#                          },
#                    'l4': {'exc2inh': 3.70, 'inh2exc': -0.85,
#                           'exc2exc': 1.10, 'inh2inh': -1.55,
#                           'exc2l2e': 4.00, 'inh2l2e': -1.75,
#                           'inh2l2i': -1.5,
#                          },
#                    'l5': {'exc2inh': 0.90, 'inh2exc': -1.20,
#                           'exc2exc': 1.70, 'inh2inh': -1.20,
#                           'exc2l2e': 0.30,
#                          },
#                   }

column_conn_wgt = {'l0': {'exc2inh': g_w2s / 50., 'inh2exc': g_w2s*1.1,
                          'exc2exc': g_w2s / 50., 'inh2inh': g_w2s*1.1}}

w_conv = 1.#(g_w2s)/5.2 #5.2 is abs max weight in dict
for l in column_conn_wgt:
    for c in column_conn_wgt[l]:
        column_conn_wgt[l][c] *= w_conv

# column_conn_prob = {'l2': {'exc2inh': 0.21, 'inh2exc': 0.16,
#                            'exc2exc': 0.26, 'inh2inh': 0.25,
#                            'exc2l5e': 0.55, 'inh2l5e': 0.20,
#                            'exc2l4i': 0.08,
#                           },
#                     'l4': {'exc2inh': 0.19, 'inh2exc': 0.10,
#                            'exc2exc': 0.17, 'inh2inh': 0.50,
#                            'exc2l2e': 0.28, 'inh2l2e': 0.50,
#                            'inh2l2i': 0.20,
#                           },
#                     'l5': {'exc2inh': 0.10, 'inh2exc': 0.12,
#                            'exc2exc': 0.09, 'inh2inh': 0.60,
#                            'exc2l2e': 0.03,
#                           },
#                    }

column_conn_prob = {'l0': {'exc2inh': 0.10, 'inh2exc': 0.10,
                           'exc2inh': 0.10, 'inh2inh': 0.10}}

# input_conn_prob = {'main':  {'l2e': 0.20,
#                              'l4e': 0.80, 'l4i': 0.50,
#                              'l5e': 0.10,
#                             },
#                    'extra': {'l2e': 0.20,}
#                   }

input_conn_prob = {'main':  {'l0e': 1.00},
                  }

# neurons_in_column = {'l2': 40,
#                      'l4': 100,
#                      'l5': 40}
v1_exc_cell = "IF_curr_exp"
v1_exc_cell_params = { 'cm': 0.25,  # nF
                  'i_offset': 0.001,
                  'tau_m': 10.0,
                  'tau_refrac': 3.0,
                  'tau_syn_E': 1., #2
                  'tau_syn_I': 2., #4
                  'v_reset': -70.0,
                  'v_rest': -65.0,
                  'v_thresh': -55.4
                  }

neurons_in_column = {'l0': 50}
max_w_mult = 0.5
defaults_v1 = { 'unit_type': unit_type,
                'w2s': g_w2s,
                'pop_ratio': pop_ratio,
                'column_conn_prob': column_conn_prob,
                'column_conn_wgt': column_conn_wgt,
                'input_conn_prob': input_conn_prob,
                'neurons_in_column': neurons_in_column,
                'inter_unit_connect': False,
                'inter_conn_prob': 0.2,#input_conn_prob['extra']['l2e'],
                'inter_conn_weight': g_w2s*0.5,
                'inter_conn_width': 3,
                'inter_conn_delay': 1.,
                'input_delay': 1,
                'context_in_weight': 0.3,
                'context_to_context_weight': 0.5, 
                'context_to_simple_weight': 1., 
                'min_delay': 2.,
                'max_delay': 14.,
                'max_weight': g_w2s*max_w_mult,
                'wta_inh_cell': { 'cell': wta_inh_cell,
                                  'params': wta_inh_cell_params,
                                }, 
                'inh_cell': {'cell': inh_cell,
                             'params': inh_cell_params,
                            }, 
                'exc_cell': {'cell': v1_exc_cell,
                             'params': v1_exc_cell_params,
                            },
                'record': {'voltages': False, 
                           'spikes': True,
                          },
                'lat_inh': False,
                'stdp': {'tau_plus': 20,
                         'tau_minus': 24,
                         'w_max': g_w2s*max_w_mult,
                         'w_min': 0.,
                         'a_plus': 0.01,
                         'a_minus': 0.012,
                        },
                'in_receptive_width': 7, # width
                'in_receptive_step':  3, # step
                'in_receptive_start': 3, # start
                'min_scale_weight': 0.00001,
                'min_weight': 0.00001,
                'pix_in_weight': g_w2s*0.2,
                'readout_w': 0.5,
                'num_input_wta': 15,
                'num_liquid': 500,
                'num_output': 25,
                'in_to_liquid_exc_probability': 0.8,
                'in_to_liquid_inh_probability': 0.5,
                # 'col_weight_func': lambda dist: (g_w2s*0.01)*np.exp(-dist/2.),
                'col_weight_func': lambda dist: 0.01 *  np.exp(-(dist)/1.),
                'build_complex': False,
                'build_readout': False,
                'complex_recp_width': 9,
                'weight_dir': 'v1_column_weights',
                'noise_per_unit': True,
                'noise_weight': g_w2s*0.01,
                'noise_rate': 20,#Hz
               }



