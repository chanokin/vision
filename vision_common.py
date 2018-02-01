import numpy as np
import pylab as plt
import re
# import vision.sim_tools.connectors.kernel_connectors as kconn
from vision.spike_tools.vis import my_imshow, plot_spikes, \
                                   plot_output_spikes, \
                                   imgs_in_T_from_spike_array, \
                                   images_to_video, \
                                   plot_in_out_spikes, \
                                   plot_image_set

# from vision.sim_tools.vis import plot_connector_3d
# import vision.sim_tools.kernels.center_surround as csgen
# import vision.sim_tools.kernels.gabor as gabgen
from vision.sim_tools.connectors import mapping_funcs as mapf
from vision.sim_tools.common import dump_compressed, load_compressed, is_spinnaker

from mpl_toolkits.mplot3d import axes3d, Axes3D

from vision.retina import Retina, dvs_modes, MERGED
from vision.lgn import LGN
from vision.v1 import V1

from vision.spike_tools.pattern import pattern_generator as pat_gen
from vision.mnist_config import exc_cell_params as params, \
                                g_w2s as W2S
import os
import sys

# from pyNN import nest as sim
# from pyNN import spiNNaker as sim
import spynnaker7.pyNN as sim

def row_major_map(nrn_id, img_width, img_height):
    r = nrn_id//img_width
    c = nrn_id%img_width

    return r, c, True

def cam_img_map(nrn_id, img_width, img_height):
    cols_bits = np.uint32(np.ceil(np.log2(img_width)))
    cols_mask = np.uint32((1<<cols_bits) - 1)
    rows_bits = np.uint32(np.ceil(np.log2(img_height)))
    rows_mask = np.uint32((1<<rows_bits) - 1)

    col = (nrn_id >> (rows_bits + 1)) & cols_mask
    row = (nrn_id >> 1) & rows_mask
    up_dn = nrn_id & 1
    
    return row, col, up_dn


def setup_cam_pop(sim, spike_array, img_w, img_h, ch_bits=1, event_bits=0,
                    w2s=W2S):
    row_bits = int(np.ceil(np.log2(img_h)))
    col_bits = int(np.ceil(np.log2(img_w)))
    pop_size = (1 << (row_bits + col_bits + ch_bits + event_bits))
    cell = sim.IF_curr_exp

    dmy_pops = []
    dmy_prjs = []
    if is_spinnaker(sim):
        cam_pop = sim.Population(pop_size, sim.SpikeSourceArray, 
                                 {'spike_times': spike_array}, 
                                 label='Source Camera')

    else:
        cam_pop = sim.Population(pop_size, cell, params, label='camera')

        for i in range(pop_size):
            dmy_pops.append(sim.Population(1, sim.SpikeSourceArray, 
                                        {'spike_times': spike_array[i]},
                                        label='pixel (row, col) = (%d, %d)'%\
                                        (i//img_w, i%img_w)))
            conn = [(0, i, w2s, 1)]
            dmy_prjs.append(sim.Projection(dmy_pops[i], cam_pop,
                                        sim.FromListConnector(conn),
                                        target='excitatory',
                                        label='dmy to cam %d'%i))

    return cam_pop, dmy_pops, dmy_prjs


def plot_out_spikes(on_spikes, off_spikes, img_w, img_h, 
                    end_t_ms, ftime_ms, thresh, title):
    on_imgs = imgs_in_T_from_spike_array(on_spikes, img_w, img_h, 
                                        0, end_t_ms, ftime_ms, 
                                        out_array=True, thresh=thresh)

    off_imgs = imgs_in_T_from_spike_array(off_spikes, img_w, img_h, 
                                        0, end_t_ms, ftime_ms, 
                                        out_array=True, thresh=thresh)

    num_imgs = len(on_imgs)
    cols = 10
    rows = num_imgs//cols + (1 if num_imgs%cols else 0)
    figw = 1.2
    fig = plt.figure(figsize=(figw*cols, figw*rows))
    for i in range(num_imgs):
        img = np.zeros((img_w, img_h, 3), dtype=np.uint8)
        img[:, :, 1] = off_imgs[i]*4
        img[:, :, 0] = on_imgs[i]*4
        ax = plt.subplot(rows, cols, i+1)
        my_imshow(ax, img, cmap=None)
    # plot_spikes(spikes)
    plt.suptitle(title)
    plt.savefig("%s.pdf"%(title), dpi=150)
    # plt.show()

def get_spikes(pop, key):
    try:
        return pop.getSpikes(compatible_output=True)
    except Exception, e:
        print("\t\t--- Unable to get spikes for %s\n\t\t%s"%(key, e))
        
        return None