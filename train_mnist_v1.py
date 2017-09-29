from common import *
from vision.mnist_config import defaults_retina
import cv2
import time

def set_sim(sim, spikes, img_w, img_h, w2s=4.376069, competition_on=True,
            direction_on=True, learning_on=False, new_lists=None):
    
    sim.setup(timestep=1., max_delay=14., min_delay=1.)

    sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 100)
    sim.set_number_of_neurons_per_core(sim.SpikeSourceArray, 100)

    cam, dmy_ssa_cam, dmy_prj_cam = setup_cam_pop(sim, spikes, 
                                                  img_w, img_h, w2s=w2s)

    cam.record()
    rbits = int(np.ceil(np.log2(img_h)))
    ret_cfg = defaults_retina
    ret_cfg['record'] = { 'voltages': False,
                           'spikes': True,
                         }
    ret_cfg['gabor'] = False
    ret_cfg['input_mapping_func'] = mapf.row_col_to_input
    ret_cfg['row_bits'] = rbits
    ret_cfg['lateral_competition'] = competition_on
    ret_cfg['split_cam_off_arg'] = False
    ret_cfg['plot_kernels'] = True


    if not direction_on:
        ret_cfg['direction'] = False

    mode = dvs_modes[MERGED]
    retina = Retina(sim, cam, img_w, img_h, mode, cfg=ret_cfg)
    lgn = LGN(sim, retina)
    v1 = V1(sim, lgn, learning_on=learning_on, input_conns=new_lists)

    return retina, lgn, v1, cam


def add_time_to_spikes(spikes, loop, img_idx, total_imgs, run_time):
    dt = (loop*total_imgs + img_idx)*run_time
    for nid in range(len(spikes)):
        for tidx in range(len(spikes[nid])):
            spikes[nid][tidx] += dt

    return spikes


def run_sim(sim, run_time, img_w, img_h, retina, lgn, v1):

    rbits = int(np.ceil(np.log2(img_h)))

    v1_start_w = v1.get_input_weights(get_initial=True)

    start_time = time.time()
    print("\n Experiment will begin now...@ %s\n" %
          time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()))

    sim.run(run_time)
    end_time = time.time()
    time_to_run = end_time - start_time
    print("\n Experiment finished running @ %s! Took %s minutes to run\n" %
          (time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()), time_to_run/60.))

    map_w = {}
    out_spikes = {}
    bip_w = {}
    for ch in retina.pops.keys():
        out_spikes[ch] = {}
        out_spikes[ch]['cam'] = get_spikes(retina.cam[ch], 'cam__%s'%ch)
        map_w[ch] = retina._cam_map_projs[ch].getWeights(format='array')
        bip_w[ch] = {}
        for p in retina.pops[ch].keys():
            out_spikes[ch][p] = {}
            if 'bip2gang' in retina.projs[ch][p]:
                bip_w[ch][p] = [np.array(retina.projs[ch][p]['bip2gang'][0].
                                                getWeights(format='array')),
                               np.array(retina.projs[ch][p]['bip2gang'][1].
                                                getWeights(format='array'))]

            if isinstance(retina.pops[ch][p], dict):
                for t in retina.pops[ch][p].keys():
                    key = "%s__%s__%s"%(ch, p, t)
                    print("\tGettings spikes for %s"%key)
                    out_spikes[ch][p][t] = get_spikes(retina.pops[ch][p][t],
                                                     key)

    for k in map_w.keys():
        f = open("mapping_weights_%s_w.txt"%k, 'w+')
        weight_indices = np.where(~np.isnan(map_w[k]))
        for i in range(len(weight_indices[0])):
            f.write("%d (%d, %d) to %d (%d, %d) = %f\n"%
                    (weight_indices[0][i],
                     (weight_indices[0][i] >> 1) & ((1 << rbits) - 1),
                     (weight_indices[0][i] >> (rbits + 1) ),
                     weight_indices[1][i],
                     weight_indices[1][i]//img_w, weight_indices[1][i]%img_w,
                    map_w[k][weight_indices[0][i], weight_indices[1][i]]))
        f.close()

    for k in bip_w:
        for p in bip_w[k]:
            for i in range(len(bip_w[k][p])):
                f = open("weights_bip_to_gang_%s_%s_%d_w.txt"%(k, p, i), "w+")
                r = 0
                for row in bip_w[k][p][i]:
                    c = 0
                    for val in row:
                        if np.isnan(val):
                            c += 1
                            continue
                        f.write("%d to %d = %3.4f\n"%(r, c, val))
                        c += 1
                    r += 1
                f.close()

    lgn_spikes = {}
    for ch in lgn.pops.keys():
        lgn_spikes[ch] = {}
        for p in lgn.pops[ch].keys():
            lgn_spikes[ch][p] = {}
            for lyr in  lgn.pops[ch][p]:
                lgn_spikes[ch][p][lyr] = lgn.pops[ch][p][lyr].\
                                           getSpikes(compatible_output=True)

    v1_spikes = v1.get_out_spikes()

    v1_end_w = v1.get_input_weights()

    v1_dict = {'shape': (v1.cfg['in_receptive_width'], v1.cfg['in_receptive_width']),
               'spikes': v1_spikes, 'init_w': v1_start_w, 'end_w': v1_end_w}
    return out_spikes, retina.shapes, lgn_spikes, v1_dict

def delete_prev_run():
    import os
    files = os.listdir(os.getcwd())
    for file in files:
        if file.endswith(".png") or file.endswith(".m4v") or \
            file.endswith(".bz2") or file.endswith(".npy") or \
            file.endswith(".pdf") or file.endswith("w.txt"):
            os.remove(os.path.join(os.getcwd(), file))

def plot_v1_in_weights(v1_in_w, in_shape, ncols=10, figw=2., out_dir='./v1_in_w'):
    template = 'row_%d_col_%d__ch_%s_pop_%s_key_%s'
    title_tmpl = '%d, %d - %s - %s - %s'
    for r in sorted(v1_in_w):
        for c in sorted(v1_in_w[r]):
            unit_w = v1_in_w[r][c]
            for ch in unit_w:
                for p in unit_w[ch]:
                    for k in unit_w[ch][p]:
                        notification = title_tmpl % (r, c, ch, p, k)
                        sys.stdout.write("\r%s"%notification)
                        sys.stdout.flush()

                        w = unit_w[ch][p][k]
                        nrows = w.shape[1]//ncols + (1 if w.shape[1]%ncols > 0 else 0)
                        title = '(%d, %d)   ch %s   pop %s   key %s' % \
                                (r, c, ch, p, k)
                        fig = plt.figure(figsize=(ncols*figw, (nrows+1)*figw))
                        fig.suptitle(title)

                        hh = int(np.sqrt(w.shape[0]))
                        if hh**2 == w.shape[0]:
                            ww = hh
                        else:
                            ww = hh + 1
                        weights = np.zeros((hh, ww))
                        for i in range(w.shape[1]):
                            ax = plt.subplot(nrows, ncols, i+1)
                            ax.set_title("post %d"%i)
                            ax.get_xaxis().set_visible(False)
                            ax.get_yaxis().set_visible(False)
                            weights[:] = w[:, i].reshape((hh, ww))
                            weights[np.isnan(weights)] = 0
                            plt.imshow(weights, cmap='Greys_r',
                                       interpolation='none')

                        plt.draw()
                        fname = template % (r, c, ch, p, k)
                        plt.savefig(os.path.join(out_dir, "%s.pdf"%fname))
                        plt.close()


w2s = 4.376069
img_w, img_h = 32, 32
# img_w, img_h = 50, 50
# img_w, img_h = 64, 64
# img_w, img_h = 90, 90
# img_w, img_h = 96, 96
# img_w, img_h = 100, 100

# img_w, img_h = 128, 128
col_bits = int(np.ceil(np.log2(img_w)))
row_bits = int(np.ceil(np.log2(img_h)))
ch_bits = 1
num_neurons = img_w*img_h*2
n_cam_neurons = (1 << ( col_bits + row_bits + ch_bits ))
fps = 100
vid_fps = 60 # this will slow video down
vid_scale = 20
thresh_scale = 15
cam_thresh_scale = 10
frames = 200
frames = 300
frames = 500
frames = 45

thresh = int(255*0.05) # just for plotting
# thresh = int(255*0.1)

# mnist_dir = "../../pyDVS/mnist_spikes/" \
#             "mnist_behave_SACCADE_" \
#             "pol_MERGED_enc_TIME_" \
#             "thresh_12_hist_95_00_" \
#             "inh_False___45_frames_" \
#             "at_90fps_%dx%d_res_spikes/" \
#             "t10k/"

mnist_dir = "../../pyDVS/mnist_spikes/" \
            "mnist_behave_SACCADE_" \
            "pol_MERGED_enc_TIME_" \
            "thresh_12_hist_99_00_" \
            "inh_False___90_frames_" \
            "at_90fps_%dx%d_res_spikes/" \
            "train/"

mnist_dir = mnist_dir%(img_w, img_h)

print("reading spikes from:")
print(mnist_dir)
###############################################################################
#
#  S W I T C H E S 
#
###############################################################################
#
plot_cam_spikes = True if 1 else False
simulate_retina = True if 1 else False
compete_on      = True if 1 else False
learn_on        = True if 1 else False
direction_on    = True if 0 else False
plot_out_spikes = True if 0 else False
vid_out_spikes  = True if 1 else False
del_prev        = True if 1 else False
#
###############################################################################


on_time_ms  = int( frames*(1000./fps) ) # total simulation time
ftime_ms    = int( 1000./fps )*1 # how many milliseconds (from frames) to plot
run_time = int(on_time_ms*1.1)
off_time_ms = 0
start_time  = 0
delete_before = 500
delete_before = 300
delete_before = 0
# spikes_dir = os.path.join(mnist_dir, '')
spikes_dir = mnist_dir
new_lists = None
out_weight_dir = './v1_input_weights'
out_net_dir = './train_v1_dir'
num_loops  = 1
total_imgs = 100
spikes = []
imgsU = []
dump_every = 1

if del_prev:
    print("Delete previous run --------------------------------------------")
    delete_prev_run()

for loop in range(num_loops):

    print("\n\nLoop %d/%d --- Num images %d\n"%(loop+1, num_loops, total_imgs))
    first_img = True
    for img_idx in range(total_imgs):
        print("\n\nLoop %d/%d --- Image number %d/%d ---------------\n"%
              (loop+1, num_loops, img_idx+1, total_imgs))

        spikes[:] = pat_gen.img_spikes_from_to(spikes_dir, n_cam_neurons,
                            img_idx, img_idx + 1, on_time_ms, off_time_ms,
                            start_time, delete_before=delete_before)

        if loop > 0 or img_idx > 0:
            spikes[:] = add_time_to_spikes(spikes, loop,
                                img_idx, total_imgs, run_time)

        if plot_cam_spikes:
            cols = 10
            figw = 2.

            # print(spikes)
            vid_start_t = (loop*total_imgs + img_idx)*run_time
            vid_end_t = vid_start_t + on_time_ms
            imgsU[:] = imgs_in_T_from_spike_array(
                            spikes, img_w, img_h,
                            vid_start_t, vid_end_t, ftime_ms,
                            out_array=False, thresh=thresh*cam_thresh_scale,
                            map_func=cam_img_map)

            num_imgs = len(imgsU)

            rows = num_imgs//cols + (1 if num_imgs%cols else 0)
            images_to_video(imgsU, fps=vid_fps, scale=10,
                    title='camera_spikes_loop_%d_img_%d'%(loop, img_idx))

        if first_img:
            retina, lgn, v1, cam = set_sim(sim, spikes, img_w, img_h, w2s, compete_on,
                                      direction_on, learn_on, new_lists)
            first_img = False
        else:
            cam.set("spike_times", spikes)

        ret_spikes, retina.shapes, lgn_spikes, v1_dict = \
                            run_sim(sim, run_time, img_w, img_h, retina, lgn, v1)

        try:
            new_lists.clear()
            del new_lists
        except:
            pass

        # new_lists = v1.updated_in_conn_lists(v1_dict['end_w'])

        if (img_idx)%dump_every == 0:

            if not (os.path.isdir(out_weight_dir)):
                os.makedirs(out_weight_dir)

            loop_weight_dir = os.path.join(out_weight_dir,
                                           "loop_%06d_img_%06d"%(loop, img_idx))

            if not (os.path.isdir(loop_weight_dir)):
                os.makedirs(loop_weight_dir)
            else:
                for the_file in os.listdir(loop_weight_dir):
                    file_path = os.path.join(loop_weight_dir, the_file)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)

                    except Exception as e:
                        print(e)

            plot_v1_in_weights(v1_dict['end_w'], (16, 16), out_dir=loop_weight_dir)
            print("\n\n ------------------------------------------------------- \n")
            print("DUMPING NETWORK at loop %d, img %d"%(loop + 1, img_idx + 1) )
            print("\n ------------------------------------------------------- \n")
            if not os.path.isdir(os.path.join(os.getcwd(), out_net_dir)):
                os.makedirs(os.path.join(os.getcwd(), out_net_dir))

            dump_compressed({#'retina': retina, 'lgn': lgn, 'v1': v1,
                             'ret_spikes': ret_spikes, 'lgn_spikes': lgn_spikes,
                             'v1_dict': v1_dict},
                            os.path.join(out_net_dir,
                                 'network_at_loop_%d__img_%d.pickle'%(loop, img_idx)))

        # for o in [retina, lgn, v1, ret_spikes, lgn_spikes, v1_dict]:
        #     del o

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
sim.end()
