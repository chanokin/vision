from vision_common import *
import os
from vision.mnist_config import defaults_retina, defaults_lgn
import cv2
import time

def run_sim(sim, run_time, spikes, img_w, img_h, ch_bits=1, row_bits=8, w2s=4.376069, 
            competition_on=True, direction_on=True):
    
    sim.setup(timestep=1., max_delay=140., min_delay=1.)

    sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 100)
    sim.set_number_of_neurons_per_core(sim.SpikeSourceArray, 100)

    cam, dmy_ssa_cam, dmy_prj_cam = setup_cam_pop(sim, spikes, 
                                      img_w, img_h, ch_bits, w2s=w2s)

    cam.record()
    rbits = int(np.ceil(np.log2(img_h)))
    ret_cfg = defaults_retina
    ret_cfg['record'] = { 'voltages': False,
                           'spikes': True,
                        }
    ret_cfg['gabor'] =  False
    ret_cfg['row_bits'] =  rbits
    ret_cfg['lateral_competition'] =  competition_on
    ret_cfg['plot_kernels'] =  False


    if not direction_on:
        ret_cfg['direction'] = False

    mode = dvs_modes[MERGED]
    retina = Retina(sim, cam, img_w, img_h, mode, cfg=ret_cfg)
    #retina.cam['on'].record()
    
    lgn = LGN(sim, retina, cfg=defaults_lgn)

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
        bip_w[ch] = {}
        for p in retina.pops[ch].keys():
            out_spikes[ch][p] = {}

            if isinstance(retina.pops[ch][p], dict):
                for t in retina.pops[ch][p].keys():
                    key = "%s__%s__%s"%(ch, p, t)
                    print("\tGettings spikes for %s"%key)
                    out_spikes[ch][p][t] = get_spikes(retina.pops[ch][p][t],
                                                     key)

    lgn_spikes = {}
    for ch in lgn.pops.keys():
        lgn_spikes[ch] = {}
        for p in lgn.pops[ch].keys():
            lgn_spikes[ch][p] = {}
            for lyr in  lgn.pops[ch][p]:
                lgn_spikes[ch][p][lyr] = lgn.pops[ch][p][lyr].\
                                            getSpikes(compatible_output=True)



    sim.end()

    return out_spikes, retina.shapes, lgn_spikes

def delete_prev_run():
    import os
    files = os.listdir(os.getcwd())
    for file in files:
        if file.endswith(".png") or file.endswith(".m4v") or \
            file.endswith(".bz2") or file.endswith(".npy") or \
            file.endswith(".pdf") or file.endswith("w.txt"):
            os.remove(os.path.join(os.getcwd(), file))

img_w, img_h = 32, 32

col_bits = int(np.ceil(np.log2(img_w)))
row_bits = int(np.ceil(np.log2(img_h)))
ch_bits = 1
num_neurons = img_w*img_h*2
n_cam_neurons = (1 << ( col_bits + row_bits + ch_bits ))
fps = 90
vid_fps = 60 # this will slow video down
vid_scale = 20
thresh_scale = 15
cam_thresh_scale = 10
frames = 90

thresh = int(255*0.05) # just for plotting


mnist_dir = \
    "../../pyDVS/mnist_spikes" \
    "/mnist_behave_SACCADE_pol_MERGED" \
    "_enc_TIME_thresh_12_hist_99_00_" \
    "inh_False___90_frames_at_90fps_%dx%d_res_spikes/"\
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
competition_on  = True if 0 else False
direction_on    = True if 1 else False
plot_out_spikes = True if 0 else False
vid_out_spikes  = True if 1 else False
del_prev        = True if 1 else False
#
###############################################################################


on_time_ms  = int( frames*(1000./fps) ) # total simulation time
ftime_ms    = int( 1000./fps )*1 # how many milliseconds (from frames) to plot
off_time_ms = 0
start_time  = 0
delete_before = 500
delete_before = 300
delete_before = 0
# spikes_dir = os.path.join(mnist_dir, '')

if del_prev:
    print("Delete previous run --------------------------------------------")
    delete_prev_run()

spikes_dir = mnist_dir
print("Getting spikes -------------------------------------------------")
spikes = pat_gen.img_spikes_from_to(spikes_dir, n_cam_neurons, 0, 1, 
                                    on_time_ms, off_time_ms,
                                    start_time, delete_before=delete_before)

print("Plotting Camera ------------------------------------------------")
cols = 10
figw = 2.
if plot_cam_spikes:
    imgsU = imgs_in_T_from_spike_array(spikes, img_w, img_h, 
                                       0, on_time_ms, ftime_ms,
                                       out_array=False,
                                       thresh=thresh*cam_thresh_scale,
                                       map_func=cam_img_map)

    num_imgs = len(imgsU)

    rows = num_imgs//cols + (1 if num_imgs%cols else 0)
    images_to_video(imgsU, fps=vid_fps, title='camera_spikes', scale=10)

# ------------------------------------------------------------------ #
# ------------------------------------------------------------------ #
# ------------------------------------------------------------------ #

if simulate_retina:
    print("Simulating network ---------------------------------------------")
    ret_spikes, pop_shapes, lgn_spikes = run_sim(sim, on_time_ms, spikes,
                                             img_w, img_h, ch_bits=ch_bits,
                                             w2s=4.376069*1.01,
                                             competition_on=competition_on,
                                             direction_on=direction_on)

    dump_compressed({'spikes': ret_spikes, 'shapes': pop_shapes},
                    'test_retina_spikes_and_shapes.pickle')

    print("\nPlotting Retina's output spikes:\n")
    on_imgs    = []
    on_spikes  = []
    off_imgs   = []
    off_spikes = []
    in_spikes  = []
    out_spikes = []
    for pop in ret_spikes[ret_spikes.keys()[0]]:
#        if 'cam' in pop:
#            continue

        if 'dir' in pop:
            w = pop_shapes['direction']['width']
            h = pop_shapes['direction']['height']
        elif 'orient' in pop:
            w = pop_shapes['orientation']['width']
            h = pop_shapes['orientation']['height']
        else:
            w = pop_shapes[pop]['width']
            h = pop_shapes[pop]['height']

        print("\tfor %s"%(pop))

        on_imgs[:]   = []
        on_spikes[:] = []

        off_imgs[:]   = []
        off_spikes[:] = []

        ### output and camera spikes
        if 'cam' in pop: 
            on_spikes[:]  = ret_spikes['on'][pop]
            off_spikes[:] = ret_spikes['off'][pop]
        else:
            on_spikes[:]  = ret_spikes['on'][pop]['ganglion']
            off_spikes[:] = ret_spikes['off'][pop]['ganglion']

        on_imgs[:] = imgs_in_T_from_spike_array(on_spikes, w, h,
                                                0, on_time_ms, ftime_ms,
                                                out_array=True, 
                                                thresh=thresh*thresh_scale,
                                                up_down = True)
        off_imgs[:] = imgs_in_T_from_spike_array(off_spikes, w, h, 
                                                 0, on_time_ms, ftime_ms,
                                                 out_array=True, 
                                                 thresh=thresh*thresh_scale,
                                                 up_down = False)
        if plot_out_spikes:
            fname = "retina_out_spikes_filter_%s_competition_%s.pdf" % \
                                (pop, 'on' if competition_on else 'off')
            plot_image_set(on_imgs, fname, ftime_ms, off_imgs)

        # plt.show()
        # if vid_out_spikes:
        images_to_video(on_imgs, vid_fps,
                        "retina_out_video_spikes_filter_%s_competition_%s"%
                                        (pop, 'on' if competition_on else 'off'),
                        scale=vid_scale, off_images=off_imgs)


    for ch in ret_spikes:
        for p in ret_spikes[ch]:
            if 'cam' in p:
                continue

            in_spikes[:]  = ret_spikes[ch][p]['bipolar']
            out_spikes[:] = ret_spikes[ch][p]['ganglion']

            in_color = 'cyan' if ch == 'on' else 'magenta'
            fname = 'retina_raster_plot_filter_%s_channel_%s.pdf' % (p, ch)
            plot_in_out_spikes(in_spikes, out_spikes, fname, in_color, p, ch)

    # ------------------------------------------------------------------- #

    print("\nPlotting LGN's output spikes:\n")
# #
# #
    ch = lgn_spikes.keys()[0]
    for p in lgn_spikes[ch]:
        for lyr in lgn_spikes[ch][p]:
            lyr = 'relay'
            if 'cam' in p:
               continue
# #
            if 'dir' in p:
                w = pop_shapes['direction']['width']
                h = pop_shapes['direction']['height']
            elif 'orient' in p:
                w = pop_shapes['orientation']['width']
                h = pop_shapes['orientation']['height']
            else:
                w = pop_shapes[p]['width']
                h = pop_shapes[p]['height']
    # #
            on_spikes[:]  = lgn_spikes['on'][p][lyr]
            off_spikes[:] = lgn_spikes['off'][p][lyr]
    # #
            on_imgs[:] = imgs_in_T_from_spike_array(on_spikes, w, h,
                                                    0, on_time_ms, ftime_ms,
                                                    out_array=True,
                                                    thresh=thresh * thresh_scale,
                                                    up_down=True, )
    # # 
            off_imgs[:] = imgs_in_T_from_spike_array(off_spikes, w, h,
                                                     0, on_time_ms, ftime_ms,
                                                     out_array=True,
                                                     thresh=thresh * thresh_scale,
                                                     up_down=False, )
# #
        if plot_out_spikes:
            fname = "lgn_out_spikes_filter_%s_%s.pdf" % (p, lyr)
            plot_image_set(on_imgs, fname, ftime_ms, off_imgs)
# #
        # plt.show()
        if vid_out_spikes:
            images_to_video(on_imgs, vid_fps,
                    "lgn_out_video_spikes_filter_%s_competition_%s" % (p, lyr),
                    scale=vid_scale, off_images=off_imgs)
# #
# #
    for ch in lgn_spikes:
        for p in lgn_spikes[ch]:
            if 'cam' in p:
                continue
# #
            in_spikes[:]  = ret_spikes[ch][p]['ganglion']
            out_spikes[:] = lgn_spikes[ch][p]['relay']
# #
            in_color = 'cyan' if ch == 'on' else 'magenta'
            fname = 'lgn_raster_plot_filter_%s_channel_%s.pdf' % (p, ch)
            plot_in_out_spikes(in_spikes, out_spikes, fname, in_color, p, ch)


