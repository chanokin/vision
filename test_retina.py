from common import *

import cv2

def run_sim(sim, run_time, spikes, img_w, img_h, row_bits=8, w2s=4.376069, 
            competition_on=True, direction_on=True):
    
    if sim.__name__ == 'pyNN.spiNNaker':
        sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 150)
        # sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 100)
        # sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 50)
        # sim.set_number_of_neurons_per_core(sim.SpikeSourceArray, 1000)
        # sim.set_number_of_neurons_per_core(sim.SpikeSourceArray, 500)
        sim.set_number_of_neurons_per_core(sim.SpikeSourceArray, 1000)

    sim.setup(timestep=1., max_delay=140., min_delay=1.)

    cam, dmy_ssa_cam, dmy_prj_cam = setup_cam_pop(sim, spikes, 
                                                  img_w, img_h, w2s=w2s)

    cam.record()

    ret_cfg = {'record': { 'voltages': False, 
                           'spikes': True,
                         },
               'w2s': w2s,
               'gabor': False,
               'input_mapping_func': mapf.row_col_to_input,
               'row_bits': int(np.ceil(np.log2(img_h))),
               'lateral_competition': competition_on,
               'split_cam_off_arg': False,
               'plot_kernels': True,
              }

    if not direction_on:
        ret_cfg['direction'] = False

    mode = dvs_modes[MERGED]
    retina = Retina(sim, cam, img_w, img_h, mode, cfg=ret_cfg)

    print("\n Experiment will begin now...\n")

    sim.run(run_time)

    print("\n Experiment finished running!\n")

    out_spikes = {}
    for k in retina.pops.keys():
        out_spikes[k] = {}
        out_spikes[k]['cam'] = get_spikes(retina.cam[k], 'cam__%s'%k)

        for p in retina.pops[k].keys():
            out_spikes[k][p] = {}
            if isinstance(retina.pops[k][p], dict):
                for t in retina.pops[k][p].keys():
                    key = "%s__%s__%s"%(k, p, t)
                    print("\tGettings spikes for %s"%key)
                    out_spikes[k][p][t] = get_spikes(retina.pops[k][p][t],
                                                     key)

    # exw = retina.projs['on']['cs']['bip2gang'][0].getWeights(format='array')
    # for w in exw[:, 0]:
    #     if not np.isnan(w):
    #         print(w)
    # ihw = retina.projs['on']['cs']['bip2gang'][1].getWeights(format='array')
    # for w in ihw[:, 0]:
    #     if not np.isnan(w):
    #         print(w)

    sim.end()

    return out_spikes, retina.shapes


img_w, img_h = 32, 32
# img_w, img_h = 64, 64
# img_w, img_h = 90, 90
# img_w, img_h = 100, 100

# img_w, img_h = 128, 128
num_neurons = img_w*img_h*2
n_cam_neurons = 1 << ( int(np.ceil(np.log2(img_h)))  + 
                       int(np.ceil(np.log2(img_w))) + 1 )
fps = 100
vid_fps = 30 # this will slow video down
vid_scale = 20
frames = 200
# frames = 300
frames = 500

thresh = int(255*0.05) # just for plotting
# thresh = int(255*0.1)

# mnist_dir = "../../pyDVS/mnist_spikes/" + \
#             "mnist_behave_SACCADE_pol_MERGED" + \
#             "_enc_RATE_thresh_12_hist_99_00" + \
#             "_inh_False___" + \
#             "200_frames_at_100fps_32x32_res_spikes/" + \
#             "t10k/"
#             # "t10k/" "train/"

mnist_dir = "../../pyDVS/mnist_spikes/" + \
            "mnist_behave_SACCADE_pol_MERGED" + \
            "_enc_RATE_thresh_12_hist_99_00" + \
            "_inh_False___" + \
            "100_frames_at_100fps_%dx%d_res_spikes/" + \
            "t10k/"
            # "t10k/" "train/"

mnist_dir = "../../pyDVS/mnist_spikes/" + \
            "mnist_behave_TRAVERSE_pol_MERGED" + \
            "_enc_RATE_thresh_12_hist_99_00" + \
            "_inh_False___" + \
            "500_frames_at_100fps_%dx%d_res_spikes/" + \
            "t10k/"
            # "t10k/" "train/"

# mnist_dir = "../../pyDVS/mnist_spikes/" + \
#             "img_behave_SACCADE_pol_MERGED" + \
#             "_enc_TIME_thresh_12_hist_100_00" + \
#             "_inh_False___" + \
#             "10_frames_at_100fps_%dx%d_res_spikes/" + \
#             "t10k/"
            # "t10k/" "train/"
mnist_dir = mnist_dir%(img_w, img_h)

###############################################################################
#
#  S W I T C H E S 
#
###############################################################################
#
plot_cam_spikes = True if 1 else False
simulate_retina = True if 1 else False
competition_on  = True if 1 else False
direction_on    = True if 1 else False
plot_out_spikes = True if 0 else False
vid_out_spikes  = True if 1 else False
#
###############################################################################


on_time_ms  = int( frames*(1000./fps) ) # total simulation time
ftime_ms    = int( 1000./fps )*1 # how many milliseconds (from frames) to plot
off_time_ms = 0
start_time  = 0
delete_before = 300
# spikes_dir = os.path.join(mnist_dir, '')
spikes_dir = mnist_dir
print("Getting spikes -------------------------------------------------")
spikes = pat_gen.img_spikes_from_to(spikes_dir, n_cam_neurons, 0, 1, 
                                    on_time_ms, off_time_ms,
                                    start_time, delete_before=delete_before)

print("Plotting Camera ------------------------------------------------")
cols = 10
figw = 2.
if plot_cam_spikes:
    # print(spikes)
    imgsU = imgs_in_T_from_spike_array(spikes, img_w, img_h, 
                                        0, on_time_ms, ftime_ms, 
                                        out_array=False, thresh=thresh*100,
                                        map_func=cam_img_map)

    num_imgs = len(imgsU)

    rows = num_imgs//cols + (1 if num_imgs%cols else 0)
    fig = plt.figure(figsize=(figw*cols, figw*rows))
    plt.suptitle("each square is %d ms"%(ftime_ms))
    for i in range(num_imgs):
        ax = plt.subplot(rows, cols, i+1)
        my_imshow(ax, imgsU[i], cmap=None)
    # plot_spikes(spikes)

    plt.savefig("test_retina_camera.png", dpi=300)
    plt.close()
    # plt.show()

    images_to_video(imgsU, fps=vid_fps, title='camera_spikes', scale=10)




if simulate_retina:
    print("Simulating network ---------------------------------------------")
    ret_spikes, pop_shapes = run_sim(sim, on_time_ms, spikes, img_w, img_h, 
                                     w2s=4.376069*1.01, 
                                    #  w2s=3.95*1.01, 
                                     competition_on=competition_on,
                                     direction_on=direction_on)

    dump_compressed({'spikes': ret_spikes, 'shapes': pop_shapes},
                    'test_retina_spikes_and_shapes.pickle')
    # print(ret_spikes['on'].keys())
    # print(pop_shapes.keys())
    print("\nPlotting output spikes:\n")
    on_imgs    = []
    on_spikes  = []
    off_imgs   = []
    off_spikes = []
    # for channel in ret_spikes.keys():
    for pop in ret_spikes[ret_spikes.keys()[0]]:
        if 'cam' in pop:
            continue

        if 'cam' in pop:
            w = img_w
            h = img_h
        elif 'dir' in pop:
            w = pop_shapes['direction']['width']
            h = pop_shapes['direction']['height']
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

        spikes_fig_on = plt.figure(1, figsize=(10, 7))
        plt.plot([t for (_, t) in on_spikes], 
                    [i for (i, _) in on_spikes], 'xb', markersize=4,
                    label='Output - Ganglion - (%d spikes)'%(len(on_spikes)) )

        spikes_fig_off = plt.figure(2, figsize=(10, 7))
        plt.plot([t for (_, t) in off_spikes], 
                    [i for (i, _) in off_spikes], 'xb', markersize=4,
                    label='Output - Ganglion - (%d spikes)'%(len(off_spikes))
                )


        on_imgs[:] = imgs_in_T_from_spike_array(on_spikes, w, h, 
                                                0, on_time_ms, ftime_ms,
                                                out_array=True, 
                                                thresh=thresh*100,
                                                up_down = True)
        off_imgs[:] = imgs_in_T_from_spike_array(off_spikes, w, h, 
                                                 0, on_time_ms, ftime_ms,
                                                 out_array=True, 
                                                 thresh=thresh*100,
                                                 up_down = False)
        if plot_out_spikes:
            fig = plt.figure(3)
            num_imgs = len(on_imgs)
            rows = num_imgs//cols + (1 if num_imgs%cols else 0)
            figw = 2.5
            fig = plt.figure(figsize=(figw*cols, figw*rows))
            # plt.suptitle("each square is %d ms"%(ftime_ms))
            for i in range(num_imgs):

                ax = plt.subplot(rows, cols, i+1)
                if i == 0:
                    ax.set_title("%d ms frame"%ftime_ms)
                on_imgs[i][:, :, 0] = off_imgs[i][:, :, 0]
                my_imshow(ax, on_imgs[i], cmap=None)

            plt.savefig("out_spikes_filter_%s_competition_%s.png"%
                        (pop, 'on' if competition_on else 'off'),
                        dpi=300)
            plt.close(fig)
        # plt.show()
        if vid_out_spikes:
            images_to_video(on_imgs, vid_fps, 
                            "out_video_spikes_filter_%s_competition_%s"%
                                       (pop, 'on' if competition_on else 'off'),
                            scale=vid_scale)

        #bipolar population
        if 'cam' in pop:
            continue
        else:
            on_spikes[:]  = ret_spikes['on'][pop]['bipolar']
            off_spikes[:] = ret_spikes['off'][pop]['bipolar']

        on_imgs[:] = imgs_in_T_from_spike_array(on_spikes, w, h, 
                                                0, on_time_ms, ftime_ms,
                                                out_array=True, 
                                                thresh=thresh*20,
                                                up_down = True,)

        off_imgs[:] = imgs_in_T_from_spike_array(off_spikes, w, h, 
                                                 0, on_time_ms, ftime_ms,
                                                 out_array=True, 
                                                 thresh=thresh*20,
                                                 up_down = False,)

        if plot_out_spikes:
            fig = plt.figure(3)
            num_imgs = len(on_imgs)
            rows = num_imgs//cols + (1 if num_imgs%cols else 0)
            figw = 2.5
            fig = plt.figure(figsize=(figw*cols, figw*rows))
            # plt.suptitle("each square is %d ms"%(ftime_ms))
            for i in range(num_imgs):
                ax = plt.subplot(rows, cols, i+1)
                if i == 0:
                    ax.set_title("%d ms frame"%ftime_ms)
                
                on_imgs[i][:, :, 0] = off_imgs[i][:, :, 0]
                my_imshow(ax, on_imgs[i], cmap=None)

            plt.savefig("gss_spikes_filter_%s_competition_%s.png"%
                        (pop, 'on' if competition_on else 'off'),
                        dpi=300)
            plt.close(fig)

        if vid_out_spikes:
            images_to_video(on_imgs, vid_fps, 
                            "gss_video_spikes_filter_%s_competition_%s"%
                                       (pop, 'on' if competition_on else 'off'),
                            scale=vid_scale)
        plt.figure(1)
        plt.plot([t for (_, t) in on_spikes], 
                    [i for (i, _) in on_spikes], '.', 
                    color='green', markersize=2,
                    label='Input - Bipolar - (%d spikes)'%(len(on_spikes))
                )
        plt.ylabel('Neuron Id')
        plt.xlabel('Time (ms)')
        plt.margins(0.1, 0.1)
        lgd = plt.legend(bbox_to_anchor=(1., 1.15), loc='upper right', 
                            ncol=1)
        plt.draw()
        plt.savefig('raster_plot_gss_filter_%s_channel_%s_competition_%s.png'%
                    (pop, 'on', 'on' if competition_on else 'off'),
                    bbox_extra_artists=(lgd,), bbox_inches='tight',
                    dpi=300)
        
        plt.close(spikes_fig_on)

        plt.figure(2)
        plt.plot([t for (_, t) in off_spikes], 
                    [i for (i, _) in off_spikes], '.', 
                    color='red', markersize=2,
                    label='Input - Bipolar - (%d spikes)'%(len(off_spikes))
                )
        plt.ylabel('Neuron Id')
        plt.xlabel('Time (ms)')
        plt.margins(0.1, 0.1)
        lgd = plt.legend(bbox_to_anchor=(1., 1.15), loc='upper right', 
                         ncol=1)
        plt.draw()
        plt.savefig('raster_plot_gss_filter_%s_channel_%s_competition_%s.png'%
                    (pop, 'off', 'on' if competition_on else 'off'),
                    bbox_extra_artists=(lgd,), bbox_inches='tight',
                    dpi=300)
        
        plt.close(spikes_fig_off)

