from common import *

import cv2

def run_sim(sim, run_time, spikes, img_w, img_h, row_bits=8, w2s=4.376069, 
            competition_on=True, direction_on=True):
    
    sim.setup(timestep=1., max_delay=140., min_delay=1.)

    if is_spinnaker(sim):
        sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 200)
        # sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 100)
        # sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 50)
        # sim.set_number_of_neurons_per_core(sim.SpikeSourceArray, 1000)
        # sim.set_number_of_neurons_per_core(sim.SpikeSourceArray, 500)
        sim.set_number_of_neurons_per_core(sim.SpikeSourceArray, 1000)

    cam, dmy_ssa_cam, dmy_prj_cam = setup_cam_pop(sim, spikes, 
                                                  img_w, img_h, w2s=w2s)

    cam.record()
    rbits = int(np.ceil(np.log2(img_h)))
    ret_cfg = {'record': { 'voltages': False, 
                           'spikes': True,
                         },
               # 'w2s': w2s,
               'gabor': False,
               'input_mapping_func': mapf.row_col_to_input,
               'row_bits': rbits,
               'lateral_competition': competition_on,
               'split_cam_off_arg': False,
               'plot_kernels': True,
               'orientation': False,
              }

    if not direction_on:
        ret_cfg['direction'] = False

    mode = dvs_modes[MERGED]
    retina = Retina(sim, cam, img_w, img_h, mode, cfg=ret_cfg)

    print("\n Experiment will begin now...\n")

    sim.run(run_time)

    print("\n Experiment finished running!\n")

    map_w = {}
    out_spikes = {}
    bip_w = {}
    for k in retina.pops.keys():
        out_spikes[k] = {}
        out_spikes[k]['cam'] = get_spikes(retina.cam[k], 'cam__%s'%k)
        map_w[k] = retina._cam_map_projs[k].getWeights(format='array')
        bip_w[k] = {}
        for p in retina.pops[k].keys():
            out_spikes[k][p] = {}
            if 'bip2gang' in retina.projs[k][p]:
                bip_w[k][p] = [np.array(retina.projs[k][p]['bip2gang'][0].
                                                getWeights(format='array')),
                               np.array(retina.projs[k][p]['bip2gang'][1].
                                                getWeights(format='array'))]

            if isinstance(retina.pops[k][p], dict):
                for t in retina.pops[k][p].keys():
                    key = "%s__%s__%s"%(k, p, t)
                    print("\tGettings spikes for %s"%key)
                    out_spikes[k][p][t] = get_spikes(retina.pops[k][p][t],
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

def delete_prev_run():
    import os
    files = os.listdir(os.getcwd())
    for file in files:
        if file.endswith(".png") or file.endswith(".m4v") or \
            file.endswith(".bz2") or file.endswith(".npy") or \
            file.endswith(".pdf") or file.endswith("w.txt"):
            os.remove(os.path.join(os.getcwd(), file))

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
frames = 200

thresh = int(255*0.05) # just for plotting
# thresh = int(255*0.1)

mnist_dir = "../pyDVS/mnist_spikes/" + \
            "mnist_behave_SACCADE_" + \
            "pol_MERGED_enc_TIME_" + \
            "thresh_12_hist_95_00_"+ \
            "inh_False___45_frames_" + \
            "at_90fps_%dx%d_res_spikes/" + \
            "t10k/"
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
direction_on    = True if 0 else False
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
    # print(spikes)
    imgsU = imgs_in_T_from_spike_array(spikes, img_w, img_h, 
                                       0, on_time_ms, ftime_ms,
                                       out_array=False,
                                       thresh=thresh*cam_thresh_scale,
                                       map_func=cam_img_map)

    num_imgs = len(imgsU)

    rows = num_imgs//cols + (1 if num_imgs%cols else 0)
    # fig = plt.figure(figsize=(figw*cols, figw*rows))
    # plt.suptitle("each square is %d ms"%(ftime_ms))
    # for i in range(num_imgs):
    #     ax = plt.subplot(rows, cols, i+1)
    #     my_imshow(ax, imgsU[i], cmap=None)
    # # plot_spikes(spikes)

    # plt.savefig("test_retina_camera.png", dpi=300)
    # plt.close()
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

        spikes_fig_on = plt.figure(1, figsize=(10, 7))
        plt.plot([t for (_, t) in on_spikes], 
                    [i for (i, _) in on_spikes], '|b', markersize=5,
                    label='Output - Ganglion - (%d spikes)'%(len(on_spikes)) )

        spikes_fig_off = plt.figure(2, figsize=(10, 7))
        plt.plot([t for (_, t) in off_spikes], 
                    [i for (i, _) in off_spikes], '|b', markersize=5,
                    label='Output - Ganglion - (%d spikes)'%(len(off_spikes))
                )


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
                            scale=vid_scale, off_images=off_imgs)

        #bipolar population
        if 'cam' in pop:
            continue
        else:
            on_spikes[:]  = ret_spikes['on'][pop]['bipolar']
            off_spikes[:] = ret_spikes['off'][pop]['bipolar']

        on_imgs[:] = imgs_in_T_from_spike_array(on_spikes, w, h, 
                                                0, on_time_ms, ftime_ms,
                                                out_array=True, 
                                                thresh=thresh*thresh_scale,
                                                up_down = True,)

        off_imgs[:] = imgs_in_T_from_spike_array(off_spikes, w, h, 
                                                 0, on_time_ms, ftime_ms,
                                                 out_array=True, 
                                                 thresh=thresh*thresh_scale,
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

            plt.savefig("bip_spikes_filter_%s_competition_%s.png"%
                        (pop, 'on' if competition_on else 'off'),
                        dpi=300)
            plt.close(fig)

        if vid_out_spikes:
            images_to_video(on_imgs, vid_fps, 
                            "bip_video_spikes_filter_%s_competition_%s"%
                                       (pop, 'on' if competition_on else 'off'),
                            scale=vid_scale, off_images=off_imgs)
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
        plt.savefig('raster_plot_filter_%s_channel_%s_competition_%s.png'%
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
        plt.savefig('raster_plot_filter_%s_channel_%s_competition_%s.png'%
                    (pop, 'off', 'on' if competition_on else 'off'),
                    bbox_extra_artists=(lgd,), bbox_inches='tight',
                    dpi=300)
        
        plt.close(spikes_fig_off)

