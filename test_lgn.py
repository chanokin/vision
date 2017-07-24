from common import *

def run_sim(sim, run_time, spikes, img_w, img_h, row_bits=8, w2s=4.376069, 
            competition_on=True):
    
    sim.setup(timestep=1., max_delay=140., min_delay=1.)

    cam, dmy_ssa_cam, dmy_prj_cam = setup_cam_pop(sim, spikes, 
                                                img_w, img_h, w2s=w2s)

    cam.record()

    ret_cfg = { 'record': { 'voltages': False, 
                            'spikes': True,
                        },
                'w2s': w2s,
                'gabor': False,
                'direction': False,
                'input_mapping_func': mapf.row_col_to_input,
                'row_bits': int(np.ceil(np.log2(img_h))),
                'lateral_competition': competition_on,
                'split_cam_off_arg': False,
                'plot_kernels': True,
            }

    mode = dvs_modes[MERGED]
    retina = Retina(sim, cam, img_w, img_h, mode, cfg=ret_cfg)

    lgn = LGN(sim, retina)

    print("\n Experiment will begin now...\n")

    sim.run(run_time)

    print("\n Experiment finished running!\n")

    retina_spikes = {}
    for k in retina.pops.keys():
        retina_spikes[k] = {}
        retina_spikes[k]['cam'] = get_spikes(retina.cam[k], 'cam__%s'%k)

        for p in retina.pops[k].keys():
            retina_spikes[k][p] = {}
            if isinstance(retina.pops[k][p], dict):
                for t in retina.pops[k][p].keys():
                    key = "Retina: %s %s %s"%(k, p, t)
                    print("\tGettings spikes for %s"%key)
                    retina_spikes[k][p][t] = get_spikes(retina.pops[k][p][t],
                                                        key)

    lgn_spikes = {}
    for k in lgn.pops.keys():
        lgn_spikes[k] = {}
        lgn_spikes[k]['cam'] = get_spikes(retina.cam[k], 'cam__%s'%k)

        for p in lgn.pops[k].keys():
            lgn_spikes[k][p] = {}
            if isinstance(lgn.pops[k][p], dict):
                for t in lgn.pops[k][p].keys():
                    key = "LGN: %s  %s  %s"%(k, p, t)
                    print("\tGettings spikes for %s"%key)
                    lgn_spikes[k][p][t] = get_spikes(lgn.pops[k][p][t],
                                                     key)
    sim.end()

    out_spikes = {'retina': retina_spikes, 'lgn': lgn_spikes}

    return out_spikes, retina.shapes


img_w, img_h = 32, 32
num_neurons = img_w*img_h*2
fps = 100
frames = 110
# frames = 300

thresh = int(255*0.05) # just for plotting
# thresh = int(255*0.1)

mnist_dir = "../../pyDVS/mnist_spikes/" + \
            "mnist_behave_SACCADE_pol_MERGED" + \
            "_enc_RATE_thresh_12_hist_97_00" + \
            "_inh_False___" + \
            "300_frames_at_100fps_32x32_res_spikes/" + \
            "t10k/"

###############################################################################
#
#  S W I T C H E S 
#
###############################################################################

plot_cam_spikes = True if 1 else False
simulate        = True if 1 else False
competition_on  = True if 0 else False

###############################################################################


on_time_ms  = int( frames*(1000./fps) )
ftime_ms    = int( 1000./fps )*5 # how many milliseconds to (frmo frames) to plot
off_time_ms = 0
start_time  = 0
# spikes_dir = os.path.join(mnist_dir, '')
spikes_dir = mnist_dir

spikes = pat_gen.img_spikes_from_to(spikes_dir, num_neurons, 0, 1, 
                                    on_time_ms, off_time_ms,
                                    start_time)

cols = 10
figw = 2.
if plot_cam_spikes:
    # print(spikes)
    imgsU = imgs_in_T_from_spike_array(spikes, img_w, img_h, 
                                        0, on_time_ms, ftime_ms, 
                                        out_array=False, thresh=thresh*10,
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



if simulate:

    all_spikes, pop_shapes = run_sim(sim, on_time_ms, spikes, img_w, img_h, 
                                     w2s=4.376069*1.01, 
                                     competition_on=competition_on)

    # print(ret_spikes['on'].keys())
    # print(pop_shapes.keys())
    print("\nPlotting output spikes:\n")
    ret_spikes = all_spikes['retina']
    out_imgs   = []
    pop_spikes = []
    for channel in ret_spikes.keys():
        for pop in ret_spikes[channel]:
            if 'cam' in pop:
                continue

            if 'cam' in pop:
                w = img_w
                h = img_h
            else:
                w = pop_shapes[pop]['width']
                h = pop_shapes[pop]['height']
            
            print("\tfor Retina - %s - %s"%(channel, pop))

            is_up = True if channel == 'on' else False

            out_imgs[:] = []
            pop_spikes[:] = []
            if 'cam' in pop:
                pop_spikes[:] = ret_spikes[channel][pop]
            else:
                try:
                    pop_spikes[:] = ret_spikes[channel][pop]['ganglion']
                    # pop_spikes[:] = ret_spikes[channel][pop]['bipolar']
                except:
                    print("No spikes for retina %s - %s"%(channel, pop))
                    continue

            out_imgs[:] = imgs_in_T_from_spike_array(pop_spikes, w, h, 
                                                    0, on_time_ms, ftime_ms,
                                                    out_array=True, 
                                                    thresh=thresh*10,
                                                    up_down = is_up,
                                                    #  map_func=row_major_map
                                                    )

            fig = plt.figure()
            num_imgs = len(out_imgs)
            rows = num_imgs//cols + (1 if num_imgs%cols else 0)
            figw = 2.5
            fig = plt.figure(figsize=(figw*cols, figw*rows))
            plt.suptitle("each square is %d ms"%(ftime_ms))
            for i in range(num_imgs):
                ax = plt.subplot(rows, cols, i+1)
                my_imshow(ax, out_imgs[i], cmap=None)

            plt.savefig("ret_spikes_filter_%s_channel_%s_competition_on_%s.png"%
                        (pop, channel, competition_on),
                        dpi=150)
            plt.close()
            # plt.show()


    lgn_spikes = all_spikes['lgn']
    for channel in lgn_spikes.keys():
        for pop in lgn_spikes[channel]:
            if 'cam' in pop:
                continue

            if 'cam' in pop:
                w = img_w
                h = img_h
            else:
                w = pop_shapes[pop]['width']
                h = pop_shapes[pop]['height']
            
            print("\tfor LGN - %s - %s"%(channel, pop))

            is_up = True if channel == 'on' else False

            out_imgs[:] = []
            pop_spikes[:] = []
            try:
                pop_spikes[:] = lgn_spikes[channel][pop]['ganglion']
            except Exception, e:
                print("No spikes for lgn %s - %s"%(channel, pop))
                print(str(e))
                continue

            
            out_imgs[:] = imgs_in_T_from_spike_array(pop_spikes, w, h, 
                                                    0, on_time_ms, ftime_ms,
                                                    out_array=True, 
                                                    thresh=thresh*10,
                                                    up_down = is_up,
                                                    #  map_func=row_major_map
                                                    )

            fig = plt.figure()
            num_imgs = len(out_imgs)
            rows = num_imgs//cols + (1 if num_imgs%cols else 0)
            figw = 2.5
            fig = plt.figure(figsize=(figw*cols, figw*rows))
            plt.suptitle("each square is %d ms"%(ftime_ms))
            for i in range(num_imgs):
                ax = plt.subplot(rows, cols, i+1)
                my_imshow(ax, out_imgs[i], cmap=None)

            plt.savefig("lgn_spikes_filter_%s_channel_%s_competition_on_%s.png"%
                        (pop, channel, competition_on),
                        dpi=150)
            plt.close()

