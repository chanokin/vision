from common import *

def run_sim(sim, run_time, spikes, img_w, img_h, row_bits=8, w2s=4.376069, 
            competition_on=True):
    
    sim.setup(timestep=1., max_delay=140., min_delay=1.)

    cam, dmy_ssa_cam, dmy_prj_cam = setup_cam_pop(sim, spikes, 
                                                img_w, img_h, w2s=w2s)
    
    cam.record()
    target_pop = sim.Population(img_h*img_w*2, sim.IZK_curr_exp, {})
    proj = sim.Projection(cam, target_pop, sim.OneToOneConnector(weights=5.),
                          target='excitatory')
    print("\n Experiment will begin now...\n")

    sim.run(run_time)

    print("\n Experiment finished running!\n")

    out_spikes = {}

    out_spikes['cam'] = get_spikes(cam, 'cam')

    sim.end()

    return out_spikes, []


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
#
plot_cam_spikes = True if 1 else False
simulate_retina = True if 1 else False
competition_on  = True if 1 else False
#
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




if simulate_retina:

    ret_spikes, pop_shapes = run_sim(sim, on_time_ms, spikes, img_w, img_h, 
                                     w2s=4.376069*1.01, 
                                     competition_on=competition_on)

    # print(ret_spikes['on'].keys())
    # print(pop_shapes.keys())
    print("\nPlotting output spikes:\n")


    out_imgs = imgs_in_T_from_spike_array(ret_spikes['cam'], img_w, img_h, 
                                          0, on_time_ms, ftime_ms,
                                          out_array=True, 
                                          thresh=thresh*10,
                                          map_func=cam_img_map,
                                         )

    fig = plt.figure()
    num_imgs = len(out_imgs)
    cols = 6
    rows = num_imgs//cols + (1 if num_imgs%cols else 0)
    figw = 2.5
    fig = plt.figure(figsize=(figw*cols, figw*rows))
    plt.suptitle("each square is %d ms"%(ftime_ms))
    for i in range(num_imgs):
        ax = plt.subplot(rows, cols, i+1)
        my_imshow(ax, out_imgs[i], cmap=None)

    plt.savefig("out_spikes_filter_%s_channel_%s_competition_on_%s.png"%
                ('cam', 'both', 'na'),
                dpi=150)
    plt.close()
    # plt.show()
