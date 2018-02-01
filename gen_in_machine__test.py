"""
Synfirechain-like example
"""
#!/usr/bin/python
import os
import glob
import spynnaker7.pyNN as p
import numpy as np, pylab as plt
from pyNN.random import NumpyRNG, RandomDistribution
import sys
import binascii
import time




def run_sim(num_pre, num_post, spike_times, run_time, weight=5.6, delay=40., gom=False,
            conn_type='one2one', use_stdp=False, mad=False, prob=0.5):
    model = p.IF_curr_exp
    p.setup( timestep = 1.0, min_delay = 1.0, max_delay = 144.0 )
    # if num_pre <= 10:
    #     p.set_number_of_neurons_per_core(model, 4)
    #     p.set_number_of_neurons_per_core(p.SpikeSourceArray, 5)
    # elif 10 < num_pre <= 50:
    #     p.set_number_of_neurons_per_core(model, 20)
    #     p.set_number_of_neurons_per_core(p.SpikeSourceArray, 21)
    # elif 50 < num_pre <= 100:
    #     p.set_number_of_neurons_per_core(model, 50)
    #     p.set_number_of_neurons_per_core(p.SpikeSourceArray, 51)
    # else:
    if use_stdp:
        p.set_number_of_neurons_per_core(model, 150)

    p.set_number_of_neurons_per_core(p.SpikeSourceArray, 2000)


    cell_params_lif = {  'cm'        : 1.0, # nF
                         'i_offset'  : 0.00,
                         'tau_m'     : 10.0,
                         'tau_refrac': 4.0,
                         'tau_syn_E' : 1.0,
                         'tau_syn_I' : 1.0,
                         'v_reset'   : -70.0,
                         'v_rest'    : -65.0,
                         'v_thresh'  : -60.0
                      }

    cell_params_pos = {
        'spike_times': spike_times,
    }
    w2s = weight
    dly = delay
    rng = NumpyRNG( seed = 1 )
    if use_stdp:
        td = p.SpikePairRule(tau_minus=1., tau_plus=1.)
        wd = p.AdditiveWeightDependence(w_min=0, w_max=20., A_plus=0.0, A_minus=0.0)
        stdp = p.STDPMechanism(timing_dependence=td, weight_dependence=wd)
        syn_dyn = p.SynapseDynamics(slow=stdp)
    else:
        syn_dyn = None

    sink = p.Population( num_post, model, cell_params_lif, label='sink')
    # sink1 = p.Population( nNeuronsPost, model, cell_params_lif, label='sink1')

    source0 = p.Population( num_pre, p.SpikeSourceArray, cell_params_pos,
                            label='source_0')

    # source1 = p.Population( nNeurons, p.SpikeSourceArray, cell_params_pos,
    #                         label='source_1')


    if conn_type == 'one2one':
        conn = p.OneToOneConnector(weights=w2s, delays=dly, generate_on_machine=gom)
    elif conn_type == 'all2all':
        conn = p.AllToAllConnector(weights=w2s, delays=dly, generate_on_machine=gom)
    elif conn_type == 'fixed_prob':
        conn = p.FixedProbabilityConnector(prob, weights=w2s, delays=dly,
                                           generate_on_machine=gom)
    else:
        raise Exception("Not a valid connector for test")

    proj = p.Projection( source0, sink, conn, target='excitatory',
                         synapse_dynamics=syn_dyn,
                         label=' source 0 to sink - EXC - delayed')


    # sink.record_v()
    # sink.record_gsyn()
    sink.record()


    print("Running for {} ms".format(run_time))
    t0 = time.time()
    p.run(run_time)
    time_to_run = time.time() - t0
    v = None
    gsyn = None
    spikes = None

    # v = np.array(sink.get_v(compatible_output=True))
    # gsyn = sink.get_gsyn(compatible_output=True)
    spikes = sink.getSpikes(compatible_output=True)
    w = proj.getWeights(format='array')
    p.end()

    return v, gsyn, spikes, w, time_to_run


ms = 8
verbose_conn = {'one2one': 'One-To-One', 'all2all': 'All-To-All',
                'fixed_prob': 'Fixed Probability'}

# delay_list = [10, 20, 40, 80, 144]
# delay_list = [10, 144]
delay_list = [10, 40]
# delay_list = [10]
delay_list = [40]
# delay_list = [144]

stdp_list = [False, True]
stdp_list = [True]
# stdp_list = [False]

conn_type_list = ['one2one', 'all2all', 'fixed_prob']
# conn_type_list = ['one2one', 'all2all']
# conn_type_list = ['one2one', 'fixed_prob']
conn_type_list = ['fixed_prob', 'all2all']
# conn_type_list = ['fixed_prob']
conn_type_list = ['all2all']
# conn_type_list = ['one2one']


run_baseline = True if 1==0 else False
run_test     = True if 1==1 else False

prob = 0.2
w = 4.5
# num_pre_list = [250, 500, 1000, 2500, 5000]
# num_pre_list = [10, 50, 100, 250, 500, 1000, 2500, 5000]
prevn = 5
currn = 8
# prevn = 377
# currn = 610
tmp = 0
num_pre_list = []
for i in range(11):
    num_pre_list.append(currn)
    tmp = prevn
    prevn = currn
    currn += tmp

# sys.exit(0)
# num_pre_list = [610]
# num_pre_list = [80]
# num_pre_list = [987]
num_pre_list += [1300, 1400, 1500]
# num_pre_list += [i*1000 for i in range(2, 5)]
num_pre_list += [i*1000 for i in range(2, 11)]
# num_pre_list = [i*1000 for i in range(8, 11)]
# num_pre_list = [499]
# num_pre_list = [10000]
# num_pre_list = [2000, 5000, 7000, 10000]
# num_pre_list = [7000, 10000]
# print(num_pre_list)
# sys.exit(0)

log_fname = "on_spinnaker_connector_build.txt"
test_log = open(log_fname, 'w')
for f in glob.glob("./post_pop*.pdf"):
    os.remove(f)

# for f in glob.glob("./host_vs_spinnaker_time_compare_*.txt"):
#     os.remove(f)


first_round = True
neuron_ids = []
spikes_ts = []

num_tests = 2
for test_idx in range(num_tests):
    for conn_type in conn_type_list:
        for d in delay_list:
            for use_stdp in stdp_list:
                for num_pre in num_pre_list:
                    # if conn_type == 'all2all' and num_pre > 1500:
                    #     continue
                    if conn_type == 'all2all' and num_pre > 8000:
                        continue


                    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    print("+\n+ %02d - conn: %s\tdelay: %s\tstdp: %s\tneurons: %d\n+"%
                          (test_idx, conn_type, d,
                           ("on" if use_stdp else "off"), num_pre))
                    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    time.sleep(1.)
                    num_post = num_pre
                    max_t = 100
                    spike_times = [[1 + np.random.randint(max_t)]
                                               for _ in range(num_pre)]
                    spike_times[ 0][0] = 1
                    spike_times[-1][0] = max_t
                    # spike_times = [[1 + i * 5] for i in range(num_pre)]
                    # spike_times = [[1 + i] for i in range(num_pre)]

                    time_log_fname = "host_vs_spinnaker_time_compare_conn_%s_" \
                                     "d_%d_stdp_%s.txt"%(conn_type, d, use_stdp)

                    time_log = open(time_log_fname, 'a+')


                    test_log  = open(log_fname, 'a+')
                    run_time = np.max(spike_times)*1.2 + d + 50.

                    if not first_round:
                        test_log.write('\n\n')

                    test_log.write("Testing connector %s\n"%verbose_conn[conn_type])
                    if conn_type == 'fixed_prob':
                        test_log.write("    Probability %03.2f %%\n"%(100*prob))

                    test_log.write("    Weights = %s\n"%w)
                    test_log.write("    Delays = %s\n"%d)
                    test_log.write("    Using STDP? %s\n"%use_stdp)
                    test_log.write("    Running for %d ms\n"%run_time)
                    test_log.write("    Num Pre %d, Post %d\n"%(num_pre, num_post))
                    test_log.write("------------------------------------------------\n")

                    base_time = 0
                    spinn_time = 0

                    time_log.write("\n")
                    time_log.write("%s, " % (num_pre))

                    if run_baseline:
                        start_t = time.time()
                        dv, dgsyn, dspikes, dw, ttr = run_sim(num_pre, num_post,
                                                         spike_times,
                                                         run_time,
                                                         weight=w, delay=d, gom=False,
                                                         conn_type=conn_type,
                                                         use_stdp=use_stdp, prob=prob)
                        base_time = time.time() - start_t
                        test_log.write("\nOn Host time = %s seconds\n"%base_time)
                    time_log.write("%s, " % (base_time))

                    if run_test:
                        start_t = time.time()
                        sv, sgsyn, sspikes, sw, ttr = run_sim(num_pre, num_post,
                                                         spike_times,
                                                         run_time,
                                                         weight=w, delay=d, gom=True,
                                                         conn_type=conn_type,
                                                         use_stdp=use_stdp, prob=prob)
                        spinn_time = time.time() - start_t
                        test_log.write("\nOn SpiNNaker time = %s seconds\n"%spinn_time)


                    time_log.write("%s"%(spinn_time))
                    time_log.close()

                    if run_test:
                        test_log.write("\nWeights generated on SpiNNaker:\n")
                        test_log.write(np.array_str(sw, max_line_width=160, precision=6))

                    if run_baseline:
                        test_log.write("\nWeights generated on Host:\n")
                        test_log.write(np.array_str(dw, max_line_width=160, precision=6))

                    test_log.write("\n\n")

                    if num_pre < 500:
                        fig = plt.figure(figsize=(20, 10))
                        ax = plt.subplot(1, 1, 1)

                        neuron_ids[:] = []
                        spikes_ts[:] = []
                        nid = 0
                        for ts in spike_times:
                            for t in ts:
                                spikes_ts.append(t)
                                neuron_ids.append(nid)
                            nid += 1
                        plt.plot(spikes_ts, neuron_ids, '^g', markersize=ms,
                                 label='Source', color=(0., 1., 0., 0.5),
                                 fillstyle = 'none', linewidth = 0.1)

                        if run_baseline:
                            ax.set_title(
                                "Spikes from both simulations (OnMachine Vs OnHost)")
                            test_log.write("Spikes on Host: %d\n"%len(dspikes))
                            neuron_ids[:] = [n_id  for (n_id, spk_t) in dspikes]
                            spikes_ts[:]  = [spk_t for (n_id, spk_t) in dspikes]
                            plt.plot(spikes_ts, neuron_ids, '+r', markersize=ms,
                                     label='Host', color=(0., 0., 1., 0.3))

                        if run_test:
                            test_log.write("Spikes on SpiNNaker: %d\n" % len(sspikes))
                            neuron_ids[:] = [n_id  for (n_id, spk_t) in sspikes]
                            spikes_ts[:]  = [spk_t for (n_id, spk_t) in sspikes]
                            plt.plot(spikes_ts, neuron_ids, 'xb', markersize=ms,
                                     label='SpiNN', color=(1., 0., 0., 0.3))

                        ax.margins(0.1, 0.1)
                        ax.legend(bbox_to_anchor=(0.8, 1), loc=2, borderaxespad=0.)
                        ax.grid(which='both')
                        plt.draw()
                        plt.savefig("post_pop_n_%d___conn_%s___d_%d___use_stdp_%s_.pdf"%
                                    (num_post, conn_type, d, use_stdp))
                        plt.close()

                    if run_baseline:
                        d_conns_per_pre = (~np.isnan(dw)).sum(1)

                    if run_test:
                        s_conns_per_pre = (~np.isnan(sw)).sum(1)

                    if run_baseline and run_test:

                        test_log.write("Spikes on Host == SpiNNaker? %s\n" %
                                       (len(dspikes) == len(sspikes)))
                        if conn_type != 'fixed_prob':

                            if np.array_equal(s_conns_per_pre, d_conns_per_pre):
                      #TODO: CHECK for same spike times in fixed number/target conns
                                test_log.write("Test passed!!!\n")
                            else:
                                test_log.write("Test failed! =(\n")
                                test_log.write("Connections per Pre-Synaptic Neuron "
                                               "generated on Host\n")
                                test_log.write(np.array_str(d_conns_per_pre))

                                test_log.write("Connections per Pre-Synaptic Neuron "
                                               "generated on SpiNNaker\n")
                                test_log.write(np.array_str(s_conns_per_pre))
                                # sys.exit(1)
                        else:
                            smean = s_conns_per_pre.mean()
                            dmean = d_conns_per_pre.mean()
                            avg_conn = (num_post * prob)
                            error = avg_conn*0.05
                            test_log.write("Expected average connections per "
                                           "Pre-Synaptic: "
                                           "%s +/- %03.3f\n" % (avg_conn, error))
                            test_log.write("SpiNNaker-generated average connections "
                                           "per Pre-Synaptic neuron: %s\n" % smean)
                            test_log.write("Host-generated average connections per " +
                                           "Pre-Synaptic neuron: %s\n" % dmean)

                            if np.abs(avg_conn - smean) < error and \
                                      np.abs(avg_conn - dmean) < error:
                                test_log.write("Test passed!!!\n")
                            else:
                                test_log.write("Test failed! =(\n")
                            #    sys.exit(1)
                    test_log.write(
                           "====================================================\n\n")

                    test_log.close()
                    first_round = False
                    # time.sleep(0.5)

test_log.close()