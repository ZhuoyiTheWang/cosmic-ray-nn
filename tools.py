import numpy as np
import os
import tensorflow as tf
from keras.layers import *
import fitting
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy import optimize
from scipy.stats import binom, binned_statistic, moment, sem

def fit_chi2(func, datax, datay, datayerror=None, p_init=None):
    if datayerror is None:
        datayerror = np.maximum(1, np.sqrt(np.abs(datay)))
        # datayerror = np.maximum(1, np.sqrt(datay))
    popt, cov = optimize.curve_fit(func, datax, datay, sigma=datayerror,
                                   absolute_sigma=True, p0=p_init, ftol=1e-6, xtol=1e-6)
    y_fit = func(datax, *popt)
    chi2 = np.sum((y_fit - datay) ** 2 / datayerror ** 2)
    if len(datax) != len(popt):
        chi2ndf = 1. * chi2 / (len(datax) - len(popt))
    else:
        chi2ndf = np.NAN
    func2 = lambda x: func(x, *popt)
    return popt, cov, chi2ndf, func2

# Generator function to stream input files from input directory
def stream_input_one_ch(file_list, batchsize, n_stations, grid_stations, tracelen, elongation_rate, weight_table):
    i = 0
    if weight_table != 'off' and weight_table != 'test':
        sample_weight_lookup = np.load(weight_table)
        xmax_bounds = sample_weight_lookup['xmax']
        weights = sample_weight_lookup['sample_weights']
    while True:
        inp_trace_ssd = np.zeros(batchsize * grid_stations * tracelen).reshape((batchsize, n_stations, n_stations, tracelen, 1))
        inp_trace_wcd = np.zeros(batchsize * grid_stations * tracelen).reshape((batchsize, n_stations, n_stations, tracelen, 1))
        inp_energy_zenith_time = np.zeros(batchsize * grid_stations * 3).reshape((batchsize, n_stations, n_stations, 3))
        inp_xmax = np.zeros(batchsize)
        if weight_table != 'off' and weight_table != 'test':
            inp_sample_weights = np.zeros(batchsize)
        for b in range(batchsize):
            if i == len(file_list):
                i = 0
                file_list = shuffle(file_list, random_state=0)
            sample = file_list[i]
            loaded_file = np.load(sample)
            i += 1
            # Read inputs from file
            inp_trace_ssd[b] = loaded_file['inp_trace_ssd'].reshape((n_stations, n_stations, tracelen, 1))
            inp_trace_wcd[b] = loaded_file['inp_trace_wcd'].reshape((n_stations, n_stations, tracelen, 1))
            inp_energy_zenith_time[b] = loaded_file['inp_energy_zenith_time']
            #Elongation rate settings
            if elongation_rate == 'off':
                inp_xmax[b] = loaded_file['inp_xmax']
            elif elongation_rate == 'simple':
                inp_xmax[b] = loaded_file['inp_xmax_er_simple']
            elif elongation_rate == 'full':
                inp_xmax[b] = loaded_file['inp_xmax_er_full']
            # Setting weights
            if weight_table != 'off' and weight_table != 'test':
                inp_sample_weights[b] = calculate_sample_weights(inp_xmax[b], xmax_bounds, weights)
        # Feed to networks
        if weight_table == 'test':
            yield ([inp_trace_ssd, inp_trace_wcd, inp_energy_zenith_time])
        elif weight_table == 'off':
            yield ([inp_trace_ssd, inp_trace_wcd, inp_energy_zenith_time], [inp_xmax])
        else:
            yield ([inp_trace_ssd, inp_trace_wcd, inp_energy_zenith_time], [inp_xmax], [inp_sample_weights])


def stream_input_two_ch(file_list, batchsize, n_stations, grid_stations, tracelen, elongation_rate, weight_table,
                        black_station_rate):
    i = 0
    # black stations that says "ni"
    ni = np.zeros(tracelen)
    black_station_probabilities = np.array([binom.pmf(1, grid_stations, black_station_rate),
                                            binom.pmf(2, grid_stations, black_station_rate),
                                            binom.pmf(3, grid_stations, black_station_rate),
                                            binom.pmf(4, grid_stations, black_station_rate)])
    if weight_table != 'off' and weight_table != 'test':
        sample_weight_lookup = np.load(weight_table)
        xmax_bounds = sample_weight_lookup['xmax']
        weights = sample_weight_lookup['sample_weights']
    while True:
        inp_traces = np.zeros(batchsize * grid_stations * tracelen * 2).reshape((batchsize, n_stations, n_stations, tracelen, 2))
        inp_energy_zenith_time = np.zeros(batchsize * grid_stations * 3).reshape((batchsize, n_stations, n_stations, 3))
        inp_xmax = np.zeros(batchsize)
        if weight_table != 'off' and weight_table != 'test':
            inp_sample_weights = np.zeros(batchsize)
        for b in range(batchsize):
            if i == len(file_list):
                i = 0
                # random state for trekkies
                file_list = shuffle(file_list, random_state=1701)
            sample = file_list[i]
            loaded_file = np.load(sample)
            i += 1
            # Read inputs from file
            ssd_trace = loaded_file['inp_trace_ssd']
            wcd_trace = loaded_file['inp_trace_wcd']
            #Applying black station bootstrap
            if black_station_rate != 0:
                black_stations = the_tanks_who_say_ni(black_station_probabilities, grid_stations)
                if not black_stations.size == 0:
                    ssd_trace = ssd_trace.reshape((grid_stations, tracelen))
                    wcd_trace = wcd_trace.reshape((grid_stations, tracelen))
                    ssd_trace[black_stations] = ni
                    wcd_trace[black_stations] = ni
                    ssd_trace = ssd_trace.reshape((n_stations, n_stations, tracelen))
                    wcd_trace = wcd_trace.reshape((n_stations, n_stations, tracelen))
            inp_traces[b] = np.stack((ssd_trace, wcd_trace), axis=-1)
            inp_energy_zenith_time[b] = np.round(loaded_file['inp_energy_zenith_time'], 2)
            #Elongation rate settings
            if elongation_rate == 'off':
                inp_xmax[b] = loaded_file['inp_xmax']
            elif elongation_rate == 'simple':
                inp_xmax[b] = loaded_file['inp_xmax_er_simple']
            elif elongation_rate == 'full':
                inp_xmax[b] = loaded_file['inp_xmax_er_full']
            # Setting weights
            if weight_table != 'off' and weight_table != 'test':
                inp_sample_weights[b] = calculate_sample_weights(inp_xmax[b], xmax_bounds, weights)
        # Feed to networks
        if weight_table == 'test':
            yield ([inp_traces, inp_energy_zenith_time])
        elif weight_table == 'off':
            yield ([inp_traces, inp_energy_zenith_time], [inp_xmax])
        else:
            yield ([inp_traces, inp_energy_zenith_time], [inp_xmax], [inp_sample_weights])


def stream_traces_cnn(file_list, batchsize, n_stations, grid_stations, tracelen, elongation_rate, weight_table):
    i = 0
    if weight_table != 'off' and weight_table != 'test':
        sample_weight_lookup = np.load(weight_table)
        xmax_bounds = sample_weight_lookup['xmax']
        weights = sample_weight_lookup['sample_weights']
    while True:
        inp_traces = np.zeros(batchsize * grid_stations * tracelen * 2).reshape((batchsize, grid_stations, tracelen, 2))
        inp_energy_zenith_time = np.zeros(batchsize * grid_stations * 3).reshape((batchsize, n_stations, n_stations, 3))
        inp_xmax = np.zeros(batchsize)
        inp_xmaxforweights = np.zeros(batchsize)
        if weight_table != 'off' and weight_table != 'test':
            inp_sample_weights = np.zeros(batchsize)
        for b in range(batchsize):
            if i == len(file_list):
                i = 0
                file_list = shuffle(file_list, random_state=0)
            sample = file_list[i]
            loaded_file = np.load(sample)
            i += 1
            # Read inputs from file
            ssd_trace = loaded_file['inp_trace_ssd'][:, :, :tracelen].reshape((grid_stations, tracelen))
            wcd_trace = loaded_file['inp_trace_wcd'][:, :, :tracelen].reshape((grid_stations, tracelen))
            inp_traces[b] = np.stack((ssd_trace, wcd_trace), axis=-1)
            inp_energy_zenith_time[b] = loaded_file['inp_energy_zenith_time']
            #Elongation rate settings
            if elongation_rate == 'off':
                inp_xmax[b] = loaded_file['inp_xmax']
                inp_xmaxforweights[b] = loaded_file['inp_xmax_er_full']
            elif elongation_rate == 'simple':
                inp_xmax[b] = loaded_file['inp_xmax_er_simple']
            elif elongation_rate == 'full':
                inp_xmax[b] = loaded_file['inp_xmax_er_full']
            # Setting weights
            if weight_table != 'off' and weight_table != 'test':
                inp_sample_weights[b] = calculate_sample_weights(inp_xmaxforweights[b], xmax_bounds, weights)
        inp_traces = inp_traces.reshape((batchsize, grid_stations, tracelen, 2, 1))
        # Feed to networks
        if weight_table == 'test':
            yield ([inp_traces, inp_energy_zenith_time])
        elif weight_table == 'off':
            yield ([inp_traces, inp_energy_zenith_time], [inp_xmax])
        else:
            yield ([inp_traces, inp_energy_zenith_time], [inp_xmax], [inp_sample_weights])


def stream_traces_lstm(file_list, batchsize, grid_stations, tracelen, type=None):
    i = 0
    while True:
        inp_traces = np.empty((0, 1200, 1))
        inp_trace_types = np.empty((0, 1))
        if type == 'test':
            yield ([file_list.reshape(batchsize,1200,1)])
        else:
            for b in range(batchsize):
                if i == len(file_list):
                    i = 0
                    file_list = shuffle(file_list, random_state=0)
                sample = file_list[i]
                loaded_file = np.load(sample)
                i += 1
                # Read inputs from file
                ssd_traces = loaded_file['inp_trace_ssd'].reshape((grid_stations, tracelen))
                ssd_traces = ssd_traces[~np.all(ssd_traces == 0, axis=1)]
                wcd_traces = loaded_file['inp_trace_wcd'].reshape((grid_stations, tracelen))
                wcd_traces = wcd_traces[~np.all(wcd_traces == 0, axis=1)]
                ssd_length = len(ssd_traces)
                wcd_length = len(wcd_traces)
                inp_traces = np.append(inp_traces, ssd_traces.reshape((ssd_length, tracelen, 1)), axis=0)
                inp_traces = np.append(inp_traces, wcd_traces.reshape((wcd_length, tracelen, 1)), axis=0)
                inp_trace_types = np.append(inp_trace_types, np.zeros(ssd_length).reshape((ssd_length, 1)))
                inp_trace_types = np.append(inp_trace_types, np.ones(wcd_length).reshape((wcd_length, 1)))
            indices = np.arange(inp_trace_types.shape[0])
            shuffled_indices = shuffle(indices, random_state=0)
            inp_traces = inp_traces[shuffled_indices]
            inp_trace_types = inp_trace_types[shuffled_indices]
            yield ([inp_traces], [inp_trace_types])


def the_tanks_who_say_ni(black_station_probabilities, grid_stations):
    rand = np.random.uniform(0.0, 1.0)
    if rand > np.sum(black_station_probabilities):
        nblack_tanks = 0
    elif rand > np.sum(black_station_probabilities[1:]):
        nblack_tanks = 1
    elif rand > np.sum(black_station_probabilities[2:]):
        nblack_tanks = 2
    elif rand > np.sum(black_station_probabilities[3:]):
        nblack_tanks = 3
    else:
        nblack_tanks = 4
    idx_black_stations = np.random.choice(range(grid_stations), nblack_tanks, replace=False)
    return idx_black_stations


def norm_arr_time(abs_time, n_stations, method='min'):
    rel_time = abs_time
    rel_time[rel_time == 0] = np.nan
    u = np.isnan(rel_time)
    if method == 'min':
        rel_time -= np.nanmin(abs_time, axis=1)[:, np.newaxis]
    elif method == 'center':
        rel_time -= np.nanmean(abs_time, axis=1)[:, np.newaxis]
        rel_time /= np.nanstd(rel_time)
    else:
        print('Please specify calculation method for rel. arrival time')
    rel_time[u] = 0
    try:
        rel_time = rel_time.reshape((abs_time.shape[0], n_stations, n_stations))
    except Exception as e:
        print(e)
        rel_time = rel_time.reshape((abs_time.shape[0], n_stations, n_stations))
    return rel_time


def weighted_avg_and_stdof(values, weights):
  average = np.average(values, weights=weights)
  variance = np.average((values - average) ** 2, weights=weights)
  return (average, (variance)**0.5)


# Metric to control for standart deviation
def resolution(y_true, y_pred):
    mean, var = tf.nn.moments((y_true - y_pred), axes=[0])
    return tf.sqrt(var)

def bias(y_true, y_pred):
    mean, var = tf.nn.moments((y_true - y_pred), axes=[0])
    return mean

# Calculates 68% quantile of distribution
def percentile68(x):
    return np.percentile(x, 68)


# Densely Connected SeparableConvolution2D Layer
def densely_connected_sep_conv(z, nfilter, **kwargs):
    c = SeparableConv2D(nfilter, (3, 3), padding='same', depth_multiplier=1, **kwargs)(z)
    return concatenate([z, c], axis=-1)


def load_mc_trace_truth_lstm(input_files, grid_stations, tracelen, mc_truth_file):
    if os.path.isfile(mc_truth_file):
        loaded = np.load(mc_truth_file)
        true_trace_type = loaded['true_trace_type']
    else:
        true_trace_type = np.empty((0, 1))
        for count, file in enumerate(input_files, 0):
            loaded = np.load(file)
            ssd_traces = loaded['inp_trace_ssd'].reshape((grid_stations, tracelen))
            ssd_traces = ssd_traces[~np.all(ssd_traces == 0, axis=1)]
            wcd_traces = loaded['inp_trace_wcd'].reshape((grid_stations, tracelen))
            wcd_traces = wcd_traces[~np.all(wcd_traces == 0, axis=1)]
            ssd_length = len(ssd_traces)
            wcd_length = len(wcd_traces)
            true_trace_type = np.append(true_trace_type, np.zeros(ssd_length).reshape((ssd_length, 1)))
            true_trace_type = np.append(true_trace_type, np.ones(wcd_length).reshape((wcd_length, 1)))
        # indices = np.arange(true_trace_type.shape[0])
        # shuffled_indices = shuffle(indices, random_state=0)
        # true_trace_type = true_trace_type[shuffled_indices]
        np.savez(mc_truth_file, true_trace_type=np.asarray(true_trace_type.flatten(), dtype=float))
        true_trace_type = np.asarray(true_trace_type.flatten())
    return true_trace_type


def load_mc_trace_truth_cnn(input_files, grid_stations, tracelen, mc_truth_file):
    if os.path.isfile(mc_truth_file):
        loaded = np.load(mc_truth_file)
        true_particle_asymmetry = loaded['true_particle_asymmetry']
    else:
        true_particle_asymmetry = np.empty((0, 2))
        true_particle_asymmetry_log = np.empty((0, 2))
        for count, file in enumerate(input_files, 0):
            loaded = np.load(file)
            # Read inputs from file
            ssd_traces = loaded['inp_trace_ssd'].reshape((grid_stations, tracelen))
            wcd_traces = loaded['inp_trace_wcd'].reshape((grid_stations, tracelen))
            mu_wcd = loaded['particle_counts_wcd'][:, 0]
            mu_ssd = loaded['particle_counts_ssd'][:, 0]
            log_mu_wcd = np.log(mu_wcd, where=mu_wcd > 1.0)
            log_mu_ssd = np.log(mu_ssd, where=mu_ssd > 1.0)
            # np.nan_to_num(log_mu_wcd, copy=False)
            # np.nan_to_num(log_mu_ssd, copy=False)
            # # Stack inputs
            stacked_traces = np.stack((ssd_traces, wcd_traces), axis=-1).reshape((81, 1200, 2, 1))
            stacked_nmu = np.stack((log_mu_ssd, log_mu_wcd), axis=-1)
            stacked_nmu2 = np.stack((mu_ssd, mu_wcd), axis=-1)
            nonzero_indices = np.unique(np.where(~np.all(stacked_traces == 0, axis=1))[0])
            stacked_traces = stacked_traces[nonzero_indices].reshape((len(nonzero_indices), tracelen, 2, 1))
            stacked_nmu = stacked_nmu[nonzero_indices]
            stacked_nmu2 = stacked_nmu2[nonzero_indices]
            true_particle_asymmetry = np.append(true_particle_asymmetry, stacked_nmu2, axis=0)
            true_particle_asymmetry_log = np.append(true_particle_asymmetry_log, stacked_nmu, axis=0)
        np.savez(mc_truth_file, true_particle_asymmetry=true_particle_asymmetry, true_particle_asymmetry_log=true_particle_asymmetry_log)
    return true_particle_asymmetry, true_particle_asymmetry_log


def load_mc_truth(input_files, mc_truth_file, elongation_rate):
    if os.path.isfile(mc_truth_file):
        print('Reading from existing MC file')
        loaded = np.load(mc_truth_file)
        x_test_xmax = loaded['xmax_true']
        x_test_energy = loaded['energy']
        x_test_zenith = loaded['zenith']
        event_id = loaded['event_id']
        particle = loaded['particle']
    else:
        print('Creating new MC file')
        x_test_xmax = np.zeros(len(input_files))
        x_test_energy = np.zeros(len(input_files))
        x_test_zenith = np.zeros(len(input_files))
        event_id = []
        particle = []
        for count, file in enumerate(input_files, 0):
            loaded = np.load(file)
            if loaded['inp_xmax_er_full'] > 1000.0:
                shutil.move(file, '/home/schroeder/')
            else:
                event_id.append(str(os.path.basename(file).split("_")[3]))
                particle.append(str(os.path.basename(file).split("_")[4]))
                x_test_energy[count] = np.round(loaded['inp_energy_zenith_time'][0, 0, 0], 3)
                x_test_zenith[count] = np.round(loaded['inp_energy_zenith_time'][0, 0, 1], 3)
                if elongation_rate == 'off':
                    x_test_xmax[count] = loaded['inp_xmax']
                if elongation_rate == 'simple':
                    x_test_xmax[count] = loaded['inp_xmax_er_simple']
                if elongation_rate == 'full':
                    x_test_xmax[count] = loaded['inp_xmax_er_full']
        np.savez(mc_truth_file, event_id=np.asarray(event_id), particle=np.asarray(particle),
                 xmax_true=np.asarray(x_test_xmax.flatten(), dtype=float),
                 energy=np.asarray(x_test_energy.flatten(), dtype=float),
                 zenith=np.asarray(x_test_zenith.flatten(), dtype=float))
        x_test_xmax = np.asarray(x_test_xmax.flatten(), dtype=float)
        x_test_energy = np.asarray(x_test_energy.flatten(), dtype=float)
        x_test_zenith = np.asarray(x_test_zenith.flatten(), dtype=float)
    return x_test_xmax, x_test_energy, x_test_zenith, event_id, particle


def remove_elongation_correction(y_true, y_pred, energy, elongation_rate):
    if elongation_rate == 'off':
        print('Elongation rate correction is off')
    if elongation_rate == 'simple':
        y_true = y_true + 60 * (energy - 19)
        y_pred = y_pred + 60 * (energy - 19)
    if elongation_rate == 'full':
        y_true = y_true + (600.00 + 63.1 * (energy - 18) - 1.97 * (energy - 18) ** 2)
        y_pred = y_pred + (600.00 + 63.1 * (energy - 18) - 1.97 * (energy - 18) ** 2)
    return y_true, y_pred


def create_sample_weights(mc_truth_file, sample_weight_lookup_table, data_list, max_weight, elongation_rate, plot_dir):
    if not os.path.isfile(sample_weight_lookup_table):
        low = 1/max_weight
        y_true, energy, zenith, event_id, particle = load_mc_truth(data_list, mc_truth_file, elongation_rate)
        lower_xmax = int(np.floor(np.min(y_true) / 10.0)) * 10
        upper_xmax = int(np.ceil(np.max(y_true) / 10.0)) * 10 + 1
        hist, bins = np.histogram(y_true, bins=np.arange(lower_xmax, upper_xmax, 10))
        weighted_hist = hist/max(hist) * (1-low) + low
        np.savez(sample_weight_lookup_table, xmax=bins[:-1], sample_weights=1/weighted_hist)
        plotting.shower_parameter_bar(bins[:-1], hist, r'$X_{max,true,corr}$', r'${g/cm^2}$',
                                      'Xmax-true_train-val', '# Events', str(len(y_true)) + ' Events', plot_dir, log_x=False, log_y=False)
        plotting.shower_parameter_bar(bins[:-1], 1/(weighted_hist), r'$X_{max,true,corr}$', r'${g/cm^2}$',
                                      'Xmax-true_weights', 'Weight', 'Event Weights', plot_dir, log_x=False, log_y=False)


def get_biases(y_pred, y_true, dependence, particle, dependence_str, plot_dir, cut=False):
    if not particle == 'full-comp':
        if not os.path.exists(plot_dir + 'fits/per_prim/'+ dependence_str):
            os.makedirs(plot_dir + 'fits/per_prim/' + dependence_str)
    else:
        if not os.path.exists(plot_dir + 'fits/'+ dependence_str):
            os.makedirs(plot_dir + 'fits/' + dependence_str)
    popt_prev = None
    if dependence_str == 'Zenith':
        if cut is True:
            bins = np.append(np.arange(10, 50, 5), float("inf"))
        else:
            bins = np.append(np.arange(0, 56, 5), float("inf"))
    if dependence_str == 'Energy':
        bins = np.append(np.arange(18.5, 20.05, 0.1), float("inf"))
    d = {dependence_str: dependence, 'xmax_mc': y_true, 'xmax_rec': y_pred}
    df_out = pd.DataFrame(data=d)
    df_out['bins'] = pd.cut(df_out[dependence_str], bins, right=False, include_lowest=True)
    test = df_out.groupby(df_out['bins']).count()
    mean_bin = np.array(round(df_out.groupby(df_out['bins'])[dependence_str].mean(),2))
    mu = []
    sigma = []
    mu_err = []
    sigma_err = []
    bin_list = []
    for i, idx in enumerate(test.index.values, 0):
        df_test = df_out.loc[df_out['bins'] == idx]
        reco = df_test['xmax_rec'] - df_test['xmax_mc']
        fig, ax = plt.subplots(1)
        plt.title(dependence_str + ': %s to %s' % (round(idx.left, 1), round(idx.right, 1)))
        low_bin, high_bin = np.floor(np.min(reco)) - 5, np.ceil(np.max(reco)) + 5
        bins = np.arange(low_bin, high_bin, 5)
        hist, bins, p = ax.hist(reco, bins=bins)
        bin_max = np.argmax(hist)
        bin_center = (bins[1] - bins[0]) / 2
        fit_bins = bins[:-1] + bin_center
        mean, std = weighted_avg_and_stdof(fit_bins, hist)
        xfine = np.arange(bins[bin_max] - 120, bins[bin_max] + 120, 1)
        try:
            # if not particle == 'full-comp':
            popt, cov, chi2ndf, func = fitting.fit_chi2(fitting.fit_gaus, fit_bins, hist,
                                                        p_init=[hist.max() * (2 * np.pi * std ** 2) ** 0.5, mean, std])
            ax.plot(xfine, popt[0] * (2 * np.pi * popt[2] ** 2) ** -0.5 * np.exp(
                -0.5 * (xfine - popt[1]) ** 2 * popt[2] ** -2),
                    color='C1', linestyle='-', linewidth=3)
            fitting.plot_fit_stats(ax, popt, cov, chi2ndf, posx=0.98, posy=0.91, ha='right',
                                   significant_figure=True, color='k',
                                   parnames=['norm', 'mu', 'sigma'])
            # else:
            #     popt, cov, chi2ndf, func = fitting.fit_chi2(fitting.fit_double_gaus, fit_bins, hist,
            #                                                 p_init=[35000, mean, std, 0.5, 2])
            #     ax.plot(xfine, popt[0] * (2 * np.pi * popt[2] ** 2) ** -0.5 * np.exp(
            #         -0.5 * (xfine - popt[1]) ** 2 * popt[2] ** -2),
            #             color='C1', linestyle='-', linewidth=3)
            #     fitting.plot_fit_stats(ax, popt, cov, chi2ndf, posx=0.98, posy=0.91, ha='right',
            #                            significant_figure=True, color='k',
            #                            parnames=['norm', 'mu', 'sigma', 'b', 'b'])
            ax.set_ylim(0, max(hist) + 5)
            ax.set_xlabel('$\Delta X_{max}\,\,/\,\,g/cm^{2}$')
            ax.set_ylabel('#Events')
            if not particle == 'full-comp':
                fig.savefig(plot_dir + 'fits/per_prim/{}/Xmax_reco_dist_{}-{}_{}.png'.
                            format(dependence_str, round(idx.left, 1), round(idx.right, 1), particle))
            else:
                fig.savefig(plot_dir + 'fits/{}/Xmax_reco_dist_{}-{}_{}.png'.
                            format(dependence_str, round(idx.left, 1), round(idx.right, 1), particle))
            mu.append(popt[1])
            mu_err.append(cov[1, 1] ** 0.5)
            sigma.append(popt[2])
            sigma_err.append(cov[2, 2] ** 0.5)
            if dependence_str == 'Energy' and round(idx.left, 1) == 20.0:
                bin_list.append(mean_bin[-1])
            else:
                bin_list.append(round(mean_bin[i], 2))
        except RuntimeError:
            print('Error - curve_fit failed hard')
            ax.set_ylim(0, max(hist) + 5)
            ax.set_xlabel('$\Delta X_{max}\,\,/\,\,g/cm^{2}$')
            ax.set_ylabel('#Events')
            if not particle == 'full-comp':
                fig.savefig(plot_dir + 'fits/per_prim/{}/Xmax_reco_dist_{}-{}_{}_failed.png'.
                            format(dependence_str, round(idx.left, 1), round(idx.right, 1), particle))
            else:
                fig.savefig(plot_dir + 'fits/{}/Xmax_reco_dist_{}-{}_{}_failed.png'.
                            format(dependence_str, round(idx.left, 1), round(idx.right, 1), particle))
            mu.append(0)
            mu_err.append(100)
            sigma.append(0)
            sigma_err.append(100)
            bin_list.append(round(idx.left, 1))
    np.savetxt(plot_dir + 'bias_resolution_{}_{}.txt'.format(dependence_str, particle),
               np.c_[np.asarray(bin_list), np.asarray(mu), np.asarray(mu_err), np.asarray(sigma),
                     np.asarray(sigma_err)])


def create_input_chunks(file, production_version,tracelen_og, tracelen, cutoff_start, cutoff_end, n_stations,
                  grid_stations, ssd_charge_muon_calib, wcd_charge_muon_calib, name):
    particle = os.path.basename(file).split("_")[2].split(".")[0]
    eventID = os.path.basename(file).split("_")[3].split(".")[0]
    if not os.path.isfile('../../../data/SSDSim_v3r99/input_files_9x9_' + production_version + '/' + name + '_'
                          + eventID + '_' + particle + '_9x9.npz'):
        try:
            print('Processing next chunk...')
            df_network = pd.read_pickle(file)
            df_network.reset_index(inplace=True)
            df_network.set_index(['EventID', 'Energy', 'Particle'], inplace=True)

            # General Shower Info
            Xmax = df_network['Xmax']
            Time = norm_arr_time(df_network['StationTime'][0].reshape((1, grid_stations)), n_stations,
                                 method='min').flatten()
            Energy = df_network.index.get_level_values('Energy')[0]
            Zenith = (df_network['ShowerAxis'][0])[2]
            Energy_Zenith_Time = (np.stack((np.log10(Energy) * np.ones(grid_stations), Zenith * np.ones(grid_stations),
                                            Time), axis=-1)).reshape(n_stations, n_stations, 3)

            # SSD Trace
            df_network_trace_ssd = (df_network['SSDHGTrace'][0] / ssd_charge_muon_calib).reshape((grid_stations,
                                                                                                  tracelen_og))
            HGTrace_ssd = np.array(df_network_trace_ssd[:, cutoff_start:(tracelen_og - cutoff_end)].
                                   reshape((1, n_stations, n_stations, tracelen, 1)))

            # WCD Traces
            df_network_trace_wcd1 = df_network['WCD1HGTrace'][0].reshape((grid_stations, tracelen_og))
            df_network_trace_wcd2 = df_network['WCD2HGTrace'][0].reshape((grid_stations, tracelen_og))
            df_network_trace_wcd3 = df_network['WCD3HGTrace'][0].reshape((grid_stations, tracelen_og))
            df_network_trace_wcd = np.dstack([df_network_trace_wcd1, df_network_trace_wcd2, df_network_trace_wcd3])
            df_network_trace_wcd = (np.mean(df_network_trace_wcd, axis=2) / wcd_charge_muon_calib)
            HGTrace_wcd = np.array(df_network_trace_wcd[:, cutoff_start:(tracelen_og - cutoff_end)].
                                   reshape((1, n_stations, n_stations, tracelen, 1)))
            inp_trace_ssd = HGTrace_ssd
            inp_trace_wcd = HGTrace_wcd
            inp_energy_zenith_time = Energy_Zenith_Time
            inp_xmax = Xmax
            # Save input arrays to chunk file
            np.savez('../../../data/SSDSim_v3r99/input_files_9x9_' + production_version + '/' + name + '_'
                          + eventID + '_' + particle + '_9x9.npz', inp_trace_ssd=inp_trace_ssd,
                     inp_trace_wcd=inp_trace_wcd, inp_energy_zenith_time=inp_energy_zenith_time, inp_xmax=inp_xmax)
        except Exception as e:
            print(e, file)
    else:
        print('File exists')


def create_split_data(shuffled_network_data, sample_size, test_fraction, val_fraction, input_dir):
    test_size = int(test_fraction * sample_size)
    val_size = int(val_fraction * test_size)
    for file in shuffled_network_data[:-test_size]:
        shutil.move(file, input_dir + 'train/')
    for file in shuffled_network_data[-test_size:-val_size]:
        shutil.move(file, input_dir + 'test/')
    for file in shuffled_network_data[-val_size:]:
        shutil.move(file, input_dir + 'validate/')


def find_nearest_by_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def calculate_sample_weights(inp_xmax, xmax_bounds, sample_weight_table):
    if inp_xmax < min(xmax_bounds):
        weight = 10
    if inp_xmax > max(xmax_bounds):
        weight = 10
    else:
        index = find_nearest_by_index(xmax_bounds, inp_xmax)
        weight = sample_weight_table[index]
    return weight


def calc_mean(xmax):
    sum_xmax = 0
    for i in range(len(xmax)):
        sum_xmax += xmax[i]
    return sum_xmax/len(xmax)


def calc_mean_error(xmax, mean):
    var = calc_variance(xmax, mean)
    return np.sqrt(var / len(xmax))


def calc_variance(xmax, mean):
    sum_xmax = 0
    for i in range(len(xmax)):
        sum_xmax += (xmax[i] - mean)**2
    return 1/(len(xmax)-1) * sum_xmax


def calc_std(xmax, mean):
    return np.sqrt(calc_variance(xmax, mean))


def calc_std_error(xmax, mean):
    var = calc_variance(xmax, mean)
    sum_dings = 0
    mom4 = 0
    for i in range(len(xmax)):
        mean_diff = xmax[i] - mean
        sum_dings += mean_diff**4
    mom4 = sum_dings/len(xmax)
    var_var = (1 / len(xmax)) * (mom4 - (len(xmax) - 3) / (len(xmax) - 1) * var**2)
    return np.sqrt(var_var) / np.sqrt(4 * var)


def calc_merit_factors(energy, y_pred, primaries, name):
    prim_colors = ['C0', 'C1']
    markers = ['^', 'v']
    bins_e = np.append(np.arange(18.5, 20.05, 0.1), float("inf"))
    for j, particle in enumerate(primaries):
        d = {'energy': energy[j], 'xmax_rec_' + particle: y_pred[j]}
        df_out = pd.DataFrame(data=d)
        df_out['bins'] = pd.cut(df_out['energy'], bins_e, right=False, include_lowest=True)
        test = df_out.groupby(df_out['bins']).count()
        mean_bin = np.array(round(df_out.groupby(df_out['bins'])['energy'].mean(),2))
        mu = []
        sigma = []
        mu_err = []
        sigma_err = []
        bin_list = []
        for i, idx in enumerate(test.index.values, 0):
            df_test = df_out.loc[df_out['bins'] == idx]
            # fig, ax = plt.subplots(1)
            reco = np.asarray(df_test['xmax_rec_' + particle]).flatten()
            mean = calc_mean(reco)
            mean_error = calc_mean_error(reco, mean)
            std = calc_std(reco, mean)
            std_error = calc_std_error(reco, mean)
            mu.append(mean)
            mu_err.append(mean_error)
            sigma.append(std)
            sigma_err.append(std_error)
            bin_list.append(round(mean_bin[i], 2))
        np.savetxt('gumpel_fits_{}_{}.txt'.format(particle, name),
                   np.c_[np.asarray(bin_list), np.asarray(mu), np.asarray(mu_err), np.asarray(sigma),
                         np.asarray(sigma_err)])


def get_rms(values):
    rms = np.sqrt(np.mean(values ** 2))
    # rms = np.sqrt(np.sum(values ** 2) / len(values))
    return rms

def get_mean_and_err(x, y, nbins, useSE=False):

    unique_values = np.unique(x)
    bin_edges = np.append(unique_values, unique_values[-1] + 0.00000001)

    mean_results = binned_statistic(x, y, statistic='mean', bins=bin_edges)
    std_results = binned_statistic(x, y, statistic=sem, bins=bin_edges)
    
    mean = mean_results.statistic
    std = std_results.statistic
    if useSE:
        count_results = binned_statistic(x, y, statistic='count', bins=nbins)
        std /= np.sqrt(count_results.statistic)
    
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0

    return bin_centers, mean, std, 0


def get_std_and_err(x, y, nbins, useSE=False):

    unique_values = np.unique(x)
    bin_edges = np.append(unique_values, unique_values[-1] + 0.00000001)

    mean_results = binned_statistic(x, y, statistic='std', bins=bin_edges)
    std_results = binned_statistic(x, y, statistic=get_rmse, bins=bin_edges)
    mean = mean_results.statistic
    std = std_results.statistic
    if useSE:
        count_results = binned_statistic(x, y, statistic='count', bins=nbins)
        std /= np.sqrt(count_results.statistic)
    
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0

    return bin_centers, mean, std, 0

def get_mean_and_std(x, y, nbins, useSE=False):

    mean_results = binned_statistic(x, y, statistic='mean', bins=nbins)
    std_results = binned_statistic(x, y, statistic='std', bins=nbins)
    mean = mean_results.statistic
    std = std_results.statistic
    if useSE:
        count_results = binned_statistic(x, y, statistic='count', bins=nbins)
        std /= np.sqrt(count_results.statistic)
    bin_centers = (mean_results.bin_edges[1:] + mean_results.bin_edges[:-1]) / 2.
    bin_width = bin_centers - mean_results.bin_edges[1:]

    return bin_centers, mean, std, np.abs(bin_width)


def get_rmse(values):
    # n = len(values)
    # m4 = scipy.stats.moment(values, moment=4)
    # print(m4)
    # V_err = 1 / (n) * (m4 - (n - 3) / (n - 1) * scipy.std(values)**4)
    # rmse = np.sqrt(V_err) / np.sqrt(4 * scipy.std(values)**2)
    # return rmse
    n = len(values)
    m4 = moment(values, moment=4)
    V_err = (1 / n) * (m4 - ((n - 3) / (n - 1)) * get_rms(values)**4)
    rmse = np.sqrt(np.abs(V_err)) / np.sqrt(4 * get_rms(values)**2)
    return rmse
