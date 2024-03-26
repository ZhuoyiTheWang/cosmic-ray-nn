import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import tools
import uncertainties.unumpy as unp
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as stds
from scipy.stats import pearsonr


plt.ioff()


def fit_e(x, p0, p1, p2):
    return np.sqrt((p0 * np.sqrt(x)**2) + (p1 * x)**2 + (p2)**2)

def fit_pol2(x, p0, p1, p2):
    return p0 * x**2 + p1 * x + p2


def lnA_correlation(mass_true, mass_pred, plot_dir, noise):
    diff = mass_pred - mass_true
    correlation_coefficient = np.corrcoef(mass_pred, mass_true)[0, 1]
    print(pearsonr(mass_pred, mass_true, alternative='two-sided')[0])
    binning = np.arange(-0.5, 5.05, 0.05)
    _, mean_mf, std_mf, _ = tools.get_mean_and_std(mass_true, mass_pred, 48)
    merit_factor = np.abs(mean_mf[0] - mean_mf[-6])/np.sqrt(std_mf[0]**2 + std_mf[-6]**2)
    bin_centers, mean, mean_err, bin_width = tools.get_mean_and_err(mass_true, diff, 48)
    bin_centers, std, std_err, bin_width = tools.get_std_and_err(mass_true, diff, 48)


    fig = plt.figure(figsize=(20, 9))
    # Plot 1
    ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
    counts, xedges, yedges, im = ax1.hist2d(mass_true, mass_pred, bins=[binning, binning], cmin=1,
                                               norm=mpl.colors.LogNorm())
    # im = ax1.hexbin(mass_true, mass_pred, gridsize=len(binning), bins='log',
    #                                  extent=[-0.5, 4.5, -0.5, 4.5], mincnt=1, cmap='viridis', zorder=0)
    ax1.set_xlabel(r"$\ln{A}_\mathrm{MC}$")
    ax1.set_ylabel(r"$\ln{A}_\mathrm{DNN}$")
    ax1.set_xlim(-0.5, 4.5)
    ax1.set_ylim(-0.5, 4.5)
    ax1.set_box_aspect(1)
    stat_box = r"$\mu$ = %.3f" % np.mean(diff) + "\n" + "$\sigma$ = %.3f" % np.std(
        diff) + "\n" + "$r$ = %.3f" % correlation_coefficient
    ax1.text(0.55, 0.2, stat_box, verticalalignment="top", horizontalalignment="left",
                transform=ax1.transAxes, backgroundcolor="w")
    ax1.plot([np.min(mass_true), np.max(mass_true)], [np.min(mass_true), np.max(mass_true)], color="C1")
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('# Events')

    # Plot 2
    ax2 = plt.subplot2grid((2, 2), (0, 1), colspan=1)
    ax2.errorbar(bin_centers, mean, xerr=bin_width, yerr=mean_err, color='grey', fmt=".", linewidth=2.5, zorder=1)
    ax2.errorbar(bin_centers[0], mean[0], xerr=bin_width[0], yerr=mean_err[0], color='red', fmt=".", linewidth=2.5,
                    label="proton", zorder=1)
    ax2.errorbar(bin_centers[-1], mean[-1], xerr=bin_width[-1], yerr=mean_err[-1], color='blue', fmt=".", linewidth=2.5,
                    label="Iron", zorder=1)
    ax2.legend(loc='upper right')
    ax2.set_ylabel(r"$\langle\Delta\ln{A}\rangle_\mathrm{(DNN-MC)}$")

    # Plot 3
    ax3 = plt.subplot2grid((2, 2), (1, 1), colspan=1)
    ax3.errorbar(bin_centers, std, xerr=bin_width, yerr=std_err, color='grey', fmt=".", linewidth=2.5, zorder=1)
    ax3.errorbar(bin_centers[0], std[0], xerr=bin_width[0], yerr=std_err[0], color='red', fmt=".", linewidth=2.5,
                    label="proton", zorder=1)
    ax3.errorbar(bin_centers[-1], std[-1], xerr=bin_width[-1], yerr=std_err[-1], color='blue', fmt=".", linewidth=2.5,
                    label="Iron", zorder=1)
    stat_box = r"$\mathrm{MeritFactor}=\frac{\langle\ln{A}\rangle_\mathrm{Fe} - \langle\ln{A}\rangle_\mathrm{p}}{\sigma(\ln{A})_\mathrm{Fe}\oplus\sigma(\ln{A})_\mathrm{p}}$ = %.3f" % round(merit_factor, 2)
    ax3.text(0.05, 0.3, stat_box, verticalalignment="top", horizontalalignment="left",
                transform=ax3.transAxes, backgroundcolor="w")
    ax3.legend(loc='upper right')
    ax3.set_xlabel(r"$\ln{A}_\mathrm{MC}$")
    ax3.set_ylabel(r"$\sigma\left(\Delta\ln{A}\right)_\mathrm{(DNN-MC)}$")

    fig.tight_layout()
    print('done')
    fig.savefig(plot_dir + f"/dnn_results{noise}")


    # ax.plot([np.min(binning) - 5, np.max(binning) + 5], [np.min(binning) - 5, np.max(binning) + 5], color='grey',
    #         linestyle='--')
    # ax.plot(np.NaN, np.NaN, 's', markersize=3, color='C0', label=r'$C_\mathrm{corr}$ = ' + str(corr_coeff))
    # popt, cov, chi2ndf, func = fitting.fit_chi2(fitting.fitfunc, y_true, y_pred)
    # textstr = fitting.plot_fit_stats(ax, popt, cov, chi2ndf, posx=0.98, posy=0.83, ha='right', significant_figure=True,
    #                                  color='k', parnames=[r'p_{0}', r'p_{1}'], plot_box=False)
    # xfine = binning
    # ax.plot(xfine, popt[0] * xfine + popt[1], color='C1', label=textstr)


def mf_plot(mfs, noise, plot_dir):

    fig, ax = plt.subplots(figsize=(10, 8))
    # Plot 1
    ax.plot(noise, mfs, marker='o',linestyle='None')
    #FIT 1
    popt, cov, chi2ndf, func = fit_chi2(fit_e, noise, mfs)
    print(popt)
    # textstr = plot_fit_stats(ax, popt, cov, chi2ndf, posx=0.98, posy=0.83, ha='right', significant_figure=True,
    #                                  color='k', parnames=[r'p_{0}', r'p_{1}'], plot_box=False)
    # xfine = binning
    x_new = np.linspace(0, 500, 500)
    ax.plot(x_new, fit_e(x_new, popt[0], popt[1], popt[2]), '--', color='grey')#, label=textstr)
    ax.set_xlabel(r"Noise")
    ax.set_ylabel(r"MeritFactor")
    # ax.set_xlim(-0.5, 4.5)
    # ax.set_ylim(-0.5, 4.5)
    ax.set_box_aspect(1)
    # stat_box = r"$\mu$ = %.3f" % np.mean(diff) + "\n" + "$\sigma$ = %.3f" % np.std(
    #     diff) + "\n" + "$r$ = %.3f" % correlation_coefficient
    # ax1.text(0.55, 0.2, stat_box, verticalalignment="top", horizontalalignment="left",
    #             transform=ax1.transAxes, backgroundcolor="w")
    # ax1.plot([np.min(mass_true), np.max(mass_true)], [np.min(mass_true), np.max(mass_true)], color="C1")
    # divider = make_axes_locatable(ax1)
    # cax = divider.append_axes("right", size="3%", pad=0.05)
    # cbar = fig.colorbar(im, cax=cax)
    # cbar.set_label('# Events')

    fig.tight_layout()
    print('done')
    fig.savefig(plot_dir + f"/mfs")

folder_name = '(1) General Model [l: 0.1551, vl: 0.1337]'

results = np.load(f'/home/zwang/cosmic-ray-nn/testing/{folder_name}/model_predictions.npz')

truth = results['actual']
prediction = results['predicted']

lnA_correlation(truth, prediction, f'/home/zwang/cosmic-ray-nn/testing/{folder_name}/', None)