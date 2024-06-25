import matplotlib.pyplot as plt
import matplotlib.colors as colors

import matplotlib.dates as mdates
import numpy as np


def draw_simple_plots(orig_tensor, corr_tensor, true_tensor, channel=2,
                      input_loss=-1, test_loss=-1, era_metric=None, date=None, mask=None):
    fig, axs = plt.subplots(4, 1, figsize=(10, 28))
    vmin = min(orig_tensor[channel].min(), corr_tensor[channel].min(), true_tensor[channel].min())
    vmax = max(orig_tensor[channel].max(), corr_tensor[channel].max(), true_tensor[channel].max())
    im = axs[0].imshow(orig_tensor[channel].cpu().numpy(), interpolation='none', vmin=vmin, vmax=vmax,
                       extent=[0, 280, 0, 210],)
    axs[1].imshow((corr_tensor[channel]*mask[channel]).cpu(), interpolation='none', vmin=vmin, vmax=vmax,
                  extent=[0, 280, 0, 210],)
    axs[2].imshow(true_tensor[channel].cpu(), interpolation='none',
                  extent=[0, 280, 0, 210], vmin=vmin, vmax=vmax)
    imc = axs[3].imshow((corr_tensor[channel]*mask[channel]-orig_tensor[channel]*mask[channel]).cpu(),
                        interpolation='none', extent=[0, 280, 0, 210],)
    axs[0].set_xlabel('Original Glorys data')
    axs[0].text(50, 28, f'loss={round(input_loss, 3)}')
    axs[1].set_xlabel('Corrected Glorys data')
    axs[1].text(50, 28, f'loss={round(test_loss, 3)}')
    axs[2].set_xlabel('Glorys reanalysis')
    axs[3].set_xlabel('Correction')
    for i in range(4):
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].xaxis.set_label_coords(.5, -.01)
    if era_metric:
        axs[3].text(0, -85, f'era metric={round(era_metric, 3)}')
    if date:
        axs[3].text(0, 5, f'{date}')
    fig.colorbar(im, ax=axs[:3], orientation='vertical', fraction=0.1, aspect=21)
    fig.colorbar(imc, ax=axs[3], orientation='vertical', fraction=0.1, aspect=6)

    return fig, axs


def draw_so_means(data, names, date, timestamp="D", locators='sparse'):
    fig, ax = plt.subplots(1, dpi=200, layout='constrained')
    # names = ['era5', 'wrf_hindcast', 'wrf_operative']
    for dtype, name in zip(data, names):
        ax.plot_date(date, dtype, label=name, linestyle='solid', marker='None')
    # Major ticks every half year, minor ticks every month,
    if locators == 'sparse':
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
    elif locators == "frequent":
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        # ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1] + list(range(0, 31, 5))))
        ax.set_xlim(date[0], date[-1])

    ax.grid(True)
    ax.set_ylabel(r'so')
    ax.set_title('Mean Volumetric Salinity in Barents & Kara Seas',
                 fontsize='medium')
    # Text in the x-axis will be displayed in 'YYYY-mm' format.
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b-%d'))
    # Rotates and right-aligns the x labels so they don't crowd each other.
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')
    plt.legend(fontsize="7")
    return fig