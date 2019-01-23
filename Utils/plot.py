import numpy as np
import matplotlib
import matplotlib.patches as mpatches
matplotlib.use('Agg')
import matplotlib.pylab as pylab
import sys
import os
import logging
from matplotlib import colors as mcolors
from scipy.signal import savgol_filter


def plotOneLine(lossVector, xAxisLst, xtitleStr, ytitleStr, titleStr, outputDir, filename):
    '''

    '''

    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] - %(message)s')
    ld = logging.debug
    fig = pylab.figure()
    pylab.xlabel(xtitleStr)
    pylab.ylabel(ytitleStr)
    pylab.suptitle(titleStr)
    pylab.plot(xAxisLst, lossVector)
    axes = pylab.gca()
    axes.set_xlim([min(xAxisLst), max(xAxisLst)])
    # axes.set_ylim([min(lossVector),max(lossVector)+0.1])
    if max(lossVector) / min(lossVector) > 1000:
        axes.set_yscale('log')
    try:
        # Save to file both to the required format and to png
        pylab.savefig(os.path.join(outputDir, filename), dpi=300)
        ld("Saved plot to:" + os.path.join(outputDir, filename))

        filename_list = filename.split(".")
        filename_list[-1] = ".eps"
        filename_eps = "".join(filename_list)
        pylab.savefig(os.path.join(outputDir, filename_eps), format='eps', dpi=1000)
        ld("Saved plot to:" + os.path.join(outputDir, filename_eps))
    except:
        pass
    pylab.close(fig)


def plotTwoLines(trainAcc, validAcc, xAxisLst, xtitleStr, ytitleStr, titleStr, outputDir, filename, isAnnotated=False,
                 trainStr='Train', validStr='Validation', customPointAnnotation=None):
    '''

    : param customPointAnnotation - set to a particular x value to mark both of the plot-lines
                                    with an arrow at this x value.
                                    can be a list - if multiple annotated points are deisred

    Legend "loc" arguments:
    'best'         : 0, (only implemented for axes legends)
    'upper right'  : 1,
    'upper left'   : 2, <--- we chose it
    'lower left'   : 3,
    'lower right'  : 4,
    'right'        : 5,
    'center left'  : 6,
    'center right' : 7,
    'lower center' : 8,
    'upper center' : 9,
    'center'       : 10,
    '''
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] - %(message)s')
    ld = logging.debug
    fig = pylab.figure()
    pylab.xlabel(xtitleStr)
    pylab.ylabel(ytitleStr)
    pylab.suptitle(titleStr)
    plot_train, = pylab.plot(xAxisLst, trainAcc, label=trainStr, linestyle="--")
    plot_valid, = pylab.plot(xAxisLst, validAcc, label=validStr, linestyle="-")
    pylab.legend([plot_train, plot_valid], [trainStr, validStr], loc=0)
    if (isAnnotated):
        maxAccId = np.argmax(validAcc)
        maxAccVal = validAcc[maxAccId]
        minAcc_total = min(min(validAcc), min(trainAcc))
        maxAcc_total = max(max(validAcc), max(trainAcc))
        str = validStr + " %.1f%%" % maxAccVal + " at epoch " + str((maxAccId + 1))
        pylab.plot([maxAccId + 1], [maxAccVal], 'o')
        pylab.annotate(str, xy=(maxAccId + 1, maxAccVal),
                       xytext=(maxAccId + 1 - len(xAxisLst) * 0.25, maxAccVal - (maxAcc_total - minAcc_total) / 10),
                       arrowprops=dict(facecolor='orange', shrink=0.05))
    if not(customPointAnnotation is None):
        if type(customPointAnnotation)!=list:
            customPointAnnotation = [customPointAnnotation]
        for pt in customPointAnnotation:
            if (pt in xAxisLst):
                pylab.plot([pt, pt], [trainAcc[pt], validAcc[pt]], 'o', markersize=3)
                pylab.annotate("", xy=(pt, trainAcc[pt]),
                               arrowprops=dict(facecolor='orange', shrink=0.05))
                pylab.annotate("", xy=(pt, validAcc[pt]),
                               arrowprops=dict(facecolor='orange', shrink=0.05))
            else:
                ld("Warning: cannot annotate plot at point {}, since it's out of the range of X-axis".format(pt))
    # Save to file both to the required format and to png
    pylab.savefig(os.path.join(outputDir, filename), dpi=300)
    ld("Saved plot to:" + os.path.join(outputDir, filename))

    filename_list = filename.split(".")
    filename_list[-1] = ".eps"
    filename_eps = "".join(filename_list)
    pylab.savefig(os.path.join(outputDir, filename_eps), format='eps', dpi=1000)
    ld("Saved plot to:" + os.path.join(outputDir, filename_eps))
    pylab.close(fig)

def plotManyLines(common_x_lst, y_lst_of_lists, legends_lst, xtitleStr, ytitleStr, titleStr, outputDir, filename):
    '''


    for python 3.5
    :param common_x_lst:
    :param y_lst_of_lists:
    :param legends_lst:
    :param xtitleStr:
    :param ytitleStr:
    :param titleStr:
    :param outputDir:
    :param filename:
    :return:
    '''
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] - %(message)s')
    ld = logging.debug
    fig = pylab.figure()
    pylab.xlabel(xtitleStr)
    pylab.ylabel(ytitleStr)
    pylab.suptitle(titleStr)
    colors = list(mcolors.CSS4_COLORS.values())[14:14+len(legends_lst)]


    axes = [pylab.plot(common_x_lst, y_lst, color=cllr) for cllr,y_lst in zip(colors,y_lst_of_lists)]
    pylab.legend(handles = [mpatches.Patch(color =cllr, label=legenda) for cllr, legenda in zip(colors,legends_lst)])

    # Save to file both to the required format and to png
    pylab.savefig(os.path.join(outputDir, filename), dpi=300)
    ld("Saved plot to:" + os.path.join(outputDir, filename))

    filename_list = filename.split(".")
    filename_list[-1] = ".eps"
    filename_eps = "".join(filename_list)
    pylab.savefig(os.path.join(outputDir, filename_eps), format='eps', dpi=1000)
    ld("Saved plot to:" + os.path.join(outputDir, filename_eps))

def plotListOfPlots(x_lst_of_lists, y_lst_of_lists, legends_lst, xtitleStr, ytitleStr, titleStr, outputDir, filename, lpf=None, colorLst=None, fontsize=None, showGrid=False):
    '''
        :param lpf: the window-length of averaging. This is used for smoothing, and implemented by the Savitzky-Golay
                    filter.
    '''
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] - %(message)s')
    ld = logging.debug

    fig = pylab.figure()
    pylab.xlabel(xtitleStr, fontsize=fontsize)
    pylab.ylabel(ytitleStr, fontsize=fontsize)

    if not titleStr is None and titleStr != "":
        pylab.suptitle(titleStr)
    #colors = list(mcolors.CSS4_COLORS.values())
    if colorLst is None:
        colorLst = ['red', 'orange', 'green', 'blue', 'darkblue', 'purple', 'black']

    if lpf != None:
        y_lst_of_lists_new =[savgol_filter(np.array(data), lpf, 1) for data in y_lst_of_lists]
        y_lst_of_lists = y_lst_of_lists_new

    if not fontsize is None:
        #matplotlib.rcParams.update({'font.size': fontsize})
        ##matplotlib.rc('xtick', labelsize=fontsize)
        #matplotlib.rc('ytick', labelsize=fontsize)
        # pylab.rc('font', size=fontsize)  # controls default text sizes
        # pylab.rc('axes', titlesize=fontsize)  # fontsize of the axes title
        # pylab.rc('axes', labelsize=fontsize)  # fontsize of the x and y labels
        # pylab.rc('xtick', labelsize=fontsize)  # fontsize of the tick labels
        # pylab.rc('ytick', labelsize=fontsize)  # fontsize of the tick labels
        # pylab.rc('legend', fontsize=fontsize)  # legend fontsize
        # pylab.rc('figure', titlesize=fontsize)  # fontsize of the figure title
        pass

    axes = [pylab.plot(x_lst, y_lst, color=cllr, linewidth=3.0) for cllr, x_lst,y_lst in zip(colorLst, x_lst_of_lists, y_lst_of_lists)]
    if not legends_lst is None and len(legends_lst) == len(x_lst_of_lists):
        pylab.legend(handles = [mpatches.Patch(color =cllr, label=legenda) for cllr, legenda in zip(colorLst, legends_lst)])
    #pylab.legend(axes, legends_lst, loc=0)    # old legend generation
    if showGrid:
        pylab.gca().grid(True, which='both', linestyle=':')
    # Save to file both to the required format and to png
    pylab.savefig(os.path.join(outputDir, filename), dpi=300)
    ld("Saved plot to:" + os.path.join(outputDir, filename))

    filename_list = filename.split(".")
    filename_list[-1] = ".eps"
    filename_eps = "".join(filename_list)
    pylab.savefig(os.path.join(outputDir, filename_eps), format='eps', dpi=1000)
    ld("Saved plot to:" + os.path.join(outputDir, filename_eps))
    pylab.close(fig)

def plotListOfScatters(x_lst_of_lists, y_lst_of_lists, legends_lst, xtitleStr, ytitleStr, titleStr, outputDir, filename):

    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] - %(message)s')
    ld = logging.debug

    fig = pylab.figure()
    pylab.xlabel(xtitleStr)
    pylab.ylabel(ytitleStr)
    pylab.suptitle(titleStr)
    axes = [pylab.scatter(x_lst, y_lst, s=2) for x_lst,y_lst in zip(x_lst_of_lists, y_lst_of_lists)]
    pylab.legend(axes, legends_lst, loc=0)

    # Save to file both to the required format and to png
    pylab.savefig(os.path.join(outputDir, filename), dpi=300)
    ld("Saved plot to:" + os.path.join(outputDir, filename))

    filename_list = filename.split(".")
    filename_list[-1] = ".eps"
    filename_eps = "".join(filename_list)
    pylab.savefig(os.path.join(outputDir, filename_eps), format='eps', dpi=1000)
    ld("Saved plot to:" + os.path.join(outputDir, filename_eps))
    pylab.close(fig)

def plotBetweens(y_list, xAxisLst, xtitleStr, ytitleStr, titleStr, legendStrLst, outputDir, filename):
    '''
    The y_list constains multiple y-values, all consistent with the xAxisLst ticks
    The plot will consist of len(Y_list) plots, with a different coloor fill between two consecutive plots.

    Pre-conditions:
    1) The first array (or list) in y_list is assumed to be the lowest one in its height.
    2) The first array (or list) in y_list will be filled downwards till the x-axis.

    y
    |                                /
    |       color=white             /  color2
    |                              /
    |                       ______/___________ y_list[1]
    |                      /
    |_____________________/     color1
    |                   ______________________ y_list[0]
    |  color1   _______/
    |          /              color0
    |_________/
    ------------------------------------------->x

    Legend "loc" arguments:
    'best'         : 0, (only implemented for axes legends)
    'upper right'  : 1,
    'upper left'   : 2, <--- we chose it
    'lower left'   : 3,
    'lower right'  : 4,
    'right'        : 5,
    'center left'  : 6,
    'center right' : 7,
    'lower center' : 8,
    'upper center' : 9,
    'center'       : 10,
    '''
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] - %(message)s')
    ld = logging.debug

    fig = pylab.figure(figsize=(8,2))
    pylab.xlabel(xtitleStr)
    pylab.ylabel(ytitleStr)
    pylab.suptitle(titleStr)

    # Add the artificial "zero-plot" to be the reference plot
    zerolst = len(xAxisLst) * [0]
    y_arr_with_zerolst = [zerolst] + y_list
    axis_lst = [pylab.plot(xAxisLst, zerolst, "k-")]
    legend_handles = []

    # Color list for assining different colors to different fills
    color_lst = ['tab:black', 'tab:orange', 'tab:red', 'tab:brown', 'tab:purple', 'tab:pink', 'tab:olive', 'tab:green',
                 'tab:gray', 'tab:cyan', 'tab:blue']
    num_plots = len(y_arr_with_zerolst)
    num_colors = len(color_lst)

    # Add the plot-lines and fill the space between each to consequitive ones
    for i in range(1, num_plots):
        fill_color = color_lst[i % min(num_plots, num_colors)]
        axis_lst.append(pylab.plot(xAxisLst, y_arr_with_zerolst[i], "k-"))
        pylab.fill_between(xAxisLst, y_arr_with_zerolst[i], y_arr_with_zerolst[i - 1], facecolor=fill_color,
                           interpolate=True)
        legend_handles.append(mpatches.Patch(color=fill_color, label=legendStrLst[i - 1]))

    # Legends

    pylab.legend(handles=legend_handles[::-1])

    # Save to file both to the required format and to png
    pylab.savefig(os.path.join(outputDir, filename), dpi=300)
    ld("Saved plot to:" + os.path.join(outputDir, filename))

    filename_list = filename.split(".")
    filename_list[-1] = ".eps"
    filename_eps = "".join(filename_list)
    pylab.savefig(os.path.join(outputDir, filename_eps), format='eps', dpi=1000)
    ld("Saved plot to:" + os.path.join(outputDir, filename_eps))
    pylab.close(fig)

    #subsample the bar data
def subsample(x,y,factor):
    new_x = []
    new_y = []
    for i in range(0,len(x),factor):
        new_x.append(x[i])
        new_y.append(0)
        for j in range(factor):
            new_y[-1] += y[i+j]

    return new_x, new_y

def plotListOfPlots_and_Bars(x_lst_of_lists, y_lst_of_lists, legends_lst, xtitleStr, ytitleStr, titleStr, outputDir, filename, n_lines, lpf=None, colorLst=None, fontsize=None, showGrid=False):
    '''
    : param nlines indicates how many lines there are in the list. after this amount of lines - the bars begin
        :param lpf: the window-length of averaging. This is used for smoothing, and implemented by the Savitzky-Golay
                    filter.
    '''
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] - %(message)s')
    ld = logging.debug

    fig = pylab.figure()
    pylab.xlabel(xtitleStr, fontsize=fontsize)
    pylab.ylabel(ytitleStr, fontsize=fontsize)

    if not titleStr is None and titleStr != "":
        pylab.suptitle(titleStr)
    #colors = list(mcolors.CSS4_COLORS.values())
    if colorLst is None:
        colorLst = ['red', 'orange', 'green', 'blue', 'darkblue', 'purple', 'black']

    if lpf != None:
        y_lst_of_lists_new =[savgol_filter(np.array(data), lpf, 1) for data in y_lst_of_lists]
        y_lst_of_lists = y_lst_of_lists_new

    if not fontsize is None:
        #matplotlib.rcParams.update({'font.size': fontsize})
        ##matplotlib.rc('xtick', labelsize=fontsize)
        #matplotlib.rc('ytick', labelsize=fontsize)
        # pylab.rc('font', size=fontsize)  # controls default text sizes
        # pylab.rc('axes', titlesize=fontsize)  # fontsize of the axes title
        # pylab.rc('axes', labelsize=fontsize)  # fontsize of the x and y labels
        # pylab.rc('xtick', labelsize=fontsize)  # fontsize of the tick labels
        # pylab.rc('ytick', labelsize=fontsize)  # fontsize of the tick labels
        # pylab.rc('legend', fontsize=fontsize)  # legend fontsize
        # pylab.rc('figure', titlesize=fontsize)  # fontsize of the figure title
        pass

    axes = []
    n_bars = len(y_lst_of_lists) - n_lines
    subsample_bar_factor = 10 # by how much is the bar plot's resolution is lower than the reliable accuracy's
    n_bins = len(y_lst_of_lists[0]) / subsample_bar_factor
    ld("Using " + str(n_bins) + " bins and " + str(n_bars) + "bars")

    bar_width = (max(x_lst_of_lists[0])-min(x_lst_of_lists[0])) / float(n_bars*n_bins)
    bin_offset = bar_width*(n_bars/2)


    for cllr, x_lst, y_lst,i in zip(colorLst, x_lst_of_lists, y_lst_of_lists, range(n_bars+n_lines)):
        if i < n_lines:
            axes.append(pylab.plot(x_lst, y_lst, color=cllr, linewidth=3.0))
        else:
            new_x, new_y = subsample(x_lst, y_lst, subsample_bar_factor)
            axes.append(pylab.bar(x_lst-bin_offset+(i-n_lines)*bar_width, y_lst, color=cllr, align='center', width=bar_width))


    if not legends_lst is None and len(legends_lst) == len(x_lst_of_lists):
        pylab.legend(handles = [mpatches.Patch(color =cllr, label=legenda) for cllr, legenda in zip(colorLst, legends_lst)])
    #pylab.legend(axes, legends_lst, loc=0)    # old legend generation
    if showGrid:
        pylab.gca().grid(True, which='both', linestyle=':')
    # Save to file both to the required format and to png
    pylab.savefig(os.path.join(outputDir, filename), dpi=300)
    ld("Saved plot to:" + os.path.join(outputDir, filename))

    filename_list = filename.split(".")
    filename_list[-1] = ".eps"
    filename_eps = "".join(filename_list)
    pylab.savefig(os.path.join(outputDir, filename_eps), format='eps', dpi=1000)
    ld("Saved plot to:" + os.path.join(outputDir, filename_eps))
    pylab.close(fig)