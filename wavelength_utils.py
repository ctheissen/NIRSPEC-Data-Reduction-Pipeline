# -*- coding: utf-8 -*-
"""
Created on Wed Jul 03 14:35:13 2013

@author: jholt
with minimal modifications and declassification by rcohen
"""
import os
import numpy as np
from numpy import fft
import scipy
from scipy import optimize
import pylab as pl
#import itertools
#from scipy.signal import argrelextrema
#import statsmodels.api as smapi
from statsmodels.formula.api import ols
import config
import matplotlib.pyplot as plt
import nirspec_constants

import logging
from scipy.signal._peak_finding import argrelextrema, find_peaks_cwt

logger = logging.getLogger('obj')


MAX_SHIFT         = 50
DISP_LOWER_LIMIT  = 0.95
DISP_UPPER_LIMIT  = 1.05
#OH_FILE_NAME = './ir_ohlines.dat'
GAUSSIAN_VARIANCE = 0.2


def synthesize_sky(oh_wavelengths, oh_intensities, wavelength_scale_calc, eta=None, arc=None):
    
    x = np.array(oh_wavelengths)
    y = np.array(oh_intensities)

    limit = 0.01
    if arc is not None: 
        limit = 0
    
    if y.any():
        synthesized_sky = y[0]
    else:
        raise ValueError('no reference OH/Etalon/Arc line data')

    for i in np.arange(0, x.size):
        if y[i] > limit:
            if eta is not None:
                g = y[i] * np.exp(-(wavelength_scale_calc - x[i]) ** 2.0 / (2.0 * (2*GAUSSIAN_VARIANCE) ** 2.0))
            else:
                g = y[i] * np.exp(-(wavelength_scale_calc - x[i]) ** 2.0 / (2.0 * GAUSSIAN_VARIANCE ** 2.0))
            
            synthesized_sky = synthesized_sky + g

    return synthesized_sky
        


def line_id(order, oh_wavelengths, oh_intensities, eta=None, arc=None):
    """
    Given real sky spectrum, synthesized sky spectrum, estimated wavelength scale based on 
    evaluation of grating equation, and accepted OH emission line wavelengths and relative
    intensities, match observed sky lines with accepted OH lines.
    
    First find_wavelength_shift() is called to determine the offset between the estimated 
    wavelength scale and the actual wavelength scale using the method of cross correlation.
    
    identify() is called to make the actual associate between observed and accepted lines.
    identify() has not yet been recoded, detailed explanation of algorithm forthcoming.  
    
    This function returns (column, wavelength) pairs as a list of tuples.
    
    """
    # find wavelength shift

    #plt.figure(111)
    #plt.scatter(oh_wavelengths, oh_intensities, c='r', alpha=0.5)

    if eta is not None:
        order.waveShift = find_wavelength_shift(order.etalonSpec, order.synthesizedSkySpec,
                order.flatOrder.gratingEqWaveScale, eta=eta)

    elif arc is not None:
        order.waveShift = find_wavelength_shift(order.arclampSpec, order.synthesizedSkySpec,
                order.flatOrder.gratingEqWaveScale, arc=arc)

    else:
        order.waveShift = find_wavelength_shift(order.skySpec['A'], order.synthesizedSkySpec,
                order.flatOrder.gratingEqWaveScale)
 
    if abs(order.waveShift) > MAX_SHIFT:
        logger.warning('measured wavelength shift of {:.1f} pixels exceeds threshold of {:.0f}'.format(
                order.waveShift, MAX_SHIFT))
        return None
        
    #logger.info('wavelength scale shift = ' + str(round(order.waveShift, 3)) + ' pixels')   
    logger.info('wavelength scale shift = ' + str(round(order.waveShift, 3)) + ' angstroms')   
    wavelength_scale_shifted = order.flatOrder.gratingEqWaveScale + order.waveShift   

    ### TESTING
    '''
    print(wavelength_scale_shifted)
    print(oh_wavelengths)
    print(oh_intensities)
    print(order.arclampSpec)
    plt.figure(1011)
    #plt.plot(wavelength_scale_shifted, order.etalonSpec, c='b', alpha=0.5)
    plt.plot(wavelength_scale_shifted, order.arclampSpec, c='b', alpha=0.5)
    plt.scatter(oh_wavelengths, oh_intensities, c='r', alpha=0.5)
    plt.show()
    '''
    ### TESTING
    

    # match sky lines
    if eta is not None:
        id_tuple = identify(
                order.etalonSpec, wavelength_scale_shifted, oh_wavelengths, oh_intensities, eta=eta)
    elif arc is not None:
        id_tuple = identify(
                order.arclampSpec, wavelength_scale_shifted, oh_wavelengths, oh_intensities, arc=arc)
    else:
        id_tuple = identify(
                order.skySpec['A'], wavelength_scale_shifted, oh_wavelengths, oh_intensities)

    if id_tuple is not None:
        matchesdx, matchesohx, matchesidx = id_tuple
    else:
        matchesdx, matchesohx, matchesidx = np.array([]), np.array([]), np.array([])

    ### TESTING
    '''
    plt.figure(1012)
    #plt.plot(wavelength_scale_shifted, order.etalonSpec, c='b', alpha=0.5)
    plt.plot(wavelength_scale_shifted, order.arclampSpec, c='b', alpha=0.5)
    plt.scatter(oh_wavelengths, oh_intensities, c='r', alpha=0.5)
    plt.show()
    sys.exit()
    '''
    ### TESTING
    
#     if order.isPair:
#         id_tuple = identify(
#                 order.skySpec['B'], wavelength_scale_shifted, oh_wavelengths, oh_intensities)
#         if id_tuple is not None:
#             matchesdx = np.concatenate((matchesdx, id_tuple[0]))
#             matchesohx = np.concatenate((matchesohx, id_tuple[1]))
#             matchesidx = np.concatenate((matchesidx, id_tuple[2]))
    
    if len(matchesdx) < 1:
        return None
          
    p = np.polyfit(matchesohx, matchesdx, deg=1)
    # polyfit() returns highest power polynomial coefficient first, so p[0] is slope.

    #plt.figure(1078)
    #plt.scatter(matchesohx, matchesdx)
    #plt.plot(np.linspace(np.min(matchesohx), np.max(matchesohx)), z00(np.linspace(np.min(matchesohx), np.max(matchesohx))), 'r--')
    #plt.show()
    disp = p[0]
    
    if DISP_UPPER_LIMIT > abs(disp) > DISP_LOWER_LIMIT:
        logger.info('slope of accepted vs measured wavelengths = ' + str(round(disp, 3)))

        # XXX Add in the RMS of the fit to the logger
        z00 = np.poly1d(p)
        residual = np.abs(  z00(matchesohx) - matchesdx)            
        var      = ((residual ** 2).sum()) / (len(matchesdx) - 1)
        sigma    = np.sqrt(var)
        logger.info('RMS fit residual (Angstroms) = ' + str(round(sigma, 3)))

        # return column, wavelength pairs as a list of tuples
        lines = []
        for i in range(len(matchesdx)):
            lines.append((matchesidx[i], matchesohx[i]))
        return lines 
    else:
        logger.warning('per-order wavelength fit slope out of limits, not using sky/etalon/arc lines from this order')
        return None

    #plt.figure(1078)
    #plt.scatter(matchesohx, matchesdx)
    #plt.plot(np.linspace(np.min(matchesohx), np.max(matchesohx)), z00(np.linspace(np.min(matchesohx), np.max(matchesohx))), 'r--')
    #plt.show()


# def read_oh_file():
#     """
#     """
#     fn = './ir_ohlines.dat'
#     try:
#         lines = open(OH_FILE_NAME).readlines()
#     except:
#         logger.critical('failed to open OH emission line file: ' + fn)
#         return None, None
# 
#     oh_wavelengths = []
#     oh_intensities = []
# 
#     print('*** reading oh file')
#     for line in lines:
#         tokens = line.split(" ")
#         if float(tokens[1]) > 0:
#             oh_wavelengths.append(float(tokens[0]))
#             oh_intensities.append(float(tokens[1]))
# 
#     return oh_wavelengths, oh_intensities


def get_oh_lines():
    """
    Reads OH line wavelengths and intensities from data file.
    Once the data is read, it is saved in static-like variables 
    so the file is read only once.  
    
    Returns two parallel arrays, one containing wavelengths and
    the other containing the corresponding intensities, as a tuple.
    
    Raises IOError if data file cannot be opened or read
    """
    
    try:
        return get_oh_lines.oh_wavelengths, get_oh_lines.oh_intensities
    
    except AttributeError:
        
        if config.params['oh_envar_override']:
            oh_filename = config.params['oh_filename']
        else:
            oh_filename = os.environ.get(config.params['oh_envar_name'])
            if oh_filename is None:
                oh_filename = config.params['oh_filename']
             
        logger.info('reading OH line data from ' + oh_filename)
        
        try:
            lines = open(oh_filename).readlines()
        except:
            logger.error('failed to open OH emission line file: ' + oh_filename)
            raise
    
        get_oh_lines.oh_wavelengths = []
        get_oh_lines.oh_intensities = []
    
        for line in lines:
            tokens = line.split(" ")
            if float(tokens[1]) > 0:
                get_oh_lines.oh_wavelengths.append(float(tokens[0]))
                get_oh_lines.oh_intensities.append(float(tokens[1]))
    
        return get_oh_lines.oh_wavelengths, get_oh_lines.oh_intensities



def get_etalon_lines():
    """
    Reads Etalon line wavelengths and intensities from data file.
    Once the data is read, it is saved in static-like variables 
    so the file is read only once.  
    
    Returns two parallel arrays, one containing wavelengths and
    the other containing the corresponding intensities, as a tuple.
    
    Raises IOError if data file cannot be opened or read
    """
    
    try:
        return get_etalon_lines.etalon_wavelengths, get_etalon_lines.etalon_intensities
    
    except AttributeError:
        
        if config.params['etalon_envar_override']:
            etalon_filename = config.params['etalon_filename']
        else:
            etalon_filename = os.environ.get(config.params['etalon_envar_name'])
            if etalon_filename is None:
                etalon_filename = config.params['etalon_filename']
             
        logger.info('reading etalon line data from ' + etalon_filename)
        
        try:
            lines = open(etalon_filename).readlines()
        except:
            logger.error('failed to open etalon emission line file: ' + etalon_filename)
            raise
    
        get_etalon_lines.etalon_wavelengths = []
        get_etalon_lines.etalon_intensities = []
    
        for line in lines:
            tokens = line.split(" ")
            if float(tokens[1]) > 0:
                get_etalon_lines.etalon_wavelengths.append(float(tokens[0]))
                get_etalon_lines.etalon_intensities.append(float(tokens[1]))
    
        return get_etalon_lines.etalon_wavelengths, get_etalon_lines.etalon_intensities

 
def get_arclamp_lines():
    """
    Reads Etalon line wavelengths and intensities from data file.
    Once the data is read, it is saved in static-like variables 
    so the file is read only once.  
    
    Returns two parallel arrays, one containing wavelengths and
    the other containing the corresponding intensities, as a tuple.
    
    Raises IOError if data file cannot be opened or read
    """
    
    try:
        return get_arclamp_lines.arclamp_wavelengths, get_arclamp_lines.arclamp_intensities
    
    except AttributeError:
        
        if config.params['arclamp_envar_override']:
            arclamp_filename = config.params['arclamp_filename']
        else:
            arclamp_filename = os.environ.get(config.params['arclamp_envar_name'])
            if arclamp_filename is None:
                arclamp_filename = config.params['arclamp_filename']
             
        logger.info('reading arc lamp line data from ' + arclamp_filename)
        
        try:
            lines = open(arclamp_filename).readlines()
        except:
            logger.error('failed to open arc lamp emission line file: ' + arclamp_filename)
            raise
    
        get_arclamp_lines.arclamp_wavelengths = []
        get_arclamp_lines.arclamp_intensities = []
    
        for line in lines:
            tokens = line.split(" ")
            if float(tokens[1]) > 0:
                get_arclamp_lines.arclamp_wavelengths.append(float(tokens[0]))
                get_arclamp_lines.arclamp_intensities.append(float(tokens[1]))
    
        return get_arclamp_lines.arclamp_wavelengths, get_arclamp_lines.arclamp_intensities


def gen_synthesized_sky(oh_wavelengths, oh_intensities, wavelength_scale_calc, eta=None, arc=None):
    """
    """
    x = np.array(oh_wavelengths)
    y = np.array(oh_intensities)
    if y.any():
        all_g = y[0]
    else:
        logger.warning('no OH/etalon/arc lines in wavelength range')
        return None

    for i in np.arange(0, x.size):
        if y[i] > 0.01:
            if eta is not None:
                g = y[i] * np.exp(-(wavelength_scale_calc - x[i]) ** 2.0 / (2.0 * (2*GAUSSIAN_VARIANCE) ** 2.0))
            elif arc is not None:
                g = y[i] * np.exp(-(wavelength_scale_calc - x[i]) ** 2.0 / (2.0 * (2*GAUSSIAN_VARIANCE) ** 2.0))
            else:
                g = y[i] * np.exp(-(wavelength_scale_calc - x[i]) ** 2.0 / (2.0 * GAUSSIAN_VARIANCE ** 2.0))
            all_g = all_g + g

    return all_g



def find_wavelength_shift(sky, gauss_sky, grating_eq_wave_scale, eta=None, arc=None):
    
    if len(sky) > 0:
        if eta is not None:
            sky_n = sky
        elif arc is not None:
            sky_n = sky
        else:
            sky_n = sky - sky.mean()
    else:
        logger.error('sky/etalon/arc spectrum length is zero')
        return None
 
    ohg = np.array([grating_eq_wave_scale, gauss_sky])  # ohg is a synthetic spectrum of gaussians

    ### TESTING
    #plt.figure(102)
    #plt.plot(sky, alpha=0.5, label='data')
    #plt.plot(gauss_sky, alpha=0.5, label='synthesized')
    #plt.legend()
    #plt.show()
    ### TESTING
 
    if not ohg.any():
        logger.error('no synthetic sky/etalon/arc lines in wavelength range')
        return None
 
    xcorrshift = max_corr(ohg[1], sky_n)
    if xcorrshift is None:
        logger.error('failed to find wavelength shift')
        return None
 
    delta_x = (ohg[0][-1] - ohg[0][0]) / float(ohg[0].size)

    if eta is not None or arc is not None:

        sky_n -= np.amin(sky_n)
        coeffs = np.polyfit(np.arange(len(sky_n)), sky_n, 9)
        s_fit  = np.polyval(coeffs, np.arange(len(sky_n)))
        sky_n  = sky_n - s_fit + 0.9
        if arc is not None:
            sky_n += -0.9 # remove the baseline
            ohg[1] += -np.median(ohg[1]) # set the floor to zero
            ohg[1] = ohg[1] / np.max(ohg[1]) * np.max(sky_n) # scale to the sky
        
        ### TESTING
        #plt.figure(111, figsize=(12,6))
        #plt.plot(ohg[0], ohg[1], c='m', alpha=0.5, label='synthesized')
        #plt.plot(ohg[0], sky_n, c='b', alpha=0.5, label='data')
        #plt.legend()
        #plt.show(block=True)
        #sys.exit()
        ### TESTING
        
        import scipy.interpolate as sci
        length = len(ohg[1])
        # Calculate the cross correlation
        drvs = np.arange(-40, 40, 0.1)
        cc   = np.zeros(len(drvs))
        tw   = np.arange(len(ohg[1]))
        w    = np.arange(len(sky_n))
        tf   = ohg[1]
        f    = sky_n
        for i, rv in enumerate(drvs):
            fi = sci.interp1d(tw+rv, tf, fill_value=0.9, bounds_error=False)
            # Shifted template evaluated at location of spectrum
            cc[i] = np.sum(f * fi(w))
        maxind   = np.argmax(cc)
        pixShift = drvs[maxind]
        
        ### TESTING
        '''
        plt.figure(191, figsize=(12,6))
        plt.plot(ohg[1], c='b', alpha=0.5, label='calib')
        plt.plot(sky_n, c='r', alpha=0.5, label='before')
        plt.plot(np.arange(len(sky_n))+pixShift, sky_n, c='m', alpha=0.5, label='pos')
        plt.plot(np.arange(len(sky_n))-pixShift, sky_n, c='c', alpha=0.5, label='neg')
        plt.legend()
        plt.show(block=True)
        #sys.exit()
        '''
        ### TESTING
        
        return -pixShift * delta_x

    return xcorrshift * delta_x

SKY_LINE_MIN = 10
SKY_OVERLAP_THRESHOLD = 0.6
SKY_THRESHOLD = 3.0

   

def identify(sky, wavelength_scale_shifted, oh_wavelengths, oh_intensities, eta=None, arc=None):
    """
    """
    debug    = False
    theory_x = np.array(wavelength_scale_shifted)
         
    # if theory_x.min() < 20500:
    dy       = sky
    '''
    import pylab as pl
    pl.figure(facecolor="white")
    pl.cla()
    pl.xlabel('Wavelength (Angstroms)')
    pl.ylabel('Relative Intensity')
    pl.plot(wavelength_scale_shifted, dy, 'b-', alpha=0.5, label='data')
    pl.scatter(oh_wavelengths, oh_intensities, color='r', alpha=0.5, label='lines')
    pl.legend()
    pl.show()
    '''
    # ## Open, narrow down, and clean up line list ###
    # only look at the part of sky line list that is around the theory locations
    if eta is not None or arc is not None:
        #print(theory_x[-1] + 100, theory_x[0] - 100)
        #print(np.where( (oh_wavelengths < theory_x[-1] + 100) & (oh_wavelengths > theory_x[0] - 100)))
        #print(np.where( (oh_wavelengths < theory_x[-1] + 100) & (oh_wavelengths > theory_x[0] - 100))[0])
        #print(np.array(oh_wavelengths)[np.where( (oh_wavelengths < theory_x[-1] + 100) & (oh_wavelengths > theory_x[0] - 100) )[0]])
        ohxsized = np.array(oh_wavelengths)[np.where( (oh_wavelengths < theory_x[-1] + 100) & (oh_wavelengths > theory_x[0] - 100) )[0]]
        ohysized = np.array(oh_intensities)[np.where( (oh_wavelengths < theory_x[-1] + 100) & (oh_wavelengths > theory_x[0] - 100) )[0]]

    else:
        locohx = np.intersect1d(np.where(oh_wavelengths < theory_x[-1] + 20)[0],
                                np.where(oh_wavelengths > theory_x[0] - 20)[0])
 
        ohxsized = np.array(oh_wavelengths[locohx[0]:locohx[-1]])
        ohysized = np.array(oh_intensities[locohx[0]:locohx[-1]])
 
    # ignore small lines in sky line list
    if eta is not None:
        bigohy = ohysized[np.where(ohysized > 0.5)]
        bigohx = ohxsized[np.where(ohysized > 0.5)]
    elif arc is not None:
        bigohy = ohysized[np.where(ohysized > 0)]
        bigohx = ohxsized[np.where(ohysized > 0)]
    else:
        bigohy = ohysized[np.where(ohysized > SKY_LINE_MIN)]
        bigohx = ohxsized[np.where(ohysized > SKY_LINE_MIN)]

    '''
    import pylab as pl
    pl.figure(facecolor="white")
    pl.cla()
    pl.xlabel('Wavelength (Angstroms)')
    pl.ylabel('Relative Intensity')
    pl.plot(wavelength_scale_shifted, dy, 'b-', alpha=0.5, label='data')
    pl.scatter(oh_wavelengths, oh_intensities, color='r', alpha=0.5, label='lines')
    pl.scatter(bigohx, bigohy, color='m', marker='x', alpha=0.5, label='lines (in range)')
    pl.legend()
    pl.show()
    '''
    # bigohx, y are lines from the data file, in the expected wavelength range
    # with intensity greater than  SKY_LINE_MIN

    deletelist = []
 
    # remove 'overlapping' or too close sky lines
    if bigohy.any():
        for i in range(1, len(bigohy)):
            if abs(bigohx[i] - bigohx[i - 1]) < SKY_OVERLAP_THRESHOLD:
                deletelist.append(i)
        bigohy = np.delete(bigohy, deletelist, None)
        bigohx = np.delete(bigohx, deletelist, None)
    else:
        # there were no sky lines in the table that match theoretical wavelength range
        logger.warning('could not find known sky/etalon/arc lines in expected wavelength range')
        return []
 
        
    # ## Open, narrow down, clean up sky line list
    # look for relative maxes in dy (real sky line peak values)
#     if argrelextrema(dy, np.greater)[0].any():
#         relx = theory_x[argrelextrema(dy, np.greater)[0]]
#         rely = dy[argrelextrema(dy, np.greater)[0]]
#         idx1 = argrelextrema(dy, np.greater)[0]
#  
#         # bixdx is the locations (in x) of any sky peak maximums greater than threshold sig
#         bigdx = relx[np.where(rely > SKY_THRESHOLD * rely.mean())]
#         # bigidx are the intensities of the sky peak maximums
#         bigidx = idx1[np.where(rely > SKY_THRESHOLD * rely.mean())]
#         
#     else:
#         # couldn't find any relative maxes in sky
#         logger.info('could not find any relative maxes in sky lines')
#         return []

    if config.params['lla'] == 1:
        bigidx = find_peaks_1(dy)
    else:
        bigidx = find_peaks_2(dy, eta=eta, arc=arc)

    bigdx = theory_x[bigidx]
    logger.debug('n sky/etalon/arc line peaks = {}'.format(len(bigidx)))
    
    deletelist = []

    ### XXX TESTING AREA
    #print(bigidx)
    #plt.plot(dy)
    #plt.scatter(bigidx)
    ### XXX TESTING AREA
 
    # remove 'overlapping' real sky line values
    for i in range(1, len(bigdx)):
        if abs(bigdx[i] - bigdx[i - 1]) < SKY_OVERLAP_THRESHOLD:
            deletelist.append(i)
 
    bigdx  = np.delete(bigdx, deletelist, None)
    bigidx = np.delete(bigidx, deletelist, None)
 
    # The two arrays to match are bigdx and bigohx
 
    matchesohx = []
    matchesohy = []
    matchesdx  = []
    matchesidx = []

    #plt.plot(sky)
 
    if bigohx.any() and bigdx.any():
 
        # ## First look for doublets ###
 
        # search for shifted doublet
        bigdx2  = bigdx
        bigohx2 = bigohx
        bigohy2 = bigohy
        bigidx2 = bigidx
        # happened is a counter of doublets matched, removed from bigdx, bigohx and added to match list
        happened = 0
 
        for i in range(0, len(bigdx) - 1):
            if eta is not None or arc is not None: 
                waveLimit = 10
            else:
                waveLimit = 2
            if bigdx[i + 1] - bigdx[i] < waveLimit:
                if debug:
                    print(bigdx[i], ' and ', bigdx[i + 1], 'possible doublet')
 
                # locx is the section of bigohx  within +/- 4 angstrom of the bigdx possible doublet
                locx = np.intersect1d(np.where(bigohx2 > (bigdx[i] - 4))[0],
                            np.where(bigohx2 < (bigdx[i + 1] + 4))[0])
                if debug:
                    print('locx=', locx)
 
                if len(locx) > 1:  # there has to be more than two lines within the range for matched doublet
 
                    if len(locx) > 2:
                        # found more than 2 possible sky lines to match with doublet
                        # 'happened' is how many doubles already removed from bigohy
 
                        # yslice is the part of bigohy that corresponds to bigohx (with a 'happened' fix for
                        # removed doublet fails)
                        yslice = np.array(bigohy2[locx[0] - 2 * happened:locx[-1] - 2 * happened + 1])
 
                        locxfix = np.zeros(2, dtype=np.int)
 
                        if len(yslice) > 0:
                            # location of the peak in the yslice
                            locxfix[0] = np.argmax(yslice)  #
                        else:
                            continue
 
                        yslice = np.delete(yslice, locxfix[0])  # remove the max from yslice
 
                        locxfix[1] = np.argmax(
                            yslice)  # find the location of the next max; second biggest in original slice
 
                        if locxfix[1] <= locxfix[0]:
                            locxfix[1] += 1  # if lowest peak then highest peak
 
                        locx = locx[locxfix]
                        locx.sort()
                        if debug:
                            print('locx=', locx)
 
                    ohslice = np.array(bigohx2[locx[0] - 2 * happened:locx[1] - 2 * happened + 1])
                    if debug:
                        print('ohslice=', ohslice, ' are in the same location as', bigdx[i], bigdx[i + 1])
                    if len(ohslice) > 1:
                        for j in range(0, 1):
                            if debug:
                                print('j=', j)
                            if ((ohslice[j + 1] - ohslice[j]) < 2 and abs(ohslice[j] - bigdx2[i - 2 * happened]) < 6
                                    and abs(ohslice[j + 1] - bigdx2[i + 1 - 2 * happened]) < 6):
                                if debug:
                                    print(ohslice[j], ohslice[j + 1], 'is same doublet as ', \
                                        bigdx2[i - 2 * happened], bigdx2[i + 1 - 2 * happened])
 
                                matchesohx.append(ohslice[j])
                                matchesohx.append(ohslice[j + 1])
                                matchesohy.append(bigohy2[locx[0] - 2 * happened + j])
                                matchesohy.append(bigohy2[locx[0] - 2 * happened + j + 1])
                                matchesdx.append(bigdx2[i - 2 * happened])
                                matchesdx.append(bigdx2[i - 2 * happened + 1])
                                matchesidx.append(bigidx[i - 2 * happened])
                                matchesidx.append(bigidx[i - 2 * happened + 1])
 
                                if debug:
                                    print('removing bigdxs', bigdx2[i - 2 * happened], bigdx2[i - 2 * happened + 1])
                                    print('removing bigoxs', bigohx2[locx[0] - 2 * happened + j], \
                                        bigohx2[locx[0] - 2 * happened + j + 1])
                                    print('before dx2=', bigdx2)
                                    print('before oh2=', bigohx2)
 
                                bigdx2  = np.delete(bigdx2, i - 2 * happened)
                                bigdx2  = np.delete(bigdx2, i - 2 * happened)  # this removes the "i+1"
                                bigohx2 = np.delete(bigohx2, locx[0] - 2 * happened + j)
                                bigohx2 = np.delete(bigohx2, locx[0] - 2 * happened + j)  # this removes the "j+1"
                                bigohy2 = np.delete(bigohy2, locx[0] - 2 * happened + j)
                                bigohy2 = np.delete(bigohy2, locx[0] - 2 * happened + j)  # this removes the "j+1"
                                bigidx2 = np.delete(bigidx2, i - 2 * happened)
                                bigidx2 = np.delete(bigidx2, i - 2 * happened)
 
                                happened += 1


 
        bigdx  = bigdx2
        bigidx = bigidx2
        bigohx = bigohx2
        bigohy = bigohy2
 
        if debug:
            print('bigohx=', bigohx)
            print('bigohy=', bigohy)
            print('bigdx=', bigdx)
            print('bigidx=', bigidx)
 
        for j in range(0, len(bigohx)):
            minimum = min((abs(bigohx[j] - i), i) for i in bigdx)
 
            if eta is not None or arc is not None:
                Minimum = 10.0
            else: 
                Minimum = 4.0

            if (minimum[0]) < Minimum:
                matchesohx.append(bigohx[j])
                matchesohy.append(bigohy[j])
                matchesdx.append(minimum[1])
 
                for idx in range(len(bigidx)):
                    if bigdx[idx] == minimum[1]:
                        matchesidx.append(bigidx[idx])
 
    if len(matchesdx) > 2:
        if debug:
            print('matchesdx:', matchesdx)
            print('matchesohx:', matchesohx)
            print('matchesohy:', matchesohy)
            print('matchesidx:', matchesidx)
        # check for duplicates
        matchesdx2 = matchesdx[:]
        matchesohx2 = matchesohx[:]
        matchesohy2 = matchesohy[:]
        matchesidx2 = matchesidx[:]
        happened = 0
        for j in range(0, len(matchesdx) - 1):
 
            if abs(matchesdx[j + 1] - matchesdx[j]) < 0.01:
                if debug:
                    print('duplicate=', matchesdx[j + 1], matchesdx[j])
                # find which oh does it actually belongs to 
 
                if min(matchesdx[j + 1] - matchesohx[j + 1], matchesdx[j + 1] - matchesohx[j]) == 0:
                    matchesdx2.pop(j + 1 - happened)
                    matchesidx2.pop(j + 1 - happened)
                    matchesohx2.pop(j + 1 - happened)
                    matchesohy2.pop(j + 1 - happened)
                else:
                    matchesdx2.pop(j - happened)
                    matchesidx2.pop(j - happened)
                    matchesohx2.pop(j - happened)
                    matchesohy2.pop(j - happened)
 
                happened += 1
 
        matchesdx = np.array(matchesdx2)
        matchesohx = np.array(matchesohx2)
        matchesohy = np.array(matchesohy2)
        matchesidx = np.array(matchesidx2)
 
        matchesdx.sort()
        matchesidx.sort()
 
        oh_sort_indices = matchesohx.argsort()
        matchesohy = matchesohy[oh_sort_indices]
        # matchesohx.sort()
        matchesohx = matchesohx[oh_sort_indices]
        
#         print('***** ' + str(len(matchesohx)) + ' matches found'))
#         print('matchesdx: ' + str(matchesdx))
#         print('matchesohx: ' + str(matchesohx))
#         print('*****')
#         raw_input('waiting')
        
#         return [matchesdx, matchesohx, matchesohy, bigohx, bigohy, 1, matchesidx]
        return [matchesdx, matchesohx, matchesidx]


# def sanity_check(orig_pix_x, order_number_array, matched_sky_line):
#     """
#     tries to fit a line to each ID/OH value, throws out bad fits
#     :param orig_pix_x:
#     :param order_number_array:
#     :param matched_sky_line:
#     :return:
#     """
# 
#     i = 0
#     while i < len(order_number_array):
#         f1, residuals1, rank1, singular_values1, rcond1 = np.polyfit(orig_pix_x[i], matched_sky_line[i], 1, full=True)
#         f2, residuals2, rank2, singular_values2, rcond2 = np.polyfit(orig_pix_x[i], matched_sky_line[i], 2, full=True)
# 
#         if float(residuals2) > 500:
#             print('order number ', order_number_array[i][0], ' is a bad fit')
#             orig_pix_x.pop(i)
#             order_number_array.pop(i)
#             matched_sky_line.pop(i)
#         i += 1
# 
#     return orig_pix_x, order_number_array, matched_sky_linex



def max_corr(a, b):
    """ 
    Find the maximum of the cross-correlation - includes upsampling
    """
    if len(a.shape) > 1:
        logger.error('array dimension greater than 1')
        return None
    
    length = len(a)
    if not length % 2 == 0:
        logger.error('cannot cross correlate an odd length array')
        return None

    if not a.shape == b.shape:
        logger.error('cannot cross correlate arrays of different shapes')
        return None

    # Start by finding the coarse discretised arg_max
    coarse_max = np.argmax(np.correlate(a, b, mode='full')) - length + 1

    omega = np.zeros(length)
    omega[0:length // 2] = (2 * np.pi * np.arange(length // 2)) / length
    omega[length // 2 + 1:] = (2 * np.pi *
                              (np.arange(length // 2 + 1, length) - length)) / length

    fft_a = fft.fft(a)

    def correlate_point(tau):
        rotate_vec = np.exp(1j * tau * omega)
        rotate_vec[length // 2] = np.cos(np.pi * tau)

        return np.sum((fft.ifft(fft_a * rotate_vec)).real * b)

    start_arg, end_arg = (float(coarse_max) - 1, float(coarse_max) + 1)

    max_arg = optimize.fminbound(lambda tau: -correlate_point(tau),
                                 start_arg, end_arg)
    # print('coarse_max=',coarse_max,' max_arg=',max_arg)

    return max_arg



def __residual(params, f, x, y):
    """ 
    Define fit function; 
    Return residual error. 
    """
    
#     print('params: ' + str(params))
#     print('f: ' + str(f))
#     print('x: ' + str(x))
#     print('y: ' + str(y))
#     raw_input('waiting')
    
    a0, a1, a2, a3, a4, a5 = params
    return np.ravel(a0 + a1 * x + a2 * x ** 2 + a3 * y + a4 * x * y + a5 * (x ** 2) * y - f)

LOWER_LEN_POINTS = 10.0
#SIGMA_MAX = 0.3
SIGMA_MAX = 1.0
MIN_N_LINES = 6



def twodfit(dataX, dataY, dataZ):
#def twodfit(cols, orders, wavelengths):
    """

    :param dataX: First independent variable
    :param dataY: Second independent variable
    :param dataZ: Dependent variable
    :param logger: logger instance
    :param lower_len_points: lowest
    :param sigma_max:
    :return:
    """
    
#     print(('datax: ' + str(dataX))
#     print(('datay: ' + str(dataY))
#     print(('dataz: ' + str(dataZ))
#     raw_input('waiting')

    if len(dataX) < MIN_N_LINES:
        logger.warning('not enough lines to compute wavelength solution, n = {}, min = {}'.format(
                str(len(dataX)), str(MIN_N_LINES)))
        return None, None, None, None

    testing = False

#     newoh = 9999
    newoh = None

    dataXX, dataYY = scipy.meshgrid(dataX, dataY)

    # # guess initial values for parameters
    p0 = [137.9, 0., 1. / 36, 750000, 10, 0.]
    bad_points = []
    # print(__residual(p0, dataZZ, dataXX, dataYY)
    sigma = 100.

    # This call is just to set up the plots
    p1, pcov, infodict, errmsg, success = optimize.leastsq(__residual, x0=p0, args=(dataZ, dataX, dataY),
                                                                 full_output=1)


    k = 0

    if testing:
        pl.figure(14, figsize=(15, 8))
        pl.clf()
        ax1 = pl.subplot(211)
        pl.title("2d fitting")
        ax2 = pl.subplot(212)

        points = ['r.', 'g.', 'c.', 'k.', 'm.', 'b.', 'y.',
                  'rx', 'gx', 'cx', 'kx', 'mx', 'bx', 'yx',
                  'r*', 'g*', 'c*', 'k*', 'm*', 'b*', 'y*', 'r.', 'g.', 'c.', 'k.', 'm.', 'b.', 'y.',
                  'rx', 'gx', 'cx', 'kx', 'mx', 'bx', 'yx',
                  'r*', 'g*', 'c*', 'k*', 'm*', 'b*', 'y*']

        lines = ['r-.', 'g.-', 'c-.', 'k-.', 'm-.', 'b-.', 'y-.',
                 'r--', 'g--', 'c--', 'k--', 'm--', 'b--', 'y--',
                 'r-', 'g-', 'c-', 'k-', 'm-', 'b-', 'y-', 'r-.', 'g.-', 'c-.', 'k-.', 'm-.', 'b-.', 'y-.',
                 'r--', 'g--', 'c--', 'k--', 'm--', 'b--', 'y--',
                 'r-', 'g-', 'c-', 'k-', 'm-', 'b-', 'y-']

        ax2.plot(__residual(p1, dataZ, dataX, dataY),
                         points[k], __residual(p1, dataZ, dataX, dataY), lines[k],
                         label=str(k) + ' fit')

    dataZ_new = np.copy(dataZ)
    dataY_new = np.copy(dataY)
    dataX_new = np.copy(dataX)

    residual   = __residual(p1, dataZ, dataX, dataY)
    x_res      = np.arange(len(residual))
    regression = ols("data ~ x_res", data=dict(data=residual, x=x_res)).fit()
    test       = regression.outlier_test()
    outliers   = ((x_res[i], residual[i]) for i,t in enumerate(test.iloc[:, 2]) if t < 0.9)
    #print('outliers=',list(outliers))
    x          = list(outliers)
    #logger.info('residual outliers='+str(x))
    xhap=0

    for j in range(len(x)):
        logger.debug('deleting outlier from 2d fit, col={:d}, order={:d}, wave accepted = {:.1f}'.format(
                int(dataX_new[x[j][0]-xhap]),
                int(1/dataY_new[x[j][0]-xhap]),
                dataZ_new[x[j][0]-xhap]))
        dataZ_new = np.delete(dataZ_new, x[j][0]-xhap)
        dataX_new = np.delete(dataX_new, x[j][0]-xhap)
        dataY_new = np.delete(dataY_new, x[j][0]-xhap)

        xhap+=1


    happened=0
    while len(dataZ_new) > LOWER_LEN_POINTS - 1. and sigma > SIGMA_MAX:

        p1, pcov, infodict, errmsg, success = optimize.leastsq(__residual, x0=p0, args=(dataZ_new, dataX_new, dataY_new),
                                                                     full_output=1)


        residual = __residual(p1, dataZ_new, dataX_new, dataY_new)
        x_res = np.arange(len(residual))
        
        regression = ols("data ~ x_res", data=dict(data=residual, x=x_res)).fit()
        test = regression.outlier_test()
        outliers = ((x_res[i], residual[i]) for i,t in enumerate(test.iloc[:, 2]) if t < 0.9)
        #print('outliers=',list(outliers))
        x=list(outliers)
        xhap=0

        for j in range(len(x)):
            logger.debug('deleting outlier from 2d fit, col={:d}, order={:d}, accepted wavelength={:.1f}'.format(
                    int(dataX_new[x[j][0]-xhap]),
                    int(1/dataY_new[x[j][0]-xhap]),
                    dataZ_new[x[j][0]-xhap]))
            dataZ_new = np.delete(dataZ_new, x[j][0]-xhap)
            dataX_new = np.delete(dataX_new, x[j][0]-xhap)
            dataY_new = np.delete(dataY_new, x[j][0]-xhap)

            xhap+=1

        dataX_new_forplot = np.copy(dataX_new)
        dataY_new_forplot = np.copy(dataY_new)
        dataZ_new_forplot = np.copy(dataZ_new)

        newoh = np.ravel(p1[0] + p1[1] * dataX_new + p1[2] * dataX_new ** 2 + p1[3] * dataY_new + p1[4] * dataX_new * dataY_new + p1[5] * (
            dataX_new ** 2) * dataY_new)
        
        # pl.figure('fit')
        # pl.cla()
        # pl.plot(newoh, dataZ_new)
        # pl.show()

        if (len(dataZ_new) > len(p0)) and pcov is not None:

            if testing:
                
                ax1.plot(newoh, dataZ_new, points[k], newoh, dataZ_new, lines[k], label='fit')
                ax2.plot(__residual(p1, dataZ_new_forplot, dataX_new_forplot, dataY_new_forplot),
                         points[k], __residual(p1, dataZ_new_forplot, dataX_new_forplot, dataY_new_forplot), lines[k],
                         label=str(k) + ' fit')

            residual = np.abs(__residual(p1, dataZ_new, dataX_new, dataY_new))
                        
            var = ((residual ** 2).sum()) / (len(dataZ_new) - 1)

            sigma = np.sqrt(var)


        # new arrays made for second pass
        if sigma > SIGMA_MAX and len(dataZ_new) > LOWER_LEN_POINTS:

            bad_points.append(residual.argmax())
            logger.debug('2d fit rms residual={:.3f}'.format(sigma))

            if testing:
                dataZ_new_forplot[residual.argmax()-happened ] = dataZ_new_forplot[residual.argmin()]
                dataX_new_forplot[residual.argmax()-happened ] = dataX_new_forplot[residual.argmin()]
                dataY_new_forplot[residual.argmax()-happened ] = dataY_new_forplot[residual.argmin()]


            #logger.info('removing residual val='+str(residual[residual.argmax()])+' index = '+str(residual.argmax()))
            #logger.info(' removing datax='+str(dataX_new[residual.argmax()])+' datay='+str(1/dataY_new[residual.argmax()])+' dataz=',dataZ_new[residual.argmax()])

            try:
                dataZ_new = np.delete(dataZ_new, residual.argmax())
                dataX_new = np.delete(dataX_new, residual.argmax())
                dataY_new = np.delete(dataY_new, residual.argmax())
                happened +=1
            except:
                logger.error('failed to delete outlier, continuing')
#                 exit()


        elif sigma > SIGMA_MAX:
            logger.info('minimum number of lines reached ({}), sigma={:.3f}'.format(
                    LOWER_LEN_POINTS, sigma))
            break

        k += 1
    

    if sigma >= 100.0:
        logger.critical('wavelength calibration failed, fit residual too large')
        return None, None, None, None

    logger.success('wavelength calibration rms fit residual = ' + str(round(sigma, 3)))


    logger.info('wavelength eq coefficients: ' + ', '.join('{:.2E}'.format(k) for k in p1))
    
    dataZZ = p1[0] + p1[1] * dataXX + p1[2] * dataXX ** 2 + p1[3] * dataYY + p1[4] * dataXX * dataYY + p1[5] * (
        dataXX ** 2) * dataYY

    if testing:
        ax2.plot(__residual(p1, dataZ_new_forplot, dataX_new_forplot, dataY_new_forplot), 'ko-')
        ax2.legend()
        pl.show()
    #return p1, newoh, dataZZ
    return p1, newoh, dataZ_new, sigma



def applySolution(order_object, p1):
    if len(p1) > 0:

        if nirspec_constants: endPix = 2048
        else: endPix = 1024
        
        newdx = np.arange(endPix)
        newy = 1. / order_object.sciorder.order_num
        newoh = np.ravel(
            p1[0] + p1[1] * newdx + p1[2] * newdx ** 2 + p1[3] * newy + p1[4] * newdx * newy + p1[5] * (
                newdx ** 2) * newy)

        # setattr(reduced_order_object.sciorder, "dx_2dfit", astro_math.conv_ang_to_mu(newoh))
        # reduced_order_object.sciorder.dx = astro_math.conv_ang_to_mu(reduced_order_object.sciorder.dx)
        # reduced_order_object.lineobj.matchesohx = astro_math.conv_ang_to_mu(reduced_order_object.lineobj.matchesohx)
        # reduced_order_object.lineobj.bigohx = astro_math.conv_ang_to_mu(reduced_order_object.lineobj.bigohx)
        return newoh
    else:
        return []
   


def find_peaks_1(s):
    """
    """
    logger.info('using calibration line location algorithm 1')
    peaks_i = argrelextrema(s, np.greater)[0]
    peaks_y = s[peaks_i]
    return(peaks_i[np.where(peaks_y > 3.0 * peaks_y.mean())])
    


def find_peaks_2(s, eta=None, arc=None):
    """ 
    """
    logger.info('using calibration line location algorithm 2')
    s         = s[:-20] # Crop out the back portion
    s        -= np.amin(s)
    coeffs    = np.polyfit(np.arange(len(s)), s, 10)
    s_fit     = np.polyval(coeffs, np.arange(len(s)))
    sp        = s - s_fit
    if eta is not None or arc is not None:
        sp[0:20] = 0. # Set the first few pixels to zero since we don't want false lines
    sp       -= np.amin(sp)
    sp       -= np.median(sp)
    #print(np.median(sp), 1.1*np.median(sp))
    if eta is not None or arc is not None:
        #peaks_i   = argrelextrema(sp, np.greater, order=2)
        #print(peaks_i)
        peaks_i   = find_peaks_cwt(sp, [4,5,6,7,8])#, min_length=20)#, min_snr=3)
        #print('PEAKS', peaks_i)
    else:
        peaks_i   = argrelextrema(sp, np.greater)
    peaks_y   = sp[peaks_i]
    #for p,z in zip(peaks_i[0], peaks_y):
    #    print(p, z, z/peaks_y.mean())
#     big_peaks = peaks_i[0][np.where(peaks_y > 1.7 * peaks_y.mean())]
    if eta is not None:
        #big_peaks = peaks_i[np.where(peaks_y > 1.1 * sp.mean())]
        #big_peaks = peaks_i[np.where(peaks_y > 1.1 * np.median(sp))]
        big_peaks = peaks_i[np.where(peaks_y >= 0.15)] # XXX We could change this to use percentiles to estimate the cutoff
        #print(big_peaks)
    elif arc is not None:
        s         = s - np.median(s)
        peaks_y   = s[peaks_i]
        big_peaks = peaks_i[np.where(peaks_y >= 0.005)] # XXX We could change this to use percentiles to estimate the cutoff
    else:
        big_peaks = peaks_i[0][np.where(peaks_y > 1.4 * peaks_y.mean())]
    
    ### TESTING
    '''
    import pylab as pl
    pl.figure(figsize=(15, 5))
    pl.cla()
    if eta is not None or arc is not None: offsetPlot = 0
    else: offsetPlot = 20
    pl.plot(s + offsetPlot, 'r-')
    pl.plot(sp, 'k-')
    pl.plot(s_fit + offsetPlot, 'm-', alpha=0.5)
    for peak in peaks_i:
        if peak == peaks_i[0]: pl.axvline(peak, color='r', ls=':', lw=1.5, label='peaks')
        else: pl.axvline(peak, color='r', ls=':', lw=1.5)
    #pl.scatter(big_peaks, sp[big_peaks], color='C0')
    for peak in big_peaks:
        if peak == big_peaks[0]: pl.axvline(peak, color='b', ls='--', lw=0.9, label='big peaks')
        else: pl.axvline(peak, color='b', ls='--', lw=0.9)
    pl.legend()
    pl.show()
    #sys.exit()
    '''
    ### TESTING
    
    
    return(big_peaks)
    
