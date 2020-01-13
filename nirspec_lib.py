import numpy as np
from scipy.signal._peak_finding import argrelextrema

import logging
import config
import nirspec_constants

import tracer
import matplotlib.pyplot as plt

logger = logging.getLogger('obj')
      
        
def calc_noise_img(obj, flat, integration_time):
    """
    flat is expected to be normalized and both obj and flat are expected to be rectified
    """
    
    G  = 5.8  # e-/ADU    
    RN = 23.0 # e-/pixel
    DC = 0.8  # e-/second/pixel
    if nirspec_constants.upgrade: # updated from website (9/25/2018)
        G  = 2.85 # e-/ADU    
        RN = 10.  # e-/pixel
        DC = 0.67 # e-/second/pixel
    
    # calculate photon noise
    noise = obj      # This is already in ADU!
    #noise = obj / G  # This is in ADU (only works if obj is in electrons!)
    #noise = obj * G  # This is in electrons
    
    # add read noise
    noise += np.square(RN / G) # This is in ADU
    #noise += np.square(RN) # This is in electrons
    
    # add dark current noise
    noise += DC * integration_time / np.square(G) # This is in ADU
    #noise += (DC / G) * integration_time # This is in electrons
    
    # divide by normalized flat squared
    noise /= np.square(flat)
    
    return noise


ORDER_EDGE_SEARCH_WIDTH = 10
ORDER_EDGE_BG_WIDTH     = 30
ORDER_EDGE_JUMP_THRESH  = 1.9
ORDER_EDGE_JUMP_LIMIT   = 200



def trace_order_edge(data, start):

    ORDER_EDGE_SEARCH_WIDTH = 10
    ORDER_EDGE_BG_WIDTH     = 30
    ORDER_EDGE_JUMP_THRESH  = 1.9
    ORDER_EDGE_JUMP_LIMIT   = 200
        
    trace, nJumps =  tracer.trace_edge(
            data, start, ORDER_EDGE_SEARCH_WIDTH, ORDER_EDGE_BG_WIDTH, ORDER_EDGE_JUMP_THRESH)
    
    if trace is None:
        logger.warning('trace failed')
        return None
    
#     if slit_name.endswith('24'):
#         x = np.arange(len(trace))
#         coeffs = np.polyfit(x, trace, 1)
#         y_fit = np.polyval(coeffs, x)
#         res1 = trace - y_fit
# #         stdev1 = np.std(res1)
# 
#         if abs(coeffs[0]) < 1e-2:
#             logger.warning('long slit edge criteria not met')
#             return None
#         else:
#             return trace
    
    if nJumps > ORDER_EDGE_JUMP_LIMIT:
        
            logger.debug('order edge trace jump limit exceeded')
            logger.debug('reducing search width to {:.1f}'.format(ORDER_EDGE_SEARCH_WIDTH / 2))
            trace, nJumps =  tracer.trace_edge(
            data, start, ORDER_EDGE_SEARCH_WIDTH / 2, ORDER_EDGE_BG_WIDTH, ORDER_EDGE_JUMP_THRESH)
            
            if nJumps > ORDER_EDGE_JUMP_LIMIT:
                ORDER_EDGE_SEARCH_WIDTH = 3
        
                logger.debug('order edge trace jump limit exceeded')
                logger.debug('reducing search width to {:.1f}'.format(ORDER_EDGE_SEARCH_WIDTH))
                trace, nJumps =  tracer.trace_edge(
                data, start, ORDER_EDGE_SEARCH_WIDTH, ORDER_EDGE_BG_WIDTH, ORDER_EDGE_JUMP_THRESH)

                if trace is None:
                    logger.warning('trace failed')
                    return None
                
                if nJumps > ORDER_EDGE_JUMP_LIMIT:
                    logger.warning('order edge trace jump limit exceeded: n jumps=' + 
                            str(nJumps) + ' limit=' + str(ORDER_EDGE_JUMP_LIMIT))
                    if config.params['spatial_jump_override'] is True:
                        logger.warning('spatial jump override enabled, edge not rejected')
                    else:
                        logger.info('edge rejected')
                        return None
    return trace
    
SKY_LINE_SEARCH_WIDTH = 3
SKY_LINE_BG_WIDTH     = 0
SKY_LINE_JUMP_THRESH  = 0.8
SKY_LINE_JUMP_LIMIT   = 10
        


def trace_sky_line(data, start, eta=None, arc=None):

    # updated params based on trial and error
    SKY_LINE_SEARCH_WIDTH = 5
    #print('ETA1', eta)
    if eta is not None: # Make a specific modification for etalon lamps
        SKY_LINE_SEARCH_WIDTH = 11
    if arc is not None: # Make a specific modification for arc lamps
        SKY_LINE_SEARCH_WIDTH = 11
    SKY_LINE_BG_WIDTH     = 0
    SKY_LINE_JUMP_THRESH  = 1
    SKY_LINE_JUMP_LIMIT   = 12
    if nirspec_constants.upgrade:
        SKY_LINE_JUMP_LIMIT   = 24 # Double this although the order width shouldn't be different

    trace, nJumps =  tracer.trace_edge_line(
            data, start, SKY_LINE_SEARCH_WIDTH, SKY_LINE_BG_WIDTH, SKY_LINE_JUMP_THRESH, eta=eta, arc=arc)
    
    if trace is None:
        logger.warning('sky/etalon/arc line trace failed')
        return None
    if nJumps > SKY_LINE_JUMP_LIMIT:
        logger.debug('sky/etalon/arc line trace jump limit exceeded: n jumps=' + 
                str(nJumps) + ' limit=' + str(SKY_LINE_JUMP_LIMIT))        
        return None
    logger.debug('sky/etalon/arc line accepted')  
    return trace



def smooth_spatial_trace(y_raw):
    """
    """
    
    deg          = 3
    n_end_ignore = 20
    threshold    = 3
    
    mask    = np.ones(y_raw.shape[0] - n_end_ignore, dtype=bool)
    mask    = np.append(mask, np.zeros(n_end_ignore, dtype=bool))
    
    x       = np.arange(y_raw.shape[0])
    
    coeffs  = np.polyfit(x[mask], y_raw[mask], deg)
    y_fit   = np.polyval(coeffs, x)
    res1    = y_raw - y_fit
    stdev1  = np.std(res1)
    
    greater = np.greater(np.absolute(res1), threshold * stdev1)
    mask    = np.logical_and(mask, np.logical_not(greater))
    
    coeffs  = np.polyfit(x[mask], y_raw[mask], deg)
    y_fit   = np.polyval(coeffs, x)
    res2    = y_raw - y_fit
    stdev2  = np.std(res2)
 
    return y_fit, mask



SKY_SIGMA           = 1.1
EXTRA_PADDING       = 5
MIN_LINE_SEPARATION = 5



def find_spectral_trace(data, numrows=5, eta=None, arc=None, plot=False):
    """
    Locates sky/etalon lines in the bottom 5 rows (is this really optimal?) of the order image data. 
    I fixed the above lines to check the bottom 5 rows and top 5 rows for which has more sky. - CAT
    Finds strongest peaks, sorts them, traces them, returns the average of the traces.
    Rejects line pairs that are too close together.
    Returns spectral trace as 1-d array.  Throws exception if can't find or trace lines.
    """
    SKY_SIGMA           = 2. #2.25 # old value
    MIN_LINE_SEPARATION = 5
    
    # transpose the array because spectroid can only read horizontal peaks for now
    data_t0  = data.transpose()

#     data_t = data_t[:, padding + 5:data_t.shape[1] - 5 - padding]
    data_t   = data_t0[:, 5:data_t0.shape[1] - 5]    

    crit_val = np.median(data_t) # Get a value for the background
    #print('Crit', crit_val, 2*crit_val)

    # This is a check to see if we should use the front rows or a few rows in
    if len(np.where(data_t[:, 0:numrows].flatten() > 2*crit_val)[0]) > 1000: 
        s = np.sum(data_t[:, numrows:numrows*2], axis=1) # This is a few rows in
    else:
        s = np.sum(data_t[:, 0:numrows], axis=1) # This is the first few rows

    if plot:
        import pylab as pl
        sky_thres = SKY_SIGMA * np.median(s)
        #print('SIG', np.median(s), SKY_SIGMA, sky_thres)
        pl.figure(facecolor='white')
        pl.cla()
        pl.plot(s, 'k-')
        pl.axhline(SKY_SIGMA * np.median(s), c='r', ls=':', label='sigma * median')
        pl.axhline(2.25 * np.median(s), c='b', ls=':', label='2.25 * median')
        pl.xlim(0, data_t.shape[0])
        pl.xlabel('column (pixels)')
        pl.ylabel('intensity summed over 5 rows (DN)')
        ymin, ymax = pl.ylim()
        pl.ylim(0, ymax)
        pl.legend()
        pl.show()

    # finds column indices of maxima
    if eta is not None:
        maxima_c = argrelextrema(s, np.greater, order=3) 
    elif arc is not None:
        maxima_c = argrelextrema(s, np.greater, order=3) 
    else:
        maxima_c = argrelextrema(s, np.greater)    
    
    # find indices in maxima_c of maxima with intensity 
    # greater than SKY_SIGMA * median extrema height
    sky_thres = SKY_SIGMA * np.median(s)
    locmaxes  = np.where(s[maxima_c[0]] > sky_thres)

    # indices in s or peaks
    maxes = np.array(maxima_c[0][locmaxes[0]])

    logger.debug('n sky/etalon/arc line peaks with intensity > {:.0f} = {}'.format(
                sky_thres, len(maxes)))

    deletelist = []
           
    # remove adjacent sky lines that are closer than MIN_LINE_SEPARATION pixels
    for i in range(1, len(maxes)):
        if abs(maxes[i] - maxes[i - 1]) < MIN_LINE_SEPARATION:
            deletelist.append(i)
    maxes = np.delete(maxes, deletelist, None)

    peaks = s[maxes] 

    sortorder = np.argsort(peaks)
            
    maxes = maxes[sortorder]
    maxes = maxes[::-1]

    # Try to find some fainter lines if the threshold was too large
    #print('MAXES', maxes, len(maxes), SKY_SIGMA)
    if len(maxes) < 5:
        for SKY_SIGMA in [1.5, 1.2]: 
            sky_thres = SKY_SIGMA * np.median(s)
            locmaxes  = np.where(s[maxima_c[0]] > sky_thres)

            # indices in s or peaks
            maxes = np.array(maxima_c[0][locmaxes[0]])
            
            logger.debug('n sky/etalon/arc line peaks with intensity > {:.0f} = {}'.format(
                    sky_thres, len(maxes)))

            deletelist = []
           
            # remove adjacent sky lines that are closer than MIN_LINE_SEPARATION pixels
            for i in range(1, len(maxes)):
                if abs(maxes[i] - maxes[i - 1]) < MIN_LINE_SEPARATION:
                    deletelist.append(i)
            maxes = np.delete(maxes, deletelist, None)

            peaks = s[maxes] 

            sortorder = np.argsort(peaks)
                    
            maxes = maxes[sortorder]
            maxes = maxes[::-1]

            if len(maxes) >= 5: # We have enough sky/etalon/arc lines
                break

    logger.info('n unblended sky/etalon/arc line peaks with intensity > {:.0f} = {}'.format(
                    sky_thres, len(maxes)))
    
    if plot:
        print('MAXES', maxes)
        pl.figure(facecolor='white')
        pl.cla()
        pl.plot(s, 'k-')
        pl.axhline(sky_thres, c='r', ls=':')
        pl.xlim(0, data_t.shape[0])
        pl.xlabel('column (pixels)')
        pl.ylabel('intensity summed over 5 rows (DN)')
        ymin, ymax = pl.ylim()
        pl.ylim(0, ymax)
        #pl.show()
        for max1 in maxes: pl.axvline(max1, color='b', ls='--', alpha=0.5, lw=0.5)

        pl.figure(27847)
        pl.imshow(data_t, origin='lower', aspect='auto')
        for max1 in maxes: pl.axhline(max1, color='r', ls='--', alpha=0.5, lw=0.5)
        pl.show()
    
    centroid_sky_sum = np.zeros(data_t.shape[1])
    fitnumber        = 0

    centroids = []

    lowlim, uplim = 10, 1024-14
    if nirspec_constants.upgrade:
        lowlim, uplim = 20, 2048-48

    for maxskyloc in maxes:
        if lowlim < maxskyloc < uplim:
            
            centroid_sky = trace_sky_line(data_t, maxskyloc, eta=eta, arc=arc)
           
            if centroid_sky is None:
                continue

            centroids.append(centroid_sky)

            fitnumber += 1

            #if eta is None:
            #    if fitnumber > 2: # Why are we limiting this?
            #        break



    if fitnumber > 0:
        logger.info(str(fitnumber) + ' sky/etalon/arc lines selected for spectral rectification')
        return centroids
        #return centroid_sky_sum / fitnumber
    
    logger.warning('failed to find sky/etalon/arc line trace')
    raise StandardError('failed to find sky/etalon/arc line trace')
    
    

def smooth_spectral_trace(data, l, eta=None, arc=None, version2=True, plot=False):
    
    if version2 == True:
        AllPix   = []
        AllCents = []
        for centroids in data:

            Pixels = np.arange(len(centroids))
            p0 = np.polyfit(Pixels, centroids, deg=1)  # end always drops off
            """
            if plot:
                fig1 = plt.figure(161, figsize=(8,4))
                ax1 = fig1.add_subplot(121)
                ax2 = fig1.add_subplot(122)
                z0 = np.poly1d(p0)
                ax1.scatter(Pixels, centroids)
                Xs = np.arange(np.min(Pixels), np.max(Pixels))
                ax1.plot(Xs, z0(Xs), 'r--')
                ax2.plot(Pixels, centroids - z0(Pixels))
                ax2.axhline(0, c='r', ls='--')
            """
            from astropy.stats import sigma_clip
            SigCut = True
            newData = centroids
            newPix2  = Pixels
            while SigCut:

                p0 = np.polyfit(newPix2, newData, deg=1)  # end always drops off
                z0 = np.poly1d(p0)

                filtered_data = sigma_clip(newData - z0(newPix2), sigma=2., maxiters=None)

                newCent = newData[~filtered_data.mask]
                newPix  = newPix2[~filtered_data.mask]
                """
                if plot:
                    fig2 = plt.figure(162, figsize=(8,4))
                    ax3 = fig2.add_subplot(121)
                    ax4 = fig2.add_subplot(122)
                    ax3.scatter(newPix, newCent)
                    ax3.plot(Xs, z0(Xs), 'r--')
                    ax4.plot(newPix, newCent - z0(newPix))
                    ax4.axhline(0, c='r', ls='--')
                    plt.show()
                """
                if len(newData) == len(newCent): 
                    SigCut = False
                newData = newCent
                newPix2 = newPix

            logger.debug('spectral tilt of line is {:.3f} pixels/pixel'.format(p0[0]))
            """
            # Moved this to later in the script
            # Remove lines with large RMS (> 0.1 pixels)
            # This is a sign that we did not resolve doublets or lines got noisy at ends
            rmse = np.sqrt(np.mean((newCent - z0(newPix))**2))
            logger.debug('RMSE of the line is {:.3f} pixels'.format(rmse))
            if rmse > 0.1: 
                continue
            """
            AllPix.append(newPix2)
            AllCents.append(newData - z0(0))

    
        PlotPix    = []
        PlotCent   = []
        PlotSlopes = []

        if plot:
            fig11 = plt.figure(165, figsize=(10,7))
            ax111 = fig11.add_subplot(221)
            ax222 = fig11.add_subplot(222)
            ax333 = fig11.add_subplot(223)
            ax444 = fig11.add_subplot(224)
        
        for Pixels, centroids in zip(AllPix, AllCents):
            p0 = np.polyfit(Pixels, centroids, deg=1)  # end always drops off
            z0 = np.poly1d(p0)
            
            if plot:
                ax111.scatter(Pixels, centroids)
                Xs = np.linspace(np.min(Pixels), np.max(Pixels))
                ax111.plot(Xs, z0(Xs), 'k:', lw=1, alpha=0.5)
            
            # Get the RMS of the line for later filtering (> 0.15 pixels)
            # This is a sign that we did not resolve doublets or lines got noisy at ends
            rmse = np.sqrt(np.mean((centroids - z0(Pixels))**2))
            if rmse > 0.15: 
            	logger.debug('RMSE of the line is {:.3f} pixels (outlier)'.format(rmse))
            else:
            	logger.debug('RMSE of the line is {:.3f} pixels'.format(rmse))
            if rmse > 0.15: continue

            PlotPix.append(Pixels)
            PlotCent.append(centroids)
            PlotSlopes.append(p0[0])

        PlotSlopes = np.array(PlotSlopes)

        p0 = np.polyfit(np.concatenate(PlotPix).ravel(), np.concatenate(PlotCent).ravel(), deg=1)
        z0 = np.poly1d(p0)
        if plot:
            ax111.plot(np.linspace(0,45), z0(np.linspace(0,45)), 'k-', alpha=0.5, lw=1.5)
            ax222.hist(PlotSlopes, bins = int(np.sqrt(len(PlotSlopes))))
            ax222.axvline(np.mean(PlotSlopes), color='k', ls='--')
            ax222.axvline(np.mean(PlotSlopes) -   np.std(PlotSlopes), color='k', ls=':')
            ax222.axvline(np.mean(PlotSlopes) +   np.std(PlotSlopes), color='k', ls=':')
            ax222.axvline(np.mean(PlotSlopes) - 2*np.std(PlotSlopes), color='b', ls=':')
            ax222.axvline(np.mean(PlotSlopes) + 2*np.std(PlotSlopes), color='b', ls=':')
            ax222.axvline(np.mean(PlotSlopes) - 3*np.std(PlotSlopes), color='r', ls=':')
            ax222.axvline(np.mean(PlotSlopes) + 3*np.std(PlotSlopes), color='r', ls=':')
            xmin, xmax = ax222.get_xlim()
        
        # Set a mask for things that are 2-sigma outliers.
        # We want a tight relation for spectral rectification
        mask = np.zeros(len(PlotSlopes))
        mask[np.where( (PlotSlopes < np.mean(PlotSlopes)-2*np.std(PlotSlopes)) | 
                       (PlotSlopes > np.mean(PlotSlopes)+2*np.std(PlotSlopes)) 
                     ) ] = 1

        PlotPix2    = []
        PlotCent2   = []
        PlotSlopes2 = []

        linecount=0
        for i in range(len(mask)):
            if mask[i] == 0:
                PlotPix2.append(PlotPix[i])
                PlotCent2.append(PlotCent[i])
                PlotSlopes2.append(PlotSlopes[i])
                linecount+=1
                
                if plot:
                    ax333.scatter(PlotPix[i], PlotCent[i])
                    Xs = np.linspace(np.min(PlotPix[i]), np.max(PlotPix[i]))
                    p0 = np.polyfit(PlotPix[i], PlotCent[i], deg=1)  # end always drops off
                    z0 = np.poly1d(p0)
                    ax333.plot(Xs, z0(Xs), 'k:', lw=1, alpha=0.5)
                    
            else: 
                continue

        p0 = np.polyfit(np.concatenate(PlotPix2).ravel(), np.concatenate(PlotCent2).ravel(), deg=1)
        z0 = np.poly1d(p0)

        # Log the number of lines used
        logger.success(str(linecount) + ' sky/etalon/arc lines used for spectral rectification')

        if plot:
            ax333.plot(np.linspace(0,45), z0(np.linspace(0,45)), 'k-', alpha=0.5, lw=1.5)
            ax444.hist(PlotSlopes2, bins = int(np.sqrt(len(PlotSlopes2))))
            ax444.axvline(np.mean(PlotSlopes2), color='k', ls='--')
            ax444.axvline(np.mean(PlotSlopes2) -   np.std(PlotSlopes2), color='k', ls=':')
            ax444.axvline(np.mean(PlotSlopes2) +   np.std(PlotSlopes2), color='k', ls=':')
            ax444.axvline(np.mean(PlotSlopes2) - 2*np.std(PlotSlopes2), color='b', ls=':')
            ax444.axvline(np.mean(PlotSlopes2) + 2*np.std(PlotSlopes2), color='b', ls=':')
            ax444.axvline(np.mean(PlotSlopes2) - 3*np.std(PlotSlopes2), color='r', ls=':')
            ax444.axvline(np.mean(PlotSlopes2) + 3*np.std(PlotSlopes2), color='r', ls=':')
            ax444.set_xlim(xmin, xmax)
        
        logger.debug('spectral tilt of lines are {:.3f} pixels/pixel'.format(p0[0]))
        
        if plot:
            plt.tight_layout()
            plt.show()
            #sys.exit()
        

        # Set the offset to zero
        p0[1] -= p0[1]
        
        logger.info('spectral tilt is {:.3f} pixels/pixel'.format(p0[0]))
        fit = np.polyval(p0, np.arange(l))
        #print('fit', fit)
        return fit
    

