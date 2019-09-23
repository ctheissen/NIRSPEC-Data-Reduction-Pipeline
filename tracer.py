import numpy as np
import scipy.ndimage

import CAT_Functions as cat
import scipy.optimize as op
import matplotlib.pyplot as plt
import nirspec_constants

def trace_edge(data, start, searchWidth, bgWidth, jumpThresh, plot=False):

    # initialize trace array
    trace = np.zeros(data.shape[1])

    # nJumps is the number of times successive centroids differed by more than threshold
    nJumps = 0

    # first centroid assumed to be at start
    trace[0] = start
        
    # find centroids for the rest of the columns in data

    for i in range(1, data.shape[1]):

        if nirspec_constants.upgrade:
            if i < 5: 
                trace[i] = start
                continue
        
        # define search window
        ymin = int(trace[i - 1] - searchWidth)
        ymax = int(trace[i - 1] + searchWidth)
        #print('limits', ymin, ymax)
        
        # clip search window at top and bottom of column
        if abs(ymax) > data.shape[0]:
            ymax = int(data.shape[0])

        if ymin < 1:  # don't let it trace the bottom of the detector
            ymin = 1
        if ymax <= 0:
            # how can this happen?
            ymax = int(trace[i] + searchWidth) + 1

        if bgWidth <= 0: 
            bgMean = 0.0;
        
        else:
            # If bgWidth > 0 then we will subtract average of pixel values at two locations
            # from each pixel value in search window.  Two locations are previous centroid
            # plus and minus search width.
        
            bgMin = trace[i - 1] - bgWidth
            if bgMin < 0:
                bgMin = 0
            bgMax = trace[i - 1] + bgWidth
            if bgMax > data.shape[0]:
                bgMax = data.shape[0] - 1
            
            try:
                bgMean = (data[bgMin, i] + data[bgMax, i]) / 2.0
            except:
                bgMean = 0.0
        
        #print('limits', ymin, ymax)
        #print('0', trace[i],  trace[i-1], ymin)
        trace[i] = scipy.ndimage.measurements.center_of_mass(
                data[int(ymin):int(ymax) + 1, i] - bgMean)[0] + ymin
        #print('1', trace[i])

        plotwidth = 3  # This is for order edges
        #plotwidth = 11 # This is for sky lines and etalons
        if plot and searchWidth==plotwidth:
            import pylab as pl
            x0 = max(0, int(trace[i - 1]) - 50)
            x1 = min(data.shape[0]-1, int(trace[i-1]) + 50)
            
            pl.figure()
            pl.cla()
            pl.plot(data[:,i])
            #pl.plot([trace[i], trace[i]], pl.ylim())
            pl.axvline(trace[i], color='r', ls='--')
            pl.axvline(ymin, color='r', ls=':')
            pl.axvline(ymax, color='r', ls=':')

            pl.figure()
            pl.cla()
            pl.plot(data[int(ymin):int(ymax) + 1, i] - bgMean)
            #pl.plot([trace[i], trace[i]], pl.ylim())
            #pl.axvline(trace[i], color='r', ls='--')

            pl.figure()
            pl.cla()
            pl.plot(np.arange(x0, x1), data[x0:x1, i])
            #pl.plot([trace[i], trace[i]], pl.ylim())
            pl.axvline(trace[i], color='r', ls='--')
            pl.axvline(ymin, color='r', ls=':')
            pl.axvline(ymax+1, color='r', ls=':')

            
            #pl.plot(data[x0:x1, i], 'ro', alpha=0.5)
            #pl.plot(data[int(ymin):int(ymax) + 1, i] - bgMean, 'go', alpha=0.5)
            #pl.plot([trace[i], trace[i]], [0, pl.ylim()[0]], 'g-')
            #print(trace[max(0, i-10):i])
            
     
            pl.show()

        if trace[i] is np.inf or trace[i] is -np.inf:  
            # went off array
            print('went off array')
            return None, None
    
        # centroid jumped more than traceDelta
        if np.abs(trace[i] - trace[i - 1]) > jumpThresh:
            nJumps += 1
            if i > 4:
                # jump is past beginning, use past three centroids
                trace[i] = trace[i - 3:i - 1].mean()
            elif i > 1:
                # average as many traces as we have gone through
                trace[i] = trace[i - 2:i - 1].mean()
            else:
                # use the first one found
                trace[i] = trace[i - 1]
        #print('Final', i, trace[i])
    return trace, nJumps



def trace_edge_line(data, start, searchWidth, bgWidth, jumpThresh, eta=None, arc=None, plotvid=False):

    # initialize trace array
    trace = np.zeros(data.shape[1])

    # nJumps is the number of times successive centroids differed by more than threshold
    nJumps = 0

    # find centroids for the rest of the columns in data
    stepcount = 3
    if eta is not None: # Need some extra signal for etalon lamps
        stepcount = 7 
    if arc is not None: # Need some extra signal for arc lamps
        stepcount = 7 


    # first centroid assumed to be at start
    trace[0] = start 

    #plt.imshow(data, origin='lower', aspect='auto')
    #plt.show()
    for i in range(1, data.shape[1], 1):
        #print('I',i)
        
        # define search window
        ymin = int(trace[i - 1] - searchWidth)
        ymax = int(trace[i - 1] + searchWidth)
        
        # clip search window at top and bottom of column
        if abs(ymax) > data.shape[0]:
            ymax = int(data.shape[0])

        if ymin < 1:  # don't let it trace the bottom of the detector
            ymin = 1
        if ymax <= 0:
            # how can this happen?
            ymax = int(trace[i] + searchWidth) + 1

        if bgWidth <= 0: 
            bgMean = 0.0;
        
        else:
            # If bgWidth > 0 then we will subtract average of pixel values at two locations
            # from each pixel value in search window.  Two locations are previous centroid
            # plus and minus search width.
        
            bgMin = trace[i - 1] - bgWidth
            if bgMin < 0:
                bgMin = 0
            bgMax = trace[i - 1] + bgWidth
            if bgMax > data.shape[0]:
                bgMax = data.shape[0] - 1
            
            try:
                bgMean = (data[bgMin, i] + data[bgMax, i]) / 2.0
            except:
                bgMean = 0.0


        if i < stepcount:
            Xs     = np.arange(len(np.sum(data[int(ymin):int(ymax)+1, i:i+stepcount+1], axis=1))) + ymin
            Ys     = np.sum(data[int(ymin):int(ymax)+1, i:i+stepcount+1], axis=1)
            guess1 = Xs[int(len(Xs)/2)]
            try:
                popt, pcov = op.curve_fit(cat.NormDist, Xs, Ys, 
                                      p0     = [guess1, 2., np.min(Ys), np.max(Ys)], 
                                      bounds = ( (guess1-3, 1., -100., 0.), (guess1+3, 8., 1e10, 1e10) ),
                                      maxfev = 1000000) 
                trace[i] = popt[0]
            except:
                nJumps += 1
                continue

            if plotvid:
                fig0 = plt.figure(198, figsize=(8,4))
                ax1 = fig0.add_subplot(121)
                ax2 = fig0.add_subplot(122)
                ax1.imshow(data[int(ymin):int(ymax) + 1, i:i+stepcount+1], origin='lower', aspect='auto')
                ax1.axhline(popt[0]-ymin, c='r', ls=':')
                ax2.plot(Ys, Xs)
                Xs2 = np.linspace(np.min(Xs), np.max(Xs))
                ax2.plot(cat.NormDist(Xs2, *popt), Xs2, 'r--')
                ax2.axhline(popt[0], c='r', ls=':')
                ax1.minorticks_on()
                ax2.minorticks_on()
                plt.draw()
                plt.pause(0.05)
                plt.close('all')
                #plt.show()
                #sys.exit() 

        else:

            Xs = np.arange(len(np.sum(data[int(ymin):int(ymax)+1, i-stepcount:i+stepcount+1], axis=1))) + ymin
            Ys = np.sum(data[int(ymin):int(ymax)+1, i-stepcount:i+stepcount+1], axis=1)
            guess1 = Xs[int(len(Xs)/2)]
            try:
                popt, pcov = op.curve_fit(cat.NormDist, Xs, Ys, 
                                      p0     = [guess1, 2., np.min(Ys), np.max(Ys)], 
                                      bounds = ( (guess1-3, 1., -100., 0.), (guess1+3, 8., 1e10, 1e10) ),
                                      maxfev = 1000000) 
                trace[i] = popt[0]
            except:
                nJumps += 1
                continue

            if plotvid:
                fig0 = plt.figure(198, figsize=(8,4))
                ax1 = fig0.add_subplot(121)
                ax2 = fig0.add_subplot(122)
                ax1.imshow(data[int(ymin):int(ymax) + 1, i-stepcount:i+stepcount+1], origin='lower', aspect='auto')
                ax1.axhline(popt[0]-ymin, c='r', ls=':')
                ax2.plot(Ys, Xs)
                Xs2 = np.linspace(np.min(Xs), np.max(Xs))
                ax2.plot(cat.NormDist(Xs2, *popt), Xs2, 'r--')
                ax2.axvline(popt[0], c='r', ls=':')
                ax1.minorticks_on()
                ax2.minorticks_on()
                plt.draw()
                plt.pause(0.05)
                plt.close('all')
                #plt.show()
                #sys.exit() 
             

            # pl.figure(101)
            # pl.cla()
            # pl.plot(np.arange(x0, x1), np.sum(data[x0:x1, i:i+5], axis=1))
            # pl.plot([trace[i], trace[i]], pl.ylim())
            # pl.plot([trace[i-1], trace[i-1]], pl.ylim(), 'r--')
            # pl.plot([ymin, ymin], pl.ylim(), 'r-')
            # pl.plot([ymax, ymax], pl.ylim(), 'r-')
            # #pl.plot([trace[i], trace[i]], [0, pl.ylim()[0]], 'g-')

            # pl.figure(102)
            # pl.plot(data[x0:x1, i], 'ro')
            # pl.plot(data[int(ymin):int(ymax) + 1, i] - bgMean, 'go')
            # pl.plot([trace[i], trace[i]], [0, pl.ylim()[0]], 'g-')
            # print(trace[max(0, i-10):i])
            # print(np.abs(trace[i] - trace[i - 1]), trace[i], trace[i-1], jumpThresh)
            # print('TEST2', np.flatnonzero(data[int(ymin):int(ymax) + 1, i] - bgMean).mean()+0.5)
            # print(nJumps)
            
            # pl.show()

        if trace[i] is np.inf or trace[i] is -np.inf:  
            # went off array
            print('went off array')
            return None, None
    
        # centroid jumped more than traceDelta
        #print('dist', np.abs(trace[i] - trace[i - 1]), 1)
        if np.abs(trace[i] - trace[i - 1]) > jumpThresh:
            nJumps += 1
            if i > 4:
                # jump is past beginning, use past three centroids
                trace[i] = trace[i - 3:i - 1].mean()
            elif i > 1:
                # average as many traces as we have gone through
                trace[i] = trace[i - 2:i - 1].mean()
            else:
                # use the first one found
                trace[i] = trace[i]# - 1]


    return trace, nJumps

