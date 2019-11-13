# Functions obtained from Ian Crossfield's library of astro routines
import logging
import numpy as np
import scipy.optimize as op
import image_lib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy import signal
import nirspec_constants
from astropy.visualization import ZScaleInterval, ImageNormalize, SquaredStretch

from numpy import polyfit, polyval, isfinite, ones
from numpy.linalg import LinAlgError
from pylab import plot, legend, title, figure, show
from scipy import signal
#from nsdata import imshow, bfixpix

logger = logging.getLogger('obj')

def optimalExtract(frame, variance, gain, readnoise, **kw): 
    """
    Extract spectrum, following Horne 1986.

    :INPUTS:
       data : 2D Numpy array
         Appropriately calibrated frame from which to extract
         spectrum.  Should be in units of ADU, not electrons!

       variance : 2D Numpy array
         Variances of pixel values in 'data'.

       gain : scalar
         Detector gain, in electrons per ADU

       readnoise : scalar
         Detector readnoise, in electrons.

    :OPTIONS:
       goodpixelmask : 2D numpy array
         Equals 0 for bad pixels, 1 for good pixels

       bkg_radii : 2- or 4-sequence
         If length 2: inner and outer radii to use in computing
         background. Note that for this to be effective, the spectral
         trace should be positions in the center of 'data.'
         
         If length 4: start and end indices of both apertures for
         background fitting, of the form [b1_start, b1_end, b2_start,
         b2_end] where b1 and b2 are the two background apertures, and
         the elements are arranged in strictly ascending order.

       extract_radius : int or 2-sequence
         radius to use for both flux normalization and extraction.  If
         a sequence, the first and last indices of the array to use
         for spectral normalization and extraction.


       dispaxis : bool
         0 for horizontal spectrum, 1 for vertical spectrum

       bord : int >= 0
         Degree of polynomial background fit.

       bsigma : int >= 0
         Sigma-clipping threshold for computing background.

       pord : int >= 0
         Degree of polynomial fit to construct profile.

       psigma : int >= 0
         Sigma-clipping threshold for computing profile.

       csigma : int >= 0
         Sigma-clipping threshold for cleaning & cosmic-ray rejection.

       finite : bool
         If true, mask all non-finite values as bad pixels.

       nreject : int > 0
         Number of pixels to reject in each iteration.
             
    :RETURNS:
       3-tuple:
          [0] -- spectrum flux (in electrons)

          [1] -- uncertainty on spectrum flux

          [1] -- background flux


    :EXAMPLE:
      ::


    :SEE_ALSO:
      :func:`superExtract`.

    :NOTES:
      Horne's classic optimal extraction algorithm is optimal only so
      long as the spectral traces are very nearly aligned with
      detector rows or columns.  It is *not* well-suited for
      extracting substantially tilted or curved traces, for the
      reasons described by Marsh 1989, Mukai 1990.  For extracting
      such spectra, see :func:`superExtract`.
    """

    # 2012-08-20 08:24 IJMC: Created from previous, low-quality version.
    # 2012-09-03 11:37 IJMC: Renamed to replace previous, low-quality
    #                        version. Now bkg_radii and extract_radius
    #                        can refer to either a trace-centered
    #                        coordinate system, or the specific
    #                        indices of all aperture edges. Added nreject.


    # Parse options:
    if 'goodpixelmask' in kw.keys():
        goodpixelmask = np.array(kw['goodpixelmask'], copy=True).astype(bool)
    else:
        goodpixelmask = np.ones(frame.shape, dtype=bool)

    if 'dispaxis' in kw.keys():
        if kw['dispaxis']==1:
            frame         = frame.transpose()
            variance      = variance.transpose()
            goodpixelmask = goodpixelmask.transpose()
            logger.debug("Flipping dispersion axis")

    if 'verbose' in kw.keys():
        verbose = kw['verbose']
    else:
        verbose = False

    if 'bkg_radii' in kw.keys():
        bkg_radii = kw['bkg_radii']
    else:
        bkg_radii = [15, 20]
        if verbose: 
            logger.debug("Setting option 'bkg_radii' to: " + str(bkg_radii))

    if 'extract_radius' in kw.keys():
        extract_radius = kw['extract_radius']
    else:
        extract_radius = 10
        if verbose: 
            logger.debug("Setting option 'extract_radius' to: " + str(extract_radius))

    if 'bord' in kw.keys():
        bord = kw['bord']
    else:
        bord = 1
        if verbose: 
            logger.debug("Setting option 'bord' to: " + str(bord))

    if 'bsigma' in kw.keys():
        bsigma = kw['bsigma']
    else:
        bsigma = 3
        if verbose: 
            logger.debug("Setting option 'bsigma' to: " + str(bsigma))

    if 'pord' in kw.keys():
        pord = kw['pord']
    else:
        pord = 2
        if verbose: 
            logger.debug("Setting option 'pord' to: " + str(pord))

    if 'psigma' in kw.keys():
        psigma = kw['psigma']
    else:
        psigma = 4
        if verbose: 
            logger.debug("Setting option 'psigma' to: " + str(psigma))

    if 'csigma' in kw.keys():
        csigma = kw['csigma']
    else:
        csigma = 5
        if verbose: 
            logger.debug("Setting option 'csigma' to: " + str(csigma))

    if 'finite' in kw.keys():
        finite = kw['finite']
    else:
        finite = True
        if verbose: 
            logger.debug("Setting option 'finite' to: " + str(finite))

    if 'nreject' in kw.keys():
        nreject = kw['nreject']
    else:
        nreject = 100
        if verbose: 
            logger.debug("Setting option 'nreject' to: " + str(nreject))
    '''
    plt.figure(10922)
    norm = ImageNormalize(frame, interval=ZScaleInterval(), stretch=SquaredStretch())
    plt.imshow(frame, origin='lower', aspect='auto', norm=norm)
    plt.figure(10923)
    norm = ImageNormalize(variance, interval=ZScaleInterval(), stretch=SquaredStretch())
    plt.imshow(frame, origin='lower', aspect='auto', norm=norm)
    plt.show()
    '''
    if finite:
        goodpixelmask *= (np.isfinite(frame) * np.isfinite(variance))

    #plt.figure(10924)
    #plt.imshow(frame, origin='lower', aspect='auto')
    
    variance[np.where(1 - goodpixelmask == 1)] = np.max(frame[np.where(goodpixelmask==1)]) * 1e9 # Need to figure out what is going on with good pixel mask
    fitwidth, nlam = frame.shape

    #plt.figure(10925)
    #plt.imshow(frame, origin='lower', aspect='auto')

    xxx  = np.arange(-fitwidth/2, fitwidth/2)
    xxx0 = np.arange(fitwidth)
    if len(bkg_radii)==4: # Set all borders of background aperture:
        backgroundAperture = ((xxx0 > bkg_radii[0]) * (xxx0 <= bkg_radii[1])) + \
            ((xxx0 > bkg_radii[2]) * (xxx0 <= bkg_radii[3]))
    else: # Assume trace is centered, and use only radii.
        backgroundAperture = (np.abs(xxx) > bkg_radii[0]) * (np.abs(xxx) <= bkg_radii[1])

    if hasattr(extract_radius, '__iter__'):
        extractionAperture = (xxx0 > extract_radius[0]) * (xxx0 <= extract_radius[1])
    else:
        extractionAperture = np.abs(xxx) < extract_radius

    nextract = extractionAperture.sum()
    xb       = xxx[backgroundAperture]

    # Step 3: Sky Subtraction
    if bord==0: # faster to take weighted mean:
        '''
        print('Background aperature', backgroundAperture)
        print(backgroundAperture.shape)
        print(frame.shape)
        print(frame[backgroundAperture, :].shape)
        #plt.figure(10927)
        #plt.imshow(frame, origin='lower', aspect='auto')
        #plt.axhline()
        
        plt.figure(10927)
        norm = ImageNormalize(frame[backgroundAperture, :], interval=ZScaleInterval(), stretch=SquaredStretch())
        plt.imshow(frame[backgroundAperture, :], origin='lower', aspect='auto')
        #plt.show()

        plt.figure(10928)
        plt.imshow((goodpixelmask/variance)[backgroundAperture, :], origin='lower', aspect='auto')
        plt.show()
        #sys.exit()
        '''
        background = wmean(frame[backgroundAperture, :], (goodpixelmask/variance)[backgroundAperture, :], axis=0)
    else:
        background = 0. * frame
        for ii in range(nlam):
            #print(xb)
            #print(frame[ii, backgroundAperture])
            #print(bord)
            #print(bsigma)
            #print((goodpixelmask/variance)[ii, backgroundAperture])
            #print()
            fit = polyfitr(xb, frame[backgroundAperture, ii], bord, bsigma, w=(goodpixelmask/variance)[backgroundAperture, ii], verbose=verbose-1)
            background[:, ii] = np.polyval(fit, xxx)

    # (my 3a: mask any bad values)
    #plt.figure(10927)
    #plt.plot(background.flatten())#, origin='lower', aspect='auto', norm=norm)

    badBackground = 1 - np.isfinite(background)
    #print(background)
    #print('Finite', np.isfinite(background))
    #print(frame.shape, background.shape, badBackground.shape)

    #plt.figure(10928)
    #plt.plot(badBackground)#, origin='lower', aspect='auto', norm=norm)

    background[np.where(badBackground==1)] = 0.
    if verbose and badBackground.any():
        print("Found bad background values at: ", badBackground.nonzero())

    skysubFrame = frame - background

    '''
    # Step 4: Extract 'standard' spectrum and its variance
    plt.figure(10929)
    #norm = ImageNormalize(background, interval=ZScaleInterval(), stretch=SquaredStretch())
    plt.plot(background.flatten())#, origin='lower', aspect='auto', norm=norm)

    plt.figure(10930)
    norm = ImageNormalize(frame, interval=ZScaleInterval(), stretch=SquaredStretch())
    plt.imshow(frame, origin='lower', aspect='auto', norm=norm)

    plt.figure(10931)
    norm = ImageNormalize(skysubFrame, interval=ZScaleInterval(), stretch=SquaredStretch())
    plt.imshow(skysubFrame, origin='lower', aspect='auto', norm=norm)
    #plt.axhline(extract_radius[0], c='r', ls=':')
    #plt.axhline(extract_radius[1], c='r', ls=':')
    #plt.show()
    #print('Extraction aperture:', extractionAperture)

    plt.figure(10932)
    norm = ImageNormalize(skysubFrame[extractionAperture, :], interval=ZScaleInterval(), stretch=SquaredStretch())
    plt.imshow(skysubFrame[extractionAperture, :], origin='lower', aspect='auto', norm=norm)
    '''
    standardSpectrum    = nextract * wmean(skysubFrame[extractionAperture, :], goodpixelmask[extractionAperture, :], axis=0) 
    varStandardSpectrum = nextract * wmean(variance[extractionAperture, :], goodpixelmask[extractionAperture, :], axis=0)
    '''
    plt.figure(10134)    
    norm = ImageNormalize(skysubFrame[extractionAperture, :], interval=ZScaleInterval(), stretch=SquaredStretch())
    plt.imshow(skysubFrame[extractionAperture, :], origin='lower', aspect='auto', norm=norm)
    print(nextract)
    plt.figure(10933)
    plt.plot(standardSpectrum.flatten())
    #plt.show()
    #plt.figure(10934)
    #plt.plot(varStandardSpectrum.flatten())
    #plt.show()
    '''
    # (my 4a: mask any bad values)
    badSpectrum                                     = 1 - (np.isfinite(standardSpectrum))
    standardSpectrum[np.where(badSpectrum == 1)]    = 1.
    varStandardSpectrum[np.where(badSpectrum == 1)] = varStandardSpectrum[np.where(1 - badSpectrum == 1)].max() * 1e9


    # Step 5: Construct spatial profile; enforce positivity & normalization
    normData    = skysubFrame / standardSpectrum
    varNormData = variance / standardSpectrum**2
    '''
    plt.figure(10934)    
    norm = ImageNormalize(normData, interval=ZScaleInterval(), stretch=SquaredStretch())
    plt.imshow(normData, origin='lower', aspect='auto', norm=norm)
    plt.show()
    '''
    # Iteratively clip outliers
    newBadPixels = True
    iter1        = 0
    iter0        = -1
    if verbose: 
        print("Looking for bad pixel outliers in profile construction.")
    xl = np.linspace(-1., 1., nlam)

    while newBadPixels:
        iter0 += 1

        if pord==0: # faster to take weighted mean:
            profile = np.tile(wmean(normData, (goodpixelmask/varNormData), axis=1), (1, nlam))

        else:
            profile = 0. * frame
            for ii in np.arange(fitwidth):
                #print(ii, np.arange(fitwidth))
                #plt.figure(1)
                #plt.plot(xl, normData[ii, :])
                #plt.show()
                fit = np.polyfit(xl, normData[ii, :], pord, np.inf, w=(goodpixelmask/varNormData)[ii, :])
                profile[ii, :] = np.polyval(fit, xl)

        #norm = ImageNormalize(profile, interval=ZScaleInterval(), stretch=SquaredStretch())
        #plt.figure(9999)
        #plt.imshow(profile, origin='lower', aspect='auto', norm=norm)
        #plt.show()
        #plt.figure(99999)
        #plt.plot(wmean(normData, (goodpixelmask/varNormData), axis=1).flatten())
        #plt.show()

        if profile.min() < 0:
            profile[np.where(profile < 0)] = 0.
        #print('profile1.5', profile)
        #print(np.sum(np.isnan(profile)))
        #print('profile1.6', profile.sum(1).reshape(1, nlam))
        #print('profile1.7', np.nansum(profile, axis=1).reshape(1, nlam))
        #profile /= profile.nansum(1).reshape(nlam, 1)
        profile /= np.nansum(profile, axis=0).reshape(1, nlam)
        #print('profile2', profile)

        # Step 6: Revise variance estimates 
        modelData = standardSpectrum * profile + background
        variance  = (np.abs(modelData)/gain + (readnoise/gain)**2) / \
            (goodpixelmask + 1e-9) # Avoid infinite variance

        outlierSigmas = (frame - modelData)**2 / variance
        #print('outlier sigmas:', outlierSigmas)
        if outlierSigmas.max() > psigma**2:
            maxRejectedValue             = max(psigma**2, np.sort(outlierSigmas[extractionAperture, :].ravel())[-nreject])
            worstOutliers                = (outlierSigmas >= maxRejectedValue).nonzero()
            goodpixelmask[worstOutliers] = False
            newBadPixels                 = True
            numberRejected               = len(worstOutliers[0])
            iter1                       += numberRejected
        else:
            newBadPixels                 = False
            numberRejected               = 0

        if verbose: 
            print("Rejected %i pixels on this iteration " % numberRejected)

        #Step 5: Construct spatial profile; enforce positivity & normalization
        varNormData = variance / standardSpectrum**2

    if verbose: 
        print("%i bad pixels found over %s iterations" % (iter1, iter0))


    # Iteratively clip Cosmic Rays
    newBadPixels = True
    iter1        = 0
    iter0        = -1
    if verbose: 
        print("Looking for bad pixel outliers in optimal extraction.")
    while newBadPixels:
        iter0 += 1

        #Step 8: Extract optimal spectrum and its variance
        gp          = goodpixelmask * profile
        denom       = (gp * profile / variance)[extractionAperture, :].sum(0)
        spectrum    = ((gp * skysubFrame  / variance)[extractionAperture, :].sum(0) / denom).reshape(1, nlam)
        varSpectrum = (gp[extractionAperture, :].sum(0) / denom).reshape(1, nlam)

        #norm = ImageNormalize(profile, interval=ZScaleInterval(), stretch=SquaredStretch())
        #print(goodpixelmask)
        #print(profile)
        #print(gp)
        #print(skysubFrame  / variance)
        #print(denom)
        #plt.figure(999999)
        #plt.plot(spectrum.flatten())
        ##plt.imshow(profile, origin='lower', aspect='auto', norm=norm)
        #plt.show()


        # Step 6: Revise variance estimates 
        modelData = spectrum * profile + background
        variance = (np.abs(modelData)/gain + (readnoise/gain)**2) / \
            (goodpixelmask + 1e-9) # Avoid infinite variance


        # Iterate until worse outliers are all identified:
        outlierSigmas = (frame - modelData)**2 / variance
        if outlierSigmas.max() > csigma**2:
            maxRejectedValue             = max(csigma**2, np.sort(outlierSigmas[extractionAperture, :].ravel())[-nreject])
            worstOutliers                = (outlierSigmas>=maxRejectedValues).nonzero()
            goodpixelmask[worstOutliers] = False
            newBadPixels                 = True
            numberRejected               = len(worstOutliers[0])
            iter1                       += numberRejected

        else:
            newBadPixels                 = False
            numberRejected               = 0

        if verbose: 
            print("Rejected %i pixels on this iteration " % numberRejected)


    if verbose: 
        print("%i bad pixels found over %s iterations" % (iter1, iter0))

    ret = (spectrum.flatten(), varSpectrum.flatten(), profile, background, goodpixelmask)

    plt.figure(101)
    plt.plot(spectrum.flatten())
    plt.figure(102)
    plt.plot(varSpectrum.flatten())
    plt.figure(103)
    plt.plot(profile)
    plt.figure(104)
    plt.plot(background.flatten())
    plt.figure(105)
    plt.plot(goodpixelmask)
    plt.figure(106)
    plt.plot(spectrum.flatten() / np.sqrt(varSpectrum.flatten()))
    plt.show()
    #sys.exit()

    return  ret


def wmean(a, w, axis=None, reterr=False):
    """wmean(a, w, axis=None)

    Perform a weighted mean along the specified axis.

    :INPUTS:
      a : sequence or Numpy array
        data for which weighted mean is computed

      w : sequence or Numpy array
        weights of data -- e.g., 1./sigma^2

      reterr : bool
        If True, return the tuple (mean, err_on_mean), where
        err_on_mean is the unbiased estimator of the sample standard
        deviation.

    :SEE ALSO:  :func:`wstd`
    """
    # 2008-07-30 12:44 IJC: Created this from ...
    # 2012-02-28 20:31 IJMC: Added a bit of documentation
    # 2012-03-07 10:58 IJMC: Added reterr option

    newdata    = np.array(a, subok=True, copy=True)
    newweights = np.array(w, subok=True, copy=True)

    if axis==None:
        newdata    = newdata.ravel()
        newweights = newweights.ravel()
        axis       = 0

    ash       = list(newdata.shape)
    wsh       = list(newweights.shape)

    nsh       = list(ash)
    nsh[axis] = 1

    if ash != wsh:
        logger.warning('Data and weight must be arrays of same shape.')
        return []
    
    wsum = newweights.sum(axis=axis).reshape(nsh) 

    weightedmean = (a * newweights).sum(axis=axis).reshape(nsh) / wsum
    '''
    plt.figure(947632)
    print(a.shape)
    print(a)
    plt.plot(a)
    plt.figure(947633)
    print(newweights.shape)
    plt.plot(newweights)
    print(newweights)
    plt.figure(947634)
    testx = (a * newweights).sum(axis=axis).reshape(nsh)
    print(testx.shape)
    print(testx)
    plt.plot(testx)
    plt.figure(947635)
    print(wsum.shape)
    plt.plot(wsum)
    plt.figure(947636)
    print(weightedmean.shape)
    plt.plot(weightedmean)
    plt.show()
    '''
    if reterr:
        # Biased estimator:
        #e_weightedmean = sqrt((newweights * (a - weightedmean)**2).sum(axis=axis) / wsum)

        # Unbiased estimator:
        #e_weightedmean = sqrt((wsum / (wsum**2 - (newweights**2).sum(axis=axis))) * (newweights * (a - weightedmean)**2).sum(axis=axis))
        
        # Standard estimator:
        e_weightedmean = np.sqrt(1./newweights.sum(axis=axis))

        ret = weightedmean, e_weightedmean
    else:
        ret = weightedmean

    return ret


def polyfitr(x, y, N, s, fev=100, w=None, diag=False, clip='both', \
                 verbose=True, plotfit=False, plotall=False, eps=1e-13, catchLinAlgError=False):
    """Matplotlib's polyfit with weights and sigma-clipping rejection.

    :DESCRIPTION:
      Do a best fit polynomial of order N of y to x.  Points whose fit
      residuals exeed s standard deviations are rejected and the fit is
      recalculated.  Return value is a vector of polynomial
      coefficients [pk ... p1 p0].

    :OPTIONS:
        w:   a set of weights for the data; uses CARSMath's weighted polynomial 
             fitting routine instead of numpy's standard polyfit.

        fev:  number of function evaluations to call before stopping

        'diag'nostic flag:  Return the tuple (p, chisq, n_iter)

        clip: 'both' -- remove outliers +/- 's' sigma from fit
              'above' -- remove outliers 's' sigma above fit
              'below' -- remove outliers 's' sigma below fit

        catchLinAlgError : bool
          If True, don't bomb on LinAlgError; instead, return [0, 0, ... 0].

    :REQUIREMENTS:
       :doc:`CARSMath`

    :NOTES:
       Iterates so long as n_newrejections>0 AND n_iter<fev. 


     """
    # 2008-10-01 13:01 IJC: Created & completed
    # 2009-10-01 10:23 IJC: 1 year later! Moved "import" statements within func.
    # 2009-10-22 14:01 IJC: Added 'clip' options for continuum fitting
    # 2009-12-08 15:35 IJC: Automatically clip all non-finite points
    # 2010-10-29 09:09 IJC: Moved pylab imports inside this function
    # 2012-08-20 16:47 IJMC: Major change: now only reject one point per iteration!
    # 2012-08-27 10:44 IJMC: Verbose < 0 now resets to 0
    # 2013-05-21 23:15 IJMC: Added catchLinAlgError

    if verbose < 0:
        verbose = 0

    xx = np.array(x, copy=False)
    yy = np.array(y, copy=False)
    noweights = (w==None)

    if w is not None:
        ww = np.array(w, copy=False)
    else:
        ww = np.ones(xx.shape, float)

    ii   = 0
    nrej = 1

    if w is not None:
        goodind = isfinite(xx)*isfinite(yy)*isfinite(ww)
    else:
        goodind = isfinite(xx)*isfinite(yy)

    xx2 = xx[np.where(goodind==1)]
    yy2 = yy[np.where(goodind==1)]
    ww2 = ww[np.where(goodind==1)]

    while (ii<fev and (nrej != 0)):
        if w is not None:
            if catchLinAlgError:
                try:
                    p = np.polyfit(xx2, yy2, N, w=ww2)
                except LinAlgError:
                    p = np.zeros(N+1, dtype=float)
            else:
                p = np.polyfit(xx2, yy2, N, w=ww2)

            p = p#[::-1]  # polyfit uses reverse coefficient ordering
            residual = (yy2 - np.polyval(p, xx2)) * np.sqrt(ww2)
            clipmetric = s
        else:
            p           = np.polyfit(xx2, yy2, N)
            residual    = yy2 - np.polyval(p, xx2)
            stdResidual = np.std(residual)
            clipmetric  = s * stdResidual

        if clip=='both':
            worstOffender = abs(residual).max()
            #pdb.set_trace()
            if worstOffender <= clipmetric or worstOffender < eps:
                ind = np.ones(residual.shape, dtype=bool)
            else:
                ind = abs(residual) < worstOffender
        elif clip=='above':
            worstOffender = residual.max()
            if worstOffender <= clipmetric:
                ind = np.ones(residual.shape, dtype=bool)
            else:
                ind = residual < worstOffender
        elif clip=='below':
            worstOffender = residual.min()
            if worstOffender >= -clipmetric:
                ind = np.ones(residual.shape, dtype=bool)
            else:
                ind = residual > worstOffender
        else:
            ind = np.ones(residual.shape, dtype=bool)
    
        xx2 = xx2[ind]
        yy2 = yy2[ind]
        if w is not None:
            ww2 = ww2[ind]
        ii += 1
        nrej = len(residual) - len(xx2)
        if plotall:
            figure()
            plot(x, y, '.', xx2, yy2, 'x', x, np.polyval(p, x), '--')
            legend(['data', 'fit data', 'fit'])
            title('Iter. #' + str(ii) + ' -- Close all windows to continue....')
            show()

        if verbose:
            print(str(len(x)-len(xx2)) + ' points rejected on iteration #' + str(ii))

    if (plotfit or plotall):
        figure()
        plot(x, y, '.', xx2, yy2, 'x', x, np.polyval(p, x), '--')
        legend(['data', 'fit data', 'fit'])
        title('Close window to continue....')
        show()

    if diag:
        chisq = ( (residual)**2 / yy2 ).sum()
        p = (p, chisq, ii)

    return p




def superExtract(frame, variance, gain, readnoise, *args, **kw):
    """
    Optimally extract curved spectra, following Marsh 1989.

    :INPUTS:
       data : 2D Numpy array
         Appropriately calibrated frame from which to extract
         spectrum.  Should be in units of ADU, not electrons!

       variance : 2D Numpy array
         Variances of pixel values in 'data'.

       gain : scalar
         Detector gain, in electrons per ADU

       readnoise : scalar
         Detector readnoise, in electrons.

    :OPTIONS:
       trace : 1D numpy array
         location of spectral trace.  If None, :func:`traceorders` is
         invoked.

       goodpixelmask : 2D numpy array
         Equals 0 for bad pixels, 1 for good pixels

       npoly : int
         Number of profile polynomials to evaluate (Marsh's
         "K"). Ideally you should not need to set this -- instead,
         play with 'polyspacing' and 'extract_radius.' For symmetry,
         this should be odd.

       polyspacing : scalar
         Spacing between profile polynomials, in pixels. (Marsh's
         "S").  A few cursory tests suggests that the extraction
         precision (in the high S/N case) scales as S^-2 -- but the
         code slows down as S^2.

       pord : int
         Order of profile polynomials; 1 = linear, etc.

       bkg_radii : 2-sequence
         inner and outer radii to use in computing background

       extract_radius : int
         radius to use for both flux normalization and extraction

       dispaxis : bool
         0 for horizontal spectrum, 1 for vertical spectrum

       bord : int >= 0
         Degree of polynomial background fit.

       bsigma : int >= 0
         Sigma-clipping threshold for computing background.

       tord : int >= 0
         Degree of spectral-trace polynomial (for trace across frame
         -- not used if 'trace' is input)

       csigma : int >= 0
         Sigma-clipping threshold for cleaning & cosmic-ray rejection.

       finite : bool
         If true, mask all non-finite values as bad pixels.

       qmode : str ('fast' or 'slow')
         How to compute Marsh's Q-matrix.  Valid inputs are
         'fast-linear', 'slow-linear', 'fast-nearest,' 'slow-nearest,'
         and 'brute'.  These select between various methods of
         integrating the nearest-neighbor or linear interpolation
         schemes as described by Marsh; the 'linear' methods are
         preferred for accuracy.  Use 'slow' if you are running out of
         memory when using the 'fast' array-based methods.  'Brute' is
         both slow and inaccurate, and should not be used.
         
       nreject : int
         Number of outlier-pixels to reject at each iteration. 

       retall : bool
         If true, also return the 2D profile, background, variance
         map, and bad pixel mask.
             
    :RETURNS:
       object with fields for:
         spectrum

         varSpectrum

         trace


    :EXAMPLE:
      ::

        import spec
        import numpy as np
        import pylab as py

        def gaussian(p, x):
           if len(p)==3:
               p = concatenate((p, [0]))
           return (p[3] + p[0]/(p[1]*sqrt(2*pi)) * exp(-(x-p[2])**2 / (2*p[1]**2)))

        # Model some strongly tilted spectral data:
        nx, nlam = 80, 500
        x0 = np.arange(nx)
        gain, readnoise = 3.0, 30.
        background = np.ones(nlam)*10
        flux =  np.ones(nlam)*1e4
        center = nx/2. + np.linspace(0,10,nlam)
        FWHM = 3.
        model = np.array([gaussian([flux[ii]/gain, FWHM/2.35, center[ii], background[ii]], x0) for ii in range(nlam)])
        varmodel = np.abs(model) / gain + (readnoise/gain)**2
        observation = np.random.normal(model, np.sqrt(varmodel))
        fitwidth = 60
        xr = 15

        output_spec = spec.superExtract(observation, varmodel, gain, readnoise, polyspacing=0.5, pord=1, bkg_radii=[10,30], extract_radius=5, dispaxis=1)

        py.figure()
        py.plot(output_spec.spectrum.squeeze() / flux)
        py.ylabel('(Measured flux) / (True flux)')
        py.xlabel('Photoelectrons')
        


    :TO_DO:
      Iterate background fitting and reject outliers; maybe first time
      would be unweighted for robustness.

      Introduce even more array-based, rather than loop-based,
      calculations.  For large spectra computing the C-matrix takes
      the most time; this should be optimized somehow.

    :SEE_ALSO:

    """

    # 2012-08-25 20:14 IJMC: Created.
    # 2012-09-21 14:32 IJMC: Added error-trapping if no good pixels
    #                      are in a row. Do a better job of extracting
    #                      the initial 'standard' spectrum.



    # Parse inputs:
    #frame, variance, gain, readnoise = args[0:4]

    frame                  = gain * np.array(frame, copy=False)
    variance               = gain**2 * np.array(variance, copy=False)
    variance[variance<=0.] = readnoise**2

    # Parse options:
    if 'verbose' in kw.keys():
        verbose = kw['verbose']
    else:
        verbose = False
    if verbose: from time import time


    if 'goodpixelmask' in kw.keys():
        goodpixelmask = np.array(kw['goodpixelmask'], copy=True).astype(bool)
    else:
        goodpixelmask = np.ones(frame.shape, dtype=bool)


    if 'dispaxis' in kw.keys():
        dispaxis = kw['dispaxis']
    else:
        dispaxis = 0

    if dispaxis==0:
        frame         = frame.transpose()
        variance      = variance.transpose()
        goodpixelmask = goodpixelmask.transpose()


    if 'pord' in kw.keys():
        pord = kw['pord']
    else:
        pord = 2

    if 'polyspacing' in kw.keys():
        polyspacing = kw['polyspacing']
    else:
        polyspacing = 1

    if 'bkg_radii' in kw.keys():
        bkg_radii = kw['bkg_radii']
    else:
        bkg_radii = [15, 20]
        if verbose: logger.debug("Setting option 'bkg_radii' to: " + str(bkg_radii))

    if 'extract_radius' in kw.keys():
        extract_radius = kw['extract_radius']
    else:
        extract_radius = 10
        if verbose: logger.debug("Setting option 'extract_radius' to: " + str(extract_radius))

    if 'npoly' in kw.keys():
        npoly = kw['npoly']
    else:
        npoly = 2 * int((2.0 * extract_radius) / polyspacing / 2.) + 1

    if 'bord' in kw.keys():
        bord = kw['bord']
    else:
        bord = 1
        if verbose: logger.debug("Setting option 'bord' to: " + str(bord))

    if 'tord' in kw.keys():
        tord = kw['tord']
    else:
        tord = 3
        if verbose: logger.debug("Setting option 'tord' to: " + str(tord))

    if 'bsigma' in kw.keys():
        bsigma = kw['bsigma']
    else:
        bsigma = 3
        if verbose: logger.debug("Setting option 'bsigma' to: " + str(bsigma))

    if 'csigma' in kw.keys():
        csigma = kw['csigma']
    else:
        csigma = 5
        if verbose: logger.debug("Setting option 'csigma' to: " + str(csigma))

    if 'qmode' in kw.keys():
        qmode = kw['qmode']
    else:
        qmode = 'fast'
        if verbose: logger.debug("Setting option 'qmode' to: " + str(qmode))

    if 'nreject' in kw.keys():
        nreject = kw['nreject']
    else:
        nreject = 100
        if verbose: logger.debug("Setting option 'nreject' to: " + str(nreject))

    if 'finite' in kw.keys():
        finite = kw['finite']
    else:
        finite = True
        if verbose: logger.debug("Setting option 'finite' to: " + str(finite))


    if 'retall' in kw.keys():
        retall = kw['retall']
    else:
        retall = False


    if finite:
        goodpixelmask *= (np.isfinite(frame) * np.isfinite(variance))

    variance[np.where(1 - goodpixelmask == 1)] = frame[np.where(goodpixelmask==1)].max() * 1e9
    fitwidth, nlam = frame.shape

    # Define trace (Marsh's "X_j" in Eq. 9)
    if 'trace' in kw.keys():
        trace = kw['trace']
    else:
        trace = None

    if trace is None:
        trace = 5
    '''
    if not hasattr(trace, '__iter__'):
        if verbose: logger.debug("Tracing not fully tested; dispaxis may need adjustment.")
        #pdb.set_trace()
        tracecoef = traceorders(frame, pord=trace, nord=1, verbose=verbose-1, plotalot=verbose-1, g=gain, rn=readnoise, badpixelmask=True-goodpixelmask, dispaxis=dispaxis, fitwidth=min(fitwidth, 80))
        trace     = np.polyval(tracecoef.ravel(), np.arange(nlam))
    '''
    
    #xxx = np.arange(-fitwidth/2, fitwidth/2)
    #backgroundAperture = (np.abs(xxx) > bkg_radii[0]) * (np.abs(xxx) < bkg_radii[1])
    #extractionAperture = np.abs(xxx) < extract_radius
    #nextract = extractionAperture.sum()
    #xb = xxx[backgroundAperture]

    xxx                 = np.arange(fitwidth) - trace.reshape(nlam,1)
    backgroundApertures = (np.abs(xxx) > bkg_radii[0]) * (np.abs(xxx) <= bkg_radii[1])
    extractionApertures = np.abs(xxx) <= extract_radius

    nextracts           = extractionApertures.sum(1)

    #Step3: Sky Subtraction
    background = 0. * frame
    for ii in range(nlam):
        if goodpixelmask[ii, backgroundApertures[ii]].any():
            fit = polyfitr(xxx[ii,backgroundApertures[ii]], frame[ii, backgroundApertures[ii]], bord, bsigma, w=(goodpixelmask/variance)[ii, backgroundApertures[ii]], verbose=verbose-1)
            background[ii, :] = np.polyval(fit, xxx[ii])
        else:
            background[ii] = 0.

    plt.figure(1948)
    plt.imshow(background, origin='lower', aspect='auto')
    plt.show()

    background_at_trace = np.array([np.interp(0, xxx[j], background[j]) for j in np.arange(nlam)])

    # (my 3a: mask any bad values)
    badBackground = True - np.isfinite(background)
    background[badBackground] = 0.
    if verbose and badBackground.any():
        logger.debug("Found bad background values at: ", badBackground.nonzero())

    skysubFrame = frame - background


    # Interpolate and fix bad pixels for extraction of standard
    # spectrum -- otherwise there can be 'holes' in the spectrum from
    # ill-placed bad pixels.
    fixSkysubFrame = bfixpix(skysubFrame, True-goodpixelmask, n=8, retdat=True)

    #Step4: Extract 'standard' spectrum and its variance
    standardSpectrum    = np.zeros((nlam, 1), dtype=float)
    varStandardSpectrum = np.zeros((nlam, 1), dtype=float)
    for ii in np.arange(nlam):
        thisrow_good            = extractionApertures[ii] #* goodpixelmask[ii] * 
        standardSpectrum[ii]    = fixSkysubFrame[ii, thisrow_good].sum()
        varStandardSpectrum[ii] = variance[ii, thisrow_good].sum()


    spectrum    = standardSpectrum.copy()
    varSpectrum = varStandardSpectrum

    # Define new indices (in Marsh's appendix):
    N  = pord + 1
    mm = np.tile(np.arange(N).reshape(N,1), (npoly)).ravel()
    nn = mm.copy()
    ll = np.tile(np.arange(npoly), N)
    kk = ll.copy()
    pp = N * ll + mm
    qq = N * kk + nn

    jj = np.arange(nlam)  # row (i.e., wavelength direction)
    ii = np.arange(fitwidth) # column (i.e., spatial direction)
    jjnorm = np.linspace(-1, 1, nlam) # normalized X-coordinate
    jjnorm_pow = jjnorm.reshape(1,1,nlam) ** (np.arange(2*N-1).reshape(2*N-1,1,1))

    # Marsh eq. 9, defining centers of each polynomial:
    constant = 0.  # What is it for???
    poly_centers = trace.reshape(nlam, 1) + polyspacing * np.arange(-npoly/2+1, npoly/2+1) + constant


    # Marsh eq. 11, defining Q_kij    (via nearest-neighbor interpolation)
    #    Q_kij =  max(0, min(S, (S+1)/2 - abs(x_kj - i)))
    if verbose: tic = time() 
    if qmode=='fast-nearest': # Array-based nearest-neighbor mode.
        if verbose: tic = time()
        Q = np.array([np.zeros((npoly, fitwidth, nlam)), np.array([polyspacing * np.ones((npoly, fitwidth, nlam)), 0.5 * (polyspacing+1) - np.abs((poly_centers - ii.reshape(fitwidth, 1, 1)).transpose(2, 0, 1))]).min(0)]).max(0)

    elif qmode=='slow-linear': # Code is a mess, but it works.
        invs = 1./polyspacing
        poly_centers_over_s = poly_centers / polyspacing
        xps_mat = poly_centers + polyspacing
        xms_mat = poly_centers - polyspacing
        Q = np.zeros((npoly, fitwidth, nlam), dtype=float)
        for i in range(fitwidth):
            ip05 = i + 0.5
            im05 = i - 0.5
            for j in range(nlam):
                for k in range(npoly):
                    xkj = poly_centers[j,k]
                    xkjs = poly_centers_over_s[j,k]
                    xps = xps_mat[j,k] #xkj + polyspacing
                    xms = xms_mat[j,k] # xkj - polyspacing

                    if (ip05 <= xms) or (im05 >= xps):
                        qval = 0.
                    elif (im05) > xkj:
                        lim1 = im05
                        lim2 = min(ip05, xps)
                        qval = (lim2 - lim1) * \
                            (1. + xkjs - 0.5*invs*(lim1+lim2))
                    elif (ip05) < xkj:
                        lim1 = max(im05, xms)
                        lim2 = ip05
                        qval = (lim2 - lim1) * \
                            (1. - xkjs + 0.5*invs*(lim1+lim2))
                    else:
                        lim1 = max(im05, xms)
                        lim2 = min(ip05, xps)
                        qval = lim2 - lim1 + \
                            invs * (xkj*(-xkj + lim1 + lim2) - \
                                        0.5*(lim1*lim1 + lim2*lim2))
                    Q[k,i,j] = max(0, qval)


    elif qmode=='fast-linear': # Code is a mess, but it's faster than 'slow' mode
        invs = 1./polyspacing
        xps_mat = poly_centers + polyspacing
        Q = np.zeros((npoly, fitwidth, nlam), dtype=float)
        for j in np.arange(nlam):
            xkj_vec  = np.tile(poly_centers[j,:].reshape(npoly, 1), (1, fitwidth))
            xps_vec  = np.tile(xps_mat[j,:].reshape(npoly, 1), (1, fitwidth))
            xms_vec  = xps_vec - 2*polyspacing

            ip05_vec = np.tile(np.arange(fitwidth) + 0.5, (npoly, 1))
            im05_vec = ip05_vec - 1
            ind00    = ((ip05_vec <= xms_vec) + (im05_vec >= xps_vec))
            ind11    = ((im05_vec > xkj_vec) * (True - ind00))
            ind22    = ((ip05_vec < xkj_vec) * (True - ind00))
            ind33    = (True - (ind00 + ind11 + ind22)).nonzero()
            ind11    = ind11.nonzero()
            ind22    = ind22.nonzero()

            n_ind11  = len(ind11[0])
            n_ind22  = len(ind22[0])
            n_ind33  = len(ind33[0])

            if n_ind11 > 0:
                ind11_3d    = ind11 + (np.ones(n_ind11, dtype=int)*j,)
                lim2_ind11  = np.array((ip05_vec[ind11], xps_vec[ind11])).min(0)
                Q[ind11_3d] = ((lim2_ind11 - im05_vec[ind11]) * invs * \
                                   (polyspacing + xkj_vec[ind11] - 0.5*(im05_vec[ind11] + lim2_ind11)))
            
            if n_ind22 > 0:
                ind22_3d    = ind22 + (np.ones(n_ind22, dtype=int)*j,)
                lim1_ind22  = np.array((im05_vec[ind22], xms_vec[ind22])).max(0)
                Q[ind22_3d] = ((ip05_vec[ind22] - lim1_ind22) * invs * \
                                   (polyspacing - xkj_vec[ind22] + 0.5*(ip05_vec[ind22] + lim1_ind22)))
            
            if n_ind33 > 0:
                ind33_3d    = ind33 + (np.ones(n_ind33, dtype=int)*j,)
                lim1_ind33  = np.array((im05_vec[ind33], xms_vec[ind33])).max(0)
                lim2_ind33  = np.array((ip05_vec[ind33], xps_vec[ind33])).min(0)
                Q[ind33_3d] = ((lim2_ind33 - lim1_ind33) + invs * \
                                   (xkj_vec[ind33] * (-xkj_vec[ind33] + lim1_ind33 + lim2_ind33) - 0.5*(lim1_ind33*lim1_ind33 + lim2_ind33*lim2_ind33)))
            

    elif qmode=='brute': # Neither accurate, nor memory-frugal.
        oversamp      = 4.
        jj2           = np.arange(nlam*oversamp, dtype=float) / oversamp
        trace2        = np.interp(jj2, jj, trace)
        poly_centers2 = trace2.reshape(nlam*oversamp, 1) + polyspacing * np.arange(-npoly/2+1, npoly/2+1, dtype=float) + constant
        x2            = np.arange(fitwidth*oversamp, dtype=float)/oversamp
        Q             = np.zeros((npoly, fitwidth, nlam), dtype=float)
        for k in np.arange(npoly):
            Q[k,:,:] = an.binarray((np.abs(x2.reshape(fitwidth*oversamp,1) - poly_centers2[:,k]) <= polyspacing), oversamp)

        Q /= oversamp*oversamp*2

    else:  # 'slow' Loop-based nearest-neighbor mode: requires less memory
        if verbose: tic = time()
        Q = np.zeros((npoly, fitwidth, nlam), dtype=float)
        for k in np.arange(npoly):
            for i in np.arange(fitwidth):
                for j in np.arange(nlam):
                    Q[k,i,j] = max(0, min(polyspacing, 0.5*(polyspacing+1) - np.abs(poly_centers[j,k] - i)))

    if verbose: logger.debug('%1.2f s to compute Q matrix (%s mode)' % (time() - tic, qmode))
        

    # Some quick math to find out which dat columns are important, and
    #   which contain no useful spectral information:
    Qmask  = Q.sum(0).transpose() > 0
    Qind   = Qmask.transpose().nonzero()
    Q_cols = [Qind[0].min(), Qind[0].max()]
    nQ     = len(Qind[0])
    Qsm    = Q[:,Q_cols[0]:Q_cols[1]+1,:]

    # Prepar to iteratively clip outliers
    newBadPixels = True
    iter1 = -1
    if verbose: logger.debug("Looking for bad pixel outliers.")
    while newBadPixels:
        iter1 += 1
        if verbose: logger.debug("Beginning iteration %i" % iter1)


        # Compute pixel fractions (Marsh Eq. 5):
        #     (Note that values outside the desired polynomial region
        #     have Q=0, and so do not contribute to the fit)
        #E = (skysubFrame / spectrum).transpose()
        invEvariance        = (spectrum**2 / variance).transpose()
        weightedE           = (skysubFrame * spectrum / variance).transpose() # E / var_E
        invEvariance_subset = invEvariance[Q_cols[0]:Q_cols[1]+1,:]

        # Define X vector (Marsh Eq. A3):
        if verbose: tic = time()
        X  = np.zeros(N * npoly, dtype=float)
        X0 = np.zeros(N * npoly, dtype=float)
        for q in qq:
            X[q] = (weightedE[Q_cols[0]:Q_cols[1]+1,:] * Qsm[kk[q],:,:] * jjnorm_pow[nn[q]]).sum() 
        if verbose: logger.debug('%1.2f s to compute X matrix' % (time() - tic))

        # Define C matrix (Marsh Eq. A3)
        if verbose: tic = time()
        C = np.zeros((N * npoly, N*npoly), dtype=float)

        buffer1 = 1.1 # C-matrix computation buffer (to be sure we don't miss any pixels)
        for p in pp:
            qp = Qsm[ll[p],:,:]
            for q in qq:
                #  Check that we need to compute C:
                if np.abs(kk[q] - ll[p]) <= (1./polyspacing + buffer1):
                    if q>=p: 
                        # Only compute over non-zero columns:
                        C[q, p] = (Qsm[kk[q],:,:] * qp * jjnorm_pow[nn[q]+mm[p]] * invEvariance_subset).sum() 
                    if q>p:
                        C[p, q] = C[q, p]


        if verbose: logger.debug('%1.2f s to compute C matrix' % (time() - tic))

        ##################################################
        ##################################################
        # Just for reference; the following is easier to read, perhaps, than the optimized code:
        if False: # The SLOW way to compute the X vector:
            X2 = np.zeros(N * npoly, dtype=float)
            for n in nn:
                for k in kk:
                    q = N * k + n
                    xtot = 0.
                    for i in ii:
                        for j in jj:
                            xtot += E[i,j] * Q[k,i,j] * (jjnorm[j]**n) / Evariance[i,j]
                    X2[q] = xtot

            # Compute *every* element of C (though most equal zero!)
            C = np.zeros((N * npoly, N*npoly), dtype=float)
            for p in pp:
                for q in qq:
                    if q>=p:
                        C[q, p] = (Q[kk[q],:,:] * Q[ll[p],:,:] * (jjnorm.reshape(1,1,nlam)**(nn[q]+mm[p])) / Evariance).sum()
                    if q>p:
                        C[p, q] = C[q, p]
        ##################################################
        ##################################################

        # Solve for the profile-polynomial coefficients (Marsh Eq. A)4: 
        if np.abs(np.linalg.det(C)) < 1e-10:
            Bsoln = np.dot(np.linalg.pinv(C), X)
        else:
            Bsoln = np.linalg.solve(C, X)

        Asoln = Bsoln.reshape(N, npoly).transpose()

        # Define G_kj, the profile-defining polynomial profiles (Marsh Eq. 8)
        Gsoln = np.zeros((npoly, nlam), dtype=float)
        for n in np.arange(npoly):
            Gsoln[n] = np.polyval(Asoln[n,::-1], jjnorm) # reorder polynomial coef.


        # Compute the profile (Marsh eq. 6) and normalize it:
        if verbose: tic = time()
        profile = np.zeros((fitwidth, nlam), dtype=float)
        for i in np.arange(fitwidth):
            profile[i,:] = (Q[:,i,:] * Gsoln).sum(0)

        #P = profile.copy() # for debugging 
        if profile.min() < 0:
            profile[profile < 0] = 0. 
        profile /= profile.sum(0).reshape(1, nlam)
        profile[1 - np.isfinite(profile)] = 0.
        if verbose: logger.debug('%1.2f s to compute profile' % (time() - tic))

        #Step6: Revise variance estimates 
        modelSpectrum = spectrum * profile.transpose()
        modelData = modelSpectrum + background
        variance0 = np.abs(modelData) + readnoise**2
        variance = variance0 / (goodpixelmask + 1e-9) # De-weight bad pixels, avoiding infinite variance

        outlierVariances = (frame - modelData)**2/variance

        if outlierVariances.max() > csigma**2:
            newBadPixels = True
            # Base our nreject-counting only on pixels within the spectral trace:
            maxRejectedValue = max(csigma**2, np.sort(outlierVariances[Qmask])[-nreject])
            worstOutliers = (outlierVariances>=maxRejectedValue).nonzero()
            goodpixelmask[worstOutliers] = False
            numberRejected = len(worstOutliers[0])
            #pdb.set_trace()
        else:
            newBadPixels = False
            numberRejected = 0
        
        if verbose: logger.debug("Rejected %i pixels on this iteration " % numberRejected)

            
        # Optimal Spectral Extraction: (Horne, Step 8)
        fixSkysubFrame = bfixpix(skysubFrame, True-goodpixelmask, n=8, retdat=True)
        spectrum       = np.zeros((nlam, 1), dtype=float)
        #spectrum1      = np.zeros((nlam, 1), dtype=float)
        varSpectrum    = np.zeros((nlam, 1), dtype=float)
        goodprof       =  profile.transpose() #* goodpixelmask
        for ii in np.arange(nlam):
            thisrow_good = extractionApertures[ii] #* goodpixelmask[ii]
            denom        = (goodprof[ii, thisrow_good] * profile.transpose()[ii, thisrow_good] / variance0[ii, thisrow_good]).sum()
            if denom==0:
                spectrum[ii]    = 0.
                varSpectrum[ii] = 9e9
            else:
                spectrum[ii]    = (goodprof[ii, thisrow_good] * skysubFrame[ii, thisrow_good] / variance0[ii, thisrow_good]).sum() / denom
                #spectrum1[ii]   = (goodprof[ii, thisrow_good] * modelSpectrum[ii, thisrow_good] / variance0[ii, thisrow_good]).sum() / denom
                varSpectrum[ii] = goodprof[ii, thisrow_good].sum() / denom
            #if spectrum.size==1218 and ii>610:
            #    pdb.set_trace()

        #if spectrum.size==1218: pdb.set_trace()

    ret             = baseObject()
    ret.spectrum    = spectrum
    ret.raw         = standardSpectrum
    ret.varSpectrum = varSpectrum
    ret.trace       = trace
    ret.units       = 'electrons'
    ret.background  = background_at_trace

    ret.function_name = 'spec.superExtract'

    if retall:
        ret.profile_map         = profile
        ret.extractionApertures = extractionApertures
        ret.background_map      = background
        ret.variance_map        = variance0
        ret.goodpixelmask       = goodpixelmask
        ret.function_args       = args
        ret.function_kw         = kw

    return ret


