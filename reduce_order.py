import logging
import numpy as np
import scipy.stats
import scipy.optimize
#import scipy.ndimage
import coloredlogs, verboselogs

import config
import image_lib
import nirspec_lib
import wavelength_utils
import Line
import DrpException

from astropy.visualization import ZScaleInterval, ImageNormalize
import fixpix

import CAT_Functions as cat
import matplotlib.pyplot as plt 
import nirspec_constants

verboselogs.install()
logger = logging.getLogger('obj')
#logger = verboselogs.VerboseLogger('obj')

def reduce_order(order, eta=None, arc=None):
        
    #print('ETA BEGINNING', eta)
    #print('REDUCE ORDER BEGINNING', order.flatOrder.orderNum)
    #if order.flatOrder.orderNum != 33: return 0
    #if order.flatOrder.orderNum != 68: return 0
    #if order.flatOrder.orderNum != 77: return 0
    #sys.exit()

    # flatten object images for this order
    __flatten(order, eta=eta, arc=arc)

    ### TEST PLOT XXX
    '''
    from astropy.visualization import ZScaleInterval, ImageNormalize
    plt.figure(198)
    norm = ImageNormalize(order.ffEtaImg, interval=ZScaleInterval())
    plt.imshow(order.ffEtaImg, origin='lower')
    plt.show()
    #sys.exit()
    '''
    ### TEST PLOT XXX

    ### XXX TESTING AREA
    
    if config.params['no_clean']:
        logger.info("bad pixel rejection on object frame/order inhibited by command line flag")

    else:
        for frame in order.frames:
            if frame == 'AB': continue # Don't need to do this one
            logger.info('bad pixel cleaning object frame %s, order %s'%(frame, order.flatOrder.orderNum))
            order.ffObjImg[frame] = fixpix.fixpix_rs(order.ffObjImg[frame])
            logger.debug('bad pixel cleaning object frame %s, order %s complete'%(frame, order.flatOrder.orderNum))
    
        if eta is not None:
            logger.info('bad pixel cleaning etalon frame %s, order %s'%(frame, order.flatOrder.orderNum))
            order.ffEtaImg = fixpix.fixpix_rs(order.ffEtaImg)
            logger.debug('bad pixel cleaning etalon frame %s, order %s complete'%(frame, order.flatOrder.orderNum))

        if arc is not None:
            logger.info('bad pixel cleaning arc lamp frame %s, order %s'%(frame, order.flatOrder.orderNum))
            order.ffArcImg = fixpix.fixpix_rs(order.ffArcImg)
            logger.debug('bad pixel cleaning arc lamp frame %s, order %s complete'%(frame, order.flatOrder.orderNum))
    
    ### XXX TESTING AREA

    '''
    #plt.figure(3)
    #norm = ImageNormalize(order.ffEtaImg, interval=ZScaleInterval())
    #plt.imshow(order.ffEtaImg, origin='lower', norm=norm, aspect='auto')
    #np.save('unrect_%s.npy'%order.flatOrder.orderNum, order.ffEtaImg) 
    #plt.savefig('%s_unrect.png'%order.flatOrder.orderNum, dpi=600, bbox_inches='tight')
    plt.figure(33)
    norm = ImageNormalize(order.ffObjImg['A'], interval=ZScaleInterval())
    plt.imshow(order.ffObjImg['A'], origin='lower', aspect='auto', norm=norm)
    #np.save('unrect_%s.npy'%order.flatOrder.orderNum, order.ffEtaImg) 
    #plt.savefig('%s_unrect.png'%order.flatOrder.orderNum, dpi=600, bbox_inches='tight')
    plt.figure(333)
    norm = ImageNormalize(order.ffObjImg['B'], interval=ZScaleInterval())
    plt.imshow(order.ffObjImg['B'], origin='lower', aspect='auto', norm=norm)
    plt.show(block=True)
    '''

    # rectify obj and flattened obj in spatial dimension
    __rectify_spatial(order, eta=eta, arc=arc)
 
    '''
    print('SHAPE:', order.ffObjImg['A'].shape, order.ffEtaImg.shape)
    plt.figure(3)
    norm = ImageNormalize(order.ffEtaImg, interval=ZScaleInterval())
    plt.imshow(order.ffEtaImg, origin='lower', norm=norm, aspect='auto')
    #np.save('unrect_%s.npy'%order.flatOrder.orderNum, order.ffEtaImg) 
    #plt.savefig('%s_unrect.png'%order.flatOrder.orderNum, dpi=600, bbox_inches='tight')
    '''

    # trim rectified order
    __trim(order, eta=eta, arc=arc)

    '''
    from astropy.visualization import ZScaleInterval, ImageNormalize, SquaredStretch
    norm = ImageNormalize(order.ffObjImg['A'], interval=ZScaleInterval(), stretch=SquaredStretch())
    plt.figure(1)
    plt.imshow(order.ffObjImg['A'], origin='lower', aspect='auto', norm=norm)

    norm = ImageNormalize(order.ffObjImg['B'], interval=ZScaleInterval(), stretch=SquaredStretch())
    plt.figure(2)
    plt.imshow(order.ffObjImg['B'], origin='lower', aspect='auto', norm=norm)

    plt.figure(3)
    plt.plot(np.sum(order.ffObjImg['A'], axis=1))

    plt.figure(4)
    plt.plot(np.sum(order.ffObjImg['B'], axis=1))

    plt.show()
    #sys.exit()
    '''

    '''
    plt.figure(4)
    print('SHAPE:', order.ffObjImg['A'].shape, order.ffEtaImg.shape)
    norm = ImageNormalize(order.ffEtaImg, interval=ZScaleInterval())
    plt.imshow(order.ffEtaImg, origin='lower', norm=norm, aspect='auto')
    #np.save('unrect_%s.npy'%order.flatOrder.orderNum, order.ffEtaImg) 
    #plt.savefig('%s_unrect.png'%order.flatOrder.orderNum, dpi=600, bbox_inches='tight')
    plt.figure(44)
    norm = ImageNormalize(order.ffObjImg['A'], interval=ZScaleInterval())
    plt.imshow(order.ffObjImg['A'], origin='lower', aspect='auto', norm=norm)
    #np.save('unrect_%s.npy'%order.flatOrder.orderNum, order.ffEtaImg) 
    #plt.savefig('%s_unrect.png'%order.flatOrder.orderNum, dpi=600, bbox_inches='tight')
    plt.figure(444)
    norm = ImageNormalize(order.ffObjImg['B'], interval=ZScaleInterval())
    plt.imshow(order.ffObjImg['B'], origin='lower', aspect='auto', norm=norm)
    plt.show(block=True)
    sys.exit()
    '''

    # save spatially rectified images before spectral rectify for diagnostics 
    # if AB pair then subtract B from A
    if order.isPair:
        order.objImg['AB']   = np.subtract(order.objImg['A'], order.objImg['B'])
        order.ffObjImg['AB'] = np.subtract(order.ffObjImg['A'], order.ffObjImg['B'])
        # reFlatten
        if np.amin(order.ffObjImg['AB']) < 0: order.ffObjImg['AB'] -= np.amin(order.ffObjImg['AB'])
        
    order.srNormFlatImg = order.flatOrder.rectFlatImg
    for frame in order.frames:
        order.srFfObjImg[frame] = order.ffObjImg[frame]

        if eta is not None:
            if frame == 'B':
                order.srNormEtaImgB = order.ffEtaImgB
            else:
                order.srNormEtaImg  = order.ffEtaImg

        if arc is not None:
            if frame == 'B':
                order.srNormArcImgB = order.ffArcImgB
            else:
                order.srNormArcImg  = order.ffArcImg
    
    # find spatial profile and peak
    __find_spatial_profile_and_peak(order)
    
    # characterize spatial profile by fitting to Gaussian
    __characterize_spatial_profile(order)

    # Try to find the spectral trace using etalon lamps if provided
    for frame in order.frames:
        #print('FRAME', frame)
        if frame == 'AB': 
            continue # Build the AB frame using A and B later
        if frame in ['A']:
            if eta is not None: # Do spectral rectification using the etalon lamps
                try:           
                    logger.info('attempting rectification of frame {} order {} in spectral dimension (etalon)'.format(
                                frame, order.flatOrder.orderNum))
                    order.spectralTrace[frame] = nirspec_lib.smooth_spectral_trace(
                                                 nirspec_lib.find_spectral_trace(order.ffEtaImg, eta=eta), 
                                                 order.ffObjImg['A'].shape[0])
                except Exception as e:
                    logger.warning('not rectifying frame {} order {} in spectral dimension (etalon)'.format(
                                   frame, order.flatOrder.orderNum))

            elif arc is not None: # Do spectral rectification using the arc lamps
                try:           
                    logger.info('attempting rectification of frame {} order {} in spectral dimension (arc lamp)'.format(
                                frame, order.flatOrder.orderNum))
                    order.spectralTrace[frame] = nirspec_lib.smooth_spectral_trace(
                                                 nirspec_lib.find_spectral_trace(order.ffArcImg, arc=arc), 
                                                 order.ffObjImg['A'].shape[0])
                except Exception as e:
                    logger.warning('not rectifying frame {} order {} in spectral dimension (arc lamp)'.format(
                                   frame, order.flatOrder.orderNum))

            else: # no etalons
                try:
                    logger.info('attempting rectification of frame {} order {} in spectral dimension'.format(
                                frame, order.flatOrder.orderNum))
                    order.spectralTrace[frame] = nirspec_lib.smooth_spectral_trace(
                                                 nirspec_lib.find_spectral_trace(order.ffObjImg['A']), 
                                                 order.ffObjImg['A'].shape[0])
                except Exception as e:
                    logger.warning('not rectifying frame {} order {} in spectral dimension'.format(
                                   frame, order.flatOrder.orderNum))

        else: # B frame
            if eta is not None: # Do spectral rectification using the etalon lamps
                try:           
                    logger.info('attempting rectification of frame {} order {} in spectral dimension (etalon)'.format(
                                frame, order.flatOrder.orderNum))
                    order.spectralTrace[frame] = nirspec_lib.smooth_spectral_trace(
                                                 nirspec_lib.find_spectral_trace(order.ffEtaImgB, eta=eta), 
                                                 order.ffObjImg['B'].shape[0])
                except Exception as e:
                    logger.warning('not rectifying frame {} order {} in spectral dimension (etalon)'.format(
                                   frame, order.flatOrder.orderNum))

            elif arc is not None: # Do spectral rectification using the arc lamps
                try:           
                    logger.info('attempting rectification of frame {} order {} in spectral dimension (arc lamps)'.format(
                                frame, order.flatOrder.orderNum))
                    order.spectralTrace[frame] = nirspec_lib.smooth_spectral_trace(
                                                 nirspec_lib.find_spectral_trace(order.ffArcImgB, arc=arc), 
                                                 order.ffObjImg['B'].shape[0])
                except Exception as e:
                    logger.warning('not rectifying frame {} order {} in spectral dimension (arc lamps)'.format(
                                   frame, order.flatOrder.orderNum))

            else: # no etalons
                try:
                    logger.info('attempting rectification of frame {} order {} in spectral dimension'.format(
                        frame, order.flatOrder.orderNum))
                    order.spectralTrace[frame] = nirspec_lib.smooth_spectral_trace(
                            nirspec_lib.find_spectral_trace(
                                    order.ffObjImg['B']), order.ffObjImg['B'].shape[0])
                except Exception as e:
                    logger.warning('not rectifying frame {} order {} in spectral dimension'.format(
                                   frame, order.flatOrder.orderNum))
    try: 
        __rectify_spectral(order, eta=eta, arc=arc)
    except:
        logger.warning('not able to rectify all of order {} in spectral dimension'.format(
                                   order.flatOrder.orderNum))

    # if AB pair then subtract B from A
    if order.isPair:
        order.objImg['AB']   = np.subtract(order.objImg['A'], order.objImg['B'])
        order.ffObjImg['AB'] = np.subtract(order.ffObjImg['A'], order.ffObjImg['B'])
        # reFlatten
        if np.amin(order.ffObjImg['AB']) < 0: order.ffObjImg['AB'] -= np.amin(order.ffObjImg['AB'])
    

    # TEST PLOT XXX
    '''
    from skimage import exposure
    plt.figure(666, figsize=(10,6))
    #plt.imshow(order.ffEtaImg, origin='lower')
    plt.imshow(exposure.equalize_hist(order.ffArcImg), origin='lower', aspect='auto')
    #np.save('rect_%s.npy'%order.flatOrder.orderNum, order.ffEtaImg) 
    #plt.savefig('%s_rect.png'%order.flatOrder.orderNum, dpi=600, bbox_inches='tight')
    plt.show()
    #sys.exit()
    '''
    # TEST PLOT XXX
    

    # compute noise image
    for frame in order.frames:
        order.noiseImg[frame] = nirspec_lib.calc_noise_img(
                order.objImg[frame], order.flatOrder.rectFlatImg, order.integrationTime)

    # extract spectra
    __extract_spectra(order, eta=eta, arc=arc)
            
    # calculate approximate SNR
    __calc_approximate_snr(order)
            
    # find and identify sky/etalon/arc lines   
    line_pairs = None # line_pairs are (column number, accepted wavelength)
    try:
        if eta is not None:
            etalon_wavelengths, etalon_intensities = wavelength_utils.get_etalon_lines()
        elif arc is not None:
            arclamp_wavelengths, arclamp_intensities = wavelength_utils.get_arclamp_lines()
        else:
            oh_wavelengths, oh_intensities = wavelength_utils.get_oh_lines()
    except IOError as e:
        logger.critical('cannot read OH/etalon/arc line file: ' + str(e))
        raise
        

    try:
        # synthesize sky/etalon/arc spectrum and store in order object
        if eta is not None:
            order.synthesizedSkySpec = wavelength_utils.synthesize_sky(
                    etalon_wavelengths, etalon_intensities, order.flatOrder.gratingEqWaveScale, eta=eta)
         
            # identify lines and return list of (column number, accepted wavelength) tuples
            line_pairs = wavelength_utils.line_id(order, etalon_wavelengths, etalon_intensities, eta=eta)


        elif arc is not None:
            order.synthesizedSkySpec = wavelength_utils.synthesize_sky(
                    arclamp_wavelengths, arclamp_intensities, order.flatOrder.gratingEqWaveScale, arc=arc)
         
            # identify lines and return list of (column number, accepted wavelength) tuples
            line_pairs = wavelength_utils.line_id(order, arclamp_wavelengths, arclamp_intensities, arc=arc)

        else:
            order.synthesizedSkySpec = wavelength_utils.synthesize_sky(
                    oh_wavelengths, oh_intensities, order.flatOrder.gratingEqWaveScale)
         
            # identify lines and return list of (column number, accepted wavelength) tuples
            line_pairs = wavelength_utils.line_id(order, oh_wavelengths, oh_intensities)
        
    except (IOError, ValueError) as e:
        logger.warning('sky/etalon/arc line matching failed: ' + str(e))
        
    if line_pairs is not None:
        
        logger.success(str(len(line_pairs)) + ' matched sky/etalon/arc lines found in order')

        # add line pairs to Order object as Line objects
        if nirspec_constants.upgrade:
            endPix = 2048
        else:
            endPix = 1024

        for line_pair in line_pairs:
            col, waveAccepted = line_pair
            peak = order.skySpec['A'][col]
            cent = image_lib.centroid(order.skySpec['A'], endPix, 5, col)
            line = Line.Line(order.baseNames['A'], order.flatOrder.orderNum, 
                    waveAccepted, col, cent, peak)
            order.lines.append(line)
            
        nlinelimit = 3
        if arc is not None: nlinelimit = 2 # Some orders have very few lines

        if len(order.lines) >= nlinelimit:
            # do local wavelength fit
            measured = []
            accepted = []
            for line in order.lines:
                measured.append(order.flatOrder.gratingEqWaveScale[line.col])
                accepted.append(line.waveAccepted)
            (order.orderCalSlope, order.orderCalIncpt, order.orderCalCorrCoeff, p, e) = \
                    scipy.stats.linregress(np.array(measured), np.array(accepted))  
            order.orderCalNLines = len(order.lines)       
            logger.info('per order wavelength fit: n = {}, a = {:.6f}, b = {:.6f}, r = {:.6f}'.format(
                    len(order.lines), order.orderCalIncpt, order.orderCalSlope, 
                    order.orderCalCorrCoeff))

            for line in order.lines:
                line.orderWaveFit = order.orderCalIncpt + \
                    (order.orderCalSlope * order.flatOrder.gratingEqWaveScale[line.col])    
                line.orderFitRes = abs(line.orderWaveFit - line.waveAccepted)  
                line.orderFitSlope = (order.orderCalSlope * 
                        (order.flatOrder.gratingEqWaveScale[endPix-1] - 
                         order.flatOrder.gratingEqWaveScale[0]))/float(endPix)
    else:
        logger.warning('not enough matched sky/etalon/arc lines in order ' + str(order.flatOrder.orderNum))
        order.orderCal = False 

    # TEST PLOT FLAT XXX
    '''
    from skimage import exposure
    plt.figure(figsize=(10,6))
    plt.imshow(exposure.equalize_hist(order.flatOrder.cutout), aspect='auto')
    plt.show()
    '''
    # TEST PLOT FLAT XXX
                        
    return



def __flatten(order, eta=None, arc=None):
    """Flat field object image[s] but keep originals for noise calculation.
    """
    
    for frame in order.frames:
        
        order.objImg[frame]   = np.array(order.objCutout[frame]) 
        order.ffObjImg[frame] = np.array(order.objCutout[frame] / order.flatOrder.normFlatImg)

        #Also cut out the flat fielded object
        order.ffObjCutout[frame] = np.array(image_lib.cut_out(order.ffObjImg[frame], 
                    order.flatOrder.highestPoint, order.flatOrder.lowestPoint, order.flatOrder.cutoutPadding))
        # Add then mask it
        order.ffObjCutout[frame] = np.ma.masked_array(order.objCutout[frame], mask=order.flatOrder.offOrderMask)
        
        if frame != 'AB':
            if np.amin(order.ffObjImg[frame]) < 0:
                order.ffObjImg[frame] -= np.amin(order.ffObjImg[frame])

        if eta is not None:
            if frame == 'B':
                order.etaImgB   = np.array(order.etaCutout) 
                order.ffEtaImgB = np.array(order.etaCutout / order.flatOrder.normFlatImg)
            else:
                order.etaImg   = np.array(order.etaCutout) 
                order.ffEtaImg = np.array(order.etaCutout / order.flatOrder.normFlatImg)

        if arc is not None:
            if frame == 'B':
                order.arcImgB   = np.array(order.arcCutout) 
                order.ffArcImgB = np.array(order.arcCutout / order.flatOrder.normFlatImg)
            else:
                order.arcImg   = np.array(order.arcCutout) 
                order.ffArcImg = np.array(order.arcCutout / order.flatOrder.normFlatImg)

    
    order.flattened = True
    logger.info('order has been flat fielded')
    return



def __rectify_spatial(order, eta=None, arc=None):
    """
    """     
    
    for frame in order.frames:
        if frame == 'AB': continue # Skip the AB frame, we will subtract them after rectification

        logger.info('attempting spatial rectification of frame {} using object trace'.format(frame))

        try:
            if frame in ['A', 'B']:
                #print('FRAME', frame)
                if config.params['onoff'] == True and frame == 'B': 
                    order.objImg[frame]   = image_lib.rectify_spatial(order.objImg[frame], polyVals1)
                    order.ffObjImg[frame] = image_lib.rectify_spatial(order.ffObjImg[frame], polyVals2)
                    logger.info('frame {}, order {} rectified using object trace'.format(
                        frame, order.flatOrder.orderNum))
                
                elif config.params['spatial_rect_flat'] == True:

                    logger.info('order will be rectified in the spatial dimension using the median order trace')
                    order.objImg[frame] = image_lib.rectify_spatial(
                        order.objImg[frame], order.flatOrder.smoothedSpatialTrace)
                    order.ffObjImg[frame] = image_lib.rectify_spatial(
                        order.ffObjImg[frame], order.flatOrder.smoothedSpatialTrace)

                    logger.info('frame {}, order {} rectified in the spatial dimension using flat frame'.format(
                        frame, order.flatOrder.orderNum))

                else:
                    #polyVals1             = cat.CreateSpatialMap(order.objImg[frame])  
                    polyVals1             = cat.CreateSpatialMap(order.objCutout[frame])  
                    order.objImg[frame]   = image_lib.rectify_spatial(order.objImg[frame], polyVals1)
                    logger.info('frame {}, order {} rectified using object trace'.format(
                        frame, order.flatOrder.orderNum))
                    #polyVals2             = cat.CreateSpatialMap(order.ffObjImg[frame])  
                    polyVals2             = cat.CreateSpatialMap(order.ffObjCutout[frame])  
                    order.ffObjImg[frame] = image_lib.rectify_spatial(order.ffObjImg[frame], polyVals2)
                    logger.info('flat fielded frame {}, order {} rectified using object trace'.format(
                        frame, order.flatOrder.orderNum))

                if eta is not None:
                    if frame == 'B':
                        if config.params['onoff'] == True:
                            order.etaImgB     = order.etaImg
                            order.ffEtaImgB   = order.ffEtaImg
                        else:
                            order.etaImgB     = image_lib.rectify_spatial(order.etaImgB, polyVals1)
                            order.ffEtaImgB   = image_lib.rectify_spatial(order.ffEtaImgB, polyVals2)
                            logger.info('etalon frame {}, order {} rectified using object trace'.format(
                                frame, order.flatOrder.orderNum))

                    else:
                        order.etaImg      = image_lib.rectify_spatial(order.etaImg, polyVals1)
                        order.ffEtaImg    = image_lib.rectify_spatial(order.ffEtaImg, polyVals2)
                        logger.info('etalon frame {}, order {} rectified using object trace'.format(
                            frame, order.flatOrder.orderNum))

                if arc is not None:
                    if frame == 'B':
                        if config.params['onoff'] == True:
                            order.arcImgB     = order.arcImg
                            order.ffArcImgB   = order.ffArcImg
                        else:
                            order.arcImgB     = image_lib.rectify_spatial(order.arcImgB, polyVals1)
                            order.ffArcImgB   = image_lib.rectify_spatial(order.ffArcImgB, polyVals2)
                            logger.info('arc lamp frame {}, order {} rectified using object trace'.format(
                                frame, order.flatOrder.orderNum))

                    else:
                        order.arcImg      = image_lib.rectify_spatial(order.arcImg, polyVals1)
                        order.ffArcImg    = image_lib.rectify_spatial(order.ffArcImg, polyVals2)
                        logger.info('arc lamp frame {}, order {} rectified using object trace'.format(
                            frame, order.flatOrder.orderNum))
            else:
                order.objImg[frame]   = image_lib.rectify_spatial(order.objImg[frame], polyVals1)
                order.ffObjImg[frame] = image_lib.rectify_spatial(order.ffObjImg[frame], polyVals2)

        except:
            logger.warning('could not rectify using object trace, falling back to edge trace')
            order.objImg[frame]   = image_lib.rectify_spatial(order.objImg[frame], 
                                                              order.flatOrder.smoothedSpatialTrace)
            order.ffObjImg[frame] = image_lib.rectify_spatial(order.ffObjImg[frame], 
                                                              order.flatOrder.smoothedSpatialTrace)
            if eta is not None:
                if frame == 'B':
                    order.etaImgB     = image_lib.rectify_spatial(order.etaImgB, 
                                                                  order.flatOrder.smoothedSpatialTrace)
                    order.ffEtaImgB   = image_lib.rectify_spatial(order.ffEtaImgB, 
                                                                  order.flatOrder.smoothedSpatialTrace)

                else:
                    order.etaImg      = image_lib.rectify_spatial(order.etaImg, 
                                                                  order.flatOrder.smoothedSpatialTrace)
                    order.ffEtaImg    = image_lib.rectify_spatial(order.ffEtaImg, 
                                                                  order.flatOrder.smoothedSpatialTrace)
            if arc is not None:
                if frame == 'B':
                    order.arcImgB     = image_lib.rectify_spatial(order.arcImgB, 
                                                                  order.flatOrder.smoothedSpatialTrace)
                    order.ffArcImgB   = image_lib.rectify_spatial(order.ffArcImgB, 
                                                                  order.flatOrder.smoothedSpatialTrace)

                else:
                    order.arcImg      = image_lib.rectify_spatial(order.arcImg, 
                                                                  order.flatOrder.smoothedSpatialTrace)
                    order.ffArcImg    = image_lib.rectify_spatial(order.ffArcImg, 
                                                                  order.flatOrder.smoothedSpatialTrace)


    order.spatialRectified = True
    logger.success('order has been rectified in the spatial dimension')
        
    return   
 

    
def __trim(order, eta=None, arc=None):
    """
    """
    for frame in order.frames:
        if frame == 'AB': continue
        order.objImg[frame]   = order.objImg[frame][order.flatOrder.botTrim:order.flatOrder.topTrim, :]
        order.ffObjImg[frame] = order.ffObjImg[frame][order.flatOrder.botTrim:order.flatOrder.topTrim, :]

        if eta is not None:
            if frame == 'B':
                order.etaImgB   = order.etaImgB[order.flatOrder.botTrim:order.flatOrder.topTrim, :]
                order.ffEtaImgB = order.ffEtaImgB[order.flatOrder.botTrim:order.flatOrder.topTrim, :]
            else:
                order.etaImg    = order.etaImg[order.flatOrder.botTrim:order.flatOrder.topTrim, :]
                order.ffEtaImg  = order.ffEtaImg[order.flatOrder.botTrim:order.flatOrder.topTrim, :]

        if arc is not None:
            if frame == 'B':
                order.arcImgB   = order.arcImgB[order.flatOrder.botTrim:order.flatOrder.topTrim, :]
                order.ffArcImgB = order.ffArcImgB[order.flatOrder.botTrim:order.flatOrder.topTrim, :]
            else:
                order.arcImg    = order.arcImg[order.flatOrder.botTrim:order.flatOrder.topTrim, :]
                order.ffArcImg  = order.ffArcImg[order.flatOrder.botTrim:order.flatOrder.topTrim, :]
        
    return



def __rectify_spectral(order, eta=None, arc=None):
    """
    """   
    for frame in order.frames:
        #print('FRAME', frame)
        if frame == 'AB': continue
        if config.params['onoff'] == True and frame == 'B':
            order.objImg[frame], peak1   = image_lib.rectify_spectral(order.objImg[frame], order.spectralTrace['A'], returnpeak=True)
            order.ffObjImg[frame], peak2 = image_lib.rectify_spectral(order.ffObjImg[frame], order.spectralTrace['A'], returnpeak=True)
        else:
            order.objImg[frame], peak1   = image_lib.rectify_spectral(order.objImg[frame], order.spectralTrace[frame], returnpeak=True)
            order.ffObjImg[frame], peak2 = image_lib.rectify_spectral(order.ffObjImg[frame], order.spectralTrace[frame], returnpeak=True)

        if frame == 'A':
            order.flatOrder.rectFlatImg, peak0 = image_lib.rectify_spectral(order.flatOrder.rectFlatImg, order.spectralTrace[frame], returnpeak=True)

        if eta is not None:
            if frame == 'B':
                if config.params['onoff'] == True:
                    order.etaImgB   = order.etaImg
                    order.ffEtaImgB = iorder.ffEtaImg
                else:
                    order.etaImgB   = image_lib.rectify_spectral(order.etaImgB, order.spectralTrace[frame], peak1)
                    order.ffEtaImgB = image_lib.rectify_spectral(order.ffEtaImgB, order.spectralTrace[frame], peak2)
            else:
                order.etaImg    = image_lib.rectify_spectral(order.etaImg, order.spectralTrace[frame], peak1)
                order.ffEtaImg  = image_lib.rectify_spectral(order.ffEtaImg, order.spectralTrace[frame], peak2)

        if arc is not None:
            if frame == 'B':
                if config.params['onoff'] == True:
                    order.arcImgB   = order.arcImg
                    order.ffArcImgB = iorder.ffArcImg
                else:
                    order.arcImgB   = image_lib.rectify_spectral(order.arcImgB, order.spectralTrace[frame], peak1)
                    order.ffArcImgB = image_lib.rectify_spectral(order.ffArcImgB, order.spectralTrace[frame], peak2)
            else:
                order.arcImg    = image_lib.rectify_spectral(order.arcImg, order.spectralTrace[frame], peak1)
                order.ffArcImg  = image_lib.rectify_spectral(order.ffArcImg, order.spectralTrace[frame], peak2)
    
    return     


              
def __extract_spectra(order, eta=None, arc=None):
    
    if order.isPair:        
        # get object extraction range for AB
        order.objWindow['AB'], _, _ = \
                image_lib.get_extraction_ranges(order.objImg['AB'].shape[0], 
                order.peakLocation['AB'], config.params['obj_window'], None, None)   
                
        logger.info('frame AB extraction window width = {}'.format(str(len(order.objWindow['AB']))))

        # extract object spectrum from AB
        order.objSpec['AB'] = np.sum(order.ffObjImg['AB'][i, :] for i in order.objWindow['AB']) 
        
        frames = ['A', 'B']
    else:
        frames = ['A']
        
    for frame in frames:

        # get sky extraction ranges for A or A and B
        order.objWindow[frame], order.topSkyWindow[frame], order.botSkyWindow[frame] = \
                image_lib.get_extraction_ranges(order.objImg[frame].shape[0], 
                order.peakLocation[frame], config.params['obj_window'], 
                config.params['sky_window'], config.params['sky_separation'])


        
        ### TEST PLOT SKY EXTRACTION REGION XXX
        '''
        from skimage import exposure
        fig = plt.figure(3285)
        #plt.imshow(self.rectFlatImg, origin='lower', aspect='auto', norm=norm)
        plt.imshow(exposure.equalize_hist(order.objImg[frame]), origin='lower', aspect='auto')#, norm=norm)
        plt.axhline(np.min(order.objWindow[frame]), c='b', ls='-')
        plt.axhline(np.max(order.objWindow[frame]), c='b', ls='-')
        plt.axhline(np.min(order.topSkyWindow[frame]), c='r', ls='-')
        plt.axhline(np.max(order.topSkyWindow[frame]), c='r', ls='-')
        plt.axhline(np.min(order.botSkyWindow[frame]), c='r', ls=':')
        plt.axhline(np.max(order.botSkyWindow[frame]), c='r', ls=':')
        #plt.axhline(self.lowestPoint, c='r', ls='--')
        #plt.axhline(self.highestPoint, c='b', ls='--')
        fig.suptitle('Extraction Regions')
        plt.show()
        #sys.exit()
        '''
        ### TEST PLOT SKY EXTRACTION REGION XXX
        
                
        logger.info('frame {} extraction window width = {}'.format(
                frame, str(len(order.objWindow[frame]))))
        logger.info('frame {} top background window width = {}'.format(
                frame, str(len(order.topSkyWindow[frame]))))
        if len(order.topSkyWindow[frame]) > 0:
            logger.info('frame {} top background window separation = {}'.format(
                    frame, str(order.topSkyWindow[frame][0] - order.objWindow[frame][-1])))
        logger.info('frame {} bottom background window width = {}'.format(
                frame, str(len(order.botSkyWindow[frame]))))
        if len(order.botSkyWindow[frame]) > 0:
            logger.info('frame {} bottom background window separation = {}'.format(
                    frame, str(order.objWindow[frame][0] - order.botSkyWindow[frame][-1])))       
                
        if eta is not None:
            if frame == 'B':
                # extract object, sky, etalon, and noise spectra for A and B and flat spectrum
                order.objSpec[frame], order.flatSpec, order.etalonSpec, order.skySpec[frame], order.noiseSpec[frame], \
                        order.topBgMean[frame], order.botBgMean[frame] = image_lib.extract_spectra(
                                order.ffObjImg[frame], order.flatOrder.rectFlatImg, order.noiseImg[frame], 
                                order.objWindow[frame], order.topSkyWindow[frame], 
                                order.botSkyWindow[frame], eta=order.ffEtaImgB) 
            else:
                # extract object, sky, etalon, and noise spectra for A and B and flat spectrum
                order.objSpec[frame], order.flatSpec, order.etalonSpec, order.skySpec[frame], order.noiseSpec[frame], \
                        order.topBgMean[frame], order.botBgMean[frame] = image_lib.extract_spectra(
                                order.ffObjImg[frame], order.flatOrder.rectFlatImg, order.noiseImg[frame], 
                                order.objWindow[frame], order.topSkyWindow[frame], 
                                order.botSkyWindow[frame], eta=order.ffEtaImg) 
        
        elif arc is not None:
            if frame == 'B':
                # extract object, sky, arc, and noise spectra for A and B and flat spectrum
                order.objSpec[frame], order.flatSpec, order.arclampSpec, order.skySpec[frame], order.noiseSpec[frame], \
                        order.topBgMean[frame], order.botBgMean[frame] = image_lib.extract_spectra(
                                order.ffObjImg[frame], order.flatOrder.rectFlatImg, order.noiseImg[frame], 
                                order.objWindow[frame], order.topSkyWindow[frame], 
                                order.botSkyWindow[frame], arc=order.ffArcImgB) 
            else:
                # extract object, sky, arc, and noise spectra for A and B and flat spectrum
                order.objSpec[frame], order.flatSpec, order.arclampSpec, order.skySpec[frame], order.noiseSpec[frame], \
                        order.topBgMean[frame], order.botBgMean[frame] = image_lib.extract_spectra(
                                order.ffObjImg[frame], order.flatOrder.rectFlatImg, order.noiseImg[frame], 
                                order.objWindow[frame], order.topSkyWindow[frame], 
                                order.botSkyWindow[frame], arc=order.ffArcImg)

        else:
            # extract object, sky, and noise spectra for A and B and flat spectrum
            order.objSpec[frame], order.flatSpec, order.skySpec[frame], order.noiseSpec[frame], \
                    order.topBgMean[frame], order.botBgMean[frame] = image_lib.extract_spectra(
                            order.ffObjImg[frame], order.flatOrder.rectFlatImg, order.noiseImg[frame], 
                            order.objWindow[frame], order.topSkyWindow[frame], 
                            order.botSkyWindow[frame])  

    ### XXX TESTING AREA
    '''
    # Try the defringe filter
    import matplotlib.pyplot as plt
    from scipy import fftpack
    from scipy import signal

    for frame in frames:
        if order.flatOrder.orderNum == 33:

            #fit1 = np.polyfit(np.arange(len(order.objSpec['A'])), order.objSpec['A'], 10)
            #z1   = np.poly1d(fit1)

            #orderspec2 = order.objSpec[frame]# - z1(np.arange(len(order.objSpec['A'])))

            #plt.plot(order.objSpec[frame], 'C0')
            #plt.plot(z1(np.arange(len(order.objSpec['A']))), 'C1')


            """
            f, Pxx_den = signal.periodogram(orderspec2, fs=len(order.objSpec[frame]))
            plt.figure(2)
            plt.semilogy(f, Pxx_den)
            #plt.ylim([1e-7, 1e2])
            plt.xlabel('frequency [Hz]')
            plt.ylabel('PSD [V**2/Hz]')
            """
            ## IDL CODE
            xdim   = len(order.objSpec[frame])
            nfil   = xdim/2 + 1
            cal1   = order.objSpec[frame]
            """
            f_high = 130
            f_low  = 150
            freq   = np.arange(nfil/2+1) / (nfil / float(xdim))
            fil    = np.zeros(len(freq))
            fil[np.where((freq > f_low) | (freq < f_high))] = 1
            fil    = np.append(fil, np.flip(fil[1:]))
            fil    = fftpack.ifft(fil)
            fil    = fil/nfil
            fil    = np.roll(fil, nfil/2)
            fil    = fil*np.hanning(nfil)
            """

            ##
            """
            cal1orig    = order.objSpec[frame]
            freq        = np.arange(nfil)
            cal1origmed = np.median(cal1orig)
            cal1smooth  = scipy.ndimage.median_filter(cal1, size=30)
            cal1fft     = fftpack.rfft(cal1orig-cal1smooth)
            yp          = abs(cal1fft[0:nfil])**2
            ypmax       = np.max(yp)
            yp          = yp / ypmax
            plt.figure(1011)
            plt.plot(freq, yp)
            """
            ##
            """
            cal1smooth = scipy.ndimage.median_filter(cal1, size=30)
            cal2smooth = scipy.ndimage.median_filter(cal2, size=30)

            cal1       = np.convolve(cal1-cal1smooth, fil, mode='same') + cal1smooth
            cal2       = np.convolve(cal2-cal2smooth, fil, mode='same') + cal2smooth
            #cal2smooth = median(cal2,30)
            #cal2       = convol(cal2-cal2smooth,fil,/edge_wrap)+cal2smooth
            plt.figure(101)
            plt.plot(cal1)
            plt.figure(102)
            plt.plot(cal2)
            plt.figure(103)
            plt.plot(cal1-cal2)
            #plt.show()
            #sys.exit()
            """
            ## IDL CODE
            """
            W        = fftpack.fftfreq(order.objSpec[frame].size, d=1./1024)
            toplot   = np.fft.fftshift(W)
            f_signal = fftpack.rfft(order.objSpec[frame])
            W1       = np.arange(len(W))
            
            plt.figure(3)
            plt.plot(W1, f_signal)
            
            plt.figure(4)
            plt.plot(toplot, f_signal)
            """
            f_high   = 300
            f_low    = 260
            W        = fftpack.fftfreq(order.objSpec[frame].size, d=1./1024)
            fftval   = fftpack.rfft(order.objSpec[frame])
            fftval[np.where((W > f_low) & (W < f_high))] = 0
            newsig   = fftpack.irfft(fftval)  
            """
            plt.figure(5)
            plt.plot(newsig)
            plt.show()
            sys.exit()
            """
            order.objSpec[frame] = newsig
    
    '''
    ### XXX TESTING AREA
        
    if order.isPair:
        #order.noiseSpec['AB'] = order.noiseSpec['A'] + order.noiseSpec['B']
        order.noiseSpec['AB'] = np.sqrt(np.square(order.noiseSpec['A']) + np.square(order.noiseSpec['B']))
        order.skySpec['AB']   = order.skySpec['A'] + order.skySpec['B'] # XXX This is a dirty fix
        
            
    return



def __calc_approximate_snr(order):
    
    if order.isPair:
        frames = ['A', 'B']
    else:
        frames = ['A']
        
    for frame in frames:
        
        bg = 0.0
        
        if order.topBgMean[frame] is not None:
            bg += order.topBgMean[frame]
        if order.botBgMean[frame] is not None:
            bg += order.botBgMean[frame]
        if order.topBgMean[frame] is not None and order.botBgMean[frame] is not None:
            bg /= 2
            
        order.snr[frame] = np.absolute(
                np.mean(order.ffObjImg[frame]\
                        [order.peakLocation[frame] : order.peakLocation[frame] + 1, :]) / bg)   
        
        logger.info('frame {} signal-to-noise ratio = {:.1f}'.format(frame, order.snr[frame]))
        
    if order.isPair:
        order.snr['AB'] = 0.0
        if order.snr['A'] is not None:
            order.snr['AB'] = order.snr['A']
        if order.snr['B'] is not None:
            order.snr['AB'] = order.snr['AB'] + order.snr['B']
        if order.snr['A'] is not None and order.snr['B'] is not None:
            order.snr['AB'] = order.snr['AB'] / 2.0
        
    return
    


def __characterize_spatial_profile(order):
    
    for frame in order.frames:
        try:
            if frame == 'B' and config.params['onoff']  == True:
                logger.debug('using frame A window width = {}'.format(abs(order.gaussianParams['A'][2])))
                order.gaussianParams[frame] = order.gaussianParams['A']
            else:
                for w in range(10, 30, 10):
                    logger.debug('gaussian window width = {}'.format(2 * w))
                    x0 = max(0, order.peakLocation[frame] - w)
                    x1 = min(len(order.spatialProfile[frame]) - 1, order.peakLocation[frame] + w)
                    x  = range(x1 - x0)
                    order.gaussianParams[frame], pcov = scipy.optimize.curve_fit(
                            image_lib.gaussian, x, order.spatialProfile[frame][x0:x1] - \
                            np.amin(order.spatialProfile[frame][x0:x1]))
                    order.gaussianParams[frame][1] += x0
                    if order.gaussianParams[frame][2] > 1.0:
                        break
        except Exception as e:
            logger.warning('cannot fit frame {} spatial profile to Gaussian'.format(frame))
            order.gaussianParams[frame] = None
        else:
            logger.info('frame {} spatial peak width = {:.1f} pixels'.format(
                    frame, abs(order.gaussianParams[frame][2])))
        
    return



def __find_spatial_profile_and_peak(order):
    """
    """
    
    MARGIN = 5
    
    for frame in order.frames:
        
        # find spatial profile(s)
        if frame == 'B' and config.params['onoff'] == True:
            order.spatialProfile[frame] = order.spatialProfile['A']
            order.peakLocation[frame]   = order.peakLocation['A']
            order.centroid[frame]       = order.centroid['A']

        else:
            order.spatialProfile[frame] = order.ffObjImg[frame].mean(axis=1)

            if len(order.spatialProfile[frame]) < (2 * MARGIN) + 2:
                raise DrpException.DrpException(
                        'cannot find spatial profile for frame {} order {}'.format(
                        frame, order.flatOrder.orderNum))
                
            # find peak locations
            order.peakLocation[frame] = np.argmax(order.spatialProfile[frame][MARGIN:-MARGIN]) + MARGIN
            logger.info('frame {} spatial profile peak intensity row {:d}'.format(
                    frame, order.peakLocation[frame]))
        
            # fit peak to Gaussian, save Gaussian parameters and real centroid location
            p0 = order.peakLocation[frame] - (config.params['obj_window'] // 2)
            p1 = order.peakLocation[frame] + (config.params['obj_window'] // 2)
            order.centroid[frame] = (scipy.ndimage.measurements.center_of_mass(
                order.spatialProfile[frame][p0:p1]))[0] + p0 
            logger.info('frame {} spatial profile peak centroid row {:.1f}'.format(
                    frame, float(order.centroid[frame])))     
    
    return



def __calculate_SNR(order):
    
    bg = 0.0

    if order.topBgMean is not None:
        bg += order.topBgMean
    if order.botBgMean is not None:
        bg += order.botBgMean
    if order.topBgMean is not None and order.botBgMean is not None:
        bg /= 2
    order.snr = np.absolute(np.mean(
            order.ffObjImg['A'][order.peakLocation['A']:order.peakLocation['A'] + 1, :]) / bg)
    
    
    