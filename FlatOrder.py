import logging
import numpy as np
import image_lib
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval, ImageNormalize
import nirspec_constants
import config

#logger = logging.getLogger('obj')

class FlatOrder:
    """
    
    Top refers to high row numbers, longer wavelengths.
    Bottom refers to low row numbers, shorter wavelengths.
    LHS refers to left hand side of order, low column numbers, shorter wavelengths.
    """
    
    def __init__(self, baseName, orderNum, logger):
        
        self.flatBaseName       = baseName
        self.orderNum           = orderNum
        self.logger             = logger
        
        self.valid              = False
        
        self.topCalc            = None  # LHS top row of order, according to grating eq
        self.botCalc            = None  # LHS bottom row of order, according to grating eq
        self.gratingEqWaveScale = None  # wavelength scale, according to grating eq
        
        self.topMeas            = None  # measured LHS top row of order
        self.botMeas            = None  # measured LHS bottom row of order
        
        self.topEdgeTrace       = None # top edge trace
        self.botEdgeTrace       = None # bot edge trace
        self.avgEdgeTrace       = None

        self.longSlitEdgeMargin = 0
        self.cutoutPadding      = 0

        self.extraTrim          = 0
        
        self.highestPoint       = None
        self.lowestPoint        = None
        self.topTrim            = None
        self.botTrim            = None
                
        self.onOrderMask        = None
        self.offOrderMask       = None
        
        self.mean               = None
        self.median             = None
        
        self.cutout             = None
#         self.flatImg = None
        self.normFlatImg        = None
        self.rectFlatImg        = None
        
        self.normalized         = False
        self.spatialRectified   = False
        self.spectralRectified  = False

        self.smoothedSpatialTrace    = None
        self.spatialTraceMask        = None
        self.spatialTraceFitResidual = None
        
        
    def reduce(self):
        
        self.logger.info('reducing flat order {}'.format(self.orderNum))
        
        # normalize flat
        self.normFlatImg, self.median =  image_lib.normalize(
                self.cutout, self.onOrderMask, self.offOrderMask)
        self.normalized = True
        self.logger.info('flat normalized, flat median = ' + str(round(self.median, 1)))
        
        # spatially rectify flat
        self.rectFlatImg  = image_lib.rectify_spatial(self.normFlatImg, self.smoothedSpatialTrace)
        """
        self.rectFlatImgA = image_lib.rectify_spatial(self.normFlatImg, self.smoothedSpatialTraceA)
        self.rectFlatImgB = image_lib.rectify_spatial(self.normFlatImg, self.smoothedSpatialTraceB)
        """

        self.spatialRectified = True
        
        # compute top and bottom trim points
        self.calcTrimPoints()
        
        ### TESTING PLOT XXX
        '''
        from skimage import exposure
        #norm = ImageNormalize(self.rectFlatImg, interval=ZScaleInterval())

        fig = plt.figure(1985)
        #plt.imshow(self.rectFlatImg, origin='lower', aspect='auto', norm=norm)
        plt.imshow(self.cutout, origin='lower', aspect='auto')#, norm=norm)
        plt.axhline(self.botTrim, c='r', ls=':')
        plt.axhline(self.topTrim, c='b', ls=':')
        #plt.axhline(self.lowestPoint, c='r', ls='--')
        #plt.axhline(self.highestPoint, c='b', ls='--')
        fig.suptitle('Order: %s'%self.orderNum)

        fig = plt.figure(1986)
        #plt.imshow(self.rectFlatImg, origin='lower', aspect='auto', norm=norm)
        plt.imshow(self.normFlatImg, origin='lower', aspect='auto')#, norm=norm)
        plt.axhline(self.botTrim, c='r', ls=':')
        plt.axhline(self.topTrim, c='b', ls=':')
        #plt.axhline(self.lowestPoint, c='r', ls='--')
        #plt.axhline(self.highestPoint, c='b', ls='--')
        fig.suptitle('Order: %s'%self.orderNum)

        fig = plt.figure(1987)
        #plt.imshow(self.rectFlatImg, origin='lower', aspect='auto', norm=norm)
        plt.imshow(exposure.equalize_hist(self.rectFlatImg), origin='lower', aspect='auto')#, norm=norm)
        plt.axhline(self.botTrim, c='r', ls=':')
        plt.axhline(self.topTrim, c='b', ls=':')
        #plt.axhline(self.botTrim+10, c='r', ls='--')
        #plt.axhline(self.topTrim-10, c='b', ls='--')
        #plt.axhline(self.lowestPoint, c='r', ls='--')
        #plt.axhline(self.highestPoint, c='b', ls='--')
        fig.suptitle('Order: %s'%self.orderNum)
        print(self.cutoutPadding)
        print(self.highestPoint, self.lowestPoint, self.topEdgeTrace - self.botEdgeTrace)
        plt.show()
        sys.exit()
        '''
        ### TESTING PLOT XXX

        # trim rectified flat order images
        self.rectFlatImg  = self.rectFlatImg[self.botTrim:self.topTrim, :]
        #self.rectFlatImgA = self.rectFlatImgA[self.botTrimA:self.topTrimA, :]
        #self.rectFlatImgB = self.rectFlatImgB[self.botTrimB:self.topTrimB, :]
        
        self.logger.debug('reduction of flat order {} complete'.format(self.orderNum))
        
        return


    
    def calcTrimPoints(self):

        if self.lowestPoint > self.cutoutPadding:
            self.topTrim = self.highestPoint - self.lowestPoint + self.cutoutPadding - 3
        else:
            self.topTrim = self.highestPoint - 3
        h = np.amin(self.topEdgeTrace - self.botEdgeTrace) # Old way
        h = int(np.around(np.mean(self.topEdgeTrace - self.botEdgeTrace))) # New way

        if nirspec_constants.upgrade:
            endPix = 2048
        else:
            endPix = 1024

        self.botTrim = self.topTrim - h + 3
        self.botTrim = int(max(0, self.botTrim))
        self.topTrim = int(min(self.topTrim, endPix-1))

        # Trim a little more from the edges. Useful for overlapping edges.
        if config.params['extra_cutout']:
            self.botTrim += 10
            self.topTrim -= 10
        
        return


        
        
        
        