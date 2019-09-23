import numpy as np
from astropy.io import fits
import DrpException
import nirspec_constants


class RawDataSet:
    """Represents a complete raw data set consisting an object frame or an object frame pair,
    one or more flats and zero or more darks.
    
    Objects of this class can include a single object frame or an AB pair.
    
    Args:
        objAFn (str): filename of A object frame
        
        objBFn (str): filename of B object frame
        
        objHeader (Header): FITS header to be associated with this data set, always from object A.
        
    Raises:
        DrpException: If either objAFn or objHeader is none

    """
    
    def __init__(self, objAFn, objBFn, objHeader, eta=None, arc=None, dark=None):

        if objAFn is None:
            raise DrpException('objAFn cannot be None')
        
        if objHeader is None:
            raise DrpException('header cannot be None')
            
        self.objAFn = objAFn
        self.objBFn = objBFn
        
        self.objHeader = objHeader
        
        self.baseNames = {}
        
        if self.objAFn is not None and self.objBFn is not None:
            self.isPair = True
            self.frames = ['A', 'B', 'AB']
            self.baseNames['A'] = self.__deriveBaseName(self.objAFn)
            self.baseNames['B'] = self.__deriveBaseName(self.objBFn)
            self.baseNames['AB'] = self.__combineBaseNames(
                    self.baseNames['A'], self.baseNames['B'])
        else:
            self.isPair = False
            self.frames = ['A']
            self.baseNames['A'] = self.__deriveBaseName(self.objAFn)

        
        self.flatFns = []
        self.darkFns = []
        self.etaFns  = []
        self.arcFns  = []

        if eta is not None:
            self.etaFns = eta

        if arc is not None:
            self.arcFns = arc

        if dark is not None:
            self.darkFns = [dark]
 
        
    def __deriveBaseName(self, fn):
        """Derives a base object file name to be used in data products.
        """
        if fn.find('NS.') == 0:
            # Filename is a KOA ID of the form path/NS.20100429.40400.fits.
            # A directory path prefix may or may not be present.
            # The file name extension may be .gz instead of .fits
            baseName = fn[fn.find('NS'):].rstrip('.gz').rstrip('.fits')
            
        else:
            # Filename is any valid filename with a .fits extension.
            # A directory path prefix may or may not be present.
            if fn.find('/') >= 0:
                i0 = fn.rfind('/') + 1
                i1 = fn.lower().find('.fits')
                baseName = fn[i0:i1]
            else:
                baseName = fn[:fn.lower().find('.fits')]
                
        return baseName
    
    
    def __combineBaseNames(self, A, B):
        
        if A.startswith('NS.') and B.startswith('NS.') and \
                A[3:].find('.') == 8 and B[3:].find('.') == 8:
            return A + '-' + B[12:]

        for i in range(len(A)):
            if A[i] != B[i]:
                break

        return A + '-' + B[i:]
        
                
    def isPair(self):
        return self.pair

    
#     def getObjFn(self):
#         return(self.objAFn)
# 
#     
#     def getObjAFn(self):
#         return(self.objAFn)
# 
# 
#     def getObjHeader(self):
#         return(self.objHeader)
 

    def getShape(self):
        return((self.objHeader['NAXIS1'], self.objHeader['NAXIS2']))
 
    
    def toString(self):
        return 'raw data set: fn={}, ut={}, disppos={}, echlpos={}, filname={}, slitname={}'.format(
                self.baseName, self.objHeader['UTC'], self.objHeader['disppos'], 
                self.objHeader['echlpos'], self.objHeader['filname'], self.objHeader['slitname'])

    
    def combineFlats(self): # Can't tell if this is being used, or if these functions are in Flat.py
        """Median combines flats and returns resulting image array
        """
        if len(self.flatFns) == 0:
            return None
        if len(self.flatFns) == 1:
            return(fits.getdata(self.flatFns[0], ignore_missing_end=True))
        flatDataList = []
        for fn in self.flatFns:
            flatDataList.append(fits.getdata(fn, ignore_missing_end=True))   
        return np.median(flatDataList, axis=0)

         
    def combineDarks(self): # I don't think this is being implemented yet      
        """Median combines darks and returns resulting image array
        """
        if len(self.darkFns) == 0:
            return None
        if len(self.darkFns) == 1:
            return(fits.getdata(self.darkFns[0], ignore_missing_end=True))
        if len(self.darkFns) > 1:
            darkData = []
            for fn in self.darkFns:
                darkData.append(fits.getdata(fn, ignore_missing_end=True))   
            return np.median(darkData, axis=0)


    def combineEtas(self):   # XXX This is not working yet    
        """Median combines Etalons and returns resulting image array
        """
        if len(self.etaFns) == 0:
            return None
        if len(self.etaFns) == 1:
            return(fits.getdata(self.etaFns[0], ignore_missing_end=True))
        if len(self.etaFns) > 1:
            etaData = []
            for fn in self.etaFns:
                etaData.append(fits.getdata(fn, ignore_missing_end=True))   
            return np.median(etaData, axis=0)


    def combineArcs(self):   # XXX This is not working yet    
        """Median combines Arc lamps and returns resulting image array
        """
        if len(self.arcFns) == 0:
            return None
        if len(self.arcFns) == 1:
            return(fits.getdata(self.arcFns[0], ignore_missing_end=True))
        if len(self.arcFns) > 1:
            arcData = []
            for fn in self.arcFns:
                arcData.append(fits.getdata(fn, ignore_missing_end=True))   
            return np.median(arcData, axis=0)
            
