import os
import logging
from astropy.io import fits
import nirspec_constants
import RawDataSet
import create_raw_data_sets
import reduce_frame
import config
import products
import dgn
import DrpException
from datetime import datetime

def process_frame(fn1, fn2, obj_B_fn, out_dir, dark=None, eta=None, arc=None, override=False):

    logger = logging.getLogger('main')
   
    flat_fn    = None
    obj_fn     = None
    
    fn1_header = fits.PrimaryHDU.readfrom(fn1, ignore_missing_end=True).header
    fn2_header = fits.PrimaryHDU.readfrom(fn2, ignore_missing_end=True).header

    # Get the observation date to determine if this is upgraded NIRSPEC
    if datetime.strptime(fn1_header['date-obs'], '%Y-%m-%d') > datetime.strptime('2018-10-10', '%Y-%m-%d'):
        nirspec_constants.upgrade = True

    # Do the case for the etalon lamp
    if eta is not None:
        eta_header = fits.PrimaryHDU.readfrom(eta, ignore_missing_end=True).header
        if nirspec_constants.upgrade:
            if eta_header['etalon'] == 'On' and eta_header['calmpos'] == 'In':
                eta_fn = eta
        else:
            if eta_header['etalon'] == 1 and eta_header['calmpos'] == 1:
                eta_fn = eta

    # Do the case for the arc lamp
    if arc is not None:
        arc_header = fits.PrimaryHDU.readfrom(arc, ignore_missing_end=True).header
        if nirspec_constants.upgrade:
            if (arc_header['neon'] == 'On' or arc_header['argon'] == 'On' or arc_header['krypton'] == 'On' or \
                arc_header['xenon'] == 'On') and arc_header['calmpos'] == 'In':
                arc_fn = arc
        else:
            if (arc_header['neon'] == 1 or arc_header['argon'] == 1 or arc_header['krypton'] == 1 or \
                arc_header['xenon'] == 1) and arc_header['calmpos'] == 1:
                arc_fn = arc

    # Get the master dark frame if given
    if dark is not None:
        dark_header = fits.PrimaryHDU.readfrom(dark, ignore_missing_end=True).header
        if nirspec_constants.upgrade:
            if dark_header['halogen'] == 'On' and dark_header['calmpos'] == 'In' and \
               dark_header['neon'] != 'On' and dark_header['argon'] != 'On' and dark_header['krypton'] != 'On' and \
               dark_header['xenon'] != 'On':
                dark_fn = dark
        else:
            if dark_header['flat'] == 0 and dark_header['calmpos'] == 1 and \
               dark_header['neon'] != 1 and dark_header['argon'] != 1 and dark_header['krypton'] != 1 and \
               dark_header['xenon'] != 1:
                dark_fn = dark

    # Get the flat and object fits
    if nirspec_constants.upgrade:
        if fn1_header['halogen'] == 'On' and fn1_header['calmpos'] == 'In':
            flat_fn     = fn1
            obj_fn      = fn2
            obj_header  = fn2_header
            flat_header = fn1_header
        if fn1_header['halogen'] == 'On' and fn1_header['calmpos'] == 'In':
            flat_fn     = fn1
            obj_fn      = fn2
            obj_header  = fn2_header
            flat_header = fn1_header
    else:
        if fn1_header['flat'] == 1 and fn1_header['calmpos'] == 1:
            flat_fn     = fn1
            obj_fn      = fn2
            obj_header  = fn2_header
            flat_header = fn1_header
        if fn2_header['flat'] == 1 and fn2_header['calmpos'] == 1:
            if flat_fn is not None:
                raise DrpException.DrpException('two flats, no object frame')
            else:
                flat_fn     = fn2
                obj_fn      = fn1
                obj_header  = fn1_header
                flat_header = fn2_header
    if flat_fn is None:
        raise DrpException.DrpException('no flat')

    if obj_header['ECHLPOS'] > 100:
        print('ERROR: cannot reduce low-resolution image (ECHLPOS > 100')
        exit(1)
        
    # Check the array size is correct
    if nirspec_constants.upgrade:
        if obj_header['NAXIS1'] != nirspec_constants.N_COLS_upgrade:
            raise DrpException.DrpException('NAXIS1 != {}'.format(nirspec_constants.N_COLS_upgrade))
        if obj_header['NAXIS2'] != nirspec_constants.N_ROWS_upgrade:
            raise DrpException.DrpException('NAXIS2 != {}'.format(nirspec_constants.N_COLS_upgrade))
        filtername1, filtername2 = obj_header['SCIFILT2'], ''
        if obj_header['SCIFILT1'] == 'AO-stop': filtername2 = '-AO'
        if filtername1.upper()+filtername2.upper() not in nirspec_constants.filter_names:
            raise DrpException.DrpException('unsupported filter: {}'.format(obj_header['SCIFILT2']))
    else:
        if obj_header['NAXIS1'] != nirspec_constants.N_COLS:
            raise DrpException.DrpException('NAXIS1 != {}'.format(nirspec_constants.N_COLS))
        if obj_header['NAXIS2'] != nirspec_constants.N_ROWS:
            raise DrpException.DrpException('NAXIS2 != {}'.format(nirspec_constants.N_COLS))
        if obj_header['FILNAME'].upper() not in nirspec_constants.filter_names:
            raise DrpException.DrpException('unsupported filter: {}'.format(obj_header['FILNAME']))
    
    if create_raw_data_sets.flat_criteria_met(obj_header, flat_header, ignore_dispers=True) is False:
        raise DrpException.DrpException('flat is not compatible with object frame')

    if dark is not None:
        if create_raw_data_sets.dark_criteria_met(obj_header, dark_header) is False:
            raise DrpException.DrpException('dark is not compatible with object frame')
    
    if obj_B_fn is not None:
        # confirm that A and B are not the same files
        if obj_fn == obj_B_fn:
            raise DrpException.DrpException('frames A and B are the same')
        
        obj_B_header = fits.PrimaryHDU.readfrom(obj_B_fn, ignore_missing_end=True).header
        if create_raw_data_sets.is_valid_pair(obj_header, obj_B_header, override=override):
            logger.debug('Reducing AB pair, A= ' + obj_fn + ' , B= ' + obj_B_fn)
            rawDataSet = RawDataSet.RawDataSet(obj_fn, obj_B_fn, obj_header, eta=eta, arc=arc, dark=dark)

        else:
            raise DrpException.DrpException('frames A and B are not a valid pair')
    else:
        rawDataSet = RawDataSet.RawDataSet(obj_fn, None, obj_header, eta=eta, arc=arc, dark=dark)

    rawDataSet.flatFns.append(flat_fn)
     
    if not os.path.exists(out_dir):
        try: 
            os.mkdir(out_dir)
        except: 
            msg = 'output directory {} does not exist and cannot be created'.format(out_dir)
            raise IOError(msg)
                
    # generate reduced data set by reducing raw data set
    reducedDataSet = reduce_frame.reduce_frame(rawDataSet, out_dir, eta=eta, arc=arc, dark=dark)
    
    # write reduction summary to log file
    write_summary(reducedDataSet)
        
    # produce data products from reduced data set
    if config.params['no_products'] is True:
        logger.info('data product generation inhibited by command line switch')
    else:
        products.gen(reducedDataSet, out_dir, eta=eta, arc=arc)

    # if diagnostics mode is enabled, then produce diagnostic data products
    if config.params['dgn'] is True:
        logger.info('diagnostic mode enabled, generating diagnostic data products')
        dgn.gen(reducedDataSet, out_dir, eta=eta, arc=arc)
        
    return    


def write_summary(rds):    
    
    logger = logging.getLogger('obj')
    logger.info("summary:")
    v = []
    v.append(('base name' , '{}', rds.getBaseName()))
    v.append(('observation time (UT)',              '{}',       
              rds.getDate() + ' ' + rds.getTime()))
    v.append(('target name',                        '{}',       rds.getTargetName()))
    v.append(('filter',                             '{}',       rds.getFilter()))
    v.append(('slit',                               '{}',       rds.getSlit()))
    v.append(('cross disperser angle (deg)',        '{:.2f}',   rds.getDispPos()))
    v.append(('Echelle angle (deg)',                '{:.2f}',   rds.getEchPos()))
    v.append(('integration time (sec)',             '{:.0f}',   rds.getITime()))
#     v.append(('n coadds',                       '{:d}',     rds.getNCoadds()))
    v.append(('n orders expected',                  '{:d}',     rds.Flat.nOrdersExpected))
    v.append(('n orders reduced',                   '{:d}',     rds.Flat.nOrdersFound))
    v.append(('SNR mean',                           '{:.1f}',   rds.snrMean))
    v.append(('SNR min',                            '{:.1f}',   rds.snrMin))
    v.append(('spatial peak width mean (pixels)',   '{:.1f}',   rds.wMean))
    v.append(('spatial peak width max (pixels)',    '{:.1f}',   rds.wMax))
    v.append(('n sky/etalon/arc lines found',       '{:d}',     rds.nLinesFound))
    v.append(('n sky/etalon/arc lines used',        '{:d}',     rds.nLinesUsed))
    v.append(('RMS fit residual (Angstroms)',       '{:.3f}',   rds.frameCalRmsRes))
    
    for val in v:
        try:
            logger.info('{:>34}'.format(val[0]) + ' = ' + str(val[1]).format(val[2]))
        except ValueError as e:
            logger.info('{:>27}'.format(val[0]) + ' = ')
        except TypeError as e:
            logger.debug('Excepting TypeError: {}'.format(e))
            pass
