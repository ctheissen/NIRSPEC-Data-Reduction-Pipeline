import numpy as np
import datetime
import config

import nirspec_constants as const

class GratingEq:
    
    def __init__(self, logger):
        self.logger = logger
        
    def evaluate(self, order, filtername, slit, echlpos, disppos, dateobs=None):
        """
        use grating equation with coefficients empirically found for each
        filter, grating angle, and echelle angle to determine starting
        locations to look for each order on array
    
        lambda = k1 * sin(theta_echelle) - k2 * (512 - pixel) * cos(theta_echelle) / order
    
        """
        #print('filt', filtername)
        left_indent = 50
    
        coeffs = dict()
        if const.upgrade:
            coeffs['K-AO']      = { 'c1': 0.376787803, 'c2': 5.35337265e6, 'y0': -3.12732704e6,
                                    'r1': 0.23482994, 'r2': -33591.707, 'z0': 14891.3158}; 
                                
            #coeffs['NIRSPEC-7'] = { 'c1': 0.376655776, 'c2': -4.22366351e6, 'y0': 2.45448586e6,
            #                        'r1': 0.23482994, 'r2': -33591.707, 'z0': 14891.3158};    
                                          
            coeffs['NIRSPEC-7'] = { 'c1': 0.37673628855616464, 'c2': -16820.86460210653, 'y0': 2620.4274343405577,
                                    'r1': 0.23482994, 'r2': -33591.707, 'z0': 14891.3158};  

            coeffs['NIRSPEC-6'] = { 'c1': 0.24986411, 'c2': -35961.453, 'y0': 15944.8337,
                                    'r1': 0.23686484, 'r2': -33685.61,  'z0': 14901.32};   
                                                 
            coeffs['NIRSPEC-5'] = { 'c1': 0.37030507, 'c2': -36304.523, 'y0': 16215.28,
                                    'r1': 0.35065575, 'r2': -33928.81,  'z0': 15115.5859};      
        
            coeffs['NIRSPEC-4'] = { 'c1': 0.37318182, 'c2': -36031.358, 'y0': 16009.5052,
                                    'r1': 0.35333187, 'r2': -33646.524, 'z0': 14908.2313};    
        
            coeffs['NIRSPEC-3'] = { 'c1': 0.5669286175211347, 'c2': -2257995.9452759516, 'y0': 1259626.5524847843,
                                    'r1': 0.35514352, 'r2': -36175.709, 'z0': 16283.995};    
                                
            coeffs['NIRSPEC-2'] = { 'c1': 0.49449345, 'c2': -35965.39,  'y0': 15987.1423,
                                    'r1': 0.46779492, 'r2': -31416.355, 'z0': 13601.3478};                                    
        
            #coeffs['NIRSPEC-1'] = { 'c1': 0.754348427, 'c2': -1.34342463e6, 'y0': 7.62950418e5,
            #                        'r1': 0.4713783,  'r2': -38876.842, 'z0': 17880.5877}; 

            coeffs['NIRSPEC-1'] = { 'c1': 0.754607318, 'c2': -2.88170696e6, 'y0': 1.64417296e6,
                                    'r1': 0.4713783,  'r2': -38876.842, 'z0': 17880.5877}; 

        else:
            coeffs['K-AO']      = { 'c1': 0.24792775, 'c2': -35906.947, 'y0': 15955.4515,
                                    'r1': 0.23482994, 'r2': -33591.707, 'z0': 14891.3158}; 
                                    
            coeffs['NIRSPEC-7'] = { 'c1': 0.24792775, 'c2': -35906.947, 'y0': 15955.4515,
                                    'r1': 0.23482994, 'r2': -33591.707, 'z0': 14891.3158};    
                                          
            coeffs['NIRSPEC-6'] = { 'c1': 0.24986411, 'c2': -35961.453, 'y0': 15944.8337,
                                    'r1': 0.23686484, 'r2': -33685.61,  'z0': 14901.32};   
                                                 
            coeffs['NIRSPEC-5'] = { 'c1': 0.37030507, 'c2': -36304.523, 'y0': 16215.28,
                                    'r1': 0.35065575, 'r2': -33928.81,  'z0': 15115.5859};      
        
            coeffs['NIRSPEC-4'] = { 'c1': 0.37318182, 'c2': -36031.358, 'y0': 16009.5052,
                                    'r1': 0.35333187, 'r2': -33646.524, 'z0': 14908.2313};    
        
            coeffs['NIRSPEC-3'] = { 'c1': 0.37495832, 'c2': -36067.086, 'y0': 15987.1927,
                                    'r1': 0.35514352, 'r2': -36175.709, 'z0': 16283.995};    
                                
            coeffs['NIRSPEC-2'] = { 'c1': 0.49449345, 'c2': -35965.39,  'y0': 15987.1423,
                                    'r1': 0.46779492, 'r2': -31416.355, 'z0': 13601.3478};                                    
        
            coeffs['NIRSPEC-1'] = { 'c1': 0.49777509, 'c2': -38653.878, 'y0': 17488.344,
                                    'r1': 0.4713783,  'r2': -38876.842, 'z0': 17880.5877};      
                             
        c1 = coeffs[filtername.upper()[:9]]['c1']
        c2 = coeffs[filtername.upper()[:9]]['c2']
        y0 = coeffs[filtername.upper()[:9]]['y0']
       
        if const.upgrade: 
            pixel = np.arange(const.N_COLS_upgrade, dtype=float)

            #k1, k2 = 8.55030015e5, 1.61866180e1 # Found these coeffs empirically using the new format simulator
            k1, k2 = 8.54942246e5, 1.65988312e1 # New test coeffs
        else:
            pixel = np.arange(const.N_COLS, dtype=float)
    
            k1, k2 = 8.528e5, 2.413e1
    
        # solve for wavelength scale
        if const.upgrade: # 
            wavelength_scale = (k1 * np.sin(np.radians(echlpos)) -
                k2 * ((const.N_COLS_upgrade / 2) - pixel) * np.cos(np.radians(echlpos))) / order
        
            wavelength_left = (k1 * np.sin(np.radians(echlpos)) -
                k2 * ((const.N_COLS_upgrade / 2) - left_indent) * np.cos(np.radians(echlpos))) / order
        else:
            wavelength_scale = (k1 * np.sin(np.radians(echlpos)) -
                k2 * ((const.N_COLS / 2) - pixel) * np.cos(np.radians(echlpos))) / order
        
            wavelength_left = (k1 * np.sin(np.radians(echlpos)) -
                k2 * ((const.N_COLS / 2) - left_indent) * np.cos(np.radians(echlpos))) / order
    
        # solve for location of the beginning of the middle of the order in spatial axis
        left_mid_row = c1 * wavelength_left + c2 * np.sin(np.radians(disppos)) + y0
        #print('LEFT:', wavelength_left, left_mid_row, const.upgrade)
    
        if config.params['sowc'] is True:
            self.logger.info('using simple order width calculation')
            if '24' in slit:
                orderWidth = 24.0 / 0.193
                self.logger.info('long slit order width = {:.0f} pixels'.format(orderWidth))
            else:
                if filtername.upper() == 'K-AO':
                    orderWidth = 12.0 / 0.1
                else:
                    orderWidth = 12.0 / 0.193
                    self.logger.info('short slit order width = {:.0f} pixels'.format(orderWidth))

            left_top_row = left_mid_row + (orderWidth / 2.0)
            left_bot_row = left_mid_row - (orderWidth / 2.0)

        else:
            
            self.logger.info('using complicated order width calculation')
            # width of order depends on disperser angle
            if const.upgrade: 

                # Old values but work for our data (NIRSPEC upgrade N7)
                #k3 = -3.91408142e6 # Found these coeffs empirically using the new format simulator
                #k4 = -1.09796060e5

                #k3, k4 = 6.51668027e4, 1.82380583e3 # New test coeffs
                k3, k4 = 25943.42645610225, 724.5138607638191 # New test coeffs for N7

                if 'NIRSPEC-1' in filtername:
                    #k3, k4 = 5.23052614e6, 1.49654716e5 # old coeffs
                    k3, k4 = -2.47020399e6, -7.06809106e4 # New test coeffs

                if 'NIRSPEC-3' in filtername:
                    k3, k4 = 6635335.343062301, 194638.89613176414

                if 'K-' in filtername:
                    k3, k4 = -3.75106282e6, -1.05224419e5

            else: # pre-upgrade values
                k3, k4 = 9.9488e1, 1.0517e0
            
            left_top_row = left_mid_row + ((k3 - (k4 * disppos)) / 2.0)
            left_bot_row = left_mid_row - ((k3 - (k4 * disppos)) / 2.0)
            #print('left top:', left_top_row)
            #print('left bot:', left_bot_row)
                
            # apply empirical corrections
            
            long_slit_y_corr    = 20
            low_res_slit_y_corr = 30
        #    date_y_corr         = 50
            date_y_corr         = 0
            filter_7_y_corr     = 45
            filter_4_5_6_y_corr = 30
            filter_3_y_corr     = 50
            filter_1_y_corr     = 50
            if const.upgrade: 
            	filter_1_y_corr     = 20
            filter_K_y_corr     = 70
        
            if '24' in slit:
                
                if '1' not in filtername:
                    self.logger.debug('applying +/-' + str(long_slit_y_corr) + 
                                ' pixel long slit y correction for slit ' + slit)
                    left_top_row += long_slit_y_corr
                    left_bot_row -= long_slit_y_corr
                
                if '3' in filtername:
                    self.logger.info('applying N-3 long slit correction')
                    left_top_row -= 25
                    left_bot_row -= 25 
                    
                if '5' in filtername:
                    self.logger.info('applying N-5 long slit correction')
                    left_top_row += 10
                    left_bot_row += 10
                    
                if '6' in filtername:
                    self.logger.info('applying N-6 long slit correction')
                    left_top_row -= 25
                    left_bot_row -= 25 
                
            elif '42x' in slit:
                self.logger.debug('applying +/-' + str(low_res_slit_y_corr) + 
                            ' pixel low res slit y correction for slit ' + slit)
                left_top_row += low_res_slit_y_corr
                left_bot_row -= low_res_slit_y_corr
                
            elif '2.26' in slit:
                self.logger.debug ('applying x2.26 AO slit correction')
                left_bot_row = left_top_row - ((left_top_row - left_bot_row) * 2)

            elif '1.13' in slit:
                self.logger.debug ('applying no correction for x1.13 AO slit correction')
                
            if dateobs is not None:
                obs_date   = datetime.datetime.strptime(dateobs, '%Y-%m-%d')
                shift_date = datetime.datetime.strptime('2004-05-24', '%Y-%m-%d')
                
                if obs_date < shift_date:
                    self.logger.debug('applying +' + str(date_y_corr) + 
                                ' pixel pre-' + shift_date.strftime('%x') + ' y correction')
                    left_top_row += date_y_corr
                    left_bot_row += date_y_corr
                
            if 'NIRSPEC-7' in filtername:
                self.logger.debug('applying + ' + str(filter_7_y_corr) + ' pixel y corr for filter ' + filtername)
                left_top_row += filter_7_y_corr
                left_bot_row += filter_7_y_corr
                
            elif 'NIRSPEC-6' in filtername or 'NIRSPEC-5' in filtername or 'NIRSPEC-4' in filtername:
                self.logger.debug('applying +' + str(filter_4_5_6_y_corr) + 
                            ' pixel N-4,5,6 filter y corr for filter ' + filtername)
                left_top_row += filter_4_5_6_y_corr
                left_bot_row += filter_4_5_6_y_corr
                
            elif 'NIRSPEC-3' in filtername:
                self.logger.debug('applying +' + str(filter_3_y_corr) + 
                            ' pixel N-3 filter y corr for filter ' + filtername )
                left_top_row += filter_3_y_corr
                left_bot_row += filter_3_y_corr
                
            elif 'NIRSPEC-1' in filtername:
                self.logger.debug('applying +' + str(filter_1_y_corr) + 
                            ' pixel N-1 filter y corr for filter ' + filtername )
                left_top_row += filter_1_y_corr
                left_bot_row += filter_1_y_corr

            elif 'K-' in filtername:
                self.logger.debug('applying + ' + str(filter_K_y_corr) + ' pixel y corr for filter ' + filtername)
                left_top_row += filter_K_y_corr
                left_bot_row += filter_K_y_corr
                
                self.logger.info('order width = {:.0f} pixels'.format(left_top_row - left_bot_row))
            
        wavelength_shift = 0
        
        if order > 55:
            wavelength_shift = 50.0
            if const.upgrade: wavelength_shift = 20.0
        elif order < 38:
            wavelength_shift = 70.0
            if const.upgrade: wavelength_shift = 50.0
        elif order < 40:
            wavelength_shift = 70.0
        elif order < 45:
            wavelength_shift = 60.0
        elif order < 51:
            wavelength_shift = 50.0
        elif order < 55:
            wavelength_shift = 50.0


        #print(left_top_row, left_bot_row, left_top_row - left_bot_row) 
     
        self.logger.debug('applying ' + str(wavelength_shift) + ' Angstrom order-dependent wavelength shift')
        wavelength_scale += wavelength_shift
     
        return left_top_row, left_bot_row, wavelength_scale
    
    
    def is_on_detector(self, left_top_row, left_bot_row):
        padding = 20.0
        if const.upgrade: 
            pixYlimit = 2048
        else: 
            pixYlimit = 1024
        if left_top_row < (pixYlimit - padding + 50) and left_bot_row > 0:
            return True
        else:
            return False
    
