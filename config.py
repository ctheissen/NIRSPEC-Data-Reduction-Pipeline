params = {}

# command line arguments
params['cmnd_line_mode']        = False
params['debug']                 = False     # comment
params['no_cosmic']             = False     # cosmic ray rejection inhibited if True
params['no_products']           = False     # data product generation inhibited if True
params['obj_window']            = 9         # comment
params['sky_window']            = 8         # comment
params['sky_separation']        = 2         # comment
params['upgrade']               = False     # New version of NIRSPEC (after Oct 2018)

params['oh_filename']           = './ir_ohlines.dat'
params['oh_envar_name']         = 'NSDRP_OH_FILENAME'
params['oh_envar_override']     = False     # if True then use params['oh_filename'] 
                                            # even if envar is set
                                            
params['spatial_jump_override'] = False
params['spatial_rect_flat']     = False     # Do not trace the object spectra for spatial rectification (low S/N usage)
params['boost_signal']          = False     # Boost the signal for the spatial rectification trace. Useful for faint sources

params['int_c']                 = False
params['dgn']                   = False     # diagnostic data product generation enabled if True
params['npy']                   = False
params['verbose']               = False     # all log file messages printed to stdout if True
params['subdirs']               = False     # data products in per-frame subdirectory if True
params['lla']                   = 2         # sky line location algorithm
params['pipes']                 = False
params['shortsubdir']           = True
params['ut']                    = None
params['gunzip']                = False
params['out_dir']               = './nsdrp_out'       # used only in command line mode
#params['serialize_rds']         = False
params['jpg']                   = False     # if True then write preview plots in JPG not PNG

# configuration and tuning parameters
params['max_n_flats']           = 8
params['max_n_darks']           = 8
params['max_n_etas']            = 4
params['max_spatial_trace_res'] = 1.0


params['long_slit_edge_margin']    = 1         # cut out margin in pixels
params['K-AO_edge_margin']         = 1
params['large_tilt_threshold']     = 20
params['large_tilt_extra_padding'] = 10
params['overscan_width']           = 10

params['extra_cutout'] = False  # do not do any extra trimming

params['sowc'] = False  # simple order width calculation

params['log_dir']  = False  # log directory
params['log_file'] = False  # full path to the log file


# expected order number at bottom of detector
starting_order = {
        'NIRSPEC-1': 80, 
        'NIRSPEC-2': 70, 
        'NIRSPEC-3': 67, 
        'NIRSPEC-4': 61, 
        'NIRSPEC-5': 53, 
        'NIRSPEC-6': 49, 
        'NIRSPEC-7': 41,
        'K-AO': 38
}

def get_starting_order(filtername):
    return starting_order[filtername.upper()[:9]]

# order edge location error threshold
max_edge_location_error = {
        'NIRSPEC-1': 40, 
        'NIRSPEC-2': 50, 
        'NIRSPEC-3': 50, 
        'NIRSPEC-4': 20, 
        'NIRSPEC-5': 50, 
        'NIRSPEC-6': 20, 
        'NIRSPEC-7': 60, 
        'K-AO': 60  
}

def get_max_edge_location_error(filtername, slit):
    if '24' in slit:
        if 'NIRSPEC-7' in filtername.upper():
            return 35
            #return 50
        else:
            return 30
    else:
        return max_edge_location_error[filtername.upper()[:9]]
    
# order cutout padding
long_slit_cutout_padding = {
    'NIRSPEC-1': 0, 
    'NIRSPEC-2': 0, 
    'NIRSPEC-3': 0, 
    'NIRSPEC-4': 0, 
    'NIRSPEC-5': 0, 
    'NIRSPEC-6': 1, 
    'NIRSPEC-7': 3,
    'K-AO': 0        
}
short_slit_cutout_padding = {
    'NIRSPEC-1': 0, 
    'NIRSPEC-2': 0, 
    'NIRSPEC-3': 1, 
    'NIRSPEC-4': 1, 
    'NIRSPEC-5': 1, 
    'NIRSPEC-6': 1, 
    'NIRSPEC-7': 3,
    'K-AO': 0          
}

extra_cutout = {
    'NIRSPEC-1': 10, 
    'NIRSPEC-2': 10, 
    'NIRSPEC-3': 10, 
    'NIRSPEC-4': 10, 
    'NIRSPEC-5': 10, 
    'NIRSPEC-6': 10, 
    'NIRSPEC-7': 10,
    'K-AO': 10          
}

def get_cutout_padding(filtername, slit):
    if '24' in slit:
        return(long_slit_cutout_padding[filtername.upper()[:9]])
    else:
        return(short_slit_cutout_padding[filtername.upper()[:9]])


def get_extra_cutout(filtername, slit):
    return(extra_cutout[filtername.upper()[:9]])

    