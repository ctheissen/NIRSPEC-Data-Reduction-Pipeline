import numpy as np
import scipy.ndimage as ndimage
#from skimage.feature.peak import peak_local_max
import cosmics
#from __builtin__ import None
import matplotlib.pyplot as plt
import os
import nirspec_constants


def rectify_spatial(data, curve):
    """
    Shift data, column by column, along y-axis according to curve.
    
    Returns shifted image.
    
    Throws IndexError exception if length of curve 
    is not equal to number of columns in data. 
    """
    
    # shift curve to be centered at middle of order 
    # and change sign so shift is corrective
#     curve_p = -1.0 * (curve - (data.shape[0] / 2))
    curve_p = -1.0 * curve
    curve_p = curve_p - np.amin(curve_p)
    """
    import pylab as pl
    pl.figure()
    pl.cla()
    pl.plot(curve, 'r-')
    pl.plot(curve_p, 'g-')
    pl.show()
    
    pl.figure()
    pl.cla()
    pl.imshow(data, vmin=0, vmax=256)
    pl.show()
    """
    rectified = []
    for i in range(0, len(curve_p)):
        s = data[:, i]
        rectified.append(ndimage.interpolation.shift(
                s, curve_p[i], order=3, mode='nearest', prefilter=True))
    """
    pl.figure()
    pl.cla()
    pl.imshow((np.array(rectified)).transpose(), vmin=0, vmax=256)
    pl.show()
    """
    return((np.array(rectified)).transpose())



def rectify_spectral(data, curve, peak=None, returnpeak=None):
    """
    Shift data, row by row, along x-axis according to curve.
    
    Returns shifted image.
    
    Throws IndexError exception if length of curve
    is not equal to number of rows in data.
    """
    #print('TEST2')
    #print(curve)
    #print(curve.shape)
    #print(data.shape)
    #sys.exit()
    
    # pivot curve around peak 
    # and change sign so shift is corrective 
    profile  = data.sum(axis=1)
    #import matplotlib.pyplot as plt
    #plt.plot(profile)
    #print('PEAK', peak)
    #plt.show()
    if peak == None:
        peak     = np.argmax(profile)
    curve_p  = -1.0 * (curve - curve[peak])
        
    rectified = []
    for i in range(0, len(curve_p)):
        s = data[i, :]
        rectified.append(ndimage.interpolation.shift(
                s, curve_p[i], order=3, mode='nearest', prefilter=True)) 

    if returnpeak == True:
        return(np.array(rectified), peak)

    return(np.array(rectified))



def normalize(data, on_order, off_order):
    """
    data is the image cut-out plus padding
    
    on_order is array of same size as data with 
    on-order pixels set to 1.0 and off order (padding) pixels set to 0.0.
    
    off_order is array of same size as data with 
    off-order (padding) pixels set to 1.0 and on order pixels set to 0.0.
    
    returns normalized data array and median(mean) of the on-order pixels
    """
    
    #m    = np.mean(data)
    m    = np.median(data)
    non  = np.count_nonzero(on_order)
    noff = np.count_nonzero(off_order)
    
    data_copy = data
    
    if np.count_nonzero(on_order) == 0:
        return

    # ignore pixels beyond the dropoff at the red end by setting value to 1.0
    if nirspec_constants.upgrade:
        data_copy[:, 2048-48:] = 1.0
    else:    
        data_copy[:, 1024-24:] = 1.0

    # take median (mean) of only the on-order pixels
    #mean = np.ma.masked_array(data_copy, mask=off_order).mean()
    median = np.ma.median(np.ma.masked_array(data_copy, mask=off_order))
    
    # create normalized data array
    #normalized = (data_copy * on_order) / mean
    normalized = (data_copy * on_order) / median

    # around the edges of the order can blow up when div by median (mean), set those to one
    normalized[np.where(normalized > 10.0)] = 1.0

    # avoid zeroes (not too sure about these)
    normalized[np.where(normalized == 0.0)] = 1.0
    normalized[np.where(normalized < 0.2)] = 1.0

    #return normalized, mean
    return normalized, median



def cosmic_clean(data):
    """
    """
    max_iter = 3
    sig_clip = 5.0
    sig_frac = 0.3
    obj_lim = 5.0
    
    c = cosmics.cosmicsImage(data, sigclip=sig_clip, sigfrac=sig_frac, objlim=obj_lim, 
            verbose=False)
    
    c.run(max_iter)
    return(c.cleanarray)



def get_extraction_ranges(image_width, peak_location, obj_w, sky_w, sky_dist):
    """
    This function was modified so it can be used to define object window only or object and sky 
    windows.  If sky_w and sky_dist are None then only image window pixel list is computed and
    returned.
    
    Truncate windows that extend beyond top or bottom of order.
    
    Args:
        image_width:
        peak_location:
        obj_w:
        sky_w:
        sky_dist:
        
    Returns:
        three element tuple consisting of:
            0: Extraction range list.
            1: Top sky range list or None.
            2: Bottom sky range list or None.
    """
    
    if obj_w % 2:
        ext_range = np.array(range(int((1 - obj_w) / 2.0), int((obj_w + 1) / 2.0))) + peak_location
    else:  
        ext_range = np.array(range((-obj_w) / 2, obj_w / 2)) + peak_location
    ext_range = np.ma.masked_less(ext_range, 0).compressed()
    ext_range = np.ma.masked_greater_equal(ext_range, image_width).compressed()
    ext_range_list = ext_range.tolist()


    if sky_w is not None and sky_dist is not None:
        sky_range_top = np.array(range(ext_range[-1] + sky_dist, ext_range[-1] + sky_dist + sky_w))
        sky_range_top = np.ma.masked_less(sky_range_top, 0).compressed()
        sky_range_top = np.ma.masked_greater_equal(sky_range_top, image_width).compressed()
        sky_range_top_list = sky_range_top.tolist()
    
        sky_range_bot = np.array(range(ext_range[0] - sky_dist - sky_w + 1,
                ext_range[0] - sky_dist + 1))
        sky_range_bot = np.ma.masked_less(sky_range_bot, 0).compressed()
        sky_range_bot = np.ma.masked_greater_equal(sky_range_bot, image_width).compressed()
        sky_range_bot_list = sky_range_bot.tolist()
    else:
        sky_range_top_list = None
        sky_range_bot_list = None
    
    return ext_range_list, sky_range_top_list, sky_range_bot_list


   
def extract_spectra(obj, flat, noise, obj_range, sky_range_top, sky_range_bot, eta=None, arc=None):
    
    """
    """
    #print('OBJ RANGE:', obj_range)
    #sys.exit()

    ### TESTING AREA XXX
    '''
    print(obj_range)
    print(sky_range_top, sky_range_bot)
    import matplotlib.pyplot as plt
    plt.figure(20)
    plt.imshow(obj)
    plt.figure(21)
    plt.imshow(flat, aspect='auto', origin='lower')
    plt.figure(22)
    plt.imshow(noise, aspect='auto', origin='lower')
    #plt.show()
    #sys.exit()
    '''
    
    obj_sum     = np.sum(obj[i, :] for i in obj_range)
    flat_sum    = np.sum(flat[i, :] for i in obj_range)
    
    flat_sp     = flat_sum / len(obj_range)

    sky_top_sum = np.sum(obj[i, :] for i in sky_range_top)
    sky_bot_sum = np.sum(obj[i, :] for i in sky_range_bot)
    
    if len(sky_range_top) > 0:
        top_bg_mean = (sky_top_sum / len(sky_range_top)).mean()
    else:
        top_bg_mean = None
    if len(sky_range_bot) > 0:
        bot_bg_mean = (sky_bot_sum / len(sky_range_bot)).mean()
    else:
        bot_bg_mean = None
    
    sky_mean = (sky_top_sum + sky_bot_sum) / (len(sky_range_top) + len(sky_range_bot))
    """
    print(top_bg_mean)
    print(bot_bg_mean)
    print(sky_mean)
    """


#     sky_mean -= np.median(sky_mean) 
    #print('Obj sum', obj_sum)
    #print('Sky mean', sky_mean)
    obj_sp            = obj_sum - (len(obj_range) * sky_mean)
    #print('Obj sp', obj_sp)
    #sys.exit()

    sky_sp            = sky_mean - sky_mean.mean() # why this?
    
    obj_noise_sum     = np.sum(noise[i, :] for i in obj_range)
    sky_noise_top_sum = np.sum(noise[i, :] for i in sky_range_top)
    sky_noise_bot_sum = np.sum(noise[i, :] for i in sky_range_bot)
    
    k = float(np.square(len(obj_range))) / float(np.square((len(sky_range_top) + len(sky_range_bot))))
    '''
    print(k)
    print(np.square(len(obj_range)) / np.square((len(sky_range_top) + len(sky_range_bot))))
    print(np.square(len(obj_range)))
    print(np.square((len(sky_range_top) + len(sky_range_bot))))
    print()
    '''
    noise_sp = np.sqrt(obj_noise_sum + (k * (sky_noise_top_sum + sky_noise_bot_sum)))
    '''
    print(obj_range, sky_range_top, sky_range_bot)
    print(obj_noise_sum)
    print(k)
    print(sky_noise_top_sum)
    print(sky_noise_bot_sum)
    print((k * (sky_noise_top_sum + sky_noise_bot_sum)))
    print(noise_sp)

    plt.figure()
    plt.plot(obj_sp, label='Object')
    plt.plot(noise_sp, label='Noise')
    plt.show()
    #sys.exit()
    '''
    
    if eta is not None:
        #etalon_sum  = np.sum(eta[i, :] for i in obj_range) 
        etalon_sum  = np.sum(eta[10:-10, :], axis=0)
        etalon_sub  = etalon_sum - np.median(etalon_sum) # put the floor at ~0
        etalon_norm = etalon_sub / np.max(etalon_sub) * 0.9 # normalize for comparison to synthesized etalon
        etalon_sp   = etalon_norm + 0.9 # Add to the continuum for comparison to synthesized etalon
        
        return obj_sp, flat_sp, etalon_sp, sky_sp, noise_sp, top_bg_mean, bot_bg_mean

    if arc is not None:
        #arclamp_sum  = np.sum(eta[i, :] for i in obj_range) 
        arclamp_sum  = np.sum(arc[10:-10, :], axis=0)
        arclamp_sub  = arclamp_sum - np.median(arclamp_sum) # put the floor at ~0
        arclamp_norm = arclamp_sub / np.max(arclamp_sub) * 0.9 # normalize for comparison to synthesized arc lamp
        arclamp_sp   = arclamp_norm + 0.9 # Add to the continuum for comparison to synthesized arc lamp
        
        return obj_sp, flat_sp, arclamp_sp, sky_sp, noise_sp, top_bg_mean, bot_bg_mean

    else:
        return obj_sp, flat_sp, sky_sp, noise_sp, top_bg_mean, bot_bg_mean



def gaussian(x, a, b, c):
    return(a * np.exp(-(x - b)**2 / c**2))



def cut_out(data, top, bot, padding):
    try:    
        if bot > padding:
            return data[bot - padding:top + padding, :]
        else:
            return data[0:top + padding, :]
    except TypeError:
        if bot > padding:
            return data[int(bot) - int(padding):int(top) + int(padding), :]
        else:
            return data[0:int(top) + int(padding), :]


def centroid(spec, width, window, approx):
    p0 = max(0, approx - (window // 2))
    p1 = min(width - 1, approx + (window // 2)) + 1
    c  = p0 + ndimage.center_of_mass(spec[p0:p1])[0]
    
    if abs(c - approx) > 1:
        return(approx)    
    
    return(c)            
