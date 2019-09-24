import logging
import numpy as np
import scipy.optimize as op
import image_lib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy import signal
import nirspec_constants


def NormDist(x, mean, sigma, baseline, amplitude):

	Dist1 = amplitude * 1. / np.sqrt(2. * np.pi * sigma**2) * np.exp(-1*(x - mean)**2 / (2.*sigma**2) ) + baseline

	return Dist1



def CreateSpatialMap(image, numrows=5, clip=15, cutoff=10, plot=False, plotvid=False, plotfinal=False):

	# Make some changes for the upgraded version of NIRSPEC
	if nirspec_constants.upgrade: 
		numrows, cutoff = 11, 40

	if nirspec_constants.boost_signal:
		numrows *= 3

	#print(image.shape)
	Centroids = []
	Pixels    = []
	for i in range(image.shape[1]-cutoff):
		#print(i)
		if i >= image.shape[1]-cutoff:
			#print('--0--')
			continue

		if i < numrows:
			#print('--1--')
			#guess1 = np.where(np.sum(image[clip:-clip,0:numrows+1], axis=1) == np.max(np.sum(image[clip:-clip,0:numrows+1], axis=1)))[0][0]
			#guess1 = len(np.sum(image[7:-7,0:numrows+1], axis=1)) / 2. + 7
			#print('Guess', guess1)
			
			if nirspec_constants.upgrade:
				Xs0  = np.arange(len(np.sum(image[:, 0:200], axis=1)))
				Ys0  = np.sum(image[:, 0:200], axis=1)

				Xs00 = np.arange(len(np.sum(image[:, -200:], axis=1)))
				Ys00 = np.sum(image[:, -200:], axis=1)

			else:
				Xs0  = np.arange(len(np.sum(image[:, 0:200], axis=1)))
				Ys0  = np.sum(image[:, 0:200], axis=1)

				Xs00 = np.arange(len(np.sum(image[:, -200:], axis=1)))
				Ys00 = np.sum(image[:, -200:], axis=1)

			Xs  = np.arange(len(np.sum(image[:, 0:numrows*2+1], axis=1)))
			Ys  = np.sum(image[clip:-clip, 0:numrows*2+1], axis=1)

			guess1 = Xs0[np.where(Ys0 == np.ma.max(Ys0))][0]
			guess2 = Xs00[np.where(Ys00 == np.ma.max(Ys00))][0]

			#print('Guess', guess1, guess2)
			#print(np.min([guess1, guess2]))
			#print(np.max([guess1, guess2]))
			range1 = np.ma.min([guess1, guess2]) - 15
			range2 = np.ma.max([guess1, guess2]) + 15
			#print('RANGE1, RANGE2:', range1, range2)
			#print('GUESS1, GUESS2:', guess1, guess2)
			if range1 < 0: 
				range1 = 0
			if range2 > image.shape[0]: 
				range2 = image.shape[0]
			#print('RANGE1, RANGE2:', range1, range2)
			#print(image.shape)

			if plot:
				fig0 = plt.figure(198, figsize=(10,6))
				ax1 = fig0.add_subplot(121)
				ax2 = fig0.add_subplot(122)
				#ax1.plot(np.arange(len(np.sum(image[clip:-clip, 0:numrows+1], axis=1))) + clip,
				#	     np.sum(image[clip:-clip,0:numrows+1], axis=1))
				if nirspec_constants.upgrade:
					ax1.plot(np.arange(len(np.sum(image[:, 0:200], axis=1))),
						     np.sum(image[:, 0:200], axis=1) )#+ np.sum(image[:, -40:], axis=1))
					ax2.plot(np.arange(len(np.sum(image[:, -200:], axis=1))),
						     np.sum(image[:, -200:], axis=1) )#+ np.sum(image[:, -40:], axis=1))
				else:
					ax1.plot(np.arange(len(np.sum(image[:, 0:200], axis=1))),
						     np.sum(image[:, 0:200], axis=1) )#+ np.sum(image[:, -40:], axis=1))
					ax2.plot(np.arange(len(np.sum(image[:, -200:], axis=1))),
						     np.sum(image[:, -200:], axis=1) )#+ np.sum(image[:, -40:], axis=1))

				ax1.axvline(guess1, c='r', ls='--')
				ax1.axvline(range1, c='m', ls=':')
				ax1.axvline(range2, c='m', ls=':')
				ax2.axvline(guess2, c='r', ls='--')
				ax2.axvline(range1, c='m', ls=':')
				ax2.axvline(range2, c='m', ls=':')
				plt.show()

			#print('Guess1', guess1)
			#guess2 = signal.find_peaks_cwt(Ys, np.arange(1,30))#, min_length=5)
			#guess2, _ = signal.find_peaks(Ys, width=3)
			#print('Guess2', guess2+clip)
			
			if i == 0:
				if plot:
					fig0 = plt.figure(198, figsize=(8,6))
					ax1 = fig0.add_subplot(111)
					#ax1.plot(np.arange(len(np.sum(image[clip:-clip, 0:numrows*2+1], axis=1))) + clip,
					#	     np.sum(image[clip:-clip,0:numrows*2+1], axis=1))
					ax1.plot(np.arange(len(np.sum(image[:, 0:numrows*2+1], axis=1))),
						     np.sum(image[:,0:numrows*2+1], axis=1))
					ax1.axvline(guess1, c='r', ls='--')
					#for G in guess2:
					#	ax1.axvline(G+clip, c='b', ls='--')
					ax1.axvline(range1, c='m', ls=':')
					ax1.axvline(range2, c='m', ls=':')
					print(range1, range2)
					plt.show()
					#sys.exit()
			
			try:
				#print(range1, range2)
				#Xs = np.arange(len(np.sum(image[clip:-clip, 0:numrows*2+1], axis=1))) + clip
				#Ys = np.sum(image[clip:-clip, 0:numrows*2+1], axis=1)
				Xs = np.arange(len(np.sum(image[range1:range2+1, 0:numrows*2+1], axis=1))) + range1
				#print('Xs', Xs)
				Ys = np.sum(image[range1:range2+1, 0:numrows*2+1], axis=1)
				#print('Ys', Ys)
				guess1 = Xs[np.where(Ys == np.ma.max(Ys))]
				#print('Guess1', guess1)
				#guess2 = signal.find_peaks_cwt(Ys, np.arange(8,20), min_length=9)
				#print('Guess2', guess1)
				#guess2, _ = signal.find_peaks(Ys, width=3)
				#guess1 = Xs[np.where(Ys[guess2] == np.max(Ys[guess2]))]
				#print('Guess11', guess1)
				#guess1 = Xs[len(Xs)//2]
				
				popt, pcov = op.curve_fit(NormDist, Xs, Ys, 
										  p0=[guess1, 2., np.ma.median(Ys), np.ma.max(Ys)], 
										  bounds = ( (guess1-7., 1., -1000., 0.), (guess1+7., 4., 1e7, 1e7) ),
										  maxfev=100000) # Where should a pixel start? (0, 1, 0.5?)
				#print(i,popt)
				
				#popt = [guess1]
				if plotvid:
					fig0 = plt.figure(i, figsize=(8,4))
					ax1 = fig0.add_subplot(121)
					ax2 = fig0.add_subplot(122)
					#ax1.imshow(image[clip:-clip, 0:numrows*2+1], origin='lower', aspect='auto')
					#ax1.axhline(popt[0]-clip, c='r', ls=':')
					ax1.imshow(image[range1:range2+1, 0:numrows*2+1], origin='lower', aspect='auto')
					#ax1.axhline(popt[0]-clip, c='r', ls=':')
					ax1.axhline(popt[0]-range1, c='r', ls=':')
					ax2.plot(Xs, Ys)
					Xs2 = np.linspace(np.ma.min(Xs), np.ma.max(Xs))
					ax2.plot(Xs2, NormDist(Xs2, *popt), 'r--')
					ax2.axvline(popt[0], c='r', ls=':')
					ax1.minorticks_on()
					ax2.minorticks_on()
					#plt.show()
					plt.draw()
					plt.pause(0.01)
					plt.close('all')
				
				Pixels.append(i)
				Centroids.append(popt[0])
			except: 
				continue
				#Pixels.append(0)

		else:
			#print('--2--')
			#guess1 = np.where(np.sum(image[clip:-clip, i-numrows:i+numrows+1], axis=1) == np.max(np.sum(image[clip:-clip, i-numrows:i+numrows+1], axis=1)))[0][0]
			#guess1 = len(np.sum(image[7:-7,0:numrows+1], axis=1)) / 2. + 7
			#print('Guess', guess1)
			'''
			if plot:
				fig0 = plt.figure(198, figsize=(8,6))
				ax1 = fig0.add_subplot(111)
				#ax1.plot(np.arange(len(np.sum(image[clip:-clip, i-numrows:i+numrows+1], axis=1))) + clip,
				#         np.sum(image[clip:-clip,0:numrows+1], axis=1))
				ax1.plot(np.arange(len(np.sum(image[range1:range2+1, i-numrows:i+numrows+1], axis=1))) + range1,
				         np.sum(image[range1:range2+1, i-numrows:i+numrows+1], axis=1))
				plt.show()
			'''
			try:
				#Xs = np.arange(len(np.sum(image[clip:-clip, i-numrows:i+numrows+1], axis=1))) + clip
				#Ys = np.sum(image[clip:-clip, i-numrows:i+numrows+1], axis=1)
				Xs     = np.arange(len(np.sum(image[range1:range2+1, i-numrows:i+numrows+1], axis=1))) + range1
				Ys     = np.sum(image[range1:range2+1, i-numrows:i+numrows+1], axis=1)
				guess1 = Xs[np.where(Ys == np.ma.max(Ys))]
				#guess1 = popt[0]
				#guess2, _ = signal.find_peaks(Ys, width=3)
				#guess1 = Xs[np.where(Ys[guess2] == np.max(Ys[guess2]))]+clip
				#guess1 = Xs[len(Xs)//2]
				#print('Guess', guess1)
				#print('Guess1', guess1)
				#print('Guess2', popt)
				#prevGuess = Centroids[-1]
				#print('Starts:', guess1, 2., np.ma.median(Ys), np.ma.max(Ys))
				
				popt, pcov = op.curve_fit(NormDist, Xs, Ys, 
										  p0=[guess1, 2., np.ma.median(Ys), np.ma.max(Ys)],
										  bounds = ( (guess1-7., 1., -1000., 0.), (guess1+7., 4., 1e7, 1e7) ), 
										  maxfev=100000) # Where should a pixel start? (0, 1, 0.5?)
				#print(i,popt)
				#popt = [guess1]
				#print(abs(popt[0]-prevGuess))
				#if abs(popt[0]-prevGuess) > 10: 
				#	continue

				if plotvid:
					fig0 = plt.figure(i, figsize=(8,4))
					ax1 = fig0.add_subplot(121)
					ax2 = fig0.add_subplot(122)
					#ax1.imshow(image[clip:-clip, i-numrows:i+numrows+1], origin='lower', aspect='auto')
					#ax1.axhline(popt[0]-clip, c='r', ls=':')
					ax1.imshow(image[range1:range2+1, i-numrows:i+numrows+1], origin='lower', aspect='auto')
					ax1.axhline(popt[0]-range1, c='r', ls=':')
					ax2.plot(Xs, Ys)
					Xs2 = np.linspace(np.ma.min(Xs), np.ma.max(Xs))
					ax2.plot(Xs2, NormDist(Xs2, *popt), 'r--')
					ax2.axvline(popt[0], c='r', ls=':')
					ax1.minorticks_on()
					ax2.minorticks_on()
					#plt.show()
					plt.draw()
					plt.pause(0.01)
					plt.close('all')
				

				Pixels.append(i)
				Centroids.append(popt[0])
			except: 
				continue
				#Pixels.append(0)
	
	
	Centroids   = np.array(Centroids)
	pixels      = np.array(Pixels)
	#print(list(pixels))
	#print(list(Centroids))
	#print('LENGTHS:',len(pixels), len(Centroids))
	#pixels      = np.arange(image.shape[1])

	unFitted = True
	count    = 0

	if plotfinal:
		fig = plt.figure(3)
		ax = fig.add_subplot(111)
		ax.scatter(pixels, Centroids, c='0.5', s=3, alpha=0.5)

	while unFitted:
		#plt.figure(3)
		#plt.scatter(pixels, Pixels, c='0.5')
		z1 = np.polyfit(pixels, Centroids, 3)
		p1 = np.poly1d(z1)
		#plt.plot(pixels, p1(pixels), 'r--')
		#plt.show()
		#sys.exit()

		# Do the fit
		hist = Centroids - p1(pixels)
		sig  = np.std(hist)
		#print('sigma', sig, 2*sig, 5*sig)
		if count == 0: 
			ind1 = np.where(abs(hist) < 1*sig)
		else:
			ind1 = np.where(abs(hist) < 2.5*sig)

		#plt.scatter(pixels[ind1], Pixels[ind1], marker='x', c='r')
		#plt.show()
		newpix  = pixels[ind1].flatten()
		newCent = Centroids[ind1].flatten()

		#print(len(Pixels) == len(newPix))
		#print(unFitted)
		if len(Centroids) == len(newCent): 
			
			if plotfinal:
				ax.plot(np.arange(image.shape[1]), p1(np.arange(image.shape[1])), 'r--')
				ax.scatter(newpix, newCent, marker='x', c='b', s=3)

			# Calc the RMS
			rms   = np.sqrt(np.mean(hist**2))
			sumsq = np.sum(hist**2)
			#print(rms, sumsq)
			#print(z1)
			if plotfinal:
				ax.plot(np.arange(image.shape[1]), p1(np.arange(image.shape[1])), 'r--')
				ax.scatter(newpix, newCent, marker='x', c='b', s=3)
				"""
				at = AnchoredText('RMS = %0.4f'%(rms) + '\n' + 'Coeff: %0.3E %0.3E %0.3E %0.3E'%(z1[-1], z1[-2], z1[-3], z1[-4]),
				    prop=dict(size=8), frameon=False,
				    loc=2,
				    )
				"""
				at = AnchoredText('RMS =  %0.4f'%(rms) + '\n' + \
					              'Sum of Squared Errors =  %0.4f'%(sumsq) + \
				                  '\n' + 'Coeff =' + \
				                  '\n' + '%0.3E'%(z1[-1]) + \
				                  '\n' + '%0.3E'%(z1[-2]) + \
				                  '\n' + '%0.3E'%(z1[-3]) + \
				                  '\n' + '%0.3E'%(z1[-4]),
				                  prop=dict(size=8), frameon=False, loc=2)
				ax.add_artist(at)
				ax.set_ylim(np.min(p1(np.arange(image.shape[1])))-3, np.max(p1(np.arange(image.shape[1])))+3)
			
			unFitted = False


		#print(unFitted)
		pixels    = pixels[ind1].flatten()
		Centroids = Centroids[ind1].flatten()

		count+=1
	
	#plt.figure(1)
	#plt.imshow(image, origin='lower')
	if plotfinal: 
		plt.show()
	

	return p1(np.arange(image.shape[1]))



def spectral_trace(calimage, linelist='apohenear.dat', interac=True,
                   fmask=(1,), display=True,
                   tol=10, fit_order=2, previous='', mode='poly',
                   second_pass=True):
    """
    Determine the wavelength solution to be used for the science images.
    Can be done either automatically (buyer beware) or manually. Both the
    manual and auto modes use a "slice" through the chip center to learn
    the wavelengths of specific lines. Emulates the IDENTIFY
    function in IRAF.

    If the automatic mode is selected (interac=False), program tries to
    first find significant peaks in the "slice", then uses a brute-force
    guess scheme based on the grating information in the header. While
    easy, your mileage may vary with this method.

    If the interactive mode is selected (interac=True), you click on
    features in the "slice" and identify their wavelengths.

    Parameters
    ----------
    calimage : str
        Etalon lamp image
    linelist : str, optional
        The linelist file to use in the resources/linelists/ directory.
        Only used in automatic mode. (Default is etalon.dat)
    interac : bool, optional
        Should the etalon identification be done interactively (manually)?
        (Default is True)
    trim : bool, optional
        Trim the image using the DATASEC keyword in the header, assuming
        has format of [0:1024,0:512] (Default is False) XXX remove this option, trim already done
    fmask : array-like, optional
        A list of illuminated rows in the spatial direction (Y), as
        returned by flatcombine.
    display : bool, optional
    tol : int, optional
        When in automatic mode, the tolerance in pixel units between
        linelist entries and estimated wavelengths for the first few
        lines matched... use carefully. (Default is 10)
    mode : str, optional
        What type of function to use to fit the entire 2D wavelength
        solution? Options include (poly, spline2d). (Default is poly)
    fit_order : int, optional
        The polynomial order to use to interpolate between identified
        peaks in the HeNeAr (Default is 2)
    previous : string, optional
        name of file containing previously identified peaks. Still has to
        do the fitting.

    Returns
    -------
    wfit : bivariate spline object or 2d polynomial
        The wavelength solution at every pixel. Output type depends on the
        mode keyword above (poly is recommended)
    """

    print('Finding etalon lines')

    # silence the polyfit warnings
    warnings.simplefilter('ignore', np.RankWarning)

    img = calimage

    # this approach will be very DIS specific
    #disp_approx = hdu[0].header['DISPDW']
    #wcen_approx = hdu[0].header['DISPWC']
    disp_approx = 1.66
    wcen_approx = 7300

    # take a slice thru the data (+/- 10 pixels) in center row of chip
    slice = img[img.shape[0]/2-10:img.shape[0]/2+10,:].sum(axis=0)

    # use the header info to do rough solution (linear guess)
    wtemp = (np.arange(len(slice))-len(slice)/2) * disp_approx * sign + wcen_approx


    ######   IDENTIFY   (auto and interac modes)
    # = = = = = = = = = = = = = = = =
    #-- automatic mode
    if (interac is False) and (len(previous)==0):
        print("Doing automatic wavelength calibration on HeNeAr.")
        print("Note, this is not very robust. Suggest you re-run with interac=True")
        # find the linelist of choice

        linelists_dir = os.path.dirname(os.path.realpath(__file__))+ '/resources/linelists/'
        # if (len(linelist)==0):
        #     linelist = os.path.join(linelists_dir, linelist)

        # import the linelist
        linewave = np.loadtxt(os.path.join(linelists_dir, linelist), dtype='float',
                              skiprows=1,usecols=(0,),unpack=True)


        pcent_pix, wcent_pix = find_peaks(wtemp, slice, pwidth=10, pthreshold=97)

    #   loop thru each peak, from center outwards. a greedy solution
    #   find nearest list line. if not line within tolerance, then skip peak
        pcent = []
        wcent = []

        # find center-most lines, sort by dist from center pixels
        ss = np.argsort(np.abs(wcent_pix-wcen_approx))

        #coeff = [0.0, 0.0, disp_approx*sign, wcen_approx]
        coeff = np.append(np.zeros(fit_order-1),(disp_approx*sign, wcen_approx))

        for i in range(len(pcent_pix)):
            xx = pcent_pix-len(slice)/2
            #wcent_pix = coeff[3] + xx * coeff[2] + coeff[1] * (xx*xx) + coeff[0] * (xx*xx*xx)
            wcent_pix = np.polyval(coeff, xx)

            if display is True:
                plt.figure()
                plt.plot(wtemp, slice, 'b')
                plt.scatter(linewave,np.ones_like(linewave)*np.nanmax(slice),marker='o',c='cyan')
                plt.scatter(wcent_pix,np.ones_like(wcent_pix)*np.nanmax(slice)/2.,marker='*',c='green')
                plt.scatter(wcent_pix[ss[i]],np.nanmax(slice)/2., marker='o',c='orange')

            # if there is a match w/i the linear tolerance
            if (min((np.abs(wcent_pix[ss][i] - linewave))) < tol):
                # add corresponding pixel and *actual* wavelength to output vectors
                pcent = np.append(pcent,pcent_pix[ss[i]])
                wcent = np.append(wcent, linewave[np.argmin(np.abs(wcent_pix[ss[i]] - linewave))] )

                if display is True:
                    plt.scatter(wcent,np.ones_like(wcent)*np.nanmax(slice),marker='o',c='red')

                if (len(pcent)>fit_order):
                    coeff = np.polyfit(pcent-len(slice)/2, wcent, fit_order)

            if display is True:
                plt.xlim((min(wtemp),max(wtemp)))
                plt.show()

        print('Matches')
        print(pcent)
        print(wcent)
        lout = open(calimage+'.lines', 'w')
        lout.write("# This file contains the HeNeAr lines identified [auto] Columns: (pixel, wavelength) \n")
        for l in range(len(pcent)):
            lout.write(str(pcent[l]) + ', ' + str(wcent[l])+'\n')
        lout.close()

        # the end result is the vector "coeff" has the wavelength solution for "slice"
        # update the "wtemp" vector that goes with "slice" (fluxes)
        wtemp = np.polyval(coeff, (np.arange(len(slice))-len(slice)/2))


    # = = = = = = = = = = = = = = = =
    #-- manual (interactive) mode
    elif (interac is True):
        if (len(previous)==0):
            print('')
            print('Using INTERACTIVE HeNeAr_fit mode:')
            print('1) Click on HeNeAr lines in plot window')
            print('2) Enter corresponding wavelength in terminal and press <return>')
            print('   If mis-click or unsure, just press leave blank and press <return>')
            print('3) To delete an entry, click on label, enter "d" in terminal, press <return>')
            print('4) Close plot window when finished')

            xraw = np.arange(len(slice))
            class InteracWave(object):
                # http://stackoverflow.com/questions/21688420/callbacks-for-graphical-mouse-input-how-to-refresh-graphics-how-to-tell-matpl
                def __init__(self):
                    self.fig = plt.figure()
                    self.ax = self.fig.add_subplot(111)
                    self.ax.plot(wtemp, slice, 'b')
                    plt.xlabel('Wavelength')
                    plt.ylabel('Counts')

                    self.pcent = [] # the pixel centers of the identified lines
                    self.wcent = [] # the labeled wavelengths of the lines
                    self.ixlib = [] # library of click points

                    self.cursor = Cursor(self.ax, useblit=False, horizOn=False,
                                         color='red', linewidth=1 )
                    self.connect = self.fig.canvas.mpl_connect
                    self.disconnect = self.fig.canvas.mpl_disconnect
                    self.clickCid = self.connect("button_press_event",self.OnClick)

                def OnClick(self, event):
                    # only do stuff if toolbar not being used
                    # NOTE: this subject to change API, so if breaks, this probably why
                    # http://stackoverflow.com/questions/20711148/ignore-matplotlib-cursor-widget-when-toolbar-widget-selected
                    if self.fig.canvas.manager.toolbar._active is None:
                        ix = event.xdata
                        print('onclick point:', ix)

                        # if the click is in good space, proceed
                        if (ix is not None) and (ix > np.nanmin(wtemp)) and (ix < np.nanmax(wtemp)):
                            # disable button event connection
                            self.disconnect(self.clickCid)

                            # disconnect cursor, and remove from plot
                            self.cursor.disconnect_events()
                            self.cursor._update()

                            # get points nearby to the click
                            nearby = np.where((wtemp > ix-10*disp_approx) &
                                              (wtemp < ix+10*disp_approx) )

                            # find if click is too close to an existing click (overlap)
                            kill = None
                            if len(self.pcent)>0:
                                for k in range(len(self.pcent)):
                                    if np.abs(self.ixlib[k]-ix)<tol:
                                        kill_d = raw_input('> WARNING: Click too close to existing point. To delete existing point, enter "d"')
                                        print('You entered:', kill_d, kill_d=='d')
                                        if kill_d=='d':
                                            kill = k
                                if kill is not None:
                                    del(self.pcent[kill])
                                    del(self.wcent[kill])
                                    del(self.ixlib[kill])


                            # If there are enough valid points to possibly fit a peak too...
                            if (len(nearby[0]) > 4) and (kill is None):
                                print('Fitting Peak')
                                imax = np.nanargmax(slice[nearby])

                                pguess = (np.nanmax(slice[nearby]), np.median(slice), xraw[nearby][imax], 2.)
                                try:
                                    popt,pcov = curve_fit(_gaus, xraw[nearby], slice[nearby], p0=pguess)
                                    self.ax.plot(wtemp[int(popt[2])], popt[0], 'r|')
                                except ValueError:
                                    print('> WARNING: Bad data near this click, cannot centroid line with Gaussian. I suggest you skip this one')
                                    popt = pguess
                                except RuntimeError:
                                    print('> WARNING: Gaussian centroid on line could not converge. I suggest you skip this one')
                                    popt = pguess

                                # using raw_input sucks b/c doesn't raise terminal, but works for now
                                try:
                                    number=float(raw_input('> Enter Wavelength: '))
                                    print('Pixel Value:',popt[2])
                                    self.pcent.append(popt[2])
                                    self.wcent.append(number)
                                    self.ixlib.append((ix))
                                    self.ax.plot(wtemp[int(popt[2])], popt[0], 'ro')
                                    print('  Saving '+str(number))
                                except ValueError:
                                    print("> Warning: Not a valid wavelength float!")

                            elif (kill is None):
                                print('> Error: No valid data near click!')

                            # reconnect to cursor and button event
                            self.clickCid = self.connect("button_press_event",self.OnClick)
                            self.cursor = Cursor(self.ax, useblit=False,horizOn=False,
                                             color='red', linewidth=1 )
                    else:
                        pass

            # run the interactive program
            wavefit = InteracWave()
            plt.show() #activate the display - GO!

            # how I would LIKE to do this interactively:
            # inside the interac mode, do a split panel, live-updated with
            # the wavelength solution, and where user can edit the fit_order

            # how I WILL do it instead
            # a crude while loop here, just to get things moving

            # after interactive fitting done, get results fit peaks
            pcent = np.array(wavefit.pcent,dtype='float')
            wcent = np.array(wavefit.wcent, dtype='float')

            print('> You have identified '+str(len(pcent))+' lines')
            lout = open(calimage+'.lines', 'w')
            lout.write("# This file contains the HeNeAr lines identified [manual] Columns: (pixel, wavelength) \n")
            for l in range(len(pcent)):
                lout.write(str(pcent[l]) + ', ' + str(wcent[l])+'\n')
            lout.close()


        if (len(previous)>0):
            pcent, wcent = np.loadtxt(previous, dtype='float',
                                      unpack=True, skiprows=1,delimiter=',')


        #---  FIT SMOOTH FUNCTION ---

        # fit polynomial thru the peak wavelengths
        # xpix = (np.arange(len(slice))-len(slice)/2)
        # coeff = np.polyfit(pcent-len(slice)/2, wcent, fit_order)
        xpix = np.arange(len(slice))
        coeff = np.polyfit(pcent, wcent, fit_order)
        wtemp = np.polyval(coeff, xpix)

        done = str(fit_order)
        while (done != 'd'):
            fit_order = int(done)
            coeff = np.polyfit(pcent, wcent, fit_order)
            wtemp = np.polyval(coeff, xpix)

            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            ax1.plot(pcent, wcent, 'bo')
            ax1.plot(xpix, wtemp, 'r')

            ax2.plot(pcent, wcent - np.polyval(coeff, pcent),'ro')
            ax2.set_xlabel('pixel')
            ax1.set_ylabel('wavelength')
            ax2.set_ylabel('residual')
            ax1.set_title('fit_order = '+str(fit_order))

            # ylabel('wavelength')

            print(" ")
            print('> How does this look?  Enter "d" to be done (accept), ')
            print('  or a number to change the polynomial order and re-fit')
            print('> Currently fit_order = '+str(fit_order))
            print(" ")

            plt.show(block=False)

            _CheckMono(wtemp)

            print(' ')
            done = str(raw_input('ENTER: "d" (done) or a # (poly order): '))


    # = = = = = = = = = = = = = = = = = =
    # now rough wavelength is found, either via interactive or auto mode!

    #-- SECOND PASS
    second_pass = False
    if second_pass is True:
        linelists_dir = os.path.dirname(os.path.realpath(__file__))+ '/resources/linelists/'
        hireslinelist = 'henear.dat'
        linewave2 = np.loadtxt(os.path.join(linelists_dir, hireslinelist), dtype='float',
                               skiprows=1, usecols=(0,), unpack=True)

        tol2 = tol # / 2.0
        print(wtemp)
        """
        plt.figure(999)
        plt.plot(wtemp, slice)
        plt.show()
        """
        pcent_pix2, wcent_pix2 = find_peaks(wtemp, slice, pwidth=10, pthreshold=80)
        print(pcent_pix2)
        print(wcent_pix2)

        pcent2 = []
        wcent2 = []
        # sort from center wavelength out
        ss = np.argsort(np.abs(wcent_pix2-wcen_approx))

        # coeff should already be set by manual or interac mode above
        # coeff = np.append(np.zeros(fit_order-1),(disp_approx*sign, wcen_approx))
        for i in range(len(pcent_pix2)):
            xx = pcent_pix2-len(slice)/2
            wcent_pix2 = np.polyval(coeff, xx)

            if (min((np.abs(wcent_pix2[ss][i] - linewave2))) < tol2):
                # add corresponding pixel and *actual* wavelength to output vectors
                pcent2 = np.append(pcent2, pcent_pix2[ss[i]])
                wcent2 = np.append(wcent2, linewave2[np.argmin(np.abs(wcent_pix2[ss[i]] - linewave2))] )
                #print(pcent2, wcent2)

            #-- update in real time. maybe not good for 2nd pass
            # if (len(pcent2)>fit_order):
            #     coeff = np.polyfit(pcent2-len(slice)/2, wcent2, fit_order)

            if display is True:
                plt.figure()
                plt.plot(wtemp, slice, 'b')
                plt.scatter(linewave2,np.ones_like(linewave2)*np.nanmax(slice),
                            marker='o',c='cyan')
                plt.scatter(wcent_pix2,np.ones_like(wcent_pix2)*np.nanmax(slice)/2.,
                            marker='*',c='green')
                plt.scatter(wcent_pix2[ss[i]],np.nanmax(slice)/2.,
                            marker='o',c='orange')
                plt.text(np.nanmin(wcent_pix2), np.nanmax(slice)*0.95, hireslinelist)
                plt.text(np.nanmin(wcent_pix2), np.nanmax(slice)/2.*1.1, 'detected lines')

                plt.scatter(wcent2,np.ones_like(wcent2)*np.nanmax(slice)*1.05,marker='o',c='red')
                plt.text(np.nanmin(wcent_pix2), np.nanmax(slice)*1.1, 'matched lines')

                plt.ylim((np.nanmin(slice), np.nanmax(slice)*1.2))
                plt.xlim((min(wtemp),max(wtemp)))
                plt.show()
        wtemp = np.polyval(coeff, (np.arange(len(slice))-len(slice)/2))

        lout = open(calimage+'.lines2', 'w')
        lout.write("# This file contains the HeNeAr lines identified [2nd pass] Columns: (pixel, wavelength) \n")
        for l in range(len(pcent2)):
            lout.write(str(pcent2[l]) + ', ' + str(wcent2[l])+'\n')
        lout.close()

        xpix = np.arange(len(slice))
        coeff = np.polyfit(pcent2, wcent2, fit_order)
        wtemp = np.polyval(coeff, xpix)


        #---  FIT SMOOTH FUNCTION ---
        if interac is True:
            done = str(fit_order)
            while (done != 'd'):
                fit_order = int(done)
                coeff = np.polyfit(pcent2, wcent2, fit_order)
                wtemp = np.polyval(coeff, xpix)

                fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
                ax1.plot(pcent2, wcent2, 'bo')
                ax1.plot(xpix, wtemp, 'r')

                ax2.plot(pcent2, wcent2 - np.polyval(coeff, pcent2),'ro')
                ax2.set_xlabel('pixel')
                ax1.set_ylabel('wavelength')
                ax2.set_ylabel('residual')
                ax1.set_title('2nd pass, fit_order = '+str(fit_order))

                # ylabel('wavelength')

                print(" ")
                print('> How does this look?  Enter "d" to be done (accept), ')
                print('  or a number to change the polynomial order and re-fit')
                print('> Currently fit_order = '+str(fit_order))
                print(" ")

                plt.show(block=False)

                _CheckMono(wtemp)

                print(' ')
                done = str(raw_input('ENTER: "d" (done) or a # (poly order): '))

    #-- trace the peaks vertically --
    xcent_big, ycent_big, wcent_big = line_trace(img, pcent, wcent,
                                                 fmask=fmask, display=display)

    #-- turn these vertical traces in to a whole chip wavelength solution
    wfit = lines_to_surface(img, xcent_big, ycent_big, wcent_big,
                            mode=mode, fit_order=fit_order)

    print('This is the FIT:', wfit)
    print(wfit.shape)
    plt.figure(1)
    plt.plot(wfit[0])
    plt.show()
    return wfit