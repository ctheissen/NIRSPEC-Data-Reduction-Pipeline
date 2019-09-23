# nsdrp (modified by Chris Theissen & Dino Hsu)

This version is modified by Chris Theissen & Dino Hsu. This includes some bugs fixed in Python 3, with the modifications mentioned in the codes. Ideally, this version of NSDRP needs to be processed together with [nirspec_pip](https://github.com/chihchunhsu/nirspec_pip).

PREREQUISITES:

1)    NSDRP was developed with Anaconda python and that distribution is recommended, but can be run with Ureka or other python installations. Anaconda comes installed with the required modules: numpy, os, math, subprocess, fnmatch, logging, pylab, errno, datetime, warnings, astropy, scipy, argparse, and statsmodels. Depending on your python distribution of choice, you may be required to manually download and install one or more of these modules.

2)    Verify correct installation by typing "python" on the command line. The terminal will display the version number and the distribution, i.e. Python2.7.8, Anaconda 2.1.0, etc. To verify installation of the required python modules, please refer to step one in the EXECUTING NSDRP section below.

3)    The NSDRP software is maintained on GitHub. Download the code from https://github.com/ctheissen/NIRSPEC-Data-Reduction-Pipeline.git 
and install by decompressing the folder in the directory of your choice.

CONSTRAINTS:

The NSDRP is designed to be fully automated to run without any user's interaction. In
addition, it is intended to produce only quick-look, browse extracted spectra for the 
purpose of assessing the scientific quality and content of the raw data. It may not be able
to reduce all types of NIRSPEC data. Specifically, it can only reduce data obtained in
high-dispersion mode, and is optimized for and works best on frames having:

- single point source in slit
- reasonably bright targets with detectable continuum 
- NIRSPEC-1 through NIRSPEC-7 filters
- well-separated orders without overlapping
- sufficient exposure times (~> 30s) with detectable sky lines
	
Finally, the NSDRP should be run only on files downloaded from the Keck Observatory Archive 
(KOA, http://koa.ipac.caltech.edu/) since it needs to know the correct image type 
(on-sky object or calibrations) of the data frames. Only NIRSPEC files ingested into the archive 
are guaranteed to contain this information.

EXECUTING NSDRP:

1)    Simple command line usage: python nsdrp.py in_dir out_dir

      Note that you must provide the full directory path of the python file (nsdrp.py) if your current working 
      directory is not the source code. 
      
      Any required python packages that are not installed will be printed on the terminal console and NSDRP will halt             execution.
      
      OH sky lines are used to refine the theoretical wavelength scale. These lines are listed in the 
      ir_ohlines.dat file which is included in the download. In case you would like to use a different file, 
      it can be specified by using an additional command line argument described in advanced usage.  

      in_dir: Path to directory which contains the raw FITS files to be reduced. This must include at least 
      1 object frame and 1 flat image. For testing purposes, two sample NIRSPEC files are included in the 
      download under the directory 'rawfits'. To check that the installation is successful, try running the 
      pipeline on this directory and verify that it completes without any errors. The descriptions of the 
      output products are provided in the document NSDRP.Products.pdf.

      out_dir: Path to the root of the output directory to use for storage of generated data products. 
      This directory will be created if it does not pre-exist. 

2)    Advanced Usage
    
      Additional command line arguments are available. These are listed and described after executing: 
      python nsdrp.py -h

3)   Beta stage for upgraded NIRSPEC

      Changes have been made to make the latest version compatible with the upgraded NIRSEPC (end of 2018). 
      Wavelength solutions may or may not be accurate, contact Chris Theissen for more info.
      Additional testing for KOA mode is needed. Best results are given we used in AB pair mode.

