import importlib

modules = ['os', 'numpy', 'math', 'subprocess', 'fnmatch', 'logging', 'pylab', 'errno', 'datetime', 
           'warnings', 'astropy', 'scipy', 'argparse', 'statsmodels', 'PIL', 'coloredlogs']
missingModules = []

def is_missing():
    for m in modules:
        
        try:
        	package1 = importlib.util.find_spec(m)
        except ImportError:
            print('Module %s not installed' % m)
            missingModules.append(m)

        if package1 == None:
            print('Module %s not installed' % m)
            missingModules.append(m)

    return missingModules

