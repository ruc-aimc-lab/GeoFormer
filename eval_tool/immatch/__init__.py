import os
os.system('echo `pwd`')

try:
    from .modules.geoformer import GeoFormer

except ImportError as e:
    print(e)
    # print('Can not import sparsencnet')
    pass    
