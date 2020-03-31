import os
from pathlib import Path

p = Path(os.path.abspath(os.path.dirname(__file__)))

rgbn = str(p / 'rgbn.tif')
rgbn_suba = str(p / 'rgbn_suba.tif')
rgbn_subb = str(p / 'rgbn_subb.tif')
rgbn_20160101 = str(p / 'rgbn_20160101.tif')
rgbn_20160401 = str(p / 'rgbn_20160401.tif')
rgbn_20160517 = str(p / 'rgbn_20160517.tif')
rgbn_20170203 = str(p / 'rgbn_20170203.tif')

rgbn_time_list = [rgbn_20160101, rgbn_20160401, rgbn_20160517, rgbn_20170203]

oli = str(p / 'oli_2016_1213.tif')
