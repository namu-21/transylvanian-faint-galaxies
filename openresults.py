#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 11:39:18 2026

@author: javfdez
"""

from astropy.io import fits
import numpy as np

galaxy_number = 5
result_folder_path = f'weaveprocessed/cube_{galaxy_number}'
datafile = result_folder_path+'/results_bin_1/postprocess_results.fits'
data = fits.open(datafile)
vals = data[3].data
# usar vals.field(i) para acceder