#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 15:22:45 2026

@author: javfdez
"""

def import_galaxy_data(galaxy_number):
    from galaxy import Ring
    from galaxy import Galaxy
    from astropy.io import fits
    import numpy as np

    result_folder_path = f'weaveprocessed/cube_{galaxy_number}'
    map_path = f'WEAVE/cube{galaxy_number}/map_cube{galaxy_number}.npz'
    redshift_path = f'WEAVE/cube{galaxy_number}/dataCube{galaxy_number}/redshift.txt'

    cube = Galaxy()
    cube.segmap = np.load(map_path)['arr_0']
    cube.redshift = np.loadtxt(redshift_path).item()

    for i in range(1,11):
        datafile = result_folder_path+f'/results_bin_{i}/postprocess_results.fits'
        data = fits.open(datafile)
        vals = data[3].data
        cube.add_ring(i, Ring(
            logssfr8 = vals.field(21)[1:4],
            logssfr9 = vals.field(15)[1:4],
            stellar_mass = vals.field(23)[1:4],
            metallicity = vals.field(9)[1:4],
            Av = vals.field(5)[1:4]
            ))
        data.close()
    return cube

    # for i in range(1,11):
    #     datafile = result_folder_path+f'/results_bin_{i}/postprocess_results.fits'
    #     data = fits.open(datafile)
    #     cube.add_ring(i, Ring(
    #         ssfrlog = [
    #             data[1].header['P010MN'], #ssfrlog_8.00
    #             data[1].header['P009MN'], #ssfrlog_8.48
    #             data[1].header['P008MN'], #ssfrlog_8.70
    #             data[1].header['P007MN'], #ssfrlog_9.00
    #             data[1].header['P006MN'], #ssfrlog_9.48
    #             data[1].header['P005MN'], #ssfrlog_9.70
    #             ],
    #         stellar_mass = data[1].header['P011MN'],
    #         metallicity = data[1].header['P004MN'],
    #         Av = data[1].header['P002MN']
    #         ))
    #     data.close()
    # return cube