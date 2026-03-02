#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 8 17:00:25 2025

@author: javfd

"""

halpha = 6562.8
def redshift(line):
    return (line-halpha)/halpha

def flux_show(flux):
    from astropy.visualization import simple_norm
    import matplotlib.pyplot as plt

    plt.imshow(flux, cmap = 'Spectral', norm=simple_norm(flux, 'log'), origin='lower', aspect='equal')

def bocao(arr, p1, p2):
    '''
    p1 = (y1,x1) tuple del punto inferior izquierdo del paralelogramo que se quiere borrar
    p2 = (y2,x2) tuple del punto superior derecho del paralelogramo que se quiere borrar
    '''
    out = arr.copy()

    out[p1[1]:p2[1]+1, p1[0]:p2[0]+1] = 0

    return out

def segmentation_map(image, brightness_ranges):
    if len(brightness_ranges) != 10:
        raise ValueError("No se han dado diez rangos")

    segmap = np.zeros_like(image, dtype=np.uint8)

    for i, (Bmin, Bmax) in enumerate(brightness_ranges):
        mask = (image >= Bmin) & (image <= Bmax)

        segmap[mask] = i + 1

    return segmap

def avg_spectra(cube, segmap, n_masks=10, do_sum = False):
    Nlambda, Nx, Ny = cube.shape
    spectra = {}

    for i in range(1, n_masks + 1):
        mask = (segmap == i)

        if not np.any(mask):
            spectra[i] = np.zeros(Nlambda)
            continue

        masked_cube = cube[:, mask]
        if do_sum:
            spectra[i] = masked_cube.sum(axis=1)
        else:
            spectra[i] = masked_cube.mean(axis=1)

    return spectra

if __name__ == "__main__":

    from astropy.io import fits
    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.wcs import WCS

    # Loading the data from fits file

    filered = "stackcube_3132971.fit"
    fileblue ="stackcube_3132972.fit"

    # Red part of the spectrum

    red = fits.open(filered)
    rdata = red[1].data         # Cube data
    rflux = red[6].data         # Collapsed flux
    rerr = red[2].data          # Inverse variance loading
    rwcs = WCS(red[1].header)   # Extracting wcs variables
    rsens = red[5].data         # Sensibility function

    # Red wavelength array extraction
    Rwav = range(rwcs.spectral.array_shape[0])
    Rwav = rwcs.spectral.pixel_to_world(Rwav).value
    red.close()

    # Blue part of the spectrum (same structure as red)
    blue = fits.open(fileblue)
    bdata = blue[1].data
    bflux = blue[6].data
    berr = blue[2].data
    bsens = blue[5].data

    bwcs = WCS(blue[1].header)
    Bwav = range(bwcs.spectral.array_shape[0])
    Bwav = bwcs.spectral.pixel_to_world(Bwav).value

    blue.close()

    # Adding flux regions for total flux (slight overlap)
    flux = rflux + bflux

    #%%

    #Star flux cleaning

    cflux = bocao(flux, (56,128), (70,139))
    cflux = bocao(cflux, (152,90), (162,110))
    cflux = bocao(cflux, (140,32), (154,39))
    cflux = bocao(cflux, (1,36), (26,57))
    cflux = bocao(cflux, (68,8), (77,16))
    cflux = bocao(cflux, (90,131), (109,142))
    cflux = bocao(cflux, (152,53), (160,56))


    flux_show(cflux)

    #%%
    #Brightness regions selection (made by eye)

    bright_tuples =[
        (6.4e6, 1e9),
        (1.65e6,6.4e6),
        (1.19e6, 1.65e6),
        (9.5e5, 1.19e6),
        (8.0e5, 9.5e5),
        (6.9e5, 8.0e5),
        (4.1e5,6.9e5),
        (2.35e5, 4.1e5),
        (1.34e5, 2.35e5),
        (7e4,1.34e5)
        ]

    segmap = segmentation_map(cflux, bright_tuples)
    red_spectra = avg_spectra(rdata, segmap)
    blue_spectra = avg_spectra(bdata, segmap)

    # Turning inverse variance into sigma

    red_err = avg_spectra(np.divide(1, np.sqrt(rerr), where = rerr!=0), segmap, do_sum = False)
    blue_err = avg_spectra(np.divide(1, np.sqrt(berr), where = berr!=0), segmap, do_sum = False)

    # Calculation of correction over the errors by estimating median error intensity

    red_correction = np.ones(11)
    for i in range(1,11):
        med = np.sqrt(np.nanmedian(0.5*(red_spectra[i][1:]-red_spectra[i][:-1])**2))
        mu = np.nanmedian(red_err[i])
        red_correction[i] = med/mu

    blue_correction = np.ones(11)
    for i in range(1,11):
        med = np.sqrt(0.5*np.nanmedian((blue_spectra[i][1:]-blue_spectra[i][:-1])**2))
        mu = np.nanmedian(blue_err[i])
        blue_correction[i] = med/mu

    # Applying the sensitivity function to calculated spectra for calibration

    for i in range(1,11):
        red_spectra[i] = rsens*red_spectra[i]
        red_err[i] = rsens*red_err[i]

        blue_spectra[i] = bsens*blue_spectra[i]
        blue_err[i] = bsens*blue_err[i]

    # Plot of total flux
    flux_plot = True
    if flux_plot:
        plt.figure(1)
        flux_show(cflux)

    # Plot of masked
    mask_plot = True
    if mask_plot:
        plt.figure(2)
        plt.imshow(segmap, origin='lower')
        plt.show()

    save_segmap = False
    if save_segmap:
        np.savez('map_cubei.npz', segmap)

    # Plot of selected mask spectrum
    mask = 1
    def plot_spectral(mask):
        fig, ax = plt.subplots()
        ax.plot(Rwav*1e10, red_spectra[mask],'r')
        ax.fill_between(Rwav*1e10, red_spectra[mask]-red_err[mask], red_spectra[mask]+red_err[mask], color ='red', alpha = 0.3)
        ax.plot(Bwav*1e10, blue_spectra[mask],'b')
        ax.fill_between(Bwav*1e10, blue_spectra[mask]-blue_err[mask], blue_spectra[mask]+blue_err[mask], color = 'blue', alpha = 0.3)
        plt.xlabel('Wavelength (Armstrong)')
        plt.ylabel('Intensity (erg/s cm^2)')
        plt.grid(True)
        plt.title(f'Spectra of mask {mask}')

    def write_redshift(line):
        with open('redshift.txt', 'w') as f:
            f.write(f'{redshift(line)}')
    # Function for exporting results into text files
    def write_results():
        for i in range(1,11):
            with open(f'red_galaxy_mask_{i}.txt', 'w') as f:
                with open(f'red_weights_mask_{i}.txt', 'w') as w:
                    for j in range(len(Rwav)):
                        if red_spectra[i][j] == 0.0:
                            weight = 0.0
                        else:
                            weight = 1.0
                        f.write(f'{Rwav[j]} {red_spectra[i][j]} {red_err[i][j]}\n')
                        w.write(f'{weight}\n')

        for i in range(1,11):
            with open(f'blue_galaxy_mask_{i}.txt', 'w') as f:
                with open(f'blue_weights_mask_{i}.txt', 'w') as w:
                    for j in range(len(Bwav)):
                        if blue_spectra[i][j] == 0.0:
                            weight = 0.0
                        else:
                            weight = 1.0
                        f.write(f'{Bwav[j]} {blue_spectra[i][j]} {blue_err[i][j]}\n')
                        w.write(f'{weight}\n')
