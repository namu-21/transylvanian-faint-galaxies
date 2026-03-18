#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 16:11:35 2026

@author: javfdez
"""
from extfunc import import_galaxy_data
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

plt.close('all')

galaxy_select = [0,3,4,5]

cube = {}

plt.rcParams['lines.marker'] = 'x'
plt.rcParams['lines.linestyle'] = ':'
fig, axs = plt.subplots(4, figsize = (6,7), layout = 'tight')

alpha = 0.15

for i in galaxy_select:
    cube[i] = import_galaxy_data(i)
    fiberarea = 2*np.log10(np.pi*(1.3*np.sin(1*u.arcsec)*cube[i].angular_distance))
    surfacedensity = cube[i].mass_array(1)-fiberarea
    err_lower = np.sqrt( (cube[i].ssfr8_array(1) - cube[i].ssfr8_array(0))**2
                        + (cube[i].ssfr9_array(2) - cube[i].ssfr9_array(1))**2 )

    err_upper = np.sqrt( (cube[i].ssfr8_array(2) - cube[i].ssfr8_array(1))**2
                        + (cube[i].ssfr9_array(1) - cube[i].ssfr9_array(0))**2 )
    center = cube[i].ssfr8_array(1)-cube[i].ssfr9_array(1)


    axs[0].plot(surfacedensity,
                cube[i].metallicity_array(1),
                label=f'Galaxy {i}')
    axs[0].fill_between(surfacedensity,
                        cube[i].metallicity_array(0),
                        cube[i].metallicity_array(2),
                        alpha = alpha)
    axs[1].plot(surfacedensity,
                center,
                label = f'Galaxy {i}')
    axs[1].fill_between(surfacedensity,
                        center - err_lower,
                        center + err_upper,
                        alpha = alpha)

    axs[2].plot(surfacedensity,
                cube[i].ssfr9_array(1),
                label=f'Galaxy {i}')
    axs[2].fill_between(surfacedensity,
                        cube[i].ssfr9_array(0),
                        cube[i].ssfr9_array(2),
                        alpha = alpha)

    axs[3].plot(surfacedensity,
                cube[i].ssfr8_array(1),
                label=f'Galaxy {i}')
    axs[3].fill_between(surfacedensity,
                        cube[i].ssfr8_array(0),
                        cube[i].ssfr8_array(2),
                        alpha = alpha)

axs[0].set_ylabel(r'Z [$Z_{\odot}$]')
axs[1].set_ylabel(r'log$_{10}$(sSFR$_8$/sSFR$_9$)')
axs[2].set_ylabel(r'log$_{10}$(sSFR$_9$)')
axs[3].set_ylabel(r'log$_{10}$(sSFR$_8)$')
axs[0].legend()

for ax in axs:
    ax.set_xlabel(r'$\Sigma^* (M_{\odot}/kpc)$')
    ax.label_outer()
    ax.grid()
plt.show()

for i in galaxy_select:
    cube[i].plot_maps(i)
