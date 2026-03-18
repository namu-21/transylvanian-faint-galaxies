#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 18:43:38 2026

@author: javfdez
"""
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

@dataclass
class Ring:
    logssfr9: np.ndarray(3)
    logssfr8: np.ndarray(3)
    stellar_mass: np.ndarray(3)
    metallicity: np.ndarray(3)
    Av: np.ndarray(3)

class Galaxy:
    segmap: np.ndarray
    redshift: np.float64

    def __init__(self):
        self.rings = {}

    def __post_init__(self):
        self.segmap = self.segmap.astype(int)

    @property
    def angular_distance(self):
        c = 299792.458 #km/s
        H0 = 0.070 #km/s/kpc
        return self.redshift*c/H0

    def __getitem__(self, index):
        return self.rings[index]

    def add_ring(self, index, ring):
        self.rings[index] = ring

    def metallicity_array(self,index):
        indices = sorted(self.rings.keys())
        return np.array([self.rings[i].metallicity[index] for i in indices])

    def mass_array(self,index):
        indices = sorted(self.rings.keys())
        return np.array([self.rings[i].stellar_mass[index] for i in indices])

    def ssfr8_array(self,index):
        indices = sorted(self.rings.keys())
        return np.array([self.rings[i].logssfr8[index] for i in indices])

    def ssfr9_array(self,index):
        indices = sorted(self.rings.keys())
        return np.array([self.rings[i].logssfr9[index] for i in indices])

    def plot_mass_map(self, galaxy_index):
        seg = self.segmap
        massmap = np.full(seg.shape, np.nan)
        valid = seg > 0
        massmap[valid] = self.mass_array()[seg[valid] - 1]

        fig, ax = plt.subplots()
        mapped = ax.imshow(massmap, cmap = 'magma')
        fig.colorbar(mapped, ax = ax)
        ax.set_title('$\Sigma^*$ galactic map')
        plt.show()

    def plot_maps(self, galaxy_index):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        seg = self.segmap
        valid = seg > 0
        ssfr8_map = np.full(seg.shape, np.nan)
        ssfr9_map = np.full(seg.shape, np.nan)
        massmap = np.full(seg.shape, np.nan)

        ssfr8_map[valid] = self.ssfr8_array(1)[seg[valid] - 1]
        ssfr9_map[valid] = self.ssfr9_array(1)[seg[valid] - 1]
        massmap[valid] = self.mass_array(1)[seg[valid] - 1]
        fig, ax = plt.subplots(1,3, figsize = (9,3))

        ax[0].set_title(r'$\Sigma^* (M_{\odot}/kpc)$')
        ax[1].set_title(r'log$_{10}$(sSFR$_8$)')
        ax[2].set_title(r'log$_{10}$(sSFR$_9$)')
        cmap = 'Spectral'
        vmin = -11
        vmax = -9
        fiberarea = 2*np.log10(np.pi*(1.3*np.sin(1*u.arcsec)*self.angular_distance))
        masses = ax[0].imshow(massmap-fiberarea, cmap='magma')
        ssfr8 = ax[1].imshow(ssfr8_map, cmap=cmap, vmin = vmin, vmax = vmax)
        ssfr9 = ax[2].imshow(ssfr9_map, cmap=cmap, vmin = vmin, vmax = vmax)

        for i, im in enumerate([masses, ssfr8, ssfr9]):
            divider = make_axes_locatable(ax[i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            ax[i].set_aspect('equal')
            fig.colorbar(im, cax=cax)

        fig.suptitle(f'Galaxia {galaxy_index}')

        plt.tight_layout()
        plt.show()

