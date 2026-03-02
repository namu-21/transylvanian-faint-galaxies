#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 18:43:38 2026

@author: javfdez
"""
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class Ring:
    ssfrlog: np.ndarray #(8, 8.48, 8.70, 9.00, 9.48, 9.70)
    stellar_mass: tuple #(red, blue)
    metallicity: float
    Av: float

    def __post_init__(self):
        self.ssfrlog = np.array(self.ssfrlog)

    def plot_ssfrevo(self):
        logtime = np.array([8, 8.48, 8.70, 9.00, 9.48, 9.70])
        time_gyr = 10**logtime/1e9
        plt.figure(figsize=(5,5), layout='constrained')
        plt.plot(time_gyr, self.ssfrlog, '+', ms = 11)
        plt.xlabel('Time ago [Gyr]')
        plt.ylabel('logssfr/logyr')
        plt.grid()
        plt.show()

class Galaxy:
    segmap: np.ndarray
    redshift: np.float64

    def __init__(self):
        self.rings = {}

    def __post_init__(self):
        self.segmap = self.segmap.astype(int)

    def __getitem__(self, index):
        return self.rings[index]

    def add_ring(self, index, ring):
        self.rings[index] = ring

    def metallicity_array(self):
        indices = sorted(self.rings.keys())
        return np.array([self.rings[i].metallicity for i in indices])

    def mass_array(self):
        indices = sorted(self.rings.keys())
        return np.array([self.rings[i].stellar_mass for i in indices])

    def ssfr8_array(self):
        indices = sorted(self.rings.keys())
        return np.array([self.rings[i].ssfrlog[0] for i in indices])

    def ssfr9_array(self):
        indices = sorted(self.rings.keys())
        return np.array([self.rings[i].ssfrlog[3] for i in indices])

    def plot_mass_map(self):
        seg = self.segmap
        massmap = np.full(seg.shape, np.nan)
        valid = seg > 0
        massmap[valid] = self.mass_array()[seg[valid] - 1]

        fig, ax = plt.subplots()
        mapped = ax.imshow(massmap, cmap = 'magma')
        fig.colorbar(mapped, ax = ax)
        ax.set_title('$\Sigma^*$ galactic map')
        plt.show()

    def plot_ssfr9_map(self):
        seg = self.segmap
        ssfr9_map = np.full(seg.shape, np.nan)
        valid = seg > 0
        ssfr9_map[valid] = 10**self.ssfr9_array()[seg[valid] - 1]

        fig, ax = plt.subplots()
        mapped = ax.imshow(ssfr9_map, cmap = 'seismic')
        fig.colorbar(mapped, ax = ax)
        ax.set_title('sSFR at 9 logyr galactic map')
        plt.show()

    def plot_ssfr8_map(self):
        seg = self.segmap
        ssfr8_map = np.full(seg.shape, np.nan)
        valid = seg > 0
        ssfr8_map[valid] = 10**self.ssfr8_array()[seg[valid] - 1]

        fig, ax = plt.subplots()
        mapped = ax.imshow(ssfr8_map, cmap = 'plasma')
        fig.colorbar(mapped, ax = ax)
        ax.set_title('sSFR at 8 logyr galactic map')
        plt.show()

    def plot_metallicity(self):
        indices = sorted(self.rings.keys())
        metallicities = self.metallicity_array()

        plt.figure(figsize=(5,5), layout='constrained')
        plt.plot(indices, metallicities, '+', ms = 11)
        plt.xlabel("Ring index (approx. radius proxy)")
        plt.ylabel("Metallicity (in solar Z)")
        plt.grid()
        plt.show()

    def plot_mass_gradient(self):

        indices = sorted(self.rings.keys())

        masses = [(self.rings[i].stellar_mass) for i in indices]

        plt.figure(figsize=(5,5), layout='constrained')
        plt.plot(indices, masses, '+', ms = 11)

        plt.xlabel("Ring index (approx. radius proxy)")
        plt.ylabel("Stellar Mass")
        plt.grid()
        plt.show()

    def plot_ssfr9(self):
        indices = sorted(self.rings.keys())
        ssfr9 = np.array([self.rings[i].ssfrlog[3] for i in indices])
        plt.figure(figsize=(5,5), layout='constrained')
        plt.plot(indices, ssfr9, '+', ms = 11)
        plt.xlabel("Ring index (approx. radius proxy)")
        plt.ylabel("sSFR over 9 logyr")
        plt.grid()
        plt.show()

    def plot_ssfr_all(self):

        logtime = np.array([8, 8.48, 8.70, 9.00, 9.48, 9.70])
        time_gyr = 10**logtime
        plt.figure(figsize=(5,5), layout='constrained')

        for i in sorted(self.rings.keys()):
            ring = self.rings[i]
            plt.semilogx(time_gyr, ring.ssfrlog,'+--', label=f"Ring {i}", ms = 11)

        plt.xlabel('Time ago (Gyr)')
        plt.ylabel('log10(sSFR)')
        plt.grid()
        plt.legend()

        plt.show()
