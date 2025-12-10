# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 17:00:25 2025

@author: javfd
"""

def flux_show(flux):
    '''
    Realiza un gráfico del flujo integrado de una imagen
    
    Parameters:
    - flux : array de flujos
    '''
    from astropy.visualization import simple_norm
    import matplotlib.pyplot as plt
    
    inv_flux = np.transpose(flux)
    
    plt.imshow(inv_flux, cmap = 'Spectral', norm=simple_norm(inv_flux, 'log'), origin='lower', aspect='equal')
def bocao(arr, x1, y1, x2, y2):
    """
    Pone a cero un rectángulo definido en coordenadas (x, y).
    """

    out = arr.copy()

    # Ordenamos correctamente
    xmin, xmax = sorted([x1, x2])
    ymin, ymax = sorted([y1, y2])

    # Convertimos a (fila=y, columna=x)
    out[ymin:ymax+1, xmin:xmax+1] = 0

    return out


def segmentation_map(image, brightness_ranges):
    """
    Genera un segmentation map a partir de una imagen y 10 rangos de brillo.

    Parameters
    ----------
    image : np.ndarray
        Imagen 2D de flujo.
    brightness_ranges : list of tuples
        Lista de 10 tuplas (Bmin, Bmax).

    Returns
    -------
    segmap : np.ndarray
        Mapa 2D donde cada píxel contiene un entero de 1 a 10,
        indicando la máscara correspondiente. 0 = no pertenece a ninguna.
    """

    if len(brightness_ranges) != 10:
        raise ValueError("Debes proporcionar exactamente 10 rangos (10 máscaras).")

    # Inicialmente, ningún píxel pertenece a nada
    seg = np.zeros_like(image, dtype=np.uint8)

    for i, (Bmin, Bmax) in enumerate(brightness_ranges):
        mask = (image >= Bmin) & (image <= Bmax)

        # Etiquetamos la máscara con valor i+1 (1..10)
        seg[mask] = i + 1

    return seg


def sum_spectra(cube, segmap, n_masks=10):
    """
    Suma los espectros contenidos en cada máscara para cubos con formato (Nλ, Nx, Ny).

    Parameters
    ----------
    cube : np.ndarray
        Cubo espectral con shape (Nlambda, Nx, Ny)
    segmap : np.ndarray
        Mapa de segmentos con valores 0..n_masks
        Debe tener shape (Nx, Ny)
    n_masks : int
        Número de máscaras, por defecto 10.

    Returns
    -------
    spectra : dict
        Diccionario con:
            spectra[1] = espectro integrado de máscara 1 (Nlambda)
            ...
            spectra[n_masks]
    """

    Nlambda, Nx, Ny = cube.shape
    spectra = {}

    for i in range(1, n_masks + 1):
        mask = (segmap == i)

        if not np.any(mask):
            spectra[i] = np.zeros(Nlambda)
            continue

        # Extraemos los píxeles pertenecientes a la máscara para cada λ
        # cube[:, mask] => matriz de shape (Nlambda, N_pixels_mask)
        masked_cube = cube[:, mask]

        # Sumamos sobre los píxeles
        spectra[i] = masked_cube.sum(axis=1)

    return spectra

if __name__ == "__main__":

    from astropy.io import fits
    from astropy.visualization import simple_norm
    import matplotlib.pyplot as plt
    import numpy as np
    
    filered = "WEAVE/stackcube_3120047.fit"
    fileblue ="WEAVE/stackcube_3120048.fit"

    red = fits.open(filered)
    rd = red[1].data
    rflux = red[6].data
    
    red.close()
    
    blue = fits.open(fileblue)
    bd = blue[1].data
    bflux = blue[6].data
    
    blue.close()
    
    flux = rflux + bflux
    
    #Limpieza de estrellas
    
    cflux = bocao(flux, 27, 35, 60, 15)
    cflux = bocao(cflux, 54, 173, 64, 165)
    cflux = bocao(cflux, 152, 127, 163, 117)
    
    bright_tuples =[
        (9e6, 1e9),
        (6.1e6,9e6),
        (5e6, 6.1e6),
        (4e6, 5e6),
        (3e6, 4e6),
        (2e6, 3e6),
        (1e6,2e6),
        (5e5, 1e6),
        (2e5, 5e5),
        (1e5,2e5)
        ]
    
    seg = segmentation_map(cflux, bright_tuples)
    red_spectra = sum_spectra(rd, seg)
    blue_spectra = sum_spectra(bd, seg)
    
    Rrange = (5790, 9590)
    Brange = (3660, 6060)
    
    Rwav = np.arange(Rrange[0], Rrange[1], (Rrange[1] - Rrange[0]) / len(rd))
    Bwav = np.arange(Brange[0], Brange[1], (Brange[1] - Brange[0]) / len(bd))
    
    plt.figure(1)
    flux_show(cflux)
    
    mask_plot = True
    plt.figure(2)
    if mask_plot:
        plt.imshow(np.transpose(seg), origin='lower', cmap='tab10')
        plt.show()
            
    mask2plot = 2
    plt.figure(3)
    plt.plot(Rwav, red_spectra[mask2plot],Bwav, blue_spectra[mask2plot])
    plt.xlabel('Wavelength (Armstrong)')
    plt.ylabel('Intensity (au)')
    plt.grid(True)
    
