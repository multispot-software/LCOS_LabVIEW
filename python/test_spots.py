from __future__ import division

from gen_image_qt4 import *
import patternlib_simple as pat
from patternlib_simple import LCOS

import numpy as np
import matplotlib.pyplot as plt
imshow = plt.imshow

def single_pattern_steer(xm, ym, mask=None, a=None,
                         steer_slope=1, steer_offset=0):
    """Single spot  linear steering phase pattern (1 = pi).

    Centered at (xm,ym) in LCOS pixel units.
    The pattern is computed on the subset of LCOS pixels selected by mask.

    `a` is an (optional) array in which to store the pattern.
    `steer_slope`: max linear phase (1 = pi) for spot steering may be < 0
    """
    if a is None: a = LCOS.zeros()
    if mask is None: mask = np.ones(a.shape, dtype=bool)
    ycenter = 0.5*(LCOS.YL[mask].max() + LCOS.YL[mask].min())
    a[mask] += (LCOS.YL[mask] - ycenter)*steer_slope + steer_offset
    return a

def pshow(a):
    im = plt.imshow(a, interpolation='nearest', cmap=plt.gray())
    plt.colorbar()
    return im

def zeros_lcos(dtye=None):
    return np.zeros((600, 800), dtype=dtype)

lens_params = dict(wl=532e-9, f=32e-3, phi_max=4, phase_factor=65,
                   ph_wrapping=True)
steer_params = dict(vmax=120, lw=1, horizontal=False)

## Low-level test
#Xm = np.array([[200, 300], [200, 300]])
#Ym = np.array([[100, 100], [200, 200]])
#C = pat.get_spot_limits(Xm, Ym)
#a = pat.pattern_sep(Xm, Ym, C, dtype=np.uint8, **lens_params)

# Low-level test
Xm = np.array([[400]])
Ym = np.array([[450]])
pat_size = 25
half_pat = pat_size/2

C = np.zeros((Xm.shape[0], Xm.shape[1], 4))
C[...] = Xm[0,0]-half_pat, Xm[0,0]+half_pat, Ym[0,0]-half_pat, Ym[0,0]+half_pat

a = pat.pattern_sep(Xm, Ym, C, dtype=np.uint8, **lens_params)

im = pshow(a)
show_pattern_twin(a, im)
