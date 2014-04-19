"""
The following variables must be set in the session before running the script.

Arguments:
    Xm (float array): X position of the spots in LOCOS coordinates
    Ym (float array): Y posiition of the spots in LOCOS coordinates
    phi_max (int): phi_max*pi is the max phase delay of the pattern reached
        at the center of each spot. This is the initial phase set as maximum
        but the max can wrap down to a lower value.
    wl (float): laser wavelength (in m)
    f (float): single-spot lens focal length (in m)
    phase_factor (int, 0..255): 8bit level correspondic to a phase of pi
    dark_spot (int): spot number of a spot to hide
    darken (bool): if True hides spot `dark_spot`
    lw (int): line-width of the steering pattern
    vmax (int, 0..255): max level fot the steering pattern
    PhWrap (bool): if True wrap the phase above 2*pi
    pad (int): how many pixel of padding to leave around the spot pattern
        before stating the steerring pattern
    dark_all (bool): make a black pattern
    nospot (bool): pattern with no spot (the pattern can have a steering
        pattern)
    steer_horiz (bool): if True horizontal lines in steering, else vertical
    debug (bool): if True prints debugging info to a file
"""

import sys
PATH = "C:/Data/Antonio/software/LCOS/LCOS_CompleteLV/python"
if PATH not in sys.path:
    sys.path.insert(0, PATH)

from pattern_main import pattern_wrapper

a = pattern_wrapper(Xm, Ym, f, wl, phi_max, phase_factor, center_spot,
                    darken_cspot, lw, vmax, ph_wrapping, pad, dark_all,
                    nospot, steer_horiz, debug=debug)

s = 'test'

#import numpy as np
#a = (np.arange(800*600).reshape(800, 600).T*255./(800*600)).tolist()
