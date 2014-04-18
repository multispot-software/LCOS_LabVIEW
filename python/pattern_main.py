import numpy as np
from patternlib_simple import get_spot_pattern, fprint, fprint_kw


def pattern_wrapper(Xm, Ym, f, wl, phi_max, phase_factor, center_spot,
                    darken_cspot, lw, vmax, ph_wrapping, pad, dark_all, nospot):
    """
    Wrapper function to generate the pattern using input parameters from LV.

    Parameters:
    `Xm`, `Ym`: 2D arrays of spot centers
    `f`: (float) focal length of generated lenses (m)
    `wl`: (float)wavelength of laser (m)
    `phi_max`: (float) constant phase to add to the pattern (in pi units)
    `phase_factor`: (uint8) the 8-bit value (grey scale) corresponding to pi
    `center_spot`: spot number considered as center
    `darken_cspot`: (bool) if True darken the center spot
    `lw`: (uint) line-width of the steering pattern
    `vmax` (uint8) max value for the steering pattern (min value is 0)
    `ph_wrapping`: (uint8) if "true" wraps all the phase values < 0 to 0..2pi
    `pad`: (uint) #pixels of padding for the lens pattern
    `dark_all` (uint8) if "true" return an array of zeros
    `nospot` (uint8) if "true" return only the steering pattern
    """
    if lw < 0: # avoids divisions by zero
        lw = 1
        vmax = 0

    Xm, Ym = np.array(Xm), np.array(Ym)
    lens_params = dict(wl=wl, f=f, phi_max=phi_max, phase_factor=phase_factor,
                       ph_wrapping=bool(ph_wrapping))
    steer_params = dict(vmax=int(vmax), lw=int(lw), horizontal=True)

    fprint('')
    fprint_kw(Xm=Xm, Ym=Ym)
    fprint_kw(**lens_params)
    fprint_kw(**steer_params)
    fprint_kw(pad=pad,
              darken_cspot=darken_cspot, CD=(0, center_spot),
              dark_all=dark_all, nospot=nospot)
    #a = (np.arange(800*600).reshape(800, 600).T*255./(800*600)).tolist()
    a = get_spot_pattern(Xm, Ym, lens_params, steer_params, pad=pad,
                        darken_cspot=darken_cspot, CD=(0, center_spot),
                        dark_all=dark_all, nospot=nospot)
    a = a.tolist()
    return a


if __name__ == '__main__':
    # Run the module to execute the following test
    phi0=4; wl=532e-9; f=32e-3; phase_factor=65; center_spot=4; darken_cspot=0;
    lw=1; vmax=120; ph_wrapping=0; pad=1; dark_all=0; nospot=0

    Xm = np.r_[21.72,  44.87,  68.20,  91.45, 114.72, 137.87, 160.97, 184.04]
    Ym = np.r_[13.84,  13.78,  13.79,  13.83,  13.75,  13.79,  13.74,  13.68]

    a = pattern_wrapper(Xm, Ym, f, wl, phi0, phase_factor, center_spot,
                     darken_cspot, lw, vmax, ph_wrapping, pad, dark_all, nospot)

