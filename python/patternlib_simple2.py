import numpy as np


LOG_FILE = 'C:/Data/Antonio/software/LCOS/python_out.txt'
LCOS_X_SIZE, LCOS_Y_SIZE, LCOS_PIX_SIZE = 800, 600, 20e-6

# Allocate the (x, y) grid once when the module is imported
YL, XL = np.mgrid[:LCOS_Y_SIZE, :LCOS_X_SIZE]


def fprint(s):
    with open(LOG_FILE, "a") as myfile:
        myfile.write(str(s)+'\n')


def fprint_kw(**kwargs):
    with open(LOG_FILE, "a") as myfile:
        s = ', '.join([k +'='+str(v) for k,v in kwargs.items()])
        myfile.write(s+'\n')


def black_pattern(dtype=np.uint8):
    return np.zeros((LCOS_Y_SIZE, LCOS_X_SIZE), dtype=dtype)


def phase_spherical(r, f, wavelen=532e-9):
    """Phase profile (in pi units) at distance r (x,y) from spot center
    for a spherical wave converging at a distance f from the screen.
    Inputs in SI units. The formula is exact, not approximated."""
    return -(2/wavelen) * (np.sqrt(r**2 + f**2) - f)


def get_steer_pattern(lw, vmax=255, horizontal=True, debug=False):
    """Horizontal or vertical pattern for beam steering.

    Arguments:
        lw (uint): line-width in LCOS pixels for the steering pattern
        vmax (int, 0..255): max value for the steering pattern
        horizontal (bool): if True draw horizontal line, else vertical
    """
    a = black_pattern()
    if vmax > 0:
        if debug:
            fprint_kw(lw_pos_assert=(lw > 0), horizontal=horizontal)
            assert lw > 0
        row_wise_a = a
        if not horizontal:
            row_wise_a = a.T
        for i in range(0, row_wise_a.shape[0], 2*lw):
            row_wise_a[i:i+lw] = vmax
    return a


def get_spot_limits(X, Y, debug=False):
    """
    From an array of X, Y spot positions compute the boundaries for each spot.

    Returns:
        An array of shape (Y_dim, X_dim, 4), where for each spot (x, y)
        there are 4 values representing the spot boundaries in the following
        order: (xmin, xmax, ymin, ymax).
    """
    if debug:
        fprint_kw(X_Y_shape_assert=(X.shape == Y.shape))
        assert X.shape == Y.shape
    Y_dim, X_dim = X.shape
    alpha = 0.5

    ## Assumes at least one dimension > 1
    xpitch = np.mean(np.diff(X,axis=1))
    if Y_dim == 1:
        ypitch = xpitch # avoids Warning
    else:
        ypitch = np.mean(np.diff(Y,axis=0))
    if X_dim == 1: xpitch = ypitch
    if Y_dim == 1: ypitch = xpitch
    if debug:
        condition = (not np.isnan(xpitch)) and (not np.isnan(ypitch))
        fprint_kw(x_y_pitch_nan_assert=condition)
        assert condition

    X_sep = (np.diff(X, axis=1)/2. + X[:,:-1])  # separators between spots
    # For each row (Y_dim)
    Xstart = np.array([(xl[0]-alpha*xpitch)-1 for xl in X]).reshape(Y_dim,1)
    Xend = np.array([(xl[-1]+alpha*xpitch)+1 for xl in X]).reshape(Y_dim,1)
    X_R = np.concatenate((Xstart,X_sep,Xend), axis=1)

    Y_sep = (np.diff(Y, axis=0)/2. + Y[:-1,:])  # separators between spots
    # For each col (X_dim)
    Ystart = np.array([(yl[0]-alpha*ypitch)-1 for yl in Y.T]).reshape(1,X_dim)
    Yend = np.array([(yl[-1]+alpha*ypitch)+1 for yl in Y.T]).reshape(1,X_dim)
    Y_R = np.concatenate((Ystart,Y_sep,Yend), axis=0)

    # C contains the limits (4 values) for each spot (one row per spot)
    C = np.zeros((Y_dim, X_dim, 4), dtype=np.int64)
    C[:,:,0] = np.ceil(X_R[:,:-1])  # xmin
    C[:,:,1] = np.floor(X_R[:,1:])  # xmax
    C[:,:,2] = np.ceil(Y_R[:-1,:])  # ymin
    C[:,:,3] = np.floor(Y_R[1:,:])  # ymax
    return C


def single_pattern_steer(xm, ym, mask=None, a=None, phase_max=0, f=30e-3,
                         wavelen=532e-9, steer_slope=None, steer_offset=0):
    """Single spot lens and linear steering phase pattern (1 = pi).

    Centered at (xm,ym) in LCOS pixel units.
    The pattern is computed on the subset of LCOS pixels selected by mask.
    `phase_max` is constant phase to add to the pattern (in pi units).
    `a` is an (optional) array in which to store the pattern.
    `steer_slope`: max linear phase (1 = pi) for spot steering may be < 0
    """
    if a is None:
        a = black_pattern(float)
    if mask is None:
        mask = np.ones(a.shape, dtype=bool)
    radius = lambda x, y: np.sqrt(x**2 + y**2)
    R = radius((XL[mask] - xm)*LCOS_PIX_SIZE, (YL[mask] - ym)*LCOS_PIX_SIZE)
    a[mask] = phase_max + phase_spherical(R, f=f, wavelen=wavelen)

    if steer_slope is not None:
        ycenter = 0.5*(YL[mask].max() + YL[mask].min())
        a[mask] += (YL[mask] - ycenter)*steer_slope + steer_offset
    return a


def pattern_sep(X, Y, C, phase_max, f=30e-3, wavelen=532e-9, phase_factor=1,
                phase_wrap=False, clip=True, dtype=np.uint8, debug=False,
                steer_slope=None, steer_offset=0):
    """Pattern for spots centered in X,Y and rectangular limits defined in C.

    Arguments:
        X, Y (2d arrays): center positions of the spots
        C (3d array): for each spot has 4 values (xmin, xmax, ymin, ymax)
            that defines the spot boundaries
        phase_max (float): constant phase added to the pattern (in pi units)
        f (float): focal length of generated lenses (m)
        wavelen (float): wavelength of laser
        phase_factor (uint8): the 8-bit value (gray scale) corresponding to pi
        phase_wrap (bool): if True wraps all the phase values < 0 to 0..2pi
        clip (bool): if True clips to 255 all the phase values > 255
        dtype (numpy.dtype): data type to use in the returned array.
            Default uint8.
        debug (bool): if True prints debugging info into the log file.
        steer_slope (None or float): a parameter for per-spot steering.
        steer_offset (None or float): a parameter for per-spot steering.
    """
    a = black_pattern(float)

    for iy in range(X.shape[0]):
        for ix in range(X.shape[1]):
            xm, ym = X[iy, ix], Y[iy, ix]
            xmin, xmax, ymin, ymax = C[iy, ix]
            mask = (XL >= xmin) * (XL <= xmax) * (YL >= ymin) * (YL <= ymax)
            single_pattern_steer(xm, ym, mask=mask, a=a, phase_max=phase_max, f=f,
                                 wavelen=wavelen,
                                 steer_slope=steer_slope,
                                 steer_offset=steer_offset)
    neg_phase = a < 0
    if phase_wrap:
        a[neg_phase] = a[neg_phase] % phase_max
        if debug:
            fprint_kw(a_pos_assert=(a>=0).all())
            assert (a>=0).all()
    a[neg_phase] = 0
    a *= phase_factor
    if clip:
        a[a > 255] = 255
    return a.round().astype(dtype)


def get_outer_mask(C, pad=0):
    """Get a rectangular mask that selects outside the spot pattern.
    `pad`: an additional padding that can be added around the spot pattern.
    `sigma`: if > 0 makes the transition smooth (std_dev of Gaussian filter)
            in this case the mask is not boolean anymore but float in [0..1]
    """
    mask = np.ones(XL.shape)
    Xmin, Xmax = C[:, :, 0].min(), C[:, :, 1].max()
    Ymin, Ymax = C[:, :, 2].min(), C[:, :, 3].max()
    mask[Ymin-pad : Ymax+pad+1, Xmin-pad : Xmax+pad+1] = 0
    mask = mask.astype(bool)
    return mask


def get_spot_pattern(Xm, Ym, lens_params, steer_params, pad=2, ref_spot=4,
                     ref_spot_dark=False, dark_all=False, nospot=False,
                     debug=False,
                     steer_slope=None, steer_offset=0,
                     ):
    """Return the pattern with the multi-spot lenses and the beam steering.

    Arguments:
        pad (uint): # pixels of zero-padding around the lens pattern before
            the steering pattern starts.
        ref_spot (int): index of the spot considered as reference (e.g. center).
        ref_spot_dark (bool): if True darken the reference spot.
        dark_all (bool): if True return an array of zeros.
        nospot (bool): if True return only the steering pattern with no spots.
        debug (bool): if True prints debugging info into the log file.
    """
    steer_params.update(debug=debug)
    lens_params.update(debug=debug)
    if dark_all:
        return black_pattern()
    if nospot:
        return get_steer_pattern(**steer_params)
    Xm = Xm.copy() + LCOS_X_SIZE/2.
    Ym = Ym.copy() + LCOS_Y_SIZE/2.
    XM, YM = np.atleast_2d(Xm), np.atleast_2d(Ym)
    if debug:
        fprint_kw(XM_YM_shape_assert=(len(XM.shape) == len(YM.shape) == 2))
        assert len(XM.shape) == len(YM.shape) == 2

    C = get_spot_limits(XM, YM, debug=debug)
    a = pattern_sep(XM, YM, C, dtype=np.uint8,
                    steer_slope=steer_slope,
                    steer_offset=steer_offset,
                    **lens_params)
    if ref_spot_dark:
        if ref_spot >= 0 and ref_spot < XM.size:
            nrows, ncols = XM.shape
            rspot_y = ref_spot // ncols
            rspot_x = ref_spot % ncols
            xmin, xmax, ymin, ymax = C[rspot_y, rspot_x]
            a[ymin:ymax + 1, xmin:xmax + 1] = 0
        else:
            print('WARNING: ref_spot out of range: %d' % ref_spot)
    if steer_params['vmax'] > 0:
        steer_img = get_steer_pattern(**steer_params)
        mask = get_outer_mask(C, pad=pad)
        a[mask] = steer_img[mask]
    return a


def spot_coord_grid(nrows, ncols, pitch_x=25, pitch_y=25,
                    center_x=0, center_y=0):
    """Returns the coordinates of spots arranged on a rectangular grid.

    Returns:
        Two arrays of X and Y coordinates. These arrays can be directly
        passed to `get_spot_pattern` to generate a pattern of spots.
    """
    xm = (np.arange(0, ncols, dtype=float) - (ncols-1)/2) * pitch_x
    ym = (np.arange(0, nrows, dtype=float) - (nrows-1)/2) * pitch_y
    return np.meshgrid(xm, ym)

def get_test_pattern():
    a = np.arange(LCOS_Y_SIZE * LCOS_X_SIZE, dtype=float)
    a *= (255 / a.max())
    return a.astype('uint8').reshape(LCOS_Y_SIZE, LCOS_X_SIZE)

def pattern_from_dict(ncols, nrows, rotation, spotsize, pitch_x, pitch_y,
                 center_x, center_y, wavelen, steer_lw, steer_vmax,
                 ref_spot, focal, phase_max, phase_factor, steer_pad,
                 Xm, Ym, phase_wrap, phase_oversh, steer_horiz, test_pattern,
                 nospot, grid, dark_all, ref_spot_dark, debug):
    if test_pattern:
        return get_test_pattern()

    if grid:
        Xm, Ym = spot_coord_grid(nrows=nrows, ncols=ncols,
                                 pitch_x=pitch_x, pitch_y=pitch_y,
                                 center_x=center_x, center_y=center_y)
    else:
        Xm = np.array(Xm)
        Ym = np.array(Ym)

    lens_params = dict(wavelen=wavelen, f=focal, phase_max=phase_max,
                       phase_factor=phase_factor, phase_wrap=phase_wrap)

    steer_params = dict(vmax=steer_vmax, lw=steer_lw,
                        horizontal=steer_horiz)
    a = get_spot_pattern(Xm, Ym, lens_params, steer_params, pad=steer_pad,
                         ref_spot_dark=ref_spot_dark, ref_spot=ref_spot,
                         dark_all=dark_all, nospot=nospot, debug=debug)
    return a


if __name__ == "__main__":
    lens_params = dict(wavelen=532e-9, f=32e-3, phase_max=4, phase_factor=65,
                       phase_wrap=False)
    steer_params = dict(vmax=120, lw=180, horizontal=False)

    Xm = np.array([  23.87  ,   46.99  ,   70.1875,   93.4175,  116.6275,
                   139.6825,  162.715 ,  185.765 ])
    Ym = np.array([ 15.5825,  15.56  ,  15.5875,  15.595 ,  15.5225,  15.545 ,
                   15.5125,  15.47  ])

    a = get_spot_pattern(Xm,Ym, lens_params, steer_params,
                         pad=2, darken_cspot=0)
