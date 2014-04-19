import numpy as np


LCOS_X_SIZE, LCOS_Y_SIZE, LCOS_PIX_SIZE = 800, 600, 20e-6
YL, XL = np.mgrid[:LCOS_Y_SIZE, :LCOS_X_SIZE]

is_odd = lambda x: bool(x & 1)

def fprint(s):
    with open(r"C:\Data\Antonio\software\LCOS\python_out.txt", "a") as myfile:
        myfile.write(str(s)+'\n')

def fprint_kw(**kwargs):
    with open(r"C:\Data\Antonio\software\LCOS\python_out.txt", "a") as myfile:
        s = ', '.join([k +'='+str(v) for k,v in kwargs.iteritems()])
        myfile.write(s+'\n')


def black_pattern():
    return np.zeros((800,600), dtype=np.uint8)

def get_steer_pattern(lw=20, vmax=255, horizontal=True, debug=False):
    """Horizontal or vertical pattern for beam steering.

    Arguments:
        lw (uint): line-width of the steering pattern
        vmax (int, 0..255): max value for the steering pattern
        horizontal (bool): if True draw orizontal line, else vertical
    """
    a = np.zeros((LCOS_Y_SIZE,LCOS_X_SIZE), dtype=np.uint8)
    if vmax > 0:
        if debug:
            fprint_kw(lw_pos_assert=(lw > 0), horizontal=horizontal)
            assert lw > 0
        ai = a if horizontal else a.T
        for i, line in enumerate(ai):
            if is_odd(i/lw): line[:] = vmax
    return a

def get_spot_limits(X, Y, debug=False):
    """
    From X,Y (spot positions) compute the limits for each spot (stored in C)
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

    X_sep = (np.diff(X, axis=1)/2. + X[:,:-1]) # separators between spots
    # For each row (Y_dim)
    Xstart = np.array([(xl[0]-alpha*xpitch)-1 for xl in X]).reshape(Y_dim,1)
    Xend = np.array([(xl[-1]+alpha*xpitch)+1 for xl in X]).reshape(Y_dim,1)
    X_R = np.concatenate((Xstart,X_sep,Xend), axis=1)

    Y_sep = (np.diff(Y, axis=0)/2. + Y[:-1,:]) # separators between spots
    # For each col (X_dim)
    Ystart = np.array([(yl[0]-alpha*ypitch)-1 for yl in Y.T]).reshape(1,X_dim)
    Yend = np.array([(yl[-1]+alpha*ypitch)+1 for yl in Y.T]).reshape(1,X_dim)
    Y_R = np.concatenate((Ystart,Y_sep,Yend), axis=0)

    # C contains the limits (4 values) for each spot
    C = np.zeros((Y_dim,X_dim,4))
    C[:,:,0] = np.ceil(X_R[:,:-1])  # xmin
    C[:,:,1] = np.floor(X_R[:,1:])  # xmax
    C[:,:,2] = np.ceil(Y_R[:-1,:])  # ymin
    C[:,:,3] = np.floor(Y_R[1:,:])  # ymax
    return C

def phase_spherical(r, f, wl=532e-9):
    """Phase profile (in pi units) at distance r (x,y) from spot center
    for a spherical wave converging at a distance f from the screen.
    Inputs in SI units. NOTE: This is equivalent to phase_exact()."""
    return -(2/wl)*(np.sqrt(r**2+f**2)-f)

def single_pattern(xm, ym, mask=None, a=None, phi_max=0, f=30e-3, wl=532e-9):
    """Pattern for a single spot centered at (xm,ym) in LCOS pixel units.
    The pattern is computed on the subset of LCOS pixels selected by mask.
    `phi_max` is constant phase to add to the pattern (in pi units).
    `a` is an (optional) array in which to store the pattern.
    """
    #if a is None: a = zeros((LCOS_Y_SIZE,LCOS_X_SIZE))
    #if mask is None: mask = ones(a.shape, dtype=bool)
    radius = lambda x,y: np.sqrt(x**2 + y**2)
    R = radius((XL[mask] - xm)*LCOS_PIX_SIZE, (YL[mask] - ym)*LCOS_PIX_SIZE)
    a[mask] = phi_max + phase_spherical(R, f=f, wl=wl)
    return a

def pattern_sep(X, Y, C, phi_max, f=30e-3, wl=532e-9, phase_factor=1,
                ph_wrapping=False, clip=True, dtype=np.uint8, debug=False):
    """Pattern for spots centered in X,Y and rectangular limits defined in C.

    Arguments:
        X, Y (2d arrays): center positions of the spots
        C (3d array): for each spot has 4 values (xmin, xmax, ymin, ymax)
            that define the spot boundaries
        phi_max (float): constant phase added to the pattern (in pi units)
        f (float): focal length of generated lenses (m)
        wl (float): wavelength of laser
        phase_factor (uint8): the 8-bit value (grey scale) corresponding to pi
        ph_wrapping (bool): if True wraps all the phase values < 0 to 0..2pi
        clip (bool): if True clips to 255 all the phase values > 255
        debug (bool): if True prints debugging info to a file
    """
    a = np.zeros((LCOS_Y_SIZE, LCOS_X_SIZE))

    for iy in xrange(X.shape[0]):
        for ix in xrange(X.shape[1]):
            xm, ym = X[iy, ix], Y[iy, ix]
            xmin, xmax, ymin, ymax = C[iy, ix]
            mask = (XL >= xmin)*(XL <= xmax)*(YL >= ymin)*(YL <= ymax)
            single_pattern(xm, ym, mask=mask, a=a, phi_max=phi_max, f=f, wl=wl)
    if ph_wrapping:
        a[a<0] = a[a<0] % phi_max
        if debug:
            fprint_kw(a_pos_assert=(a>=0).all())
            assert (a>=0).all()
    a[a<0] = 0
    a *= phase_factor
    if clip: a[a>255] = 255
    return a.round().astype(dtype)

def get_outer_mask(C, pad=0):
    """Get a rectaungular mask that selects outside the spot pattern.
    `pad`: an additional padding that can be added around the spot pattern.
    `sigma`: if > 0 makes the transition smooth (std_dev of gaussian filter)
            in this case the mask is not boolean anymore but float in [0..1]
    """
    mask = np.ones(XL.shape)
    Xmin, Xmax = C[:, :, 0].min(), C[:, :, 1].max()
    Ymin, Ymax = C[:, :, 2].min(), C[:, :, 3].max()
    mask[Ymin-pad : Ymax+pad+1, Xmin-pad : Xmax+pad+1] = 0
    mask = mask.astype(bool)
    return mask

def get_spot_pattern(Xm, Ym, lens_params, steer_params, pad=2, CD=(0,4),
                     darken_cspot=False, dark_all=False, nospot=False,
                     debug=False):
    """Return the pattern with the multi-spot lenses and the beam steering.

    Arguments:
        pad (uint): #pixels of padding for the lens pattern
        CD (tuple): coordinates of the pixel considerred the center one
        darken_cspot (bool): if True darken the center spot
        dark_all (bool): if True return an array of zeros
        nospot (bool): if True return only hte steering pattern
        debug (bool): if True prints debugging info to a file
    """
    steer_params.update(debug=debug)
    lens_params.update(debug=debug)
    if dark_all: return np.zeros((LCOS_Y_SIZE, LCOS_X_SIZE), dtype=np.uint8)
    if nospot: return get_steer_pattern(**steer_params)
    Xm += LCOS_X_SIZE/2.
    Ym += LCOS_Y_SIZE/2.
    XM, YM = np.atleast_2d(Xm), np.atleast_2d(Ym)
    if debug:
        fprint_kw(XM_YM_shape_assert=(len(XM.shape) == len(YM.shape) == 2))
        assert len(XM.shape) == len(YM.shape) == 2

    if darken_cspot: assert (CD[0] < XM.shape[0]) and (CD[1] < XM.shape[1])

    C = get_spot_limits(XM, YM, debug=debug)
    a = pattern_sep(XM, YM, C, dtype=np.uint8, **lens_params)
    if darken_cspot:
        xmin, xmax, ymin, ymax = C[CD[0], CD[1]]
        a[ymin:ymax+1, xmin:xmax+1] = 0
    if steer_params['vmax'] > 0:
        ad = get_steer_pattern(**steer_params)
        mask = get_outer_mask(C, pad=pad)
        a[mask] = ad[mask]
    return a

if __name__ == "__main__":
    lens_params = dict(wl=532e-9, f=32e-3, phi_max=4, phase_factor=65,
                       ph_wrapping=False)
    steer_params = dict(vmax=120, lw=1, horizontal=False)
    Xm = np.array([  23.87  ,   46.99  ,   70.1875,   93.4175,  116.6275,
                   139.6825,  162.715 ,  185.765 ])
    Ym = np.array([ 15.5825,  15.56  ,  15.5875,  15.595 ,  15.5225,  15.545 ,
                   15.5125,  15.47  ])

    #ym = r_[21.72,  44.87,  68.20,  91.45, 114.72, 137.87, 160.97, 184.04]
    #ym = r_[13.84,  13.78,  13.79,  13.83,  13.75,  13.79,  13.74,  13.68]


    ## 2D Pattern
    #Xm = vstack((xm,xm)) + LCOS_X_SIZE/2.
    #Ym = vstack((ym,ym+23)) + LCOS_Y_SIZE/2.

    ## 1D  vertical pattern
    #Xm = array([[400],[405]])
    #Ym = array([[300], [350]])

    ## 1D  horizontal pattern
    #Xm = array([[400, 450]])
    #Ym = array([[250, 255]])

    #XM, YM = atleast_2d(Xm), atleast_2d(Ym)
    a = get_spot_pattern(Xm,Ym, lens_params, steer_params, pad=2, darken_cspot=0)

#    C = get_spot_limits(XM,YM)
#    a = pattern_sep(XM,YM,C, **pattern_params)
#    a[a>255] = 255
#    a[a<0] = 0
#    a = a.astype(uint8)
#
#    ad = get_steer_pattern(vmax=120, lw=1, horizontal=1)
#    mask = get_outer_mask(C, pad=2)
#    grad = get_outer_mask(C, pad=12, sigma=5)
#    ad *= grad
#    a[mask] = ad[mask]
