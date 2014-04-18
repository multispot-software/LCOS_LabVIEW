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
"""

import numpy as np

LCOS_X_SIZE, LCOS_Y_SIZE, LCOS_PIX_SIZE = 800, 600, 20e-6
YL, XL = np.mgrid[:LCOS_Y_SIZE,:LCOS_X_SIZE]


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

def get_steer_pattern(lw=20, vmax=255, horizontal=True):
    """Horizontal or vertical pattern for beam steering.

    Arguments:
        lw (uint): line-width of the steering pattern
        vmax (int, 0..255): max value for the steering pattern
        horizontal (bool): if True draw orizontal line, else vertical
    """
    a = np.zeros((LCOS_Y_SIZE,LCOS_X_SIZE), dtype=np.uint8)
    if vmax > 0:
        fprint_kw(lw_pos_assert=(lw > 0))
        assert lw > 0
        for v in range(LCOS_Y_SIZE):
                if is_odd(v/lw): a[v] = vmax
    return a

def get_spot_limits(X, Y):
    """
    From X,Y (spot positions) compute the limits for each spot (stored in C)
    """
    fprint_kw(X=X, Y=Y)
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
    fprint_kw(x_y_pitch_nan_assert=(not np.isnan(xpitch)) and (not np.isnan(ypitch)))
    assert (not np.isnan(xpitch)) and (not np.isnan(ypitch))

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
                ph_wrapping=False, clip=True, dtype=np.uint8):
    """Pattern for spots centered in X,Y and rectangular limits defined in C.
    `phi_max`: constant phase added to the pattern (in pi units)
    `f`: focal length of generated lenses (m)
    `wl`: wavelength of laser
    `phase_factor`: (uint8) the 8-bit value (grey scale) corresponding to pi
    `ph_wrapping`: (bool) if True wraps all the phase values < 0 to 0..2pi
    `clip`: (bool) if True clips to 255 all the phase values > 255
    """
    a = np.zeros((LCOS_Y_SIZE,LCOS_X_SIZE))
    #for xm, ym, c in zip(X.ravel(), Y.ravel(), C.ravel()):
    for iy in xrange(X.shape[0]):
        for ix in xrange(X.shape[1]):
            xm, ym = X[iy,ix], Y[iy,ix]
            xmin,xmax,ymin,ymax = C[iy,ix]
            mask = (XL >= xmin)*(XL <= xmax)*(YL >= ymin)*(YL <= ymax)
            single_pattern(xm, ym, mask=mask, a=a, phi_max=phi_max, f=f, wl=wl)
    if ph_wrapping:
        a[a<0] = a[a<0] % phi_max
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
    Xmin, Xmax = C[:,:,0].min(), C[:,:,1].max()
    Ymin, Ymax = C[:,:,2].min(), C[:,:,3].max()
    mask[Ymin-pad:Ymax+pad+1, Xmin-pad:Xmax+pad+1] = 0
    mask = mask.astype(bool)
    return mask

def get_spot_pattern(Xm,Ym, lens_params, steer_params, pad=2, CD=(0,4),
                     darken_cspot=False, dark_all=False, nospot=False):
    """Return the pattern with the multi-spo lenses and the beam steering.
    `pad`: (uint) #pixels of padding for the lens pattern
    `CD`: coordinates of the pixel considerred the center one
    `darken_cspot`: (bool) if True darken the center spot
    `dark_all`: (bool) if True return an array of zeros
    `nospot`: (bool) if True return only hte steering pattern
    """
    if dark_all: return np.zeros((LCOS_Y_SIZE,LCOS_X_SIZE), dtype=np.uint8)
    if nospot: return get_steer_pattern(**steer_params)
    fprint_kw(Xm=Xm, Ym=Ym)
    Xm += LCOS_X_SIZE/2.
    Ym += LCOS_Y_SIZE/2.
    XM, YM = np.atleast_2d(Xm), np.atleast_2d(Ym)
    fprint_kw(XM_YM_shape_assert=(len(XM.shape) == len(YM.shape) == 2))
    assert len(XM.shape) == len(YM.shape) == 2
    if darken_cspot: assert (CD[0] < XM.shape[0]) and (CD[1] < XM.shape[1])

    C = get_spot_limits(XM,YM)
    a = pattern_sep(XM,YM,C, dtype=np.uint8, **lens_params)
    if darken_cspot:
        xmin, xmax, ymin, ymax = C[CD[0], CD[1]]
        a[ymin:ymax+1, xmin:xmax+1] = 0
    if steer_params['vmax'] > 0:
        ad = get_steer_pattern(**steer_params)
        mask = get_outer_mask(C, pad=pad)
        a[mask] = ad[mask]
    return a


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



a = pattern_wrapper(Xm, Ym, f, wl, phi_max, phase_factor, center_spot,
                    darken_cspot, lw, vmax, ph_wrapping, pad, dark_all, nospot)

s = 'test'



#import numpy as np
#a = (np.arange(800*600).reshape(800, 600).T*255./(800*600)).tolist()
