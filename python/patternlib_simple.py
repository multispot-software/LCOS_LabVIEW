from numpy import (atleast_2d, array, zeros, ones, bool, concatenate, mgrid, 
                   mean, diff, isnan, ceil, floor, uint8, sqrt)

LCOS_X_SIZE, LCOS_Y_SIZE, LCOS_PIX_SIZE = 800, 600, 20e-6
YL, XL = mgrid[:LCOS_Y_SIZE,:LCOS_X_SIZE]

is_odd = lambda x: bool(x & 1)

def black_pattern():
    return zeros((800,600), dtype=uint8)

def get_steer_pattern(vmax=255, horizontal=True, lw=20):
    """Horizontal or vertical pattern fro beam steering.
    `lw`: (uint) line-width of the steering pattern
    """
    a = zeros((LCOS_Y_SIZE,LCOS_X_SIZE), dtype=uint8)
    if vmax > 0:
        assert lw > 0
        for v in range(LCOS_Y_SIZE):
                if is_odd(v/lw): a[v] = vmax
    return a

def get_spot_limits(X,Y):
    """
    From X,Y (spot positions) compute the limits for each spot (stored in C)
    """
    assert X.shape == Y.shape
    Y_dim, X_dim = X.shape
    alpha = 0.5

    ## Assumes at least one dimension > 1
    xpitch = mean(diff(X,axis=1))
    if Y_dim == 1:
        ypitch = xpitch # avoids Warning
    else: 
        ypitch = mean(diff(Y,axis=0))    
    if X_dim == 1: xpitch = ypitch    
    if Y_dim == 1: ypitch = xpitch
    assert (not isnan(xpitch)) and (not isnan(ypitch))
    
    X_sep = (diff(X, axis=1)/2. + X[:,:-1]) # separators between spots
    # For each row (Y_dim)
    Xstart = array([(xl[0]-alpha*xpitch)-1 for xl in X]).reshape(Y_dim,1)
    Xend = array([(xl[-1]+alpha*xpitch)+1 for xl in X]).reshape(Y_dim,1)
    X_R = concatenate((Xstart,X_sep,Xend), axis=1)

    Y_sep = (diff(Y, axis=0)/2. + Y[:-1,:]) # separators between spots
    # For each col (X_dim)
    Ystart = array([(yl[0]-alpha*ypitch)-1 for yl in Y.T]).reshape(1,X_dim)
    Yend = array([(yl[-1]+alpha*ypitch)+1 for yl in Y.T]).reshape(1,X_dim)
    Y_R = concatenate((Ystart,Y_sep,Yend), axis=0)

    # C contains the limits (4 values) for each spot
    C = zeros((Y_dim,X_dim,4))
    C[:,:,0] = ceil(X_R[:,:-1])  # xmin
    C[:,:,1] = floor(X_R[:,1:])  # xmax
    C[:,:,2] = ceil(Y_R[:-1,:])  # ymin
    C[:,:,3] = floor(Y_R[1:,:])  # ymax
    return C

def phase_spherical(r, f, wl=532e-9):
    """Phase profile (in pi units) at distance r (x,y) from spot center 
    for a spherical wave converging at a distance f from the screen.
    Inputs in SI units. NOTE: This is equivalent to phase_exact()."""
    return -(2/wl)*(sqrt(r**2+f**2)-f)

def single_pattern(xm, ym, mask=None, a=None, phi0=0, f=30e-3, wl=532e-9):
    """Pattern for a single spot centered at (xm,ym) in LCOS pixel units.
    The pattern is computed on the subset of LCOS pixels selected by mask.
    `phi0` is constant phase to add to the pattern (in pi units).
    `a` is an (optional) array in which to store the pattern.
    """
    #if a is None: a = zeros((LCOS_Y_SIZE,LCOS_X_SIZE))
    #if mask is None: mask = ones(a.shape, dtype=bool)
    radius = lambda x,y: sqrt(x**2 + y**2)
    R = radius((XL[mask]-xm)*LCOS_PIX_SIZE,(YL[mask]-ym)*LCOS_PIX_SIZE)
    a[mask] = phi0 + phase_spherical(R, f=f, wl=wl)
    return a

def pattern_sep(X, Y, C, phi0, f=30e-3, wl=532e-9, phase_factor=1,
                ph_wrapping=False, clip=True, dtype=uint8):
    """Pattern for spots centered in X,Y and rectangular limits defined in C.
    `phi0`: constant phase added to the pattern (in pi units)
    `f`: focal length of generated lenses (m)
    `wl`: wavelength of laser
    `phase_factor`: (uint8) the 8-bit value (grey scale) corresponding to pi
    `ph_wrapping`: (bool) if True wraps all the phase values < 0 to 0..2pi
    `clip`: (bool) if True clips to 255 all the phase values > 255
    """
    a = zeros((LCOS_Y_SIZE,LCOS_X_SIZE))
    #for xm, ym, c in zip(X.ravel(), Y.ravel(), C.ravel()):
    for iy in xrange(X.shape[0]):
        for ix in xrange(X.shape[1]):
            xm, ym = X[iy,ix], Y[iy,ix]
            xmin,xmax,ymin,ymax = C[iy,ix]
            mask = (XL >= xmin)*(XL <= xmax)*(YL >= ymin)*(YL <= ymax)
            single_pattern(xm, ym, mask=mask, a=a, phi0=phi0, f=f, wl=wl)
    if ph_wrapping: 
        a[a<0] = a[a<0] % phi0
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
    mask = ones(XL.shape)
    Xmin, Xmax = C[:,:,0].min(), C[:,:,1].max()
    Ymin, Ymax = C[:,:,2].min(), C[:,:,3].max()
    mask[Ymin-pad:Ymax+pad+1,Xmin-pad:Xmax+pad+1] = 0
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
    if dark_all: return zeros((LCOS_Y_SIZE,LCOS_X_SIZE), dtype=uint8)
    if nospot: return get_steer_pattern(**steer_params)

    Xm += LCOS_X_SIZE/2.
    Ym += LCOS_Y_SIZE/2.
    XM, YM = atleast_2d(Xm), atleast_2d(Ym)
    assert len(XM.shape) == len(YM.shape) == 2
    if darken_cspot: assert (CD[0] < XM.shape[0]) and (CD[1] < XM.shape[1])
    
    C = get_spot_limits(XM,YM)
    a = pattern_sep(XM,YM,C, dtype=uint8, **lens_params)
    if darken_cspot:
        xmin, xmax, ymin, ymax = C[CD[0], CD[1]]
        a[ymin:ymax+1, xmin:xmax+1] = 0
    if steer_params['vmax'] > 0:
        ad = get_steer_pattern(**steer_params)
        mask = get_outer_mask(C, pad=pad)
        a[mask] = ad[mask]
    return a

if __name__ == "__main__":
    lens_params = dict(wl=532e-9, f=32e-3, phi0=4, phase_factor=65, 
                       ph_wrapping=False)
    steer_params = dict(vmax=120, lw=1, horizontal=True)
    Xm = array([  23.87  ,   46.99  ,   70.1875,   93.4175,  116.6275,  139.6825,  162.715 ,  185.765 ])
    Ym = array([ 15.5825,  15.56  ,  15.5875,  15.595 ,  15.5225,  15.545 ,  15.5125,  15.47  ])
    
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
    a = get_spot_pattern(Xm,Ym, lens_params, steer_params, pad=2, dark_spot=0)
    
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
    