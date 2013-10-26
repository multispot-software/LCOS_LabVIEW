# Paste this in LabView Python-ScriptNode
import os
os.chdir("C:/Data/Antonio/software/LCOS/LCOS_CompleteLV/python/")
s = os.getcwd()
from pattern_main import pattern_wrapper
a = pattern_wrapper(Xm, Ym, f, wl, phi0, phase_factor, center_spot, 
                    darken_cspot, lw, vmax, ph_wrapping, pad, dark_all, nospot)
