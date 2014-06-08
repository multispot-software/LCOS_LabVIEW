"""
Show a frameless image in a QT4 window
"""

from PyQt4 import QtCore, QtGui
import numpy as np
import matplotlib.pyplot as plt

screen_1_xresolution = 1920


label = QtGui.QLabel()
label.setWindowFlags(QtCore.Qt.FramelessWindowHint)
label.resize(800,600)
label.move(0,0)
label.setWindowTitle('LCOS Pattern')

is_odd = lambda x: bool(x & 1)

def gen_test_pattern(vmin=0,vmax=255,horizontal=True,lw=4):
    """Horizontal or vertical pattern."""
    a = np.zeros((800,600), dtype=np.uint8)
    for x in range(800):
        for y in range(600):
            l = x if horizontal else y
            a[x,y] = vmin if is_odd(l/lw) else vmax
    return a

def phase_exact(r, f, wl):
    """Phase profile at distance r from center for focusing at f from screen."""
    return 2*(1-np.cos(np.arctan(r/f)))*np.sqrt(r**2+f**2)/wl

def show_pattern(a):
    i = QtGui.QImage(a.tostring(), 800, 600, QtGui.QImage.Format_Indexed8)
    p = QtGui.QPixmap.fromImage(i)
    label.setPixmap(p)
    label.show()

def show_pattern_twin(a, im):
    show_pattern(a)
    im.set_data(a)
    plt.draw()

def clear_pattern():
    label.hide()

def move_to_2nd_screen():
    label.move(screen_1_xresolution, 0)

def move_from_2nd_screen():
    label.move(0, 0)

if __name__ == '__main__':
    a = gen_test_pattern()
    show_pattern(a)

