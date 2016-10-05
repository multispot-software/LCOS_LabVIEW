
import yaml
import numpy as np
import socketserver
from patternlib_simple2 import get_spot_pattern

def pattern_yaml(wl, lw, Xm, center_spot, phase_factor, phi_max, PhWrap,
                 darken_cspot, pad, f, debug, steer_horiz, vmax, nospot, Ym, dark_all):

    lens_params = dict(wl=wl, f=f, phi_max=phi_max, phase_factor=phase_factor,
                       ph_wrapping=PhWrap)
    steer_params = dict(vmax=vmax, lw=lw, horizontal=steer_horiz)
    a = get_spot_pattern(Xm, Ym, lens_params, steer_params, pad=pad,
                        darken_cspot=darken_cspot, CD=(0, center_spot),
                        dark_all=dark_all, nospot=nospot, debug=debug)
    return a

def process_recv_data(data):
    print(data)
    params = yaml.load(data)
    params['Xm']  = np.array(params['Xm'])
    params['Ym']  = np.array(params['Ym'])
    print(params)
    a = pattern_yaml(**params)
    return a.tobytes()

class MyTCPHandler(socketserver.BaseRequestHandler):

    def handle(self):
        # self.request is the TCP socket connected to the client
        self.data = self.request.recv(1024).strip()
        print("- Data received.")
        response = process_recv_data(self.data)
        self.request.sendall(response)

if __name__ == "__main__":
    HOST, PORT = "localhost", 9999
    server = socketserver.TCPServer((HOST, PORT), MyTCPHandler)
    print('Serving...')
    server.serve_forever()
