import socketserver
import yaml
from patternlib_simple2 import pattern_from_dict


def process_recv_data(data):
    print(data)
    params = yaml.load(data)
    print(params)
    a = pattern_from_dict(**params)
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
