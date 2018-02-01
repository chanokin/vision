import enum
import numpy as np
import socket

import matplotlib.animation as animation
import matplotlib.colors as col
import matplotlib.pyplot as plt

COLOUR_LEVEL = 0.9

# ----------------------------------------------------------------------------
# Visualiser
# ----------------------------------------------------------------------------
class RetinaVisualiser(object):
    # How many bits are used to represent colour
    colour_bits = 1

    def __init__(self, udp_ports, retina_config):
        cfg = retina_config
        # Open socket to receive datagrams
        channels = udp_ports.keys()
        self.colours = {'on': 1, 'off': 0}

        keys = sorted(udp_ports[channels[0]].keys())
        self.sockets = {}
        for ch in channels:
            self.sockets[ch] = {}
            for k in keys:
                self.sockets[ch][k] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.sockets[ch][k].bind(("0.0.0.0", udp_ports[ch][k]))
                self.sockets[ch][k].setblocking(False)

        # Create image plot to display game screen
        self.fig = plt.figure("Retina outputs")

        axlist = [plt.subplot(2, 2, i+1) for i in range(len(keys))]

        self.axes = {keys[i]: axlist[i] for i in range(len(keys))}

        self.data = {}
        for k in keys:
            self.data[k] = np.zeros((cfg[k]['height'], cfg[k]['width'], 3))

        self.images = (self.axes[k].imshow(
                                self.data[k], interpolation="nearest")
                                                            for k in keys)

        self.keys = keys
        self.channels = channels

    def init(self):
        for i, k in enumerate(self.keys):
            self.axes[k].set_title("pop %s"%k)

            # Hide grid
            self.axes[k].grid(False)
            self.axes[k].set_xticklabels([])
            self.axes[k].set_yticklabels([])
            self.axes[k].axes.get_xaxis().set_visible(False)
            self.data[k][:] = 0
            self.images[i].set_array(self.data[k])

    # ------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------
    def show(self):
        # Play animation
        self.animation = animation.FuncAnimation(self.fig, self._update,
                                                 init_func=self.init,
                                                 interval=20.0, blit=True)
        # Show animated plot (blocking)
        try:
            plt.show()
        except:
            pass

    # ------------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------------
    def _update(self, frame):

        # Read all datagrams received during last frame
        while True:
            for ch in self.channels:
                for k in self.keys:
                    try:
                        raw_data = self.sockets[ch][k].recv(512)
                    except socket.error:
                        # If error isn't just a non-blocking read fail, print it
                        # if e != "[Errno 11] Resource temporarily unavailable":
                        #    print "Error '%s'" % e
                        # Stop reading datagrams
                        break
                    else:
                        # Slice off EIEIO header and convert to numpy array of uint32
                        payload = np.fromstring(raw_data[6:], dtype="uint32")

                        y   = np.clip(payload // self.data[k].shape[1],
                                      0, self.data[k].shape[0] - 1)
                        x   = np.clip(payload % self.data[k].shape[1],
                                      0, self.data[k].shape[1] - 1)
                        chi = self.colours[ch]
                        # Set valid pixels
                        try:
                            self.data[k][y, x, chi] = COLOUR_LEVEL

                        except IndexError as e:
                            print(
                              "Packet contains invalid pixels: (x, y, c) == %d, %d, %s"%
                              (x, y, ch))
                            self.data[k][:,:, chi] = 0


        for i in len(range(self.keys)):
            k = self.keys[i]
            # Set image data
            try:
                self.images[i].set_array(self.data[k])
            except NameError:
                pass

        # Return list of artists which we have updated
        # **YUCK** order of these dictates sort order
        # **YUCK** score_text must be returned whether it has
        # been updated or not to prevent overdraw
        return self.images

