import numpy as np
import matplotlib.pyplot as pp

x = np.array([0, 1, -1])
y = np.array([0, 1, -1])
z = np.array([[0, 1, -1],[1, 2, 0],[-1, 0, -2]])
f = np.fft.fft2(z)
w_x = np.fft.fftfreq(f.shape[0])
w_y = np.fft.fftfreq(f.shape[1])

pp.figure()
pp.imshow(np.abs(f))
pp.xticks(np.arange(0,len(w_x)), np.round(w_x,2))
pp.yticks(np.arange(0,len(w_y)), np.round(w_y,2))

f = np.fft.fftshift(f)
w_x = np.fft.fftshift(w_x)
w_y = np.fft.fftshift(w_y)

pp.figure()
pp.imshow(np.abs(f))
pp.xticks(np.arange(0,len(w_x)), np.round(w_x,2))
pp.yticks(np.arange(0,len(w_y)), np.round(w_y,2))
pp.show()