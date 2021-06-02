# https://www.henryschmale.org/2021/01/07/pygame-linein-audio-viz.html
# https://github.com/pygame/pygame/blob/main/examples/audiocapture.py
# https://www.oreilly.com/library/view/elegant-scipy/9781491922927/ch04.html
# https://www.gaussianwaves.com/2020/01/how-to-plot-fft-in-python-fft-of-basic-signals-sine-and-cosine-waves/
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html
# https://machinelearningmastery.com/curve-fitting-with-python/

import pyaudio
import numpy as np
from math import sqrt
from scipy.interpolate import UnivariateSpline
from scipy.signal import butter, lfilter
from scipy.optimize import curve_fit
import scipy as sp
import time
import pygame
from pygame import gfxdraw
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

terms = 100 # number of terms for the Fourier series

pygame.init()
clock = pygame.time.Clock()

# set up a bunch of constants
BLUE       = (  0,   0, 255)
WHITE      = (255, 255, 255)
DARKRED    = (128,   0,   0)
DARKBLUE   = (  0,   0, 128)
RED        = (255,   0,   0)
GREEN      = (  0, 255,   0)
DARKGREEN  = (  0, 128,   0)
YELLOW     = (255, 255,   0)
DARKYELLOW = (128, 128,   0)
BLACK      = (  0,   0,   0)

T = 0.04
RATE = 192000
# CHUNK = int((1/30) * RATE)
# CHUNK = 1024 *4
CHUNK = int(RATE*T)
FORMAT = pyaudio.paInt16
# FORMAT = pyaudio.paInt32
# print (CHUNK)

f_vec = RATE * np.arange(CHUNK / 2) / CHUNK

lowcut = 2000.0
highcut = 80000.0

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def linear(x, a, b):
	return a * x + b    

def gauss(x, a, mu, sig):
    return a**sp.exp(-(x-mu)**2/(2.*sig**2))

def gaussian(x, amp, cen, wid):
    return (amp / (sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))

def poly2(x, a, b, c, d):
	return a * np.sin(b - x) + c * x**2 + d    

def poly5(x, a, b, c, d, e, f):
	return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + (e * x**5) + f

def quadratic(x, a, b, c):
	return a * x + b * x**2 + c

def exp_fit(x, a, b, c):
    return a * np.exp(-b * x) + c

def log_fit(x, p1,p2):
    return p1*np.log(x)+p2    
    
def dB(y):
    "Calculate the log ratio of y / max(y) in decibel."
    y = np.abs(y)
    y /= y.max()
    return 20 * np.log10(y)    

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
    channels=1,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK)

def stereoToMono(audiodata):
    newaudiodata = []
    for i in range(len(audiodata)):
        d = audiodata[i][0]/2 + audiodata[i][1]/2
        newaudiodata.append(d)

    return np.array(newaudiodata, dtype='int16')    

print("*recording")

SCREEN_HEIGHT = 600
# screen = pygame.display.set_mode((CHUNK, SCREEN_HEIGHT))
screen = pygame.display.set_mode((1600, SCREEN_HEIGHT))

done = False
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
            break
    start = time.time()
    buff = stream.read(CHUNK)
    data = np.frombuffer(buff, dtype=np.int16)
    # data = butter_bandpass_filter(data, lowcut, highcut, RATE, order=4)
    data = data * np.hamming(len(data))
    # 0
    # fft_complex = np.fft.rfft(data)
    # 1
    fft_complex = np.fft.fft(data, n=CHUNK)
    left, right = np.split(np.abs(fft_complex), 2)
    fft_complex = np.add(left, right[::-1])
    # 2
    # fft_complex = (np.abs(np.fft.fft(data))[0:int(np.floor(CHUNK / 2))]) / CHUNK
    # fft_complex[1:] = 2 * fft_complex[1:]
    
    # fft_complex = np.multiply(20, np.log10(fft_complex))
    # fft_complex = dB(fft_complex)
 
    #fft_distance = np.zeros(len(fft_complex))


    screen.fill((0,0,0))
    color = (0,128,1)
    color1 = (0,128,128)
    s = 0
    # fft_complex = np.argwhere(fft_complex)
    # fft_complex = np.ma.masked_equal(fft_complex,0)

    # newy = fft_complex
    # newx = np.arange(len(newy))
    # # idx = np.nonzero(fft_complex) 
    # idx = np.where(fft_complex!=0)
    # interp = interp1d(newx[idx],newy[idx])
    # fft_complex = interp(newx)

    # fit
    x = np.linspace(0, len(fft_complex), len(fft_complex))
    # spline = UnivariateSpline(x, fft_complex-(np.max(fft_complex)/2), s=0)
    # fit 2
    try:
        (a, mu, sig), _ = curve_fit(gauss, x, fft_complex, maxfev = 2000)
        fitted_curve = gauss(x, a, mu, sig)
        fitted_curve = fitted_curve - np.min(fitted_curve)
########################################################################        
        # (a, mu, sig), _ = curve_fit(gaussian, x, fft_complex, maxfev = 2000)
        # fitted_curve = gaussian(x, a, mu, sig)
        # fitted_curve = fitted_curve - np.min(fitted_curve)        
########################################################################
        # popt, _ = curve_fit(poly5, x, fft_complex)
        # a, b, c, d, e, f = popt
        # fitted_curve = poly5(x, a, b, c, d, e, f)
########################################################################
        # popt, _ = curve_fit(exp_fit, x, fft_complex)
        # a, b, c = popt
        # fitted_curve = exp_fit(x, a, b, c)          
########################################################################
        # popt, _ = curve_fit(quadratic, x, fft_complex)
        # a, b, c = popt
        # fitted_curve = quadratic(x, a, b, c)
########################################################################
        # popt, _ = curve_fit(poly2, x, fft_complex)
        # a, b, c, d = popt
        # fitted_curve = poly2(x, a, b, c, d)          
    except:
        fitted_curve = []    
    max_fitted = np.max(fitted_curve)
    fitted_curve = fitted_curve - max_fitted * 0.5
    r = np.sqrt(fitted_curve)
    r[np.isnan(r)] = 0
###### Fitting 0dB frequency #########
    fit_point = 5
    x_slope = np.arange(0, fit_point, 1)
    _3dB_slope = [i for i in r if i != 0]
    _3dB_slope = _3dB_slope[:fit_point]
    popt, _ = curve_fit(linear, x_slope, _3dB_slope)
    a, b = popt
    _3dB_fit = linear(x_slope, *popt)
    y_center = _3dB_slope[int(len(_3dB_slope)/2)]
    _0dB_f = int((y_center-b)/a) 
######################################
    print('0dB',_0dB_f)

    r = [idx for idx, val in enumerate(r) if val != 0]
    print(r[0],r[-1])
    # scale_value = SCREEN_HEIGHT / max_fitted
    # fitted_curve = max_fitted * scale_value
    # plt.xlim([0, 500])

    # fitted_curve = np.multiply(20, np.log10(fitted_curve))

    # plt.plot(fitted_curve,'r')
    plt.plot(_3dB_fit,'g')      
    # plt.plot(fft_complex,'b')

    # r = spline.roots()
    f0= (f_vec[-1]*r[_0dB_f])/len(f_vec)
    s0= (f_vec[-1]*r[0])/len(f_vec)
    s1= (f_vec[-1]*r[-1])/len(f_vec)
    print(s0)
    print(s1)
    print('f0',f0)
    # plt.plot(x, spline(x), 'b', lw=1)
    plt.show()
  

    # Y = np.fft.fft(fft_complex)
    # np.put(Y, range(terms+1, len(fft_complex)), 0.0) # zero-ing coefficients above "terms"
    # fft_complex = np.fft.ifft(Y)

    max_val = sqrt(max(v.real * v.real + v.imag * v.imag for v in fft_complex))
    # max_val = np.max(fft_complex)
    scale_value = SCREEN_HEIGHT / max_val
    for i,v in enumerate(fft_complex):
        #v = complex(v.real / dist1, v.imag / dist1)
        dist = sqrt(v.real * v.real + v.imag * v.imag)
        # dist = np.real(fft_complex[i])
        mapped_dist = dist * scale_value
        # s += mapped_dist

        # pygame.draw.aaline(screen, DARKRED,[i, SCREEN_HEIGHT], [i, SCREEN_HEIGHT - mapped_dist],5)
        # pygame.draw.line(screen, color, (i, SCREEN_HEIGHT), (i, SCREEN_HEIGHT - mapped_dist))
        # pygame.draw.line(screen, RED, (i, SCREEN_HEIGHT), (i, SCREEN_HEIGHT - mapped_dist),1)
        # gfxdraw.pixel(screen,i,SCREEN_HEIGHT-int(mapped_dist),RED)
        pygame.draw.circle(screen,RED,[i,SCREEN_HEIGHT-int(mapped_dist)],2)
    # print(s/len(fft_complex))

    pygame.display.flip()
    end = time.time()
    # clock.tick(1)
    # print(end - start)

    
