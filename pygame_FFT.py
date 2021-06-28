import pyaudio
import os
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import butter, lfilter, filtfilt, resample
from scipy.optimize import curve_fit
import scipy as sp
import time
import pygame
from pygame.locals import *
from pygame import gfxdraw
import matplotlib.pyplot as plt
from chirp3 import lchirp

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

sweeps = {
    0: [18000, 34000, 0.004],
    1: [7000, 17000, 0.004],
    2: [12000, 24000, 0.004],
    3: [4000, 10000, 0.004],
    4: [48000, 78000, 0.004],
    5: [3000, 80000, 0.004],
}

T = 0.02
RATE = 192000
CHUNK = int(RATE*T)
FORMAT = pyaudio.paInt16

sweep = 5
blocksize = 1024 * 4
g_amplitude = 17750  # 17750 - 1.0V P2P
chirp_x = 0
chirp_y = []
sound = []
buffer = []

f_vec = RATE * np.arange(CHUNK / 2) / CHUNK

lowcut = 2000.0
highcut = 100000.0

def sweep_gen():
    global chirp_x, chirp_y, g_amplitude, sound
    Tn = sweeps[sweep][2]
    N = int(RATE * Tn)
    chirp_x = np.arange(0, int(Tn * RATE)) / RATE
    tmin = 0
    tmax = Tn
    w0 = lchirp(N, tmin=tmin, tmax=tmax, fmin=sweeps[sweep][0], fmax=sweeps[sweep][1],zero_phase_tmin=True, cos=False)
    w180 = w0 * -1
    chirp_y = np.column_stack((w0, w180))

    chirp_y = chirp_y * g_amplitude
    chirp_y = chirp_y.astype(np.int16)
    sound = pygame.sndarray.make_sound(chirp_y)

def normalize(data):
    amp = 32767/np.max(np.abs(data))
    mean = np.mean(data)
    norm = [(data[i] - mean) * amp for i, k in enumerate(data)]
    return norm 
    
def dB(y):
    "Calculate the log ratio of y / max(y) in decibel."
    y = np.abs(y)
    y /= y.max()
    return 20 * np.log10(y)

lowcut = 3000.0
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

def gauss(x, a, mu, sig):
    return a**np.exp(-(x-mu)**2/(2.*sig**2))

def Lorentzian3(x, amp1, cen1, wid1, amp2,cen2,wid2, amp3,cen3,wid3):
    return (amp1*wid1**2/((x-cen1)**2+wid1**2)) +\
            (amp2*wid2**2/((x-cen2)**2+wid2**2)) +\
                (amp3*wid3**2/((x-cen3)**2+wid3**2))

def Lorentzian4(x, amp1,cen1,wid1, amp2,cen2,wid2, amp3,cen3,wid3, amp4,cen4,wid4):
    return (amp1*wid1**2/((x-cen1)**2+wid1**2)) +\
            (amp2*wid2**2/((x-cen2)**2+wid2**2)) +\
             (amp3*wid3**2/((x-cen3)**2+wid3**2)) +\
              (amp4*wid4**2/((x-cen4)**2+wid4**2))

def linear(x, a, b):
	return a * x + b 

def pre_amp(i):
    switcher={
        11:'7/17',
        12:'7/17',
        13:'7/17',
        15:'7/34',
        18:'12/24',
        19:'12/24',
        20:'12/24',
        25:'18/34',
        26:'18/34',
        46:'40/80',
        47:'40/80',
        6:'xxxxx'
        }
    return switcher.get(i,"Invalid pre AMP")

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
    channels=1,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK)

print("*recording")

os.putenv("SDL_FBDEV", "/dev/fb1")  
os.putenv("SDL_MOUSEDRV", "TSLIB")  
# Mouse is PiTFT touchscreen
os.putenv("SDL_MOUSEDEV", "/dev/input/touchscreen")
os.putenv("SDL_AUDIODRIVER", "alsa")

pygame.mixer.pre_init(frequency=int(RATE), size=-16, channels=2, buffer=blocksize)
pygame.init()
pygame.mouse.set_visible(False)
pygame.event.set_blocked(pygame.MOUSEMOTION)
pygame.event.set_blocked(pygame.MOUSEBUTTONUP)
pygame.font.init()

clock = pygame.time.Clock()

SCREEN_WIDTH = 320
SCREEN_HEIGHT = 240
band_font = pygame.font.Font('freesansbold.ttf', round(0.07*SCREEN_HEIGHT))
level_font = pygame.font.Font('freesansbold.ttf', round(0.07*SCREEN_HEIGHT))
preamp_font = pygame.font.Font('freesansbold.ttf', round(0.07*SCREEN_HEIGHT))
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

band = [7000,17000]
bg_color = 60
terms = 25

sweep_gen()
sound.set_volume(0.01)
sound.play(-1)

done = False
while not done:
    screen.fill((0,0,0))
    s = 0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
            break
    start = time.time()
    buff = stream.read(CHUNK,exception_on_overflow=False) 
    data = np.frombuffer(buff, dtype=np.int16)
    data = butter_bandpass_filter(data, lowcut, highcut, RATE, order=4)
    max_level = np.max(data)
    # mean_level = np.mean(data)
    data = normalize(data)
    data = data * np.hamming(len(data))

    fft_complex = np.fft.fft(data, n=CHUNK)
    left, right = np.split(np.abs(fft_complex), 2)
    fft_complex = np.add(left, right[::-1])

    x = np.linspace(0, len(fft_complex), len(fft_complex))

    try:
        spline = UnivariateSpline(x, fft_complex, s=0) 
        fitted_curve = spline(x)
########################################################################    
        # (a, mu, sig), _ = curve_fit(gauss, x, fft_complex, maxfev = 2000)
        # fitted_curve = gauss(x, a, mu, sig)
        # fitted_curve = fitted_curve - np.min(fitted_curve)
########################################################################         
        # popt, pcov = curve_fit(Lorentzian3, x, fft_complex)
        # fitted_curve = Lorentzian3(x, *popt)                
    except:
        fitted_curve = np.zeros_like(x)    

    terms = 20 # number of terms for the Fourier series
    Y = np.fft.fft(fitted_curve)
    np.put(Y, range(terms+1, len(fitted_curve)), 0.0) # zero-ing coefficients above "terms"
    fitted_curve = np.fft.ifft(Y)  

    max_fitted = np.max(fitted_curve)
    fitted_curve = fitted_curve - max_fitted * 0.5
    k = [idx for idx, val in enumerate(fitted_curve) if val > 0]

    try:
        s0 = (f_vec[-1]*k[0])/len(f_vec)
        s1 = (f_vec[-1]*k[-1])/len(f_vec)
        center_f3dB = np.sqrt(s0*s1)
    except:
        s0 = 1
        s1 = 1
        center_f3dB = 1

###### Fitting 0dB frequency #########
    fit_point = 25
    x_slope = np.arange(0, fit_point, 1)
    _3dB_rest = [i for i in fitted_curve if i > 0]
    if _3dB_rest:
        # x_mean = int(len(_3dB_rest)/2) 
        # y_center = np.mean(_3dB_rest[x_mean-fit_point:x_mean+fit_point])
        y_center = np.mean(_3dB_rest)
        # y_center = np.median(_3dB_rest)
        max_fitted2 = y_center * 4
    else:
        y_center = 1 

####  left  
    _3dB_restL = _3dB_rest[:fit_point]
    try:
        popt, _ = curve_fit(linear, x_slope, _3dB_restL)
        a, b = popt
        b = 0
        _0dB_fL = int((y_center-b)/a)
        x_L = np.arange(0, _0dB_fL, 1)
        left_fit = linear(x_L, a, b)
        _0dB_fL = _0dB_fL + k[0]
    except:
        _0dB_fL = 0

####  right
    _3dB_restR = _3dB_rest[-fit_point:]
    try:
        popt, _ = curve_fit(linear, x_slope, _3dB_restR)
        a, b = popt
        b = 0
        _0dB_fR = int((y_center-b)/a)
        x_R = np.arange(0, np.abs(_0dB_fR), 1)
        right_fit = linear(x_R, a, b)
        _0dB_fR = k[-1] + _0dB_fR
    except:
        _0dB_fR = 0         

    s2= (f_vec[-1]*_0dB_fL)/len(f_vec)
    s3= (f_vec[-1]*_0dB_fR)/len(f_vec)
    center_f0dB = np.sqrt(s2*s3)

    try:
        band[0]= s0
        band[1]= s1
        band[2]= s2
        band[3]= s3
    except:
        band = [0000,0000,0000,0000]     
   

    max_val = np.max(fft_complex)
    scale_value = SCREEN_HEIGHT / max_val
    scale_fitted = SCREEN_HEIGHT / max_fitted
    scale_y = SCREEN_WIDTH/len(f_vec)

    fft_complex = resample(fft_complex,SCREEN_WIDTH)
    fitted_curve = resample(fitted_curve,SCREEN_WIDTH)
    center_f = np.ceil(center_f0dB/1000)
    pre_str = pre_amp(center_f)
 
    for i,v in enumerate(fft_complex):
        dist = np.real(fft_complex[i])
        mapped_dist = dist * scale_value

        mapped_fitted = fitted_curve[i] * scale_fitted
        
        pygame.draw.line(screen, DARKYELLOW, (i, SCREEN_HEIGHT), (i, SCREEN_HEIGHT - int(mapped_fitted)),1)
        pygame.draw.circle(screen,RED,[i,SCREEN_HEIGHT-int(mapped_dist)],2)

        band_text = band_font.render('Band: %.0f / %.0f' %(band[0],band[1]), True, (255, 255, 255) , (bg_color, bg_color, bg_color))
        band_textRect = band_text.get_rect()
        band_textRect.x, band_textRect.y = round(0.015*SCREEN_WIDTH), round(0.09*SCREEN_HEIGHT)

        level_text = level_font.render('Level: %.0f' %(20 * np.log10(max_level)), True, (255, 255, 255) , (bg_color, bg_color, bg_color))
        level_textRect = level_text.get_rect()
        level_textRect.x, level_textRect.y = round(0.015*SCREEN_WIDTH), round(0.2*SCREEN_HEIGHT)

        # preamp_text = preamp_font.render('Pre Amp: '+ (pre_str), True, (255, 255, 255) , (bg_color, bg_color, bg_color))
        preamp_text = preamp_font.render('Pre Amp: %.0f / %.0f' %(band[2],band[3]), True, (255, 255, 255) , (bg_color, bg_color, bg_color))
        preamp_textRect = preamp_text.get_rect()
        preamp_textRect.x, preamp_textRect.y = round(0.015*SCREEN_WIDTH), round(0.3*SCREEN_HEIGHT)

        screen.blit(band_text, band_textRect)
        screen.blit(level_text, level_textRect)
        screen.blit(preamp_text, preamp_textRect)

    pygame.draw.line(screen, WHITE, (k[0]*scale_y,SCREEN_HEIGHT), (int(_0dB_fL*scale_y), max_fitted2 * scale_value),2)
    pygame.draw.line(screen, WHITE, (_0dB_fR*scale_y, max_fitted2 * scale_value),(k[-1]*scale_y,SCREEN_HEIGHT),2)
    pygame.draw.line(screen, WHITE, (_0dB_fL*scale_y, max_fitted2 * scale_value),(_0dB_fR*scale_y,max_fitted2 * scale_value),2)

    pygame.display.flip()
    end = time.time()
    # clock.tick(25)
    # print(end - start)

    
