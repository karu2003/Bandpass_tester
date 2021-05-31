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
g_amplitude = 17750  # 18550 - 3.3V P2P
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
    data = normalize(data)
    data = data * np.hamming(len(data))

    fft_complex = np.fft.fft(data, n=CHUNK)
    left, right = np.split(np.abs(fft_complex), 2)
    fft_complex = np.add(left, right[::-1])

    x = np.linspace(0, len(fft_complex), len(fft_complex))
    spline = UnivariateSpline(x, fft_complex, s=0) 
    fitted_curve = spline(x)
  
    fitted_curve = fitted_curve - np.min(fitted_curve)
    max_fitted = np.max(fitted_curve)
    fitted_curve2 = fitted_curve - max_fitted*0.5
    r = [idx for idx, val in enumerate(fitted_curve2) if val > 0]
  
    try:
        band[0]= (f_vec[-1]*r[0])/len(f_vec)
        band[1]= (f_vec[-1]*r[-1])/len(f_vec)
    except:
        band = [0000,0000]     

    max_val = np.max(fft_complex)
    scale_value = SCREEN_HEIGHT / max_val
    scale_fitted = SCREEN_HEIGHT / max_fitted

    fft_complex = resample(fft_complex,SCREEN_WIDTH)
    fitted_curve = resample(fitted_curve,SCREEN_WIDTH)
    fr = np.sqrt(band[0]*band[1])
    center_f = np.ceil(fr/1000)
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

        preamp_text = preamp_font.render('Pre Amp: '+ (pre_str), True, (255, 255, 255) , (bg_color, bg_color, bg_color))
        preamp_textRect = preamp_text.get_rect()
        preamp_textRect.x, preamp_textRect.y = round(0.015*SCREEN_WIDTH), round(0.3*SCREEN_HEIGHT)

        screen.blit(band_text, band_textRect)
        screen.blit(level_text, level_textRect)
        screen.blit(preamp_text, preamp_textRect)

    pygame.display.flip()
    end = time.time()
    # clock.tick(25)
    # print(end - start)

    
