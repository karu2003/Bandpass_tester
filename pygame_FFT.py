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
from pygame import event, fastevent 
import RPi.GPIO as GPIO
import matplotlib.pyplot as plt
from chirp3 import lchirp
from SC18IS602B import SC18IS602B
from MCP230XX import MCP230XX
from LTC1380 import LTC1380

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

SCREEN_WIDTH = 320
SCREEN_HEIGHT = 240
SCREEN_SIZE = (SCREEN_WIDTH, SCREEN_HEIGHT)

splash_screen = {
    0: [(SCREEN_HEIGHT / 30) / 2, 25, 30, "couriernew", "PreAmp Tester"]
}

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
        23:'18/34',
        24:'18/34',
        25:'18/34',
        26:'18/34',
        46:'40/80',
        47:'40/80',
        6:'xxxxx'
        }
    return switcher.get(i,"Invalid pre AMP")

# --Define a function to show a screen of text with button labels
# Note: text items in the screen dictionary can be changed before displaying
def show_text_menu(menuname, highlite, buttons):  # buttons can be None
    # screen.fill(BLACK)  # blank the display
    # Build button labels first, so menu can overlap on leading blanks
    if buttons != None:  # see if there are buttons to show
        line = 0  # reset our line count for the labels
        for line in buttons:  # go through the  button line vslues
            linedata = buttons[line]
            # myfont = pygame.font.SysFont(linedata[3], linedata[1])  # use the font selected
            myfont = pygame.font.Font(linedata[3], linedata[1])
            textsurface = myfont.render(linedata[4], False, WHITE, BLACK)  # write the text
            # show the text
            screen.blit(textsurface, (linedata[2], linedata[1] * linedata[0]))
            line = line + 1  # go to the next line
    # Build the rest of the menu
    if menuname != None:
        line = 0  # start showing at line zero
        for line in menuname:  # go through the line values
        # get the value list from the menu dictionary
            linedata = menuname[line]
            myfont = pygame.font.SysFont(linedata[3], linedata[1])  # use the font selected
            # Build text and position & highlighting a line, if within range
            if line == highlite:  # check if we should highlight this line
                textsurface = myfont.render(
                    linedata[4], False, BLACK, WHITE
                )  # highlight it
            else:
                textsurface = myfont.render(
                    linedata[4], False, WHITE, BLACK
                )  # no highlight
            # add the line to the screen
            screen.blit(textsurface, (linedata[2], linedata[1] * linedata[0]))
            line = line + 1
    # Show the new screen
    # pygame.display.update()  # show it all

button_menu1 = {
    0: [2.5, 15, 268, 'freesansbold.ttf',   "Start->"],
    1: [6.5, 15, 260, 'freesansbold.ttf',   " -0dB->"],
    2: [10.5, 15, 260, 'freesansbold.ttf',   "Gain->"],
    3: [14.8, 15, 268, 'freesansbold.ttf',   "Main->"],
    4: [10, 20, 120, 'freesansbold.ttf', "PreAmp"],
}
gains  = {
    # 0:['-120dB',0x00],
    0:['   0dB->',0x11],
    1:['   6dB->',0x22],
    2:['  12dB->',0x33],
    3:['18.1dB->',0x44],
    4:['24.1dB->',0x55],
    5:['30.1dB->',0x66],
    # 6:['36.1dB->',0x77],
    # 8:['-12xdB',0x88],
}

gain = 0
Bands = {
     7:[28,40,52,63,75,86,66,78,'7/17'],
    12:[26,38,50,62,74,85,66,77,'12/24'],
    18:[26,38,50,62,74,85,65,77,'18/34'],
    40:[25,36,48,60,72,83,66,75,'40/80'],
}

def check_Level(max_level_dB,num,accuracy):
    if max_level_dB in range(num-accuracy,num+accuracy):
        return True
    return False

def check_Band(num):
    global Channel, Typ
    if not Typ:
        if Channel: # Lim
            if num in range(11,13):
                return 7    
            if num in range(16,18):
                return 12
            if num in range(23,26):
                return 18                
            if num in range(39,40):
                return 40    
        else: #Main
            if num in range(11,13):
                return 7    
            if num in range(18,20):
                return 12
            if num in range(23,26):
                return 18                
            if num in range(46,47):
                return 40
    else:
        pass            

def gpiobut(channel):
    if channel == 17:  # check for button 1
        fastevent.post(pygame.event.Event(pygame.USEREVENT + 2, button=1))
    elif channel == 22:  # check for button 2
        fastevent.post(pygame.event.Event(pygame.USEREVENT + 2, button=2))
    elif channel == 23:  # check for button 3
        fastevent.post(pygame.event.Event(pygame.USEREVENT + 2, button=3))
    elif channel == 27:  # check for button 4
        fastevent.post(pygame.event.Event(pygame.USEREVENT + 2, button=4))

def set_start():
    global Run, button_menu1
    if Run:
        button_menu1[0][4] = " Stop->"
    else:
        button_menu1[0][4] = "Start->"

def set_gain_low():
    global G_Low, button_menu1
    if G_Low:
        button_menu1[1][4] = "-20dB->"
    else:
        button_menu1[1][4] = " -0dB->"

def set_gain(gain):
    global gains, button_menu1
    button_menu1[2][4] = gains[gain][0]        

def set_typ():
    global Typ, button_menu1
    if Typ:
        button_menu1[4][4] = "USBL"
    else:
        button_menu1[4][4] = "PreAmp"

def set_channel():
    global Channel, button_menu1
    if Channel:
        button_menu1[3][4] = "Lim->"
    else:
        button_menu1[3][4] = "Main->"



p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
    channels=1,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK)

USBL_On = 1
PreAmp_On = 2
Gain_Low  = 3
RX_LIM  = 5
RX_MAIN  = 6
ON = 1
OFF = 0

MCP = MCP230XX('MCP23008', i2cAddress=0x20)
MUX = LTC1380(i2cAddress=0x48)

MCP.set_mode(USBL_On, 'output')
MCP.set_mode(PreAmp_On, 'output')
MCP.set_mode(Gain_Low, 'output')
MUX.SetChannel(RX_MAIN)
MCP.output(USBL_On, OFF)
MCP.output(PreAmp_On, OFF)
MCP.output(Gain_Low, OFF)

LTC69122 = SC18IS602B(i2cAddress=0x28, speed="CLK_1843_kHz", mode="MODE_0", order="MSB")
# LTC69122.spiTransfer(slaveNum=0, txData=[gains[gain][1]], rxLen=len([gains[gain][1]]))


print("*recording")

os.putenv("SDL_FBDEV", "/dev/fb1")  
os.putenv("SDL_MOUSEDRV", "TSLIB")  
# Mouse is PiTFT touchscreen
os.putenv("SDL_MOUSEDEV", "/dev/input/touchscreen")
os.putenv("SDL_AUDIODRIVER", "alsa")

pygame.mixer.pre_init(frequency=int(RATE), size=-16, channels=2, buffer=blocksize)
pygame.init()
pygame.mouse.set_visible(False)
screen = pygame.display.set_mode(SCREEN_SIZE)
fastevent.init()
pygame.event.set_blocked(pygame.MOUSEMOTION)
pygame.event.set_blocked(pygame.MOUSEBUTTONUP)
# pygame.font.init()

clock = pygame.time.Clock()

Step_interval = 500 # ms

pygame.time.set_timer(USEREVENT + 1, Step_interval)

band_font = pygame.font.Font('freesansbold.ttf', round(0.07*SCREEN_HEIGHT))
level_font = pygame.font.Font('freesansbold.ttf', round(0.07*SCREEN_HEIGHT))
preamp_font = pygame.font.Font('freesansbold.ttf', round(0.07*SCREEN_HEIGHT))
button_font = pygame.font.Font('freesansbold.ttf', round(0.05*SCREEN_HEIGHT))



band = [7000,17000]
M_Band = 0
bg_color = 60
terms = 30
Run = 0
Typ = 0
G_Low = 0
Channel = 0
max_level = 0
max_level_dB = 0
fault = 1 
set_start()
set_gain_low() 
set_gain(gain)
set_typ()
set_channel()

show_text_menu(splash_screen, None, None)

GPIO.setmode(GPIO.BCM)  # use BCM chip's numbering scheme vs. pin numbers
GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # PiTFT button 1
GPIO.setup(22, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # PiTFT button 2
GPIO.setup(23, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # PiTFT button 3
GPIO.setup(27, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # PiTFT button 4
# Define GPIO button event handlers for the PiTFT 2423
GPIO.add_event_detect(17, GPIO.FALLING, callback=gpiobut, bouncetime=300)
GPIO.add_event_detect(22, GPIO.FALLING, callback=gpiobut, bouncetime=300)
GPIO.add_event_detect(23, GPIO.FALLING, callback=gpiobut, bouncetime=300)
GPIO.add_event_detect(27, GPIO.FALLING, callback=gpiobut, bouncetime=300)

sweep_gen()
sound.set_volume(0.01)
sound.play(-1)
step = 0

done = False
while not done:
    screen.fill((0,0,0))
    s = 0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
            MCP.output(PreAmp_On, OFF)
            MCP.output(USBL_On, OFF)
            break

        elif event.type == pygame.USEREVENT + 1:
            if Run and M_Band:      
                step_len = len(Bands[M_Band])-1
                if step == step_len-1:
                    Channel = 0
                    G_Low = 0
                    set_channel()
                    MUX.SetChannel(RX_MAIN)
                    Run = False
                    set_start()
                    MCP.output(PreAmp_On, ON if Run else OFF)
                    if fault:
                        button_menu1[4][4] = "PreAmp OK"
                    else:
                        button_menu1[4][4] = "PreAmp Fault"
                fault = fault and check_Level(int(max_level_dB),Bands[M_Band][step],2)
                if not fault:
                    print(step)
                step += 1
                step = step % step_len
                if step in range(len(gains)):
                    LTC69122.spiTransfer(slaveNum=0, txData=[gains[step][1]], rxLen=len([gains[step][1]]))
                    set_gain(step)
                    time.sleep(0.02)
                if step == step_len-2:
                    G_Low = 1
                    set_gain_low()
                    MCP.output(Gain_Low, 1)
                    time.sleep(0.02)    
                if step == step_len-1:
                    G_Low = 0
                    set_gain_low()
                    MCP.output(Gain_Low, 0)
                    time.sleep(0.02) 
                    Channel = 1
                    set_channel()
                    MUX.SetChannel(RX_LIM)
                    time.sleep(0.02)

        elif event.type == pygame.USEREVENT + 2:
            if event.button == 1:    # button 1 = GPIO 17
                Run = not Run
                fault = 1
                set_start()
                set_typ()
                MCP.output(PreAmp_On, ON if Run else OFF)
                LTC69122.spiTransfer(slaveNum=0, txData=[gains[gain][1]], rxLen=len([gains[gain][1]]))
                time.sleep(0.02) 
                if not Run:
                    gain = 0
                    set_gain(gain)                    

            elif event.button == 2:  # button 2 = GPIO 22
                G_Low = not G_Low
                set_gain_low()
                MCP.output(Gain_Low, ON if G_Low else OFF)

            elif event.button == 3:  # button 3 = GPIO 23
                if Run:
                    gain += 1
                    gain = gain % len(gains)
                    set_gain(gain)
                    LTC69122.spiTransfer(slaveNum=0, txData=[gains[gain][1]], rxLen=len([gains[gain][1]]))

            elif event.button == 4:  # button 3 = GPIO 27
                if not Typ:
                    Channel = not Channel 
                    set_channel()
                    if Channel:
                        MUX.SetChannel(RX_LIM)
                    else:
                        MUX.SetChannel(RX_MAIN)     

        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            Typ = not Typ 
            set_typ()
         

    start = time.time()
    buff = stream.read(CHUNK,exception_on_overflow=False) 
    data = np.frombuffer(buff, dtype=np.int16)
    data = butter_bandpass_filter(data, lowcut, highcut, RATE, order=4)
    max_level = np.max(data)
    max_level_dB = 20 * np.log10(max_level)
    data = normalize(data)
    data = data * np.hamming(len(data))

    fft_complex = np.fft.fft(data, n=CHUNK)
    left, right = np.split(np.abs(fft_complex), 2)
    fft_complex = np.add(left, right[::-1])

    Y = np.fft.fft(fft_complex)
    np.put(Y, range(terms+1, len(fft_complex)), 0.0) # zero-ing coefficients above "terms"
    fft_complex = np.fft.ifft(Y)
    fft_complex = fft_complex - np.min(fft_complex)


    x = np.linspace(0, len(fft_complex), len(fft_complex))
    spline = UnivariateSpline(x, fft_complex, s=0) 
    fitted_curve = spline(x)
  
    max_fitted = np.max(fitted_curve)
    fitted_curve2 = fitted_curve - max_fitted * 0.5
    fitted_curve2 = np.sqrt(fitted_curve2)
    fitted_curve2[np.isnan(fitted_curve2)] = 0
 
###### Fitting 0dB frequency #########
    # fit_point = 5
    # x_slope = np.arange(0, fit_point, 1)
    # _3dB_slope = [i for i in fitted_curve2 if i > 0]
    # x_center = int(len(_3dB_slope)/2)
    # y_center = _3dB_slope[x_center]
    # # y_center = np.mean(_3dB_slope[x_center-fit_point:x_center-fit_point])
    # _3dB_slopeL = _3dB_slope[:fit_point]

    # popt, _ = curve_fit(linear, x_slope, _3dB_slopeL)
    # a, b = popt
    # _0dB_fL = int((y_center-b)/a)

    # _3dB_slopeR = _3dB_slope[-fit_point:]
    # popt, _ = curve_fit(linear, x_slope, _3dB_slopeR)
    # a, b = popt
    # _0dB_fR = int((y_center-b)/a)
###################################### 


    r = [idx for idx, val in enumerate(fitted_curve2) if val > 0]
  
    try:
        band[0]= (f_vec[-1]*r[0])/len(f_vec)
        band[1]= (f_vec[-1]*r[-1])/len(f_vec)
        band[2]= 0#(f_vec[-1]*r[_0dB_fL])/len(f_vec)
        band[3]= 0#(f_vec[-1]*r[-_0dB_fR])/len(f_vec)s
    except:
        band = [0000,0000,0000,0000]     

    max_val = np.max(fft_complex)
    scale_value = SCREEN_HEIGHT / max_val
    scale_fitted = SCREEN_HEIGHT / max_fitted

    fft_complex = resample(fft_complex,SCREEN_WIDTH)
    fitted_curve = resample(fitted_curve,SCREEN_WIDTH)
    fr = np.sqrt(band[0]*band[1])
    center_f = np.ceil(fr/1000)
    # print(center_f)
    # pre_str = pre_amp(center_f)
    M_Band = check_Band(center_f)
    if M_Band:
        pre_str = Bands[M_Band][-1]
    else:
        pre_str = "Invalid pre AMP"    
 
    for i,v in enumerate(fft_complex):
        dist = np.real(fft_complex[i])
        mapped_dist = dist * scale_value

        mapped_fitted = fitted_curve[i] * scale_fitted
        
        pygame.draw.line(screen, DARKYELLOW, (i, SCREEN_HEIGHT), (i, SCREEN_HEIGHT - int(mapped_fitted)),1)
        pygame.draw.circle(screen,RED,[i,SCREEN_HEIGHT-int(mapped_dist)],2)

        band_text = band_font.render('Band: %.0f / %.0f' %(band[0],band[1]), True, (255, 255, 255) , (bg_color, bg_color, bg_color))
        band_textRect = band_text.get_rect()
        band_textRect.x, band_textRect.y = round(0.015*SCREEN_WIDTH), round(0.09*SCREEN_HEIGHT)

        level_text = level_font.render('Level: %.0f' %(max_level_dB), True, (255, 255, 255) , (bg_color, bg_color, bg_color))
        level_textRect = level_text.get_rect()
        level_textRect.x, level_textRect.y = round(0.015*SCREEN_WIDTH), round(0.2*SCREEN_HEIGHT)

        preamp_text = preamp_font.render('Pre Amp: '+ (pre_str), True, (255, 255, 255) , (bg_color, bg_color, bg_color))
        # preamp_text = preamp_font.render('Pre Amp: %.0f / %.0f' %(band[2],band[3]), True, (255, 255, 255) , (bg_color, bg_color, bg_color))
        preamp_textRect = preamp_text.get_rect()
        preamp_textRect.x, preamp_textRect.y = round(0.015*SCREEN_WIDTH), round(0.3*SCREEN_HEIGHT)

        screen.blit(band_text, band_textRect)
        screen.blit(level_text, level_textRect)
        screen.blit(preamp_text, preamp_textRect)
        
    show_text_menu(None, None, button_menu1)
    pygame.display.flip()
    end = time.time()
    # clock.tick(25)
    # print(end - start)

    
