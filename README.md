![Screenshot](bandpass.jpg)
# Bandpass filter Tester on RPi 
measures the cutoff frequency of the bandpass filter. input signal up to 80Khz.
Used pygame-1.9.5,  Adafruit PiTFT - 2.8" Touchscreen Display for Raspberry Pi, HIFIBERRY DAC+ ADC.
in Display, you need to cut the grio 18. 

# install 
use only Raspberry Pi OS Lite.

I could not figure out why in "Raspberry Pi OS with desktop" the input range is limited to 20Khz. 

# install PiTFT
https://learn.adafruit.com/adafruit-2-8-pitft-capacitive-touch

git clone https://github.com/adafruit/Raspberry-Pi-Installer-Scripts.git

cd Raspberry-Pi-Installer-Scripts/

sudo pip3 install Click adafruit-python-shell

sudo python3 adafruit-pitft.py --display=28c --rotation=90 --install-type=console

# install pygame
sudo apt-get install libsdl2-2.0-0 libsdl2-gfx-1.0-0 libsdl2-image-2.0-0 libsdl2-mixer-2.0-0 libsdl2-ttf-2.0-0 python3-sdl2

sudo pip3 install pygame==1.9.5

test of available fonts
import pygame
pygame.init()
print(pygame.font.get_fonts())
if None
sudo apt-get install ttf-mscorefonts-installer

# edit in /boot/config.txt
#dtparam=audio=on

dtoverlay=hifiberry-dacplusadc

# edit for user PI
sudo adduser pi tty

sudo chmod g+rw /dev/tty0

# edit udev
sudo nano /lib/udev/rules.d/50-udev-default.rules

change line:

SUBSYSTEM=="tty", KERNEL=="tty[0-9]*", GROUP="tty", MODE="0620"

to:

SUBSYSTEM=="tty", KERNEL=="tty[0-9]*", GROUP="tty", MODE="0660"

# Features

# References:

# fix Git error: object file is empty
https://stackoverflow.com/questions/11706215/how-to-fix-git-error-object-file-is-empty