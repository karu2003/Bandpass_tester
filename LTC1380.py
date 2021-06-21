#!/usr/bin/env python
"""python module for the LTC1380

created 17, 06, 2021
last modified 17, 06, 2021
Copyright 2021 Andrew Buckin
"""
import smbus
import time

Enable = 8

class LTC1380:

    def __init__(self, i2cAddress=0x48):

        self.i2cAddress = i2cAddress
        self.bus = smbus.SMBus(1)

        try:
            self.Enable()
        except IOError:
            print("No i2c device at address:", self.i2cAddress,)
        self.Desable()    
        return

    def Enable(self):
        self.bus.write_byte(self.i2cAddress, Enable)
        return

    def Desable(self):
        self.bus.write_byte(self.i2cAddress, 0x00)
        return

    def SetChannel(self, Channel):        
        self.bus.write_byte(self.i2cAddress, Enable | Channel)
        return    

if __name__ == "__main__":

    MUX = LTC1380(i2cAddress=0x48)
    Channel = list(range(0, 8, 1))  # data loop DO>DI
    for i in Channel:
        print(i)
        MUX.SetChannel(i)
        time.sleep(0.5)
    MUX.Desable


