#!/usr/bin/env python
"""python module for the SC18IS602B I2C-bus to SPI bridge chips

created 09, 06, 2021
last modified 10, 06, 2021
Copyright 2021 Andrew Buckin
"""
import smbus
import time

GPIOPinMode = {
    "QUASI_BIDIRECTIONAL":    0B00,
    "PUSH_PULL":              0B01,
    "INPUT_ONLY":             0B10,
    "OPEN_DRAIN":             0B11
}

# SPI speeds
SPI_Speed = {
    "CLK_1843_kHz": 0B00,
    "CLK_461_kHz":  0B01,
    "CLK_115_kHz":  0B10,
    "CLK_58_kHz":   0B11
}

# SPI speeds
Order = {
    "MSB":  0B0 << 5,
    "LSB":  0B1 << 5
}

# SPI modes
SPI_Mode = {
    "MODE_0": 0B00 << 2,        # CPOL: 0  CPHA: 0
    "MODE_1": 0B01 << 2,        # CPOL: 0  CPHA: 1
    "MODE_2": 0B10 << 2,        # CPOL: 1  CPHA: 0
    "MODE_3": 0B11 << 2         # CPOL: 1  CPHA: 1
}

# Function IDs F0..F7
SPI_Interface = 0xF0
Clear_Interrupt = 0xF1
Idle_mode = 0xF2
GPIO_Write = 0xF4
GPIO_Read = 0xF5
GPIO_Enable = 0xF6
GPIO_Configuration = 0xF7
DATABUFFER_DEPTH = 32  # 200


class SC18IS602B:

    def __init__(self, i2cAddress=0x28, speed="CLK_1843_kHz", mode="MODE_0", order="MSB"):

        if speed not in SPI_Speed:
            raise ValueError(
                'SPI_Speed must be one of: CLK_1843_kHz,CLK_461_kHz,CLK_115_kHz,CLK_58_kHz')
        if mode not in SPI_Mode:
            raise ValueError(
                'SPI_Mode must be one of: MODE_0,MODE_2,MODE_3,MODE_4')
        if order not in Order:
            raise ValueError('Order must be one of: MSB,LSB')

        self.i2cAddress = i2cAddress
        self.bus = smbus.SMBus(1)
        self.speed = SPI_Speed[speed]
        self.mode = SPI_Mode[mode]
        self.order = Order[order]

        try:
            self.bus.write_byte_data(
                self.i2cAddress, SPI_Interface, self.order | self.mode | self.speed)
        except IOError:
            print("No i2c device at address:", self.i2cAddress,)
        return

    def clearInterrupt(self):
        self.bus.write_byte(self.i2cAddress, Clear_Interrupt)
        return

    def setLowPowerMode(self):
        self.bus.write_byte(self.i2cAddress, Idle_mode)
        return

    def spiTransfer(self, slaveNum=0, txData=[], rxLen=1):

        if(slaveNum < 0 or slaveNum > 3):
            print('CS Data is more as 32 Byte')
            return False

        if(rxLen > DATABUFFER_DEPTH):
            print('Data is more as 32 Byte')
            return False

        functionID = (1 << slaveNum)

        # if (txLen == 1):
        #     txData = txData[0]
        #     self.bus.write_byte_data(self.i2cAddress, functionID, txData)
        #     return self.bus.read_byte(self.i2cAddress)

        self.bus.write_i2c_block_data(self.i2cAddress, functionID, txData)
        time.sleep(0.00008 * rxLen)
        return self.bus.read_i2c_block_data(self.i2cAddress, 0xFF, rxLen)


if __name__ == "__main__":

    SC = SC18IS602B(i2cAddress=0x28, speed="CLK_1843_kHz",
                    mode="MODE_0", order="MSB")
    # SC = SC18IS602B()
    txData = list(range(0, 32, 1))  # data loop DO>DI
    # txData = [0x55]
    rxData = SC.spiTransfer(slaveNum=0, txData=txData, rxLen=len(txData))
    SC.clearInterrupt()
    # print("0x{0:02x}".format(rxData))
    print(rxData)
