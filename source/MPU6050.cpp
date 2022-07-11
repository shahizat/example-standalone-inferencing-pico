#include<stdio.h>
#include "pico/stdlib.h"
#include "pico/time.h"
#include "hardware/i2c.h"
#include "hardware/gpio.h"
#include "mpu6050.h"


void mpu6050_reset()
{
    uint8_t reg[] = {0x6B,0x00};
    i2c_write_blocking(I2C_PORT,MPU6050_ADDRESS,reg,sizeof(reg),false);
}
void mpu6050_read(int16_t accelerometer[3])
{
    uint8_t buffer[6];
    uint8_t reg = 0x3B;
    i2c_write_blocking(I2C_PORT,MPU6050_ADDRESS,&reg,sizeof(reg),true);
    i2c_read_blocking(I2C_PORT,MPU6050_ADDRESS,buffer,sizeof(buffer),false);

    accelerometer[0] = (buffer[0] << 8) | buffer[1]; 
    accelerometer[1] = (buffer[2] << 8) | buffer[3];
    accelerometer[2] = (buffer[4] << 8) | buffer[5];

}