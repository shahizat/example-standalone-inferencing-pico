#include <stdio.h>
#include <string.h>
#include "pico/stdlib.h"
#include "pico/binary_info.h"
#include "hardware/i2c.h"
#include "mpu6050.h"
#include "ei_run_classifier.h"

const uint LED_PIN = 25;
#define SCL 1
#define SDA 0
#define CONVERT_G_TO_MS2    9.80665f

static bool debug_nn = false;

int main()
{
  stdio_usb_init();

  gpio_init(LED_PIN);
  gpio_set_dir(LED_PIN, GPIO_OUT);

    i2c_init(I2C_PORT,100000);
    gpio_set_function(SCL,GPIO_FUNC_I2C);
    gpio_set_function(SDA,GPIO_FUNC_I2C);
    gpio_pull_up(SCL);
    gpio_pull_up(SDA);
    mpu6050_reset();
    
 int16_t accelerometer[3];
  ei_impulse_result_t result = {0};

  while (true)
  {
    ei_printf("Edge Impulse standalone inferencing (Raspberry Pi Pico)\n");


    while (1)
    {
      // blink LED
      gpio_put(LED_PIN, !gpio_get(LED_PIN));
      float buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE] = { 0 };

    for (size_t ix = 0; ix < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE; ix += 3) {
        // Determine the next tick (and then sleep later)
        uint64_t next_tick = ei_read_timer_us() + (EI_CLASSIFIER_INTERVAL_MS * 1000);
        mpu6050_read(accelerometer);
        buffer[ix + 0]= accelerometer[0];
        buffer[ix + 1]= accelerometer[1];
        buffer[ix + 2]= accelerometer[2];
        //IMU.readAcceleration(buffer[ix], buffer[ix + 1], buffer[ix + 2]);

        buffer[ix + 0] *= CONVERT_G_TO_MS2;
        buffer[ix + 1] *= CONVERT_G_TO_MS2;
        buffer[ix + 2] *= CONVERT_G_TO_MS2;
        sleep_us(next_tick - ei_read_timer_us());
    }

      // the features are stored into flash, and we don't want to load everything into RAM
        signal_t signal;
        int err = numpy::signal_from_buffer(buffer, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);
        if (err != 0) {
            ei_printf("Failed to create signal from buffer (%d)\n", err);
            return 1;
        }

        // Run the classifier
        ei_impulse_result_t result = { 0 };

        err = run_classifier(&signal, &result, debug_nn);
        if (err != EI_IMPULSE_OK) {
            ei_printf("ERR: Failed to run classifier (%d)\n", err);
            return 1;
        }

        // print the predictions
        ei_printf("Predictions ");
        ei_printf("(DSP: %d ms., Classification: %d ms., Anomaly: %d ms.)",
            result.timing.dsp, result.timing.classification, result.timing.anomaly);
        ei_printf(": \n");
        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            ei_printf("->  %s: %.4f%s\n", result.classification[ix].label, (result.classification[ix].value), "%");
        }
    #if EI_CLASSIFIER_HAS_ANOMALY == 1
        ei_printf("    anomaly score: %.3f\n", result.anomaly);
    #endif
        gpio_put(LED_PIN, 0);
    }

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_ACCELEROMETER
    #error "Invalid model for current sensor"
#endif

    return 0;
}
}