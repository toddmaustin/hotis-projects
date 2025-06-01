#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <wiringPi.h>
#include <mcp3004.h>
#include "core.h"

#define	X_INPUT	100

void
setup(void)
{
  printf("INFO: Raspberry Pi joystick reader.\n") ;

  wiringPiSetup();
  wiringPiSetupPinType(WPI_PIN_WPI);

  if (!mcp3004Setup (/* pinBase */100, /* spiChannel */0))
  {
    printf("ERROR: Cannot open ADC chip.\n");
    exit(1);
  }
}

void
loop(void)
{
  for (;;)
  {
    unsigned xval = analogRead(X_INPUT);
    printf("INFO: analog joystick: x == %4d, y == %4d\n", xval, 0);
    delay(50); // mS
  }
}
