#include <stdio.h>
#include <stdlib.h>
#include <wiringPi.h>
#include <mcp3004.h>

// LED Pin - wiringPi pin 0 is BCM_GPIO 17.

#define	X_INPUT	0
#define	Y_INPUT	1

int main(void)
{
  printf("Raspberry Pi joystick reader\n");

  wiringPiSetup();
  wiringPiSetupPinType(WPI_PIN_WPI);

  if (!mcp3004Setup (/* pinBase */100, /* spiChannel */0))
  {
    printf("ERROR: Cannot open ADC chip.\n");
    exit(1);
  }

  // pinMode(X_INPUT, INPUT);
  // pinMode(Y_INPUT, INPUT);

  for (;;)
  {
    unsigned xval = analogRead(100);
    // unsigned yval = analogRead(Y_INPUT);
    printf("Analog joystick: x == %4d, y == %4d\n", xval, 0);
    delay(50); // mS
  }
  return 0 ;
}
