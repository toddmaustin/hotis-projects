#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <wiringPi.h>
#include <mcp3004.h>

#define	L293_ENABLE	26
#define	L293_INPUT1	21
#define	L293_INPUT2	22

#define PWM_MAXVAL	1024

void
setup(void)
{
  printf("INFO: Raspberry Pi joystick motor controller.\n");

  wiringPiSetup();
  wiringPiSetupPinType(WPI_PIN_WPI);

  pinMode(L293_ENABLE, PWM_OUTPUT);
  pinMode(L293_INPUT1, OUTPUT);
  pinMode(L293_INPUT2, OUTPUT);

  if (!mcp3004Setup (/* pinBase */100, /* spiChannel */0))
  {
    printf("ERROR: Cannot open ADC chip.\n");
    exit(1);
  }
}

void
loop(void) 
{
  unsigned xval = analogRead(100);
  unsigned speed;

  if (xval < PWM_MAXVAL/2)
  {
    // forward
    digitalWrite(L293_INPUT1, HIGH);
    digitalWrite(L293_INPUT2, LOW);
    speed = ((PWM_MAXVAL/2) - xval) * 2;
  }
  else
  {
    // backward
    digitalWrite(L293_INPUT1, LOW);
    digitalWrite(L293_INPUT2, HIGH);
    speed = (xval - (PWM_MAXVAL/2)) * 2;
  }

  pwmWrite(L293_ENABLE, speed);
  printf("INFO: Analog joystick: x == %4d, y == %4d, pwmVal = %4d\n", xval, 0, speed);
}

