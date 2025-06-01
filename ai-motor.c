#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/param.h>
#include <wiringPi.h>
#include <mcp3004.h>
#include "core.h"

#define	L293_ENABLE	26
#define	L293_INPUT1	21
#define	L293_INPUT2	22


#define PWM_MAXVAL	1024

struct HandState lHand, rHand;
FILE *fp;

void
setup(void)
{
  printf("INFO: Raspberry Pi AI motor controller.\n");

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

  // open the AI engine
  fp = open_process("stdbuf -oL ./handstate.sh");
}

void
loop(void)
{
  unsigned xval, speed;

  get_handstate(fp, &lHand, &rHand);

  // compute new xval
  xval = (int)(abs(lHand.theta) * 512.0/180.0);
    

  if (lHand.theta >= 0)
  {
    // forward
    digitalWrite(L293_INPUT1, HIGH);
    digitalWrite(L293_INPUT2, LOW);
    speed = MIN(1024, xval * 6);
  }
  else
  {
    // backward
    digitalWrite(L293_INPUT1, LOW);
    digitalWrite(L293_INPUT2, HIGH);
    speed = MIN(1024, xval * 6);
  }

  pwmWrite(L293_ENABLE, speed);
  printf("INFO: Analog joystick: theta == %4d, speed = %4d\n", lHand.theta, speed);
}

