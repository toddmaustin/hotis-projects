#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wiringPi.h>
#include <mcp3004.h>

// LED Pin - wiringPi pin 0 is BCM_GPIO 17.

#define	L293_ENABLE	26
#define	L293_INPUT1	21
#define	L293_INPUT2	22

#define PWM_MAXVAL	1024

char linebuf[1024];

// 1) Open the program via a shell command
FILE* open_process(const char* cmd) {
    FILE *fp = popen(cmd, "r");
    if (!fp) return NULL;

    // drain the incoming pipe until process start #st#
    do {
      fgets(linebuf, sizeof(linebuf), fp);
    } while (strcmp(linebuf, "#st#\n") != 0);

    return fp;
}

// 3) Close the stream and terminate the process
void close_process(FILE *fp) {
    pclose(fp);
}

int main(void)
{
  printf("Raspberry Pi AI motor controller\n");

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
  FILE *ps = open_process("stdbuf -oL ./handstate.sh");

  printf("Raspberry Pi AI motor controller\n");

  for (;;)
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
    // printf("Analog joystick: x == %4d, y == %4d, pwmVal = %4d\n", xval, 0, speed);
  }

  close_process(ps);
  return 0 ;
}

