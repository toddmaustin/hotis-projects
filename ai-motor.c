#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/param.h>
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

struct HandState {
  int x, y;
  int theta;
  int mag;
};

struct HandState lHand, rHand;

void
get_handstate(FILE *fp, struct HandState *lHand, struct HandState *rHand)
{
  do {
    fgets(linebuf, sizeof(linebuf), fp);
  } while (!(linebuf[0] == '#' && linebuf[3] == '#'));

  // we have a valid "#xx#" command, parse it...
  if (strncmp(linebuf, "#fi#", 4) == 0)
  {
    // finish processing
    printf("#fi# command received...shutting down\n");
    pclose(fp);
    exit(0);
  }
  else if (strncmp(linebuf, "#lh#", 4) == 0)
  {
    printf("#lh# command received...\n");
    sscanf(linebuf, "#lh# ( %d , %d ), %d , %d", &lHand->x, &lHand->y, &lHand->theta, &lHand->mag);
    printf("#lh# (%d,%d),%d,%d\n", lHand->x, lHand->y, lHand->theta, lHand->mag);
  }
  else if (strncmp(linebuf, "#rh#", 4) == 0)
  {
    printf("#rh# command received...\n");
    sscanf(linebuf, "#rh# ( %d , %d ), %d , %d", &rHand->x, &rHand->y, &rHand->theta, &rHand->mag);
    printf("#rh# (%d,%d),%d,%d\n", rHand->x, rHand->y, rHand->theta, rHand->mag);
  }
  else
    printf("WARNING: Invalid command received...\n");
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
  FILE *fp = open_process("stdbuf -oL ./handstate.sh");

  printf("Raspberry Pi AI motor controller\n");

  for (;;)
  {
    unsigned xval = analogRead(100);
    unsigned speed;

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
    printf("Analog joystick: theta == %4d, speed = %4d\n", lHand.theta, speed);
  }

  return 0 ;
}

