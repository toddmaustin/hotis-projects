#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/param.h>
#include "core.h"

void setup(void);
void loop(void);

char linebuf[1024];

// open the program via a shell command
FILE *
open_process(const char* cmd)
{
  FILE *fp = popen(cmd, "r");
  if (!fp) return NULL;

  // drain the incoming pipe until process start #st#
  do {
    fgets(linebuf, sizeof(linebuf), fp);
  } while (strcmp(linebuf, "#st#\n") != 0);

  return fp;
}

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

// close the stream and terminate the process
void
close_process(FILE *fp)
{
  pclose(fp);
}

int
main(void)
{
  setup();

  for (;;)
  {
    loop();
  }

  // should not get here!
  abort();
}

