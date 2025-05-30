#ifndef __CORE_H__
#define __CORE_H__

struct HandState {
  int x, y;
  int theta;
  int mag;
};

// open a subordinate process, read its outputs from FILE*
FILE *open_process(const char* cmd);

// get the current state of the user's left and right hands
void get_handstate(FILE *fp, struct HandState *lHand, struct HandState *rHand);

// close the stream and terminate the process
void close_process(FILE *fp);

#endif /* __CORE_H__ */

