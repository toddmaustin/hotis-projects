#! /bin/bash

. /home/user/mx/bin/activate
cd mediapipe_hands/src/python
stdbuf -oL python -u run.py
