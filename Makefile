CC = gcc
OFLAGS = -Wall -O0 -g
LIBS = -lwiringPi

build: motor joystick joymotor ai-motor

motor: motor.c
	$(CC) $(OFLAGS) -o motor motor.c $(LIBS)

joystick: joystick.c
	$(CC) $(OFLAGS) -o joystick joystick.c $(LIBS)

joymotor: joymotor.c
	$(CC) $(OFLAGS) -o joymotor joymotor.c $(LIBS)

ai-motor: ai-motor.c
	$(CC) $(OFLAGS) -o ai-motor ai-motor.c $(LIBS)

clean:
	rm -f motor joystick joymotor ai-motor FOO* *.o
