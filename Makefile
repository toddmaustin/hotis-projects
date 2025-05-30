CC = gcc
OFLAGS = -Wall -O0 -g
LIBS = -lwiringPi -lm

build: motor joystick joymotor ai-motor

motor: core.c motor.c
	$(CC) $(OFLAGS) -o motor core.c motor.c $(LIBS)

joystick: core.c joystick.c
	$(CC) $(OFLAGS) -o joystick core.c joystick.c $(LIBS)

joymotor: core.c joymotor.c
	$(CC) $(OFLAGS) -o joymotor core.c joymotor.c $(LIBS)

ai-motor: core.c ai-motor.c
	$(CC) $(OFLAGS) -o ai-motor core.c ai-motor.c $(LIBS)

clean:
	rm -f motor joystick joymotor ai-motor FOO* *.o
