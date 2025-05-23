CC = gcc
OFLAGS = -Wall -O
LIBS = -lwiringPi

build: motor joystick joymotor

motor: motor.c
	$(CC) $(OFLAGS) -o motor motor.c $(LIBS)

joystick: joystick.c
	$(CC) $(OFLAGS) -o joystick joystick.c $(LIBS)

joymotor: joymotor.c
	$(CC) $(OFLAGS) -o joymotor joymotor.c $(LIBS)

clean:
	rm -f motor joystick joymotor *.o
