CC = gcc
OFLAGS = -Wall -O
LIBS = -lwiringPi

build: motor joystick

motor: motor.c
	$(CC) $(OFLAGS) -o motor motor.c $(LIBS)

joystick: joystick.c
	$(CC) $(OFLAGS) -o joystick joystick.c $(LIBS)

clean:
	rm -f motor joystick *.o
