#include <stdio.h>
#include <wiringPi.h>

// LED Pin - wiringPi pin 0 is BCM_GPIO 17.

#define	L293_ENABLE	26
#define	L293_INPUT1	21
#define	L293_INPUT2	22

#define PWM_MAXVAL	1024

int main(void)
{
  printf("Raspberry Pi motor controller\n") ;

  wiringPiSetup();
  wiringPiSetupPinType(WPI_PIN_WPI);

  // pinMode(L293_ENABLE, OUTPUT);
  pinMode(L293_ENABLE, PWM_OUTPUT);
  pinMode(L293_INPUT1, OUTPUT);
  pinMode(L293_INPUT2, OUTPUT);

  for (;;)
  {
    printf("DC motor forward...\n");
    // digitalWrite(L293_ENABLE, HIGH);
    pwmWrite(L293_ENABLE, PWM_MAXVAL);
    digitalWrite(L293_INPUT1, HIGH);
    digitalWrite(L293_INPUT2, LOW);
    delay(2000); // mS

    printf("DC motor backward...\n");
    // digitalWrite(L293_ENABLE, HIGH);
    pwmWrite(L293_ENABLE, PWM_MAXVAL);
    digitalWrite(L293_INPUT1, LOW);
    digitalWrite(L293_INPUT2, HIGH);
    delay(2000); // mS

    printf("DC motor forward (1/2 speed)...\n");
    // digitalWrite(L293_ENABLE, HIGH);
    pwmWrite(L293_ENABLE, PWM_MAXVAL/2);
    digitalWrite(L293_INPUT1, HIGH);
    digitalWrite(L293_INPUT2, LOW);
    delay(2000); // mS

    printf("DC motor backward (1/2 speed)...\n");
    // digitalWrite(L293_ENABLE, HIGH);
    pwmWrite(L293_ENABLE, PWM_MAXVAL/2);
    digitalWrite(L293_INPUT1, LOW);
    digitalWrite(L293_INPUT2, HIGH);
    delay(2000); // mS

    printf("DC motor stopped...\n");
    // digitalWrite(L293_ENABLE, LOW);
    pwmWrite(L293_ENABLE, 0);
    delay(2000); // mS
  }
  return 0;
}
