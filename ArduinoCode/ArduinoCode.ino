#include <Servo.h>

// Setup stepper pinouts
#define X_SERVO_PIN 3
#define Y_SERVO_PIN 11


#define StaticServoA 4
#define StaticServoB 5
#define StaticServoC 6

// Setup laser pinout
#define LASER_PIN 12

// Define Servo objects
Servo xServo;
Servo yServo;

Servo staticServoA;
Servo staticServoB;
Servo staticServoC;


// Setup position variables
int baseXPos = 90;
int baseYPos = 90;

void setup() {
  //Serial
  Serial.begin(9600);

  //Pinmodes
  pinMode(LASER_PIN, OUTPUT);
  digitalWrite(LASER_PIN, HIGH); // Ensure laser is on

  pinMode(StaticServoA, OUTPUT);
  staticServoB.attach(StaticServoA);
  staticServoB.write(180);

  pinMode(StaticServoB, OUTPUT);
  staticServoB.attach(StaticServoB);
  staticServoB.write(130);

  pinMode(StaticServoC, OUTPUT);
  staticServoC.attach(StaticServoC);
  staticServoC.write(30);

  xServo.attach(X_SERVO_PIN);
  yServo.attach(Y_SERVO_PIN);

  xServo.write(baseXPos);
  yServo.write(baseYPos);

  delay(200);  // Wait for servos to reach position
  Serial.println("Alive");
}


void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    if (cmd.startsWith("MOVE")) {
      int angles[2];
      sscanf(cmd.c_str(), "MOVE %d %d", 
             &angles[0], &angles[1]);

      // constrain first so smoothMove never gets invalid targets
      angles[0] = constrain(angles[0], 0, 180);
      angles[1] = constrain(angles[1], 0, 130);

      xServo.write(angles[0]);

      yServo.write(angles[1]);
    Serial.print(angles[0]);
    Serial.print(" ");
    Serial.println(angles[1]);
    }
  }
}
