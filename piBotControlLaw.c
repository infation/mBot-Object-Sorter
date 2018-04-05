#include "MeMCore.h"
#include "MeIR.h"
#include <SoftwareSerial.h>


MeUltrasonicSensor sonic_sensor(3);
MeBuzzer buzzer;

MeDCMotor motor1(M1);
MeDCMotor motor2(M2);

MeIR ir_remote;

MeLineFollower line(PORT_2);

MeRGBLed led(PORT_4);

int speed = 150;

const int yellow_cube_placement = 0;
const int red_sphere_placement = 1;
const int blue_cube_placement = 2;
const int green_pyramid_placement = 3;

const int BLUE = 0;
const int GREEN = 1;
const int RED = 2;
const int YELLOW = 3;

// 0 - north
// 1 - east
// 2 - south
// 3 - west
static int direction = 3; // facing west at beginning

void setup() {
    Serial.begin(9600);
    ir_remote.begin();
}

void loop() {
/*
    // listen for the data
  if ( Serial.available()) {
        Serial.println("I recieved some data!");

        // read a numbers from serial port
        int count = Serial.read();


        // print out the received number
        if (count == '5') {
            buzzer.tone(5000, 500);
            //delay(10000);
            Serial.print("You have input: ");
            Serial.println(String(count));
        }

        if(count == '6') {
            motor1.run(speed);
            motor2.run(speed);
        }

        if(count == '7') {
            motor1.stop();
            motor2.stop();
        }
    }
*/

  roam_avoidObstacle();
  //Serial.println(sonic_sensor.distanceCm());
  //Forward();
  //gatherData();

}


void Forward() {
  motor1.run(-speed);
  motor2.run(speed);
  delay(300);
}

void SlowForward() {
  motor1.run(-(speed - 40));
  motor2.run(speed - 40);
  delay(300);
}

void Backward() {
  motor1.run(speed);
  motor2.run(-speed);
  delay(700);
}

void TurnLeft() {
  motor1.run(-speed/10);
  motor2.run(speed);
  delay(470);
}

void TurnRight() {
  motor1.run(-speed);
  motor2.run(speed/10);
  delay(470);
}

void Stop() {
  motor1.run(0);
  motor2.run(0);
  delay(100);
}



// 0 - nothing on picture
// 1 - box
// 2 - flacon

/**
* Main driver control law
* roams around and checks for objects, after which it calls the raspberry pi
*/
void roam_avoidObstacle()
{
  int distance = sonic_sensor.distanceCm();
  randomSeed(analogRead(6));
  int rand = random(2);

  if (distance > 25) {
    //Forward();
    lineFollowing();
  }
  else if (distance < 20) {

    // stop the mbot
    Stop();

    while(Serial.available()) {
       int num = Serial.parseInt(); // exhaust buffer
    }

    // tell raspberry to take a picture
    Serial.write("0");

    // wait for response from raspberry
    while(! Serial.available()) {
      delay(100);
    }

    //get result
    int neural_output = Serial.read();

    if(neural_output >= 48 && neural_output <= 57) { // numerals 0-9
      neural_output -= 48;
    }
    else if (neural_output == 97) { // letter 'a'
      neural_output = 10;
    }
    else { // letter 'b'
      neural_output = 11;
    }

    sort_object_by_shape_color(neural_output);
    /*switch (rand) {
      case 0:
        Backward();
        TurnLeft();
        change_direction(direction, 0);
        break;

      case 1:
        Backward();
        TurnRight();
        change_direction(direction, 1);
        break;
    }*/
  }
  else
  {
    SlowForward();
    delay(200);
  }
}


// if turn = 0, bot made left turn
// if turn = 1, bot made right turn
void change_direction(int & current_direction, int turn) {
  if (turn == 0) { // bot made left turn
    if(current_direction == 0) { // facing north
      current_direction = 3; // switch west
    }
    else if(current_direction == 3) { // facing west
      current_direction = 2; // switch south
    }
    else if(current_direction == 2) { // facing south
      current_direction = 1; // switch east
    }
    else if (current_direction == 1) { // facing east
      current_direction = 0; // switch north
    }
  }
  else { // bot made right turn
    if(current_direction == 0) { // facing north
      current_direction = 1; // turn east
    }
    else if(current_direction == 1) { // facing east
      current_direction = 2; // turn south
    }
    else if(current_direction == 2) { // facing south
      current_direction = 3; // switch west
    }
    else if (current_direction == 3) { // facing west
      current_direction = 0; // turn north
    }
  }
}


void gatherData(){
  //Forward();

  /*lineFollowing();
  randomSeed(analogRead(6));
  int rand = random(2);

  int distance = sonic_sensor.distanceCm();
  //Serial.println(distance);

  if (distance < 20) {
    //take input from remote and send to pi

    // stop the mbot
    Stop();*/

      // read the remote controller
      int button = 0;
      while(button == 0) {
        button = ir_remote.getCode();
      }
      //Serial.println(button);

      //send data to pi
      switch(button) {
        // Good Luck understanding this..
         case 22: Serial.write("0"); break;
         case 12: Serial.write("1"); break;
         case 24: Serial.write("2"); break;
         case 94: Serial.write("3"); break;
         case 8: Serial.write("4");  break;
         case 28: Serial.write("5"); break;
         case 90: Serial.write("6"); break;
         case 66: Serial.write("7"); break;
         case 82: Serial.write("8"); break;
         case 74: Serial.write("9"); break;
         case 69: Serial.write("a"); break;
         case 70: Serial.write("b"); break;
         case 71: Serial.write("c"); break;
         // you can try the lottery with these numbers..
         // the cases are the values the arduino recieves when the respective ir remote buttons are pressed
      }

      while(Serial.available()) {
        int num = Serial.parseInt(); // exhaust buffer
      }

      // wait for pi to send a message that it's ready!
      while(!Serial.available()) {
        delay(50);
      }

      /*  Backward();

        switch (rand) {
          case 0:
            Backward();
            TurnLeft();
            break;

          case 1:
            Backward();
            TurnRight();
            break;
        }*/

  //}
}

void lineFollowing()
{
  int reading = line.readSensors();

  switch (reading) {
    case S1_IN_S2_IN:
      Forward();
      //LineFollowFlag=10;
      break;

    case S1_IN_S2_OUT:
      Backward();
      TurnRight();
      change_direction(direction, 1);
      break;

    case S1_OUT_S2_IN:
      Backward();
      TurnLeft();
      change_direction(direction, 0);
      break;

    case S1_OUT_S2_OUT:
      Backward();
      TurnLeft();
      change_direction(direction, 0);
      TurnLeft();
      change_direction(direction, 0);
      break;
  }
}


// current facing direction
// placement - where the object has to go

void move_object(int current_direction, int placement) {
  Stop();

  int distance = sonic_sensor.distanceCm();

  while (distance > 7) {
    SlowForward();
    distance = sonic_sensor.distanceCm();
  }

  int turns_needed = abs(current_direction - placement);

  for(int i = 0; i < turns_needed; ++i) {
    if(current_direction > placement) {
      TurnLeft();
      change_direction(direction, 0);
    }

    if(current_direction < placement) {
      TurnRight();
      change_direction(direction, 1);
    }
  }

  int reading = line.readSensors();

  while(reading == S1_IN_S2_IN) {
    //Forward();
    lineFollowing();
    reading = line.readSensors();
  }

  Backward();
  TurnLeft();
  change_direction(direction, 0);
  TurnLeft();
  change_direction(direction, 0);

  // end of moving object, gets execution back to main function
}
void setLED(int color) {
  for(int index = 0; index < 4; ++index) {
    switch(color) {
      case BLUE:
        led.setColorAt(index, 0, 0, 255);
        break;

      case GREEN:
        led.setColorAt(index, 0, 255, 0);
        break;

      case RED:
        led.setColorAt(index, 255, 0, 0);
        break;

      case YELLOW:
        led.setColorAt(index, 255, 255, 0);
        break;
    }
  }

  led.show();
  delay(500);
}

void sort_object_by_shape_color(int neural_output) {

  switch(neural_output) {

    case 0: // blue cube
      setLED(BLUE);
      buzzer.tone(3000, 200);
      move_object(direction, blue_cube_placement);
      break;

    case 1: // blue pyramid
      setLED(BLUE);
      buzzer.tone(3000, 200);
      delay(300);
      buzzer.tone(3000, 200);
      move_object(direction, blue_cube_placement);
      break;

    case 2: // blue sphere
      setLED(BLUE);
      buzzer.tone(3000, 200);
      delay(300);
      buzzer.tone(3000, 200);
      delay(300);
      buzzer.tone(3000, 200);
      move_object(direction, blue_cube_placement);
      break;

    case 3: // green cube
      setLED(GREEN);
      buzzer.tone(3000, 200);
      move_object(direction, green_pyramid_placement);
      break;

    case 4: // green pyramid
      setLED(GREEN);
      buzzer.tone(3000, 200);
      delay(300);
      buzzer.tone(3000, 200);
      move_object(direction, green_pyramid_placement);
      break;

    case 5: // green sphere
      setLED(GREEN);
      buzzer.tone(3000, 200);
      delay(300);
      buzzer.tone(3000, 200);
      delay(300);
      buzzer.tone(3000, 200);
      move_object(direction, green_pyramid_placement);
      break;

    case 6: // red cube
      setLED(RED);
      buzzer.tone(3000, 200);
      move_object(direction, red_sphere_placement);
      break;

    case 7: // red pyramid
      setLED(RED);
      buzzer.tone(3000, 200);
      delay(300);
      buzzer.tone(3000, 200);
      move_object(direction, red_sphere_placement);
      break;

    case 8: // red sphere
      setLED(RED);
      buzzer.tone(3000, 200);
      delay(300);
      buzzer.tone(3000, 200);
      delay(300);
      buzzer.tone(3000, 200);
      move_object(direction, red_sphere_placement);
      break;

    case 9: // yellow cube
      setLED(YELLOW);
      buzzer.tone(3000, 200);
      move_object(direction, yellow_cube_placement);
      break;

    case 10: // yellow pyramid case 'a'
      setLED(YELLOW);
      buzzer.tone(3000, 200);
      delay(300);
      buzzer.tone(3000, 200);
      move_object(direction, yellow_cube_placement);
      break;

    case 11: // yellow sphere case 'b'
      setLED(YELLOW);
      buzzer.tone(3000, 200);
      delay(300);
      buzzer.tone(3000, 200);
      delay(300);
      buzzer.tone(3000, 200);
      move_object(direction, yellow_cube_placement);
      break;

    default:
      buzzer.tone(5000, 500);
      move_object(direction, blue_cube_placement);
  }

}
