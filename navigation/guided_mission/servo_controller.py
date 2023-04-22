import os
if os.getenv("MOCK_PAYLOAD") is not None:
    class ServoController:
        def __init__(self):
            pass
        def openServo(self, servoIndex : int):
            print("Opening servo {}".format(servoIndex)) 
        def closeServo(self, servoIndex : int):
            print("Closing servo {}".format(servoIndex)) 
        def openAllServos(self):
            print("Opening all servos") 
        def closeAllServos(self):
            print("Closing all servos") 
else:
    print("Initializing servo controller...")
    from adafruit_servokit import ServoKit

    # max range of servo in degrees
    servoMax = 26

    #We're using a 16 channel servo controller
    kit = ServoKit(channels=16)

    #Initializes the servo controller
    def setupController():
        for i in range(16):
            kit.servo[i].actuation_range = servoMax
            kit.servo[0].set_pulse_width_range(1000, 2000)

    #Open specified servo
    def openServo(servoIndex : int):
        kit.servo[servoIndex].angle = 0

    #Close specified servo
    def closeServo(servoIndex : int):
        kit.servo[servoIndex].angle = servoMax

    #Open all servos
    def openAllServos():
        for i in range(5):
            openServo(i)

    #Close all servos
    def closeAllServos():
        for i in range(5):
            closeServo(i)

    class ServoController:
        def __init__(self):
            setupController()
        def openServo(self, servoIndex : int):
            openServo(servoIndex)
        def closeServo(self, servoIndex : int):
            closeServo(servoIndex)
        def openAllServos(self):
            openAllServos()
        def closeAllServos(self):
            closeAllServos()
    print("Servo controller ready")

if __name__=="__main__":
    servoController = ServoController()
    while 1:
        user_input = input("Enter o to open all servos, c to close all, and c0-c4 or o0-o4 to open/close a specific servo, or q to quit\n> ")
        if user_input=='q':
            break
        elif len(user_input)==1:
            if user_input=='o':
                openAllServos()
            else:
                closeAllServos()
        else:
            idx = int(user_input[1])
            if user_input[0]=='o':
                openServo(idx)
            else:
                closeServo(idx)
        
    