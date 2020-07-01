from djitellopy import Tello
import cv2
import numpy as np
import time
import datetime
import os

# Speed of the drone ( for modifications, change here )
S = 30  # fast speed
Sslow = 15  # low speed

UDOffset = 150 # offset distance under the face
safetyX = 100  # offset on the x axis . Range 0-480
safetyY = 55  # offset on the y axis . Range 0-360

# this is the bound box sizes that openCV returns
faceSizes = [508, 342, 228, 152, 101, 68, 45]
targetDistance = 3

# when True, it allows to see the face detection on the screen without taking off
TEST_VID = False  # DO NOT TAKE-OFF WHEN THIS IS TRUE

# Frames per second of the window display
FPS = 25
dimensions = (960, 720)

# choose a cascade from the 'cascades' folder
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

class ConssistCheck(object):
    def __init__(self, size):
        self.array = []
        self.maxSize = size

    def add(self, m):
        self.array.append(m)
        if len(self.array) > self.maxSize:
            self.array.pop(0)

    def isStable(self, percent=0.5):
        count = 0
        if len(self.array) < self.maxSize:
            return 0

        for res in self.array:
            if res is True:
                count += 1

        if count / len(self.array) < percent:
            ans = 0
        else:
            ans = 1

        return ans


scanCheck = ConssistCheck(4)
stableCheck = ConssistCheck(10)


class FrontEnd(object):

    def __init__(self):
        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()
        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10
        self.countFace = 0
        self.send_rc_control = False
        self.finishscan = False
        # self.stable = False
        self.stabling = False
        self.inAction = False
        self.imgTimer = 0
        self.stopTimer = 0
        self.timerOn = False

    def run(self):

        if not self.tello.connect():
            print("Tello not connected")
            return

        if not self.tello.set_speed(self.speed):
            print("Not set speed to lowest possible")
            return

        # In case streaming is on. This happens when we quit this program without the escape key.
        if not self.tello.streamoff():
            print("Could not stop video stream")
            return

        if not self.tello.streamon():
            print("Could not start video stream")
            return

        frame_read = self.tello.get_frame_read()
        start = 0
        should_stop = False
        secondBox = 30
        tDistance = targetDistance
        self.tello.get_battery()
        resetCount = 0
        startDeliver = False
        deliverAction = 1
        # Safety Zone X
        szX = safetyX
        # Safety Zone Y
        szY = safetyY

        while not should_stop:
            self.update()
            resetCount += 1
            self.imgTimer += 1

            if self.timerOn:
                if self.imgTimer > self.stopTimer:
                    self.timerOn = False
                    self.inAction = False

            if frame_read.stopped:
                frame_read.stop()
                break

            frame = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)
            frameRet = frame_read.frame

            time.sleep(1 / FPS)

            # Listen for key presses
            k = cv2.waitKey(20)

            # Set the goal distance from the person 0-6 : [1026, 684, 456, 304, 202, 136, 90]
            if k == ord('0'):
                print("Distance = 0")
                tDistance = 0

            if k == ord('1'):
                print("Distance = 1")
                tDistance = 1

            if k == ord('2'):
                print("Distance = 2")
                tDistance = 2

            if k == ord('3'):
                print("Distance = 3")
                tDistance = 3

            if k == ord('4'):
                print("Distance = 4")
                tDistance = 4

            if k == ord('5'):
                print("Distance = 5")
                tDistance = 5

            if k == ord('6'):
                print("Distance = 6")
                tDistance = 6

            if k == ord('t') and not TEST_VID:  # T to take off
                start = datetime.datetime.now()
                self.finishscan = False
                print("Taking Off")
                self.tello.takeoff()
                time.sleep(1)
                self.send_rc_control = True
                self.up_down_velocity = 50
                self.setTimer(0.9)
                self.tello.get_battery()

            if k == ord('l'):  # L to land
                print("Landing")
                self.tello.land()
                self.send_rc_control = False

            if k == 27:  # Quit the software
                should_stop = True
                break

            if startDeliver and not self.inAction:
                if deliverAction == 1:
                    self.for_back_velocity = 18
                    self.up_down_velocity = 35
                    self.setTimer(1)
                elif deliverAction == 2:
                    self.for_back_velocity = 0
                    self.up_down_velocity = 0
                    self.setTimer(1)
                else:
                    self.tello.land()
                    self.setTimer(2)
                deliverAction += 1

            gray = cv2.cvtColor(frameRet, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=2)

            # Target size
            tSize = faceSizes[tDistance]

            # These are our center dimensions
            cWidth = int(dimensions[0] / 2)
            cHeight = int(dimensions[1] / 2)

            noFaces = len(faces) == 0
            # memo.add(noFaces)
            scanCheck.add(not noFaces)

            if scanCheck.isStable(1):
                self.finishscan = True

            # if we've given rc controls & get face coords returned
            if self.send_rc_control or TEST_VID:
                for (x, y, w, h) in faces:

                    roi_gray = gray[y:y + h, x:x + w]  # (ycord_start, ycord_end)
                    roi_color = frameRet[y:y + h, x:x + w]
                    # setting Face Box properties
                    fbCol = (255, 0, 0)  # BGR 0-255
                    fbStroke = 2
                    # end coords are the end of the bounding box x & y
                    end_cord_x = x + w
                    end_cord_y = y + h
                    faceWidth = w
                    end_size = w * 2

                    # these are our target coordinates
                    targ_cord_x = int((end_cord_x + x) / 2)
                    targ_cord_y = int((end_cord_y + y) / 2) + UDOffset

                    # This calculates the vector from your face to the center of the screen
                    vTrue = np.array((cWidth, cHeight, tSize))
                    vTarget = np.array((targ_cord_x, targ_cord_y, faceWidth))
                    vDistance = vTrue - vTarget

                    if (abs(vDistance[0]) < safetyX + 10) and (abs(vDistance[1]) < safetyY + 10) and (
                            abs(vDistance[2]) < 1):
                        self.stabling = True
                    else:
                        self.stabling = False

                    if not self.inAction and self.finishscan:
                        # for turning
                        if vDistance[0] < -szX:
                            if -szX - vDistance[0] > secondBox:
                                self.yaw_velocity = S
                            else:
                                self.yaw_velocity = Sslow
                        elif vDistance[0] > szX:
                            if vDistance[0] - szX > secondBox:
                                self.yaw_velocity = -S
                            else:
                                self.yaw_velocity = -Sslow
                        else:
                            self.yaw_velocity = 0

                        # for up & down
                        if vDistance[1] > szY:
                            if vDistance[1] - szY > secondBox:
                                self.up_down_velocity = S
                            else:
                                self.up_down_velocity = Sslow
                        elif vDistance[1] < -szY:
                            if -szY - vDistance[1] > secondBox:
                                self.up_down_velocity = -S
                            else:
                                self.up_down_velocity = -Sslow
                        else:
                            self.up_down_velocity = 0


                        # for forward back
                        if vDistance[2] > 0:
                            self.for_back_velocity = Sslow
                        elif vDistance[2] < 0:
                            self.for_back_velocity = -S
                        else:
                            self.for_back_velocity = 0

                    # Draw the face bounding box
                    cv2.rectangle(frameRet, (x, y), (end_cord_x, end_cord_y), fbCol, fbStroke)

                    # Draw the target as a circle
                    cv2.circle(frameRet, (targ_cord_x, targ_cord_y), 10, (0, 255, 0), 2)

                    # Draw the safety zone
                    cv2.rectangle(frameRet, (targ_cord_x - szX, targ_cord_y - szY),
                                  (targ_cord_x + szX, targ_cord_y + szY), (0, 255, 0), fbStroke)

                    # Draw the estimated drone vector position in relation to face bounding box
                    cv2.putText(frameRet, str(vDistance), (0, 64), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # if there are no faces detected, don't do anything
                if noFaces and not self.finishscan and not self.inAction and not TEST_VID:
                    end = datetime.datetime.now()
                    elapsed = end - start
                    if elapsed.seconds > 50:
                        self.tello.land()
                        time.sleep(5)
                        should_stop = True
                    self.yaw_velocity = 25
                    self.up_down_velocity = 0
                    self.for_back_velocity = 0
                    print("SCANNING")
                    # cv2.putText(frameRet, "Scanning", (400, 664), cv2.FONT_HERSHEY_SIMPLEX, 1, dCol,
                    #             2)  # Draw the distance choosen

                if noFaces and self.finishscan and not self.inAction:
                    self.yaw_velocity = 0
                    self.up_down_velocity = 0
                    self.for_back_velocity = 0
                    print("NO TARGET")

                if self.finishscan:
                    if not noFaces and self.stabling:
                        stableCheck.add(True)
                    else:
                        stableCheck.add(False)

                res = stableCheck.isStable(0.5)  # returns 1 or 0 if stable or not
                self.countFace = self.countFace + res
                if res > 0:
                    resetCount = 0
                print(self.countFace)
                if resetCount > 15:
                    self.countFace = 0
                # if 20 < self.countFace < 45:
                #     cv2.putText(frameRet, "Stabilizing", (400, 664), cv2.FONT_HERSHEY_SIMPLEX, 1, dCol,
                #                 2)

                if self.countFace > 45:
                    print("STABLE")
                    startDeliver = True

            cv2.circle(frameRet, (cWidth, cHeight), 10, (0, 0, 255),
                       2)  # Draw the center of screen circle, this is what the drone tries to match with the target coords
            dCol = lerp(np.array((0, 0, 255)), np.array((255, 255, 255)), tDistance + 1 / 7)
            show = "Distance: {}".format(str(tDistance))
            cv2.putText(frameRet, show, (32, 664), cv2.FONT_HERSHEY_SIMPLEX, 1, dCol, 2)  # Draw the distance choosen

            # if startDeliver:
            #     cv2.putText(frameRet, "Delivering Package", (400, 664), cv2.FONT_HERSHEY_SIMPLEX, 1, dCol,
            #                 2)  # Draw the distance choosen

            cv2.imshow(f'AsafBarakTello', frameRet)  # Display the resulting frame

        # On exit, print the battery
        self.tello.get_battery()

        # When everything done, release the capture
        cv2.destroyAllWindows()

        # Call it always before finishing. I deallocate resources.
        self.tello.end()

    def battery(self):
        return self.tello.get_battery()[:2]

    def update(self):
        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity,
                                       self.yaw_velocity)

    def setTimer(self, seconds=0.0): # set time in which the drone will not get commands to move
        # we don't use time.sleep because it freezes the camera feed
        self.timerOn = True
        self.imgTimer = 0
        self.inAction = True
        self.stopTimer = seconds * FPS


def lerp(a, b, c):
    return a + c * (b - a)


def main():
    frontend = FrontEnd()

    # run frontend
    frontend.run()


if __name__ == '__main__':
    main()
