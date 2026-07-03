#Code to test anything on tello
import cv2
from djitellopy import Tello

tello = Tello()
tello.connect()
tello.streamon()

reader = tello.get_frame_read()

while True:

    frame = reader.frame

    if frame is None:
        continue
    print("RAW ToF:", tello.get_distance_tof())
    # RAW
    cv2.imshow("RAW", frame)

    # BGR -> RGB -> DISPLAY AGAIN
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("RGB_CONVERTED", rgb)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

tello.streamoff()
cv2.destroyAllWindows()