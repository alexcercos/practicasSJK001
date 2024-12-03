import GUI
import HAL
import cv2
import utm

import numpy as np


def rotateImage(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def goToLocation(pos_x, pos_y, height):
    pass


# TODO find faces : rotate img in 30 increments
# avoid calling when there is only water

# TODO find others : fly higher, detect points ?? go towards them?

# Enter sequential code!

boat_lat = 40 + 16 / 60 + 48.2 / 3600
boat_lon = -3 + 49 / 60 + 3.5 / 3600

survivors_lat = 40 + 16 / 60 + 47.23 / 3600
survivors_lon = -3 + 49 / 60 + 1.78 / 3600

boat_coord = utm.from_latlon(boat_lat, boat_lon)
surv_coord = utm.from_latlon(survivors_lat, survivors_lon)

diff_x = -(surv_coord[0] - boat_coord[0])
diff_y = surv_coord[1] - boat_coord[1]  # el sucio ajuste

HAL.takeoff(3)

# TODO remove
diff_x = 38
diff_y = -31

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

while True:

    drone_coord = HAL.get_position()

    frontal_img = HAL.get_frontal_image()
    ventral_img = rotateImage(HAL.get_ventral_image(), -30)

    # Convert the image to grayscale
    gray = cv2.cvtColor(ventral_img, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
    )

    # Draw rectangles around the detected faces
    for x, y, w, h in faces:
        cv2.rectangle(ventral_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    GUI.showImage(frontal_img)
    GUI.showLeftImage(ventral_img)

    faces = face_cascade.detectMultiScale(
        ventral_img, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
    )

    HAL.set_cmd_pos(diff_x, diff_y, 3, 0)

    print(diff_x, diff_y, drone_coord)
    print(faces)
