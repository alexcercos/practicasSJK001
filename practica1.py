import GUI
import HAL

import cv2

i = 0
err_past = 0

# kp kd ki max_angle reset_integral integral_decay

# 125.02 P -> 0.01 0.0 0.0 2.0 1000 1.0
# 124.98 PD -> 0.02 0.025 0.0 2.0 1000 1.0
# 125.92 PID -> 0.01 0.02 0.001 2.0 1000 1.0

kp = 0.02
kd = 0.025
ki = 0.0

# Absolute values (-/+)
max_angle = 2.0  # Clamp angle
reset_integral = 1000 #Avoid large integrals
integral_decay = 0.4  # Reduce weight in previous values (exp. average)
clamp_image = 3/4 #Check upper pixels only

integral = 0


def clamp(value, limit):
    return max(min(value, limit), -limit)


while True:

    img = HAL.getImage()
    img = img[:int(len(img)*clamp_image)]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, (0, 125, 125), (30, 255, 255))

    contours, hierarchy = cv2.findContours(
        red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    M = cv2.moments(contours[0])

    if M["m00"] != 0:
        cX = M["m10"] / M["m00"]
        cY = M["m01"] / M["m00"]
    else:
        cX, cY = 0, 0

    dt = 0
    if cX > 0:
        err = 320 - cX
        dt = err - err_past
        err_past = err
        integral = integral * integral_decay + err

        giro = kp * err + kd * dt + ki * integral

        HAL.setV(4)
        HAL.setW(clamp(giro, max_angle))

        if integral > reset_integral or integral < -reset_integral:
          integral = 0

    GUI.showImage(red_mask)
    print(
        "%d cX: %.2f cY: %.2f err: %.2f dt %.2f it %.2f giro=%.2f"
        % (i, cX, cY, err, dt, integral, giro)
    )
    i += 1
