import GUI
import HAL

import cv2

# Enter sequential code!
i=0
err_past = 0


kp = 0.01
kd = 0.02
ki = 0.000001

integral = 0

while True:
    
    img = HAL.getImage()
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv,
                            (0,125,125),
                            (30,255,255))
                            
    contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    M = cv2.moments(contours[0])
    
    if M["m00"] != 0:
      cX = M["m10"] / M["m00"]
      cY = M["m01"] / M["m00"]
    else:
      cX, cY = 0,0
    
    dt = 0
    if cX>0:
      err = 320 - cX
      dt = err - err_past
      err_past = err
      integral+=err
      
      HAL.setV(4)
      HAL.setW(kp*err + kd * dt +ki * integral)
      
      
      
    GUI.showImage(red_mask)
    print("%d cX: %.2f cY: %.2f err: %.2f dt %.2f it %.2f" % (i,cX,cY, err, dt, integral))
    i += 1
