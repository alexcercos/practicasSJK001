import GUI
import HAL
import cv2
import utm
import math

import numpy as np

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

faces_detected = []
potential_locations = []
debug_colors = [(0, 0, 255),(0, 255, 255),(0, 255, 0), (200, 200, 50),(200, 50, 200),(50, 200, 200),(50, 50, 200),(200, 50, 50),(50, 200, 50),(200, 200, 200),(0,0,50)]

# Hay muchos ajustes a mano, especialmente en temas de coordenadas de pixeles 
# en imagen y coordenadas reales (no esta claro como obtener cierta informacion)

def update_potential_location(lx, ly, weight, pixel_count):

    #iterate previous locations
    for i,location in enumerate(potential_locations):
        cx,cy = location[0]
        d = ((cx-lx)**2 + (cy-ly)**2)**0.5

        if d<2.2: #update location
            location[3] = max(pixel_count, location[3]) #normalization
            if location[3]>0:
                weight *= (pixel_count / location[3])
            cx = cx * (1.0-weight) + lx*weight
            cy = cy * (1.0-weight) + ly*weight
            location[0] = (cx,cy)
            location[2] = (weight*0.2+location[2]*0.8) #to favour more centered regions when choosing target
            
            if location[1]:
                return (255,0,0) #blue for visited
            return debug_colors[min(i,len(debug_colors))]
    
    # append new (unchecked) location
    potential_locations.append([(lx,ly), False, weight, 0])
    return debug_colors[min(len(potential_locations),len(debug_colors))-1]

def get_next_location():
    track_loc = None
    best_weight = 0
    for location in potential_locations:
        if location[1]:
            continue
        
        if location[2]>best_weight:
            best_weight = location[2]
            track_loc = location
    
    if track_loc is None:
        return (0,0)

    track_loc[1] = True #mark as seen
    return track_loc[0]

def relative_position_pixel(pixel_x, pixel_y, drone_height, img_width, img_height):
    # Constants
    fov_horizontal = 30  # Degrees
    #drone_height = 10  # Example height in meters
    #img_width = 640  # Image width in pixels
    #img_height = 480  # Image height in pixels

    # Point in the image (x, y) (example point)
    #pixel_x = 320  # X-coordinate in the image (center of the image)
    #pixel_y = 240  # Y-coordinate in the image (center of the image)

    # Calculate angular resolution per pixel
    angle_per_pixel_horizontal = fov_horizontal / img_width

    # Calculate pixel offset from the image center
    x_offset = pixel_x - (img_width / 2)
    y_offset = pixel_y - (img_height / 2)

    # Calculate angular offsets
    theta_x = math.radians(x_offset * angle_per_pixel_horizontal)
    theta_y = math.radians(y_offset * angle_per_pixel_horizontal)

    # Calculate real-world distances relative to the drone
    # -> coordinates fix: in image they are oriented to be
    # +real x = -img y
    # -real y = +img x
    real_y = - drone_height * math.tan(theta_x) * 2
    real_x = - drone_height * math.tan(theta_y) * 2

    return real_x, real_y
    # print(f"Real-world position relative to the drone: X={real_x:.2f}, Y={real_y:.2f}")

def rotateImage(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, scale)
    
    # Calculate the cosine and sine of the angle to determine the new bounding dimensions
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])
    
    # Compute new width and height of the rotated image
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)
    
    # Adjust the rotation matrix to account for the translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # cyan
    most_common_color = (255, 150, 0)

    # Create a canvas with the most common color
    canvas = np.full((new_h, new_w, 3), most_common_color, dtype=np.uint8)
    
    # Perform the rotation with the new dimensions
    rotated = cv2.warpAffine(image, M, (new_w, new_h))
    
    # Overlay the rotated image on the canvas
    mask = cv2.warpAffine(np.ones_like(image, dtype=np.uint8), M, (new_w, new_h))
    canvas[mask > 0] = rotated[mask > 0]

    return canvas

def goToLocation(pos_x, pos_y, height):
    """
    Moves the drone to a specific location (pos_x, pos_y, height).
    Continuously adjusts position until close enough to the target.
    Images are displayed while moving, and progress information is printed.
    """
    tolerance = 0.5  # Acceptable distance in meters to consider the drone at the target

    print(f"Going to: x={pos_x:.2f}, y={pos_y:.2f}, z={height:.2f}")

    while True:

        # Get the drone's current position
        drone_coord = HAL.get_position()
        drone_x, drone_y, drone_z = drone_coord[0], drone_coord[1], drone_coord[2]

        # Calculate position differences
        error_x = pos_x - drone_x
        error_y = pos_y - drone_y
        error_z = height - drone_z

        # Check if the drone is within tolerance of the target position
        if abs(error_x) < tolerance and abs(error_y) < tolerance and abs(error_z) < tolerance:
            print(f"Target reached successfully!: x={drone_x:.2f}, y={drone_y:.2f}, z={drone_z:.2f}")
            break

        # Command the drone to move towards the target
        HAL.set_cmd_pos(pos_x, pos_y, height, 0)

        # Display the images during movement
        frontal_img = HAL.get_frontal_image()
        ventral_img = HAL.get_ventral_image()
        #GUI.showImage(frontal_img)
        GUI.showLeftImage(ventral_img)

def detectBlackRegions():

    black_threshold = 50  # Threshold to define black regions (intensity <= 50)

    # Capture the ventral image
    ventral_img = HAL.get_ventral_image()
    GUI.showLeftImage(ventral_img)
    gray = cv2.cvtColor(ventral_img, cv2.COLOR_BGR2GRAY)

    # Threshold the image to detect black regions
    _, threshold_img = cv2.threshold(gray, black_threshold, 255, cv2.THRESH_BINARY_INV)

    # Find contours of the black regions
    contours, _ = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No black regions detected.")
        GUI.showImage(ventral_img)
        return

    img_height, img_width = ventral_img.shape[:2]

    detected_contours = []

    drone_coord = HAL.get_position()

    for contour in contours:
        
        boundary_touched = False  # Flag for boundary-touching regions

        # Check if the region touches the boundary
        for point in contour:
            x, y = point[0]
            if x == 0 or y == 0 or x == img_width - 1 or y == img_height - 1:
                boundary_touched = True
                break

        # Calculate the center of the region
        M = cv2.moments(contour)
        zclr = (0,0,0)
        if M["m00"] != 0:  # Avoid division by zero
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # Draw a blue circle at the center of the region

            rx,ry = relative_position_pixel(cX, cY, drone_coord[2], ventral_img.shape[1], ventral_img.shape[0])

            zclr = update_potential_location(drone_coord[0]+rx,drone_coord[1]+ry, 0.0 if boundary_touched else 0.2, len(contour))

        cv2.drawContours(ventral_img, [contour], -1, zclr, -1)

    print("; ".join([f"({l[0][0]:.2f}, {l[0][1]:.2f} -> {l[1]})" for l in potential_locations]))

    # Show the updated image
    GUI.showImage(ventral_img)

def detectFaces():

    total_faces = 0
    shown_img = False

    for angle in range(-180, 180):
        # Get ventral image and rotate
        ventral_img = rotateImage(HAL.get_ventral_image(), angle)
        gray = cv2.cvtColor(ventral_img, cv2.COLOR_BGR2GRAY)

        # Perform face detection
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
        )

        if (angle%4==0 and len(faces)>0) or (angle%30==0):
            for x, y, w, h in faces:
                cv2.rectangle(ventral_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            GUI.showImage(ventral_img)

        total_faces += 1 if len(faces)>0 else 0

    return total_faces #amount of angles with face

##

# Enter sequential code!

boat_lat = 40 + 16 / 60 + 48.2 / 3600
boat_lon = -3 + 49 / 60 + 3.5 / 3600

survivors_lat = 40 + 16 / 60 + 47.23 / 3600
survivors_lon = -3 + 49 / 60 + 1.78 / 3600

boat_coord = utm.from_latlon(boat_lat, boat_lon)
surv_coord = utm.from_latlon(survivors_lat, survivors_lon)

diff_x = -(surv_coord[0] - boat_coord[0]) # el sucio ajuste
diff_y = surv_coord[1] - boat_coord[1]


if HAL.get_landed_state() == 3:
    print("RESET THE IMAGE")
    # If this happens, errors occur (should not be in the air yet)

HAL.takeoff(3)
print("Takeoff finished")

# potential locations: exp average (if red more weight), with 1m radius tolerance, checked bool?
# verified locations: detected face (calculate coords once)
target_x=diff_x
target_y=diff_y

# go to first location
goToLocation(target_x, target_y, 3)
goToLocation(target_x, target_y, 20)

while True:
    for i in range(50):
        detectBlackRegions()

    target_x,target_y = get_next_location()
    goToLocation(target_x, target_y, 3)
    
    
    if target_x==0 and target_y==0:
        HAL.land()
        print("PEOPLE COORDINATES: ",faces_detected)
        break
        #finished (returned to base)
    else:
        positives = 0
        for i in range(5):
            amount = detectFaces()
            print("Detected:",amount)
            if amount>10: #instead of >0 for noise
                positives+=1
        
        if positives>=3:
            print("Added face")
            faces_detected.append((target_x,target_y))
        else:
            print("No face confirmed")

    goToLocation(target_x, target_y, 20)
