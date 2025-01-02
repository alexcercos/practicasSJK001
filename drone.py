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

# Hay muchos ajustes a mano, especialmente en temas de coordenadas de pixeles en imagen y coordenadas reales (no esta claro como obtener cierta informacion)

def update_potential_location(lx, ly, weight, pixel_count):

    #iterate previous locations
    for i,location in enumerate(potential_locations):
        cx,cy = location[0]
        d = ((cx-lx)**2 + (cy-ly)**2)**0.5

        if d<2.5: #update location
            location[3] = max(pixel_count, location[3]) #normalization
            if location[3]>0:
                weight *= (pixel_count / location[3])
            cx = cx * (1.0-weight) + lx*weight
            cy = cy * (1.0-weight) + ly*weight
            location[0] = (cx,cy)
            location[2] = max(weight,location[2]) #track max weight
            
            if location[1]:
                return (255,0,0) #blue for visited
            return debug_colors[min(i,len(debug_colors))]
    
    # append new (unchecked) location
    potential_locations.append([(lx,ly), False, weight, 0])
    return debug_colors[min(len(potential_locations),len(debug_colors))-1]

def get_next_location():
    track_loc = (0,0)
    for location in potential_locations:
        if not location[1]:
            if location[2]>0.01:
                location[1] = True #mark as seen
                return location[0]

            # track, but not mark (if only seen in image bounds)
            track_loc = location[0] 
    
    return track_loc #Go to base

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

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

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

        # Print progress information
        #print(f"Current position: x={drone_x:.2f}, y={drone_y:.2f}, z={drone_z:.2f}")

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

        if len(contour) < 20: #Remove small regions
            continue
        
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


def detectFaces(): #TODO simplify (avoid centering, use relative pixel coords)
    """
    Detects faces by rotating the ventral image in 30-degree increments.
    If a face is found, it centers the drone on the face, lowers the altitude by 1 meter,
    and verifies if the face is still detected. Returns the drone's current coordinates
    if successful, otherwise returns an empty list.
    """
    
    rotation_angles = [i * 30 for i in range(-6, 7)]  # [-180, -150, ..., 0, ..., 150, 180]
    step_size = 0.5  # Movement step size for centering
    height_reduction = 1  # Meters to descend when a face is detected
    tolerance = 20  # Tolerance in pixels for centering
    
    def centerFace(x, y, w, h, ventral_img):
        """
        Adjusts the drone's position to center the detected face in the ventral image.
        """
        # Calculate the image center
        img_center_x = ventral_img.shape[1] // 2
        img_center_y = ventral_img.shape[0] // 2
        
        # Calculate offsets between face center and image center
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        offset_x = face_center_x - img_center_x
        offset_y = face_center_y - img_center_y

        print(f"Face detected. Offsets: X={offset_x}, Y={offset_y}")
        
        # Adjust position based on offsets
        drone_coord = HAL.get_position()
        move_x = -step_size if offset_x > tolerance else (step_size if offset_x < -tolerance else 0)
        move_y = -step_size if offset_y > tolerance else (step_size if offset_y < -tolerance else 0)

        HAL.set_cmd_pos(drone_coord[0] + move_x, drone_coord[1] + move_y, drone_coord[2], 0)
        print(f"Centering on face... Moving to: ({drone_coord[0] + move_x}, {drone_coord[1] + move_y}, {drone_coord[2]})")

    for angle in rotation_angles:
        # Get ventral image and rotate
        ventral_img = rotateImage(HAL.get_ventral_image(), angle)
        gray = cv2.cvtColor(ventral_img, cv2.COLOR_BGR2GRAY)

        # Perform face detection
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
        )

        # If faces are found, center the drone and descend
        if len(faces) > 0:
            print(f"Face(s) detected at rotation angle {angle} degrees.")
            for x, y, w, h in faces:
                # Visualize the detected face
                cv2.rectangle(ventral_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                GUI.showImage(ventral_img)

                # Center on the face
                centerFace(x, y, w, h, ventral_img)

                # Lower altitude by 1 meter
                drone_coord = HAL.get_position()
                HAL.set_cmd_pos(drone_coord[0], drone_coord[1], drone_coord[2] - height_reduction, 0)
                print("Lowering altitude by 1 meter to confirm face presence...")

                # Confirm face detection at new height
                ventral_img = rotateImage(HAL.get_ventral_image(), angle)
                gray = cv2.cvtColor(ventral_img, cv2.COLOR_BGR2GRAY)
                faces_confirm = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
                )

                if len(faces_confirm) > 0:
                    print("Face still detected after descent. Returning current coordinates.")
                    final_coord = HAL.get_position()
                    return [final_coord[0], final_coord[1], final_coord[2]]
                else:
                    print("Face no longer detected after descent.")

    print("No faces detected in any rotation. Returning an empty list.")
    return []


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


# TODO go to initial location
#go to 20m height
#detect black regions, center on closest unseen (estimate position)
#lower to 3m, detect faces (check all rotations)
#if detected, add to list
#go to 20m height, repeat

# potential locations: exp average (if red more weight), with 1m radius tolerance, checked bool?
# verified locations: detected face (calculate coords once)
target_x=diff_x
target_y=diff_y

# go to first location
goToLocation(target_x, target_y, 3)
goToLocation(target_x, target_y, 20)

for i in range(50):
    detectBlackRegions()

while True:
    
    target_x,target_y = get_next_location()
    goToLocation(target_x, target_y, 3)
    
    
    if target_x==0 and target_y==0:
        HAL.land()
        break #finished (returned to base)
    else:
        for i in range(5): #TODO remove
            detectBlackRegions()
        #TODO detect faces

    goToLocation(target_x, target_y, 20)
    for i in range(50):
        detectBlackRegions()

print("Faces detected: ",faces_detected)
