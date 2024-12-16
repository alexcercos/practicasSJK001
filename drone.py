import GUI
import HAL
import cv2
import utm

import numpy as np

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

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
            print("Target reached successfully!")
            print(f"Final position: x={drone_x:.2f}, y={drone_y:.2f}, z={drone_z:.2f}")
            break

        # Command the drone to move towards the target
        HAL.set_cmd_pos(pos_x, pos_y, height, 0)

        # Display the images during movement
        frontal_img = HAL.get_frontal_image()
        ventral_img = HAL.get_ventral_image()
        GUI.showImage(frontal_img)
        GUI.showLeftImage(ventral_img)

        # Print progress information
        print(f"Current position: x={drone_x:.2f}, y={drone_y:.2f}, z={drone_z:.2f}")

def centerOnBlackRegion():
    """
    Detects black regions in the ventral image, differentiates them, and centers
    the drone's camera on one of them (largest region).
    """
    black_threshold = 50  # Threshold to define black regions (intensity <= 50)
    movement_step = 0.5   # Step size for drone adjustments
    tolerance = 20        # Tolerance in pixels for centering

    while True:

        # Capture the ventral image
        ventral_img = HAL.get_ventral_image()
        gray = cv2.cvtColor(ventral_img, cv2.COLOR_BGR2GRAY)

        # Threshold the image to detect black regions
        _, threshold_img = cv2.threshold(gray, black_threshold, 255, cv2.THRESH_BINARY_INV)

        # Find contours of the black regions
        contours, _ = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("No black regions detected.")
            GUI.showLeftImage(ventral_img)
            return

        # Find the largest contour (assuming it's the target region)
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)

        if M["m00"] == 0:  # Avoid division by zero
            print("Unable to calculate center of the black region.")
            return

        # Calculate the center of the black region
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # Draw contours and center for visualization
        cv2.drawContours(ventral_img, [largest_contour], -1, (0, 255, 0), 2)
        cv2.circle(ventral_img, (cX, cY), 5, (255, 0, 0), -1)

        # Get image dimensions and calculate offsets from center
        img_center_x = ventral_img.shape[1] // 2
        img_center_y = ventral_img.shape[0] // 2
        offset_x = cX - img_center_x
        offset_y = cY - img_center_y

        print(f"Offset: X={offset_x}, Y={offset_y}")

        # Centering logic: Move the drone to center the black region
        if abs(offset_x) < tolerance and abs(offset_y) < tolerance:
            print("Black region centered.")
            GUI.showLeftImage(ventral_img)
            return

        # Adjust the drone's position based on offsets
        move_x = -movement_step if offset_x > tolerance else (movement_step if offset_x < -tolerance else 0)
        move_y = -movement_step if offset_y > tolerance else (movement_step if offset_y < -tolerance else 0)

        drone_coord = HAL.get_position()
        HAL.set_cmd_pos(drone_coord[0] + move_x, drone_coord[1] + move_y, drone_coord[2], 0)

        # Show the updated image
        GUI.showLeftImage(ventral_img)

def detectFaces():
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
                GUI.showLeftImage(ventral_img)

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


def spiralSearch(start_x, start_y, height, step_size=3, max_loops=10):
    """
    Perform a spiral search starting from a given location.
    
    Parameters:
    - start_x, start_y, height: Starting coordinates for the search.
    - step_size: Distance (in meters) for each movement step.
    - max_loops: Maximum number of loops for the spiral.
    """
    # Move to the starting location
    print(f"Going to start location: ({start_x}, {start_y}, {height})")
    goToLocation(start_x, start_y, height)
    
    # Initialize spiral parameters
    current_x, current_y = start_x, start_y
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # [right, up, left, down]
    loop = 0
    direction_index = 0  # Start with the first direction (right)

    # Start the spiral search
    while loop < max_loops:
        # Check for black regions and center on one if found
        print("Checking for black regions...")
        centerOnBlackRegion()
        
        # If a black region is detected, call a placeholder function
        # Placeholder: replace this with your own logic
        print("Detected a black region. Processing...")
        detectFaces()  # Uncomment and implement this function
        
        # Move in the current direction
        for _ in range(2):  # Perform two moves in each direction before turning
            if loop >= max_loops:
                break

            # Calculate the next position
            dx, dy = directions[direction_index]
            next_x = current_x + dx * step_size
            next_y = current_y + dy * step_size
            
            # Move to the next position
            print(f"Moving to ({next_x}, {next_y}, {height})")
            goToLocation(next_x, next_y, height)
            
            # Update the current position
            current_x, current_y = next_x, next_y

        # Change direction (right -> up -> left -> down -> repeat)
        direction_index = (direction_index + 1) % 4

        # Increase loop count and expand the spiral radius
        loop += 1
        step_size += 1  # Expand the spiral

    print("Spiral search completed.")



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

# spiralSearch(diff_x, diff_y, 3)
# goToLocation(diff_x, diff_y, 3)
# centerOnBlackRegion()

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
