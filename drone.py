import GUI
import HAL

import utm
# Enter sequential code!

boat_lat = 40 + 16/60 + 48.2/3600
boat_lon = -3 + 49/60 + 3.5/3600

survivors_lat = 40 + 16/60 + 47.23/3600
survivors_lon = -3 + 49/60 + 1.78/3600

boat_coord = utm.from_latlon(boat_lat, boat_lon)
surv_coord = utm.from_latlon(survivors_lat, survivors_lon)

diff_x = -(surv_coord[0] - boat_coord[0])
diff_y = surv_coord[1] - boat_coord[1] # el sucio ajuste

HAL.takeoff(3)

while True:

    drone_coord = HAL.get_position()

    frontal_img = HAL.get_frontal_image()
    ventral_img = HAL.get_ventral_image()

    GUI.showImage(frontal_img)
    GUI.showLeftImage(ventral_img)

    HAL.set_cmd_pos(diff_x, diff_y, 3, 0)

    print(diff_x,diff_y,drone_coord)

    # HAL.get_velocity()
    # HAL.get_yaw_rate()
    # HAL.get_orientation()
    # HAL.get_roll()
    # HAL.get_pitch()
    # HAL.get_yaw()
    # HAL.get_landed_state()
