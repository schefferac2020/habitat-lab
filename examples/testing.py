import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import cv2
import numpy as np


FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"

image = np.ones((512, 512, 3))

fovia_size = 30 # pixels
fovial_layers = 5
fovia_layer_sizes = fovia_size*(2**np.arange(fovial_layers))

def crop_around_pixel(img, center_x, center_y, crop_width, crop_height):
    img_height, img_width = img.shape[:2]

    half_width = crop_width // 2
    half_height = crop_height // 2

    start_x = max(center_x - half_width, 0)
    start_y = max(center_y - half_height, 0)
    end_x = min(center_x + half_width, img_width)
    end_y = min(center_y + half_height, img_height)

    cropped_image = np.zeros((crop_height, crop_width, img.shape[2]), dtype=np.uint8)

    crop_start_x = max(half_width - center_x, 0)
    crop_start_y = max(half_height - center_y, 0)
    crop_end_x = crop_start_x + (end_x - start_x)
    crop_end_y = crop_start_y + (end_y - start_y)

    cropped_image[crop_start_y:crop_end_y, crop_start_x:crop_end_x] = img[start_y:end_y, start_x:end_x]

    return cropped_image

def center_image(background, overlay):
    # Get dimensions of both images
    bg_height, bg_width = background.shape[:2]
    ol_height, ol_width = overlay.shape[:2]

    # Compute the starting coordinates for centering
    start_x = (bg_width - ol_width) // 2
    start_y = (bg_height - ol_height) // 2

    # Create a copy of the background to place the overlay
    centered_image = background.copy()

    # Determine the region to place the overlay, ensuring it fits within the background
    end_x = min(start_x + ol_width, bg_width)
    end_y = min(start_y + ol_height, bg_height)
    ol_end_x = end_x - start_x
    ol_end_y = end_y - start_y

    # Copy the overlay image into the center of the background
    centered_image[start_y:end_y, start_x:end_x] = overlay[:ol_end_y, :ol_end_x]

    return centered_image

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        # Get the pixel value at the (x, y) position
        fovial_stack_arr = create_fovial_stack(x, y)

        fovial_img_arr = make_fovial_img(fovial_stack_arr)

        print(f"Cursor at {x}, {y}")

        cv2.imshow("FOVIAL IMAGE", fovial_img_arr)

def create_fovial_stack(x, y):
    global image
    fovial_stack = np.zeros((fovia_size, fovia_size, fovial_layers*3), dtype=np.uint8) # 3 each for rgb


    half_fovia = int(fovia_size/2)

    x = int(np.clip(x, half_fovia, image.shape[1] - half_fovia))
    y = int(np.clip(y, half_fovia, image.shape[0] - half_fovia))

    # cropped_img = crop_around_pixel(image, x, y, 100, 100)

    for i, crop_width in enumerate(fovia_layer_sizes):
        cropped_img = crop_around_pixel(image, x, y, crop_width, crop_width)

        cropped_img = cv2.resize(cropped_img, (fovia_size, fovia_size), interpolation=cv2.INTER_LINEAR) #TODO: Is this the best way to squash the image down?
        fovial_stack[:, :, 3*i:3*i+3] = cropped_img


    return fovial_stack

def make_fovial_img(fovial_stack):
    fovial_image = np.zeros((fovia_layer_sizes[-1], fovia_layer_sizes[-1], 3), dtype=fovial_stack.dtype)

    for i, crop_width in reversed(list(enumerate(fovia_layer_sizes))):
        downsampled_img = fovial_stack[:, :, 3*i:3*i+3]
        print(i, crop_width)
        downsampled_img = cv2.resize(downsampled_img, (crop_width, crop_width), interpolation=cv2.INTER_LINEAR)

        fovial_image = center_image(fovial_image, downsampled_img)



    return fovial_image     

def example():
    global image

    env = habitat.Env(
        config=habitat.get_config("benchmark/nav/pointnav/drew_test_big_sensor.yaml")
    )

    print("Environment creation successful")
    observations = env.reset()
    print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
        observations["pointgoal_with_gps_compass"][0],
        observations["pointgoal_with_gps_compass"][1]))

    image = transform_rgb_bgr(observations["rgb"])
    cv2.imshow("RGB", image)
    cv2.setMouseCallback("RGB", on_mouse)

    print("Agent stepping around inside environment.")

    count_steps = 0
    while not env.episode_over:
        keystroke = cv2.waitKey(0)

        if keystroke == ord(FORWARD_KEY):
            action = HabitatSimActions.move_forward
            print("action: FORWARD")
        elif keystroke == ord(LEFT_KEY):
            action = HabitatSimActions.turn_left
            print("action: LEFT")
        elif keystroke == ord(RIGHT_KEY):
            action = HabitatSimActions.turn_right
            print("action: RIGHT")
        elif keystroke == ord(FINISH):
            action = HabitatSimActions.stop
            print("action: FINISH")
        else:
            print("INVALID KEY")
            continue

        observations = env.step(action)
        count_steps += 1

        print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
            observations["pointgoal_with_gps_compass"][0],
            observations["pointgoal_with_gps_compass"][1]))

        image = transform_rgb_bgr(observations["rgb"])
        cv2.imshow("RGB", image)

    if (
        action == HabitatSimActions.stop
        and observations["pointgoal_with_gps_compass"][0] < 0.2
    ):
        print("you successfully navigated to destination point")
    else:
        print("your navigation was unsuccessful")


if __name__ == "__main__":
    example()