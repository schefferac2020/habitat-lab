import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import cv2
import numpy as np

from fovial_imagery import FovialImageFactory
from utils import *

FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"

fovial_factory = None

def on_mouse(event, x, y, flags, param):
    global fovial_factory

    if event == cv2.EVENT_MOUSEMOVE:
        fovial_img = fovial_factory.get_fovial_image(x, y)

        print("This is the shape: ",fovial_img.shape)

        print(f"Cursor at {x}, {y}")

        cv2.imshow("FOVIAL IMAGE", fovial_img)

def example():
    global fovial_factory

    env = habitat.Env(
        config=habitat.get_config("benchmark/nav/pointnav/drew_test_big_sensor.yaml")
    )

    print("Environment creation successful")
    observations = env.reset()
    print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
        observations["pointgoal_with_gps_compass"][0],
        observations["pointgoal_with_gps_compass"][1]))

    image = transform_rgb_bgr(observations["rgb"])
    fovial_factory = FovialImageFactory(image, 30, 5)
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
        fovial_factory.set_image(image)
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