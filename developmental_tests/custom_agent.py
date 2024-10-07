'''
A script that does some simple continuous navigation in Habitat_SIM

This doesn't have any sense of motion blur...
'''

import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut



from fovial_imagery import FovialImageFactory
from utils import *

from matplotlib import pyplot as plt
import magnum as mp
import numpy as np
import imageio
import random
import cv2

from habitat.core.agent import Agent
from habitat.core.simulator import ActionSpaceConfiguration


## UTIL FUNCTIONS ##
def make_configuration(settings):
    '''Basic configuration with a simple rgb camera'''

    # backend
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = settings["scene"]
    backend_cfg.enable_physics = True
    backend_cfg.allow_sliding = True

    # agent
    CAMsensor_cfg = habitat_sim.CameraSensorSpec()
    CAMsensor_cfg.sensor_type = habitat_sim.sensor.SensorType.COLOR
    CAMsensor_cfg.uuid = "rgb"
    CAMsensor_cfg.resolution = [settings["width"],settings["height"]]
    CAMsensor_cfg.position = [0.0, settings["sensor_height"], 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [CAMsensor_cfg]

    return habitat_sim.Configuration(backend_cfg,[agent_cfg])



# Define action space for both base movement and camera pan/tilt
class CustomActionSpace(ActionSpaceConfiguration):
    def __init__(self, config):
        self.config = config

    def get(self):
        return {
            "move_base": {
                "type": "VECTOR",  # Linear and angular velocities (x, y, z)
                "control": habitat.Control.Space.Box(low=-1.0, high=1.0, shape=(2,)),  # Linear and angular
            },
            "move_camera": {
                "type": "VECTOR",  # Pan and tilt angles
                "control": habitat.Control.Space.Box(low=-1.0, high=1.0, shape=(2,)),  # Pan and tilt
            },
        }

# Agent class with velocity control for base and pan/tilt for camera
class CustomAgent(Agent):
    def act(self, observations):
        # Define actions for base movement (linear_velocity, angular_velocity)
        base_action = {
            "move_base": [0.5, 0.0],  # Move forward with 0.5 speed, no rotation
        }
        
        # Define actions for camera movement (pan, tilt)
        camera_action = {
            "move_camera": [0.1, -0.1],  # Slight pan to right, tilt downwards
        }

        return {**base_action, **camera_action}

# Configuration of the agent's sensors (e.g., RGB, depth, etc.)
def configure_agent():
    config = habitat.get_config("benchmark/nav/pointnav/drew_test_big_sensor.yaml")
    print(dir(config))
    config.SIMULATOR.ACTION_SPACE_CONFIG = CustomActionSpace(config)

    # config.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
    # config.SIMULATOR.AGENT_0.HEIGHT = 1.5  # Camera height

    return config

# Create the environment and run a simulation
def run_simulation():
    config = configure_agent()
    env = habitat.Env(config)
    
    agent = CustomAgent()

    observations = env.reset()
    for _ in range(100):
        action = agent.act(observations)
        observations = env.step(action)
        env.render()  # Visualize the environment

if __name__ == "__main__":
    run_simulation()