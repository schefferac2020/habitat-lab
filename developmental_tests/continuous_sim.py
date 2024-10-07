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


# Open the test scene and setup config
test_scene = "./data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb"
sim_settings = {
    "scene": test_scene,  # Scene path
    "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
    "width": 512,  # Spatial resolution of the observations
    "height": 512,
}
cfg = make_configuration(sim_settings)
sim = habitat_sim.Simulator(cfg)

sim_time = 10 # in seconds
control_frequency = 5 # in Hz
frame_skip = 12 # integration frames per action
fps = control_frequency*frame_skip
print(f"Running at {fps} fps")

# Initilalize the agent
agent = sim.initialize_agent(0) # agent 0
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([-0.6, 0.0, 0.0])  # in world space
agent.set_state(agent_state)
# action_names = list(cfg.agents[0].action_space.keys()) # Get the discrete action names
# print("The discrete action space is: ", action_names)

# Populate a predetermined, random control sequence...
control_sequence = []
for _action in range(int(sim_time * control_frequency)):
    # allow forward velocity and y rotation to vary
    control_sequence.append(
        {
            "forward_velocity": (random.random()) * 4,  # [0,2)
            "rotation_velocity": (random.random() - 0.5) * 2.0,  # [-1,1)
        }
    )


# Set up a new VELOCITY CONTROL STRUCTURE
vel_control = habitat_sim.physics.VelocityControl()
vel_control.controlling_lin_vel = True
vel_control.lin_vel_is_local = True
vel_control.controlling_ang_vel = True
vel_control.ang_vel_is_local = True



observations = []
time_step_per_frame = 1/fps

for action in control_sequence:
    vel_control.linear_velocity = np.array([0, 0, -action["forward_velocity"]]) # local forward is -z direction
    vel_control.angular_velocity = np.array([0, action["rotation_velocity"], 0]) # local up is +y direction

    for _integration_frame in range(frame_skip):
        agent_state = agent.state
        previous_rigid_state = habitat_sim.RigidState(
            utils.quat_to_magnum(agent_state.rotation), agent_state.position
        )

        # manually integrate the rigid state
        target_rigid_state = vel_control.integrate_transform(
            time_step_per_frame, previous_rigid_state
        )

        # snap rigid state to navmesh and set state to object/agent
        # calls pathfinder.try_step or self.pathfinder.try_step_no_sliding
        end_pos = sim.step_filter(
            previous_rigid_state.translation, target_rigid_state.translation
        )

        # set the computed state
        agent_state.position = end_pos
        agent_state.rotation = utils.quat_from_magnum(
            target_rigid_state.rotation
        )
        agent.set_state(agent_state)

        # Check if a collision occured
        dist_moved_before_filter = (
            target_rigid_state.translation - previous_rigid_state.translation
        ).dot()
        dist_moved_after_filter = (
            end_pos - previous_rigid_state.translation
        ).dot()

        # NB: There are some cases where ||filter_end - end_pos|| > 0 when a
        # collision _didn't_ happen. One such case is going up stairs.  Instead,
        # we check to see if the the amount moved after the application of the filter
        # is _less_ than the amount moved before the application of the filter
        observation = sim.get_sensor_observations()
        EPS = 1e-5
        collided = (dist_moved_after_filter + EPS) < dist_moved_before_filter
        if collided:
            print("Collided with something!!!")
            
            block_size = 30
            new_img = observation['rgb']
            print("The shape is", new_img.shape)

            center_y, center_x = new_img.shape[1] // 2, new_img.shape[2] // 2
            half_block_size = block_size // 2

            # Set the red block in the center
            new_img[0:100, 0:100] = [255, 0, 0, 255]  # Red channel


            observation['rgb'] = new_img


        sim.step_physics(time_step_per_frame)
        observations.append(observation)


print("total frames = " + str(len(observations)))

if True:
    vut.make_video(
        observations=observations,
        primary_obs="rgb",
        primary_obs_type="color",
        video_file="continuous_nav",
        fps=fps
    )