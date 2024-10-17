#!/usr/bin/env python3



# TODO: Remove everything todo with an arm.

import argparse
import os
import os.path as osp
import time
from collections import defaultdict
from typing import Any, Dict, List

import magnum as mn
import numpy as np

import habitat
import habitat.tasks.rearrange.rearrange_task
from habitat.articulated_agent_controllers import HumanoidRearrangeController
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    GfxReplayMeasureMeasurementConfig,
    PddlApplyActionConfig,
    ThirdRGBSensorConfig,
    HeadRGBSensorConfig,
)
from habitat.core.logging import logger
from habitat.tasks.rearrange.actions.actions import ArmEEAction
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import euler_to_quat, write_gfx_replay
from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
)
from habitat.config.default_structured_configs import ActionConfig
from habitat.tasks.nav.nav import SimulatorTaskAction
from habitat_sim.utils import viz_utils as vut

import random
from dataclasses import dataclass


try:
    import pygame
except ImportError:
    pygame = None

# Create a new camera move action
@dataclass
class CamControlActionConfig(ActionConfig):
    ang_speed: float = 0.0 # change this in config
    noise_amount: float = 0.0

# Define the new action
# the __init__ method receives a sim and config argument.
'''
Here are the options:
- Global position
- Relative position (let's do this for now)
- camera pitch/yaw angular velocity
'''

@habitat.registry.register_task_action
class CamVelocityAction(SimulatorTaskAction):
    def __init__(self, *args, config, sim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._sim = sim
        self._ang_speed = config.ang_speed
        self._noise_amount = config.noise_amount

    def _get_uuid(self, *args, **kwargs):
        return "cam_velocity_control"

    def step(self, *args, **kwargs):
        # print(
        #     f"Calling {self._get_uuid()} d={self._ang_speed}m noise={self._noise_amount}"
        # )
        # This is where the code for the new action goes. Here we use a
        # helper method but you could directly modify the simulation here.
        # _strafe_body(self._sim, self._move_amount, 90, self._noise_amount)
        # offset_rpy = np.array([0, 0, 0])

        # xyz_offset = np.array([10, 10, 10])
        # quat = euler_to_quat(offset_rpy) # TODO: no clue what frame this is in
        # trans = mn.Matrix4.from_(
        #     quat.to_matrix(), mn.Vector3(*xyz_offset)
        # )

        # print("The current state is ", dir(self._sim.get_agent(0)))
        # print("The current state is ", self._sim.get_agent(0).get_state())

        # val = np.array([0.0, 0.0, 0.0])

        # curr_state = self._sim.get_agent(0).get_state()
        # curr_state.sensor_states["head_rgb"].position = mn.Vector3(np.array([0.0, 0.0, 0.0]))
        # curr_state.position = mn.Vector3(val)
        
        agent = self._sim.get_agent(0)
        agent_state = agent.get_state()
        
        print("Starting camera position: ", agent_state.sensor_states["head_rgb"].position)
        
        cam_pos_offset = np.random.uniform(-1, 1, 3) * .1
        
        # Get the agent's current state
        

        # Get the current position of the head camera
        current_cam_pos = agent_state.sensor_states["head_rgb"].position

        # Update the position by adding the offset
        new_cam_pos = mn.Vector3(
            current_cam_pos[0] + cam_pos_offset[0],
            current_cam_pos[1] + cam_pos_offset[1],
            current_cam_pos[2] + cam_pos_offset[2]
        )

        # Update the camera position in the sensor state
        agent_state.sensor_states["head_rgb"].position = new_cam_pos

        # Set the updated agent state back
        agent.set_state(agent_state, reset_sensors=False)

        # Print the updated camera position for debugging
        print("Updated camera position: ", agent_state.sensor_states["head_rgb"].position)
        
        observations = env._sim.get_sensor_observations()["head_rgb"]
        observations = self._sim.get_observations_at(
            agent_state.sensor_states["head_rgb"].position, agent_state.rotation
        )
        return observations

        raise "I want to see the stack trace"

        pass

        

        # We are doing local position




# Please reach out to the paper authors to obtain this file
# DEFAULT_CFG = "benchmark/rearrange/play/play.yaml"
DEFAULT_CFG = "drew_test.yaml"

DEFAULT_RENDER_STEPS_LIMIT = 600
SAVE_VIDEO_DIR = "./data/vids"

def step_env(env, action_name, action_args):
    return env.step({"action": action_name, "action_args": action_args})

i = 0

def get_input_vel_ctlr(
    skip_pygame,
    cfg,
    env,
    not_block_input,
):
    global i 
    agent_k = ""

    if "spot" in cfg:
        base_action_name = f"{agent_k}base_velocity_non_cylinder"
    else:
        base_action_name = f"{agent_k}base_velocity"
    base_key = "base_vel"


    base_action = [0, 0]
    end_ep = False

    if skip_pygame:
        base_action = [(random.random()-0.5)*2, (random.random()-0.5)*2]
        args: Dict[str, Any] = {}

        if base_action is not None and base_action_name in env.action_space.spaces:
            name = base_action_name
            args = {base_key: base_action}

        return step_env(env, name, args), end_ep

    keys = pygame.key.get_pressed()

    if keys[pygame.K_ESCAPE]:
        return None, False
    elif keys[pygame.K_m]:
        end_ep = True
    elif keys[pygame.K_n]:
        env._sim.navmesh_visualization = not env._sim.navmesh_visualization


    # Base control
    if keys[pygame.K_j]:
        # Left
        base_action = [0, 0.25]
    elif keys[pygame.K_l]:
        # Right
        base_action = [0, -0.25]
    elif keys[pygame.K_k]:
        # Back
        base_action = [-0.25, 0]
    elif keys[pygame.K_i]:
        # Forward
        base_action = [0.25, 0]
    

    if keys[pygame.K_PERIOD]:
        # Print the current position of the articulated agent, useful for debugging.
        pos = [
            float("%.3f" % x)
            for x in env._sim.articulated_agent.sim_obj.translation
        ]

        rot = env._sim.articulated_agent.sim_obj.rotation
        ee_pos = env._sim.articulated_agent.ee_transform().translation
        logger.info(
            f"Agent state: pos = {pos}, rotation = {rot}, ee_pos = {ee_pos}"
        )


    args: Dict[str, Any] = {}

    # Print out the available actions
    # print("These are the action spaces: ", env.action_space.spaces) # looks like theres an action space for the arm and the body...

    if base_action is not None and base_action_name in env.action_space.spaces:
        name = base_action_name
        args = {base_key: base_action}

    # name=  "velocity_control",
    # args = {
    #         "angular_velocity": 0.2,
    #         "linear_velocity": 0.2,
    # }
    # if i > 10:
    #     name = "DO_NEW_ACTION"
    #     args = {}
    # i += 1
    name = (base_action_name, "DO_NEW_ACTION")
    args = {
        base_key: base_action
    }

    return step_env(env, name, args), end_ep


class FreeCamHelper:
    def __init__(self):
        self._is_free_cam_mode = False
        self._last_pressed = 0
        self._free_rpy = np.zeros(3)
        self._free_xyz = np.zeros(3)

    @property
    def is_free_cam_mode(self):
        return self._is_free_cam_mode

    def update(self, env, step_result, update_idx):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_z] and (update_idx - self._last_pressed) > 60:
            self._is_free_cam_mode = not self._is_free_cam_mode
            logger.info(f"Switching camera mode to {self._is_free_cam_mode}")
            self._last_pressed = update_idx        

        if self._is_free_cam_mode:
            offset_rpy = np.zeros(3)
            if keys[pygame.K_u]:
                offset_rpy[1] += 1
            elif keys[pygame.K_o]:
                offset_rpy[1] -= 1
            elif keys[pygame.K_i]:
                offset_rpy[2] += 1
            elif keys[pygame.K_k]:
                offset_rpy[2] -= 1
            elif keys[pygame.K_j]:
                offset_rpy[0] += 1
            elif keys[pygame.K_l]:
                offset_rpy[0] -= 1

            offset_xyz = np.zeros(3)
            if keys[pygame.K_q]:
                offset_xyz[1] += 1
            elif keys[pygame.K_e]:
                offset_xyz[1] -= 1
            elif keys[pygame.K_w]:
                offset_xyz[2] += 1
            elif keys[pygame.K_s]:
                offset_xyz[2] -= 1
            elif keys[pygame.K_a]:
                offset_xyz[0] += 1
            elif keys[pygame.K_d]:
                offset_xyz[0] -= 1
            offset_rpy *= 0.1
            offset_xyz *= 0.1
            self._free_rpy += offset_rpy
            self._free_xyz += offset_xyz
            if keys[pygame.K_b]:
                self._free_rpy = np.zeros(3)
                self._free_xyz = np.zeros(3)
            
            cam_position = env._sim.get_agent(0).get_state().sensor_states["head_rgb"].position
            print("The position of the camera is", cam_position)
            
            # cam_position = np.array([0, 0, 1])

            rpy = np.random.uniform(-1, 1, 3) * 0.1
            quat = euler_to_quat(rpy)
            trans = mn.Matrix4.from_(
                quat.to_matrix(), mn.Vector3(*cam_position)
            )
            env._sim._sensors["head_rgb"]._sensor_object.node.transformation = trans
            step_result = env._sim.get_sensor_observations()
            return step_result
        return step_result


def play_env(env, args, config):
    render_steps_limit = None
    if args.no_render:
        render_steps_limit = DEFAULT_RENDER_STEPS_LIMIT

    obs = env.reset()

    # Set the initial arm configuration
    '''
    These are the limits: 
        min: array([-1.6056, -1.221 ,    -inf, -2.251 ,    -inf, -2.16  ,    -inf],
        max: array([1.6056, 1.518 ,    inf, 2.251 ,    inf, 2.16  ,    inf]
    '''
    env._sim.articulated_agent.set_fixed_arm_joint_pos([1.57, 1.50, 0, 1.57, 0.0, 1.57, 0.0])
    # print("This is one things", env._sim.articulated_agent)
    print("This is the other thing: ", env._sim.get_agent(0))
    print("This is the sensor: ", env._sim.get_agent(0)._sensors["head_rgb"])


    if not args.no_render:
        draw_obs = observations_to_image(obs, {})
        pygame.init()
        screen = pygame.display.set_mode(
            [draw_obs.shape[1], draw_obs.shape[0]]
        )

    update_idx = 0
    target_fps = 60
    prev_time = time.time()
    all_obs = []
    total_reward = 0

    free_cam = FreeCamHelper()

    while True:
        if render_steps_limit is not None and update_idx > render_steps_limit:
            break

        step_result, end_ep = get_input_vel_ctlr(
            args.no_render,
            args.cfg,
            env,
            not free_cam.is_free_cam_mode,
        )

        if step_result is None:
            break

        if end_ep:
            total_reward = 0
            # Clear the saved keyframes.
            env.reset() #TODO maybe you want to just reset env here?

        if not args.no_render:
            step_result = free_cam.update(env, step_result, update_idx)
        update_idx += 1

        obs = step_result
        
        info = env.get_metrics()

        reward_key = [k for k in info if "reward" in k]
        if len(reward_key) > 0:
            reward = info[reward_key[0]]
        else:
            reward = 0.0

        total_reward += reward
        info["Total Reward"] = total_reward

        if free_cam.is_free_cam_mode:
            cam = obs["head_rgb"]
            use_ob = np.zeros(draw_obs.shape)
            use_ob[:, : cam.shape[1]] = cam[:, :, :3]

        else:
            use_ob = observations_to_image(obs, info)

        draw_ob = use_ob[:]

        if not args.no_render:
            draw_ob = np.transpose(draw_ob, (1, 0, 2))
            draw_obuse_ob = pygame.surfarray.make_surface(draw_ob)
            screen.blit(draw_obuse_ob, (0, 0))
            pygame.display.update()
        if args.save_obs:
            all_obs.append(draw_ob)  # type: ignore[assignment]

        if not args.no_render:
            pygame.event.pump()
        if env.episode_over:
            total_reward = 0
            env.reset()

        curr_time = time.time()
        diff = curr_time - prev_time
        delay = max(1.0 / target_fps - diff, 0)
        # if (1.0 / target_fps - diff > 0):
        #     print(f"Able to play at {target_fps}")
        # else:
        #     print(f"cant play that fast here is diff {diff}s")
        time.sleep(delay)
        prev_time = curr_time


    if args.save_obs:
        all_obs = np.array(all_obs)  # type: ignore[assignment]
        all_obs = np.transpose(all_obs, (0, 2, 1, 3))  # type: ignore[assignment]
        os.makedirs(SAVE_VIDEO_DIR, exist_ok=True)
        vut.make_video(
            np.expand_dims(all_obs, 1),
            0,
            "color",
            osp.join(SAVE_VIDEO_DIR, args.save_obs_fname),
        )


    if not args.no_render:
        pygame.quit()


def has_pygame():
    return pygame is not None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-render", action="store_true", default=False)
    parser.add_argument("--save-obs", action="store_true", default=False)
    parser.add_argument("--save-obs-fname", type=str, default="fetch.mp4")
    parser.add_argument("--save-actions", action="store_true", default=False)
    parser.add_argument(
        "--save-actions-fname", type=str, default="play_actions.txt"
    )
    parser.add_argument(
        "--save-actions-count",
        type=int,
        default=200,
        help="""
            The number of steps the saved action trajectory is clipped to. NOTE
            the episode must be at least this long or it will terminate with
            error.
            """,
    )
    parser.add_argument("--play-cam-res", type=int, default=512)
    parser.add_argument(
        "--skip-render-text", action="store_true", default=False
    )
    parser.add_argument(
        "--same-task",
        action="store_true",
        default=False,
        help="If true, then do not add the render camera for better visualization",
    )

    parser.add_argument(
        "--never-end",
        action="store_true",
        default=False,
        help="If true, make the task never end due to reaching max number of steps",
    )

    parser.add_argument("--load-actions", type=str, default=None)
    parser.add_argument("--cfg", type=str, default=DEFAULT_CFG)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    if not has_pygame() and not args.no_render:
        raise ImportError(
            "Need to install PyGame (run `pip install pygame==2.0.1`)"
        )

    config = habitat.get_config(args.cfg, args.opts)
    with habitat.config.read_write(config):
        env_config = config.habitat.environment
        sim_config = config.habitat.simulator
        task_config = config.habitat.task

        task_config.actions["DO_NEW_ACTION"] = CamControlActionConfig(
            type="CamVelocityAction",
            ang_speed=1.0,
            noise_amount=0.0
        )


        if not args.same_task:
            sim_config.debug_render = True
            agent_config = get_agent_config(sim_config=sim_config)
            
            agent_config.sim_sensors.update(
                {
                    "head_rgb_sensor": HeadRGBSensorConfig(
                        height=args.play_cam_res, width=args.play_cam_res
                    )
                }
            )
            agent_config.sim_sensors.update(
                {
                    "third_rgb_sensor": ThirdRGBSensorConfig(
                        height=args.play_cam_res, width=args.play_cam_res
                    )
                }
            )



        if args.never_end:
            env_config.max_episode_steps = 0

    with habitat.Env(config=config) as env:
        play_env(env, args, config)
