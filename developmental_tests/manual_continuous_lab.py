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
import quaternion

try:
    import pygame
except ImportError:
    pygame = None
    
def quat_2_list(quat: mn.Quaternion):
    return [quat.vector[0], quat.vector[1], quat.vector[2], quat.scalar]
    

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
        
        self.current_x = 0 # Left to right motion
        self.current_y = 0 # 

    def _get_uuid(self, *args, **kwargs):
        return "cam_velocity_control"

    def step(self, *args, **kwargs):
        eyes_ang_offset = kwargs["eyes_ang_offset"]
        if eyes_ang_offset is None:
            raise "This is bad. Should be passed in I think"

        self.current_x += eyes_ang_offset[0]
        self.current_y += eyes_ang_offset[1]
        
        
        
        
        rand_vec = np.random.uniform(-1, 1, 3) * 0.1
        quat = euler_to_quat(rand_vec)

        '''
        These are the things we can modify:
        
        attached_link_id', 'cam_look_at_pos', 'cam_offset_pos', 'cam_orientation', 'relative_transform'
            :property cam_offset_pos: The 3D position of the camera relative to the transformation of the attached link.
            :property cam_look_at_pos: The 3D of where the camera should face relative to the transformation of the attached link.
        '''
        
        env._sim.articulated_agent.params.cameras["head"].cam_look_at_pos = mn.Vector3(0, 0, 0) # Set this to 0 to disable it.

        # [pitch down->up, turn right->left, roll]
        env._sim.articulated_agent.params.cameras["head"].cam_orientation = mn.Vector3(0+self.current_y, -(1.57 + self.current_x), 0)
        
        #TODO: Is there some weird gimbal locking thing here?



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
):
    global i 
    agent_k = ""

    if "spot" in cfg:
        base_action_name = f"{agent_k}base_velocity_non_cylinder"
    else:
        base_action_name = f"{agent_k}base_velocity"
    base_key = "base_vel"
    
    eyes_action_name = "eyes_action"
    eyes_key = "eyes_ang_offset"


    base_action = [0, 0]
    eyes_action = [0, 0]
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
        
    # Eyes Control
    if keys[pygame.K_w]:
        # up
        eyes_action = [0, 0.01]
    elif keys[pygame.K_a]:
        # left
        eyes_action = [-0.01, 0]
    elif keys[pygame.K_s]:
        # down
        eyes_action = [0, -0.01]
    elif keys[pygame.K_d]:
        # Forward
        eyes_action = [0.01, 0]
    

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

    name = (base_action_name, eyes_action_name)
    args = {
        base_key: base_action,
        eyes_key: eyes_action
    }

    return step_env(env, name, args), end_ep

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

    while True:
        if render_steps_limit is not None and update_idx > render_steps_limit:
            break

        step_result, end_ep = get_input_vel_ctlr(
            args.no_render,
            args.cfg,
            env,
        )

        if step_result is None:
            break

        if end_ep:
            total_reward = 0
            # Clear the saved keyframes.
            env.reset() #TODO maybe you want to just reset env here?

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

        task_config.actions["eyes_action"] = CamControlActionConfig(
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
        if args.never_end:
            env_config.max_episode_steps = 0

    with habitat.Env(config=config) as env:
        play_env(env, args, config)
