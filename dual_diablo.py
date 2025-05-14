# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Diablo Biped-Wheeled robot."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# Configuration
##

DUAL_DIABLO_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="Your path to the file/DualDiabloFlat_PayloadV7.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=0.1,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.25),
        rot = (1.0,0.0,0.0,0.0),
        joint_pos={
            "left_j1": 0.505,  # Can be modified to adjust the height of the legs
            "left_j2": -1.01,
            "right_j1": 0.505,  
            "right_j2": -1.01, 
        },
    ),
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=["left_.*", "right_.*"],
         
            min_delay=0,  # physics time steps (min: 5.0 * 0 = 0.0ms)
            max_delay=4,  # physics time steps (max: 5.0 * 4 = 20.0ms)
            stiffness={
                "left_j3": 0.0,
                "right_j3": 0.0,
                "left_j1" : 60.0,
                "left_j2" : 60.0,
                "right_j1" : 60.0,
                "right_j2" : 60.0,
                
            },
            damping={
                "left_j3": 0.7,
                "right_j3": 0.7,
                "left_j1" : 0.7,
                "left_j2" : 0.7,
                "right_j1" : 0.7,
                "right_j2" : 0.7,
            },
        ),

    },
)

DUAL_DIABLO_CFG2 = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot2",
    spawn=sim_utils.UsdFileCfg(
        usd_path="Your path to the file/DualDiabloFlat_PayloadV7.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=0.1,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0.5, 0.0, 0.25),
        rot = (1.0,0.0,0.0,0.0),
        joint_pos={
            "left_j1": 0.505,  # Can be modified to adjust the height of the legs
            "left_j2": -1.01,
            "right_j1": 0.505,  
            "right_j2": -1.01,  
        },
    ),
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=["left_.*", "right_.*"],
            
            min_delay=0,  # physics time steps (min: 5.0 * 0 = 0.0ms)
            max_delay=4,  # physics time steps (max: 5.0 * 4 = 20.0ms)
            stiffness={
                "left_j3": 0.0,
                "right_j3": 0.0,
                "left_j1" : 60.0,
                "left_j2" : 60.0,
                "right_j1" : 60.0,
                "right_j2" : 60.0,
                
            },
            damping={
                "left_j3": 0.7,
                "right_j3": 0.7,
                "left_j1" : 0.7,
                "left_j2" : 0.7,
                "right_j1" : 0.7,
                "right_j2" : 0.7,
            },
        ),

    },
)
