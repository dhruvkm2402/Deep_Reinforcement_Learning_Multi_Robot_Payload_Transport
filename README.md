## Deep Reinforcement Learning for Coordinated Payload Transport in Biped-Wheeled Robots
Deep Reinforcement Learning for Coordinated Payload Transport in Biped-Wheeled Robots  A unified PyTorch-based framework that trains a single deep reinforcement learning (DRL) agent to coordinate two biped-wheeled robots for cooperative payload transport.

A demonstration repository showing:

1. **Deep Reinforcement Learning**-based payload transport in simulation
2. **Sim-to-Real** deployment on the Diablo biped-wheeled robots

---

## 🚀 Prerequisites

1. **Isaac Lab & Isaac Sim** 
   - NVIDIA Omniverse Isaac Sim (4.5.0) & Isaac Lab (2.1) installed 
   - [Installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)

2. **Workstation Requirements** 
   - GPU: NVIDIA RTX 30xx series or higher (≥16 GB VRAM) 
   - CPU: 6-core Intel i7 / AMD Ryzen 5 or better
   - RAM: ≥32 GB 
   - Ubuntu 22.04 LTS

3. **Diablo Robot Hardware** 
   - DirecDrive Tech's Diablo biped-wheeled robots (x2)
   - ROS Noetic (Linux) 
   - Diablo URDF + control stack

4. **OptiTrack Motion Capture** 
   - Motive v3.0+ installed & calibrated 
   - OptiTrack (NatNet) streaming engine with mocap_optitrack ROS package
   - [Guide](https://tuw-cpsg.github.io/tutorials/optitrack-and-ros/)

---
## Important Files and Directories
1. **dual_diablo** - Contains the environment file and RL agent files
2. **dual_diablo.py** - Contains the actuator and additional configurations of the biped-wheeled robot in simulation
3. **USD_DualDiablo** - Contains the USD files of the payload and biped-wheeled robot 

## 📁 Directory Structure

Copy the folder dual_diablo (contains the environment and RL agent files) & dual_diablo.py (robot config file) in the IsaacLab directory as shown:
```
── IsaacLab
    └── source
        ├── isaaclab_assets
        │   └── isaaclab_assets
        │       └── robots
        │           └── dual_diablo.py
        └── isaaclab_tasks
            └── isaaclab_tasks
                └── direct
                    └── dual_diablo
```
## 🕹️ Running in Simulation
### Ensure the paths of waypoints/payload path is modified in the dual_diablo_env.py file and the USD path in dual_diablo.py file
### Running the training
```
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task DualDiablo_Task_Simple --num_envs 4096 --headless
```

### Running the Evaluation
```
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task DualDiablo_Task_Simple --num_envs 4 --checkpoint /home/Your_Directory/IsaacLab/logs/rsl_rl/dualdiablo_rsl_rl/2025-05-13_20-18-28/model_500.pt
```
### 🎥 Video Demonstrations - Simulation
<table>
  <tr>
    <td>
      <!-- Training video -->
      <video controls width="320">
        <source src="Media/DualDiabloTraining_U.mp4" type="video/mp4">
        Your browser doesn’t support the video tag.
      </video>
      <p align="center"><strong>Sim Demo</strong></p>
    </td>
    <td>
      <!-- Evaluation video -->
      <video controls width="320">
        <source src="Media/SimGradualSineS1_U.mp4" type="video/mp4">
        Your browser doesn’t support the video tag.
      </video>
      <p align="center"><strong>Real-World Demo</strong></p>
    </td>
  </tr>
</table>
