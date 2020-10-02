# Reinforcement_Learning_for_Geotechnics

This repository contains the codes for the paper:

_Reinforcement Learning based Process Optimization and Strategy Development in Conventional Tunneling_

by Georg H. Erharter, Tom F. Hansen, Zhongqiang Liu and Thomas Marcher

published in ...

DOI: ...

The paper was published as part of a collaboration on Machine Learning between the __Institute of Rock Mechanics and Tunnelling (Graz University of Technology)__
and the __Norwegian Geotechnical Institute (NGI)__ in Oslo.

## Code and Folder Structure

To run the Reinfocement Learning SImulation, execute the Python files from a folder, that contains the below given files and subfolders:

- `00_main.py` ... is the main executing file
- `A_utilities.py` ... is a library containing useful functions that do not directly belong to the environment or the agent
- `B_generator.py` ... part of the environment that generates a new geology for every episode
- `C_geotechnician.py` ... part of the environment that evaluates the stability and also contains the agent itself
- `D_tunnel.py` ... part of the environment that handles the rewards and updates the progress of the excavation
- `E_plotter.py` ... plotting functionalities to visualize the training progress or render episodes
- 02_plots ... a folder that will contain plots and renderings of episodes
  - tmp ... a subfolder of 02_plots that temporarily saves single frames of rendered episodes
- 04_checkpoints ... a folder where checkpoints of the agent are saved and loaded from

## References

Besides other references we especially want to highlight the [Reinforcement Learning with Python](https://www.youtube.com/playlist?list=PLQVvvaa0QuDezJFIOU5wDdfy4e9vdnx-7)
tutorial series of [Sentdex](https://www.youtube.com/c/sentdex) which served as a basis for the agent in `C_geotechnician.py`.

- ... other references


