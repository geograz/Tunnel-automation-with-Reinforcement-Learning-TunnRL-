# Reinforcement Learning for Geotechnics

This repository contains the codes for the paper:

_Reinforcement Learning based Process Optimization and Strategy Development in Conventional Tunneling_

by Georg H. Erharter, Tom F. Hansen, Zhongqiang Liu and Thomas Marcher

published in ...

DOI: ...

The paper was published as part of a collaboration on Machine Learning between the __Institute of Rock Mechanics and Tunnelling (Graz University of Technology)__
and the __Norwegian Geotechnical Institute (NGI)__ in Oslo.

## Requirements and folder structure

Use the `requirements.txt` file to download the required packages to run the code. We recommend using a package management system like conda for this purpose.

## Code and folder structure set up

The code framework depends on a certain folder structure. The python files should be placed in the main directory. The set up should be done in the following way:
```
Reinforcement_Learning_for_Geotechnics
├── 02_plots
│   └── tmp
├── 04_checkpoints
├── 06_results
│   └── tmp
├── 00_main.py
├── 02_model_tester.py
├── 04_analyzer.py
├── A_utilities.py
├── B_generator.py
├── C_geotechnician.py
├── D_tunnel.py
└── E_plotter.py
```
Either set up the folder structure manually or on Linux run:
```bash
bash folder_structure.sh
```

## Code description

- `00_main.py` ... is the main executing file
- `02_model_tester.py` ... file that runs and tests individual checkpoints of already trained model for further analysis
- `04_analyzer.py` ... file that analyzes and visualizes the performance of agents tested with `02_model_tester.py`
- `A_utilities.py` ... is a library containing useful functions that do not directly belong to the environment or the agent
- `B_generator.py` ... part of the environment that generates a new geology for every episode
- `C_geotechnician.py` ... part of the environment that evaluates the stability and also contains the RL agent itself
- `D_tunnel.py` ... part of the environment that handles the rewards and updates the progress of the excavation
- `E_plotter.py` ... plotting functionalities to visualize the training progress or render episodes

## Pseudo - code for the utilized DQN-algorithm

(inspired by [Deeplizard](https://deeplizard.com/learn/video/ewRw996uevM))

- A. Initialize replay memory capacity ("un-correlates" the otherwise sequential correlated input)
- B. Inititalize the policy-ANN (keeps the optimal approximated Q-function) with random weights
- C. Clone the policy-ANN to a second target-ANN that is used for computing $ Q^* $ in $Q^*(s,a) - Q(s,a) = loss$
- D. For each episode:
  1. Initialize the starting state (not resetting the weights)
  2. For each time step:
      - Select an action after an epsilon-greedy strategy (exploitation or exploration)
      - Execute the selected action in and emulator
      - Observe reward and next state
      - Store experience (a tuple of old-state, action, reward, new-state) in replay memory
      - Sample a random batch from replay memory
      - Preprocess all states (an array of values) from batch
      - Pass batch of preprocessed states and next-states to policy-ANN and target-ANN. Predict Q-values for both ANN's.
      - Calculate loss between output Q-values from policy-ANN and target-ANN
      - Standard gradient descent with back propagation updates weights in the policy-ANN to minimize loss. Every xxx timestep the weights in the target-ANN is updated with weights from the policy-ANN

## References

Besides other references we especially want to highlight the [Reinforcement Learning with Python](https://www.youtube.com/playlist?list=PLQVvvaa0QuDezJFIOU5wDdfy4e9vdnx-7)
tutorial series of [Sentdex](https://www.youtube.com/c/sentdex) which served as a basis for the agent in `C_geotechnician.py`.

- ... other references
