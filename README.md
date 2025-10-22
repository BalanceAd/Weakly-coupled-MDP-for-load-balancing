# Weakly-coupled-MDP-for-load-balancing
A weakly coupled Lagrangian relaxed Model for load balancing in the context of containerized cloud compulting, this is a scientific research project.
#####################################################################################################################################################

  * File: "Toy_model.py" contains the simulation and policy iteration for the toy sized model, it contains all the necessary functions to make the simulation
    it is not a main.
  * File: "MDP_scaled.py" contains all the functions necessary to resolve the local MDP and the Lagrangian relaxation online.
  * File: "Full_sim.py" contains the functions necessary for the full simulation of the scaled MDP.
  * File: "Dispatcher.py" contains the functions of the load aware dispatcher.
  * File: "Compare_dispatcher.py" contains the functions of the simulation on the dispatcher and compares it to the baselines.
  * File: "Baselines_toy.py" contains the baselines to which we compare the toy sized model.
  * File: "Baselines_scaled.py" contains the baselines to which we compare the scaled system.

I did not commit the main since I change it often to experiment with the models.
