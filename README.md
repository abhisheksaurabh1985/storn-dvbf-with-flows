TODO: 30-July-2017
- Save and restore the models


TODO: 26-July-2017
- Redo the plots
- Talk to Max about how many data points should be plotted
- Create separate output directories for different flows
- Read about hyper-parameter search. Ask Max if Grid search in scikit learn can be used.

TODO: 01-Aug-2017
- Radial flows: alphas are positive. Initialize alphas using a uniform distribution between 0 and 1. 
Ask Max if this can ensure positive alphas.  
- Handle the case of numFlow=0. Define sum_logdet_jacobian as a tensorflow variable.
- Save the models in the directory corresponding to the flow. For example, 'Planar' directory should
 have the output as well as the saved models. 

TODO: 05-Aug-2017
- Play with the amount of noise. Noise does not have to be a standard normal. It serves as a discretization parameter. 
A/c to me, the smaller the noise, the better the reconstruction should be. 


Observations:
06-Aug-2017 - Planar flow - 2017_08_06_16_24_50 - Reconstruction even after 5000 iterations. But sine and velocity were 
not reconstructed even after 75000 iterations. 