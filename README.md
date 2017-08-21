TODO: Weekend 13th August
- Create tables in latex 

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

TODO: 21-Aug-2017
- Reshape the filter as (3,2,1) and then swap the elements as is done in case of back propagation through convolution.
- Another approach similar to NICE can be tried as well. In this case, instead of taking the entire time sequence, we
can take a chunk of varying length and then combine the results. A/c to Max this wouldn't require much of effort in terms
of implementation. 


Observations:
06-Aug-2017 - Planar flow - 2017_08_06_16_24_50 - Reconstruction even after 5000 iterations. But sine and velocity were 
not reconstructed even after 75000 iterations. 


# Generation of gif
pip install moviepy
Run the script given here: https://gist.github.com/nirum/d4224ad3cd0d71bfef6eba8f3d6ffd59
You will get the following error:

'''
NeedDownloadError: Need ffmpeg exe. You can download it by calling:
  imageio.plugins.ffmpeg.download()
'''

pip install imageio
Now: import imageio and then imageio.plugins.ffmpeg.download()

This solution is taken from : 
https://stackoverflow.com/questions/41402550/raise-needdownloaderrorneed-ffmpeg-exe-needdownloaderror-need-ffmpeg-exe

# Comparison of losses given by different models
- Standalone script in ./analysis/compare_losses.py. Set up the path to the log files and output directory.
- Run the script in terminal: python -m analysis.compare_losses. Note that this is a module. That's why there is no '/'
or '.py' at the end. 
 
# Following are the graphs which ought to be in the final report.
 - Comparison of losses given by different models. 
 - Generative signal
 - Plot showing the actual and reconstructed signals
 - Distribution of signal values in actual and reconstructed signal (I believe this should go into the appendix and 
 not in the chapter on results and discussions). 
 - Link to the video generated from the generative signals. 
- Table with losses across models. 
 
 
Tips on running the script on another machine.
- #1
*Problem:* Issue with imageio
'''
imageio.core.fetching.NeedDownloadError: Need ffmpeg exe. You can download it by calling:
 imageio.plugins.ffmpeg.download()
''' 
*Solution:* Open python in console. Run ```imageio.plugins.ffmpeg.download()```
 