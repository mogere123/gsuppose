#!/usr/bin/env python
# coding: utf-8

# # Basic example for gSUPPOSe
# 
# This notebook demonstrate the basic usage of gSUPPOSe for image deconvolution. We will take a simulation of a
# fluorescence microscopy image of four single emitters with a known gaussian PSF and we will run gSUPPOSe to resolve
# the underlying object.

# In[15]:


from gsuppose import GSUPPOSe
from gsuppose.fitness import mse, dmse
from catmu import get_available_devices
import matplotlib.pyplot as plt


# ---

# ## Load the examples
# 
# First we will load our sample and the PSF of the microscope. We will also load an initial guess for our gSUPPOSe
# solution, i.e. a list of a fixed number of N virtual sources.


from gsuppose.examples.fouremitters import sample, psf, initial_positions

print(f"The sample shape is {sample.shape}")
print(f"The PSF shape is {psf.shape}")
print(f"There are N = {len(initial_positions)} virtual sources.")

# Let's plot our initial examples. Please note that, in this case, **the PSF has a 10 times smaller pixel size** than the
# sample, so the real dimensions are not represented in the image sizes.

fig, axes = plt.subplots(ncols=3)
axes[0].imshow(sample)
axes[0].set_title(f"Sample")

axes[1].plot(initial_positions[:, 0], initial_positions[:, 1], "ok", markersize=1)
axes[1].set_title(f"Initial positions")
axes[1].set_aspect(1)
axes[1].set_xlim(axes[0].get_xlim())
axes[1].set_ylim(axes[0].get_ylim())

axes[2].imshow(psf)
axes[2].set_title(f"PSF")

plt.show()


# ## Configure gSUPPOSe

# We will initialize a GSUPPOSE object and prepare it for our run. First we tell gSUPPOSe to take our `sample`, `psf`
# and `initial_positions` loaded before. With the arguments `sample_pixel_size` and `psf_pixel_size` we tell the
# algorithm that this PSF image has a 10 times smaller grid, which is useful to reduce interpolation error. We set the
# argument `normalize_input` to `'std'` to normalize our sample by dividing by its standard deviation, which is useful
# for mantaining the same optimizer configuration (such as the learning rate) for images with different intensities.
# 
# The argument `optimizer` sets the gradient descent method that we will use, while `fitness_function` and `dfitness_
# function` sets the loss function that is minimized and its derivative. In this case we will us an ADAM optimizer with
# a Mean Squared Error loss and a `batch_size` of 5 virtual sources. The argument `global_scale` sets the learning rate
# and may be adjusted for each run.
# 
# Finally, the argument `device` sets the device (or a list of devices) to use for computation. Positive integers `0,
# 1, ...` correspond to the available GPUs while `-1` correspond to CPU. Currently, gSUPPOSe allows to distribute the
# computation up to 3 devices, but since our GPU implementation is much faster than CPU, we recommend to use GPU
# devices if possible.

suppose = GSUPPOSe(sample=sample,
                   psf=psf,
                   initial_positions=initial_positions,
                   sample_pixel_size=(1.0, 1.0),
                   psf_pixel_size=(.1, .1),
                   normalize_input='std',
                   optimizer='adam',
                   fitness_function=mse,
                   dfitness_function=dmse,
                   batch_size=5,
                   global_scale=1e-1,
                   device_index=0 if get_available_devices() else -1)


# ## Run gSUPPOSe

# Now we perform our run. We will run the algorithm through a maximum of 1000 `epochs` (iterations). The argument
# `stop_method` adds an extra stop condition which, in this case, finishes the execution when the maximum displacement
# of all virtual sources in an epoch is less than `stop_limit`. The argument `report_every` tells the algorithm to
# print its status during the run for every `100` epochs. We also enable live plots that shows the execution (plotting
# do not affect computation time since it runs in a parallel process) and we save our results in a file called
# `basic_run.npz`.

# At the current version, controlling the verbosity and the output stream is still pending.

suppose.run(epochs=5000,
            stop_method='max_displacement',
            stop_limit=1e-3,
            report_every=500,
            plot=False,
            save_path="basic_run.npz")


# ## Plot the results

# Finally we plot the results of our run. The source intensity `alpha` can also be retrieved.

print(f"The fitted source intensity (alpha) is '{suppose.alpha:.3e}'")

fig, axes = plt.subplots(ncols=3)
axes[0].imshow(sample);
axes[0].set_title(f"Sample");

axes[1].plot(initial_positions[:, 0], initial_positions[:, 1], "ok", markersize=2, alpha=.2)
axes[1].set_title(f"Initial positions")
axes[1].set_aspect(1)
axes[1].set_xlim(axes[0].get_xlim())
axes[1].set_ylim(axes[0].get_ylim())

axes[2].plot(suppose.positions[:, 0], suppose.positions[:, 1], "ok", markersize=2, alpha=.2)
axes[2].set_title(f"Final positions")
axes[2].set_aspect(1)
axes[2].set_xlim(axes[0].get_xlim())
axes[2].set_ylim(axes[0].get_ylim())

plt.show()

