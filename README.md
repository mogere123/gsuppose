# gSUPPOSe

SUPPOSe deconvolution algorithm by means of Gradient Descent methods.

### Requisites:

  * A Linux based Operating System.
  * Python 3.
  * CUDA Compiler (`nvcc`) 9.1 or higher, added to System Path.
  * CaTMU library.

**We strongly recommed to use gSUPPOSe with a NVIDIA GPU Device (with Compute Capability 5.0 or higher) since it 
drastically reduces computation times.** Note that, even if no GPU is used, CUDA Compiler is still needed in order to compile the CaTMU library. 

### Installation

Please first install [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) and add the installation folder to the 
System Path (as indicated in the installation guide from the provider). Also install NVidia Drivers to use a GPU.

  1. Install CaTMU with `pip` from its Git repository:
     ```commandline
     pip install git+https://gitlab.com/alemazzeo/catmu.git
     ```
     
  2. Install gSUPPOSe with `pip` from the Git repository:
     ```commandline
     pip install git+https://gitlab.com/labofotonica/gsuppose.git
     ```

Done! You are ready to run gSUPPOSe. You can test the installation in a terminal by running
```commandline
ipython
import pysuppose
quit
```
and see if any errors are printed.

### Usage

Please see the [examples](gsuppose/examples) folder where you may find a 
[Jupyter Notebook](https://nbviewer.org/url/gitlab.com/labofotonica/gsuppose/-/raw/master/examples/basic_run.ipynb) and 
a [Python Script](examples/basic_run.py) that shows howto run gSUPPOSe on a simulated Single Molecule Localization 
Microscopy image.