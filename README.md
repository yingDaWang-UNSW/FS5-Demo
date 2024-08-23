# FS5 Demo

FS5 is the 5th generation of Beck Engineering’s Cave Flow Simulators. It is based on the Extended Position Based Dynamics (XPBD) method. and leverages Cuda and MPI to achieve giga-scale particle simulations scaling over distributed computational resources.

It is written in Python, allowing for faster prototyping and modification, more advanced particle flow control, and leverages the Nvidia Warp framework to achieve high performance.

This is an open source demo of FS5, which runs on single GPUs. It is capable of 5 million particles per GB of memory, and approximately 25 million particle updates per second on an RTX 2080ti


## Installation - Windows Python
1. Install the latest Nvidia graphics drivers and Cuda Framework.

https://www.nvidia.com/download/index.aspx

https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64

2. double-click install.bat

3. double-click run.bat


## Installation - CONDA/PIP

1.  install nvidia drivers and cuda toolkit (via website), confirm installation with nvidia-smi command
2.  conda install python
3.  pip install warp-lang pyglet trimesh meshio Rtree gooey matplotlib
4.  python runFS5GUIDemo.py

## Known bugs

1. the file path to your ply files cannot contain spaces, this is a limitation of a legacy voxeliser.