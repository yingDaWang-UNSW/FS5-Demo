# FS5-Warp

FS5 is the 5th generation of Beck Engineering’s Cave Flow Simulators. It is based on the Extended Position Based Dynamics (XPBD) method. and leverages Cuda and MPI to achieve giga-scale particle simulations scaling over distributed computational resources.

It is written in Python, allowing for faster prototyping and modification, more advanced particle flow control, and leverages the Nvidia Warp framework to achieve high performance.

This is an open source demo of FS5, which runs on single GPUs

## Installation

1.  install nvidia drivers and cuda toolkit (via website), confirm installation with nvidia-smi command
2.  conda install python
3.  pip install warp-lang pyglet ipython trimesh Rtree