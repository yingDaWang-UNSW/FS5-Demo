
import argparse
import os
from gooey import Gooey, GooeyParser

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def str2int(v):
    if v=='M':
        return v
    try:
        v = int(v)
    except:
        raise argparse.ArgumentTypeError('int value expected.')
    return v
def str2float(v):
    if v=='M':
        return v
    try:
        v = float(v)
    except:
        raise argparse.ArgumentTypeError('float value expected.')
    return v

# @Gooey(program_name="FS5 Demo", default_size=(600, 700), force_start=True)
# def fs5args():
#     script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')  # Get the directory where the script is located
#     surface_mesh_path = os.path.join(script_dir, 'surface.ply')  # Build the full path for the surface mesh
#     particle_mesh_path = os.path.join(script_dir, 'particles.ply')  # Build the full path for the particle mesh

#     parser = GooeyParser(description="Configure your simulation parameters")
#     parser.add_argument("--surfaceMesh", widget="FileChooser", type=str, default=surface_mesh_path, help="PLY file of surface")
#     parser.add_argument("--particleMesh", widget="FileChooser", type=str, default=particle_mesh_path, help="PLY file of particle volume")
#     parser.add_argument("--gpuIDs", type=str, default='0', help="GPU ID")
#     parser.add_argument("--runtimeRendering", type=str2bool, default=True) 
#     parser.add_argument("--radius", type=str2float, default=0.5) 
#     parser.add_argument("--particleMass", type=str2float, default=10000) 
#     parser.add_argument("--swellingRatio", type=str2float, default=0.2)
#     parser.add_argument("--dynamicParticleFriction", type=str2float, default=0.1) 
#     parser.add_argument("--staticParticleFriction", type=str2float, default=0.2) 
#     parser.add_argument("--dynamicSurfaceFriction", type=str2float, default=0.15) 
#     parser.add_argument("--staticSurfaceFriction", type=str2float, default=0.25) 
#     parser.add_argument("--sleepThreshold", type=str2float, default=0.06) 
#     parser.add_argument("--restitution", type=str2float, default=0.1) 

#     inputs = parser.parse_args()
#     inputs.xRef=1
#     inputs.particle_cohesion=0
#     inputs.particle_adhesion=0
#     inputs.relaxationFactor=1.05
#     inputs.restitution=0
#     inputs.sim_dt=0.01
#     inputs.interations=4
#     inputs.sim_substeps=1
#     inputs.solverUpdates=25
#     inputs.swelling_rotation_max=20
#     inputs.FS5MassScalingFactor=1/inputs.xRef**3
#     inputs.jitter=0.5
#     inputs.yShift=0
#     inputs.staticGroundVelocityThresholdRatio=inputs.sleepThreshold+1e-6
#     inputs.staticParticleVelocityThresholdRatio=inputs.sleepThreshold+1e-6
#     inputs.meshSidedFlag=0
#     inputs.max_steps_per_stage=inputs.solverUpdates*1000
#     inputs.framesPerRuntimeRender=1
#     inputs.swellingActivationFactor=4
#     inputs.swellingActivationLocationRatio=0
#     inputs.maxRadius=inputs.radius*((1+inputs.swellingRatio)**0.3333)
#     return inputs

def fs5args():
    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')  # Get the directory where the script is located
    surface_mesh_path = os.path.join(script_dir, 'surface.ply')  # Build the full path for the surface mesh
    particle_mesh_path = os.path.join(script_dir, 'particles.ply')  # Build the full path for the particle mesh

    parser = GooeyParser(description="Configure your simulation parameters")
    parser.add_argument("--surfaceMesh", type=str, default='D:\surface.ply', help="PLY file of surface")
    parser.add_argument("--particleMesh", type=str, default='D:\particles.ply', help="PLY file of particle volume")
    parser.add_argument("--gpuIDs", type=str, default='0', help="GPU ID")
    parser.add_argument("--runtimeRendering", type=str2bool, default=True) 
    parser.add_argument("--radius", type=str2float, default=0.5) 
    parser.add_argument("--particleMass", type=str2float, default=10000) 
    parser.add_argument("--swellingRatio", type=str2float, default=0.2)
    parser.add_argument("--dynamicParticleFriction", type=str2float, default=0.1) 
    parser.add_argument("--staticParticleFriction", type=str2float, default=0.2) 
    parser.add_argument("--dynamicSurfaceFriction", type=str2float, default=0.15) 
    parser.add_argument("--staticSurfaceFriction", type=str2float, default=0.25) 
    parser.add_argument("--sleepThreshold", type=str2float, default=0.06) 
    parser.add_argument("--restitution", type=str2float, default=0.1) 

    inputs = parser.parse_args()
    inputs.xRef=1
    inputs.particle_cohesion=0
    inputs.particle_adhesion=0
    inputs.relaxationFactor=1.05
    inputs.restitution=0
    inputs.sim_dt=0.01
    inputs.interations=4
    inputs.sim_substeps=1
    inputs.solverUpdates=25
    inputs.swelling_rotation_max=20
    inputs.FS5MassScalingFactor=1/inputs.xRef**3
    inputs.jitter=0.5
    inputs.yShift=0
    inputs.staticGroundVelocityThresholdRatio=inputs.sleepThreshold+1e-6
    inputs.staticParticleVelocityThresholdRatio=inputs.sleepThreshold+1e-6
    inputs.meshSidedFlag=0
    inputs.max_steps_per_stage=inputs.solverUpdates*1000
    inputs.framesPerRuntimeRender=1
    inputs.swellingActivationFactor=4
    inputs.swellingActivationLocationRatio=0
    inputs.maxRadius=inputs.radius*((1+inputs.swellingRatio)**0.3333)
    return inputs