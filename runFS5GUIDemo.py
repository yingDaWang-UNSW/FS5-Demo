
import numpy as np
import warp as wp
np.random.seed(42)  # Set the seed
import pdb
import os
import argparse
import trimesh
import utils as utl
from utils import fs5Model
from warpOverrides import fs5WarpOverrides
from warpOverrides import fs5RendererCore
from utils import fs5PlotUtils
from utils import fs5InputReader

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

np.seterr(divide='ignore')
class ExtendedClass(fs5Model.ModelBuilder):  
    def __init__(self, soft_contact_max_value=32*1024000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if soft_contact_max_value is not None:
            self._soft_contact_max = soft_contact_max_value
        else:
            self._soft_contact_max = 32 * 1024000  
    @property
    def soft_contact_max(self):
        """Maximum number of soft contacts that can be registered"""
        return self._soft_contact_max
    @soft_contact_max.setter
    def soft_contact_max(self, value):
        """Setter for maximum number of soft contacts"""
        self._soft_contact_max = value
class simulationStage:
    def __init__(self, inputs, mesh_points, mesh_indices, particles, activeIndices):
        if inputs.runtimeRendering:
            print('Runtime rendering is on')
            renderer = fs5RendererCore.OpenGLRenderer(        
            title="Warp sim",
            scaling=1.0,
            fps=60,
            up_axis="y",
            screen_width=1024,
            screen_height=768,
            near_plane=0.01,
            far_plane=10000/inputs.xRef,
            camera_fov=75.0,
            background_color=(0,0,0),
            draw_grid=True,
            draw_sky=False,
            draw_axis=True,
            show_info=True,
            render_wireframe=False,
            axis_scale=1.0,
            vsync=False,
            headless=False,
            enable_backface_culling=True)
            self.renderer=renderer
            self.renderer=fs5PlotUtils.look_at_centroid(np.vstack([mesh_points,particles]),self.renderer,self.renderer.camera_fov)
            renderer._paused=False
            self.renderer._camera_speed = 5/inputs.xRef
        builder = ExtendedClass(soft_contact_max_value=round(activeIndices.shape[0]))
        self.mesh = fs5Model.Mesh(mesh_points, mesh_indices)

        builder.add_body(origin=wp.transform((0, 0.0, 0.0), wp.quat_identity()))
        builder.add_shape_mesh(body=-1, mesh=self.mesh, pos=(0.0, 0.0, 0.0), rot=[wp.quat_identity()], scale=(1.0, 1.0, 1.0), mu=inputs.dynamicSurfaceFriction,restitution=inputs.restitution,is_solid=False,thickness=1., density=1.,has_ground_collision=True)

        for i in range(particles.shape[0]):

            m=inputs.particleMass
            r=inputs.radius/inputs.xRef
                
            builder.add_particle(pos=(particles[i,0],particles[i,1], particles[i,2]), vel=(0.0, 0.0, 0.0), mass=m, radius=r, flags = activeIndices[i])
        self.model = builder.finalize() 

        self.model.gravity = np.array((0.0, -9.81/inputs.xRef, 0.0)) 
        self.model.particle_max_velocity=inputs.radius/inputs.xRef/0.01*4/10
        self.model.particle_cohesion=inputs.particle_cohesion
        self.model.particle_adhesion=inputs.particle_adhesion
        self.model.particle_mu=inputs.dynamicParticleFriction 
        self.model.particle_ka=0.05
        self.model.soft_contact_margin=inputs.radius/inputs.xRef
        # self.state_init_frame = self.model.state()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.swelling_rotation=0
        self.integrator = utl.integrator_xpbd.XPBDIntegrator(iterations=inputs.interations,
        soft_body_relaxation=inputs.relaxationFactor,
        soft_contact_relaxation=inputs.relaxationFactor,
        joint_linear_relaxation=0.7,
        joint_angular_relaxation=0.4,
        rigid_contact_relaxation=0.8,
        rigid_contact_con_weighting=True,
        angular_damping=1,
        enable_restitution=inputs.restitution>0)

    def update(self,inputs):
        self.model.particle_grid.build(self.state_0.particle_q, inputs.radius/inputs.xRef*4)
        for s in range(inputs.sim_substeps):
            self.state_0.clear_forces()
            utl.collide(self.model, self.state_0)
            self.integrator.simulate(self.model, self.state_0, self.state_1, inputs.sim_dt)
                
            if inputs.swellingRatio > 0:
                # only perform swelling and radius expansion on the non dump particles
                wp.launch(kernel=fs5WarpOverrides.swellParticlesStage2, 
                        dim=self.model.particle_count, 
                        inputs=[self.model.particle_flags, self.model.particle_inv_mass, self.model.particle_radius, self.swelling_rotation, inputs.swelling_rotation_max*inputs.solverUpdates, inputs.radius/inputs.xRef], 
                        device=self.model.device)

            wp.launch(kernel=fs5WarpOverrides.sleepParticles, 
                    dim=self.model.particle_count, 
                    inputs=[inputs.sleepThreshold, self.model.particle_flags, self.state_0.particle_q, self.state_1.particle_q, self.state_1.particle_qd, inputs.sim_dt], 
                    device=self.model.device)
                

            (self.state_0, self.state_1) = (self.state_1, self.state_0)
            
    def render(self,inputs):#,drawn_particle_coordinates,drawbox_extents,activeFlags):
        activeInds=self.model.particle_flags.numpy()!=0
        self.renderer.begin_frame()
        colours=fs5PlotUtils.values_to_rgb(self.model.particle_radius.numpy()[activeInds],min_val=inputs.radius/100, max_val= inputs.radius*1.3)
        self.renderer.render_points(points=self.state_0.particle_q.numpy()[activeInds,:], radius=inputs.radius/inputs.xRef, name="points", colors=colours)
        self.renderer.render_mesh(points=self.mesh.mesh.points.numpy(),indices=self.mesh.mesh.indices.numpy(),name='mesh', colors=np.tile([1,1,1],(self.mesh.mesh.points.numpy().shape[0],1)),smooth_shading=True)
        self.renderer.end_frame()
        self.renderer.update_view_matrix()

    def swell(self,inputs):
        # print(f'Swelling {1/(inputs.swelling_rotation_max-self.swelling_rotation)} of eligible particles')
        # only swell non-dump particles
        wp.launch(kernel=fs5WarpOverrides.swellParticles, 
                  dim=self.model.particle_count, 
                  inputs=[self.model.particle_flags, self.state_0.particle_q, self.model.particle_inv_mass, self.model.particle_radius, 1/(inputs.swelling_rotation_max-self.swelling_rotation), inputs.swelling_rotation_max, inputs.swellingActivationFactor * inputs.radius / inputs.xRef, inputs.jitter, inputs.yShift, inputs.swellingActivationLocationRatio], 
                  device=self.model.device)
        self.swelling_rotation=self.swelling_rotation+1
        if self.swelling_rotation>=inputs.swelling_rotation_max:
            self.swelling_rotation=0

def read_ply(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    if lines[0].strip() != "ply":
        raise ValueError("This is not a PLY file.")
    vertex_count = int(lines[[i for i, line in enumerate(lines) if "element vertex" in line][0]].split()[2])
    face_count = int(lines[[i for i, line in enumerate(lines) if "element face" in line][0]].split()[2])
    start_index = lines.index("end_header\n") + 1
    nodes = np.array([list(map(float, line.strip().split())) for line in lines[start_index:start_index+vertex_count]])
    face_start_index = start_index + vertex_count
    elements = []
    for line in lines[face_start_index:face_start_index+face_count]:
        face_data = list(map(int, line.strip().split()))
        elements.append(face_data[1:])
    elements = np.array(elements)
    return nodes, elements

def generate_displacements_np(r, n):
    # Generate random radii, dip angles, and azimuth angles
    random_radii = np.random.uniform(0, r, n)
    dip_angles = np.radians(np.random.uniform(0, 180, n))
    azimuth_angles = np.radians(np.random.uniform(0, 360, n))
    
    # Convert spherical coordinates to Cartesian coordinates
    x = random_radii * np.sin(dip_angles) * np.cos(azimuth_angles)
    y = random_radii * np.sin(dip_angles) * np.sin(azimuth_angles)
    z = random_radii * np.cos(dip_angles)
    
    return np.column_stack((x, y, z))

def generate_grid(mesh, particle_radius, jitter):
    # Calculate bounds of the mesh
    bounds = mesh.bounds
    min_corner, max_corner = bounds[0], bounds[1]
    
    # Define the grid spacing as twice the radius of particles
    spacing = 2 * particle_radius
    
    # Generate grid points
    grid_x = np.arange(min_corner[0], max_corner[0], spacing)
    grid_y = np.arange(min_corner[1], max_corner[1], spacing)
    grid_z = np.arange(min_corner[2], max_corner[2], spacing)
    # Create a meshgrid
    x, y, z = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
    grid_points = np.vstack([x.flatten(),y.flatten(),z.flatten()]).T
    if jitter > 0:
        jitter = (particle_radius) * jitter 
        _j = generate_displacements_np(jitter, len(grid_points))
    grid_points = grid_points +_j
    return grid_points

def fill_mesh_with_particles(Vp, Fp, radius, jitter):
    # Create a trimesh object for the particle mesh
    particle_mesh = trimesh.Trimesh(vertices=Vp, faces=Fp)
    
    # Generate grid points based on the particle mesh bounds and radius
    grid_points = generate_grid(particle_mesh, radius, jitter)
    
    # Check which points are inside the mesh
    # inside_points = particle_mesh.contains(grid_points)
    inside_points = np.ones(grid_points.shape[0],dtype=bool)#particle_mesh.contains(grid_points)

    # Filter points that are inside the mesh
    inside_positions = grid_points[inside_points]
    
    return inside_positions, grid_points.shape[0]

if __name__ == '__main__':
    inputs = fs5InputReader.fs5args()
    print(inputs)  # Proceed with using the args in your application

    os.environ["CUDA_VISIBLE_DEVICES"] = inputs.gpuIDs

    # load in the ply meshes

    Vs, Fs = read_ply(inputs.surfaceMesh)
    Vp, Fp = read_ply(inputs.particleMesh)

    Vs=Vs[:,[0,2,1]]

    Vp=Vp[:,[0,2,1]]
    # scale if needed




    # create the particle set
    positions, nbox = fill_mesh_with_particles(Vp, Fp, inputs.radius, inputs.jitter)
    print(f"Number of initial particles: {positions.shape[0]}/{nbox}")
    # print("Positions of particles inside the mesh:", positions)

    # create the swelling dataset
    if inputs.swellingRatio > 0:
        total_particles_to_add = int((len(positions)+1) * inputs.swellingRatio)
        swell_particles = np.sort(np.random.choice(len(positions)-1, total_particles_to_add, replace=False)) + 1
        swell_particles += np.arange(len(swell_particles))


        out_dim = len(positions)+len(swell_particles)
        delta_a = np.ones(out_dim, dtype=bool)
        delta_a[swell_particles] = 0
        delta_particles = np.zeros((out_dim, 3))
        delta_particles[delta_a] = positions
        delta_particles[swell_particles] = delta_particles[swell_particles-1]
    else:
        delta_a = np.ones(positions.shape[0], dtype=bool)
        delta_particles=positions

    # set up the warp simulation
    wp.config.kernel_cache_dir = os.path.join("tmp", f"warpcache")

    wp.init()
    wp.rand_init(42)
    simState = simulationStage(inputs,Vs,Fs.flatten(),delta_particles,delta_a)

    for pp_step in range(inputs.max_steps_per_stage//inputs.solverUpdates):
        # simState.state_init_frame=simState.state_0
        print(f'GPU Compute...frame {pp_step}')
        # simulate
        for s in range(inputs.solverUpdates):
            simState.update(inputs)

            if np.mod(s,inputs.framesPerRuntimeRender)==0:
                if inputs.runtimeRendering:
                    simState.render(inputs)
            if pp_step==0 and s==0:
                simState.state_0.particle_qd.zero_()
                simState.state_1.particle_qd.zero_()

        if inputs.swellingRatio > 0:
            simState.swell(inputs)

        # save particle positions and masses 