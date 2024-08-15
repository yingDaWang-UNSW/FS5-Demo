import os
import cv2
import numpy as np
import warp as wp
import warp.sim
import warp.sim.render
import warp.render
import time
from utils import fs4IOUtils
from utils import fs5PlotUtils
from utils import fs5VoxelParticleUtils
from oldStuff import oldFS5WarpKernels
from utils import Drawzones
import math
import pdb
from multiprocessing import Pool
from pyglet import gl
import pyglet
def renderSingleMineStage(inputs, minePeriod, renderer, out, override=[]):
    meshFileName = fs4IOUtils.findFileName(inputs.meshFilename,minePeriod)
    print(f'Loading mesh {meshFileName}')
    meshPoints, meshIndices = fs4IOUtils.loadMesh(meshFileName, inputs.VTKLowerBound, inputs.xRef, inputs.yShift, invert=False)
    if minePeriod==inputs.initialMinePeriod:
        renderer=fs5PlotUtils.look_at_centroid(meshPoints,renderer,renderer.camera_fov)
    if len(override)==0:
        print(f'Loading particles for mine period {minePeriod}')
        particlesFileName = fs4IOUtils.findFileName(inputs.particlesBinFilename,minePeriod)
        FS4_particles = fs4IOUtils.loadFS4ParticlesSimplified(particlesFileName)
        assert len(FS4_particles)>0
        nFrames=len(FS4_particles)
    else:
        nFrames=60
        activeParticles=override    
    drawzone_voxels, quota, drawn_particle_coordinates = Drawzones.generate_drawzones(minePeriod, inputs)
    for f in range(nFrames):
        if len(override)==0:
            activeParticles=FS4_particles[f][np.where(FS4_particles[f][:,3]>0),0:4]
            activeParticles=np.squeeze(activeParticles)          
        renderer.begin_frame()
        renderer.render_points(points=activeParticles[:,0:3], radius=inputs.radius/inputs.xRef, name="points", colors=activeParticles[:,0:3]/activeParticles[:,0:3]*np.array((0.8500, 0.3250, 0.0980)))
        """ body_coordinates = state_1.body_q.numpy()
        position = body_coordinates[0][:3]
        quaternion = body_coordinates[0][3:] """
        renderer.render_mesh(points=meshPoints,indices=meshIndices,name='mesh')
        """ text = f"Sim Time: {renderer.time:.1f}, Update FPS: {renderer._fps_update:.1f}, Render FPS: {renderer._fps_render:.1f}, Mine Period: {minePeriod}, Frame: {f}, numParticles: {activeParticles.shape[0]}, Position: {renderer._camera_pos}, Front: {renderer._camera_front}, Yaw: {renderer._yaw}, Pitch: {renderer._pitch}"
        renderer.info.text = text
        renderer.info.y = renderer.screen_height - 5
        renderer.info.draw() """
        renderer.end_frame()
        renderer.update_view_matrix()
        target_image=wp.array(np.zeros((renderer.screen_height, renderer.screen_width, 3)),shape=(renderer.screen_height, renderer.screen_width, 3),dtype=wp.float32)
        renderer.get_pixels(target_image, split_up_tiles=0)
        frame = np.uint8(target_image.numpy()*255)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        out.write(frame)
    renderer.clear()
def renderFullMineStages(inputs):
    wp.init()
    print('Post-Runtime rendering is on')
    renderer = wp.render.OpenGLRenderer(        
        title="Warp sim",
        scaling=1.0,
        fps=60,
        up_axis="y",
        screen_width=1024,
        screen_height=768,
        near_plane=0.01,
        far_plane=10000/inputs.xRef,
        camera_fov=75.0,
        background_color=(0.,0.,0.),
        draw_grid=False,
        draw_sky=False,
        draw_axis=True,
        show_info=False,
        render_wireframe=False,
        axis_scale=1.0,
        vsync=False,
        headless=False,
        enable_backface_culling=True)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 60.0, (renderer.screen_width, renderer.screen_height))
    renderer._paused=True
    renderer._camera_speed = 5/inputs.xRef
    if inputs.renderDrawEllipses:
        finalFrames=[]
        finalDrawInds=[]
        for minePeriod in np.arange(inputs.initialMinePeriod, inputs.finalMinePeriod+1,1):
            print(f'Loading particles for mine period {minePeriod}')
            particlesFileName = fs4IOUtils.findFileName(inputs.particlesBinFilename,minePeriod)
            FS4_particles = fs4IOUtils.loadFS4ParticlesSimplified(particlesFileName)
            finalFrameParticles=FS4_particles[-1]
            finalFrames.append(finalFrameParticles)
            finalInactiveParticleInds = np.where(finalFrameParticles[:, 3] == 0)[0]
            finalInactiveParticles=finalFrameParticles[finalInactiveParticleInds,:]
            drawzone_voxels, quota, drawn_particle_coordinates = Drawzones.generate_drawzones(minePeriod, inputs)
            if drawn_particle_coordinates!=0:
                unique_coords = []
                if isinstance(drawn_particle_coordinates, dict):
                    for coord in drawn_particle_coordinates.values():
                        unique_coords.append(np.round(coord,0))
                unique_coords=np.asarray(unique_coords)
                unique_coord_set = set(map(tuple, unique_coords))
                matching_particles = []
                for i, particle in enumerate(finalInactiveParticles[:, 0:3]):
                    rounded_particle = tuple(np.round(particle, 0))
                    if rounded_particle in unique_coord_set:
                        matching_particles.append(i)  
                matching_particles = np.asarray(matching_particles)
                if len(matching_particles)>0:
                    drawnParticleInds=finalInactiveParticleInds[matching_particles]
                    drawnParticlePositions=finalInactiveParticles[matching_particles]
                    i=0
                    for particleInd in drawnParticleInds:
                        for frameIndex in range(len(finalFrames)):
                            frame = finalFrames[frameIndex]
                            if particleInd < frame.shape[0]:
                                drawnParticlePositions[i,:3] = frame[particleInd, :3]  
                                break
                        i=i+1
                    renderSingleMineStage(inputs, minePeriod, renderer, out, override=drawnParticlePositions)
    else:
        for minePeriod in range(inputs.initialMinePeriod, inputs.finalMinePeriod+1):
            renderSingleMineStage(inputs, minePeriod, renderer, out, override=[])
        out.release
    return inputs