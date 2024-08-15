import warp as wp
import numpy as np
import pdb
import utils as utl
import time
from utils import fs5InputReader
from utils import fs5Model
from utils.fs5Model import (
    PARTICLE_FLAG_ACTIVE,
    ModelShapeMaterials,
    ModelShapeGeometry
)

from utils.collide import broadphase_collision_pairs,handle_contact_pairs
from utils.integrator_xpbd import apply_joint_torques,solve_springs,solve_tetrahedra,solve_body_joints,apply_body_delta_velocities,apply_rigid_restitution,apply_soft_restitution_ground,update_body_velocities,apply_body_deltas
from utils.integrator_euler import integrate_bodies

args=fs5InputReader.fs5args()
inputs = args
# print('OVERRIDING WARP')
staticGroundFriction=inputs.staticSurfaceFriction
staticGroundVelocityThresholdRatio=inputs.staticGroundVelocityThresholdRatio
staticParticleFriction=inputs.staticParticleFriction
staticParticleVelocityThresholdRatio=inputs.staticParticleVelocityThresholdRatio
meshSidedFlag=float(inputs.meshSidedFlag)

@wp.kernel
def sleepParticles(
    sleepThreshold: float,
    particle_flags: wp.array(dtype=wp.uint32),
    particle_positions_init: wp.array(dtype=wp.vec3),
    particle_positions_after: wp.array(dtype=wp.vec3),
    particle_velocities_after: wp.array(dtype=wp.vec3),
    dt: float
):
    tid = wp.tid()
    if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:
        return
    # if wp.length(particle_velocities_after[tid])<sleepThreshold:
    if wp.length(particle_positions_after[tid]-particle_positions_init[tid])/dt<sleepThreshold:
        particle_positions_after[tid]=particle_positions_init[tid]

@wp.kernel
def swellParticles(
    particle_flags: wp.array(dtype=wp.uint32),
    particle_position: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    chance: float,
    swelling_rotation_max: float,
    swelling_activation_factor: float,
    jitterFactor: float,
    yShift: float,
    swellingActivationLocationRatio: float
):
    tid = wp.tid()
    i=wp.uint32(tid)
    flag = particle_flags[tid]
    position=particle_position[tid]
    y=position[1]
    if y>yShift and flag==wp.uint32(1): # if particle is active
        trackedID=tid+1
        if trackedID<particle_flags.shape[0]:
            if particle_flags[trackedID]==wp.uint32(0): # check if it has a swelling particle
                trackedPosition=particle_position[trackedID]
                trackedy=trackedPosition[1]
                dx=trackedPosition - position
                dist=wp.length(dx)
                if trackedy>yShift and dist>=swelling_activation_factor and wp.randf(i)<chance: # if the swelling particle is eligible, swell it
                    mass=particle_inv_mass[tid]
                    radius=particle_radius[tid] 

                    newPosition=wp.vec3()
                    r = wp.randf(i+wp.uint32(1))*radius*jitterFactor
                    theta = wp.randf(i+wp.uint32(2))*3.14
                    phi = wp.randf(i+wp.uint32(3))*6.28
                    jitterX = r * wp.sin(theta) * wp.cos(phi)
                    jitterY = r * wp.sin(theta) * wp.sin(phi)
                    jitterZ = r * wp.cos(theta)
                    phii=wp.float32(swellingActivationLocationRatio)
                    
                    newPosition[0]=(position[0]*(1.0-phii)+trackedPosition[0]*phii) + jitterX
                    newPosition[1]=(position[1]*(1.0-phii)+trackedPosition[1]*phii) + jitterY
                    newPosition[2]=(position[2]*(1.0-phii)+trackedPosition[2]*phii) + jitterZ

                    particle_flags[trackedID]=wp.uint32(1)
                    particle_position[trackedID]=newPosition
                    particle_inv_mass[trackedID]=mass#*10000.0
                    particle_radius[trackedID]=radius/swelling_rotation_max/25.0


@wp.kernel
def swellParticlesStage2( 
    particle_flags: wp.array(dtype=wp.uint32),
    particle_inv_mass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    swelling_rotation: float,
    swelling_rotation_max: float,
    fullRadius: float,
):
    tid = wp.tid()
    if particle_flags[tid]==wp.uint32(1) and particle_radius[tid]<fullRadius:
        particle_radius[tid]=particle_radius[tid]+fullRadius/swelling_rotation_max/25.0
    # if particle_flags[tid]==wp.uint32(1) and 1.0/particle_inv_mass[tid]<10000.0:
    #     particle_inv_mass[tid]=1.0/((1.0/(particle_inv_mass[tid]))+10000.0/swelling_rotation_max/25.0)


@wp.kernel
def my_integrate_particles(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
    w: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.uint32),
    gravity: wp.vec3,
    dt: float,
    v_max: float,
    x_new: wp.array(dtype=wp.vec3),
    v_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:
        return
    x0 = x[tid]
    v0 = v[tid]
    f0 = f[tid]
    inv_mass = w[tid]
    v1 = v0 + (f0 * inv_mass + gravity * wp.step(0.0 - inv_mass)) * dt
    v1_mag = wp.length(v1)
    if v1_mag > v_max:
        v1 *= v_max / v1_mag
    x1 = x0 + v1 * dt
    x_new[tid] = x1
    v_new[tid] = v1
utl.integrator_euler.integrate_particles = my_integrate_particles
@wp.kernel
def my_apply_particle_deltas(
    x_orig: wp.array(dtype=wp.vec3),
    x_pred: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.uint32),
    delta: wp.array(dtype=wp.vec3),
    dt: float,
    v_max: float,
    x_out: wp.array(dtype=wp.vec3),
    v_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:
        return
    x0 = x_orig[tid]
    xp = x_pred[tid]
    d = delta[tid]
    if wp.isnan(d):
        d[0]=0.0
        d[1]=0.0
        d[2]=0.0
        
    x_new = xp + d
    v_new = (x_new - x0) / dt
    v_new_mag = wp.length(v_new)
    if v_new_mag > v_max:
        v_new *= v_max / v_new_mag
    x_out[tid] = x_new
    v_out[tid] = v_new
utl.integrator_xpbd.apply_particle_deltas = my_apply_particle_deltas
@wp.kernel
def my_solve_particle_ground_contacts(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    invmass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.uint32),
    ke: float,
    kd: float,
    kf: float,
    mu: float,
    staticGroundVelocityThresholdRatio: float,
    staticGroundFriction: float,
    ground: wp.array(dtype=float),
    dt: float,
    relaxation: float,
    delta: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:
        return
    wi = invmass[tid]
    if wi == 0.0:
        return
    radius = particle_radius[tid]
    x = particle_x[tid]
    v = particle_v[tid]
    n = wp.vec3(ground[0], ground[1], ground[2])
    c = wp.min(wp.dot(n, x) + ground[3] - particle_radius[tid], 0.0)
    if c > 0.0:
        return
    lambda_n = c
    delta_n = n * lambda_n
    vn = wp.dot(n, v)
    if wp.abs(vn)<staticGroundVelocityThresholdRatio:
        mu=staticGroundFriction
    vt = v - n * vn
    lambda_f = wp.max(mu * lambda_n, 0.0 - wp.length(vt) * dt)
    delta_f = wp.normalize(vt) * lambda_f
    wp.atomic_add(delta, tid, (delta_f - delta_n) / wi * relaxation * wi)
utl.integrator_xpbd.solve_particle_ground_contacts = my_solve_particle_ground_contacts
@wp.kernel
def my_solve_particle_particle_contacts(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_invmass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.uint32),
    k_mu: float,
    staticParticleVelocityThresholdRatio: float,
    staticParticleFriction: float,
    k_cohesion: float,
    max_radius: float,
    dt: float,
    relaxation: float,
    deltas: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        return
    if (particle_flags[i] & PARTICLE_FLAG_ACTIVE) == 0:
        return
    x = particle_x[i]
    v = particle_v[i]
    radius = particle_radius[i]
    w1 = particle_invmass[i]
    query = wp.hash_grid_query(grid, x, radius + max_radius + k_cohesion)
    index = int(0)
    delta = wp.vec3(0.0)
    while wp.hash_grid_query_next(query, index):
        if (particle_flags[index] & PARTICLE_FLAG_ACTIVE) != 0 and index != i:
            n = x - particle_x[index]
            d = wp.length(n)
            err = d - radius - particle_radius[index]
            w2 = particle_invmass[index]
            denom = w1 + w2
            if err <= k_cohesion and denom > 0.0:
                n = n / d
                vrel = v - particle_v[index]
                lambda_n = err
                delta_n = n * lambda_n
                vn = wp.dot(n, vrel)
                if wp.abs(vn)<staticParticleVelocityThresholdRatio:
                    k_mu=staticParticleFriction
                vt = v - n * vn
                lambda_f = wp.max(k_mu * lambda_n, -wp.length(vt) * dt)
                delta_f = wp.normalize(vt) * lambda_f
                delta += (delta_f - delta_n) / denom * w1
    wp.atomic_add(deltas, i, delta * relaxation)
utl.integrator_xpbd.solve_particle_particle_contacts = my_solve_particle_particle_contacts
@wp.kernel
def my_solve_particle_shape_contacts(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_invmass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.uint32),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_m_inv: wp.array(dtype=float),
    body_I_inv: wp.array(dtype=wp.mat33),
    shape_body: wp.array(dtype=int),
    shape_materials: ModelShapeMaterials,
    particle_mu: float,
    particle_ka: float,
    staticGroundVelocityThresholdRatio: float,
    staticGroundFriction: float,
    contact_count: wp.array(dtype=int),
    contact_particle: wp.array(dtype=int),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_max: int,
    dt: float,
    relaxation: float,
    delta: wp.array(dtype=wp.vec3),
    body_delta: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    count = min(contact_max, contact_count[0])
    if tid >= count:
        return
    shape_index = contact_shape[tid]
    body_index = shape_body[shape_index]
    particle_index = contact_particle[tid]
    if (particle_flags[particle_index] & PARTICLE_FLAG_ACTIVE) == 0:
       return
    px = particle_x[particle_index]
    pv = particle_v[particle_index]
    X_wb = wp.transform_identity()
    X_com = wp.vec3()
    if body_index >= 0:
        X_wb = body_q[body_index]
        X_com = body_com[body_index]
    bx = wp.transform_point(X_wb, contact_body_pos[tid])
    r = bx - wp.transform_point(X_wb, X_com)
    n = contact_normal[tid]
    c = wp.dot(n, px - bx) - particle_radius[particle_index]
    if c > particle_ka:
        return
    mu = 0.5 * (particle_mu + shape_materials.mu[shape_index])
    body_v_s = wp.spatial_vector()
    if body_index >= 0:
        body_v_s = body_qd[body_index]
    body_w = wp.spatial_top(body_v_s)
    body_v = wp.spatial_bottom(body_v_s)
    bv = body_v + wp.cross(body_w, r) + wp.transform_vector(X_wb, contact_body_vel[tid])
    v = pv - bv
    lambda_n = c
    delta_n = n * lambda_n
    vn = wp.dot(n, v)
    vt = v - n * vn
    if wp.abs(vn)<staticGroundVelocityThresholdRatio:
        mu=staticGroundFriction
    w1 = particle_invmass[particle_index]
    w2 = particle_invmass[particle_index]*wp.float(0.0)
    if body_index >= 0:
        angular = wp.cross(r, n)
        q = wp.transform_get_rotation(X_wb)
        rot_angular = wp.quat_rotate_inv(q, angular)
        I_inv = body_I_inv[body_index]
        w2 = body_m_inv[body_index] + wp.dot(rot_angular, I_inv * rot_angular)
    denom = w1 + w2
    if denom == 0.0:
        return
    lambda_f = wp.max(mu * lambda_n, -wp.length(vt) * dt)
    delta_f = wp.normalize(vt) * lambda_f
    delta_total = (delta_f - delta_n) / denom * relaxation * w1
    wp.atomic_add(delta, particle_index, delta_total)
    if body_index >= 0:
        delta_t = wp.cross(r, delta_total)
        wp.atomic_sub(body_delta, body_index, wp.spatial_vector(delta_t, delta_total))
utl.integrator_xpbd.solve_particle_shape_contacts = my_solve_particle_shape_contacts
@wp.kernel
def my_create_soft_contacts(
    particle_x: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.uint32),
    body_X_wb: wp.array(dtype=wp.transform),
    shape_X_bs: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    geo: ModelShapeGeometry,
    margin: float,
    soft_contact_max: int,
    soft_contact_count: wp.array(dtype=int),
    soft_contact_particle: wp.array(dtype=int),
    soft_contact_shape: wp.array(dtype=int),
    soft_contact_body_pos: wp.array(dtype=wp.vec3),
    soft_contact_body_vel: wp.array(dtype=wp.vec3),
    soft_contact_normal: wp.array(dtype=wp.vec3),
    meshSidedFlag: float,
):
    particle_index, shape_index = wp.tid()
    if (particle_flags[particle_index] & PARTICLE_FLAG_ACTIVE) == 0:
        return
    rigid_index = shape_body[shape_index]
    px = particle_x[particle_index]
    radius = particle_radius[particle_index]
    X_wb = wp.transform_identity()
    if rigid_index >= 0:
        X_wb = body_X_wb[rigid_index]
    X_bs = shape_X_bs[shape_index]
    X_ws = wp.transform_multiply(X_wb, X_bs)
    X_sw = wp.transform_inverse(X_ws)
    x_local = wp.transform_point(X_sw, px)
    geo_type = geo.type[shape_index]
    geo_scale = geo.scale[shape_index]
    d = 1.0e6
    n = wp.vec3()
    v = wp.vec3()
    if geo_type == fs5Model.GEO_MESH:
        mesh = geo.source[shape_index]
        face_index = int(0)
        face_u = float(0.0)
        face_v = float(0.0)
        sign = float(0.0)
        if wp.mesh_query_point_sign_normal(
            mesh, wp.cw_div(x_local, geo_scale), margin + radius, sign, face_index, face_u, face_v
        ):
            shape_p = wp.mesh_eval_position(mesh, face_index, face_u, face_v)
            shape_v = wp.mesh_eval_velocity(mesh, face_index, face_u, face_v)
            shape_p = wp.cw_mul(shape_p, geo_scale)
            shape_v = wp.cw_mul(shape_v, geo_scale)
            delta = x_local - shape_p
            if meshSidedFlag == 0.0:
                sign=1.0
            d = wp.length(delta) * sign
            n = wp.normalize(delta) * sign 
            v = shape_v
    if d < margin + radius:
        index = wp.atomic_add(soft_contact_count, 0, 1)
        # if index < soft_contact_max:
        body_pos = wp.transform_point(X_bs, x_local - n * d)
        body_vel = wp.transform_vector(X_bs, v)
        world_normal = wp.transform_vector(X_ws, n)
        soft_contact_shape[index] = shape_index
        soft_contact_body_pos[index] = body_pos
        soft_contact_body_vel[index] = body_vel
        soft_contact_particle[index] = particle_index
        soft_contact_normal[index] = world_normal
utl.collide.create_soft_contacts=my_create_soft_contacts
def my_collide(model, state, edge_sdf_iter: int = 10):
    """
    Generates contact points for the particles and rigid bodies in the model,
    to be used in the contact dynamics kernel of the integrator.
    Args:
        model: the model to be simulated
        state: the state of the model
        edge_sdf_iter: number of search iterations for finding closest contact points between edges and SDF
    """
    if model.particle_count and model.shape_count > 1:
        model.soft_contact_count.zero_()
        wp.launch(
            kernel=my_create_soft_contacts,
            dim=(model.particle_count, model.shape_count - 1),
            inputs=[
                state.particle_q,
                model.particle_radius,
                model.particle_flags,
                state.body_q,
                model.shape_transform,
                model.shape_body,
                model.shape_geo,
                model.soft_contact_margin,
                model.soft_contact_max,
            ],
            outputs=[
                model.soft_contact_count,
                model.soft_contact_particle,
                model.soft_contact_shape,
                model.soft_contact_body_pos,
                model.soft_contact_body_vel,
                model.soft_contact_normal,
                meshSidedFlag,
            ],
            device=model.device,
        )
    model.rigid_contact_count.zero_()
    if model.shape_contact_pair_count:
        wp.launch(
            kernel=broadphase_collision_pairs,
            dim=model.shape_contact_pair_count,
            inputs=[
                model.shape_contact_pairs,
                state.body_q,
                model.shape_transform,
                model.shape_body,
                model.shape_geo,
                model.shape_collision_radius,
                model.rigid_contact_max,
                model.rigid_contact_margin,
            ],
            outputs=[
                model.rigid_contact_count,
                model.rigid_contact_shape0,
                model.rigid_contact_shape1,
                model.rigid_contact_point_id,
            ],
            device=model.device,
            record_tape=False,
        )
    if model.ground and model.shape_ground_contact_pair_count:
        wp.launch(
            kernel=broadphase_collision_pairs,
            dim=model.shape_ground_contact_pair_count,
            inputs=[
                model.shape_ground_contact_pairs,
                state.body_q,
                model.shape_transform,
                model.shape_body,
                model.shape_geo,
                model.shape_collision_radius,
                model.rigid_contact_max,
                model.rigid_contact_margin,
            ],
            outputs=[
                model.rigid_contact_count,
                model.rigid_contact_shape0,
                model.rigid_contact_shape1,
                model.rigid_contact_point_id,
            ],
            device=model.device,
            record_tape=False,
        )
    if model.shape_contact_pair_count or model.ground and model.shape_ground_contact_pair_count:
        wp.launch(
            kernel=handle_contact_pairs,
            dim=model.rigid_contact_max,
            inputs=[
                state.body_q,
                model.shape_transform,
                model.shape_body,
                model.shape_geo,
                model.rigid_contact_margin,
                model.body_com,
                model.rigid_contact_shape0,
                model.rigid_contact_shape1,
                model.rigid_contact_point_id,
                model.rigid_contact_count,
                edge_sdf_iter,
            ],
            outputs=[
                model.rigid_contact_body0,
                model.rigid_contact_body1,
                model.rigid_contact_point0,
                model.rigid_contact_point1,
                model.rigid_contact_offset0,
                model.rigid_contact_offset1,
                model.rigid_contact_normal,
                model.rigid_contact_thickness,
            ],
            device=model.device,
        )
utl.collide = my_collide

def my_simulate(self, model, state_in, state_out, dt, requires_grad=False):
    with wp.ScopedTimer("simulate", False):
        particle_q = None
        particle_qd = None
        if model.particle_count:
            if requires_grad:
                particle_q = wp.zeros_like(state_in.particle_q)
                particle_qd = wp.zeros_like(state_in.particle_qd)
            else:
                particle_q = state_out.particle_q
                particle_qd = state_out.particle_qd
            wp.launch(
                kernel=my_integrate_particles,
                dim=model.particle_count,
                inputs=[
                    state_in.particle_q,
                    state_in.particle_qd,
                    state_in.particle_f,
                    model.particle_inv_mass,
                    model.particle_flags,
                    model.gravity,
                    dt,
                    model.particle_max_velocity,
                ],
                outputs=[particle_q, particle_qd],
                device=model.device,
            )
        if model.body_count:
            if model.joint_count:
                wp.launch(
                    kernel=apply_joint_torques,
                    dim=model.joint_count,
                    inputs=[
                        state_in.body_q,
                        model.body_com,
                        model.joint_q_start,
                        model.joint_qd_start,
                        model.joint_type,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_X_p,
                        model.joint_X_c,
                        model.joint_axis_start,
                        model.joint_axis_dim,
                        model.joint_axis,
                        model.joint_act,
                    ],
                    outputs=[state_in.body_f],
                    device=model.device,
                )
            wp.launch(
                kernel=integrate_bodies,
                dim=model.body_count,
                inputs=[
                    state_in.body_q,
                    state_in.body_qd,
                    state_in.body_f,
                    model.body_com,
                    model.body_mass,
                    model.body_inertia,
                    model.body_inv_mass,
                    model.body_inv_inertia,
                    model.gravity,
                    self.angular_damping,
                    dt,
                ],
                outputs=[state_out.body_q, state_out.body_qd],
                device=model.device,
            )
        if model.spring_count:
            model.spring_constraint_lambdas.zero_()
        if model.edge_count:
            model.edge_constraint_lambdas.zero_()
        for i in range(self.iterations):
            if model.body_count:
                if requires_grad:
                    out_body_q = wp.clone(state_out.body_q)
                    out_body_qd = wp.clone(state_out.body_qd)
                    state_out.body_deltas = wp.zeros_like(state_out.body_deltas)
                else:
                    out_body_q = state_out.body_q
                    out_body_qd = state_out.body_qd
                    state_out.body_deltas.zero_()
            else:
                out_body_q = None
                out_body_qd = None
            if model.particle_count:
                if requires_grad:
                    deltas = wp.zeros_like(state_out.particle_f)
                else:
                    deltas = state_out.particle_f
                    deltas.zero_()
                if model.ground:
                    wp.launch(
                        kernel=my_solve_particle_ground_contacts,
                        dim=model.particle_count,
                        inputs=[
                            particle_q,
                            particle_qd,
                            model.particle_inv_mass,
                            model.particle_radius,
                            model.particle_flags,
                            model.soft_contact_ke,
                            model.soft_contact_kd,
                            model.soft_contact_kf,
                            model.soft_contact_mu,
                            staticGroundVelocityThresholdRatio,
                            staticGroundFriction,
                            model.ground_plane,
                            dt,
                            self.soft_contact_relaxation,
                        ],
                        outputs=[deltas],
                        device=model.device,
                    )
                if model.shape_count > 1:
                    wp.launch(
                        kernel=my_solve_particle_shape_contacts,
                        dim=model.soft_contact_max,
                        inputs=[
                            particle_q,
                            particle_qd,
                            model.particle_inv_mass,
                            model.particle_radius,
                            model.particle_flags,
                            out_body_q,
                            out_body_qd,
                            model.body_com,
                            model.body_inv_mass,
                            model.body_inv_inertia,
                            model.shape_body,
                            model.shape_materials,
                            model.soft_contact_mu,
                            model.particle_adhesion,
                            staticGroundVelocityThresholdRatio,
                            staticGroundFriction,
                            model.soft_contact_count,
                            model.soft_contact_particle,
                            model.soft_contact_shape,
                            model.soft_contact_body_pos,
                            model.soft_contact_body_vel,
                            model.soft_contact_normal,
                            model.soft_contact_max,
                            dt,
                            self.soft_contact_relaxation,
                        ],
                        outputs=[deltas, state_out.body_deltas],
                        device=model.device,
                    )
                if model.particle_max_radius > 0.0:
                    wp.launch(
                        kernel=my_solve_particle_particle_contacts,
                        dim=model.particle_count,
                        inputs=[
                            model.particle_grid.id,
                            particle_q,
                            particle_qd,
                            model.particle_inv_mass,
                            model.particle_radius,
                            model.particle_flags,
                            model.particle_mu,
                            staticParticleVelocityThresholdRatio,
                            staticParticleFriction,
                            model.particle_cohesion,
                            model.particle_max_radius,
                            dt,
                            self.soft_contact_relaxation,
                        ],
                        outputs=[deltas],
                        device=model.device,
                    )
                if model.spring_count:
                    wp.launch(
                        kernel=solve_springs,
                        dim=model.spring_count,
                        inputs=[
                            particle_q,
                            particle_qd,
                            model.particle_inv_mass,
                            model.spring_indices,
                            model.spring_rest_length,
                            model.spring_stiffness,
                            model.spring_damping,
                            dt,
                            model.spring_constraint_lambdas,
                        ],
                        outputs=[deltas],
                        device=model.device,
                    )
                if model.tet_count:
                    wp.launch(
                        kernel=solve_tetrahedra,
                        dim=model.tet_count,
                        inputs=[
                            particle_q,
                            particle_qd,
                            model.particle_inv_mass,
                            model.tet_indices,
                            model.tet_poses,
                            model.tet_activations,
                            model.tet_materials,
                            dt,
                            self.soft_body_relaxation,
                        ],
                        outputs=[deltas],
                        device=model.device,
                    )
                if requires_grad:
                    new_particle_q = wp.clone(particle_q)
                    new_particle_qd = wp.clone(particle_qd)
                else:
                    new_particle_q = particle_q
                    new_particle_qd = particle_qd
                wp.launch(
                    kernel=my_apply_particle_deltas,
                    dim=model.particle_count,
                    inputs=[
                        state_in.particle_q,
                        particle_q,
                        model.particle_flags,
                        deltas,
                        dt,
                        model.particle_max_velocity,
                    ],
                    outputs=[new_particle_q, new_particle_qd],
                    device=model.device,
                )
                if requires_grad:
                    particle_q.assign(new_particle_q)
                    particle_qd.assign(new_particle_qd)
                else:
                    particle_q = new_particle_q
                    particle_qd = new_particle_qd
            if model.joint_count:
                wp.launch(
                    kernel=solve_body_joints,
                    dim=model.joint_count,
                    inputs=[
                        state_out.body_q,
                        state_out.body_qd,
                        model.body_com,
                        model.body_inv_mass,
                        model.body_inv_inertia,
                        model.joint_type,
                        model.joint_enabled,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_X_p,
                        model.joint_X_c,
                        model.joint_limit_lower,
                        model.joint_limit_upper,
                        model.joint_axis_start,
                        model.joint_axis_dim,
                        model.joint_axis_mode,
                        model.joint_axis,
                        model.joint_target,
                        model.joint_target_ke,
                        model.joint_target_kd,
                        model.joint_linear_compliance,
                        model.joint_angular_compliance,
                        self.joint_angular_relaxation,
                        self.joint_linear_relaxation,
                        dt,
                    ],
                    outputs=[state_out.body_deltas],
                    device=model.device,
                )
                wp.launch(
                    kernel=apply_body_deltas,
                    dim=model.body_count,
                    inputs=[
                        state_out.body_q,
                        state_out.body_qd,
                        model.body_com,
                        model.body_inertia,
                        model.body_inv_mass,
                        model.body_inv_inertia,
                        state_out.body_deltas,
                        None,
                        dt,
                    ],
                    outputs=[
                        out_body_q,
                        out_body_qd,
                    ],
                    device=model.device,
                )
            if model.body_count and requires_grad:
                state_out.body_q.assign(out_body_q)
                state_out.body_qd.assign(out_body_qd)
            if model.rigid_contact_max and (
                model.ground and model.shape_ground_contact_pair_count or model.shape_contact_pair_count
            ):
                rigid_contact_inv_weight = None
                if requires_grad:
                    body_deltas = wp.zeros_like(state_out.body_deltas)
                    rigid_active_contact_distance = wp.zeros_like(model.rigid_active_contact_distance)
                    rigid_active_contact_point0 = wp.empty_like(
                        model.rigid_active_contact_point0, requires_grad=True
                    )
                    rigid_active_contact_point1 = wp.empty_like(
                        model.rigid_active_contact_point1, requires_grad=True
                    )
                    if self.rigid_contact_con_weighting:
                        rigid_contact_inv_weight = wp.zeros_like(model.rigid_contact_inv_weight)
                else:
                    body_deltas = state_out.body_deltas
                    body_deltas.zero_()
                    rigid_active_contact_distance = model.rigid_active_contact_distance
                    rigid_active_contact_point0 = model.rigid_active_contact_point0
                    rigid_active_contact_point1 = model.rigid_active_contact_point1
                    rigid_active_contact_distance.zero_()
                    if self.rigid_contact_con_weighting:
                        rigid_contact_inv_weight = model.rigid_contact_inv_weight
                        rigid_contact_inv_weight.zero_()
                wp.launch(
                    kernel=solve_body_contact_positions,
                    dim=model.rigid_contact_max,
                    inputs=[
                        state_out.body_q,
                        state_out.body_qd,
                        model.body_com,
                        model.body_inv_mass,
                        model.body_inv_inertia,
                        model.rigid_contact_count,
                        model.rigid_contact_body0,
                        model.rigid_contact_body1,
                        model.rigid_contact_point0,
                        model.rigid_contact_point1,
                        model.rigid_contact_offset0,
                        model.rigid_contact_offset1,
                        model.rigid_contact_normal,
                        model.rigid_contact_thickness,
                        model.rigid_contact_shape0,
                        model.rigid_contact_shape1,
                        model.shape_materials,
                        self.rigid_contact_relaxation,
                        dt,
                        model.rigid_contact_torsional_friction,
                        model.rigid_contact_rolling_friction,
                    ],
                    outputs=[
                        body_deltas,
                        rigid_active_contact_point0,
                        rigid_active_contact_point1,
                        rigid_active_contact_distance,
                        rigid_contact_inv_weight,
                    ],
                    device=model.device,
                )
                if self.enable_restitution and i == 0:
                    if requires_grad:
                        model.rigid_active_contact_distance_prev = wp.clone(rigid_active_contact_distance)
                        model.rigid_active_contact_point0_prev = wp.clone(rigid_active_contact_point0)
                        model.rigid_active_contact_point1_prev = wp.clone(rigid_active_contact_point1)
                        if self.rigid_contact_con_weighting:
                            model.rigid_contact_inv_weight_prev = wp.clone(rigid_contact_inv_weight)
                        else:
                            model.rigid_contact_inv_weight_prev = None
                    else:
                        model.rigid_active_contact_distance_prev.assign(rigid_active_contact_distance)
                        model.rigid_active_contact_point0_prev.assign(rigid_active_contact_point0)
                        model.rigid_active_contact_point1_prev.assign(rigid_active_contact_point1)
                        if self.rigid_contact_con_weighting:
                            model.rigid_contact_inv_weight_prev.assign(rigid_contact_inv_weight)
                        else:
                            model.rigid_contact_inv_weight_prev = None
                if requires_grad:
                    model.rigid_active_contact_distance = rigid_active_contact_distance
                    model.rigid_active_contact_point0 = rigid_active_contact_point0
                    model.rigid_active_contact_point1 = rigid_active_contact_point1
                    body_q = wp.clone(state_out.body_q)
                    body_qd = wp.clone(state_out.body_qd)
                else:
                    body_q = state_out.body_q
                    body_qd = state_out.body_qd
                wp.launch(
                    kernel=apply_body_deltas,
                    dim=model.body_count,
                    inputs=[
                        state_out.body_q,
                        state_out.body_qd,
                        model.body_com,
                        model.body_inertia,
                        model.body_inv_mass,
                        model.body_inv_inertia,
                        body_deltas,
                        rigid_contact_inv_weight,
                        dt,
                    ],
                    outputs=[
                        body_q,
                        body_qd,
                    ],
                    device=model.device,
                )
                if requires_grad:
                    state_out.body_q = body_q
                    state_out.body_qd = body_qd
        if model.body_count and not requires_grad:
            if requires_grad:
                out_body_qd = wp.clone(state_out.body_qd)
            else:
                out_body_qd = state_out.body_qd
            wp.launch(
                kernel=update_body_velocities,
                dim=model.body_count,
                inputs=[state_out.body_q, state_in.body_q, model.body_com, dt],
                outputs=[out_body_qd],
                device=model.device,
            )
            if requires_grad:
                state_out.body_qd.assign(out_body_qd)
        if self.enable_restitution:
            if model.particle_count:
                if requires_grad:
                    new_particle_qd = wp.clone(particle_qd)
                else:
                    new_particle_qd = particle_qd
                wp.launch(
                    kernel=apply_soft_restitution_ground,
                    dim=model.particle_count,
                    inputs=[
                        particle_q,
                        particle_qd,
                        state_in.particle_q,
                        state_in.particle_qd,
                        model.particle_inv_mass,
                        model.particle_radius,
                        model.particle_flags,
                        model.soft_contact_restitution,
                        model.ground_plane,
                        dt,
                        self.soft_contact_relaxation,
                    ],
                    outputs=[new_particle_qd],
                    device=model.device,
                )
                if requires_grad:
                    particle_qd.assign(new_particle_qd)
                else:
                    particle_qd = new_particle_qd
            if model.body_count:
                if requires_grad:
                    state_out.body_deltas = wp.zeros_like(state_out.body_deltas)
                else:
                    state_out.body_deltas.zero_()
                wp.launch(
                    kernel=apply_rigid_restitution,
                    dim=model.rigid_contact_max,
                    inputs=[
                        state_out.body_q,
                        state_out.body_qd,
                        state_in.body_q,
                        state_in.body_qd,
                        model.body_com,
                        model.body_inv_mass,
                        model.body_inv_inertia,
                        model.rigid_contact_count,
                        model.rigid_contact_body0,
                        model.rigid_contact_body1,
                        model.rigid_contact_normal,
                        model.rigid_contact_shape0,
                        model.rigid_contact_shape1,
                        model.shape_materials,
                        model.rigid_active_contact_distance_prev,
                        model.rigid_active_contact_point0_prev,
                        model.rigid_active_contact_point1_prev,
                        model.rigid_contact_inv_weight_prev,
                        model.gravity,
                        dt,
                    ],
                    outputs=[
                        state_out.body_deltas,
                    ],
                    device=model.device,
                )
                wp.launch(
                    kernel=apply_body_delta_velocities,
                    dim=model.body_count,
                    inputs=[
                        state_out.body_qd,
                        state_out.body_deltas,
                    ],
                    outputs=[state_out.body_qd],
                    device=model.device,
                )
        if model.particle_count:
            state_out.particle_q.assign(particle_q)
            state_out.particle_qd.assign(particle_qd)
        return state_out
utl.integrator_xpbd.XPBDIntegrator.simulate = my_simulate



