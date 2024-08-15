
#define WP_NO_CRT
#include "builtin.h"

// avoid namespacing of float type for casting to float type, this is to avoid wp::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)

#define builtin_tid1d() wp::tid(task_index)
#define builtin_tid2d(x, y) wp::tid(x, y, task_index, dim)
#define builtin_tid3d(x, y, z) wp::tid(x, y, z, task_index, dim)
#define builtin_tid4d(x, y, z, w) wp::tid(x, y, z, w, task_index, dim)


struct ModelShapeMaterials
{
    wp::array_t<wp::float32> ke;
    wp::array_t<wp::float32> kd;
    wp::array_t<wp::float32> kf;
    wp::array_t<wp::float32> mu;
    wp::array_t<wp::float32> restitution;


    CUDA_CALLABLE ModelShapeMaterials(wp::array_t<wp::float32> const& ke = {},
    wp::array_t<wp::float32> const& kd = {},
    wp::array_t<wp::float32> const& kf = {},
    wp::array_t<wp::float32> const& mu = {},
    wp::array_t<wp::float32> const& restitution = {})
        : ke{ke}
        , kd{kd}
        , kf{kf}
        , mu{mu}
        , restitution{restitution}

    {
    }

    CUDA_CALLABLE ModelShapeMaterials& operator += (const ModelShapeMaterials& rhs)
    {
        return *this;}

};

static CUDA_CALLABLE void adj_ModelShapeMaterials(wp::array_t<wp::float32> const&,
    wp::array_t<wp::float32> const&,
    wp::array_t<wp::float32> const&,
    wp::array_t<wp::float32> const&,
    wp::array_t<wp::float32> const&,
    wp::array_t<wp::float32> & adj_ke,
    wp::array_t<wp::float32> & adj_kd,
    wp::array_t<wp::float32> & adj_kf,
    wp::array_t<wp::float32> & adj_mu,
    wp::array_t<wp::float32> & adj_restitution,
    ModelShapeMaterials & adj_ret)
{
    adj_ke = adj_ret.ke;
    adj_kd = adj_ret.kd;
    adj_kf = adj_ret.kf;
    adj_mu = adj_ret.mu;
    adj_restitution = adj_ret.restitution;
}

CUDA_CALLABLE void adj_atomic_add(ModelShapeMaterials* p, ModelShapeMaterials t)
{
    wp::adj_atomic_add(&p->ke, t.ke);
    wp::adj_atomic_add(&p->kd, t.kd);
    wp::adj_atomic_add(&p->kf, t.kf);
    wp::adj_atomic_add(&p->mu, t.mu);
    wp::adj_atomic_add(&p->restitution, t.restitution);
}



struct ModelShapeGeometry
{
    wp::array_t<wp::int32> type;
    wp::array_t<wp::uint8> is_solid;
    wp::array_t<wp::float32> thickness;
    wp::array_t<wp::uint64> source;
    wp::array_t<wp::vec_t<3,wp::float32>> scale;


    CUDA_CALLABLE ModelShapeGeometry(wp::array_t<wp::int32> const& type = {},
    wp::array_t<wp::uint8> const& is_solid = {},
    wp::array_t<wp::float32> const& thickness = {},
    wp::array_t<wp::uint64> const& source = {},
    wp::array_t<wp::vec_t<3,wp::float32>> const& scale = {})
        : type{type}
        , is_solid{is_solid}
        , thickness{thickness}
        , source{source}
        , scale{scale}

    {
    }

    CUDA_CALLABLE ModelShapeGeometry& operator += (const ModelShapeGeometry& rhs)
    {
        return *this;}

};

static CUDA_CALLABLE void adj_ModelShapeGeometry(wp::array_t<wp::int32> const&,
    wp::array_t<wp::uint8> const&,
    wp::array_t<wp::float32> const&,
    wp::array_t<wp::uint64> const&,
    wp::array_t<wp::vec_t<3,wp::float32>> const&,
    wp::array_t<wp::int32> & adj_type,
    wp::array_t<wp::uint8> & adj_is_solid,
    wp::array_t<wp::float32> & adj_thickness,
    wp::array_t<wp::uint64> & adj_source,
    wp::array_t<wp::vec_t<3,wp::float32>> & adj_scale,
    ModelShapeGeometry & adj_ret)
{
    adj_type = adj_ret.type;
    adj_is_solid = adj_ret.is_solid;
    adj_thickness = adj_ret.thickness;
    adj_source = adj_ret.source;
    adj_scale = adj_ret.scale;
}

CUDA_CALLABLE void adj_atomic_add(ModelShapeGeometry* p, ModelShapeGeometry t)
{
    wp::adj_atomic_add(&p->type, t.type);
    wp::adj_atomic_add(&p->is_solid, t.is_solid);
    wp::adj_atomic_add(&p->thickness, t.thickness);
    wp::adj_atomic_add(&p->source, t.source);
    wp::adj_atomic_add(&p->scale, t.scale);
}




extern "C" __global__ void sleepParticles_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::float32 var_sleepThreshold,
    wp::array_t<wp::uint32> var_particle_flags,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_positions_init,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_positions_after,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_velocities_after,
    wp::float32 var_dt)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        wp::uint32* var_1;
        const wp::uint32 var_2 = 1;
        wp::uint32 var_3;
        wp::uint32 var_4;
        const wp::int32 var_5 = 0;
        bool var_6;
        wp::vec_t<3,wp::float32>* var_7;
        wp::vec_t<3,wp::float32>* var_8;
        wp::vec_t<3,wp::float32> var_9;
        wp::vec_t<3,wp::float32> var_10;
        wp::vec_t<3,wp::float32> var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        bool var_14;
        wp::vec_t<3,wp::float32>* var_15;
        wp::vec_t<3,wp::float32> var_16;
        //---------
        // forward
        // def sleepParticles(                                                                    <L 28>
        // tid = wp.tid()                                                                         <L 36>
        var_0 = builtin_tid1d();
        // if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:                                  <L 37>
        var_1 = wp::address(var_particle_flags, var_0);
        var_3 = wp::load(var_1);
        var_4 = wp::bit_and(var_3, var_2);
        var_6 = (var_4 == var_5);
        if (var_6) {
            // return                                                                             <L 38>
            return;
        }
        // if wp.length(particle_positions_after[tid]-particle_positions_init[tid])/dt<sleepThreshold:       <L 40>
        var_7 = wp::address(var_particle_positions_after, var_0);
        var_8 = wp::address(var_particle_positions_init, var_0);
        var_9 = wp::load(var_7);
        var_10 = wp::load(var_8);
        var_11 = wp::sub(var_9, var_10);
        var_12 = wp::length(var_11);
        var_13 = wp::div(var_12, var_dt);
        var_14 = (var_13 < var_sleepThreshold);
        if (var_14) {
            // particle_positions_after[tid]=particle_positions_init[tid]                         <L 41>
            var_15 = wp::address(var_particle_positions_init, var_0);
            var_16 = wp::load(var_15);
            wp::array_store(var_particle_positions_after, var_0, var_16);
        }
    }
}

extern "C" __global__ void sleepParticles_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::float32 var_sleepThreshold,
    wp::array_t<wp::uint32> var_particle_flags,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_positions_init,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_positions_after,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_velocities_after,
    wp::float32 var_dt,
    wp::float32 adj_sleepThreshold,
    wp::array_t<wp::uint32> adj_particle_flags,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_particle_positions_init,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_particle_positions_after,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_particle_velocities_after,
    wp::float32 adj_dt)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        wp::uint32* var_1;
        const wp::uint32 var_2 = 1;
        wp::uint32 var_3;
        wp::uint32 var_4;
        const wp::int32 var_5 = 0;
        bool var_6;
        wp::vec_t<3,wp::float32>* var_7;
        wp::vec_t<3,wp::float32>* var_8;
        wp::vec_t<3,wp::float32> var_9;
        wp::vec_t<3,wp::float32> var_10;
        wp::vec_t<3,wp::float32> var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        bool var_14;
        wp::vec_t<3,wp::float32>* var_15;
        wp::vec_t<3,wp::float32> var_16;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::uint32 adj_1 = {};
        wp::uint32 adj_2 = {};
        wp::uint32 adj_3 = {};
        wp::uint32 adj_4 = {};
        wp::int32 adj_5 = {};
        bool adj_6 = {};
        wp::vec_t<3,wp::float32> adj_7 = {};
        wp::vec_t<3,wp::float32> adj_8 = {};
        wp::vec_t<3,wp::float32> adj_9 = {};
        wp::vec_t<3,wp::float32> adj_10 = {};
        wp::vec_t<3,wp::float32> adj_11 = {};
        wp::float32 adj_12 = {};
        wp::float32 adj_13 = {};
        bool adj_14 = {};
        wp::vec_t<3,wp::float32> adj_15 = {};
        wp::vec_t<3,wp::float32> adj_16 = {};
        //---------
        // forward
        // def sleepParticles(                                                                    <L 28>
        // tid = wp.tid()                                                                         <L 36>
        var_0 = builtin_tid1d();
        // if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:                                  <L 37>
        var_1 = wp::address(var_particle_flags, var_0);
        var_3 = wp::load(var_1);
        var_4 = wp::bit_and(var_3, var_2);
        var_6 = (var_4 == var_5);
        if (var_6) {
            // return                                                                             <L 38>
            goto label0;
        }
        // if wp.length(particle_positions_after[tid]-particle_positions_init[tid])/dt<sleepThreshold:       <L 40>
        var_7 = wp::address(var_particle_positions_after, var_0);
        var_8 = wp::address(var_particle_positions_init, var_0);
        var_9 = wp::load(var_7);
        var_10 = wp::load(var_8);
        var_11 = wp::sub(var_9, var_10);
        var_12 = wp::length(var_11);
        var_13 = wp::div(var_12, var_dt);
        var_14 = (var_13 < var_sleepThreshold);
        if (var_14) {
            // particle_positions_after[tid]=particle_positions_init[tid]                         <L 41>
            var_15 = wp::address(var_particle_positions_init, var_0);
            var_16 = wp::load(var_15);
            // wp::array_store(var_particle_positions_after, var_0, var_16);
        }
        //---------
        // reverse
        if (var_14) {
            wp::adj_array_store(var_particle_positions_after, var_0, var_16, adj_particle_positions_after, adj_0, adj_15);
            wp::adj_load(var_15, adj_15, adj_16);
            wp::adj_address(var_particle_positions_init, var_0, adj_particle_positions_init, adj_0, adj_15);
            // adj: particle_positions_after[tid]=particle_positions_init[tid]                    <L 41>
        }
        wp::adj_div(var_12, var_dt, var_13, adj_12, adj_dt, adj_13);
        wp::adj_length(var_11, var_12, adj_11, adj_12);
        wp::adj_sub(var_9, var_10, adj_7, adj_8, adj_11);
        wp::adj_load(var_8, adj_8, adj_10);
        wp::adj_load(var_7, adj_7, adj_9);
        wp::adj_address(var_particle_positions_init, var_0, adj_particle_positions_init, adj_0, adj_8);
        wp::adj_address(var_particle_positions_after, var_0, adj_particle_positions_after, adj_0, adj_7);
        // adj: if wp.length(particle_positions_after[tid]-particle_positions_init[tid])/dt<sleepThreshold:  <L 40>
        if (var_6) {
            label0:;
            // adj: return                                                                        <L 38>
        }
        wp::adj_bit_and(var_3, var_2, adj_1, adj_2, adj_4);
        wp::adj_load(var_1, adj_1, adj_3);
        wp::adj_address(var_particle_flags, var_0, adj_particle_flags, adj_0, adj_1);
        // adj: if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:                             <L 37>
        // adj: tid = wp.tid()                                                                    <L 36>
        // adj: def sleepParticles(                                                               <L 28>
        continue;
    }
}



extern "C" __global__ void swellParticles_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::uint32> var_particle_flags,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_position,
    wp::array_t<wp::float32> var_particle_inv_mass,
    wp::array_t<wp::float32> var_particle_radius,
    wp::float32 var_chance,
    wp::float32 var_swelling_rotation_max,
    wp::float32 var_swelling_activation_factor,
    wp::float32 var_jitterFactor,
    wp::float32 var_yShift,
    wp::float32 var_swellingActivationLocationRatio)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        wp::uint32 var_1;
        wp::uint32* var_2;
        wp::uint32 var_3;
        wp::uint32 var_4;
        wp::vec_t<3,wp::float32>* var_5;
        wp::vec_t<3,wp::float32> var_6;
        wp::vec_t<3,wp::float32> var_7;
        const wp::int32 var_8 = 1;
        wp::float32 var_9;
        bool var_10;
        wp::uint32 var_11;
        bool var_12;
        bool var_13;
        wp::int32 var_14;
        const wp::int32 var_15 = 0;
        wp::shape_t* var_16;
        wp::shape_t var_17;
        wp::int32 var_18;
        bool var_19;
        wp::uint32* var_20;
        wp::uint32 var_21;
        bool var_22;
        wp::uint32 var_23;
        wp::vec_t<3,wp::float32>* var_24;
        wp::vec_t<3,wp::float32> var_25;
        wp::vec_t<3,wp::float32> var_26;
        wp::float32 var_27;
        wp::vec_t<3,wp::float32> var_28;
        wp::float32 var_29;
        bool var_30;
        bool var_31;
        wp::float32 var_32;
        bool var_33;
        bool var_34;
        wp::float32* var_35;
        wp::float32 var_36;
        wp::float32 var_37;
        wp::float32* var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        wp::vec_t<3,wp::float32> var_41;
        wp::uint32 var_42;
        wp::uint32 var_43;
        wp::float32 var_44;
        wp::float32 var_45;
        wp::float32 var_46;
        const wp::int32 var_47 = 2;
        wp::uint32 var_48;
        wp::uint32 var_49;
        wp::float32 var_50;
        const wp::float32 var_51 = 3.14;
        wp::float32 var_52;
        const wp::int32 var_53 = 3;
        wp::uint32 var_54;
        wp::uint32 var_55;
        wp::float32 var_56;
        const wp::float32 var_57 = 6.28;
        wp::float32 var_58;
        wp::float32 var_59;
        wp::float32 var_60;
        wp::float32 var_61;
        wp::float32 var_62;
        wp::float32 var_63;
        wp::float32 var_64;
        wp::float32 var_65;
        wp::float32 var_66;
        wp::float32 var_67;
        wp::float32 var_68;
        wp::float32 var_69;
        wp::float32 var_70;
        const wp::float32 var_71 = 1.0;
        wp::float32 var_72;
        wp::float32 var_73;
        wp::float32 var_74;
        wp::float32 var_75;
        wp::float32 var_76;
        wp::float32 var_77;
        wp::float32* var_78;
        wp::float32 var_79;
        wp::float32 var_80;
        wp::float32 var_81;
        wp::float32 var_82;
        wp::float32 var_83;
        wp::float32 var_84;
        wp::float32 var_85;
        wp::float32* var_86;
        wp::float32 var_87;
        wp::float32 var_88;
        wp::float32 var_89;
        wp::float32 var_90;
        wp::float32 var_91;
        wp::float32 var_92;
        wp::float32 var_93;
        wp::float32* var_94;
        wp::uint32 var_95;
        //---------
        // forward
        // def swellParticles(                                                                    <L 44>
        // tid = wp.tid()                                                                         <L 56>
        var_0 = builtin_tid1d();
        // i=wp.uint32(tid)                                                                       <L 57>
        var_1 = wp::uint32(var_0);
        // flag = particle_flags[tid]                                                             <L 58>
        var_2 = wp::address(var_particle_flags, var_0);
        var_3 = wp::load(var_2);
        var_4 = wp::copy(var_3);
        // position=particle_position[tid]                                                        <L 59>
        var_5 = wp::address(var_particle_position, var_0);
        var_6 = wp::load(var_5);
        var_7 = wp::copy(var_6);
        // y=position[1]                                                                          <L 60>
        var_9 = wp::extract(var_7, var_8);
        // if y>yShift and flag==wp.uint32(1): # if particle is active                            <L 61>
        var_10 = (var_9 > var_yShift);
        var_11 = wp::uint32(var_8);
        var_12 = (var_4 == var_11);
        var_13 = var_10 && var_12;
        if (var_13) {
            // trackedID=tid+1                                                                    <L 62>
            var_14 = wp::add(var_0, var_8);
            // if trackedID<particle_flags.shape[0]:                                              <L 63>
            var_16 = &(var_particle_flags.shape);
            var_17 = wp::load(var_16);
            var_18 = wp::extract(var_17, var_15);
            var_19 = (var_14 < var_18);
            if (var_19) {
                // if particle_flags[trackedID]==wp.uint32(0): # check if it has a swelling particle       <L 64>
                var_20 = wp::address(var_particle_flags, var_14);
                var_21 = wp::uint32(var_15);
                var_23 = wp::load(var_20);
                var_22 = (var_23 == var_21);
                if (var_22) {
                    // trackedPosition=particle_position[trackedID]                               <L 65>
                    var_24 = wp::address(var_particle_position, var_14);
                    var_25 = wp::load(var_24);
                    var_26 = wp::copy(var_25);
                    // trackedy=trackedPosition[1]                                                <L 66>
                    var_27 = wp::extract(var_26, var_8);
                    // dx=trackedPosition - position                                              <L 67>
                    var_28 = wp::sub(var_26, var_7);
                    // dist=wp.length(dx)                                                         <L 68>
                    var_29 = wp::length(var_28);
                    // if trackedy>yShift and dist>=swelling_activation_factor and wp.randf(i)<chance: # if the swelling particle is eligible, swell it       <L 69>
                    var_30 = (var_27 > var_yShift);
                    var_31 = (var_29 >= var_swelling_activation_factor);
                    var_32 = wp::randf(var_1);
                    var_33 = (var_32 < var_chance);
                    var_34 = var_30 && var_31 && var_33;
                    if (var_34) {
                        // mass=particle_inv_mass[tid]                                            <L 70>
                        var_35 = wp::address(var_particle_inv_mass, var_0);
                        var_36 = wp::load(var_35);
                        var_37 = wp::copy(var_36);
                        // radius=particle_radius[tid]                                            <L 71>
                        var_38 = wp::address(var_particle_radius, var_0);
                        var_39 = wp::load(var_38);
                        var_40 = wp::copy(var_39);
                        // newPosition=wp.vec3()                                                  <L 73>
                        var_41 = wp::vec_t<3,wp::float32>();
                        // r = wp.randf(i+wp.uint32(1))*radius*jitterFactor                       <L 74>
                        var_42 = wp::uint32(var_8);
                        var_43 = wp::add(var_1, var_42);
                        var_44 = wp::randf(var_43);
                        var_45 = wp::mul(var_44, var_40);
                        var_46 = wp::mul(var_45, var_jitterFactor);
                        // theta = wp.randf(i+wp.uint32(2))*3.14                                  <L 75>
                        var_48 = wp::uint32(var_47);
                        var_49 = wp::add(var_1, var_48);
                        var_50 = wp::randf(var_49);
                        var_52 = wp::mul(var_50, var_51);
                        // phi = wp.randf(i+wp.uint32(3))*6.28                                    <L 76>
                        var_54 = wp::uint32(var_53);
                        var_55 = wp::add(var_1, var_54);
                        var_56 = wp::randf(var_55);
                        var_58 = wp::mul(var_56, var_57);
                        // jitterX = r * wp.sin(theta) * wp.cos(phi)                              <L 77>
                        var_59 = wp::sin(var_52);
                        var_60 = wp::mul(var_46, var_59);
                        var_61 = wp::cos(var_58);
                        var_62 = wp::mul(var_60, var_61);
                        // jitterY = r * wp.sin(theta) * wp.sin(phi)                              <L 78>
                        var_63 = wp::sin(var_52);
                        var_64 = wp::mul(var_46, var_63);
                        var_65 = wp::sin(var_58);
                        var_66 = wp::mul(var_64, var_65);
                        // jitterZ = r * wp.cos(theta)                                            <L 79>
                        var_67 = wp::cos(var_52);
                        var_68 = wp::mul(var_46, var_67);
                        // phii=wp.float32(swellingActivationLocationRatio)                       <L 80>
                        var_69 = wp::float32(var_swellingActivationLocationRatio);
                        // newPosition[0]=(position[0]*(1.0-phii)+trackedPosition[0]*phii) + jitterX       <L 82>
                        var_70 = wp::extract(var_7, var_15);
                        var_72 = wp::sub(var_71, var_69);
                        var_73 = wp::mul(var_70, var_72);
                        var_74 = wp::extract(var_26, var_15);
                        var_75 = wp::mul(var_74, var_69);
                        var_76 = wp::add(var_73, var_75);
                        var_77 = wp::add(var_76, var_62);
                        var_78 = wp::index(var_41, var_15);
                        wp::store(var_78, var_77);
                        // newPosition[1]=(position[1]*(1.0-phii)+trackedPosition[1]*phii) + jitterY       <L 83>
                        var_79 = wp::extract(var_7, var_8);
                        var_80 = wp::sub(var_71, var_69);
                        var_81 = wp::mul(var_79, var_80);
                        var_82 = wp::extract(var_26, var_8);
                        var_83 = wp::mul(var_82, var_69);
                        var_84 = wp::add(var_81, var_83);
                        var_85 = wp::add(var_84, var_66);
                        var_86 = wp::index(var_41, var_8);
                        wp::store(var_86, var_85);
                        // newPosition[2]=(position[2]*(1.0-phii)+trackedPosition[2]*phii) + jitterZ       <L 84>
                        var_87 = wp::extract(var_7, var_47);
                        var_88 = wp::sub(var_71, var_69);
                        var_89 = wp::mul(var_87, var_88);
                        var_90 = wp::extract(var_26, var_47);
                        var_91 = wp::mul(var_90, var_69);
                        var_92 = wp::add(var_89, var_91);
                        var_93 = wp::add(var_92, var_68);
                        var_94 = wp::index(var_41, var_47);
                        wp::store(var_94, var_93);
                        // particle_flags[trackedID]=wp.uint32(1)                                 <L 86>
                        var_95 = wp::uint32(var_8);
                        wp::array_store(var_particle_flags, var_14, var_95);
                        // particle_position[trackedID]=newPosition                               <L 87>
                        wp::array_store(var_particle_position, var_14, var_41);
                        // particle_inv_mass[trackedID]=mass                                      <L 88>
                        wp::array_store(var_particle_inv_mass, var_14, var_37);
                        // particle_radius[trackedID]=radius#/swelling_rotation_max/25.0          <L 89>
                        wp::array_store(var_particle_radius, var_14, var_40);
                    }
                }
            }
        }
    }
}

extern "C" __global__ void swellParticles_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::uint32> var_particle_flags,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_position,
    wp::array_t<wp::float32> var_particle_inv_mass,
    wp::array_t<wp::float32> var_particle_radius,
    wp::float32 var_chance,
    wp::float32 var_swelling_rotation_max,
    wp::float32 var_swelling_activation_factor,
    wp::float32 var_jitterFactor,
    wp::float32 var_yShift,
    wp::float32 var_swellingActivationLocationRatio,
    wp::array_t<wp::uint32> adj_particle_flags,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_particle_position,
    wp::array_t<wp::float32> adj_particle_inv_mass,
    wp::array_t<wp::float32> adj_particle_radius,
    wp::float32 adj_chance,
    wp::float32 adj_swelling_rotation_max,
    wp::float32 adj_swelling_activation_factor,
    wp::float32 adj_jitterFactor,
    wp::float32 adj_yShift,
    wp::float32 adj_swellingActivationLocationRatio)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        wp::uint32 var_1;
        wp::uint32* var_2;
        wp::uint32 var_3;
        wp::uint32 var_4;
        wp::vec_t<3,wp::float32>* var_5;
        wp::vec_t<3,wp::float32> var_6;
        wp::vec_t<3,wp::float32> var_7;
        const wp::int32 var_8 = 1;
        wp::float32 var_9;
        bool var_10;
        wp::uint32 var_11;
        bool var_12;
        bool var_13;
        wp::int32 var_14;
        const wp::int32 var_15 = 0;
        wp::shape_t* var_16;
        wp::shape_t var_17;
        wp::int32 var_18;
        bool var_19;
        wp::uint32* var_20;
        wp::uint32 var_21;
        bool var_22;
        wp::uint32 var_23;
        wp::vec_t<3,wp::float32>* var_24;
        wp::vec_t<3,wp::float32> var_25;
        wp::vec_t<3,wp::float32> var_26;
        wp::float32 var_27;
        wp::vec_t<3,wp::float32> var_28;
        wp::float32 var_29;
        bool var_30;
        bool var_31;
        wp::float32 var_32;
        bool var_33;
        bool var_34;
        wp::float32* var_35;
        wp::float32 var_36;
        wp::float32 var_37;
        wp::float32* var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        wp::vec_t<3,wp::float32> var_41;
        wp::uint32 var_42;
        wp::uint32 var_43;
        wp::float32 var_44;
        wp::float32 var_45;
        wp::float32 var_46;
        const wp::int32 var_47 = 2;
        wp::uint32 var_48;
        wp::uint32 var_49;
        wp::float32 var_50;
        const wp::float32 var_51 = 3.14;
        wp::float32 var_52;
        const wp::int32 var_53 = 3;
        wp::uint32 var_54;
        wp::uint32 var_55;
        wp::float32 var_56;
        const wp::float32 var_57 = 6.28;
        wp::float32 var_58;
        wp::float32 var_59;
        wp::float32 var_60;
        wp::float32 var_61;
        wp::float32 var_62;
        wp::float32 var_63;
        wp::float32 var_64;
        wp::float32 var_65;
        wp::float32 var_66;
        wp::float32 var_67;
        wp::float32 var_68;
        wp::float32 var_69;
        wp::float32 var_70;
        const wp::float32 var_71 = 1.0;
        wp::float32 var_72;
        wp::float32 var_73;
        wp::float32 var_74;
        wp::float32 var_75;
        wp::float32 var_76;
        wp::float32 var_77;
        wp::float32* var_78;
        wp::float32 var_79;
        wp::float32 var_80;
        wp::float32 var_81;
        wp::float32 var_82;
        wp::float32 var_83;
        wp::float32 var_84;
        wp::float32 var_85;
        wp::float32* var_86;
        wp::float32 var_87;
        wp::float32 var_88;
        wp::float32 var_89;
        wp::float32 var_90;
        wp::float32 var_91;
        wp::float32 var_92;
        wp::float32 var_93;
        wp::float32* var_94;
        wp::uint32 var_95;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::uint32 adj_1 = {};
        wp::uint32 adj_2 = {};
        wp::uint32 adj_3 = {};
        wp::uint32 adj_4 = {};
        wp::vec_t<3,wp::float32> adj_5 = {};
        wp::vec_t<3,wp::float32> adj_6 = {};
        wp::vec_t<3,wp::float32> adj_7 = {};
        wp::int32 adj_8 = {};
        wp::float32 adj_9 = {};
        bool adj_10 = {};
        wp::uint32 adj_11 = {};
        bool adj_12 = {};
        bool adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::shape_t adj_16 = {};
        wp::shape_t adj_17 = {};
        wp::int32 adj_18 = {};
        bool adj_19 = {};
        wp::uint32 adj_20 = {};
        wp::uint32 adj_21 = {};
        bool adj_22 = {};
        wp::uint32 adj_23 = {};
        wp::vec_t<3,wp::float32> adj_24 = {};
        wp::vec_t<3,wp::float32> adj_25 = {};
        wp::vec_t<3,wp::float32> adj_26 = {};
        wp::float32 adj_27 = {};
        wp::vec_t<3,wp::float32> adj_28 = {};
        wp::float32 adj_29 = {};
        bool adj_30 = {};
        bool adj_31 = {};
        wp::float32 adj_32 = {};
        bool adj_33 = {};
        bool adj_34 = {};
        wp::float32 adj_35 = {};
        wp::float32 adj_36 = {};
        wp::float32 adj_37 = {};
        wp::float32 adj_38 = {};
        wp::float32 adj_39 = {};
        wp::float32 adj_40 = {};
        wp::vec_t<3,wp::float32> adj_41 = {};
        wp::uint32 adj_42 = {};
        wp::uint32 adj_43 = {};
        wp::float32 adj_44 = {};
        wp::float32 adj_45 = {};
        wp::float32 adj_46 = {};
        wp::int32 adj_47 = {};
        wp::uint32 adj_48 = {};
        wp::uint32 adj_49 = {};
        wp::float32 adj_50 = {};
        wp::float32 adj_51 = {};
        wp::float32 adj_52 = {};
        wp::int32 adj_53 = {};
        wp::uint32 adj_54 = {};
        wp::uint32 adj_55 = {};
        wp::float32 adj_56 = {};
        wp::float32 adj_57 = {};
        wp::float32 adj_58 = {};
        wp::float32 adj_59 = {};
        wp::float32 adj_60 = {};
        wp::float32 adj_61 = {};
        wp::float32 adj_62 = {};
        wp::float32 adj_63 = {};
        wp::float32 adj_64 = {};
        wp::float32 adj_65 = {};
        wp::float32 adj_66 = {};
        wp::float32 adj_67 = {};
        wp::float32 adj_68 = {};
        wp::float32 adj_69 = {};
        wp::float32 adj_70 = {};
        wp::float32 adj_71 = {};
        wp::float32 adj_72 = {};
        wp::float32 adj_73 = {};
        wp::float32 adj_74 = {};
        wp::float32 adj_75 = {};
        wp::float32 adj_76 = {};
        wp::float32 adj_77 = {};
        wp::float32 adj_78 = {};
        wp::float32 adj_79 = {};
        wp::float32 adj_80 = {};
        wp::float32 adj_81 = {};
        wp::float32 adj_82 = {};
        wp::float32 adj_83 = {};
        wp::float32 adj_84 = {};
        wp::float32 adj_85 = {};
        wp::float32 adj_86 = {};
        wp::float32 adj_87 = {};
        wp::float32 adj_88 = {};
        wp::float32 adj_89 = {};
        wp::float32 adj_90 = {};
        wp::float32 adj_91 = {};
        wp::float32 adj_92 = {};
        wp::float32 adj_93 = {};
        wp::float32 adj_94 = {};
        wp::uint32 adj_95 = {};
        //---------
        // forward
        // def swellParticles(                                                                    <L 44>
        // tid = wp.tid()                                                                         <L 56>
        var_0 = builtin_tid1d();
        // i=wp.uint32(tid)                                                                       <L 57>
        var_1 = wp::uint32(var_0);
        // flag = particle_flags[tid]                                                             <L 58>
        var_2 = wp::address(var_particle_flags, var_0);
        var_3 = wp::load(var_2);
        var_4 = wp::copy(var_3);
        // position=particle_position[tid]                                                        <L 59>
        var_5 = wp::address(var_particle_position, var_0);
        var_6 = wp::load(var_5);
        var_7 = wp::copy(var_6);
        // y=position[1]                                                                          <L 60>
        var_9 = wp::extract(var_7, var_8);
        // if y>yShift and flag==wp.uint32(1): # if particle is active                            <L 61>
        var_10 = (var_9 > var_yShift);
        var_11 = wp::uint32(var_8);
        var_12 = (var_4 == var_11);
        var_13 = var_10 && var_12;
        if (var_13) {
            // trackedID=tid+1                                                                    <L 62>
            var_14 = wp::add(var_0, var_8);
            // if trackedID<particle_flags.shape[0]:                                              <L 63>
            var_16 = &(var_particle_flags.shape);
            var_17 = wp::load(var_16);
            var_18 = wp::extract(var_17, var_15);
            var_19 = (var_14 < var_18);
            if (var_19) {
                // if particle_flags[trackedID]==wp.uint32(0): # check if it has a swelling particle       <L 64>
                var_20 = wp::address(var_particle_flags, var_14);
                var_21 = wp::uint32(var_15);
                var_23 = wp::load(var_20);
                var_22 = (var_23 == var_21);
                if (var_22) {
                    // trackedPosition=particle_position[trackedID]                               <L 65>
                    var_24 = wp::address(var_particle_position, var_14);
                    var_25 = wp::load(var_24);
                    var_26 = wp::copy(var_25);
                    // trackedy=trackedPosition[1]                                                <L 66>
                    var_27 = wp::extract(var_26, var_8);
                    // dx=trackedPosition - position                                              <L 67>
                    var_28 = wp::sub(var_26, var_7);
                    // dist=wp.length(dx)                                                         <L 68>
                    var_29 = wp::length(var_28);
                    // if trackedy>yShift and dist>=swelling_activation_factor and wp.randf(i)<chance: # if the swelling particle is eligible, swell it       <L 69>
                    var_30 = (var_27 > var_yShift);
                    var_31 = (var_29 >= var_swelling_activation_factor);
                    var_32 = wp::randf(var_1);
                    var_33 = (var_32 < var_chance);
                    var_34 = var_30 && var_31 && var_33;
                    if (var_34) {
                        // mass=particle_inv_mass[tid]                                            <L 70>
                        var_35 = wp::address(var_particle_inv_mass, var_0);
                        var_36 = wp::load(var_35);
                        var_37 = wp::copy(var_36);
                        // radius=particle_radius[tid]                                            <L 71>
                        var_38 = wp::address(var_particle_radius, var_0);
                        var_39 = wp::load(var_38);
                        var_40 = wp::copy(var_39);
                        // newPosition=wp.vec3()                                                  <L 73>
                        var_41 = wp::vec_t<3,wp::float32>();
                        // r = wp.randf(i+wp.uint32(1))*radius*jitterFactor                       <L 74>
                        var_42 = wp::uint32(var_8);
                        var_43 = wp::add(var_1, var_42);
                        var_44 = wp::randf(var_43);
                        var_45 = wp::mul(var_44, var_40);
                        var_46 = wp::mul(var_45, var_jitterFactor);
                        // theta = wp.randf(i+wp.uint32(2))*3.14                                  <L 75>
                        var_48 = wp::uint32(var_47);
                        var_49 = wp::add(var_1, var_48);
                        var_50 = wp::randf(var_49);
                        var_52 = wp::mul(var_50, var_51);
                        // phi = wp.randf(i+wp.uint32(3))*6.28                                    <L 76>
                        var_54 = wp::uint32(var_53);
                        var_55 = wp::add(var_1, var_54);
                        var_56 = wp::randf(var_55);
                        var_58 = wp::mul(var_56, var_57);
                        // jitterX = r * wp.sin(theta) * wp.cos(phi)                              <L 77>
                        var_59 = wp::sin(var_52);
                        var_60 = wp::mul(var_46, var_59);
                        var_61 = wp::cos(var_58);
                        var_62 = wp::mul(var_60, var_61);
                        // jitterY = r * wp.sin(theta) * wp.sin(phi)                              <L 78>
                        var_63 = wp::sin(var_52);
                        var_64 = wp::mul(var_46, var_63);
                        var_65 = wp::sin(var_58);
                        var_66 = wp::mul(var_64, var_65);
                        // jitterZ = r * wp.cos(theta)                                            <L 79>
                        var_67 = wp::cos(var_52);
                        var_68 = wp::mul(var_46, var_67);
                        // phii=wp.float32(swellingActivationLocationRatio)                       <L 80>
                        var_69 = wp::float32(var_swellingActivationLocationRatio);
                        // newPosition[0]=(position[0]*(1.0-phii)+trackedPosition[0]*phii) + jitterX       <L 82>
                        var_70 = wp::extract(var_7, var_15);
                        var_72 = wp::sub(var_71, var_69);
                        var_73 = wp::mul(var_70, var_72);
                        var_74 = wp::extract(var_26, var_15);
                        var_75 = wp::mul(var_74, var_69);
                        var_76 = wp::add(var_73, var_75);
                        var_77 = wp::add(var_76, var_62);
                        // var_78 = wp::index(var_41, var_15);
                        // wp::store(var_78, var_77);
                        // newPosition[1]=(position[1]*(1.0-phii)+trackedPosition[1]*phii) + jitterY       <L 83>
                        var_79 = wp::extract(var_7, var_8);
                        var_80 = wp::sub(var_71, var_69);
                        var_81 = wp::mul(var_79, var_80);
                        var_82 = wp::extract(var_26, var_8);
                        var_83 = wp::mul(var_82, var_69);
                        var_84 = wp::add(var_81, var_83);
                        var_85 = wp::add(var_84, var_66);
                        // var_86 = wp::index(var_41, var_8);
                        // wp::store(var_86, var_85);
                        // newPosition[2]=(position[2]*(1.0-phii)+trackedPosition[2]*phii) + jitterZ       <L 84>
                        var_87 = wp::extract(var_7, var_47);
                        var_88 = wp::sub(var_71, var_69);
                        var_89 = wp::mul(var_87, var_88);
                        var_90 = wp::extract(var_26, var_47);
                        var_91 = wp::mul(var_90, var_69);
                        var_92 = wp::add(var_89, var_91);
                        var_93 = wp::add(var_92, var_68);
                        // var_94 = wp::index(var_41, var_47);
                        // wp::store(var_94, var_93);
                        // particle_flags[trackedID]=wp.uint32(1)                                 <L 86>
                        var_95 = wp::uint32(var_8);
                        // wp::array_store(var_particle_flags, var_14, var_95);
                        // particle_position[trackedID]=newPosition                               <L 87>
                        // wp::array_store(var_particle_position, var_14, var_41);
                        // particle_inv_mass[trackedID]=mass                                      <L 88>
                        // wp::array_store(var_particle_inv_mass, var_14, var_37);
                        // particle_radius[trackedID]=radius#/swelling_rotation_max/25.0          <L 89>
                        // wp::array_store(var_particle_radius, var_14, var_40);
                    }
                }
            }
        }
        //---------
        // reverse
        if (var_13) {
            if (var_19) {
                if (var_22) {
                    if (var_34) {
                        wp::adj_array_store(var_particle_radius, var_14, var_40, adj_particle_radius, adj_14, adj_40);
                        // adj: particle_radius[trackedID]=radius#/swelling_rotation_max/25.0     <L 89>
                        wp::adj_array_store(var_particle_inv_mass, var_14, var_37, adj_particle_inv_mass, adj_14, adj_37);
                        // adj: particle_inv_mass[trackedID]=mass                                 <L 88>
                        wp::adj_array_store(var_particle_position, var_14, var_41, adj_particle_position, adj_14, adj_41);
                        // adj: particle_position[trackedID]=newPosition                          <L 87>
                        wp::adj_array_store(var_particle_flags, var_14, var_95, adj_particle_flags, adj_14, adj_95);
                        wp::adj_uint32(var_8, adj_8, adj_95);
                        // adj: particle_flags[trackedID]=wp.uint32(1)                            <L 86>
                        wp::adj_store(var_94, var_93, adj_94, adj_93);
                        wp::adj_index(var_41, var_47, adj_41, adj_47, adj_94);
                        wp::adj_add(var_92, var_68, adj_92, adj_68, adj_93);
                        wp::adj_add(var_89, var_91, adj_89, adj_91, adj_92);
                        wp::adj_mul(var_90, var_69, adj_90, adj_69, adj_91);
                        wp::adj_extract(var_26, var_47, adj_26, adj_47, adj_90);
                        wp::adj_mul(var_87, var_88, adj_87, adj_88, adj_89);
                        wp::adj_sub(var_71, var_69, adj_71, adj_69, adj_88);
                        wp::adj_extract(var_7, var_47, adj_7, adj_47, adj_87);
                        // adj: newPosition[2]=(position[2]*(1.0-phii)+trackedPosition[2]*phii) + jitterZ  <L 84>
                        wp::adj_store(var_86, var_85, adj_86, adj_85);
                        wp::adj_index(var_41, var_8, adj_41, adj_8, adj_86);
                        wp::adj_add(var_84, var_66, adj_84, adj_66, adj_85);
                        wp::adj_add(var_81, var_83, adj_81, adj_83, adj_84);
                        wp::adj_mul(var_82, var_69, adj_82, adj_69, adj_83);
                        wp::adj_extract(var_26, var_8, adj_26, adj_8, adj_82);
                        wp::adj_mul(var_79, var_80, adj_79, adj_80, adj_81);
                        wp::adj_sub(var_71, var_69, adj_71, adj_69, adj_80);
                        wp::adj_extract(var_7, var_8, adj_7, adj_8, adj_79);
                        // adj: newPosition[1]=(position[1]*(1.0-phii)+trackedPosition[1]*phii) + jitterY  <L 83>
                        wp::adj_store(var_78, var_77, adj_78, adj_77);
                        wp::adj_index(var_41, var_15, adj_41, adj_15, adj_78);
                        wp::adj_add(var_76, var_62, adj_76, adj_62, adj_77);
                        wp::adj_add(var_73, var_75, adj_73, adj_75, adj_76);
                        wp::adj_mul(var_74, var_69, adj_74, adj_69, adj_75);
                        wp::adj_extract(var_26, var_15, adj_26, adj_15, adj_74);
                        wp::adj_mul(var_70, var_72, adj_70, adj_72, adj_73);
                        wp::adj_sub(var_71, var_69, adj_71, adj_69, adj_72);
                        wp::adj_extract(var_7, var_15, adj_7, adj_15, adj_70);
                        // adj: newPosition[0]=(position[0]*(1.0-phii)+trackedPosition[0]*phii) + jitterX  <L 82>
                        wp::adj_float32(var_swellingActivationLocationRatio, adj_swellingActivationLocationRatio, adj_69);
                        // adj: phii=wp.float32(swellingActivationLocationRatio)                  <L 80>
                        wp::adj_mul(var_46, var_67, adj_46, adj_67, adj_68);
                        wp::adj_cos(var_52, adj_52, adj_67);
                        // adj: jitterZ = r * wp.cos(theta)                                       <L 79>
                        wp::adj_mul(var_64, var_65, adj_64, adj_65, adj_66);
                        wp::adj_sin(var_58, adj_58, adj_65);
                        wp::adj_mul(var_46, var_63, adj_46, adj_63, adj_64);
                        wp::adj_sin(var_52, adj_52, adj_63);
                        // adj: jitterY = r * wp.sin(theta) * wp.sin(phi)                         <L 78>
                        wp::adj_mul(var_60, var_61, adj_60, adj_61, adj_62);
                        wp::adj_cos(var_58, adj_58, adj_61);
                        wp::adj_mul(var_46, var_59, adj_46, adj_59, adj_60);
                        wp::adj_sin(var_52, adj_52, adj_59);
                        // adj: jitterX = r * wp.sin(theta) * wp.cos(phi)                         <L 77>
                        wp::adj_mul(var_56, var_57, adj_56, adj_57, adj_58);
                        wp::adj_randf(var_55, adj_55, adj_56);
                        wp::adj_add(var_1, var_54, adj_1, adj_54, adj_55);
                        wp::adj_uint32(var_53, adj_53, adj_54);
                        // adj: phi = wp.randf(i+wp.uint32(3))*6.28                               <L 76>
                        wp::adj_mul(var_50, var_51, adj_50, adj_51, adj_52);
                        wp::adj_randf(var_49, adj_49, adj_50);
                        wp::adj_add(var_1, var_48, adj_1, adj_48, adj_49);
                        wp::adj_uint32(var_47, adj_47, adj_48);
                        // adj: theta = wp.randf(i+wp.uint32(2))*3.14                             <L 75>
                        wp::adj_mul(var_45, var_jitterFactor, adj_45, adj_jitterFactor, adj_46);
                        wp::adj_mul(var_44, var_40, adj_44, adj_40, adj_45);
                        wp::adj_randf(var_43, adj_43, adj_44);
                        wp::adj_add(var_1, var_42, adj_1, adj_42, adj_43);
                        wp::adj_uint32(var_8, adj_8, adj_42);
                        // adj: r = wp.randf(i+wp.uint32(1))*radius*jitterFactor                  <L 74>
                        // adj: newPosition=wp.vec3()                                             <L 73>
                        wp::adj_copy(var_39, adj_38, adj_40);
                        wp::adj_load(var_38, adj_38, adj_39);
                        wp::adj_address(var_particle_radius, var_0, adj_particle_radius, adj_0, adj_38);
                        // adj: radius=particle_radius[tid]                                       <L 71>
                        wp::adj_copy(var_36, adj_35, adj_37);
                        wp::adj_load(var_35, adj_35, adj_36);
                        wp::adj_address(var_particle_inv_mass, var_0, adj_particle_inv_mass, adj_0, adj_35);
                        // adj: mass=particle_inv_mass[tid]                                       <L 70>
                    }
                    wp::adj_randf(var_1, adj_1, adj_32);
                    // adj: if trackedy>yShift and dist>=swelling_activation_factor and wp.randf(i)<chance: # if the swelling particle is eligible, swell it  <L 69>
                    wp::adj_length(var_28, var_29, adj_28, adj_29);
                    // adj: dist=wp.length(dx)                                                    <L 68>
                    wp::adj_sub(var_26, var_7, adj_26, adj_7, adj_28);
                    // adj: dx=trackedPosition - position                                         <L 67>
                    wp::adj_extract(var_26, var_8, adj_26, adj_8, adj_27);
                    // adj: trackedy=trackedPosition[1]                                           <L 66>
                    wp::adj_copy(var_25, adj_24, adj_26);
                    wp::adj_load(var_24, adj_24, adj_25);
                    wp::adj_address(var_particle_position, var_14, adj_particle_position, adj_14, adj_24);
                    // adj: trackedPosition=particle_position[trackedID]                          <L 65>
                }
                wp::adj_load(var_20, adj_20, adj_23);
                wp::adj_uint32(var_15, adj_15, adj_21);
                wp::adj_address(var_particle_flags, var_14, adj_particle_flags, adj_14, adj_20);
                // adj: if particle_flags[trackedID]==wp.uint32(0): # check if it has a swelling particle  <L 64>
            }
            wp::adj_extract(var_17, var_15, adj_16, adj_15, adj_18);
            wp::adj_load(var_16, adj_16, adj_17);
            adj_particle_flags.shape = adj_16;
            // adj: if trackedID<particle_flags.shape[0]:                                         <L 63>
            wp::adj_add(var_0, var_8, adj_0, adj_8, adj_14);
            // adj: trackedID=tid+1                                                               <L 62>
        }
        wp::adj_uint32(var_8, adj_8, adj_11);
        // adj: if y>yShift and flag==wp.uint32(1): # if particle is active                       <L 61>
        wp::adj_extract(var_7, var_8, adj_7, adj_8, adj_9);
        // adj: y=position[1]                                                                     <L 60>
        wp::adj_copy(var_6, adj_5, adj_7);
        wp::adj_load(var_5, adj_5, adj_6);
        wp::adj_address(var_particle_position, var_0, adj_particle_position, adj_0, adj_5);
        // adj: position=particle_position[tid]                                                   <L 59>
        wp::adj_copy(var_3, adj_2, adj_4);
        wp::adj_load(var_2, adj_2, adj_3);
        wp::adj_address(var_particle_flags, var_0, adj_particle_flags, adj_0, adj_2);
        // adj: flag = particle_flags[tid]                                                        <L 58>
        wp::adj_uint32(var_0, adj_0, adj_1);
        // adj: i=wp.uint32(tid)                                                                  <L 57>
        // adj: tid = wp.tid()                                                                    <L 56>
        // adj: def swellParticles(                                                               <L 44>
        continue;
    }
}



extern "C" __global__ void swellParticlesStage2_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::uint32> var_particle_flags,
    wp::array_t<wp::float32> var_particle_inv_mass,
    wp::array_t<wp::float32> var_particle_radius,
    wp::float32 var_swelling_rotation,
    wp::float32 var_swelling_rotation_max,
    wp::float32 var_fullRadius)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        wp::uint32* var_1;
        const wp::int32 var_2 = 1;
        wp::uint32 var_3;
        bool var_4;
        wp::uint32 var_5;
        wp::float32* var_6;
        bool var_7;
        wp::float32 var_8;
        bool var_9;
        wp::float32* var_10;
        wp::float32 var_11;
        const wp::float32 var_12 = 25.0;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        //---------
        // forward
        // def swellParticlesStage2(                                                              <L 93>
        // tid = wp.tid()                                                                         <L 101>
        var_0 = builtin_tid1d();
        // if particle_flags[tid]==wp.uint32(1) and particle_radius[tid]<fullRadius:              <L 102>
        var_1 = wp::address(var_particle_flags, var_0);
        var_3 = wp::uint32(var_2);
        var_5 = wp::load(var_1);
        var_4 = (var_5 == var_3);
        var_6 = wp::address(var_particle_radius, var_0);
        var_8 = wp::load(var_6);
        var_7 = (var_8 < var_fullRadius);
        var_9 = var_4 && var_7;
        if (var_9) {
            // particle_radius[tid]=particle_radius[tid]+fullRadius/swelling_rotation_max/25.0       <L 103>
            var_10 = wp::address(var_particle_radius, var_0);
            var_11 = wp::div(var_fullRadius, var_swelling_rotation_max);
            var_13 = wp::div(var_11, var_12);
            var_14 = wp::load(var_10);
            var_15 = wp::add(var_14, var_13);
            wp::array_store(var_particle_radius, var_0, var_15);
        }
        if (!var_9) {
            // return                                                                             <L 105>
            return;
        }
    }
}

extern "C" __global__ void swellParticlesStage2_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::uint32> var_particle_flags,
    wp::array_t<wp::float32> var_particle_inv_mass,
    wp::array_t<wp::float32> var_particle_radius,
    wp::float32 var_swelling_rotation,
    wp::float32 var_swelling_rotation_max,
    wp::float32 var_fullRadius,
    wp::array_t<wp::uint32> adj_particle_flags,
    wp::array_t<wp::float32> adj_particle_inv_mass,
    wp::array_t<wp::float32> adj_particle_radius,
    wp::float32 adj_swelling_rotation,
    wp::float32 adj_swelling_rotation_max,
    wp::float32 adj_fullRadius)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        wp::uint32* var_1;
        const wp::int32 var_2 = 1;
        wp::uint32 var_3;
        bool var_4;
        wp::uint32 var_5;
        wp::float32* var_6;
        bool var_7;
        wp::float32 var_8;
        bool var_9;
        wp::float32* var_10;
        wp::float32 var_11;
        const wp::float32 var_12 = 25.0;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::uint32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::uint32 adj_3 = {};
        bool adj_4 = {};
        wp::uint32 adj_5 = {};
        wp::float32 adj_6 = {};
        bool adj_7 = {};
        wp::float32 adj_8 = {};
        bool adj_9 = {};
        wp::float32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::float32 adj_15 = {};
        //---------
        // forward
        // def swellParticlesStage2(                                                              <L 93>
        // tid = wp.tid()                                                                         <L 101>
        var_0 = builtin_tid1d();
        // if particle_flags[tid]==wp.uint32(1) and particle_radius[tid]<fullRadius:              <L 102>
        var_1 = wp::address(var_particle_flags, var_0);
        var_3 = wp::uint32(var_2);
        var_5 = wp::load(var_1);
        var_4 = (var_5 == var_3);
        var_6 = wp::address(var_particle_radius, var_0);
        var_8 = wp::load(var_6);
        var_7 = (var_8 < var_fullRadius);
        var_9 = var_4 && var_7;
        if (var_9) {
            // particle_radius[tid]=particle_radius[tid]+fullRadius/swelling_rotation_max/25.0       <L 103>
            var_10 = wp::address(var_particle_radius, var_0);
            var_11 = wp::div(var_fullRadius, var_swelling_rotation_max);
            var_13 = wp::div(var_11, var_12);
            var_14 = wp::load(var_10);
            var_15 = wp::add(var_14, var_13);
            // wp::array_store(var_particle_radius, var_0, var_15);
        }
        if (!var_9) {
            // return                                                                             <L 105>
            goto label0;
        }
        //---------
        // reverse
        if (!var_9) {
            label0:;
            // adj: return                                                                        <L 105>
        }
        if (var_9) {
            wp::adj_array_store(var_particle_radius, var_0, var_15, adj_particle_radius, adj_0, adj_15);
            wp::adj_add(var_14, var_13, adj_10, adj_13, adj_15);
            wp::adj_load(var_10, adj_10, adj_14);
            wp::adj_div(var_11, var_12, var_13, adj_11, adj_12, adj_13);
            wp::adj_div(var_fullRadius, var_swelling_rotation_max, var_11, adj_fullRadius, adj_swelling_rotation_max, adj_11);
            wp::adj_address(var_particle_radius, var_0, adj_particle_radius, adj_0, adj_10);
            // adj: particle_radius[tid]=particle_radius[tid]+fullRadius/swelling_rotation_max/25.0  <L 103>
        }
        wp::adj_load(var_6, adj_6, adj_8);
        wp::adj_address(var_particle_radius, var_0, adj_particle_radius, adj_0, adj_6);
        wp::adj_load(var_1, adj_1, adj_5);
        wp::adj_uint32(var_2, adj_2, adj_3);
        wp::adj_address(var_particle_flags, var_0, adj_particle_flags, adj_0, adj_1);
        // adj: if particle_flags[tid]==wp.uint32(1) and particle_radius[tid]<fullRadius:         <L 102>
        // adj: tid = wp.tid()                                                                    <L 101>
        // adj: def swellParticlesStage2(                                                         <L 93>
        continue;
    }
}



extern "C" __global__ void my_integrate_particles_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float32>> var_x,
    wp::array_t<wp::vec_t<3,wp::float32>> var_v,
    wp::array_t<wp::vec_t<3,wp::float32>> var_f,
    wp::array_t<wp::float32> var_w,
    wp::array_t<wp::uint32> var_particle_flags,
    wp::vec_t<3,wp::float32> var_gravity,
    wp::float32 var_dt,
    wp::float32 var_v_max,
    wp::array_t<wp::vec_t<3,wp::float32>> var_x_new,
    wp::array_t<wp::vec_t<3,wp::float32>> var_v_new)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        wp::uint32* var_1;
        const wp::uint32 var_2 = 1;
        wp::uint32 var_3;
        wp::uint32 var_4;
        const wp::int32 var_5 = 0;
        bool var_6;
        wp::vec_t<3,wp::float32>* var_7;
        wp::vec_t<3,wp::float32> var_8;
        wp::vec_t<3,wp::float32> var_9;
        wp::vec_t<3,wp::float32>* var_10;
        wp::vec_t<3,wp::float32> var_11;
        wp::vec_t<3,wp::float32> var_12;
        wp::vec_t<3,wp::float32>* var_13;
        wp::vec_t<3,wp::float32> var_14;
        wp::vec_t<3,wp::float32> var_15;
        wp::float32* var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        wp::vec_t<3,wp::float32> var_19;
        const wp::float32 var_20 = 0.0;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::vec_t<3,wp::float32> var_23;
        wp::vec_t<3,wp::float32> var_24;
        wp::vec_t<3,wp::float32> var_25;
        wp::vec_t<3,wp::float32> var_26;
        wp::float32 var_27;
        bool var_28;
        wp::float32 var_29;
        wp::vec_t<3,wp::float32> var_30;
        wp::vec_t<3,wp::float32> var_31;
        wp::vec_t<3,wp::float32> var_32;
        wp::vec_t<3,wp::float32> var_33;
        //---------
        // forward
        // def my_integrate_particles(                                                            <L 108>
        // tid = wp.tid()                                                                         <L 120>
        var_0 = builtin_tid1d();
        // if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:                                  <L 121>
        var_1 = wp::address(var_particle_flags, var_0);
        var_3 = wp::load(var_1);
        var_4 = wp::bit_and(var_3, var_2);
        var_6 = (var_4 == var_5);
        if (var_6) {
            // return                                                                             <L 122>
            return;
        }
        // x0 = x[tid]                                                                            <L 123>
        var_7 = wp::address(var_x, var_0);
        var_8 = wp::load(var_7);
        var_9 = wp::copy(var_8);
        // v0 = v[tid]                                                                            <L 124>
        var_10 = wp::address(var_v, var_0);
        var_11 = wp::load(var_10);
        var_12 = wp::copy(var_11);
        // f0 = f[tid]                                                                            <L 125>
        var_13 = wp::address(var_f, var_0);
        var_14 = wp::load(var_13);
        var_15 = wp::copy(var_14);
        // inv_mass = w[tid]                                                                      <L 126>
        var_16 = wp::address(var_w, var_0);
        var_17 = wp::load(var_16);
        var_18 = wp::copy(var_17);
        // v1 = v0 + (f0 * inv_mass + gravity * wp.step(0.0 - inv_mass)) * dt                     <L 127>
        var_19 = wp::mul(var_15, var_18);
        var_21 = wp::sub(var_20, var_18);
        var_22 = wp::step(var_21);
        var_23 = wp::mul(var_gravity, var_22);
        var_24 = wp::add(var_19, var_23);
        var_25 = wp::mul(var_24, var_dt);
        var_26 = wp::add(var_12, var_25);
        // v1_mag = wp.length(v1)                                                                 <L 128>
        var_27 = wp::length(var_26);
        // if v1_mag > v_max:                                                                     <L 129>
        var_28 = (var_27 > var_v_max);
        if (var_28) {
            // v1 *= v_max / v1_mag                                                               <L 130>
            var_29 = wp::div(var_v_max, var_27);
            var_30 = wp::mul(var_26, var_29);
        }
        var_31 = wp::select(var_28, var_26, var_30);
        // x1 = x0 + v1 * dt                                                                      <L 131>
        var_32 = wp::mul(var_31, var_dt);
        var_33 = wp::add(var_9, var_32);
        // x_new[tid] = x1                                                                        <L 132>
        wp::array_store(var_x_new, var_0, var_33);
        // v_new[tid] = v1                                                                        <L 133>
        wp::array_store(var_v_new, var_0, var_31);
    }
}

extern "C" __global__ void my_integrate_particles_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float32>> var_x,
    wp::array_t<wp::vec_t<3,wp::float32>> var_v,
    wp::array_t<wp::vec_t<3,wp::float32>> var_f,
    wp::array_t<wp::float32> var_w,
    wp::array_t<wp::uint32> var_particle_flags,
    wp::vec_t<3,wp::float32> var_gravity,
    wp::float32 var_dt,
    wp::float32 var_v_max,
    wp::array_t<wp::vec_t<3,wp::float32>> var_x_new,
    wp::array_t<wp::vec_t<3,wp::float32>> var_v_new,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_x,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_v,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_f,
    wp::array_t<wp::float32> adj_w,
    wp::array_t<wp::uint32> adj_particle_flags,
    wp::vec_t<3,wp::float32> adj_gravity,
    wp::float32 adj_dt,
    wp::float32 adj_v_max,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_x_new,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_v_new)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        wp::uint32* var_1;
        const wp::uint32 var_2 = 1;
        wp::uint32 var_3;
        wp::uint32 var_4;
        const wp::int32 var_5 = 0;
        bool var_6;
        wp::vec_t<3,wp::float32>* var_7;
        wp::vec_t<3,wp::float32> var_8;
        wp::vec_t<3,wp::float32> var_9;
        wp::vec_t<3,wp::float32>* var_10;
        wp::vec_t<3,wp::float32> var_11;
        wp::vec_t<3,wp::float32> var_12;
        wp::vec_t<3,wp::float32>* var_13;
        wp::vec_t<3,wp::float32> var_14;
        wp::vec_t<3,wp::float32> var_15;
        wp::float32* var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        wp::vec_t<3,wp::float32> var_19;
        const wp::float32 var_20 = 0.0;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::vec_t<3,wp::float32> var_23;
        wp::vec_t<3,wp::float32> var_24;
        wp::vec_t<3,wp::float32> var_25;
        wp::vec_t<3,wp::float32> var_26;
        wp::float32 var_27;
        bool var_28;
        wp::float32 var_29;
        wp::vec_t<3,wp::float32> var_30;
        wp::vec_t<3,wp::float32> var_31;
        wp::vec_t<3,wp::float32> var_32;
        wp::vec_t<3,wp::float32> var_33;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::uint32 adj_1 = {};
        wp::uint32 adj_2 = {};
        wp::uint32 adj_3 = {};
        wp::uint32 adj_4 = {};
        wp::int32 adj_5 = {};
        bool adj_6 = {};
        wp::vec_t<3,wp::float32> adj_7 = {};
        wp::vec_t<3,wp::float32> adj_8 = {};
        wp::vec_t<3,wp::float32> adj_9 = {};
        wp::vec_t<3,wp::float32> adj_10 = {};
        wp::vec_t<3,wp::float32> adj_11 = {};
        wp::vec_t<3,wp::float32> adj_12 = {};
        wp::vec_t<3,wp::float32> adj_13 = {};
        wp::vec_t<3,wp::float32> adj_14 = {};
        wp::vec_t<3,wp::float32> adj_15 = {};
        wp::float32 adj_16 = {};
        wp::float32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::vec_t<3,wp::float32> adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::vec_t<3,wp::float32> adj_23 = {};
        wp::vec_t<3,wp::float32> adj_24 = {};
        wp::vec_t<3,wp::float32> adj_25 = {};
        wp::vec_t<3,wp::float32> adj_26 = {};
        wp::float32 adj_27 = {};
        bool adj_28 = {};
        wp::float32 adj_29 = {};
        wp::vec_t<3,wp::float32> adj_30 = {};
        wp::vec_t<3,wp::float32> adj_31 = {};
        wp::vec_t<3,wp::float32> adj_32 = {};
        wp::vec_t<3,wp::float32> adj_33 = {};
        //---------
        // forward
        // def my_integrate_particles(                                                            <L 108>
        // tid = wp.tid()                                                                         <L 120>
        var_0 = builtin_tid1d();
        // if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:                                  <L 121>
        var_1 = wp::address(var_particle_flags, var_0);
        var_3 = wp::load(var_1);
        var_4 = wp::bit_and(var_3, var_2);
        var_6 = (var_4 == var_5);
        if (var_6) {
            // return                                                                             <L 122>
            goto label0;
        }
        // x0 = x[tid]                                                                            <L 123>
        var_7 = wp::address(var_x, var_0);
        var_8 = wp::load(var_7);
        var_9 = wp::copy(var_8);
        // v0 = v[tid]                                                                            <L 124>
        var_10 = wp::address(var_v, var_0);
        var_11 = wp::load(var_10);
        var_12 = wp::copy(var_11);
        // f0 = f[tid]                                                                            <L 125>
        var_13 = wp::address(var_f, var_0);
        var_14 = wp::load(var_13);
        var_15 = wp::copy(var_14);
        // inv_mass = w[tid]                                                                      <L 126>
        var_16 = wp::address(var_w, var_0);
        var_17 = wp::load(var_16);
        var_18 = wp::copy(var_17);
        // v1 = v0 + (f0 * inv_mass + gravity * wp.step(0.0 - inv_mass)) * dt                     <L 127>
        var_19 = wp::mul(var_15, var_18);
        var_21 = wp::sub(var_20, var_18);
        var_22 = wp::step(var_21);
        var_23 = wp::mul(var_gravity, var_22);
        var_24 = wp::add(var_19, var_23);
        var_25 = wp::mul(var_24, var_dt);
        var_26 = wp::add(var_12, var_25);
        // v1_mag = wp.length(v1)                                                                 <L 128>
        var_27 = wp::length(var_26);
        // if v1_mag > v_max:                                                                     <L 129>
        var_28 = (var_27 > var_v_max);
        if (var_28) {
            // v1 *= v_max / v1_mag                                                               <L 130>
            var_29 = wp::div(var_v_max, var_27);
            var_30 = wp::mul(var_26, var_29);
        }
        var_31 = wp::select(var_28, var_26, var_30);
        // x1 = x0 + v1 * dt                                                                      <L 131>
        var_32 = wp::mul(var_31, var_dt);
        var_33 = wp::add(var_9, var_32);
        // x_new[tid] = x1                                                                        <L 132>
        // wp::array_store(var_x_new, var_0, var_33);
        // v_new[tid] = v1                                                                        <L 133>
        // wp::array_store(var_v_new, var_0, var_31);
        //---------
        // reverse
        wp::adj_array_store(var_v_new, var_0, var_31, adj_v_new, adj_0, adj_31);
        // adj: v_new[tid] = v1                                                                   <L 133>
        wp::adj_array_store(var_x_new, var_0, var_33, adj_x_new, adj_0, adj_33);
        // adj: x_new[tid] = x1                                                                   <L 132>
        wp::adj_add(var_9, var_32, adj_9, adj_32, adj_33);
        wp::adj_mul(var_31, var_dt, adj_31, adj_dt, adj_32);
        // adj: x1 = x0 + v1 * dt                                                                 <L 131>
        wp::adj_select(var_28, var_26, var_30, adj_28, adj_26, adj_30, adj_31);
        if (var_28) {
            wp::adj_mul(var_26, var_29, adj_26, adj_29, adj_30);
            wp::adj_div(var_v_max, var_27, var_29, adj_v_max, adj_27, adj_29);
            // adj: v1 *= v_max / v1_mag                                                          <L 130>
        }
        // adj: if v1_mag > v_max:                                                                <L 129>
        wp::adj_length(var_26, var_27, adj_26, adj_27);
        // adj: v1_mag = wp.length(v1)                                                            <L 128>
        wp::adj_add(var_12, var_25, adj_12, adj_25, adj_26);
        wp::adj_mul(var_24, var_dt, adj_24, adj_dt, adj_25);
        wp::adj_add(var_19, var_23, adj_19, adj_23, adj_24);
        wp::adj_mul(var_gravity, var_22, adj_gravity, adj_22, adj_23);
        wp::adj_step(var_21, adj_21, adj_22);
        wp::adj_sub(var_20, var_18, adj_20, adj_18, adj_21);
        wp::adj_mul(var_15, var_18, adj_15, adj_18, adj_19);
        // adj: v1 = v0 + (f0 * inv_mass + gravity * wp.step(0.0 - inv_mass)) * dt                <L 127>
        wp::adj_copy(var_17, adj_16, adj_18);
        wp::adj_load(var_16, adj_16, adj_17);
        wp::adj_address(var_w, var_0, adj_w, adj_0, adj_16);
        // adj: inv_mass = w[tid]                                                                 <L 126>
        wp::adj_copy(var_14, adj_13, adj_15);
        wp::adj_load(var_13, adj_13, adj_14);
        wp::adj_address(var_f, var_0, adj_f, adj_0, adj_13);
        // adj: f0 = f[tid]                                                                       <L 125>
        wp::adj_copy(var_11, adj_10, adj_12);
        wp::adj_load(var_10, adj_10, adj_11);
        wp::adj_address(var_v, var_0, adj_v, adj_0, adj_10);
        // adj: v0 = v[tid]                                                                       <L 124>
        wp::adj_copy(var_8, adj_7, adj_9);
        wp::adj_load(var_7, adj_7, adj_8);
        wp::adj_address(var_x, var_0, adj_x, adj_0, adj_7);
        // adj: x0 = x[tid]                                                                       <L 123>
        if (var_6) {
            label0:;
            // adj: return                                                                        <L 122>
        }
        wp::adj_bit_and(var_3, var_2, adj_1, adj_2, adj_4);
        wp::adj_load(var_1, adj_1, adj_3);
        wp::adj_address(var_particle_flags, var_0, adj_particle_flags, adj_0, adj_1);
        // adj: if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:                             <L 121>
        // adj: tid = wp.tid()                                                                    <L 120>
        // adj: def my_integrate_particles(                                                       <L 108>
        continue;
    }
}



extern "C" __global__ void my_apply_particle_deltas_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float32>> var_x_orig,
    wp::array_t<wp::vec_t<3,wp::float32>> var_x_pred,
    wp::array_t<wp::uint32> var_particle_flags,
    wp::array_t<wp::vec_t<3,wp::float32>> var_delta,
    wp::float32 var_dt,
    wp::float32 var_v_max,
    wp::array_t<wp::vec_t<3,wp::float32>> var_x_out,
    wp::array_t<wp::vec_t<3,wp::float32>> var_v_out)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        wp::uint32* var_1;
        const wp::uint32 var_2 = 1;
        wp::uint32 var_3;
        wp::uint32 var_4;
        const wp::int32 var_5 = 0;
        bool var_6;
        wp::vec_t<3,wp::float32>* var_7;
        wp::vec_t<3,wp::float32> var_8;
        wp::vec_t<3,wp::float32> var_9;
        wp::vec_t<3,wp::float32>* var_10;
        wp::vec_t<3,wp::float32> var_11;
        wp::vec_t<3,wp::float32> var_12;
        wp::vec_t<3,wp::float32>* var_13;
        wp::vec_t<3,wp::float32> var_14;
        wp::vec_t<3,wp::float32> var_15;
        bool var_16;
        const wp::float32 var_17 = 0.0;
        wp::float32* var_18;
        const wp::int32 var_19 = 1;
        wp::float32* var_20;
        const wp::int32 var_21 = 2;
        wp::float32* var_22;
        wp::vec_t<3,wp::float32> var_23;
        wp::vec_t<3,wp::float32> var_24;
        wp::vec_t<3,wp::float32> var_25;
        wp::float32 var_26;
        bool var_27;
        wp::float32 var_28;
        wp::vec_t<3,wp::float32> var_29;
        wp::vec_t<3,wp::float32> var_30;
        //---------
        // forward
        // def my_apply_particle_deltas(                                                          <L 136>
        // tid = wp.tid()                                                                         <L 146>
        var_0 = builtin_tid1d();
        // if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:                                  <L 147>
        var_1 = wp::address(var_particle_flags, var_0);
        var_3 = wp::load(var_1);
        var_4 = wp::bit_and(var_3, var_2);
        var_6 = (var_4 == var_5);
        if (var_6) {
            // return                                                                             <L 148>
            return;
        }
        // x0 = x_orig[tid]                                                                       <L 149>
        var_7 = wp::address(var_x_orig, var_0);
        var_8 = wp::load(var_7);
        var_9 = wp::copy(var_8);
        // xp = x_pred[tid]                                                                       <L 150>
        var_10 = wp::address(var_x_pred, var_0);
        var_11 = wp::load(var_10);
        var_12 = wp::copy(var_11);
        // d = delta[tid]                                                                         <L 151>
        var_13 = wp::address(var_delta, var_0);
        var_14 = wp::load(var_13);
        var_15 = wp::copy(var_14);
        // if wp.isnan(d):                                                                        <L 152>
        var_16 = wp::isnan(var_15);
        if (var_16) {
            // d[0]=0.0                                                                           <L 153>
            var_18 = wp::index(var_15, var_5);
            wp::store(var_18, var_17);
            // d[1]=0.0                                                                           <L 154>
            var_20 = wp::index(var_15, var_19);
            wp::store(var_20, var_17);
            // d[2]=0.0                                                                           <L 155>
            var_22 = wp::index(var_15, var_21);
            wp::store(var_22, var_17);
        }
        // x_new = xp + d                                                                         <L 157>
        var_23 = wp::add(var_12, var_15);
        // v_new = (x_new - x0) / dt                                                              <L 158>
        var_24 = wp::sub(var_23, var_9);
        var_25 = wp::div(var_24, var_dt);
        // v_new_mag = wp.length(v_new)                                                           <L 159>
        var_26 = wp::length(var_25);
        // if v_new_mag > v_max:                                                                  <L 160>
        var_27 = (var_26 > var_v_max);
        if (var_27) {
            // v_new *= v_max / v_new_mag                                                         <L 161>
            var_28 = wp::div(var_v_max, var_26);
            var_29 = wp::mul(var_25, var_28);
        }
        var_30 = wp::select(var_27, var_25, var_29);
        // x_out[tid] = x_new                                                                     <L 162>
        wp::array_store(var_x_out, var_0, var_23);
        // v_out[tid] = v_new                                                                     <L 163>
        wp::array_store(var_v_out, var_0, var_30);
    }
}

extern "C" __global__ void my_apply_particle_deltas_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float32>> var_x_orig,
    wp::array_t<wp::vec_t<3,wp::float32>> var_x_pred,
    wp::array_t<wp::uint32> var_particle_flags,
    wp::array_t<wp::vec_t<3,wp::float32>> var_delta,
    wp::float32 var_dt,
    wp::float32 var_v_max,
    wp::array_t<wp::vec_t<3,wp::float32>> var_x_out,
    wp::array_t<wp::vec_t<3,wp::float32>> var_v_out,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_x_orig,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_x_pred,
    wp::array_t<wp::uint32> adj_particle_flags,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_delta,
    wp::float32 adj_dt,
    wp::float32 adj_v_max,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_x_out,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_v_out)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        wp::uint32* var_1;
        const wp::uint32 var_2 = 1;
        wp::uint32 var_3;
        wp::uint32 var_4;
        const wp::int32 var_5 = 0;
        bool var_6;
        wp::vec_t<3,wp::float32>* var_7;
        wp::vec_t<3,wp::float32> var_8;
        wp::vec_t<3,wp::float32> var_9;
        wp::vec_t<3,wp::float32>* var_10;
        wp::vec_t<3,wp::float32> var_11;
        wp::vec_t<3,wp::float32> var_12;
        wp::vec_t<3,wp::float32>* var_13;
        wp::vec_t<3,wp::float32> var_14;
        wp::vec_t<3,wp::float32> var_15;
        bool var_16;
        const wp::float32 var_17 = 0.0;
        wp::float32* var_18;
        const wp::int32 var_19 = 1;
        wp::float32* var_20;
        const wp::int32 var_21 = 2;
        wp::float32* var_22;
        wp::vec_t<3,wp::float32> var_23;
        wp::vec_t<3,wp::float32> var_24;
        wp::vec_t<3,wp::float32> var_25;
        wp::float32 var_26;
        bool var_27;
        wp::float32 var_28;
        wp::vec_t<3,wp::float32> var_29;
        wp::vec_t<3,wp::float32> var_30;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::uint32 adj_1 = {};
        wp::uint32 adj_2 = {};
        wp::uint32 adj_3 = {};
        wp::uint32 adj_4 = {};
        wp::int32 adj_5 = {};
        bool adj_6 = {};
        wp::vec_t<3,wp::float32> adj_7 = {};
        wp::vec_t<3,wp::float32> adj_8 = {};
        wp::vec_t<3,wp::float32> adj_9 = {};
        wp::vec_t<3,wp::float32> adj_10 = {};
        wp::vec_t<3,wp::float32> adj_11 = {};
        wp::vec_t<3,wp::float32> adj_12 = {};
        wp::vec_t<3,wp::float32> adj_13 = {};
        wp::vec_t<3,wp::float32> adj_14 = {};
        wp::vec_t<3,wp::float32> adj_15 = {};
        bool adj_16 = {};
        wp::float32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::int32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::int32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::vec_t<3,wp::float32> adj_23 = {};
        wp::vec_t<3,wp::float32> adj_24 = {};
        wp::vec_t<3,wp::float32> adj_25 = {};
        wp::float32 adj_26 = {};
        bool adj_27 = {};
        wp::float32 adj_28 = {};
        wp::vec_t<3,wp::float32> adj_29 = {};
        wp::vec_t<3,wp::float32> adj_30 = {};
        //---------
        // forward
        // def my_apply_particle_deltas(                                                          <L 136>
        // tid = wp.tid()                                                                         <L 146>
        var_0 = builtin_tid1d();
        // if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:                                  <L 147>
        var_1 = wp::address(var_particle_flags, var_0);
        var_3 = wp::load(var_1);
        var_4 = wp::bit_and(var_3, var_2);
        var_6 = (var_4 == var_5);
        if (var_6) {
            // return                                                                             <L 148>
            goto label0;
        }
        // x0 = x_orig[tid]                                                                       <L 149>
        var_7 = wp::address(var_x_orig, var_0);
        var_8 = wp::load(var_7);
        var_9 = wp::copy(var_8);
        // xp = x_pred[tid]                                                                       <L 150>
        var_10 = wp::address(var_x_pred, var_0);
        var_11 = wp::load(var_10);
        var_12 = wp::copy(var_11);
        // d = delta[tid]                                                                         <L 151>
        var_13 = wp::address(var_delta, var_0);
        var_14 = wp::load(var_13);
        var_15 = wp::copy(var_14);
        // if wp.isnan(d):                                                                        <L 152>
        var_16 = wp::isnan(var_15);
        if (var_16) {
            // d[0]=0.0                                                                           <L 153>
            // var_18 = wp::index(var_15, var_5);
            // wp::store(var_18, var_17);
            // d[1]=0.0                                                                           <L 154>
            // var_20 = wp::index(var_15, var_19);
            // wp::store(var_20, var_17);
            // d[2]=0.0                                                                           <L 155>
            // var_22 = wp::index(var_15, var_21);
            // wp::store(var_22, var_17);
        }
        // x_new = xp + d                                                                         <L 157>
        var_23 = wp::add(var_12, var_15);
        // v_new = (x_new - x0) / dt                                                              <L 158>
        var_24 = wp::sub(var_23, var_9);
        var_25 = wp::div(var_24, var_dt);
        // v_new_mag = wp.length(v_new)                                                           <L 159>
        var_26 = wp::length(var_25);
        // if v_new_mag > v_max:                                                                  <L 160>
        var_27 = (var_26 > var_v_max);
        if (var_27) {
            // v_new *= v_max / v_new_mag                                                         <L 161>
            var_28 = wp::div(var_v_max, var_26);
            var_29 = wp::mul(var_25, var_28);
        }
        var_30 = wp::select(var_27, var_25, var_29);
        // x_out[tid] = x_new                                                                     <L 162>
        // wp::array_store(var_x_out, var_0, var_23);
        // v_out[tid] = v_new                                                                     <L 163>
        // wp::array_store(var_v_out, var_0, var_30);
        //---------
        // reverse
        wp::adj_array_store(var_v_out, var_0, var_30, adj_v_out, adj_0, adj_30);
        // adj: v_out[tid] = v_new                                                                <L 163>
        wp::adj_array_store(var_x_out, var_0, var_23, adj_x_out, adj_0, adj_23);
        // adj: x_out[tid] = x_new                                                                <L 162>
        wp::adj_select(var_27, var_25, var_29, adj_27, adj_25, adj_29, adj_30);
        if (var_27) {
            wp::adj_mul(var_25, var_28, adj_25, adj_28, adj_29);
            wp::adj_div(var_v_max, var_26, var_28, adj_v_max, adj_26, adj_28);
            // adj: v_new *= v_max / v_new_mag                                                    <L 161>
        }
        // adj: if v_new_mag > v_max:                                                             <L 160>
        wp::adj_length(var_25, var_26, adj_25, adj_26);
        // adj: v_new_mag = wp.length(v_new)                                                      <L 159>
        wp::adj_div(var_24, var_dt, adj_24, adj_dt, adj_25);
        wp::adj_sub(var_23, var_9, adj_23, adj_9, adj_24);
        // adj: v_new = (x_new - x0) / dt                                                         <L 158>
        wp::adj_add(var_12, var_15, adj_12, adj_15, adj_23);
        // adj: x_new = xp + d                                                                    <L 157>
        if (var_16) {
            wp::adj_store(var_22, var_17, adj_22, adj_17);
            wp::adj_index(var_15, var_21, adj_15, adj_21, adj_22);
            // adj: d[2]=0.0                                                                      <L 155>
            wp::adj_store(var_20, var_17, adj_20, adj_17);
            wp::adj_index(var_15, var_19, adj_15, adj_19, adj_20);
            // adj: d[1]=0.0                                                                      <L 154>
            wp::adj_store(var_18, var_17, adj_18, adj_17);
            wp::adj_index(var_15, var_5, adj_15, adj_5, adj_18);
            // adj: d[0]=0.0                                                                      <L 153>
        }
        wp::adj_isnan(var_15, adj_15, adj_16);
        // adj: if wp.isnan(d):                                                                   <L 152>
        wp::adj_copy(var_14, adj_13, adj_15);
        wp::adj_load(var_13, adj_13, adj_14);
        wp::adj_address(var_delta, var_0, adj_delta, adj_0, adj_13);
        // adj: d = delta[tid]                                                                    <L 151>
        wp::adj_copy(var_11, adj_10, adj_12);
        wp::adj_load(var_10, adj_10, adj_11);
        wp::adj_address(var_x_pred, var_0, adj_x_pred, adj_0, adj_10);
        // adj: xp = x_pred[tid]                                                                  <L 150>
        wp::adj_copy(var_8, adj_7, adj_9);
        wp::adj_load(var_7, adj_7, adj_8);
        wp::adj_address(var_x_orig, var_0, adj_x_orig, adj_0, adj_7);
        // adj: x0 = x_orig[tid]                                                                  <L 149>
        if (var_6) {
            label0:;
            // adj: return                                                                        <L 148>
        }
        wp::adj_bit_and(var_3, var_2, adj_1, adj_2, adj_4);
        wp::adj_load(var_1, adj_1, adj_3);
        wp::adj_address(var_particle_flags, var_0, adj_particle_flags, adj_0, adj_1);
        // adj: if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:                             <L 147>
        // adj: tid = wp.tid()                                                                    <L 146>
        // adj: def my_apply_particle_deltas(                                                     <L 136>
        continue;
    }
}



extern "C" __global__ void my_solve_particle_ground_contacts_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_x,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_v,
    wp::array_t<wp::float32> var_invmass,
    wp::array_t<wp::float32> var_particle_radius,
    wp::array_t<wp::uint32> var_particle_flags,
    wp::float32 var_ke,
    wp::float32 var_kd,
    wp::float32 var_kf,
    wp::float32 var_mu,
    wp::float32 var_staticGroundVelocityThresholdRatio,
    wp::float32 var_staticGroundFriction,
    wp::array_t<wp::float32> var_ground,
    wp::float32 var_dt,
    wp::float32 var_relaxation,
    wp::array_t<wp::vec_t<3,wp::float32>> var_delta)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        wp::uint32* var_1;
        const wp::uint32 var_2 = 1;
        wp::uint32 var_3;
        wp::uint32 var_4;
        const wp::int32 var_5 = 0;
        bool var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        const wp::float32 var_10 = 0.0;
        bool var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::vec_t<3,wp::float32>* var_15;
        wp::vec_t<3,wp::float32> var_16;
        wp::vec_t<3,wp::float32> var_17;
        wp::vec_t<3,wp::float32>* var_18;
        wp::vec_t<3,wp::float32> var_19;
        wp::vec_t<3,wp::float32> var_20;
        wp::float32* var_21;
        const wp::int32 var_22 = 1;
        wp::float32* var_23;
        const wp::int32 var_24 = 2;
        wp::float32* var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::vec_t<3,wp::float32> var_29;
        wp::float32 var_30;
        const wp::int32 var_31 = 3;
        wp::float32* var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        wp::float32* var_35;
        wp::float32 var_36;
        wp::float32 var_37;
        wp::float32 var_38;
        bool var_39;
        wp::float32 var_40;
        wp::vec_t<3,wp::float32> var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        bool var_44;
        wp::float32 var_45;
        wp::float32 var_46;
        wp::vec_t<3,wp::float32> var_47;
        wp::vec_t<3,wp::float32> var_48;
        wp::float32 var_49;
        wp::float32 var_50;
        wp::float32 var_51;
        wp::float32 var_52;
        wp::float32 var_53;
        wp::vec_t<3,wp::float32> var_54;
        wp::vec_t<3,wp::float32> var_55;
        wp::vec_t<3,wp::float32> var_56;
        wp::vec_t<3,wp::float32> var_57;
        wp::vec_t<3,wp::float32> var_58;
        wp::vec_t<3,wp::float32> var_59;
        wp::vec_t<3,wp::float32> var_60;
        //---------
        // forward
        // def my_solve_particle_ground_contacts(                                                 <L 166>
        // tid = wp.tid()                                                                         <L 183>
        var_0 = builtin_tid1d();
        // if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:                                  <L 184>
        var_1 = wp::address(var_particle_flags, var_0);
        var_3 = wp::load(var_1);
        var_4 = wp::bit_and(var_3, var_2);
        var_6 = (var_4 == var_5);
        if (var_6) {
            // return                                                                             <L 185>
            return;
        }
        // wi = invmass[tid]                                                                      <L 186>
        var_7 = wp::address(var_invmass, var_0);
        var_8 = wp::load(var_7);
        var_9 = wp::copy(var_8);
        // if wi == 0.0:                                                                          <L 187>
        var_11 = (var_9 == var_10);
        if (var_11) {
            // return                                                                             <L 188>
            return;
        }
        // radius = particle_radius[tid]                                                          <L 189>
        var_12 = wp::address(var_particle_radius, var_0);
        var_13 = wp::load(var_12);
        var_14 = wp::copy(var_13);
        // x = particle_x[tid]                                                                    <L 190>
        var_15 = wp::address(var_particle_x, var_0);
        var_16 = wp::load(var_15);
        var_17 = wp::copy(var_16);
        // v = particle_v[tid]                                                                    <L 191>
        var_18 = wp::address(var_particle_v, var_0);
        var_19 = wp::load(var_18);
        var_20 = wp::copy(var_19);
        // n = wp.vec3(ground[0], ground[1], ground[2])                                           <L 192>
        var_21 = wp::address(var_ground, var_5);
        var_23 = wp::address(var_ground, var_22);
        var_25 = wp::address(var_ground, var_24);
        var_26 = wp::load(var_21);
        var_27 = wp::load(var_23);
        var_28 = wp::load(var_25);
        var_29 = wp::vec_t<3,wp::float32>(var_26, var_27, var_28);
        // c = wp.min(wp.dot(n, x) + ground[3] - particle_radius[tid], 0.0)                       <L 193>
        var_30 = wp::dot(var_29, var_17);
        var_32 = wp::address(var_ground, var_31);
        var_33 = wp::load(var_32);
        var_34 = wp::add(var_30, var_33);
        var_35 = wp::address(var_particle_radius, var_0);
        var_36 = wp::load(var_35);
        var_37 = wp::sub(var_34, var_36);
        var_38 = wp::min(var_37, var_10);
        // if c > 0.0:                                                                            <L 194>
        var_39 = (var_38 > var_10);
        if (var_39) {
            // return                                                                             <L 195>
            return;
        }
        // lambda_n = c                                                                           <L 196>
        var_40 = wp::copy(var_38);
        // delta_n = n * lambda_n                                                                 <L 197>
        var_41 = wp::mul(var_29, var_40);
        // vn = wp.dot(n, v)                                                                      <L 198>
        var_42 = wp::dot(var_29, var_20);
        // if wp.abs(vn)<staticGroundVelocityThresholdRatio:                                      <L 199>
        var_43 = wp::abs(var_42);
        var_44 = (var_43 < var_staticGroundVelocityThresholdRatio);
        if (var_44) {
            // mu=staticGroundFriction                                                            <L 200>
            var_45 = wp::copy(var_staticGroundFriction);
        }
        var_46 = wp::select(var_44, var_mu, var_45);
        // vt = v - n * vn                                                                        <L 201>
        var_47 = wp::mul(var_29, var_42);
        var_48 = wp::sub(var_20, var_47);
        // lambda_f = wp.max(mu * lambda_n, 0.0 - wp.length(vt) * dt)                             <L 202>
        var_49 = wp::mul(var_46, var_40);
        var_50 = wp::length(var_48);
        var_51 = wp::mul(var_50, var_dt);
        var_52 = wp::sub(var_10, var_51);
        var_53 = wp::max(var_49, var_52);
        // delta_f = wp.normalize(vt) * lambda_f                                                  <L 203>
        var_54 = wp::normalize(var_48);
        var_55 = wp::mul(var_54, var_53);
        // wp.atomic_add(delta, tid, (delta_f - delta_n) / wi * relaxation * wi)                  <L 204>
        var_56 = wp::sub(var_55, var_41);
        var_57 = wp::div(var_56, var_9);
        var_58 = wp::mul(var_57, var_relaxation);
        var_59 = wp::mul(var_58, var_9);
        var_60 = wp::atomic_add(var_delta, var_0, var_59);
    }
}

extern "C" __global__ void my_solve_particle_ground_contacts_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_x,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_v,
    wp::array_t<wp::float32> var_invmass,
    wp::array_t<wp::float32> var_particle_radius,
    wp::array_t<wp::uint32> var_particle_flags,
    wp::float32 var_ke,
    wp::float32 var_kd,
    wp::float32 var_kf,
    wp::float32 var_mu,
    wp::float32 var_staticGroundVelocityThresholdRatio,
    wp::float32 var_staticGroundFriction,
    wp::array_t<wp::float32> var_ground,
    wp::float32 var_dt,
    wp::float32 var_relaxation,
    wp::array_t<wp::vec_t<3,wp::float32>> var_delta,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_particle_x,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_particle_v,
    wp::array_t<wp::float32> adj_invmass,
    wp::array_t<wp::float32> adj_particle_radius,
    wp::array_t<wp::uint32> adj_particle_flags,
    wp::float32 adj_ke,
    wp::float32 adj_kd,
    wp::float32 adj_kf,
    wp::float32 adj_mu,
    wp::float32 adj_staticGroundVelocityThresholdRatio,
    wp::float32 adj_staticGroundFriction,
    wp::array_t<wp::float32> adj_ground,
    wp::float32 adj_dt,
    wp::float32 adj_relaxation,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_delta)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        wp::uint32* var_1;
        const wp::uint32 var_2 = 1;
        wp::uint32 var_3;
        wp::uint32 var_4;
        const wp::int32 var_5 = 0;
        bool var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        const wp::float32 var_10 = 0.0;
        bool var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::vec_t<3,wp::float32>* var_15;
        wp::vec_t<3,wp::float32> var_16;
        wp::vec_t<3,wp::float32> var_17;
        wp::vec_t<3,wp::float32>* var_18;
        wp::vec_t<3,wp::float32> var_19;
        wp::vec_t<3,wp::float32> var_20;
        wp::float32* var_21;
        const wp::int32 var_22 = 1;
        wp::float32* var_23;
        const wp::int32 var_24 = 2;
        wp::float32* var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::vec_t<3,wp::float32> var_29;
        wp::float32 var_30;
        const wp::int32 var_31 = 3;
        wp::float32* var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        wp::float32* var_35;
        wp::float32 var_36;
        wp::float32 var_37;
        wp::float32 var_38;
        bool var_39;
        wp::float32 var_40;
        wp::vec_t<3,wp::float32> var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        bool var_44;
        wp::float32 var_45;
        wp::float32 var_46;
        wp::vec_t<3,wp::float32> var_47;
        wp::vec_t<3,wp::float32> var_48;
        wp::float32 var_49;
        wp::float32 var_50;
        wp::float32 var_51;
        wp::float32 var_52;
        wp::float32 var_53;
        wp::vec_t<3,wp::float32> var_54;
        wp::vec_t<3,wp::float32> var_55;
        wp::vec_t<3,wp::float32> var_56;
        wp::vec_t<3,wp::float32> var_57;
        wp::vec_t<3,wp::float32> var_58;
        wp::vec_t<3,wp::float32> var_59;
        wp::vec_t<3,wp::float32> var_60;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::uint32 adj_1 = {};
        wp::uint32 adj_2 = {};
        wp::uint32 adj_3 = {};
        wp::uint32 adj_4 = {};
        wp::int32 adj_5 = {};
        bool adj_6 = {};
        wp::float32 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        bool adj_11 = {};
        wp::float32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::vec_t<3,wp::float32> adj_15 = {};
        wp::vec_t<3,wp::float32> adj_16 = {};
        wp::vec_t<3,wp::float32> adj_17 = {};
        wp::vec_t<3,wp::float32> adj_18 = {};
        wp::vec_t<3,wp::float32> adj_19 = {};
        wp::vec_t<3,wp::float32> adj_20 = {};
        wp::float32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::float32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::float32 adj_27 = {};
        wp::float32 adj_28 = {};
        wp::vec_t<3,wp::float32> adj_29 = {};
        wp::float32 adj_30 = {};
        wp::int32 adj_31 = {};
        wp::float32 adj_32 = {};
        wp::float32 adj_33 = {};
        wp::float32 adj_34 = {};
        wp::float32 adj_35 = {};
        wp::float32 adj_36 = {};
        wp::float32 adj_37 = {};
        wp::float32 adj_38 = {};
        bool adj_39 = {};
        wp::float32 adj_40 = {};
        wp::vec_t<3,wp::float32> adj_41 = {};
        wp::float32 adj_42 = {};
        wp::float32 adj_43 = {};
        bool adj_44 = {};
        wp::float32 adj_45 = {};
        wp::float32 adj_46 = {};
        wp::vec_t<3,wp::float32> adj_47 = {};
        wp::vec_t<3,wp::float32> adj_48 = {};
        wp::float32 adj_49 = {};
        wp::float32 adj_50 = {};
        wp::float32 adj_51 = {};
        wp::float32 adj_52 = {};
        wp::float32 adj_53 = {};
        wp::vec_t<3,wp::float32> adj_54 = {};
        wp::vec_t<3,wp::float32> adj_55 = {};
        wp::vec_t<3,wp::float32> adj_56 = {};
        wp::vec_t<3,wp::float32> adj_57 = {};
        wp::vec_t<3,wp::float32> adj_58 = {};
        wp::vec_t<3,wp::float32> adj_59 = {};
        wp::vec_t<3,wp::float32> adj_60 = {};
        //---------
        // forward
        // def my_solve_particle_ground_contacts(                                                 <L 166>
        // tid = wp.tid()                                                                         <L 183>
        var_0 = builtin_tid1d();
        // if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:                                  <L 184>
        var_1 = wp::address(var_particle_flags, var_0);
        var_3 = wp::load(var_1);
        var_4 = wp::bit_and(var_3, var_2);
        var_6 = (var_4 == var_5);
        if (var_6) {
            // return                                                                             <L 185>
            goto label0;
        }
        // wi = invmass[tid]                                                                      <L 186>
        var_7 = wp::address(var_invmass, var_0);
        var_8 = wp::load(var_7);
        var_9 = wp::copy(var_8);
        // if wi == 0.0:                                                                          <L 187>
        var_11 = (var_9 == var_10);
        if (var_11) {
            // return                                                                             <L 188>
            goto label1;
        }
        // radius = particle_radius[tid]                                                          <L 189>
        var_12 = wp::address(var_particle_radius, var_0);
        var_13 = wp::load(var_12);
        var_14 = wp::copy(var_13);
        // x = particle_x[tid]                                                                    <L 190>
        var_15 = wp::address(var_particle_x, var_0);
        var_16 = wp::load(var_15);
        var_17 = wp::copy(var_16);
        // v = particle_v[tid]                                                                    <L 191>
        var_18 = wp::address(var_particle_v, var_0);
        var_19 = wp::load(var_18);
        var_20 = wp::copy(var_19);
        // n = wp.vec3(ground[0], ground[1], ground[2])                                           <L 192>
        var_21 = wp::address(var_ground, var_5);
        var_23 = wp::address(var_ground, var_22);
        var_25 = wp::address(var_ground, var_24);
        var_26 = wp::load(var_21);
        var_27 = wp::load(var_23);
        var_28 = wp::load(var_25);
        var_29 = wp::vec_t<3,wp::float32>(var_26, var_27, var_28);
        // c = wp.min(wp.dot(n, x) + ground[3] - particle_radius[tid], 0.0)                       <L 193>
        var_30 = wp::dot(var_29, var_17);
        var_32 = wp::address(var_ground, var_31);
        var_33 = wp::load(var_32);
        var_34 = wp::add(var_30, var_33);
        var_35 = wp::address(var_particle_radius, var_0);
        var_36 = wp::load(var_35);
        var_37 = wp::sub(var_34, var_36);
        var_38 = wp::min(var_37, var_10);
        // if c > 0.0:                                                                            <L 194>
        var_39 = (var_38 > var_10);
        if (var_39) {
            // return                                                                             <L 195>
            goto label2;
        }
        // lambda_n = c                                                                           <L 196>
        var_40 = wp::copy(var_38);
        // delta_n = n * lambda_n                                                                 <L 197>
        var_41 = wp::mul(var_29, var_40);
        // vn = wp.dot(n, v)                                                                      <L 198>
        var_42 = wp::dot(var_29, var_20);
        // if wp.abs(vn)<staticGroundVelocityThresholdRatio:                                      <L 199>
        var_43 = wp::abs(var_42);
        var_44 = (var_43 < var_staticGroundVelocityThresholdRatio);
        if (var_44) {
            // mu=staticGroundFriction                                                            <L 200>
            var_45 = wp::copy(var_staticGroundFriction);
        }
        var_46 = wp::select(var_44, var_mu, var_45);
        // vt = v - n * vn                                                                        <L 201>
        var_47 = wp::mul(var_29, var_42);
        var_48 = wp::sub(var_20, var_47);
        // lambda_f = wp.max(mu * lambda_n, 0.0 - wp.length(vt) * dt)                             <L 202>
        var_49 = wp::mul(var_46, var_40);
        var_50 = wp::length(var_48);
        var_51 = wp::mul(var_50, var_dt);
        var_52 = wp::sub(var_10, var_51);
        var_53 = wp::max(var_49, var_52);
        // delta_f = wp.normalize(vt) * lambda_f                                                  <L 203>
        var_54 = wp::normalize(var_48);
        var_55 = wp::mul(var_54, var_53);
        // wp.atomic_add(delta, tid, (delta_f - delta_n) / wi * relaxation * wi)                  <L 204>
        var_56 = wp::sub(var_55, var_41);
        var_57 = wp::div(var_56, var_9);
        var_58 = wp::mul(var_57, var_relaxation);
        var_59 = wp::mul(var_58, var_9);
        // var_60 = wp::atomic_add(var_delta, var_0, var_59);
        //---------
        // reverse
        wp::adj_atomic_add(var_delta, var_0, var_59, adj_delta, adj_0, adj_59, adj_60);
        wp::adj_mul(var_58, var_9, adj_58, adj_9, adj_59);
        wp::adj_mul(var_57, var_relaxation, adj_57, adj_relaxation, adj_58);
        wp::adj_div(var_56, var_9, adj_56, adj_9, adj_57);
        wp::adj_sub(var_55, var_41, adj_55, adj_41, adj_56);
        // adj: wp.atomic_add(delta, tid, (delta_f - delta_n) / wi * relaxation * wi)             <L 204>
        wp::adj_mul(var_54, var_53, adj_54, adj_53, adj_55);
        wp::adj_normalize(var_48, var_54, adj_48, adj_54);
        // adj: delta_f = wp.normalize(vt) * lambda_f                                             <L 203>
        wp::adj_max(var_49, var_52, adj_49, adj_52, adj_53);
        wp::adj_sub(var_10, var_51, adj_10, adj_51, adj_52);
        wp::adj_mul(var_50, var_dt, adj_50, adj_dt, adj_51);
        wp::adj_length(var_48, var_50, adj_48, adj_50);
        wp::adj_mul(var_46, var_40, adj_46, adj_40, adj_49);
        // adj: lambda_f = wp.max(mu * lambda_n, 0.0 - wp.length(vt) * dt)                        <L 202>
        wp::adj_sub(var_20, var_47, adj_20, adj_47, adj_48);
        wp::adj_mul(var_29, var_42, adj_29, adj_42, adj_47);
        // adj: vt = v - n * vn                                                                   <L 201>
        wp::adj_select(var_44, var_mu, var_45, adj_44, adj_mu, adj_45, adj_46);
        if (var_44) {
            wp::adj_copy(var_staticGroundFriction, adj_staticGroundFriction, adj_45);
            // adj: mu=staticGroundFriction                                                       <L 200>
        }
        wp::adj_abs(var_42, adj_42, adj_43);
        // adj: if wp.abs(vn)<staticGroundVelocityThresholdRatio:                                 <L 199>
        wp::adj_dot(var_29, var_20, adj_29, adj_20, adj_42);
        // adj: vn = wp.dot(n, v)                                                                 <L 198>
        wp::adj_mul(var_29, var_40, adj_29, adj_40, adj_41);
        // adj: delta_n = n * lambda_n                                                            <L 197>
        wp::adj_copy(var_38, adj_38, adj_40);
        // adj: lambda_n = c                                                                      <L 196>
        if (var_39) {
            label2:;
            // adj: return                                                                        <L 195>
        }
        // adj: if c > 0.0:                                                                       <L 194>
        wp::adj_min(var_37, var_10, adj_37, adj_10, adj_38);
        wp::adj_sub(var_34, var_36, adj_34, adj_35, adj_37);
        wp::adj_load(var_35, adj_35, adj_36);
        wp::adj_address(var_particle_radius, var_0, adj_particle_radius, adj_0, adj_35);
        wp::adj_add(var_30, var_33, adj_30, adj_32, adj_34);
        wp::adj_load(var_32, adj_32, adj_33);
        wp::adj_address(var_ground, var_31, adj_ground, adj_31, adj_32);
        wp::adj_dot(var_29, var_17, adj_29, adj_17, adj_30);
        // adj: c = wp.min(wp.dot(n, x) + ground[3] - particle_radius[tid], 0.0)                  <L 193>
        wp::adj_vec_t(var_26, var_27, var_28, adj_21, adj_23, adj_25, adj_29);
        wp::adj_load(var_25, adj_25, adj_28);
        wp::adj_load(var_23, adj_23, adj_27);
        wp::adj_load(var_21, adj_21, adj_26);
        wp::adj_address(var_ground, var_24, adj_ground, adj_24, adj_25);
        wp::adj_address(var_ground, var_22, adj_ground, adj_22, adj_23);
        wp::adj_address(var_ground, var_5, adj_ground, adj_5, adj_21);
        // adj: n = wp.vec3(ground[0], ground[1], ground[2])                                      <L 192>
        wp::adj_copy(var_19, adj_18, adj_20);
        wp::adj_load(var_18, adj_18, adj_19);
        wp::adj_address(var_particle_v, var_0, adj_particle_v, adj_0, adj_18);
        // adj: v = particle_v[tid]                                                               <L 191>
        wp::adj_copy(var_16, adj_15, adj_17);
        wp::adj_load(var_15, adj_15, adj_16);
        wp::adj_address(var_particle_x, var_0, adj_particle_x, adj_0, adj_15);
        // adj: x = particle_x[tid]                                                               <L 190>
        wp::adj_copy(var_13, adj_12, adj_14);
        wp::adj_load(var_12, adj_12, adj_13);
        wp::adj_address(var_particle_radius, var_0, adj_particle_radius, adj_0, adj_12);
        // adj: radius = particle_radius[tid]                                                     <L 189>
        if (var_11) {
            label1:;
            // adj: return                                                                        <L 188>
        }
        // adj: if wi == 0.0:                                                                     <L 187>
        wp::adj_copy(var_8, adj_7, adj_9);
        wp::adj_load(var_7, adj_7, adj_8);
        wp::adj_address(var_invmass, var_0, adj_invmass, adj_0, adj_7);
        // adj: wi = invmass[tid]                                                                 <L 186>
        if (var_6) {
            label0:;
            // adj: return                                                                        <L 185>
        }
        wp::adj_bit_and(var_3, var_2, adj_1, adj_2, adj_4);
        wp::adj_load(var_1, adj_1, adj_3);
        wp::adj_address(var_particle_flags, var_0, adj_particle_flags, adj_0, adj_1);
        // adj: if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:                             <L 184>
        // adj: tid = wp.tid()                                                                    <L 183>
        // adj: def my_solve_particle_ground_contacts(                                            <L 166>
        continue;
    }
}



extern "C" __global__ void my_solve_particle_particle_contacts_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::uint64 var_grid,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_x,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_v,
    wp::array_t<wp::float32> var_particle_invmass,
    wp::array_t<wp::float32> var_particle_radius,
    wp::array_t<wp::uint32> var_particle_flags,
    wp::float32 var_k_mu,
    wp::float32 var_staticParticleVelocityThresholdRatio,
    wp::float32 var_staticParticleFriction,
    wp::float32 var_k_cohesion,
    wp::float32 var_max_radius,
    wp::float32 var_dt,
    wp::float32 var_relaxation,
    wp::array_t<wp::vec_t<3,wp::float32>> var_deltas)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        const wp::int32 var_2 = 1;
        wp::int32 var_3;
        bool var_4;
        wp::uint32* var_5;
        const wp::uint32 var_6 = 1;
        wp::uint32 var_7;
        wp::uint32 var_8;
        const wp::int32 var_9 = 0;
        bool var_10;
        wp::vec_t<3,wp::float32>* var_11;
        wp::vec_t<3,wp::float32> var_12;
        wp::vec_t<3,wp::float32> var_13;
        wp::vec_t<3,wp::float32>* var_14;
        wp::vec_t<3,wp::float32> var_15;
        wp::vec_t<3,wp::float32> var_16;
        wp::float32* var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32* var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        wp::hash_grid_query_t var_25;
        wp::int32 var_26;
        const wp::float32 var_27 = 0.0;
        wp::vec_t<3,wp::float32> var_28;
        bool var_29;
        wp::uint32* var_30;
        wp::uint32 var_31;
        wp::uint32 var_32;
        bool var_33;
        bool var_34;
        bool var_35;
        wp::vec_t<3,wp::float32>* var_36;
        wp::vec_t<3,wp::float32> var_37;
        wp::vec_t<3,wp::float32> var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        wp::float32* var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::float32* var_44;
        wp::float32 var_45;
        wp::float32 var_46;
        wp::float32 var_47;
        bool var_48;
        bool var_49;
        bool var_50;
        wp::vec_t<3,wp::float32> var_51;
        wp::vec_t<3,wp::float32>* var_52;
        wp::vec_t<3,wp::float32> var_53;
        wp::vec_t<3,wp::float32> var_54;
        wp::float32 var_55;
        wp::vec_t<3,wp::float32> var_56;
        wp::float32 var_57;
        wp::float32 var_58;
        bool var_59;
        wp::float32 var_60;
        wp::float32 var_61;
        wp::vec_t<3,wp::float32> var_62;
        wp::vec_t<3,wp::float32> var_63;
        wp::float32 var_64;
        wp::float32 var_65;
        wp::float32 var_66;
        wp::float32 var_67;
        wp::float32 var_68;
        wp::vec_t<3,wp::float32> var_69;
        wp::vec_t<3,wp::float32> var_70;
        wp::vec_t<3,wp::float32> var_71;
        wp::vec_t<3,wp::float32> var_72;
        wp::vec_t<3,wp::float32> var_73;
        wp::vec_t<3,wp::float32> var_74;
        wp::float32 var_75;
        wp::vec_t<3,wp::float32> var_76;
        wp::vec_t<3,wp::float32> var_77;
        wp::float32 var_78;
        wp::vec_t<3,wp::float32> var_79;
        wp::vec_t<3,wp::float32> var_80;
        wp::vec_t<3,wp::float32> var_81;
        //---------
        // forward
        // def my_solve_particle_particle_contacts(                                               <L 207>
        // tid = wp.tid()                                                                         <L 223>
        var_0 = builtin_tid1d();
        // i = wp.hash_grid_point_id(grid, tid)                                                   <L 224>
        var_1 = wp::hash_grid_point_id(var_grid, var_0);
        // if i == -1:                                                                            <L 225>
        var_3 = wp::neg(var_2);
        var_4 = (var_1 == var_3);
        if (var_4) {
            // return                                                                             <L 226>
            return;
        }
        // if (particle_flags[i] & PARTICLE_FLAG_ACTIVE) == 0:                                    <L 227>
        var_5 = wp::address(var_particle_flags, var_1);
        var_7 = wp::load(var_5);
        var_8 = wp::bit_and(var_7, var_6);
        var_10 = (var_8 == var_9);
        if (var_10) {
            // return                                                                             <L 228>
            return;
        }
        // x = particle_x[i]                                                                      <L 229>
        var_11 = wp::address(var_particle_x, var_1);
        var_12 = wp::load(var_11);
        var_13 = wp::copy(var_12);
        // v = particle_v[i]                                                                      <L 230>
        var_14 = wp::address(var_particle_v, var_1);
        var_15 = wp::load(var_14);
        var_16 = wp::copy(var_15);
        // radius = particle_radius[i]                                                            <L 231>
        var_17 = wp::address(var_particle_radius, var_1);
        var_18 = wp::load(var_17);
        var_19 = wp::copy(var_18);
        // w1 = particle_invmass[i]                                                               <L 232>
        var_20 = wp::address(var_particle_invmass, var_1);
        var_21 = wp::load(var_20);
        var_22 = wp::copy(var_21);
        // query = wp.hash_grid_query(grid, x, radius + max_radius + k_cohesion)                  <L 233>
        var_23 = wp::add(var_19, var_max_radius);
        var_24 = wp::add(var_23, var_k_cohesion);
        var_25 = wp::hash_grid_query(var_grid, var_13, var_24);
        // index = int(0)                                                                         <L 234>
        var_26 = wp::int(var_9);
        // delta = wp.vec3(0.0)                                                                   <L 235>
        var_28 = wp::vec_t<3,wp::float32>(var_27);
        // while wp.hash_grid_query_next(query, index):                                           <L 236>
        start_while_2:;
        var_29 = wp::hash_grid_query_next(var_25, var_26);
        if ((var_29) == false) goto end_while_2;
            // if (particle_flags[index] & PARTICLE_FLAG_ACTIVE) != 0 and index != i:             <L 237>
            var_30 = wp::address(var_particle_flags, var_26);
            var_31 = wp::load(var_30);
            var_32 = wp::bit_and(var_31, var_6);
            var_33 = (var_32 != var_9);
            var_34 = (var_26 != var_1);
            var_35 = var_33 && var_34;
            if (var_35) {
                // n = x - particle_x[index]                                                      <L 238>
                var_36 = wp::address(var_particle_x, var_26);
                var_37 = wp::load(var_36);
                var_38 = wp::sub(var_13, var_37);
                // d = wp.length(n)                                                               <L 239>
                var_39 = wp::length(var_38);
                // err = d - radius - particle_radius[index]                                      <L 240>
                var_40 = wp::sub(var_39, var_19);
                var_41 = wp::address(var_particle_radius, var_26);
                var_42 = wp::load(var_41);
                var_43 = wp::sub(var_40, var_42);
                // w2 = particle_invmass[index]                                                   <L 241>
                var_44 = wp::address(var_particle_invmass, var_26);
                var_45 = wp::load(var_44);
                var_46 = wp::copy(var_45);
                // denom = w1 + w2                                                                <L 242>
                var_47 = wp::add(var_22, var_46);
                // if err <= k_cohesion and denom > 0.0:                                          <L 243>
                var_48 = (var_43 <= var_k_cohesion);
                var_49 = (var_47 > var_27);
                var_50 = var_48 && var_49;
                if (var_50) {
                    // n = n / d                                                                  <L 244>
                    var_51 = wp::div(var_38, var_39);
                    // vrel = v - particle_v[index]                                               <L 245>
                    var_52 = wp::address(var_particle_v, var_26);
                    var_53 = wp::load(var_52);
                    var_54 = wp::sub(var_16, var_53);
                    // lambda_n = err                                                             <L 246>
                    var_55 = wp::copy(var_43);
                    // delta_n = n * lambda_n                                                     <L 247>
                    var_56 = wp::mul(var_51, var_55);
                    // vn = wp.dot(n, vrel)                                                       <L 248>
                    var_57 = wp::dot(var_51, var_54);
                    // if wp.abs(vn)<staticParticleVelocityThresholdRatio:                        <L 249>
                    var_58 = wp::abs(var_57);
                    var_59 = (var_58 < var_staticParticleVelocityThresholdRatio);
                    if (var_59) {
                        // k_mu=staticParticleFriction                                            <L 250>
                        var_60 = wp::copy(var_staticParticleFriction);
                    }
                    var_61 = wp::select(var_59, var_k_mu, var_60);
                    // vt = v - n * vn                                                            <L 251>
                    var_62 = wp::mul(var_51, var_57);
                    var_63 = wp::sub(var_16, var_62);
                    // lambda_f = wp.max(k_mu * lambda_n, -wp.length(vt) * dt)                    <L 252>
                    var_64 = wp::mul(var_61, var_55);
                    var_65 = wp::length(var_63);
                    var_66 = wp::neg(var_65);
                    var_67 = wp::mul(var_66, var_dt);
                    var_68 = wp::max(var_64, var_67);
                    // delta_f = wp.normalize(vt) * lambda_f                                      <L 253>
                    var_69 = wp::normalize(var_63);
                    var_70 = wp::mul(var_69, var_68);
                    // delta += (delta_f - delta_n) / denom * w1                                  <L 254>
                    var_71 = wp::sub(var_70, var_56);
                    var_72 = wp::div(var_71, var_47);
                    var_73 = wp::mul(var_72, var_22);
                    var_74 = wp::add(var_28, var_73);
                }
                var_75 = wp::select(var_50, var_k_mu, var_61);
                var_76 = wp::select(var_50, var_28, var_74);
                var_77 = wp::select(var_50, var_38, var_51);
            }
            var_78 = wp::select(var_35, var_k_mu, var_75);
            var_79 = wp::select(var_35, var_28, var_76);
            wp::assign(var_k_mu, var_78);
            wp::assign(var_28, var_79);
        goto start_while_2;
        end_while_2:;
        // wp.atomic_add(deltas, i, delta * relaxation)                                           <L 255>
        var_80 = wp::mul(var_28, var_relaxation);
        var_81 = wp::atomic_add(var_deltas, var_1, var_80);
    }
}

extern "C" __global__ void my_solve_particle_particle_contacts_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::uint64 var_grid,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_x,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_v,
    wp::array_t<wp::float32> var_particle_invmass,
    wp::array_t<wp::float32> var_particle_radius,
    wp::array_t<wp::uint32> var_particle_flags,
    wp::float32 var_k_mu,
    wp::float32 var_staticParticleVelocityThresholdRatio,
    wp::float32 var_staticParticleFriction,
    wp::float32 var_k_cohesion,
    wp::float32 var_max_radius,
    wp::float32 var_dt,
    wp::float32 var_relaxation,
    wp::array_t<wp::vec_t<3,wp::float32>> var_deltas,
    wp::uint64 adj_grid,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_particle_x,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_particle_v,
    wp::array_t<wp::float32> adj_particle_invmass,
    wp::array_t<wp::float32> adj_particle_radius,
    wp::array_t<wp::uint32> adj_particle_flags,
    wp::float32 adj_k_mu,
    wp::float32 adj_staticParticleVelocityThresholdRatio,
    wp::float32 adj_staticParticleFriction,
    wp::float32 adj_k_cohesion,
    wp::float32 adj_max_radius,
    wp::float32 adj_dt,
    wp::float32 adj_relaxation,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_deltas)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        const wp::int32 var_2 = 1;
        wp::int32 var_3;
        bool var_4;
        wp::uint32* var_5;
        const wp::uint32 var_6 = 1;
        wp::uint32 var_7;
        wp::uint32 var_8;
        const wp::int32 var_9 = 0;
        bool var_10;
        wp::vec_t<3,wp::float32>* var_11;
        wp::vec_t<3,wp::float32> var_12;
        wp::vec_t<3,wp::float32> var_13;
        wp::vec_t<3,wp::float32>* var_14;
        wp::vec_t<3,wp::float32> var_15;
        wp::vec_t<3,wp::float32> var_16;
        wp::float32* var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32* var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        wp::hash_grid_query_t var_25;
        wp::int32 var_26;
        const wp::float32 var_27 = 0.0;
        wp::vec_t<3,wp::float32> var_28;
        bool var_29;
        wp::uint32* var_30;
        wp::uint32 var_31;
        wp::uint32 var_32;
        bool var_33;
        bool var_34;
        bool var_35;
        wp::vec_t<3,wp::float32>* var_36;
        wp::vec_t<3,wp::float32> var_37;
        wp::vec_t<3,wp::float32> var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        wp::float32* var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::float32* var_44;
        wp::float32 var_45;
        wp::float32 var_46;
        wp::float32 var_47;
        bool var_48;
        bool var_49;
        bool var_50;
        wp::vec_t<3,wp::float32> var_51;
        wp::vec_t<3,wp::float32>* var_52;
        wp::vec_t<3,wp::float32> var_53;
        wp::vec_t<3,wp::float32> var_54;
        wp::float32 var_55;
        wp::vec_t<3,wp::float32> var_56;
        wp::float32 var_57;
        wp::float32 var_58;
        bool var_59;
        wp::float32 var_60;
        wp::float32 var_61;
        wp::vec_t<3,wp::float32> var_62;
        wp::vec_t<3,wp::float32> var_63;
        wp::float32 var_64;
        wp::float32 var_65;
        wp::float32 var_66;
        wp::float32 var_67;
        wp::float32 var_68;
        wp::vec_t<3,wp::float32> var_69;
        wp::vec_t<3,wp::float32> var_70;
        wp::vec_t<3,wp::float32> var_71;
        wp::vec_t<3,wp::float32> var_72;
        wp::vec_t<3,wp::float32> var_73;
        wp::vec_t<3,wp::float32> var_74;
        wp::float32 var_75;
        wp::vec_t<3,wp::float32> var_76;
        wp::vec_t<3,wp::float32> var_77;
        wp::float32 var_78;
        wp::vec_t<3,wp::float32> var_79;
        wp::vec_t<3,wp::float32> var_80;
        wp::vec_t<3,wp::float32> var_81;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        bool adj_4 = {};
        wp::uint32 adj_5 = {};
        wp::uint32 adj_6 = {};
        wp::uint32 adj_7 = {};
        wp::uint32 adj_8 = {};
        wp::int32 adj_9 = {};
        bool adj_10 = {};
        wp::vec_t<3,wp::float32> adj_11 = {};
        wp::vec_t<3,wp::float32> adj_12 = {};
        wp::vec_t<3,wp::float32> adj_13 = {};
        wp::vec_t<3,wp::float32> adj_14 = {};
        wp::vec_t<3,wp::float32> adj_15 = {};
        wp::vec_t<3,wp::float32> adj_16 = {};
        wp::float32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::float32 adj_24 = {};
        wp::hash_grid_query_t adj_25 = {};
        wp::int32 adj_26 = {};
        wp::float32 adj_27 = {};
        wp::vec_t<3,wp::float32> adj_28 = {};
        bool adj_29 = {};
        wp::uint32 adj_30 = {};
        wp::uint32 adj_31 = {};
        wp::uint32 adj_32 = {};
        bool adj_33 = {};
        bool adj_34 = {};
        bool adj_35 = {};
        wp::vec_t<3,wp::float32> adj_36 = {};
        wp::vec_t<3,wp::float32> adj_37 = {};
        wp::vec_t<3,wp::float32> adj_38 = {};
        wp::float32 adj_39 = {};
        wp::float32 adj_40 = {};
        wp::float32 adj_41 = {};
        wp::float32 adj_42 = {};
        wp::float32 adj_43 = {};
        wp::float32 adj_44 = {};
        wp::float32 adj_45 = {};
        wp::float32 adj_46 = {};
        wp::float32 adj_47 = {};
        bool adj_48 = {};
        bool adj_49 = {};
        bool adj_50 = {};
        wp::vec_t<3,wp::float32> adj_51 = {};
        wp::vec_t<3,wp::float32> adj_52 = {};
        wp::vec_t<3,wp::float32> adj_53 = {};
        wp::vec_t<3,wp::float32> adj_54 = {};
        wp::float32 adj_55 = {};
        wp::vec_t<3,wp::float32> adj_56 = {};
        wp::float32 adj_57 = {};
        wp::float32 adj_58 = {};
        bool adj_59 = {};
        wp::float32 adj_60 = {};
        wp::float32 adj_61 = {};
        wp::vec_t<3,wp::float32> adj_62 = {};
        wp::vec_t<3,wp::float32> adj_63 = {};
        wp::float32 adj_64 = {};
        wp::float32 adj_65 = {};
        wp::float32 adj_66 = {};
        wp::float32 adj_67 = {};
        wp::float32 adj_68 = {};
        wp::vec_t<3,wp::float32> adj_69 = {};
        wp::vec_t<3,wp::float32> adj_70 = {};
        wp::vec_t<3,wp::float32> adj_71 = {};
        wp::vec_t<3,wp::float32> adj_72 = {};
        wp::vec_t<3,wp::float32> adj_73 = {};
        wp::vec_t<3,wp::float32> adj_74 = {};
        wp::float32 adj_75 = {};
        wp::vec_t<3,wp::float32> adj_76 = {};
        wp::vec_t<3,wp::float32> adj_77 = {};
        wp::float32 adj_78 = {};
        wp::vec_t<3,wp::float32> adj_79 = {};
        wp::vec_t<3,wp::float32> adj_80 = {};
        wp::vec_t<3,wp::float32> adj_81 = {};
        //---------
        // forward
        // def my_solve_particle_particle_contacts(                                               <L 207>
        // tid = wp.tid()                                                                         <L 223>
        var_0 = builtin_tid1d();
        // i = wp.hash_grid_point_id(grid, tid)                                                   <L 224>
        var_1 = wp::hash_grid_point_id(var_grid, var_0);
        // if i == -1:                                                                            <L 225>
        var_3 = wp::neg(var_2);
        var_4 = (var_1 == var_3);
        if (var_4) {
            // return                                                                             <L 226>
            goto label0;
        }
        // if (particle_flags[i] & PARTICLE_FLAG_ACTIVE) == 0:                                    <L 227>
        var_5 = wp::address(var_particle_flags, var_1);
        var_7 = wp::load(var_5);
        var_8 = wp::bit_and(var_7, var_6);
        var_10 = (var_8 == var_9);
        if (var_10) {
            // return                                                                             <L 228>
            goto label1;
        }
        // x = particle_x[i]                                                                      <L 229>
        var_11 = wp::address(var_particle_x, var_1);
        var_12 = wp::load(var_11);
        var_13 = wp::copy(var_12);
        // v = particle_v[i]                                                                      <L 230>
        var_14 = wp::address(var_particle_v, var_1);
        var_15 = wp::load(var_14);
        var_16 = wp::copy(var_15);
        // radius = particle_radius[i]                                                            <L 231>
        var_17 = wp::address(var_particle_radius, var_1);
        var_18 = wp::load(var_17);
        var_19 = wp::copy(var_18);
        // w1 = particle_invmass[i]                                                               <L 232>
        var_20 = wp::address(var_particle_invmass, var_1);
        var_21 = wp::load(var_20);
        var_22 = wp::copy(var_21);
        // query = wp.hash_grid_query(grid, x, radius + max_radius + k_cohesion)                  <L 233>
        var_23 = wp::add(var_19, var_max_radius);
        var_24 = wp::add(var_23, var_k_cohesion);
        var_25 = wp::hash_grid_query(var_grid, var_13, var_24);
        // index = int(0)                                                                         <L 234>
        var_26 = wp::int(var_9);
        // delta = wp.vec3(0.0)                                                                   <L 235>
        var_28 = wp::vec_t<3,wp::float32>(var_27);
        // while wp.hash_grid_query_next(query, index):                                           <L 236>
        // wp.atomic_add(deltas, i, delta * relaxation)                                           <L 255>
        var_80 = wp::mul(var_28, var_relaxation);
        // var_81 = wp::atomic_add(var_deltas, var_1, var_80);
        //---------
        // reverse
        wp::adj_atomic_add(var_deltas, var_1, var_80, adj_deltas, adj_1, adj_80, adj_81);
        wp::adj_mul(var_28, var_relaxation, adj_28, adj_relaxation, adj_80);
        // adj: wp.atomic_add(deltas, i, delta * relaxation)                                      <L 255>
        start_while_2:;
        var_29 = wp::hash_grid_query_next(var_25, var_26);
        if ((var_29) == false) goto end_while_2;
        adj_30 = {};
        adj_31 = {};
        adj_32 = {};
        adj_33 = {};
        adj_34 = {};
        adj_35 = {};
        adj_36 = {};
        adj_37 = {};
        adj_38 = {};
        adj_39 = {};
        adj_40 = {};
        adj_41 = {};
        adj_42 = {};
        adj_43 = {};
        adj_44 = {};
        adj_45 = {};
        adj_46 = {};
        adj_47 = {};
        adj_48 = {};
        adj_49 = {};
        adj_50 = {};
        adj_51 = {};
        adj_52 = {};
        adj_53 = {};
        adj_54 = {};
        adj_55 = {};
        adj_56 = {};
        adj_57 = {};
        adj_58 = {};
        adj_59 = {};
        adj_60 = {};
        adj_61 = {};
        adj_62 = {};
        adj_63 = {};
        adj_64 = {};
        adj_65 = {};
        adj_66 = {};
        adj_67 = {};
        adj_68 = {};
        adj_69 = {};
        adj_70 = {};
        adj_71 = {};
        adj_72 = {};
        adj_73 = {};
        adj_74 = {};
        adj_75 = {};
        adj_76 = {};
        adj_77 = {};
        adj_78 = {};
        adj_79 = {};
            // if (particle_flags[index] & PARTICLE_FLAG_ACTIVE) != 0 and index != i:             <L 237>
            var_30 = wp::address(var_particle_flags, var_26);
            var_31 = wp::load(var_30);
            var_32 = wp::bit_and(var_31, var_6);
            var_33 = (var_32 != var_9);
            var_34 = (var_26 != var_1);
            var_35 = var_33 && var_34;
            if (var_35) {
                // n = x - particle_x[index]                                                      <L 238>
                var_36 = wp::address(var_particle_x, var_26);
                var_37 = wp::load(var_36);
                var_38 = wp::sub(var_13, var_37);
                // d = wp.length(n)                                                               <L 239>
                var_39 = wp::length(var_38);
                // err = d - radius - particle_radius[index]                                      <L 240>
                var_40 = wp::sub(var_39, var_19);
                var_41 = wp::address(var_particle_radius, var_26);
                var_42 = wp::load(var_41);
                var_43 = wp::sub(var_40, var_42);
                // w2 = particle_invmass[index]                                                   <L 241>
                var_44 = wp::address(var_particle_invmass, var_26);
                var_45 = wp::load(var_44);
                var_46 = wp::copy(var_45);
                // denom = w1 + w2                                                                <L 242>
                var_47 = wp::add(var_22, var_46);
                // if err <= k_cohesion and denom > 0.0:                                          <L 243>
                var_48 = (var_43 <= var_k_cohesion);
                var_49 = (var_47 > var_27);
                var_50 = var_48 && var_49;
                if (var_50) {
                    // n = n / d                                                                  <L 244>
                    var_51 = wp::div(var_38, var_39);
                    // vrel = v - particle_v[index]                                               <L 245>
                    var_52 = wp::address(var_particle_v, var_26);
                    var_53 = wp::load(var_52);
                    var_54 = wp::sub(var_16, var_53);
                    // lambda_n = err                                                             <L 246>
                    var_55 = wp::copy(var_43);
                    // delta_n = n * lambda_n                                                     <L 247>
                    var_56 = wp::mul(var_51, var_55);
                    // vn = wp.dot(n, vrel)                                                       <L 248>
                    var_57 = wp::dot(var_51, var_54);
                    // if wp.abs(vn)<staticParticleVelocityThresholdRatio:                        <L 249>
                    var_58 = wp::abs(var_57);
                    var_59 = (var_58 < var_staticParticleVelocityThresholdRatio);
                    if (var_59) {
                        // k_mu=staticParticleFriction                                            <L 250>
                        var_60 = wp::copy(var_staticParticleFriction);
                    }
                    var_61 = wp::select(var_59, var_k_mu, var_60);
                    // vt = v - n * vn                                                            <L 251>
                    var_62 = wp::mul(var_51, var_57);
                    var_63 = wp::sub(var_16, var_62);
                    // lambda_f = wp.max(k_mu * lambda_n, -wp.length(vt) * dt)                    <L 252>
                    var_64 = wp::mul(var_61, var_55);
                    var_65 = wp::length(var_63);
                    var_66 = wp::neg(var_65);
                    var_67 = wp::mul(var_66, var_dt);
                    var_68 = wp::max(var_64, var_67);
                    // delta_f = wp.normalize(vt) * lambda_f                                      <L 253>
                    var_69 = wp::normalize(var_63);
                    var_70 = wp::mul(var_69, var_68);
                    // delta += (delta_f - delta_n) / denom * w1                                  <L 254>
                    var_71 = wp::sub(var_70, var_56);
                    var_72 = wp::div(var_71, var_47);
                    var_73 = wp::mul(var_72, var_22);
                    var_74 = wp::add(var_28, var_73);
                }
                var_75 = wp::select(var_50, var_k_mu, var_61);
                var_76 = wp::select(var_50, var_28, var_74);
                var_77 = wp::select(var_50, var_38, var_51);
            }
            var_78 = wp::select(var_35, var_k_mu, var_75);
            var_79 = wp::select(var_35, var_28, var_76);
            wp::assign(var_k_mu, var_78);
            wp::assign(var_28, var_79);
            wp::adj_assign(var_28, var_79, adj_28, adj_79);
            wp::adj_assign(var_k_mu, var_78, adj_k_mu, adj_78);
            wp::adj_select(var_35, var_28, var_76, adj_35, adj_28, adj_76, adj_79);
            wp::adj_select(var_35, var_k_mu, var_75, adj_35, adj_k_mu, adj_75, adj_78);
            if (var_35) {
                wp::adj_select(var_50, var_38, var_51, adj_50, adj_38, adj_51, adj_77);
                wp::adj_select(var_50, var_28, var_74, adj_50, adj_28, adj_74, adj_76);
                wp::adj_select(var_50, var_k_mu, var_61, adj_50, adj_k_mu, adj_61, adj_75);
                if (var_50) {
                    wp::adj_add(var_28, var_73, adj_28, adj_73, adj_74);
                    wp::adj_mul(var_72, var_22, adj_72, adj_22, adj_73);
                    wp::adj_div(var_71, var_47, adj_71, adj_47, adj_72);
                    wp::adj_sub(var_70, var_56, adj_70, adj_56, adj_71);
                    // adj: delta += (delta_f - delta_n) / denom * w1                             <L 254>
                    wp::adj_mul(var_69, var_68, adj_69, adj_68, adj_70);
                    wp::adj_normalize(var_63, var_69, adj_63, adj_69);
                    // adj: delta_f = wp.normalize(vt) * lambda_f                                 <L 253>
                    wp::adj_max(var_64, var_67, adj_64, adj_67, adj_68);
                    wp::adj_mul(var_66, var_dt, adj_66, adj_dt, adj_67);
                    wp::adj_neg(var_65, adj_65, adj_66);
                    wp::adj_length(var_63, var_65, adj_63, adj_65);
                    wp::adj_mul(var_61, var_55, adj_61, adj_55, adj_64);
                    // adj: lambda_f = wp.max(k_mu * lambda_n, -wp.length(vt) * dt)               <L 252>
                    wp::adj_sub(var_16, var_62, adj_16, adj_62, adj_63);
                    wp::adj_mul(var_51, var_57, adj_51, adj_57, adj_62);
                    // adj: vt = v - n * vn                                                       <L 251>
                    wp::adj_select(var_59, var_k_mu, var_60, adj_59, adj_k_mu, adj_60, adj_61);
                    if (var_59) {
                        wp::adj_copy(var_staticParticleFriction, adj_staticParticleFriction, adj_60);
                        // adj: k_mu=staticParticleFriction                                       <L 250>
                    }
                    wp::adj_abs(var_57, adj_57, adj_58);
                    // adj: if wp.abs(vn)<staticParticleVelocityThresholdRatio:                   <L 249>
                    wp::adj_dot(var_51, var_54, adj_51, adj_54, adj_57);
                    // adj: vn = wp.dot(n, vrel)                                                  <L 248>
                    wp::adj_mul(var_51, var_55, adj_51, adj_55, adj_56);
                    // adj: delta_n = n * lambda_n                                                <L 247>
                    wp::adj_copy(var_43, adj_43, adj_55);
                    // adj: lambda_n = err                                                        <L 246>
                    wp::adj_sub(var_16, var_53, adj_16, adj_52, adj_54);
                    wp::adj_load(var_52, adj_52, adj_53);
                    wp::adj_address(var_particle_v, var_26, adj_particle_v, adj_26, adj_52);
                    // adj: vrel = v - particle_v[index]                                          <L 245>
                    wp::adj_div(var_38, var_39, adj_38, adj_39, adj_51);
                    // adj: n = n / d                                                             <L 244>
                }
                // adj: if err <= k_cohesion and denom > 0.0:                                     <L 243>
                wp::adj_add(var_22, var_46, adj_22, adj_46, adj_47);
                // adj: denom = w1 + w2                                                           <L 242>
                wp::adj_copy(var_45, adj_44, adj_46);
                wp::adj_load(var_44, adj_44, adj_45);
                wp::adj_address(var_particle_invmass, var_26, adj_particle_invmass, adj_26, adj_44);
                // adj: w2 = particle_invmass[index]                                              <L 241>
                wp::adj_sub(var_40, var_42, adj_40, adj_41, adj_43);
                wp::adj_load(var_41, adj_41, adj_42);
                wp::adj_address(var_particle_radius, var_26, adj_particle_radius, adj_26, adj_41);
                wp::adj_sub(var_39, var_19, adj_39, adj_19, adj_40);
                // adj: err = d - radius - particle_radius[index]                                 <L 240>
                wp::adj_length(var_38, var_39, adj_38, adj_39);
                // adj: d = wp.length(n)                                                          <L 239>
                wp::adj_sub(var_13, var_37, adj_13, adj_36, adj_38);
                wp::adj_load(var_36, adj_36, adj_37);
                wp::adj_address(var_particle_x, var_26, adj_particle_x, adj_26, adj_36);
                // adj: n = x - particle_x[index]                                                 <L 238>
            }
            wp::adj_bit_and(var_31, var_6, adj_30, adj_6, adj_32);
            wp::adj_load(var_30, adj_30, adj_31);
            wp::adj_address(var_particle_flags, var_26, adj_particle_flags, adj_26, adj_30);
            // adj: if (particle_flags[index] & PARTICLE_FLAG_ACTIVE) != 0 and index != i:        <L 237>
        goto start_while_2;
        end_while_2:;
        // adj: while wp.hash_grid_query_next(query, index):                                      <L 236>
        wp::adj_vec_t(var_27, adj_27, adj_28);
        // adj: delta = wp.vec3(0.0)                                                              <L 235>
        wp::adj_int(var_9, adj_9, adj_26);
        // adj: index = int(0)                                                                    <L 234>
        wp::adj_hash_grid_query(var_grid, var_13, var_24, adj_grid, adj_13, adj_24, adj_25);
        wp::adj_add(var_23, var_k_cohesion, adj_23, adj_k_cohesion, adj_24);
        wp::adj_add(var_19, var_max_radius, adj_19, adj_max_radius, adj_23);
        // adj: query = wp.hash_grid_query(grid, x, radius + max_radius + k_cohesion)             <L 233>
        wp::adj_copy(var_21, adj_20, adj_22);
        wp::adj_load(var_20, adj_20, adj_21);
        wp::adj_address(var_particle_invmass, var_1, adj_particle_invmass, adj_1, adj_20);
        // adj: w1 = particle_invmass[i]                                                          <L 232>
        wp::adj_copy(var_18, adj_17, adj_19);
        wp::adj_load(var_17, adj_17, adj_18);
        wp::adj_address(var_particle_radius, var_1, adj_particle_radius, adj_1, adj_17);
        // adj: radius = particle_radius[i]                                                       <L 231>
        wp::adj_copy(var_15, adj_14, adj_16);
        wp::adj_load(var_14, adj_14, adj_15);
        wp::adj_address(var_particle_v, var_1, adj_particle_v, adj_1, adj_14);
        // adj: v = particle_v[i]                                                                 <L 230>
        wp::adj_copy(var_12, adj_11, adj_13);
        wp::adj_load(var_11, adj_11, adj_12);
        wp::adj_address(var_particle_x, var_1, adj_particle_x, adj_1, adj_11);
        // adj: x = particle_x[i]                                                                 <L 229>
        if (var_10) {
            label1:;
            // adj: return                                                                        <L 228>
        }
        wp::adj_bit_and(var_7, var_6, adj_5, adj_6, adj_8);
        wp::adj_load(var_5, adj_5, adj_7);
        wp::adj_address(var_particle_flags, var_1, adj_particle_flags, adj_1, adj_5);
        // adj: if (particle_flags[i] & PARTICLE_FLAG_ACTIVE) == 0:                               <L 227>
        if (var_4) {
            label0:;
            // adj: return                                                                        <L 226>
        }
        wp::adj_neg(var_2, adj_2, adj_3);
        // adj: if i == -1:                                                                       <L 225>
        wp::adj_hash_grid_point_id(var_grid, var_0, adj_grid, adj_0, adj_1);
        // adj: i = wp.hash_grid_point_id(grid, tid)                                              <L 224>
        // adj: tid = wp.tid()                                                                    <L 223>
        // adj: def my_solve_particle_particle_contacts(                                          <L 207>
        continue;
    }
}



extern "C" __global__ void my_solve_particle_shape_contacts_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_x,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_v,
    wp::array_t<wp::float32> var_particle_invmass,
    wp::array_t<wp::float32> var_particle_radius,
    wp::array_t<wp::uint32> var_particle_flags,
    wp::array_t<wp::transform_t<wp::float32>> var_body_q,
    wp::array_t<wp::vec_t<6,wp::float32>> var_body_qd,
    wp::array_t<wp::vec_t<3,wp::float32>> var_body_com,
    wp::array_t<wp::float32> var_body_m_inv,
    wp::array_t<wp::mat_t<3,3,wp::float32>> var_body_I_inv,
    wp::array_t<wp::int32> var_shape_body,
    ModelShapeMaterials var_shape_materials,
    wp::float32 var_particle_mu,
    wp::float32 var_particle_ka,
    wp::float32 var_staticGroundVelocityThresholdRatio,
    wp::float32 var_staticGroundFriction,
    wp::array_t<wp::int32> var_contact_count,
    wp::array_t<wp::int32> var_contact_particle,
    wp::array_t<wp::int32> var_contact_shape,
    wp::array_t<wp::vec_t<3,wp::float32>> var_contact_body_pos,
    wp::array_t<wp::vec_t<3,wp::float32>> var_contact_body_vel,
    wp::array_t<wp::vec_t<3,wp::float32>> var_contact_normal,
    wp::int32 var_contact_max,
    wp::float32 var_dt,
    wp::float32 var_relaxation,
    wp::array_t<wp::vec_t<3,wp::float32>> var_delta,
    wp::array_t<wp::vec_t<6,wp::float32>> var_body_delta)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        const wp::int32 var_1 = 0;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        bool var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::uint32* var_15;
        const wp::uint32 var_16 = 1;
        wp::uint32 var_17;
        wp::uint32 var_18;
        bool var_19;
        wp::vec_t<3,wp::float32>* var_20;
        wp::vec_t<3,wp::float32> var_21;
        wp::vec_t<3,wp::float32> var_22;
        wp::vec_t<3,wp::float32>* var_23;
        wp::vec_t<3,wp::float32> var_24;
        wp::vec_t<3,wp::float32> var_25;
        wp::transform_t<wp::float32> var_26;
        wp::vec_t<3,wp::float32> var_27;
        bool var_28;
        wp::transform_t<wp::float32>* var_29;
        wp::transform_t<wp::float32> var_30;
        wp::transform_t<wp::float32> var_31;
        wp::vec_t<3,wp::float32>* var_32;
        wp::vec_t<3,wp::float32> var_33;
        wp::vec_t<3,wp::float32> var_34;
        wp::transform_t<wp::float32> var_35;
        wp::vec_t<3,wp::float32> var_36;
        wp::vec_t<3,wp::float32>* var_37;
        wp::vec_t<3,wp::float32> var_38;
        wp::vec_t<3,wp::float32> var_39;
        wp::vec_t<3,wp::float32> var_40;
        wp::vec_t<3,wp::float32> var_41;
        wp::vec_t<3,wp::float32>* var_42;
        wp::vec_t<3,wp::float32> var_43;
        wp::vec_t<3,wp::float32> var_44;
        wp::vec_t<3,wp::float32> var_45;
        wp::float32 var_46;
        wp::float32* var_47;
        wp::float32 var_48;
        wp::float32 var_49;
        bool var_50;
        const wp::float32 var_51 = 0.5;
        wp::array_t<wp::float32>* var_52;
        wp::array_t<wp::float32> var_53;
        wp::float32* var_54;
        wp::float32 var_55;
        wp::float32 var_56;
        wp::float32 var_57;
        wp::vec_t<6,wp::float32> var_58;
        bool var_59;
        wp::vec_t<6,wp::float32>* var_60;
        wp::vec_t<6,wp::float32> var_61;
        wp::vec_t<6,wp::float32> var_62;
        wp::vec_t<6,wp::float32> var_63;
        wp::vec_t<3,wp::float32> var_64;
        wp::vec_t<3,wp::float32> var_65;
        wp::vec_t<3,wp::float32> var_66;
        wp::vec_t<3,wp::float32> var_67;
        wp::vec_t<3,wp::float32>* var_68;
        wp::vec_t<3,wp::float32> var_69;
        wp::vec_t<3,wp::float32> var_70;
        wp::vec_t<3,wp::float32> var_71;
        wp::vec_t<3,wp::float32> var_72;
        wp::float32 var_73;
        wp::vec_t<3,wp::float32> var_74;
        wp::float32 var_75;
        wp::vec_t<3,wp::float32> var_76;
        wp::vec_t<3,wp::float32> var_77;
        wp::float32 var_78;
        bool var_79;
        wp::float32 var_80;
        wp::float32 var_81;
        wp::float32* var_82;
        wp::float32 var_83;
        wp::float32 var_84;
        wp::float32* var_85;
        const wp::float32 var_86 = 0.0;
        wp::float32 var_87;
        wp::float32 var_88;
        wp::float32 var_89;
        bool var_90;
        wp::vec_t<3,wp::float32> var_91;
        wp::quat_t<wp::float32> var_92;
        wp::vec_t<3,wp::float32> var_93;
        wp::mat_t<3,3,wp::float32>* var_94;
        wp::mat_t<3,3,wp::float32> var_95;
        wp::mat_t<3,3,wp::float32> var_96;
        wp::float32* var_97;
        wp::vec_t<3,wp::float32> var_98;
        wp::float32 var_99;
        wp::float32 var_100;
        wp::float32 var_101;
        wp::float32 var_102;
        wp::float32 var_103;
        bool var_104;
        wp::float32 var_105;
        wp::float32 var_106;
        wp::float32 var_107;
        wp::float32 var_108;
        wp::float32 var_109;
        wp::vec_t<3,wp::float32> var_110;
        wp::vec_t<3,wp::float32> var_111;
        wp::vec_t<3,wp::float32> var_112;
        wp::vec_t<3,wp::float32> var_113;
        wp::vec_t<3,wp::float32> var_114;
        wp::vec_t<3,wp::float32> var_115;
        wp::vec_t<3,wp::float32> var_116;
        bool var_117;
        wp::vec_t<3,wp::float32> var_118;
        wp::vec_t<6,wp::float32> var_119;
        wp::vec_t<6,wp::float32> var_120;
        //---------
        // forward
        // def my_solve_particle_shape_contacts(                                                  <L 258>
        // tid = wp.tid()                                                                         <L 287>
        var_0 = builtin_tid1d();
        // count = min(contact_max, contact_count[0])                                             <L 288>
        var_2 = wp::address(var_contact_count, var_1);
        var_3 = wp::load(var_2);
        var_4 = wp::min(var_contact_max, var_3);
        // if tid >= count:                                                                       <L 289>
        var_5 = (var_0 >= var_4);
        if (var_5) {
            // return                                                                             <L 290>
            return;
        }
        // shape_index = contact_shape[tid]                                                       <L 291>
        var_6 = wp::address(var_contact_shape, var_0);
        var_7 = wp::load(var_6);
        var_8 = wp::copy(var_7);
        // body_index = shape_body[shape_index]                                                   <L 292>
        var_9 = wp::address(var_shape_body, var_8);
        var_10 = wp::load(var_9);
        var_11 = wp::copy(var_10);
        // particle_index = contact_particle[tid]                                                 <L 293>
        var_12 = wp::address(var_contact_particle, var_0);
        var_13 = wp::load(var_12);
        var_14 = wp::copy(var_13);
        // if (particle_flags[particle_index] & PARTICLE_FLAG_ACTIVE) == 0:                       <L 294>
        var_15 = wp::address(var_particle_flags, var_14);
        var_17 = wp::load(var_15);
        var_18 = wp::bit_and(var_17, var_16);
        var_19 = (var_18 == var_1);
        if (var_19) {
            // return                                                                             <L 295>
            return;
        }
        // px = particle_x[particle_index]                                                        <L 296>
        var_20 = wp::address(var_particle_x, var_14);
        var_21 = wp::load(var_20);
        var_22 = wp::copy(var_21);
        // pv = particle_v[particle_index]                                                        <L 297>
        var_23 = wp::address(var_particle_v, var_14);
        var_24 = wp::load(var_23);
        var_25 = wp::copy(var_24);
        // X_wb = wp.transform_identity()                                                         <L 298>
        var_26 = wp::transform_identity<wp::float32>();
        // X_com = wp.vec3()                                                                      <L 299>
        var_27 = wp::vec_t<3,wp::float32>();
        // if body_index >= 0:                                                                    <L 300>
        var_28 = (var_11 >= var_1);
        if (var_28) {
            // X_wb = body_q[body_index]                                                          <L 301>
            var_29 = wp::address(var_body_q, var_11);
            var_30 = wp::load(var_29);
            var_31 = wp::copy(var_30);
            // X_com = body_com[body_index]                                                       <L 302>
            var_32 = wp::address(var_body_com, var_11);
            var_33 = wp::load(var_32);
            var_34 = wp::copy(var_33);
        }
        var_35 = wp::select(var_28, var_26, var_31);
        var_36 = wp::select(var_28, var_27, var_34);
        // bx = wp.transform_point(X_wb, contact_body_pos[tid])                                   <L 303>
        var_37 = wp::address(var_contact_body_pos, var_0);
        var_38 = wp::load(var_37);
        var_39 = wp::transform_point(var_35, var_38);
        // r = bx - wp.transform_point(X_wb, X_com)                                               <L 304>
        var_40 = wp::transform_point(var_35, var_36);
        var_41 = wp::sub(var_39, var_40);
        // n = contact_normal[tid]                                                                <L 305>
        var_42 = wp::address(var_contact_normal, var_0);
        var_43 = wp::load(var_42);
        var_44 = wp::copy(var_43);
        // c = wp.dot(n, px - bx) - particle_radius[particle_index]                               <L 306>
        var_45 = wp::sub(var_22, var_39);
        var_46 = wp::dot(var_44, var_45);
        var_47 = wp::address(var_particle_radius, var_14);
        var_48 = wp::load(var_47);
        var_49 = wp::sub(var_46, var_48);
        // if c > particle_ka:                                                                    <L 307>
        var_50 = (var_49 > var_particle_ka);
        if (var_50) {
            // return                                                                             <L 308>
            return;
        }
        // mu = 0.5 * (particle_mu + shape_materials.mu[shape_index])                             <L 309>
        var_52 = &(var_shape_materials.mu);
        var_53 = wp::load(var_52);
        var_54 = wp::address(var_53, var_8);
        var_55 = wp::load(var_54);
        var_56 = wp::add(var_particle_mu, var_55);
        var_57 = wp::mul(var_51, var_56);
        // body_v_s = wp.spatial_vector()                                                         <L 310>
        var_58 = wp::vec_t<6,wp::float32>();
        // if body_index >= 0:                                                                    <L 311>
        var_59 = (var_11 >= var_1);
        if (var_59) {
            // body_v_s = body_qd[body_index]                                                     <L 312>
            var_60 = wp::address(var_body_qd, var_11);
            var_61 = wp::load(var_60);
            var_62 = wp::copy(var_61);
        }
        var_63 = wp::select(var_59, var_58, var_62);
        // body_w = wp.spatial_top(body_v_s)                                                      <L 313>
        var_64 = wp::spatial_top(var_63);
        // body_v = wp.spatial_bottom(body_v_s)                                                   <L 314>
        var_65 = wp::spatial_bottom(var_63);
        // bv = body_v + wp.cross(body_w, r) + wp.transform_vector(X_wb, contact_body_vel[tid])       <L 315>
        var_66 = wp::cross(var_64, var_41);
        var_67 = wp::add(var_65, var_66);
        var_68 = wp::address(var_contact_body_vel, var_0);
        var_69 = wp::load(var_68);
        var_70 = wp::transform_vector(var_35, var_69);
        var_71 = wp::add(var_67, var_70);
        // v = pv - bv                                                                            <L 316>
        var_72 = wp::sub(var_25, var_71);
        // lambda_n = c                                                                           <L 317>
        var_73 = wp::copy(var_49);
        // delta_n = n * lambda_n                                                                 <L 318>
        var_74 = wp::mul(var_44, var_73);
        // vn = wp.dot(n, v)                                                                      <L 319>
        var_75 = wp::dot(var_44, var_72);
        // vt = v - n * vn                                                                        <L 320>
        var_76 = wp::mul(var_44, var_75);
        var_77 = wp::sub(var_72, var_76);
        // if wp.abs(vn)<staticGroundVelocityThresholdRatio:                                      <L 321>
        var_78 = wp::abs(var_75);
        var_79 = (var_78 < var_staticGroundVelocityThresholdRatio);
        if (var_79) {
            // mu=staticGroundFriction                                                            <L 322>
            var_80 = wp::copy(var_staticGroundFriction);
        }
        var_81 = wp::select(var_79, var_57, var_80);
        // w1 = particle_invmass[particle_index]                                                  <L 323>
        var_82 = wp::address(var_particle_invmass, var_14);
        var_83 = wp::load(var_82);
        var_84 = wp::copy(var_83);
        // w2 = particle_invmass[particle_index]*wp.float(0.0)                                    <L 324>
        var_85 = wp::address(var_particle_invmass, var_14);
        var_87 = wp::float(var_86);
        var_88 = wp::load(var_85);
        var_89 = wp::mul(var_88, var_87);
        // if body_index >= 0:                                                                    <L 325>
        var_90 = (var_11 >= var_1);
        if (var_90) {
            // angular = wp.cross(r, n)                                                           <L 326>
            var_91 = wp::cross(var_41, var_44);
            // q = wp.transform_get_rotation(X_wb)                                                <L 327>
            var_92 = wp::transform_get_rotation(var_35);
            // rot_angular = wp.quat_rotate_inv(q, angular)                                       <L 328>
            var_93 = wp::quat_rotate_inv(var_92, var_91);
            // I_inv = body_I_inv[body_index]                                                     <L 329>
            var_94 = wp::address(var_body_I_inv, var_11);
            var_95 = wp::load(var_94);
            var_96 = wp::copy(var_95);
            // w2 = body_m_inv[body_index] + wp.dot(rot_angular, I_inv * rot_angular)             <L 330>
            var_97 = wp::address(var_body_m_inv, var_11);
            var_98 = wp::mul(var_96, var_93);
            var_99 = wp::dot(var_93, var_98);
            var_100 = wp::load(var_97);
            var_101 = wp::add(var_100, var_99);
        }
        var_102 = wp::select(var_90, var_89, var_101);
        // denom = w1 + w2                                                                        <L 331>
        var_103 = wp::add(var_84, var_102);
        // if denom == 0.0:                                                                       <L 332>
        var_104 = (var_103 == var_86);
        if (var_104) {
            // return                                                                             <L 333>
            return;
        }
        // lambda_f = wp.max(mu * lambda_n, -wp.length(vt) * dt)                                  <L 334>
        var_105 = wp::mul(var_81, var_73);
        var_106 = wp::length(var_77);
        var_107 = wp::neg(var_106);
        var_108 = wp::mul(var_107, var_dt);
        var_109 = wp::max(var_105, var_108);
        // delta_f = wp.normalize(vt) * lambda_f                                                  <L 335>
        var_110 = wp::normalize(var_77);
        var_111 = wp::mul(var_110, var_109);
        // delta_total = (delta_f - delta_n) / denom * relaxation * w1                            <L 336>
        var_112 = wp::sub(var_111, var_74);
        var_113 = wp::div(var_112, var_103);
        var_114 = wp::mul(var_113, var_relaxation);
        var_115 = wp::mul(var_114, var_84);
        // wp.atomic_add(delta, particle_index, delta_total)                                      <L 337>
        var_116 = wp::atomic_add(var_delta, var_14, var_115);
        // if body_index >= 0:                                                                    <L 338>
        var_117 = (var_11 >= var_1);
        if (var_117) {
            // delta_t = wp.cross(r, delta_total)                                                 <L 339>
            var_118 = wp::cross(var_41, var_115);
            // wp.atomic_sub(body_delta, body_index, wp.spatial_vector(delta_t, delta_total))       <L 340>
            var_119 = wp::vec_t<6,wp::float32>(var_118, var_115);
            var_120 = wp::atomic_sub(var_body_delta, var_11, var_119);
        }
    }
}

extern "C" __global__ void my_solve_particle_shape_contacts_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_x,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_v,
    wp::array_t<wp::float32> var_particle_invmass,
    wp::array_t<wp::float32> var_particle_radius,
    wp::array_t<wp::uint32> var_particle_flags,
    wp::array_t<wp::transform_t<wp::float32>> var_body_q,
    wp::array_t<wp::vec_t<6,wp::float32>> var_body_qd,
    wp::array_t<wp::vec_t<3,wp::float32>> var_body_com,
    wp::array_t<wp::float32> var_body_m_inv,
    wp::array_t<wp::mat_t<3,3,wp::float32>> var_body_I_inv,
    wp::array_t<wp::int32> var_shape_body,
    ModelShapeMaterials var_shape_materials,
    wp::float32 var_particle_mu,
    wp::float32 var_particle_ka,
    wp::float32 var_staticGroundVelocityThresholdRatio,
    wp::float32 var_staticGroundFriction,
    wp::array_t<wp::int32> var_contact_count,
    wp::array_t<wp::int32> var_contact_particle,
    wp::array_t<wp::int32> var_contact_shape,
    wp::array_t<wp::vec_t<3,wp::float32>> var_contact_body_pos,
    wp::array_t<wp::vec_t<3,wp::float32>> var_contact_body_vel,
    wp::array_t<wp::vec_t<3,wp::float32>> var_contact_normal,
    wp::int32 var_contact_max,
    wp::float32 var_dt,
    wp::float32 var_relaxation,
    wp::array_t<wp::vec_t<3,wp::float32>> var_delta,
    wp::array_t<wp::vec_t<6,wp::float32>> var_body_delta,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_particle_x,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_particle_v,
    wp::array_t<wp::float32> adj_particle_invmass,
    wp::array_t<wp::float32> adj_particle_radius,
    wp::array_t<wp::uint32> adj_particle_flags,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_q,
    wp::array_t<wp::vec_t<6,wp::float32>> adj_body_qd,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_body_com,
    wp::array_t<wp::float32> adj_body_m_inv,
    wp::array_t<wp::mat_t<3,3,wp::float32>> adj_body_I_inv,
    wp::array_t<wp::int32> adj_shape_body,
    ModelShapeMaterials adj_shape_materials,
    wp::float32 adj_particle_mu,
    wp::float32 adj_particle_ka,
    wp::float32 adj_staticGroundVelocityThresholdRatio,
    wp::float32 adj_staticGroundFriction,
    wp::array_t<wp::int32> adj_contact_count,
    wp::array_t<wp::int32> adj_contact_particle,
    wp::array_t<wp::int32> adj_contact_shape,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_contact_body_pos,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_contact_body_vel,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_contact_normal,
    wp::int32 adj_contact_max,
    wp::float32 adj_dt,
    wp::float32 adj_relaxation,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_delta,
    wp::array_t<wp::vec_t<6,wp::float32>> adj_body_delta)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        const wp::int32 var_1 = 0;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        bool var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::uint32* var_15;
        const wp::uint32 var_16 = 1;
        wp::uint32 var_17;
        wp::uint32 var_18;
        bool var_19;
        wp::vec_t<3,wp::float32>* var_20;
        wp::vec_t<3,wp::float32> var_21;
        wp::vec_t<3,wp::float32> var_22;
        wp::vec_t<3,wp::float32>* var_23;
        wp::vec_t<3,wp::float32> var_24;
        wp::vec_t<3,wp::float32> var_25;
        wp::transform_t<wp::float32> var_26;
        wp::vec_t<3,wp::float32> var_27;
        bool var_28;
        wp::transform_t<wp::float32>* var_29;
        wp::transform_t<wp::float32> var_30;
        wp::transform_t<wp::float32> var_31;
        wp::vec_t<3,wp::float32>* var_32;
        wp::vec_t<3,wp::float32> var_33;
        wp::vec_t<3,wp::float32> var_34;
        wp::transform_t<wp::float32> var_35;
        wp::vec_t<3,wp::float32> var_36;
        wp::vec_t<3,wp::float32>* var_37;
        wp::vec_t<3,wp::float32> var_38;
        wp::vec_t<3,wp::float32> var_39;
        wp::vec_t<3,wp::float32> var_40;
        wp::vec_t<3,wp::float32> var_41;
        wp::vec_t<3,wp::float32>* var_42;
        wp::vec_t<3,wp::float32> var_43;
        wp::vec_t<3,wp::float32> var_44;
        wp::vec_t<3,wp::float32> var_45;
        wp::float32 var_46;
        wp::float32* var_47;
        wp::float32 var_48;
        wp::float32 var_49;
        bool var_50;
        const wp::float32 var_51 = 0.5;
        wp::array_t<wp::float32>* var_52;
        wp::array_t<wp::float32> var_53;
        wp::float32* var_54;
        wp::float32 var_55;
        wp::float32 var_56;
        wp::float32 var_57;
        wp::vec_t<6,wp::float32> var_58;
        bool var_59;
        wp::vec_t<6,wp::float32>* var_60;
        wp::vec_t<6,wp::float32> var_61;
        wp::vec_t<6,wp::float32> var_62;
        wp::vec_t<6,wp::float32> var_63;
        wp::vec_t<3,wp::float32> var_64;
        wp::vec_t<3,wp::float32> var_65;
        wp::vec_t<3,wp::float32> var_66;
        wp::vec_t<3,wp::float32> var_67;
        wp::vec_t<3,wp::float32>* var_68;
        wp::vec_t<3,wp::float32> var_69;
        wp::vec_t<3,wp::float32> var_70;
        wp::vec_t<3,wp::float32> var_71;
        wp::vec_t<3,wp::float32> var_72;
        wp::float32 var_73;
        wp::vec_t<3,wp::float32> var_74;
        wp::float32 var_75;
        wp::vec_t<3,wp::float32> var_76;
        wp::vec_t<3,wp::float32> var_77;
        wp::float32 var_78;
        bool var_79;
        wp::float32 var_80;
        wp::float32 var_81;
        wp::float32* var_82;
        wp::float32 var_83;
        wp::float32 var_84;
        wp::float32* var_85;
        const wp::float32 var_86 = 0.0;
        wp::float32 var_87;
        wp::float32 var_88;
        wp::float32 var_89;
        bool var_90;
        wp::vec_t<3,wp::float32> var_91;
        wp::quat_t<wp::float32> var_92;
        wp::vec_t<3,wp::float32> var_93;
        wp::mat_t<3,3,wp::float32>* var_94;
        wp::mat_t<3,3,wp::float32> var_95;
        wp::mat_t<3,3,wp::float32> var_96;
        wp::float32* var_97;
        wp::vec_t<3,wp::float32> var_98;
        wp::float32 var_99;
        wp::float32 var_100;
        wp::float32 var_101;
        wp::float32 var_102;
        wp::float32 var_103;
        bool var_104;
        wp::float32 var_105;
        wp::float32 var_106;
        wp::float32 var_107;
        wp::float32 var_108;
        wp::float32 var_109;
        wp::vec_t<3,wp::float32> var_110;
        wp::vec_t<3,wp::float32> var_111;
        wp::vec_t<3,wp::float32> var_112;
        wp::vec_t<3,wp::float32> var_113;
        wp::vec_t<3,wp::float32> var_114;
        wp::vec_t<3,wp::float32> var_115;
        wp::vec_t<3,wp::float32> var_116;
        bool var_117;
        wp::vec_t<3,wp::float32> var_118;
        wp::vec_t<6,wp::float32> var_119;
        wp::vec_t<6,wp::float32> var_120;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        bool adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::uint32 adj_15 = {};
        wp::uint32 adj_16 = {};
        wp::uint32 adj_17 = {};
        wp::uint32 adj_18 = {};
        bool adj_19 = {};
        wp::vec_t<3,wp::float32> adj_20 = {};
        wp::vec_t<3,wp::float32> adj_21 = {};
        wp::vec_t<3,wp::float32> adj_22 = {};
        wp::vec_t<3,wp::float32> adj_23 = {};
        wp::vec_t<3,wp::float32> adj_24 = {};
        wp::vec_t<3,wp::float32> adj_25 = {};
        wp::transform_t<wp::float32> adj_26 = {};
        wp::vec_t<3,wp::float32> adj_27 = {};
        bool adj_28 = {};
        wp::transform_t<wp::float32> adj_29 = {};
        wp::transform_t<wp::float32> adj_30 = {};
        wp::transform_t<wp::float32> adj_31 = {};
        wp::vec_t<3,wp::float32> adj_32 = {};
        wp::vec_t<3,wp::float32> adj_33 = {};
        wp::vec_t<3,wp::float32> adj_34 = {};
        wp::transform_t<wp::float32> adj_35 = {};
        wp::vec_t<3,wp::float32> adj_36 = {};
        wp::vec_t<3,wp::float32> adj_37 = {};
        wp::vec_t<3,wp::float32> adj_38 = {};
        wp::vec_t<3,wp::float32> adj_39 = {};
        wp::vec_t<3,wp::float32> adj_40 = {};
        wp::vec_t<3,wp::float32> adj_41 = {};
        wp::vec_t<3,wp::float32> adj_42 = {};
        wp::vec_t<3,wp::float32> adj_43 = {};
        wp::vec_t<3,wp::float32> adj_44 = {};
        wp::vec_t<3,wp::float32> adj_45 = {};
        wp::float32 adj_46 = {};
        wp::float32 adj_47 = {};
        wp::float32 adj_48 = {};
        wp::float32 adj_49 = {};
        bool adj_50 = {};
        wp::float32 adj_51 = {};
        wp::array_t<wp::float32> adj_52 = {};
        wp::array_t<wp::float32> adj_53 = {};
        wp::float32 adj_54 = {};
        wp::float32 adj_55 = {};
        wp::float32 adj_56 = {};
        wp::float32 adj_57 = {};
        wp::vec_t<6,wp::float32> adj_58 = {};
        bool adj_59 = {};
        wp::vec_t<6,wp::float32> adj_60 = {};
        wp::vec_t<6,wp::float32> adj_61 = {};
        wp::vec_t<6,wp::float32> adj_62 = {};
        wp::vec_t<6,wp::float32> adj_63 = {};
        wp::vec_t<3,wp::float32> adj_64 = {};
        wp::vec_t<3,wp::float32> adj_65 = {};
        wp::vec_t<3,wp::float32> adj_66 = {};
        wp::vec_t<3,wp::float32> adj_67 = {};
        wp::vec_t<3,wp::float32> adj_68 = {};
        wp::vec_t<3,wp::float32> adj_69 = {};
        wp::vec_t<3,wp::float32> adj_70 = {};
        wp::vec_t<3,wp::float32> adj_71 = {};
        wp::vec_t<3,wp::float32> adj_72 = {};
        wp::float32 adj_73 = {};
        wp::vec_t<3,wp::float32> adj_74 = {};
        wp::float32 adj_75 = {};
        wp::vec_t<3,wp::float32> adj_76 = {};
        wp::vec_t<3,wp::float32> adj_77 = {};
        wp::float32 adj_78 = {};
        bool adj_79 = {};
        wp::float32 adj_80 = {};
        wp::float32 adj_81 = {};
        wp::float32 adj_82 = {};
        wp::float32 adj_83 = {};
        wp::float32 adj_84 = {};
        wp::float32 adj_85 = {};
        wp::float32 adj_86 = {};
        wp::float32 adj_87 = {};
        wp::float32 adj_88 = {};
        wp::float32 adj_89 = {};
        bool adj_90 = {};
        wp::vec_t<3,wp::float32> adj_91 = {};
        wp::quat_t<wp::float32> adj_92 = {};
        wp::vec_t<3,wp::float32> adj_93 = {};
        wp::mat_t<3,3,wp::float32> adj_94 = {};
        wp::mat_t<3,3,wp::float32> adj_95 = {};
        wp::mat_t<3,3,wp::float32> adj_96 = {};
        wp::float32 adj_97 = {};
        wp::vec_t<3,wp::float32> adj_98 = {};
        wp::float32 adj_99 = {};
        wp::float32 adj_100 = {};
        wp::float32 adj_101 = {};
        wp::float32 adj_102 = {};
        wp::float32 adj_103 = {};
        bool adj_104 = {};
        wp::float32 adj_105 = {};
        wp::float32 adj_106 = {};
        wp::float32 adj_107 = {};
        wp::float32 adj_108 = {};
        wp::float32 adj_109 = {};
        wp::vec_t<3,wp::float32> adj_110 = {};
        wp::vec_t<3,wp::float32> adj_111 = {};
        wp::vec_t<3,wp::float32> adj_112 = {};
        wp::vec_t<3,wp::float32> adj_113 = {};
        wp::vec_t<3,wp::float32> adj_114 = {};
        wp::vec_t<3,wp::float32> adj_115 = {};
        wp::vec_t<3,wp::float32> adj_116 = {};
        bool adj_117 = {};
        wp::vec_t<3,wp::float32> adj_118 = {};
        wp::vec_t<6,wp::float32> adj_119 = {};
        wp::vec_t<6,wp::float32> adj_120 = {};
        //---------
        // forward
        // def my_solve_particle_shape_contacts(                                                  <L 258>
        // tid = wp.tid()                                                                         <L 287>
        var_0 = builtin_tid1d();
        // count = min(contact_max, contact_count[0])                                             <L 288>
        var_2 = wp::address(var_contact_count, var_1);
        var_3 = wp::load(var_2);
        var_4 = wp::min(var_contact_max, var_3);
        // if tid >= count:                                                                       <L 289>
        var_5 = (var_0 >= var_4);
        if (var_5) {
            // return                                                                             <L 290>
            goto label0;
        }
        // shape_index = contact_shape[tid]                                                       <L 291>
        var_6 = wp::address(var_contact_shape, var_0);
        var_7 = wp::load(var_6);
        var_8 = wp::copy(var_7);
        // body_index = shape_body[shape_index]                                                   <L 292>
        var_9 = wp::address(var_shape_body, var_8);
        var_10 = wp::load(var_9);
        var_11 = wp::copy(var_10);
        // particle_index = contact_particle[tid]                                                 <L 293>
        var_12 = wp::address(var_contact_particle, var_0);
        var_13 = wp::load(var_12);
        var_14 = wp::copy(var_13);
        // if (particle_flags[particle_index] & PARTICLE_FLAG_ACTIVE) == 0:                       <L 294>
        var_15 = wp::address(var_particle_flags, var_14);
        var_17 = wp::load(var_15);
        var_18 = wp::bit_and(var_17, var_16);
        var_19 = (var_18 == var_1);
        if (var_19) {
            // return                                                                             <L 295>
            goto label1;
        }
        // px = particle_x[particle_index]                                                        <L 296>
        var_20 = wp::address(var_particle_x, var_14);
        var_21 = wp::load(var_20);
        var_22 = wp::copy(var_21);
        // pv = particle_v[particle_index]                                                        <L 297>
        var_23 = wp::address(var_particle_v, var_14);
        var_24 = wp::load(var_23);
        var_25 = wp::copy(var_24);
        // X_wb = wp.transform_identity()                                                         <L 298>
        var_26 = wp::transform_identity<wp::float32>();
        // X_com = wp.vec3()                                                                      <L 299>
        var_27 = wp::vec_t<3,wp::float32>();
        // if body_index >= 0:                                                                    <L 300>
        var_28 = (var_11 >= var_1);
        if (var_28) {
            // X_wb = body_q[body_index]                                                          <L 301>
            var_29 = wp::address(var_body_q, var_11);
            var_30 = wp::load(var_29);
            var_31 = wp::copy(var_30);
            // X_com = body_com[body_index]                                                       <L 302>
            var_32 = wp::address(var_body_com, var_11);
            var_33 = wp::load(var_32);
            var_34 = wp::copy(var_33);
        }
        var_35 = wp::select(var_28, var_26, var_31);
        var_36 = wp::select(var_28, var_27, var_34);
        // bx = wp.transform_point(X_wb, contact_body_pos[tid])                                   <L 303>
        var_37 = wp::address(var_contact_body_pos, var_0);
        var_38 = wp::load(var_37);
        var_39 = wp::transform_point(var_35, var_38);
        // r = bx - wp.transform_point(X_wb, X_com)                                               <L 304>
        var_40 = wp::transform_point(var_35, var_36);
        var_41 = wp::sub(var_39, var_40);
        // n = contact_normal[tid]                                                                <L 305>
        var_42 = wp::address(var_contact_normal, var_0);
        var_43 = wp::load(var_42);
        var_44 = wp::copy(var_43);
        // c = wp.dot(n, px - bx) - particle_radius[particle_index]                               <L 306>
        var_45 = wp::sub(var_22, var_39);
        var_46 = wp::dot(var_44, var_45);
        var_47 = wp::address(var_particle_radius, var_14);
        var_48 = wp::load(var_47);
        var_49 = wp::sub(var_46, var_48);
        // if c > particle_ka:                                                                    <L 307>
        var_50 = (var_49 > var_particle_ka);
        if (var_50) {
            // return                                                                             <L 308>
            goto label2;
        }
        // mu = 0.5 * (particle_mu + shape_materials.mu[shape_index])                             <L 309>
        var_52 = &(var_shape_materials.mu);
        var_53 = wp::load(var_52);
        var_54 = wp::address(var_53, var_8);
        var_55 = wp::load(var_54);
        var_56 = wp::add(var_particle_mu, var_55);
        var_57 = wp::mul(var_51, var_56);
        // body_v_s = wp.spatial_vector()                                                         <L 310>
        var_58 = wp::vec_t<6,wp::float32>();
        // if body_index >= 0:                                                                    <L 311>
        var_59 = (var_11 >= var_1);
        if (var_59) {
            // body_v_s = body_qd[body_index]                                                     <L 312>
            var_60 = wp::address(var_body_qd, var_11);
            var_61 = wp::load(var_60);
            var_62 = wp::copy(var_61);
        }
        var_63 = wp::select(var_59, var_58, var_62);
        // body_w = wp.spatial_top(body_v_s)                                                      <L 313>
        var_64 = wp::spatial_top(var_63);
        // body_v = wp.spatial_bottom(body_v_s)                                                   <L 314>
        var_65 = wp::spatial_bottom(var_63);
        // bv = body_v + wp.cross(body_w, r) + wp.transform_vector(X_wb, contact_body_vel[tid])       <L 315>
        var_66 = wp::cross(var_64, var_41);
        var_67 = wp::add(var_65, var_66);
        var_68 = wp::address(var_contact_body_vel, var_0);
        var_69 = wp::load(var_68);
        var_70 = wp::transform_vector(var_35, var_69);
        var_71 = wp::add(var_67, var_70);
        // v = pv - bv                                                                            <L 316>
        var_72 = wp::sub(var_25, var_71);
        // lambda_n = c                                                                           <L 317>
        var_73 = wp::copy(var_49);
        // delta_n = n * lambda_n                                                                 <L 318>
        var_74 = wp::mul(var_44, var_73);
        // vn = wp.dot(n, v)                                                                      <L 319>
        var_75 = wp::dot(var_44, var_72);
        // vt = v - n * vn                                                                        <L 320>
        var_76 = wp::mul(var_44, var_75);
        var_77 = wp::sub(var_72, var_76);
        // if wp.abs(vn)<staticGroundVelocityThresholdRatio:                                      <L 321>
        var_78 = wp::abs(var_75);
        var_79 = (var_78 < var_staticGroundVelocityThresholdRatio);
        if (var_79) {
            // mu=staticGroundFriction                                                            <L 322>
            var_80 = wp::copy(var_staticGroundFriction);
        }
        var_81 = wp::select(var_79, var_57, var_80);
        // w1 = particle_invmass[particle_index]                                                  <L 323>
        var_82 = wp::address(var_particle_invmass, var_14);
        var_83 = wp::load(var_82);
        var_84 = wp::copy(var_83);
        // w2 = particle_invmass[particle_index]*wp.float(0.0)                                    <L 324>
        var_85 = wp::address(var_particle_invmass, var_14);
        var_87 = wp::float(var_86);
        var_88 = wp::load(var_85);
        var_89 = wp::mul(var_88, var_87);
        // if body_index >= 0:                                                                    <L 325>
        var_90 = (var_11 >= var_1);
        if (var_90) {
            // angular = wp.cross(r, n)                                                           <L 326>
            var_91 = wp::cross(var_41, var_44);
            // q = wp.transform_get_rotation(X_wb)                                                <L 327>
            var_92 = wp::transform_get_rotation(var_35);
            // rot_angular = wp.quat_rotate_inv(q, angular)                                       <L 328>
            var_93 = wp::quat_rotate_inv(var_92, var_91);
            // I_inv = body_I_inv[body_index]                                                     <L 329>
            var_94 = wp::address(var_body_I_inv, var_11);
            var_95 = wp::load(var_94);
            var_96 = wp::copy(var_95);
            // w2 = body_m_inv[body_index] + wp.dot(rot_angular, I_inv * rot_angular)             <L 330>
            var_97 = wp::address(var_body_m_inv, var_11);
            var_98 = wp::mul(var_96, var_93);
            var_99 = wp::dot(var_93, var_98);
            var_100 = wp::load(var_97);
            var_101 = wp::add(var_100, var_99);
        }
        var_102 = wp::select(var_90, var_89, var_101);
        // denom = w1 + w2                                                                        <L 331>
        var_103 = wp::add(var_84, var_102);
        // if denom == 0.0:                                                                       <L 332>
        var_104 = (var_103 == var_86);
        if (var_104) {
            // return                                                                             <L 333>
            goto label3;
        }
        // lambda_f = wp.max(mu * lambda_n, -wp.length(vt) * dt)                                  <L 334>
        var_105 = wp::mul(var_81, var_73);
        var_106 = wp::length(var_77);
        var_107 = wp::neg(var_106);
        var_108 = wp::mul(var_107, var_dt);
        var_109 = wp::max(var_105, var_108);
        // delta_f = wp.normalize(vt) * lambda_f                                                  <L 335>
        var_110 = wp::normalize(var_77);
        var_111 = wp::mul(var_110, var_109);
        // delta_total = (delta_f - delta_n) / denom * relaxation * w1                            <L 336>
        var_112 = wp::sub(var_111, var_74);
        var_113 = wp::div(var_112, var_103);
        var_114 = wp::mul(var_113, var_relaxation);
        var_115 = wp::mul(var_114, var_84);
        // wp.atomic_add(delta, particle_index, delta_total)                                      <L 337>
        // var_116 = wp::atomic_add(var_delta, var_14, var_115);
        // if body_index >= 0:                                                                    <L 338>
        var_117 = (var_11 >= var_1);
        if (var_117) {
            // delta_t = wp.cross(r, delta_total)                                                 <L 339>
            var_118 = wp::cross(var_41, var_115);
            // wp.atomic_sub(body_delta, body_index, wp.spatial_vector(delta_t, delta_total))       <L 340>
            var_119 = wp::vec_t<6,wp::float32>(var_118, var_115);
            // var_120 = wp::atomic_sub(var_body_delta, var_11, var_119);
        }
        //---------
        // reverse
        if (var_117) {
            wp::adj_atomic_sub(var_body_delta, var_11, var_119, adj_body_delta, adj_11, adj_119, adj_120);
            wp::adj_vec_t(var_118, var_115, adj_118, adj_115, adj_119);
            // adj: wp.atomic_sub(body_delta, body_index, wp.spatial_vector(delta_t, delta_total))  <L 340>
            wp::adj_cross(var_41, var_115, adj_41, adj_115, adj_118);
            // adj: delta_t = wp.cross(r, delta_total)                                            <L 339>
        }
        // adj: if body_index >= 0:                                                               <L 338>
        wp::adj_atomic_add(var_delta, var_14, var_115, adj_delta, adj_14, adj_115, adj_116);
        // adj: wp.atomic_add(delta, particle_index, delta_total)                                 <L 337>
        wp::adj_mul(var_114, var_84, adj_114, adj_84, adj_115);
        wp::adj_mul(var_113, var_relaxation, adj_113, adj_relaxation, adj_114);
        wp::adj_div(var_112, var_103, adj_112, adj_103, adj_113);
        wp::adj_sub(var_111, var_74, adj_111, adj_74, adj_112);
        // adj: delta_total = (delta_f - delta_n) / denom * relaxation * w1                       <L 336>
        wp::adj_mul(var_110, var_109, adj_110, adj_109, adj_111);
        wp::adj_normalize(var_77, var_110, adj_77, adj_110);
        // adj: delta_f = wp.normalize(vt) * lambda_f                                             <L 335>
        wp::adj_max(var_105, var_108, adj_105, adj_108, adj_109);
        wp::adj_mul(var_107, var_dt, adj_107, adj_dt, adj_108);
        wp::adj_neg(var_106, adj_106, adj_107);
        wp::adj_length(var_77, var_106, adj_77, adj_106);
        wp::adj_mul(var_81, var_73, adj_81, adj_73, adj_105);
        // adj: lambda_f = wp.max(mu * lambda_n, -wp.length(vt) * dt)                             <L 334>
        if (var_104) {
            label3:;
            // adj: return                                                                        <L 333>
        }
        // adj: if denom == 0.0:                                                                  <L 332>
        wp::adj_add(var_84, var_102, adj_84, adj_102, adj_103);
        // adj: denom = w1 + w2                                                                   <L 331>
        wp::adj_select(var_90, var_89, var_101, adj_90, adj_89, adj_101, adj_102);
        if (var_90) {
            wp::adj_add(var_100, var_99, adj_97, adj_99, adj_101);
            wp::adj_load(var_97, adj_97, adj_100);
            wp::adj_dot(var_93, var_98, adj_93, adj_98, adj_99);
            wp::adj_mul(var_96, var_93, adj_96, adj_93, adj_98);
            wp::adj_address(var_body_m_inv, var_11, adj_body_m_inv, adj_11, adj_97);
            // adj: w2 = body_m_inv[body_index] + wp.dot(rot_angular, I_inv * rot_angular)        <L 330>
            wp::adj_copy(var_95, adj_94, adj_96);
            wp::adj_load(var_94, adj_94, adj_95);
            wp::adj_address(var_body_I_inv, var_11, adj_body_I_inv, adj_11, adj_94);
            // adj: I_inv = body_I_inv[body_index]                                                <L 329>
            wp::adj_quat_rotate_inv(var_92, var_91, adj_92, adj_91, adj_93);
            // adj: rot_angular = wp.quat_rotate_inv(q, angular)                                  <L 328>
            wp::adj_transform_get_rotation(var_35, adj_35, adj_92);
            // adj: q = wp.transform_get_rotation(X_wb)                                           <L 327>
            wp::adj_cross(var_41, var_44, adj_41, adj_44, adj_91);
            // adj: angular = wp.cross(r, n)                                                      <L 326>
        }
        // adj: if body_index >= 0:                                                               <L 325>
        wp::adj_mul(var_88, var_87, adj_85, adj_87, adj_89);
        wp::adj_load(var_85, adj_85, adj_88);
        wp::adj_float(var_86, adj_86, adj_87);
        wp::adj_address(var_particle_invmass, var_14, adj_particle_invmass, adj_14, adj_85);
        // adj: w2 = particle_invmass[particle_index]*wp.float(0.0)                               <L 324>
        wp::adj_copy(var_83, adj_82, adj_84);
        wp::adj_load(var_82, adj_82, adj_83);
        wp::adj_address(var_particle_invmass, var_14, adj_particle_invmass, adj_14, adj_82);
        // adj: w1 = particle_invmass[particle_index]                                             <L 323>
        wp::adj_select(var_79, var_57, var_80, adj_79, adj_57, adj_80, adj_81);
        if (var_79) {
            wp::adj_copy(var_staticGroundFriction, adj_staticGroundFriction, adj_80);
            // adj: mu=staticGroundFriction                                                       <L 322>
        }
        wp::adj_abs(var_75, adj_75, adj_78);
        // adj: if wp.abs(vn)<staticGroundVelocityThresholdRatio:                                 <L 321>
        wp::adj_sub(var_72, var_76, adj_72, adj_76, adj_77);
        wp::adj_mul(var_44, var_75, adj_44, adj_75, adj_76);
        // adj: vt = v - n * vn                                                                   <L 320>
        wp::adj_dot(var_44, var_72, adj_44, adj_72, adj_75);
        // adj: vn = wp.dot(n, v)                                                                 <L 319>
        wp::adj_mul(var_44, var_73, adj_44, adj_73, adj_74);
        // adj: delta_n = n * lambda_n                                                            <L 318>
        wp::adj_copy(var_49, adj_49, adj_73);
        // adj: lambda_n = c                                                                      <L 317>
        wp::adj_sub(var_25, var_71, adj_25, adj_71, adj_72);
        // adj: v = pv - bv                                                                       <L 316>
        wp::adj_add(var_67, var_70, adj_67, adj_70, adj_71);
        wp::adj_transform_vector(var_35, var_69, adj_35, adj_68, adj_70);
        wp::adj_load(var_68, adj_68, adj_69);
        wp::adj_address(var_contact_body_vel, var_0, adj_contact_body_vel, adj_0, adj_68);
        wp::adj_add(var_65, var_66, adj_65, adj_66, adj_67);
        wp::adj_cross(var_64, var_41, adj_64, adj_41, adj_66);
        // adj: bv = body_v + wp.cross(body_w, r) + wp.transform_vector(X_wb, contact_body_vel[tid])  <L 315>
        wp::adj_spatial_bottom(var_63, adj_63, adj_65);
        // adj: body_v = wp.spatial_bottom(body_v_s)                                              <L 314>
        wp::adj_spatial_top(var_63, adj_63, adj_64);
        // adj: body_w = wp.spatial_top(body_v_s)                                                 <L 313>
        wp::adj_select(var_59, var_58, var_62, adj_59, adj_58, adj_62, adj_63);
        if (var_59) {
            wp::adj_copy(var_61, adj_60, adj_62);
            wp::adj_load(var_60, adj_60, adj_61);
            wp::adj_address(var_body_qd, var_11, adj_body_qd, adj_11, adj_60);
            // adj: body_v_s = body_qd[body_index]                                                <L 312>
        }
        // adj: if body_index >= 0:                                                               <L 311>
        // adj: body_v_s = wp.spatial_vector()                                                    <L 310>
        wp::adj_mul(var_51, var_56, adj_51, adj_56, adj_57);
        wp::adj_add(var_particle_mu, var_55, adj_particle_mu, adj_54, adj_56);
        wp::adj_load(var_54, adj_54, adj_55);
        wp::adj_address(var_53, var_8, adj_52, adj_8, adj_54);
        wp::adj_load(var_52, adj_52, adj_53);
        adj_shape_materials.mu = adj_52;
        // adj: mu = 0.5 * (particle_mu + shape_materials.mu[shape_index])                        <L 309>
        if (var_50) {
            label2:;
            // adj: return                                                                        <L 308>
        }
        // adj: if c > particle_ka:                                                               <L 307>
        wp::adj_sub(var_46, var_48, adj_46, adj_47, adj_49);
        wp::adj_load(var_47, adj_47, adj_48);
        wp::adj_address(var_particle_radius, var_14, adj_particle_radius, adj_14, adj_47);
        wp::adj_dot(var_44, var_45, adj_44, adj_45, adj_46);
        wp::adj_sub(var_22, var_39, adj_22, adj_39, adj_45);
        // adj: c = wp.dot(n, px - bx) - particle_radius[particle_index]                          <L 306>
        wp::adj_copy(var_43, adj_42, adj_44);
        wp::adj_load(var_42, adj_42, adj_43);
        wp::adj_address(var_contact_normal, var_0, adj_contact_normal, adj_0, adj_42);
        // adj: n = contact_normal[tid]                                                           <L 305>
        wp::adj_sub(var_39, var_40, adj_39, adj_40, adj_41);
        wp::adj_transform_point(var_35, var_36, adj_35, adj_36, adj_40);
        // adj: r = bx - wp.transform_point(X_wb, X_com)                                          <L 304>
        wp::adj_transform_point(var_35, var_38, adj_35, adj_37, adj_39);
        wp::adj_load(var_37, adj_37, adj_38);
        wp::adj_address(var_contact_body_pos, var_0, adj_contact_body_pos, adj_0, adj_37);
        // adj: bx = wp.transform_point(X_wb, contact_body_pos[tid])                              <L 303>
        wp::adj_select(var_28, var_27, var_34, adj_28, adj_27, adj_34, adj_36);
        wp::adj_select(var_28, var_26, var_31, adj_28, adj_26, adj_31, adj_35);
        if (var_28) {
            wp::adj_copy(var_33, adj_32, adj_34);
            wp::adj_load(var_32, adj_32, adj_33);
            wp::adj_address(var_body_com, var_11, adj_body_com, adj_11, adj_32);
            // adj: X_com = body_com[body_index]                                                  <L 302>
            wp::adj_copy(var_30, adj_29, adj_31);
            wp::adj_load(var_29, adj_29, adj_30);
            wp::adj_address(var_body_q, var_11, adj_body_q, adj_11, adj_29);
            // adj: X_wb = body_q[body_index]                                                     <L 301>
        }
        // adj: if body_index >= 0:                                                               <L 300>
        // adj: X_com = wp.vec3()                                                                 <L 299>
        // adj: X_wb = wp.transform_identity()                                                    <L 298>
        wp::adj_copy(var_24, adj_23, adj_25);
        wp::adj_load(var_23, adj_23, adj_24);
        wp::adj_address(var_particle_v, var_14, adj_particle_v, adj_14, adj_23);
        // adj: pv = particle_v[particle_index]                                                   <L 297>
        wp::adj_copy(var_21, adj_20, adj_22);
        wp::adj_load(var_20, adj_20, adj_21);
        wp::adj_address(var_particle_x, var_14, adj_particle_x, adj_14, adj_20);
        // adj: px = particle_x[particle_index]                                                   <L 296>
        if (var_19) {
            label1:;
            // adj: return                                                                        <L 295>
        }
        wp::adj_bit_and(var_17, var_16, adj_15, adj_16, adj_18);
        wp::adj_load(var_15, adj_15, adj_17);
        wp::adj_address(var_particle_flags, var_14, adj_particle_flags, adj_14, adj_15);
        // adj: if (particle_flags[particle_index] & PARTICLE_FLAG_ACTIVE) == 0:                  <L 294>
        wp::adj_copy(var_13, adj_12, adj_14);
        wp::adj_load(var_12, adj_12, adj_13);
        wp::adj_address(var_contact_particle, var_0, adj_contact_particle, adj_0, adj_12);
        // adj: particle_index = contact_particle[tid]                                            <L 293>
        wp::adj_copy(var_10, adj_9, adj_11);
        wp::adj_load(var_9, adj_9, adj_10);
        wp::adj_address(var_shape_body, var_8, adj_shape_body, adj_8, adj_9);
        // adj: body_index = shape_body[shape_index]                                              <L 292>
        wp::adj_copy(var_7, adj_6, adj_8);
        wp::adj_load(var_6, adj_6, adj_7);
        wp::adj_address(var_contact_shape, var_0, adj_contact_shape, adj_0, adj_6);
        // adj: shape_index = contact_shape[tid]                                                  <L 291>
        if (var_5) {
            label0:;
            // adj: return                                                                        <L 290>
        }
        // adj: if tid >= count:                                                                  <L 289>
        wp::adj_min(var_contact_max, var_3, adj_contact_max, adj_2, adj_4);
        wp::adj_load(var_2, adj_2, adj_3);
        wp::adj_address(var_contact_count, var_1, adj_contact_count, adj_1, adj_2);
        // adj: count = min(contact_max, contact_count[0])                                        <L 288>
        // adj: tid = wp.tid()                                                                    <L 287>
        // adj: def my_solve_particle_shape_contacts(                                             <L 258>
        continue;
    }
}



extern "C" __global__ void my_create_soft_contacts_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_x,
    wp::array_t<wp::float32> var_particle_radius,
    wp::array_t<wp::uint32> var_particle_flags,
    wp::array_t<wp::transform_t<wp::float32>> var_body_X_wb,
    wp::array_t<wp::transform_t<wp::float32>> var_shape_X_bs,
    wp::array_t<wp::int32> var_shape_body,
    ModelShapeGeometry var_geo,
    wp::float32 var_margin,
    wp::int32 var_soft_contact_max,
    wp::array_t<wp::int32> var_soft_contact_count,
    wp::array_t<wp::int32> var_soft_contact_particle,
    wp::array_t<wp::int32> var_soft_contact_shape,
    wp::array_t<wp::vec_t<3,wp::float32>> var_soft_contact_body_pos,
    wp::array_t<wp::vec_t<3,wp::float32>> var_soft_contact_body_vel,
    wp::array_t<wp::vec_t<3,wp::float32>> var_soft_contact_normal,
    wp::float32 var_meshSidedFlag)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::uint32* var_2;
        const wp::uint32 var_3 = 1;
        wp::uint32 var_4;
        wp::uint32 var_5;
        const wp::int32 var_6 = 0;
        bool var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::vec_t<3,wp::float32>* var_11;
        wp::vec_t<3,wp::float32> var_12;
        wp::vec_t<3,wp::float32> var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        wp::transform_t<wp::float32> var_17;
        bool var_18;
        wp::transform_t<wp::float32>* var_19;
        wp::transform_t<wp::float32> var_20;
        wp::transform_t<wp::float32> var_21;
        wp::transform_t<wp::float32> var_22;
        wp::transform_t<wp::float32>* var_23;
        wp::transform_t<wp::float32> var_24;
        wp::transform_t<wp::float32> var_25;
        wp::transform_t<wp::float32> var_26;
        wp::transform_t<wp::float32> var_27;
        wp::vec_t<3,wp::float32> var_28;
        wp::array_t<wp::int32>* var_29;
        wp::array_t<wp::int32> var_30;
        wp::int32* var_31;
        wp::int32 var_32;
        wp::int32 var_33;
        wp::array_t<wp::vec_t<3,wp::float32>>* var_34;
        wp::array_t<wp::vec_t<3,wp::float32>> var_35;
        wp::vec_t<3,wp::float32>* var_36;
        wp::vec_t<3,wp::float32> var_37;
        wp::vec_t<3,wp::float32> var_38;
        const wp::float32 var_39 = 1000000.0;
        wp::vec_t<3,wp::float32> var_40;
        wp::vec_t<3,wp::float32> var_41;
        const wp::int32 var_42 = 5;
        bool var_43;
        wp::array_t<wp::uint64>* var_44;
        wp::array_t<wp::uint64> var_45;
        wp::uint64* var_46;
        wp::uint64 var_47;
        wp::uint64 var_48;
        wp::int32 var_49;
        const wp::float32 var_50 = 0.0;
        wp::float32 var_51;
        wp::float32 var_52;
        wp::float32 var_53;
        wp::vec_t<3,wp::float32> var_54;
        wp::float32 var_55;
        const wp::float32 var_56 = 0.001;
        bool var_57;
        wp::vec_t<3,wp::float32> var_58;
        wp::vec_t<3,wp::float32> var_59;
        wp::vec_t<3,wp::float32> var_60;
        wp::vec_t<3,wp::float32> var_61;
        wp::vec_t<3,wp::float32> var_62;
        bool var_63;
        const wp::float32 var_64 = 1.0;
        wp::float32 var_65;
        wp::float32 var_66;
        wp::float32 var_67;
        wp::vec_t<3,wp::float32> var_68;
        wp::vec_t<3,wp::float32> var_69;
        wp::vec_t<3,wp::float32> var_70;
        wp::float32 var_71;
        wp::vec_t<3,wp::float32> var_72;
        wp::vec_t<3,wp::float32> var_73;
        wp::float32 var_74;
        wp::float32 var_75;
        wp::vec_t<3,wp::float32> var_76;
        wp::vec_t<3,wp::float32> var_77;
        wp::float32 var_78;
        bool var_79;
        const wp::int32 var_80 = 1;
        wp::int32 var_81;
        wp::vec_t<3,wp::float32> var_82;
        wp::vec_t<3,wp::float32> var_83;
        wp::vec_t<3,wp::float32> var_84;
        wp::vec_t<3,wp::float32> var_85;
        wp::vec_t<3,wp::float32> var_86;
        //---------
        // forward
        // def my_create_soft_contacts(                                                           <L 343>
        // particle_index, shape_index = wp.tid()                                                 <L 361>
        builtin_tid2d(var_0, var_1);
        // if (particle_flags[particle_index] & PARTICLE_FLAG_ACTIVE) == 0:                       <L 362>
        var_2 = wp::address(var_particle_flags, var_0);
        var_4 = wp::load(var_2);
        var_5 = wp::bit_and(var_4, var_3);
        var_7 = (var_5 == var_6);
        if (var_7) {
            // return                                                                             <L 363>
            return;
        }
        // rigid_index = shape_body[shape_index]                                                  <L 364>
        var_8 = wp::address(var_shape_body, var_1);
        var_9 = wp::load(var_8);
        var_10 = wp::copy(var_9);
        // px = particle_x[particle_index]                                                        <L 365>
        var_11 = wp::address(var_particle_x, var_0);
        var_12 = wp::load(var_11);
        var_13 = wp::copy(var_12);
        // radius = particle_radius[particle_index]                                               <L 366>
        var_14 = wp::address(var_particle_radius, var_0);
        var_15 = wp::load(var_14);
        var_16 = wp::copy(var_15);
        // X_wb = wp.transform_identity()                                                         <L 367>
        var_17 = wp::transform_identity<wp::float32>();
        // if rigid_index >= 0:                                                                   <L 368>
        var_18 = (var_10 >= var_6);
        if (var_18) {
            // X_wb = body_X_wb[rigid_index]                                                      <L 369>
            var_19 = wp::address(var_body_X_wb, var_10);
            var_20 = wp::load(var_19);
            var_21 = wp::copy(var_20);
        }
        var_22 = wp::select(var_18, var_17, var_21);
        // X_bs = shape_X_bs[shape_index]                                                         <L 370>
        var_23 = wp::address(var_shape_X_bs, var_1);
        var_24 = wp::load(var_23);
        var_25 = wp::copy(var_24);
        // X_ws = wp.transform_multiply(X_wb, X_bs)                                               <L 371>
        var_26 = wp::transform_multiply(var_22, var_25);
        // X_sw = wp.transform_inverse(X_ws)                                                      <L 372>
        var_27 = wp::transform_inverse(var_26);
        // x_local = wp.transform_point(X_sw, px)                                                 <L 373>
        var_28 = wp::transform_point(var_27, var_13);
        // geo_type = geo.type[shape_index]                                                       <L 374>
        var_29 = &(var_geo.type);
        var_30 = wp::load(var_29);
        var_31 = wp::address(var_30, var_1);
        var_32 = wp::load(var_31);
        var_33 = wp::copy(var_32);
        // geo_scale = geo.scale[shape_index]                                                     <L 375>
        var_34 = &(var_geo.scale);
        var_35 = wp::load(var_34);
        var_36 = wp::address(var_35, var_1);
        var_37 = wp::load(var_36);
        var_38 = wp::copy(var_37);
        // d = 1.0e6                                                                              <L 376>
        // n = wp.vec3()                                                                          <L 377>
        var_40 = wp::vec_t<3,wp::float32>();
        // v = wp.vec3()                                                                          <L 378>
        var_41 = wp::vec_t<3,wp::float32>();
        // if geo_type == fs5Model.GEO_MESH:                                                      <L 379>
        var_43 = (var_33 == var_42);
        if (var_43) {
            // mesh = geo.source[shape_index]                                                     <L 380>
            var_44 = &(var_geo.source);
            var_45 = wp::load(var_44);
            var_46 = wp::address(var_45, var_1);
            var_47 = wp::load(var_46);
            var_48 = wp::copy(var_47);
            // face_index = int(0)                                                                <L 381>
            var_49 = wp::int(var_6);
            // face_u = float(0.0)                                                                <L 382>
            var_51 = wp::float(var_50);
            // face_v = float(0.0)                                                                <L 383>
            var_52 = wp::float(var_50);
            // sign = float(0.0)                                                                  <L 384>
            var_53 = wp::float(var_50);
            // if wp.mesh_query_point_sign_normal(                                                <L 385>
            // mesh, wp.cw_div(x_local, geo_scale), margin + radius, sign, face_index, face_u, face_v       <L 386>
            var_54 = wp::cw_div(var_28, var_38);
            var_55 = wp::add(var_margin, var_16);
            var_57 = wp::mesh_query_point_sign_normal(var_48, var_54, var_55, var_53, var_49, var_51, var_52, var_56);
            if (var_57) {
                // shape_p = wp.mesh_eval_position(mesh, face_index, face_u, face_v)              <L 388>
                var_58 = wp::mesh_eval_position(var_48, var_49, var_51, var_52);
                // shape_v = wp.mesh_eval_velocity(mesh, face_index, face_u, face_v)              <L 389>
                var_59 = wp::mesh_eval_velocity(var_48, var_49, var_51, var_52);
                // shape_p = wp.cw_mul(shape_p, geo_scale)                                        <L 390>
                var_60 = wp::cw_mul(var_58, var_38);
                // shape_v = wp.cw_mul(shape_v, geo_scale)                                        <L 391>
                var_61 = wp::cw_mul(var_59, var_38);
                // delta = x_local - shape_p                                                      <L 392>
                var_62 = wp::sub(var_28, var_60);
                // if meshSidedFlag == 0.0:                                                       <L 393>
                var_63 = (var_meshSidedFlag == var_50);
                if (var_63) {
                    // sign=1.0                                                                   <L 394>
                }
                var_65 = wp::select(var_63, var_53, var_64);
                // d = wp.length(delta) * sign                                                    <L 395>
                var_66 = wp::length(var_62);
                var_67 = wp::mul(var_66, var_65);
                // n = wp.normalize(delta) * sign                                                 <L 396>
                var_68 = wp::normalize(var_62);
                var_69 = wp::mul(var_68, var_65);
                // v = shape_v                                                                    <L 397>
                var_70 = wp::copy(var_61);
            }
            var_71 = wp::select(var_57, var_39, var_67);
            var_72 = wp::select(var_57, var_40, var_69);
            var_73 = wp::select(var_57, var_41, var_70);
            var_74 = wp::select(var_57, var_53, var_65);
        }
        var_75 = wp::select(var_43, var_39, var_71);
        var_76 = wp::select(var_43, var_40, var_72);
        var_77 = wp::select(var_43, var_41, var_73);
        // if d < margin + radius:                                                                <L 398>
        var_78 = wp::add(var_margin, var_16);
        var_79 = (var_75 < var_78);
        if (var_79) {
            // index = wp.atomic_add(soft_contact_count, 0, 1)                                    <L 399>
            var_81 = wp::atomic_add(var_soft_contact_count, var_6, var_80);
            // body_pos = wp.transform_point(X_bs, x_local - n * d)                               <L 401>
            var_82 = wp::mul(var_76, var_75);
            var_83 = wp::sub(var_28, var_82);
            var_84 = wp::transform_point(var_25, var_83);
            // body_vel = wp.transform_vector(X_bs, v)                                            <L 402>
            var_85 = wp::transform_vector(var_25, var_77);
            // world_normal = wp.transform_vector(X_ws, n)                                        <L 403>
            var_86 = wp::transform_vector(var_26, var_76);
            // soft_contact_shape[index] = shape_index                                            <L 404>
            wp::array_store(var_soft_contact_shape, var_81, var_1);
            // soft_contact_body_pos[index] = body_pos                                            <L 405>
            wp::array_store(var_soft_contact_body_pos, var_81, var_84);
            // soft_contact_body_vel[index] = body_vel                                            <L 406>
            wp::array_store(var_soft_contact_body_vel, var_81, var_85);
            // soft_contact_particle[index] = particle_index                                      <L 407>
            wp::array_store(var_soft_contact_particle, var_81, var_0);
            // soft_contact_normal[index] = world_normal                                          <L 408>
            wp::array_store(var_soft_contact_normal, var_81, var_86);
        }
    }
}

extern "C" __global__ void my_create_soft_contacts_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_x,
    wp::array_t<wp::float32> var_particle_radius,
    wp::array_t<wp::uint32> var_particle_flags,
    wp::array_t<wp::transform_t<wp::float32>> var_body_X_wb,
    wp::array_t<wp::transform_t<wp::float32>> var_shape_X_bs,
    wp::array_t<wp::int32> var_shape_body,
    ModelShapeGeometry var_geo,
    wp::float32 var_margin,
    wp::int32 var_soft_contact_max,
    wp::array_t<wp::int32> var_soft_contact_count,
    wp::array_t<wp::int32> var_soft_contact_particle,
    wp::array_t<wp::int32> var_soft_contact_shape,
    wp::array_t<wp::vec_t<3,wp::float32>> var_soft_contact_body_pos,
    wp::array_t<wp::vec_t<3,wp::float32>> var_soft_contact_body_vel,
    wp::array_t<wp::vec_t<3,wp::float32>> var_soft_contact_normal,
    wp::float32 var_meshSidedFlag,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_particle_x,
    wp::array_t<wp::float32> adj_particle_radius,
    wp::array_t<wp::uint32> adj_particle_flags,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_X_wb,
    wp::array_t<wp::transform_t<wp::float32>> adj_shape_X_bs,
    wp::array_t<wp::int32> adj_shape_body,
    ModelShapeGeometry adj_geo,
    wp::float32 adj_margin,
    wp::int32 adj_soft_contact_max,
    wp::array_t<wp::int32> adj_soft_contact_count,
    wp::array_t<wp::int32> adj_soft_contact_particle,
    wp::array_t<wp::int32> adj_soft_contact_shape,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_soft_contact_body_pos,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_soft_contact_body_vel,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_soft_contact_normal,
    wp::float32 adj_meshSidedFlag)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::uint32* var_2;
        const wp::uint32 var_3 = 1;
        wp::uint32 var_4;
        wp::uint32 var_5;
        const wp::int32 var_6 = 0;
        bool var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::vec_t<3,wp::float32>* var_11;
        wp::vec_t<3,wp::float32> var_12;
        wp::vec_t<3,wp::float32> var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        wp::transform_t<wp::float32> var_17;
        bool var_18;
        wp::transform_t<wp::float32>* var_19;
        wp::transform_t<wp::float32> var_20;
        wp::transform_t<wp::float32> var_21;
        wp::transform_t<wp::float32> var_22;
        wp::transform_t<wp::float32>* var_23;
        wp::transform_t<wp::float32> var_24;
        wp::transform_t<wp::float32> var_25;
        wp::transform_t<wp::float32> var_26;
        wp::transform_t<wp::float32> var_27;
        wp::vec_t<3,wp::float32> var_28;
        wp::array_t<wp::int32>* var_29;
        wp::array_t<wp::int32> var_30;
        wp::int32* var_31;
        wp::int32 var_32;
        wp::int32 var_33;
        wp::array_t<wp::vec_t<3,wp::float32>>* var_34;
        wp::array_t<wp::vec_t<3,wp::float32>> var_35;
        wp::vec_t<3,wp::float32>* var_36;
        wp::vec_t<3,wp::float32> var_37;
        wp::vec_t<3,wp::float32> var_38;
        const wp::float32 var_39 = 1000000.0;
        wp::vec_t<3,wp::float32> var_40;
        wp::vec_t<3,wp::float32> var_41;
        const wp::int32 var_42 = 5;
        bool var_43;
        wp::array_t<wp::uint64>* var_44;
        wp::array_t<wp::uint64> var_45;
        wp::uint64* var_46;
        wp::uint64 var_47;
        wp::uint64 var_48;
        wp::int32 var_49;
        const wp::float32 var_50 = 0.0;
        wp::float32 var_51;
        wp::float32 var_52;
        wp::float32 var_53;
        wp::vec_t<3,wp::float32> var_54;
        wp::float32 var_55;
        const wp::float32 var_56 = 0.001;
        bool var_57;
        wp::vec_t<3,wp::float32> var_58;
        wp::vec_t<3,wp::float32> var_59;
        wp::vec_t<3,wp::float32> var_60;
        wp::vec_t<3,wp::float32> var_61;
        wp::vec_t<3,wp::float32> var_62;
        bool var_63;
        const wp::float32 var_64 = 1.0;
        wp::float32 var_65;
        wp::float32 var_66;
        wp::float32 var_67;
        wp::vec_t<3,wp::float32> var_68;
        wp::vec_t<3,wp::float32> var_69;
        wp::vec_t<3,wp::float32> var_70;
        wp::float32 var_71;
        wp::vec_t<3,wp::float32> var_72;
        wp::vec_t<3,wp::float32> var_73;
        wp::float32 var_74;
        wp::float32 var_75;
        wp::vec_t<3,wp::float32> var_76;
        wp::vec_t<3,wp::float32> var_77;
        wp::float32 var_78;
        bool var_79;
        const wp::int32 var_80 = 1;
        wp::int32 var_81;
        wp::vec_t<3,wp::float32> var_82;
        wp::vec_t<3,wp::float32> var_83;
        wp::vec_t<3,wp::float32> var_84;
        wp::vec_t<3,wp::float32> var_85;
        wp::vec_t<3,wp::float32> var_86;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::uint32 adj_2 = {};
        wp::uint32 adj_3 = {};
        wp::uint32 adj_4 = {};
        wp::uint32 adj_5 = {};
        wp::int32 adj_6 = {};
        bool adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::vec_t<3,wp::float32> adj_11 = {};
        wp::vec_t<3,wp::float32> adj_12 = {};
        wp::vec_t<3,wp::float32> adj_13 = {};
        wp::float32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::float32 adj_16 = {};
        wp::transform_t<wp::float32> adj_17 = {};
        bool adj_18 = {};
        wp::transform_t<wp::float32> adj_19 = {};
        wp::transform_t<wp::float32> adj_20 = {};
        wp::transform_t<wp::float32> adj_21 = {};
        wp::transform_t<wp::float32> adj_22 = {};
        wp::transform_t<wp::float32> adj_23 = {};
        wp::transform_t<wp::float32> adj_24 = {};
        wp::transform_t<wp::float32> adj_25 = {};
        wp::transform_t<wp::float32> adj_26 = {};
        wp::transform_t<wp::float32> adj_27 = {};
        wp::vec_t<3,wp::float32> adj_28 = {};
        wp::array_t<wp::int32> adj_29 = {};
        wp::array_t<wp::int32> adj_30 = {};
        wp::int32 adj_31 = {};
        wp::int32 adj_32 = {};
        wp::int32 adj_33 = {};
        wp::array_t<wp::vec_t<3,wp::float32>> adj_34 = {};
        wp::array_t<wp::vec_t<3,wp::float32>> adj_35 = {};
        wp::vec_t<3,wp::float32> adj_36 = {};
        wp::vec_t<3,wp::float32> adj_37 = {};
        wp::vec_t<3,wp::float32> adj_38 = {};
        wp::float32 adj_39 = {};
        wp::vec_t<3,wp::float32> adj_40 = {};
        wp::vec_t<3,wp::float32> adj_41 = {};
        wp::int32 adj_42 = {};
        bool adj_43 = {};
        wp::array_t<wp::uint64> adj_44 = {};
        wp::array_t<wp::uint64> adj_45 = {};
        wp::uint64 adj_46 = {};
        wp::uint64 adj_47 = {};
        wp::uint64 adj_48 = {};
        wp::int32 adj_49 = {};
        wp::float32 adj_50 = {};
        wp::float32 adj_51 = {};
        wp::float32 adj_52 = {};
        wp::float32 adj_53 = {};
        wp::vec_t<3,wp::float32> adj_54 = {};
        wp::float32 adj_55 = {};
        wp::float32 adj_56 = {};
        bool adj_57 = {};
        wp::vec_t<3,wp::float32> adj_58 = {};
        wp::vec_t<3,wp::float32> adj_59 = {};
        wp::vec_t<3,wp::float32> adj_60 = {};
        wp::vec_t<3,wp::float32> adj_61 = {};
        wp::vec_t<3,wp::float32> adj_62 = {};
        bool adj_63 = {};
        wp::float32 adj_64 = {};
        wp::float32 adj_65 = {};
        wp::float32 adj_66 = {};
        wp::float32 adj_67 = {};
        wp::vec_t<3,wp::float32> adj_68 = {};
        wp::vec_t<3,wp::float32> adj_69 = {};
        wp::vec_t<3,wp::float32> adj_70 = {};
        wp::float32 adj_71 = {};
        wp::vec_t<3,wp::float32> adj_72 = {};
        wp::vec_t<3,wp::float32> adj_73 = {};
        wp::float32 adj_74 = {};
        wp::float32 adj_75 = {};
        wp::vec_t<3,wp::float32> adj_76 = {};
        wp::vec_t<3,wp::float32> adj_77 = {};
        wp::float32 adj_78 = {};
        bool adj_79 = {};
        wp::int32 adj_80 = {};
        wp::int32 adj_81 = {};
        wp::vec_t<3,wp::float32> adj_82 = {};
        wp::vec_t<3,wp::float32> adj_83 = {};
        wp::vec_t<3,wp::float32> adj_84 = {};
        wp::vec_t<3,wp::float32> adj_85 = {};
        wp::vec_t<3,wp::float32> adj_86 = {};
        //---------
        // forward
        // def my_create_soft_contacts(                                                           <L 343>
        // particle_index, shape_index = wp.tid()                                                 <L 361>
        builtin_tid2d(var_0, var_1);
        // if (particle_flags[particle_index] & PARTICLE_FLAG_ACTIVE) == 0:                       <L 362>
        var_2 = wp::address(var_particle_flags, var_0);
        var_4 = wp::load(var_2);
        var_5 = wp::bit_and(var_4, var_3);
        var_7 = (var_5 == var_6);
        if (var_7) {
            // return                                                                             <L 363>
            goto label0;
        }
        // rigid_index = shape_body[shape_index]                                                  <L 364>
        var_8 = wp::address(var_shape_body, var_1);
        var_9 = wp::load(var_8);
        var_10 = wp::copy(var_9);
        // px = particle_x[particle_index]                                                        <L 365>
        var_11 = wp::address(var_particle_x, var_0);
        var_12 = wp::load(var_11);
        var_13 = wp::copy(var_12);
        // radius = particle_radius[particle_index]                                               <L 366>
        var_14 = wp::address(var_particle_radius, var_0);
        var_15 = wp::load(var_14);
        var_16 = wp::copy(var_15);
        // X_wb = wp.transform_identity()                                                         <L 367>
        var_17 = wp::transform_identity<wp::float32>();
        // if rigid_index >= 0:                                                                   <L 368>
        var_18 = (var_10 >= var_6);
        if (var_18) {
            // X_wb = body_X_wb[rigid_index]                                                      <L 369>
            var_19 = wp::address(var_body_X_wb, var_10);
            var_20 = wp::load(var_19);
            var_21 = wp::copy(var_20);
        }
        var_22 = wp::select(var_18, var_17, var_21);
        // X_bs = shape_X_bs[shape_index]                                                         <L 370>
        var_23 = wp::address(var_shape_X_bs, var_1);
        var_24 = wp::load(var_23);
        var_25 = wp::copy(var_24);
        // X_ws = wp.transform_multiply(X_wb, X_bs)                                               <L 371>
        var_26 = wp::transform_multiply(var_22, var_25);
        // X_sw = wp.transform_inverse(X_ws)                                                      <L 372>
        var_27 = wp::transform_inverse(var_26);
        // x_local = wp.transform_point(X_sw, px)                                                 <L 373>
        var_28 = wp::transform_point(var_27, var_13);
        // geo_type = geo.type[shape_index]                                                       <L 374>
        var_29 = &(var_geo.type);
        var_30 = wp::load(var_29);
        var_31 = wp::address(var_30, var_1);
        var_32 = wp::load(var_31);
        var_33 = wp::copy(var_32);
        // geo_scale = geo.scale[shape_index]                                                     <L 375>
        var_34 = &(var_geo.scale);
        var_35 = wp::load(var_34);
        var_36 = wp::address(var_35, var_1);
        var_37 = wp::load(var_36);
        var_38 = wp::copy(var_37);
        // d = 1.0e6                                                                              <L 376>
        // n = wp.vec3()                                                                          <L 377>
        var_40 = wp::vec_t<3,wp::float32>();
        // v = wp.vec3()                                                                          <L 378>
        var_41 = wp::vec_t<3,wp::float32>();
        // if geo_type == fs5Model.GEO_MESH:                                                      <L 379>
        var_43 = (var_33 == var_42);
        if (var_43) {
            // mesh = geo.source[shape_index]                                                     <L 380>
            var_44 = &(var_geo.source);
            var_45 = wp::load(var_44);
            var_46 = wp::address(var_45, var_1);
            var_47 = wp::load(var_46);
            var_48 = wp::copy(var_47);
            // face_index = int(0)                                                                <L 381>
            var_49 = wp::int(var_6);
            // face_u = float(0.0)                                                                <L 382>
            var_51 = wp::float(var_50);
            // face_v = float(0.0)                                                                <L 383>
            var_52 = wp::float(var_50);
            // sign = float(0.0)                                                                  <L 384>
            var_53 = wp::float(var_50);
            // if wp.mesh_query_point_sign_normal(                                                <L 385>
            // mesh, wp.cw_div(x_local, geo_scale), margin + radius, sign, face_index, face_u, face_v       <L 386>
            var_54 = wp::cw_div(var_28, var_38);
            var_55 = wp::add(var_margin, var_16);
            var_57 = wp::mesh_query_point_sign_normal(var_48, var_54, var_55, var_53, var_49, var_51, var_52, var_56);
            if (var_57) {
                // shape_p = wp.mesh_eval_position(mesh, face_index, face_u, face_v)              <L 388>
                var_58 = wp::mesh_eval_position(var_48, var_49, var_51, var_52);
                // shape_v = wp.mesh_eval_velocity(mesh, face_index, face_u, face_v)              <L 389>
                var_59 = wp::mesh_eval_velocity(var_48, var_49, var_51, var_52);
                // shape_p = wp.cw_mul(shape_p, geo_scale)                                        <L 390>
                var_60 = wp::cw_mul(var_58, var_38);
                // shape_v = wp.cw_mul(shape_v, geo_scale)                                        <L 391>
                var_61 = wp::cw_mul(var_59, var_38);
                // delta = x_local - shape_p                                                      <L 392>
                var_62 = wp::sub(var_28, var_60);
                // if meshSidedFlag == 0.0:                                                       <L 393>
                var_63 = (var_meshSidedFlag == var_50);
                if (var_63) {
                    // sign=1.0                                                                   <L 394>
                }
                var_65 = wp::select(var_63, var_53, var_64);
                // d = wp.length(delta) * sign                                                    <L 395>
                var_66 = wp::length(var_62);
                var_67 = wp::mul(var_66, var_65);
                // n = wp.normalize(delta) * sign                                                 <L 396>
                var_68 = wp::normalize(var_62);
                var_69 = wp::mul(var_68, var_65);
                // v = shape_v                                                                    <L 397>
                var_70 = wp::copy(var_61);
            }
            var_71 = wp::select(var_57, var_39, var_67);
            var_72 = wp::select(var_57, var_40, var_69);
            var_73 = wp::select(var_57, var_41, var_70);
            var_74 = wp::select(var_57, var_53, var_65);
        }
        var_75 = wp::select(var_43, var_39, var_71);
        var_76 = wp::select(var_43, var_40, var_72);
        var_77 = wp::select(var_43, var_41, var_73);
        // if d < margin + radius:                                                                <L 398>
        var_78 = wp::add(var_margin, var_16);
        var_79 = (var_75 < var_78);
        if (var_79) {
            // index = wp.atomic_add(soft_contact_count, 0, 1)                                    <L 399>
            // var_81 = wp::atomic_add(var_soft_contact_count, var_6, var_80);
            // body_pos = wp.transform_point(X_bs, x_local - n * d)                               <L 401>
            var_82 = wp::mul(var_76, var_75);
            var_83 = wp::sub(var_28, var_82);
            var_84 = wp::transform_point(var_25, var_83);
            // body_vel = wp.transform_vector(X_bs, v)                                            <L 402>
            var_85 = wp::transform_vector(var_25, var_77);
            // world_normal = wp.transform_vector(X_ws, n)                                        <L 403>
            var_86 = wp::transform_vector(var_26, var_76);
            // soft_contact_shape[index] = shape_index                                            <L 404>
            // wp::array_store(var_soft_contact_shape, var_81, var_1);
            // soft_contact_body_pos[index] = body_pos                                            <L 405>
            // wp::array_store(var_soft_contact_body_pos, var_81, var_84);
            // soft_contact_body_vel[index] = body_vel                                            <L 406>
            // wp::array_store(var_soft_contact_body_vel, var_81, var_85);
            // soft_contact_particle[index] = particle_index                                      <L 407>
            // wp::array_store(var_soft_contact_particle, var_81, var_0);
            // soft_contact_normal[index] = world_normal                                          <L 408>
            // wp::array_store(var_soft_contact_normal, var_81, var_86);
        }
        //---------
        // reverse
        if (var_79) {
            wp::adj_array_store(var_soft_contact_normal, var_81, var_86, adj_soft_contact_normal, adj_81, adj_86);
            // adj: soft_contact_normal[index] = world_normal                                     <L 408>
            wp::adj_array_store(var_soft_contact_particle, var_81, var_0, adj_soft_contact_particle, adj_81, adj_0);
            // adj: soft_contact_particle[index] = particle_index                                 <L 407>
            wp::adj_array_store(var_soft_contact_body_vel, var_81, var_85, adj_soft_contact_body_vel, adj_81, adj_85);
            // adj: soft_contact_body_vel[index] = body_vel                                       <L 406>
            wp::adj_array_store(var_soft_contact_body_pos, var_81, var_84, adj_soft_contact_body_pos, adj_81, adj_84);
            // adj: soft_contact_body_pos[index] = body_pos                                       <L 405>
            wp::adj_array_store(var_soft_contact_shape, var_81, var_1, adj_soft_contact_shape, adj_81, adj_1);
            // adj: soft_contact_shape[index] = shape_index                                       <L 404>
            wp::adj_transform_vector(var_26, var_76, adj_26, adj_76, adj_86);
            // adj: world_normal = wp.transform_vector(X_ws, n)                                   <L 403>
            wp::adj_transform_vector(var_25, var_77, adj_25, adj_77, adj_85);
            // adj: body_vel = wp.transform_vector(X_bs, v)                                       <L 402>
            wp::adj_transform_point(var_25, var_83, adj_25, adj_83, adj_84);
            wp::adj_sub(var_28, var_82, adj_28, adj_82, adj_83);
            wp::adj_mul(var_76, var_75, adj_76, adj_75, adj_82);
            // adj: body_pos = wp.transform_point(X_bs, x_local - n * d)                          <L 401>
            wp::adj_atomic_add(var_soft_contact_count, var_6, var_80, adj_soft_contact_count, adj_6, adj_80, adj_81);
            // adj: index = wp.atomic_add(soft_contact_count, 0, 1)                               <L 399>
        }
        wp::adj_add(var_margin, var_16, adj_margin, adj_16, adj_78);
        // adj: if d < margin + radius:                                                           <L 398>
        wp::adj_select(var_43, var_41, var_73, adj_43, adj_41, adj_73, adj_77);
        wp::adj_select(var_43, var_40, var_72, adj_43, adj_40, adj_72, adj_76);
        wp::adj_select(var_43, var_39, var_71, adj_43, adj_39, adj_71, adj_75);
        if (var_43) {
            wp::adj_select(var_57, var_53, var_65, adj_57, adj_53, adj_65, adj_74);
            wp::adj_select(var_57, var_41, var_70, adj_57, adj_41, adj_70, adj_73);
            wp::adj_select(var_57, var_40, var_69, adj_57, adj_40, adj_69, adj_72);
            wp::adj_select(var_57, var_39, var_67, adj_57, adj_39, adj_67, adj_71);
            if (var_57) {
                wp::adj_copy(var_61, adj_61, adj_70);
                // adj: v = shape_v                                                               <L 397>
                wp::adj_mul(var_68, var_65, adj_68, adj_65, adj_69);
                wp::adj_normalize(var_62, var_68, adj_62, adj_68);
                // adj: n = wp.normalize(delta) * sign                                            <L 396>
                wp::adj_mul(var_66, var_65, adj_66, adj_65, adj_67);
                wp::adj_length(var_62, var_66, adj_62, adj_66);
                // adj: d = wp.length(delta) * sign                                               <L 395>
                wp::adj_select(var_63, var_53, var_64, adj_63, adj_53, adj_64, adj_65);
                if (var_63) {
                    // adj: sign=1.0                                                              <L 394>
                }
                // adj: if meshSidedFlag == 0.0:                                                  <L 393>
                wp::adj_sub(var_28, var_60, adj_28, adj_60, adj_62);
                // adj: delta = x_local - shape_p                                                 <L 392>
                wp::adj_cw_mul(var_59, var_38, adj_59, adj_38, adj_61);
                // adj: shape_v = wp.cw_mul(shape_v, geo_scale)                                   <L 391>
                wp::adj_cw_mul(var_58, var_38, adj_58, adj_38, adj_60);
                // adj: shape_p = wp.cw_mul(shape_p, geo_scale)                                   <L 390>
                wp::adj_mesh_eval_velocity(var_48, var_49, var_51, var_52, adj_48, adj_49, adj_51, adj_52, adj_59);
                // adj: shape_v = wp.mesh_eval_velocity(mesh, face_index, face_u, face_v)         <L 389>
                wp::adj_mesh_eval_position(var_48, var_49, var_51, var_52, adj_48, adj_49, adj_51, adj_52, adj_58);
                // adj: shape_p = wp.mesh_eval_position(mesh, face_index, face_u, face_v)         <L 388>
            }
            wp::adj_mesh_query_point_sign_normal(var_48, var_54, var_55, var_53, var_49, var_51, var_52, var_56, adj_48, adj_54, adj_55, adj_53, adj_49, adj_51, adj_52, adj_56, adj_57);
            wp::adj_add(var_margin, var_16, adj_margin, adj_16, adj_55);
            wp::adj_cw_div(var_28, var_38, var_54, adj_28, adj_38, adj_54);
            // adj: mesh, wp.cw_div(x_local, geo_scale), margin + radius, sign, face_index, face_u, face_v  <L 386>
            // adj: if wp.mesh_query_point_sign_normal(                                           <L 385>
            wp::adj_float(var_50, adj_50, adj_53);
            // adj: sign = float(0.0)                                                             <L 384>
            wp::adj_float(var_50, adj_50, adj_52);
            // adj: face_v = float(0.0)                                                           <L 383>
            wp::adj_float(var_50, adj_50, adj_51);
            // adj: face_u = float(0.0)                                                           <L 382>
            wp::adj_int(var_6, adj_6, adj_49);
            // adj: face_index = int(0)                                                           <L 381>
            wp::adj_copy(var_47, adj_46, adj_48);
            wp::adj_load(var_46, adj_46, adj_47);
            wp::adj_address(var_45, var_1, adj_44, adj_1, adj_46);
            wp::adj_load(var_44, adj_44, adj_45);
            adj_geo.source = adj_44;
            // adj: mesh = geo.source[shape_index]                                                <L 380>
        }
        // adj: if geo_type == fs5Model.GEO_MESH:                                                 <L 379>
        // adj: v = wp.vec3()                                                                     <L 378>
        // adj: n = wp.vec3()                                                                     <L 377>
        // adj: d = 1.0e6                                                                         <L 376>
        wp::adj_copy(var_37, adj_36, adj_38);
        wp::adj_load(var_36, adj_36, adj_37);
        wp::adj_address(var_35, var_1, adj_34, adj_1, adj_36);
        wp::adj_load(var_34, adj_34, adj_35);
        adj_geo.scale = adj_34;
        // adj: geo_scale = geo.scale[shape_index]                                                <L 375>
        wp::adj_copy(var_32, adj_31, adj_33);
        wp::adj_load(var_31, adj_31, adj_32);
        wp::adj_address(var_30, var_1, adj_29, adj_1, adj_31);
        wp::adj_load(var_29, adj_29, adj_30);
        adj_geo.type = adj_29;
        // adj: geo_type = geo.type[shape_index]                                                  <L 374>
        wp::adj_transform_point(var_27, var_13, adj_27, adj_13, adj_28);
        // adj: x_local = wp.transform_point(X_sw, px)                                            <L 373>
        wp::adj_transform_inverse(var_26, adj_26, adj_27);
        // adj: X_sw = wp.transform_inverse(X_ws)                                                 <L 372>
        wp::adj_transform_multiply(var_22, var_25, adj_22, adj_25, adj_26);
        // adj: X_ws = wp.transform_multiply(X_wb, X_bs)                                          <L 371>
        wp::adj_copy(var_24, adj_23, adj_25);
        wp::adj_load(var_23, adj_23, adj_24);
        wp::adj_address(var_shape_X_bs, var_1, adj_shape_X_bs, adj_1, adj_23);
        // adj: X_bs = shape_X_bs[shape_index]                                                    <L 370>
        wp::adj_select(var_18, var_17, var_21, adj_18, adj_17, adj_21, adj_22);
        if (var_18) {
            wp::adj_copy(var_20, adj_19, adj_21);
            wp::adj_load(var_19, adj_19, adj_20);
            wp::adj_address(var_body_X_wb, var_10, adj_body_X_wb, adj_10, adj_19);
            // adj: X_wb = body_X_wb[rigid_index]                                                 <L 369>
        }
        // adj: if rigid_index >= 0:                                                              <L 368>
        // adj: X_wb = wp.transform_identity()                                                    <L 367>
        wp::adj_copy(var_15, adj_14, adj_16);
        wp::adj_load(var_14, adj_14, adj_15);
        wp::adj_address(var_particle_radius, var_0, adj_particle_radius, adj_0, adj_14);
        // adj: radius = particle_radius[particle_index]                                          <L 366>
        wp::adj_copy(var_12, adj_11, adj_13);
        wp::adj_load(var_11, adj_11, adj_12);
        wp::adj_address(var_particle_x, var_0, adj_particle_x, adj_0, adj_11);
        // adj: px = particle_x[particle_index]                                                   <L 365>
        wp::adj_copy(var_9, adj_8, adj_10);
        wp::adj_load(var_8, adj_8, adj_9);
        wp::adj_address(var_shape_body, var_1, adj_shape_body, adj_1, adj_8);
        // adj: rigid_index = shape_body[shape_index]                                             <L 364>
        if (var_7) {
            label0:;
            // adj: return                                                                        <L 363>
        }
        wp::adj_bit_and(var_4, var_3, adj_2, adj_3, adj_5);
        wp::adj_load(var_2, adj_2, adj_4);
        wp::adj_address(var_particle_flags, var_0, adj_particle_flags, adj_0, adj_2);
        // adj: if (particle_flags[particle_index] & PARTICLE_FLAG_ACTIVE) == 0:                  <L 362>
        // adj: particle_index, shape_index = wp.tid()                                            <L 361>
        // adj: def my_create_soft_contacts(                                                      <L 343>
        continue;
    }
}

