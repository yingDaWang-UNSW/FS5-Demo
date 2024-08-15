
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



extern "C" __global__ void update_vbo_transforms_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_instance_id,
    wp::array_t<wp::int32> var_instance_body,
    wp::array_t<wp::transform_t<wp::float32>> var_instance_transforms,
    wp::array_t<wp::vec_t<3,wp::float32>> var_instance_scalings,
    wp::array_t<wp::transform_t<wp::float32>> var_body_q,
    wp::array_t<wp::mat_t<4,4,wp::float32>> var_vbo_transforms)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32* var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        wp::transform_t<wp::float32>* var_4;
        wp::transform_t<wp::float32> var_5;
        wp::transform_t<wp::float32> var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        const wp::int32 var_10 = 0;
        bool var_11;
        wp::transform_t<wp::float32>* var_12;
        wp::transform_t<wp::float32> var_13;
        wp::transform_t<wp::float32> var_14;
        wp::transform_t<wp::float32> var_15;
        wp::transform_t<wp::float32> var_16;
        wp::transform_t<wp::float32> var_17;
        wp::vec_t<3,wp::float32> var_18;
        wp::quat_t<wp::float32> var_19;
        wp::vec_t<3,wp::float32>* var_20;
        wp::vec_t<3,wp::float32> var_21;
        wp::vec_t<3,wp::float32> var_22;
        wp::mat_t<3,3,wp::float32> var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        const wp::int32 var_27 = 1;
        wp::float32 var_28;
        wp::float32 var_29;
        wp::float32 var_30;
        const wp::int32 var_31 = 2;
        wp::float32 var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        const wp::float32 var_35 = 0.0;
        wp::float32 var_36;
        wp::float32 var_37;
        wp::float32 var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        wp::float32 var_45;
        wp::float32 var_46;
        wp::float32 var_47;
        wp::float32 var_48;
        wp::float32 var_49;
        wp::float32 var_50;
        wp::float32 var_51;
        wp::float32 var_52;
        wp::float32 var_53;
        wp::float32 var_54;
        wp::float32 var_55;
        wp::float32 var_56;
        const wp::float32 var_57 = 1.0;
        wp::mat_t<4,4,wp::float32> var_58;
        //---------
        // forward
        // def update_vbo_transforms(                                                             <L 282>
        // tid = wp.tid()                                                                         <L 291>
        var_0 = builtin_tid1d();
        // i = instance_id[tid]                                                                   <L 292>
        var_1 = wp::address(var_instance_id, var_0);
        var_2 = wp::load(var_1);
        var_3 = wp::copy(var_2);
        // X_ws = instance_transforms[i]                                                          <L 293>
        var_4 = wp::address(var_instance_transforms, var_3);
        var_5 = wp::load(var_4);
        var_6 = wp::copy(var_5);
        // if instance_body:                                                                      <L 294>
        if (var_instance_body) {
            // body = instance_body[i]                                                            <L 295>
            var_7 = wp::address(var_instance_body, var_3);
            var_8 = wp::load(var_7);
            var_9 = wp::copy(var_8);
            // if body >= 0:                                                                      <L 296>
            var_11 = (var_9 >= var_10);
            if (var_11) {
                // if body_q:                                                                     <L 297>
                if (var_body_q) {
                    // X_ws = body_q[body] * X_ws                                                 <L 298>
                    var_12 = wp::address(var_body_q, var_9);
                    var_13 = wp::load(var_12);
                    var_14 = wp::mul(var_13, var_6);
                }
                var_15 = wp::select(var_body_q, var_6, var_14);
                if (!var_body_q) {
                    // return                                                                     <L 300>
                    return;
                }
            }
            var_16 = wp::select(var_11, var_6, var_15);
        }
        var_17 = wp::select(var_instance_body, var_6, var_16);
        // p = wp.transform_get_translation(X_ws)                                                 <L 301>
        var_18 = wp::transform_get_translation(var_17);
        // q = wp.transform_get_rotation(X_ws)                                                    <L 302>
        var_19 = wp::transform_get_rotation(var_17);
        // s = instance_scalings[i]                                                               <L 303>
        var_20 = wp::address(var_instance_scalings, var_3);
        var_21 = wp::load(var_20);
        var_22 = wp::copy(var_21);
        // rot = wp.quat_to_matrix(q)                                                             <L 304>
        var_23 = wp::quat_to_matrix(var_19);
        // vbo_transforms[tid] = wp.mat44(                                                        <L 306>
        // rot[0, 0] * s[0],                                                                      <L 307>
        var_24 = wp::extract(var_23, var_10, var_10);
        var_25 = wp::extract(var_22, var_10);
        var_26 = wp::mul(var_24, var_25);
        // rot[1, 0] * s[0],                                                                      <L 308>
        var_28 = wp::extract(var_23, var_27, var_10);
        var_29 = wp::extract(var_22, var_10);
        var_30 = wp::mul(var_28, var_29);
        // rot[2, 0] * s[0],                                                                      <L 309>
        var_32 = wp::extract(var_23, var_31, var_10);
        var_33 = wp::extract(var_22, var_10);
        var_34 = wp::mul(var_32, var_33);
        // 0.0,                                                                                   <L 310>
        // rot[0, 1] * s[1],                                                                      <L 311>
        var_36 = wp::extract(var_23, var_10, var_27);
        var_37 = wp::extract(var_22, var_27);
        var_38 = wp::mul(var_36, var_37);
        // rot[1, 1] * s[1],                                                                      <L 312>
        var_39 = wp::extract(var_23, var_27, var_27);
        var_40 = wp::extract(var_22, var_27);
        var_41 = wp::mul(var_39, var_40);
        // rot[2, 1] * s[1],                                                                      <L 313>
        var_42 = wp::extract(var_23, var_31, var_27);
        var_43 = wp::extract(var_22, var_27);
        var_44 = wp::mul(var_42, var_43);
        // 0.0,                                                                                   <L 314>
        // rot[0, 2] * s[2],                                                                      <L 315>
        var_45 = wp::extract(var_23, var_10, var_31);
        var_46 = wp::extract(var_22, var_31);
        var_47 = wp::mul(var_45, var_46);
        // rot[1, 2] * s[2],                                                                      <L 316>
        var_48 = wp::extract(var_23, var_27, var_31);
        var_49 = wp::extract(var_22, var_31);
        var_50 = wp::mul(var_48, var_49);
        // rot[2, 2] * s[2],                                                                      <L 317>
        var_51 = wp::extract(var_23, var_31, var_31);
        var_52 = wp::extract(var_22, var_31);
        var_53 = wp::mul(var_51, var_52);
        // 0.0,                                                                                   <L 318>
        // p[0],                                                                                  <L 319>
        var_54 = wp::extract(var_18, var_10);
        // p[1],                                                                                  <L 320>
        var_55 = wp::extract(var_18, var_27);
        // p[2],                                                                                  <L 321>
        var_56 = wp::extract(var_18, var_31);
        // 1.0,                                                                                   <L 322>
        var_58 = wp::mat_t<4,4,wp::float32>(var_26, var_30, var_34, var_35, var_38, var_41, var_44, var_35, var_47, var_50, var_53, var_35, var_54, var_55, var_56, var_57);
        // vbo_transforms[tid] = wp.mat44(                                                        <L 306>
        wp::array_store(var_vbo_transforms, var_0, var_58);
    }
}

extern "C" __global__ void update_vbo_transforms_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_instance_id,
    wp::array_t<wp::int32> var_instance_body,
    wp::array_t<wp::transform_t<wp::float32>> var_instance_transforms,
    wp::array_t<wp::vec_t<3,wp::float32>> var_instance_scalings,
    wp::array_t<wp::transform_t<wp::float32>> var_body_q,
    wp::array_t<wp::mat_t<4,4,wp::float32>> var_vbo_transforms,
    wp::array_t<wp::int32> adj_instance_id,
    wp::array_t<wp::int32> adj_instance_body,
    wp::array_t<wp::transform_t<wp::float32>> adj_instance_transforms,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_instance_scalings,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_q,
    wp::array_t<wp::mat_t<4,4,wp::float32>> adj_vbo_transforms)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
    }
}



extern "C" __global__ void update_vbo_vertices_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float32>> var_points,
    wp::array_t<wp::float32> var_vbo_vertices)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        wp::vec_t<3,wp::float32>* var_1;
        wp::vec_t<3,wp::float32> var_2;
        wp::vec_t<3,wp::float32> var_3;
        const wp::int32 var_4 = 0;
        wp::float32 var_5;
        const wp::int32 var_6 = 1;
        wp::float32 var_7;
        const wp::int32 var_8 = 2;
        wp::float32 var_9;
        //---------
        // forward
        // def update_vbo_vertices(                                                               <L 327>
        // tid = wp.tid()                                                                         <L 332>
        var_0 = builtin_tid1d();
        // p = points[tid]                                                                        <L 333>
        var_1 = wp::address(var_points, var_0);
        var_2 = wp::load(var_1);
        var_3 = wp::copy(var_2);
        // vbo_vertices[tid, 0] = p[0]                                                            <L 334>
        var_5 = wp::extract(var_3, var_4);
        wp::array_store(var_vbo_vertices, var_0, var_4, var_5);
        // vbo_vertices[tid, 1] = p[1]                                                            <L 335>
        var_7 = wp::extract(var_3, var_6);
        wp::array_store(var_vbo_vertices, var_0, var_6, var_7);
        // vbo_vertices[tid, 2] = p[2]                                                            <L 336>
        var_9 = wp::extract(var_3, var_8);
        wp::array_store(var_vbo_vertices, var_0, var_8, var_9);
    }
}

extern "C" __global__ void update_vbo_vertices_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float32>> var_points,
    wp::array_t<wp::float32> var_vbo_vertices,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_points,
    wp::array_t<wp::float32> adj_vbo_vertices)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
    }
}



extern "C" __global__ void update_points_positions_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float32>> var_instance_positions,
    wp::array_t<wp::vec_t<3,wp::float32>> var_instance_scalings,
    wp::array_t<wp::mat_t<4,4,wp::float32>> var_vbo_transforms)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        wp::vec_t<3,wp::float32>* var_1;
        wp::vec_t<3,wp::float32> var_2;
        wp::vec_t<3,wp::float32> var_3;
        const wp::float32 var_4 = 1.0;
        wp::vec_t<3,wp::float32> var_5;
        wp::vec_t<3,wp::float32>* var_6;
        wp::vec_t<3,wp::float32> var_7;
        wp::vec_t<3,wp::float32> var_8;
        wp::vec_t<3,wp::float32> var_9;
        const wp::int32 var_10 = 0;
        wp::float32 var_11;
        const wp::float32 var_12 = 0.0;
        const wp::int32 var_13 = 1;
        wp::float32 var_14;
        const wp::int32 var_15 = 2;
        wp::float32 var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::mat_t<4,4,wp::float32> var_20;
        //---------
        // forward
        // def update_points_positions(                                                           <L 340>
        // tid = wp.tid()                                                                         <L 346>
        var_0 = builtin_tid1d();
        // p = instance_positions[tid]                                                            <L 347>
        var_1 = wp::address(var_instance_positions, var_0);
        var_2 = wp::load(var_1);
        var_3 = wp::copy(var_2);
        // s = wp.vec3(1.0)                                                                       <L 348>
        var_5 = wp::vec_t<3,wp::float32>(var_4);
        // if instance_scalings:                                                                  <L 349>
        if (var_instance_scalings) {
            // s = instance_scalings[tid]                                                         <L 350>
            var_6 = wp::address(var_instance_scalings, var_0);
            var_7 = wp::load(var_6);
            var_8 = wp::copy(var_7);
        }
        var_9 = wp::select(var_instance_scalings, var_5, var_8);
        // vbo_transforms[tid] = wp.mat44(                                                        <L 353>
        // s[0],  0.0,  0.0, 0.0,                                                                 <L 354>
        var_11 = wp::extract(var_9, var_10);
        // 0.0, s[1],  0.0, 0.0,                                                                  <L 355>
        var_14 = wp::extract(var_9, var_13);
        // 0.0,  0.0, s[2], 0.0,                                                                  <L 356>
        var_16 = wp::extract(var_9, var_15);
        // p[0], p[1], p[2], 1.0)                                                                 <L 357>
        var_17 = wp::extract(var_3, var_10);
        var_18 = wp::extract(var_3, var_13);
        var_19 = wp::extract(var_3, var_15);
        var_20 = wp::mat_t<4,4,wp::float32>(var_11, var_12, var_12, var_12, var_12, var_14, var_12, var_12, var_12, var_12, var_16, var_12, var_17, var_18, var_19, var_4);
        // vbo_transforms[tid] = wp.mat44(                                                        <L 353>
        wp::array_store(var_vbo_transforms, var_0, var_20);
    }
}

extern "C" __global__ void update_points_positions_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float32>> var_instance_positions,
    wp::array_t<wp::vec_t<3,wp::float32>> var_instance_scalings,
    wp::array_t<wp::mat_t<4,4,wp::float32>> var_vbo_transforms,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_instance_positions,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_instance_scalings,
    wp::array_t<wp::mat_t<4,4,wp::float32>> adj_vbo_transforms)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
    }
}



extern "C" __global__ void update_line_transforms_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float32>> var_lines,
    wp::array_t<wp::mat_t<4,4,wp::float32>> var_vbo_transforms)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        const wp::int32 var_1 = 0;
        wp::vec_t<3,wp::float32>* var_2;
        wp::vec_t<3,wp::float32> var_3;
        wp::vec_t<3,wp::float32> var_4;
        const wp::int32 var_5 = 1;
        wp::vec_t<3,wp::float32>* var_6;
        wp::vec_t<3,wp::float32> var_7;
        wp::vec_t<3,wp::float32> var_8;
        const wp::float32 var_9 = 0.5;
        wp::vec_t<3,wp::float32> var_10;
        wp::vec_t<3,wp::float32> var_11;
        wp::vec_t<3,wp::float32> var_12;
        wp::float32 var_13;
        wp::vec_t<3,wp::float32> var_14;
        const wp::float32 var_15 = 0.0;
        const wp::float32 var_16 = 1.0;
        wp::vec_t<3,wp::float32> var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::vec_t<3,wp::float32> var_20;
        wp::vec_t<3,wp::float32> var_21;
        wp::float32 var_22;
        wp::quat_t<wp::float32> var_23;
        wp::mat_t<3,3,wp::float32> var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        const wp::int32 var_27 = 2;
        wp::float32 var_28;
        wp::float32 var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        wp::float32 var_37;
        wp::float32 var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        wp::mat_t<4,4,wp::float32> var_41;
        //---------
        // forward
        // def update_line_transforms(                                                            <L 362>
        // tid = wp.tid()                                                                         <L 367>
        var_0 = builtin_tid1d();
        // p0 = lines[tid, 0]                                                                     <L 368>
        var_2 = wp::address(var_lines, var_0, var_1);
        var_3 = wp::load(var_2);
        var_4 = wp::copy(var_3);
        // p1 = lines[tid, 1]                                                                     <L 369>
        var_6 = wp::address(var_lines, var_0, var_5);
        var_7 = wp::load(var_6);
        var_8 = wp::copy(var_7);
        // p = 0.5 * (p0 + p1)                                                                    <L 370>
        var_10 = wp::add(var_4, var_8);
        var_11 = wp::mul(var_9, var_10);
        // d = p1 - p0                                                                            <L 371>
        var_12 = wp::sub(var_8, var_4);
        // s = wp.length(d)                                                                       <L 372>
        var_13 = wp::length(var_12);
        // axis = wp.normalize(d)                                                                 <L 373>
        var_14 = wp::normalize(var_12);
        // y_up = wp.vec3(0.0, 1.0, 0.0)                                                          <L 374>
        var_17 = wp::vec_t<3,wp::float32>(var_15, var_16, var_15);
        // angle = wp.acos(wp.dot(axis, y_up))                                                    <L 375>
        var_18 = wp::dot(var_14, var_17);
        var_19 = wp::acos(var_18);
        // axis = wp.normalize(wp.cross(axis, y_up))                                              <L 376>
        var_20 = wp::cross(var_14, var_17);
        var_21 = wp::normalize(var_20);
        // q = wp.quat_from_axis_angle(axis, -angle)                                              <L 377>
        var_22 = wp::neg(var_19);
        var_23 = wp::quat_from_axis_angle(var_21, var_22);
        // rot = wp.quat_to_matrix(q)                                                             <L 378>
        var_24 = wp::quat_to_matrix(var_23);
        // vbo_transforms[tid] = wp.mat44(                                                        <L 381>
        // rot[0, 0],     rot[1, 0],     rot[2, 0], 0.0,                                          <L 382>
        var_25 = wp::extract(var_24, var_1, var_1);
        var_26 = wp::extract(var_24, var_5, var_1);
        var_28 = wp::extract(var_24, var_27, var_1);
        // s * rot[0, 1], s * rot[1, 1], s * rot[2, 1], 0.0,                                      <L 383>
        var_29 = wp::extract(var_24, var_1, var_5);
        var_30 = wp::mul(var_13, var_29);
        var_31 = wp::extract(var_24, var_5, var_5);
        var_32 = wp::mul(var_13, var_31);
        var_33 = wp::extract(var_24, var_27, var_5);
        var_34 = wp::mul(var_13, var_33);
        // rot[0, 2],     rot[1, 2],     rot[2, 2], 0.0,                                          <L 384>
        var_35 = wp::extract(var_24, var_1, var_27);
        var_36 = wp::extract(var_24, var_5, var_27);
        var_37 = wp::extract(var_24, var_27, var_27);
        // p[0],          p[1],          p[2], 1.0,                                               <L 385>
        var_38 = wp::extract(var_11, var_1);
        var_39 = wp::extract(var_11, var_5);
        var_40 = wp::extract(var_11, var_27);
        var_41 = wp::mat_t<4,4,wp::float32>(var_25, var_26, var_28, var_15, var_30, var_32, var_34, var_15, var_35, var_36, var_37, var_15, var_38, var_39, var_40, var_16);
        // vbo_transforms[tid] = wp.mat44(                                                        <L 381>
        wp::array_store(var_vbo_transforms, var_0, var_41);
    }
}

extern "C" __global__ void update_line_transforms_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float32>> var_lines,
    wp::array_t<wp::mat_t<4,4,wp::float32>> var_vbo_transforms,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_lines,
    wp::array_t<wp::mat_t<4,4,wp::float32>> adj_vbo_transforms)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
    }
}



extern "C" __global__ void compute_gfx_vertices_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_indices,
    wp::array_t<wp::vec_t<3,wp::float32>> var_vertices,
    wp::array_t<wp::float32> var_gfx_vertices)
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
        wp::vec_t<3,wp::float32>* var_4;
        wp::vec_t<3,wp::float32> var_5;
        wp::vec_t<3,wp::float32> var_6;
        const wp::int32 var_7 = 1;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::vec_t<3,wp::float32>* var_10;
        wp::vec_t<3,wp::float32> var_11;
        wp::vec_t<3,wp::float32> var_12;
        const wp::int32 var_13 = 2;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::vec_t<3,wp::float32>* var_16;
        wp::vec_t<3,wp::float32> var_17;
        wp::vec_t<3,wp::float32> var_18;
        const wp::int32 var_19 = 3;
        wp::int32 var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::float32 var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        wp::vec_t<3,wp::float32> var_32;
        wp::vec_t<3,wp::float32> var_33;
        wp::vec_t<3,wp::float32> var_34;
        wp::vec_t<3,wp::float32> var_35;
        wp::float32 var_36;
        wp::float32 var_37;
        const wp::int32 var_38 = 4;
        wp::float32 var_39;
        const wp::int32 var_40 = 5;
        wp::float32 var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        wp::float32 var_45;
        wp::float32 var_46;
        //---------
        // forward
        // def compute_gfx_vertices(                                                              <L 391>
        // tid = wp.tid()                                                                         <L 397>
        var_0 = builtin_tid1d();
        // v0 = vertices[indices[tid, 0]]                                                         <L 398>
        var_2 = wp::address(var_indices, var_0, var_1);
        var_3 = wp::load(var_2);
        var_4 = wp::address(var_vertices, var_3);
        var_5 = wp::load(var_4);
        var_6 = wp::copy(var_5);
        // v1 = vertices[indices[tid, 1]]                                                         <L 399>
        var_8 = wp::address(var_indices, var_0, var_7);
        var_9 = wp::load(var_8);
        var_10 = wp::address(var_vertices, var_9);
        var_11 = wp::load(var_10);
        var_12 = wp::copy(var_11);
        // v2 = vertices[indices[tid, 2]]                                                         <L 400>
        var_14 = wp::address(var_indices, var_0, var_13);
        var_15 = wp::load(var_14);
        var_16 = wp::address(var_vertices, var_15);
        var_17 = wp::load(var_16);
        var_18 = wp::copy(var_17);
        // i = tid * 3                                                                            <L 401>
        var_20 = wp::mul(var_0, var_19);
        // j = i + 1                                                                              <L 402>
        var_21 = wp::add(var_20, var_7);
        // k = i + 2                                                                              <L 403>
        var_22 = wp::add(var_20, var_13);
        // gfx_vertices[i, 0] = v0[0]                                                             <L 404>
        var_23 = wp::extract(var_6, var_1);
        wp::array_store(var_gfx_vertices, var_20, var_1, var_23);
        // gfx_vertices[i, 1] = v0[1]                                                             <L 405>
        var_24 = wp::extract(var_6, var_7);
        wp::array_store(var_gfx_vertices, var_20, var_7, var_24);
        // gfx_vertices[i, 2] = v0[2]                                                             <L 406>
        var_25 = wp::extract(var_6, var_13);
        wp::array_store(var_gfx_vertices, var_20, var_13, var_25);
        // gfx_vertices[j, 0] = v1[0]                                                             <L 407>
        var_26 = wp::extract(var_12, var_1);
        wp::array_store(var_gfx_vertices, var_21, var_1, var_26);
        // gfx_vertices[j, 1] = v1[1]                                                             <L 408>
        var_27 = wp::extract(var_12, var_7);
        wp::array_store(var_gfx_vertices, var_21, var_7, var_27);
        // gfx_vertices[j, 2] = v1[2]                                                             <L 409>
        var_28 = wp::extract(var_12, var_13);
        wp::array_store(var_gfx_vertices, var_21, var_13, var_28);
        // gfx_vertices[k, 0] = v2[0]                                                             <L 410>
        var_29 = wp::extract(var_18, var_1);
        wp::array_store(var_gfx_vertices, var_22, var_1, var_29);
        // gfx_vertices[k, 1] = v2[1]                                                             <L 411>
        var_30 = wp::extract(var_18, var_7);
        wp::array_store(var_gfx_vertices, var_22, var_7, var_30);
        // gfx_vertices[k, 2] = v2[2]                                                             <L 412>
        var_31 = wp::extract(var_18, var_13);
        wp::array_store(var_gfx_vertices, var_22, var_13, var_31);
        // n = wp.normalize(wp.cross(v1 - v0, v2 - v0))                                           <L 413>
        var_32 = wp::sub(var_12, var_6);
        var_33 = wp::sub(var_18, var_6);
        var_34 = wp::cross(var_32, var_33);
        var_35 = wp::normalize(var_34);
        // gfx_vertices[i, 3] = n[0]                                                              <L 414>
        var_36 = wp::extract(var_35, var_1);
        wp::array_store(var_gfx_vertices, var_20, var_19, var_36);
        // gfx_vertices[i, 4] = n[1]                                                              <L 415>
        var_37 = wp::extract(var_35, var_7);
        wp::array_store(var_gfx_vertices, var_20, var_38, var_37);
        // gfx_vertices[i, 5] = n[2]                                                              <L 416>
        var_39 = wp::extract(var_35, var_13);
        wp::array_store(var_gfx_vertices, var_20, var_40, var_39);
        // gfx_vertices[j, 3] = n[0]                                                              <L 417>
        var_41 = wp::extract(var_35, var_1);
        wp::array_store(var_gfx_vertices, var_21, var_19, var_41);
        // gfx_vertices[j, 4] = n[1]                                                              <L 418>
        var_42 = wp::extract(var_35, var_7);
        wp::array_store(var_gfx_vertices, var_21, var_38, var_42);
        // gfx_vertices[j, 5] = n[2]                                                              <L 419>
        var_43 = wp::extract(var_35, var_13);
        wp::array_store(var_gfx_vertices, var_21, var_40, var_43);
        // gfx_vertices[k, 3] = n[0]                                                              <L 420>
        var_44 = wp::extract(var_35, var_1);
        wp::array_store(var_gfx_vertices, var_22, var_19, var_44);
        // gfx_vertices[k, 4] = n[1]                                                              <L 421>
        var_45 = wp::extract(var_35, var_7);
        wp::array_store(var_gfx_vertices, var_22, var_38, var_45);
        // gfx_vertices[k, 5] = n[2]                                                              <L 422>
        var_46 = wp::extract(var_35, var_13);
        wp::array_store(var_gfx_vertices, var_22, var_40, var_46);
    }
}

extern "C" __global__ void compute_gfx_vertices_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_indices,
    wp::array_t<wp::vec_t<3,wp::float32>> var_vertices,
    wp::array_t<wp::float32> var_gfx_vertices,
    wp::array_t<wp::int32> adj_indices,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_vertices,
    wp::array_t<wp::float32> adj_gfx_vertices)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
    }
}



extern "C" __global__ void compute_average_normals_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_indices,
    wp::array_t<wp::vec_t<3,wp::float32>> var_vertices,
    wp::array_t<wp::vec_t<3,wp::float32>> var_normals,
    wp::array_t<wp::int32> var_faces_per_vertex)
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
        const wp::int32 var_5 = 1;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        const wp::int32 var_9 = 2;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::vec_t<3,wp::float32>* var_13;
        wp::vec_t<3,wp::float32> var_14;
        wp::vec_t<3,wp::float32> var_15;
        wp::vec_t<3,wp::float32>* var_16;
        wp::vec_t<3,wp::float32> var_17;
        wp::vec_t<3,wp::float32> var_18;
        wp::vec_t<3,wp::float32>* var_19;
        wp::vec_t<3,wp::float32> var_20;
        wp::vec_t<3,wp::float32> var_21;
        wp::vec_t<3,wp::float32> var_22;
        wp::vec_t<3,wp::float32> var_23;
        wp::vec_t<3,wp::float32> var_24;
        wp::vec_t<3,wp::float32> var_25;
        wp::vec_t<3,wp::float32> var_26;
        wp::int32 var_27;
        wp::vec_t<3,wp::float32> var_28;
        wp::int32 var_29;
        wp::vec_t<3,wp::float32> var_30;
        wp::int32 var_31;
        //---------
        // forward
        // def compute_average_normals(                                                           <L 426>
        // tid = wp.tid()                                                                         <L 433>
        var_0 = builtin_tid1d();
        // i = indices[tid, 0]                                                                    <L 434>
        var_2 = wp::address(var_indices, var_0, var_1);
        var_3 = wp::load(var_2);
        var_4 = wp::copy(var_3);
        // j = indices[tid, 1]                                                                    <L 435>
        var_6 = wp::address(var_indices, var_0, var_5);
        var_7 = wp::load(var_6);
        var_8 = wp::copy(var_7);
        // k = indices[tid, 2]                                                                    <L 436>
        var_10 = wp::address(var_indices, var_0, var_9);
        var_11 = wp::load(var_10);
        var_12 = wp::copy(var_11);
        // v0 = vertices[i]                                                                       <L 437>
        var_13 = wp::address(var_vertices, var_4);
        var_14 = wp::load(var_13);
        var_15 = wp::copy(var_14);
        // v1 = vertices[j]                                                                       <L 438>
        var_16 = wp::address(var_vertices, var_8);
        var_17 = wp::load(var_16);
        var_18 = wp::copy(var_17);
        // v2 = vertices[k]                                                                       <L 439>
        var_19 = wp::address(var_vertices, var_12);
        var_20 = wp::load(var_19);
        var_21 = wp::copy(var_20);
        // n = wp.normalize(wp.cross(v1 - v0, v2 - v0))                                           <L 440>
        var_22 = wp::sub(var_18, var_15);
        var_23 = wp::sub(var_21, var_15);
        var_24 = wp::cross(var_22, var_23);
        var_25 = wp::normalize(var_24);
        // wp.atomic_add(normals, i, n)                                                           <L 441>
        var_26 = wp::atomic_add(var_normals, var_4, var_25);
        // wp.atomic_add(faces_per_vertex, i, 1)                                                  <L 442>
        var_27 = wp::atomic_add(var_faces_per_vertex, var_4, var_5);
        // wp.atomic_add(normals, j, n)                                                           <L 443>
        var_28 = wp::atomic_add(var_normals, var_8, var_25);
        // wp.atomic_add(faces_per_vertex, j, 1)                                                  <L 444>
        var_29 = wp::atomic_add(var_faces_per_vertex, var_8, var_5);
        // wp.atomic_add(normals, k, n)                                                           <L 445>
        var_30 = wp::atomic_add(var_normals, var_12, var_25);
        // wp.atomic_add(faces_per_vertex, k, 1)                                                  <L 446>
        var_31 = wp::atomic_add(var_faces_per_vertex, var_12, var_5);
    }
}

extern "C" __global__ void compute_average_normals_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_indices,
    wp::array_t<wp::vec_t<3,wp::float32>> var_vertices,
    wp::array_t<wp::vec_t<3,wp::float32>> var_normals,
    wp::array_t<wp::int32> var_faces_per_vertex,
    wp::array_t<wp::int32> adj_indices,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_vertices,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_normals,
    wp::array_t<wp::int32> adj_faces_per_vertex)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
    }
}



extern "C" __global__ void assemble_gfx_vertices_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float32>> var_vertices,
    wp::array_t<wp::vec_t<3,wp::float32>> var_normals,
    wp::array_t<wp::int32> var_faces_per_vertex,
    wp::array_t<wp::float32> var_gfx_vertices)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        wp::vec_t<3,wp::float32>* var_1;
        wp::vec_t<3,wp::float32> var_2;
        wp::vec_t<3,wp::float32> var_3;
        wp::vec_t<3,wp::float32>* var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::float32 var_7;
        wp::vec_t<3,wp::float32> var_8;
        wp::vec_t<3,wp::float32> var_9;
        const wp::int32 var_10 = 0;
        wp::float32 var_11;
        const wp::int32 var_12 = 1;
        wp::float32 var_13;
        const wp::int32 var_14 = 2;
        wp::float32 var_15;
        wp::float32 var_16;
        const wp::int32 var_17 = 3;
        wp::float32 var_18;
        const wp::int32 var_19 = 4;
        wp::float32 var_20;
        const wp::int32 var_21 = 5;
        //---------
        // forward
        // def assemble_gfx_vertices(                                                             <L 450>
        // tid = wp.tid()                                                                         <L 457>
        var_0 = builtin_tid1d();
        // v = vertices[tid]                                                                      <L 458>
        var_1 = wp::address(var_vertices, var_0);
        var_2 = wp::load(var_1);
        var_3 = wp::copy(var_2);
        // n = normals[tid] / float(faces_per_vertex[tid])                                        <L 459>
        var_4 = wp::address(var_normals, var_0);
        var_5 = wp::address(var_faces_per_vertex, var_0);
        var_6 = wp::load(var_5);
        var_7 = wp::float(var_6);
        var_8 = wp::load(var_4);
        var_9 = wp::div(var_8, var_7);
        // gfx_vertices[tid, 0] = v[0]                                                            <L 460>
        var_11 = wp::extract(var_3, var_10);
        wp::array_store(var_gfx_vertices, var_0, var_10, var_11);
        // gfx_vertices[tid, 1] = v[1]                                                            <L 461>
        var_13 = wp::extract(var_3, var_12);
        wp::array_store(var_gfx_vertices, var_0, var_12, var_13);
        // gfx_vertices[tid, 2] = v[2]                                                            <L 462>
        var_15 = wp::extract(var_3, var_14);
        wp::array_store(var_gfx_vertices, var_0, var_14, var_15);
        // gfx_vertices[tid, 3] = n[0]                                                            <L 463>
        var_16 = wp::extract(var_9, var_10);
        wp::array_store(var_gfx_vertices, var_0, var_17, var_16);
        // gfx_vertices[tid, 4] = n[1]                                                            <L 464>
        var_18 = wp::extract(var_9, var_12);
        wp::array_store(var_gfx_vertices, var_0, var_19, var_18);
        // gfx_vertices[tid, 5] = n[2]                                                            <L 465>
        var_20 = wp::extract(var_9, var_14);
        wp::array_store(var_gfx_vertices, var_0, var_21, var_20);
    }
}

extern "C" __global__ void assemble_gfx_vertices_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float32>> var_vertices,
    wp::array_t<wp::vec_t<3,wp::float32>> var_normals,
    wp::array_t<wp::int32> var_faces_per_vertex,
    wp::array_t<wp::float32> var_gfx_vertices,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_vertices,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_normals,
    wp::array_t<wp::int32> adj_faces_per_vertex,
    wp::array_t<wp::float32> adj_gfx_vertices)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
    }
}



extern "C" __global__ void copy_rgb_frame_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::uint8> var_input_img,
    wp::int32 var_width,
    wp::int32 var_height,
    wp::array_t<wp::float32> var_output_img)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        const wp::int32 var_4 = 3;
        wp::int32 var_5;
        const wp::int32 var_6 = 0;
        wp::int32 var_7;
        wp::uint8* var_8;
        wp::uint8 var_9;
        wp::float32 var_10;
        const wp::int32 var_11 = 1;
        wp::int32 var_12;
        wp::uint8* var_13;
        wp::uint8 var_14;
        wp::float32 var_15;
        const wp::int32 var_16 = 2;
        wp::int32 var_17;
        wp::uint8* var_18;
        wp::uint8 var_19;
        wp::float32 var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        const wp::float32 var_23 = 255.0;
        wp::float32 var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        //---------
        // forward
        // def copy_rgb_frame(                                                                    <L 469>
        // w, v = wp.tid()                                                                        <L 476>
        builtin_tid2d(var_0, var_1);
        // pixel = v * width + w                                                                  <L 477>
        var_2 = wp::mul(var_1, var_width);
        var_3 = wp::add(var_2, var_0);
        // pixel *= 3                                                                             <L 478>
        var_5 = wp::mul(var_3, var_4);
        // r = float(input_img[pixel + 0])                                                        <L 479>
        var_7 = wp::add(var_5, var_6);
        var_8 = wp::address(var_input_img, var_7);
        var_9 = wp::load(var_8);
        var_10 = wp::float(var_9);
        // g = float(input_img[pixel + 1])                                                        <L 480>
        var_12 = wp::add(var_5, var_11);
        var_13 = wp::address(var_input_img, var_12);
        var_14 = wp::load(var_13);
        var_15 = wp::float(var_14);
        // b = float(input_img[pixel + 2])                                                        <L 481>
        var_17 = wp::add(var_5, var_16);
        var_18 = wp::address(var_input_img, var_17);
        var_19 = wp::load(var_18);
        var_20 = wp::float(var_19);
        // v = height - v - 1                                                                     <L 483>
        var_21 = wp::sub(var_height, var_1);
        var_22 = wp::sub(var_21, var_11);
        // output_img[v, w, 0] = r / 255.0                                                        <L 484>
        var_24 = wp::div(var_10, var_23);
        wp::array_store(var_output_img, var_22, var_0, var_6, var_24);
        // output_img[v, w, 1] = g / 255.0                                                        <L 485>
        var_25 = wp::div(var_15, var_23);
        wp::array_store(var_output_img, var_22, var_0, var_11, var_25);
        // output_img[v, w, 2] = b / 255.0                                                        <L 486>
        var_26 = wp::div(var_20, var_23);
        wp::array_store(var_output_img, var_22, var_0, var_16, var_26);
    }
}

extern "C" __global__ void copy_rgb_frame_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::uint8> var_input_img,
    wp::int32 var_width,
    wp::int32 var_height,
    wp::array_t<wp::float32> var_output_img,
    wp::array_t<wp::uint8> adj_input_img,
    wp::int32 adj_width,
    wp::int32 adj_height,
    wp::array_t<wp::float32> adj_output_img)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
    }
}



extern "C" __global__ void copy_depth_frame_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_input_img,
    wp::int32 var_width,
    wp::int32 var_height,
    wp::float32 var_near,
    wp::float32 var_far,
    wp::array_t<wp::float32> var_output_img)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 1;
        wp::int32 var_6;
        const wp::float32 var_7 = 2.0;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        const wp::float32 var_11 = 1.0;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        const wp::int32 var_21 = 0;
        //---------
        // forward
        // def copy_depth_frame(                                                                  <L 490>
        // w, v = wp.tid()                                                                        <L 499>
        builtin_tid2d(var_0, var_1);
        // pixel = v * width + w                                                                  <L 500>
        var_2 = wp::mul(var_1, var_width);
        var_3 = wp::add(var_2, var_0);
        // v = height - v - 1                                                                     <L 502>
        var_4 = wp::sub(var_height, var_1);
        var_6 = wp::sub(var_4, var_5);
        // d = 2.0 * input_img[pixel] - 1.0                                                       <L 503>
        var_8 = wp::address(var_input_img, var_3);
        var_9 = wp::load(var_8);
        var_10 = wp::mul(var_7, var_9);
        var_12 = wp::sub(var_10, var_11);
        // d = 2.0 * near * far / ((far - near) * d - near - far)                                 <L 504>
        var_13 = wp::mul(var_7, var_near);
        var_14 = wp::mul(var_13, var_far);
        var_15 = wp::sub(var_far, var_near);
        var_16 = wp::mul(var_15, var_12);
        var_17 = wp::sub(var_16, var_near);
        var_18 = wp::sub(var_17, var_far);
        var_19 = wp::div(var_14, var_18);
        // output_img[v, w, 0] = -d                                                               <L 505>
        var_20 = wp::neg(var_19);
        wp::array_store(var_output_img, var_6, var_0, var_21, var_20);
    }
}

extern "C" __global__ void copy_depth_frame_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_input_img,
    wp::int32 var_width,
    wp::int32 var_height,
    wp::float32 var_near,
    wp::float32 var_far,
    wp::array_t<wp::float32> var_output_img,
    wp::array_t<wp::float32> adj_input_img,
    wp::int32 adj_width,
    wp::int32 adj_height,
    wp::float32 adj_near,
    wp::float32 adj_far,
    wp::array_t<wp::float32> adj_output_img)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
    }
}



extern "C" __global__ void copy_rgb_frame_tiles_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::uint8> var_input_img,
    wp::array_t<wp::int32> var_positions,
    wp::int32 var_screen_width,
    wp::int32 var_screen_height,
    wp::int32 var_tile_height,
    wp::array_t<wp::float32> var_output_img)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32 var_2;
        wp::array_t<wp::int32> var_3;
        const wp::int32 var_4 = 0;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        const wp::int32 var_8 = 1;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        bool var_16;
        bool var_17;
        bool var_18;
        const wp::float32 var_19 = 0.0;
        const wp::int32 var_20 = 2;
        const wp::int32 var_21 = 3;
        wp::int32 var_22;
        wp::int32 var_23;
        wp::uint8* var_24;
        wp::uint8 var_25;
        wp::float32 var_26;
        wp::int32 var_27;
        wp::uint8* var_28;
        wp::uint8 var_29;
        wp::float32 var_30;
        wp::int32 var_31;
        wp::uint8* var_32;
        wp::uint8 var_33;
        wp::float32 var_34;
        const wp::float32 var_35 = 255.0;
        wp::float32 var_36;
        wp::float32 var_37;
        wp::float32 var_38;
        //---------
        // forward
        // def copy_rgb_frame_tiles(                                                              <L 509>
        // tile, x, y = wp.tid()                                                                  <L 518>
        builtin_tid3d(var_0, var_1, var_2);
        // p = positions[tile]                                                                    <L 519>
        var_3 = wp::view(var_positions, var_0);
        // qx = x + p[0]                                                                          <L 520>
        var_5 = wp::address(var_3, var_4);
        var_6 = wp::load(var_5);
        var_7 = wp::add(var_1, var_6);
        // qy = y + p[1]                                                                          <L 521>
        var_9 = wp::address(var_3, var_8);
        var_10 = wp::load(var_9);
        var_11 = wp::add(var_2, var_10);
        // pixel = qy * screen_width + qx                                                         <L 522>
        var_12 = wp::mul(var_11, var_screen_width);
        var_13 = wp::add(var_12, var_7);
        // y = tile_height - y - 1                                                                <L 524>
        var_14 = wp::sub(var_tile_height, var_2);
        var_15 = wp::sub(var_14, var_8);
        // if qx >= screen_width or qy >= screen_height:                                          <L 525>
        var_16 = (var_7 >= var_screen_width);
        var_17 = (var_11 >= var_screen_height);
        var_18 = var_16 || var_17;
        if (var_18) {
            // output_img[tile, y, x, 0] = 0.0                                                    <L 526>
            wp::array_store(var_output_img, var_0, var_15, var_1, var_4, var_19);
            // output_img[tile, y, x, 1] = 0.0                                                    <L 527>
            wp::array_store(var_output_img, var_0, var_15, var_1, var_8, var_19);
            // output_img[tile, y, x, 2] = 0.0                                                    <L 528>
            wp::array_store(var_output_img, var_0, var_15, var_1, var_20, var_19);
            // return  # prevent out-of-bounds access                                             <L 529>
            return;
        }
        // pixel *= 3                                                                             <L 530>
        var_22 = wp::mul(var_13, var_21);
        // r = float(input_img[pixel + 0])                                                        <L 531>
        var_23 = wp::add(var_22, var_4);
        var_24 = wp::address(var_input_img, var_23);
        var_25 = wp::load(var_24);
        var_26 = wp::float(var_25);
        // g = float(input_img[pixel + 1])                                                        <L 532>
        var_27 = wp::add(var_22, var_8);
        var_28 = wp::address(var_input_img, var_27);
        var_29 = wp::load(var_28);
        var_30 = wp::float(var_29);
        // b = float(input_img[pixel + 2])                                                        <L 533>
        var_31 = wp::add(var_22, var_20);
        var_32 = wp::address(var_input_img, var_31);
        var_33 = wp::load(var_32);
        var_34 = wp::float(var_33);
        // output_img[tile, y, x, 0] = r / 255.0                                                  <L 534>
        var_36 = wp::div(var_26, var_35);
        wp::array_store(var_output_img, var_0, var_15, var_1, var_4, var_36);
        // output_img[tile, y, x, 1] = g / 255.0                                                  <L 535>
        var_37 = wp::div(var_30, var_35);
        wp::array_store(var_output_img, var_0, var_15, var_1, var_8, var_37);
        // output_img[tile, y, x, 2] = b / 255.0                                                  <L 536>
        var_38 = wp::div(var_34, var_35);
        wp::array_store(var_output_img, var_0, var_15, var_1, var_20, var_38);
    }
}

extern "C" __global__ void copy_rgb_frame_tiles_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::uint8> var_input_img,
    wp::array_t<wp::int32> var_positions,
    wp::int32 var_screen_width,
    wp::int32 var_screen_height,
    wp::int32 var_tile_height,
    wp::array_t<wp::float32> var_output_img,
    wp::array_t<wp::uint8> adj_input_img,
    wp::array_t<wp::int32> adj_positions,
    wp::int32 adj_screen_width,
    wp::int32 adj_screen_height,
    wp::int32 adj_tile_height,
    wp::array_t<wp::float32> adj_output_img)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
    }
}



extern "C" __global__ void copy_depth_frame_tiles_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_input_img,
    wp::array_t<wp::int32> var_positions,
    wp::int32 var_screen_width,
    wp::int32 var_screen_height,
    wp::int32 var_tile_height,
    wp::float32 var_near,
    wp::float32 var_far,
    wp::array_t<wp::float32> var_output_img)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32 var_2;
        wp::array_t<wp::int32> var_3;
        const wp::int32 var_4 = 0;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        const wp::int32 var_8 = 1;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        bool var_16;
        bool var_17;
        bool var_18;
        const wp::float32 var_19 = 2.0;
        wp::float32* var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        const wp::float32 var_23 = 1.0;
        wp::float32 var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::float32 var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        //---------
        // forward
        // def copy_depth_frame_tiles(                                                            <L 540>
        // tile, x, y = wp.tid()                                                                  <L 551>
        builtin_tid3d(var_0, var_1, var_2);
        // p = positions[tile]                                                                    <L 552>
        var_3 = wp::view(var_positions, var_0);
        // qx = x + p[0]                                                                          <L 553>
        var_5 = wp::address(var_3, var_4);
        var_6 = wp::load(var_5);
        var_7 = wp::add(var_1, var_6);
        // qy = y + p[1]                                                                          <L 554>
        var_9 = wp::address(var_3, var_8);
        var_10 = wp::load(var_9);
        var_11 = wp::add(var_2, var_10);
        // pixel = qy * screen_width + qx                                                         <L 555>
        var_12 = wp::mul(var_11, var_screen_width);
        var_13 = wp::add(var_12, var_7);
        // y = tile_height - y - 1                                                                <L 557>
        var_14 = wp::sub(var_tile_height, var_2);
        var_15 = wp::sub(var_14, var_8);
        // if qx >= screen_width or qy >= screen_height:                                          <L 558>
        var_16 = (var_7 >= var_screen_width);
        var_17 = (var_11 >= var_screen_height);
        var_18 = var_16 || var_17;
        if (var_18) {
            // output_img[tile, y, x, 0] = far                                                    <L 559>
            wp::array_store(var_output_img, var_0, var_15, var_1, var_4, var_far);
            // return  # prevent out-of-bounds access                                             <L 560>
            return;
        }
        // d = 2.0 * input_img[pixel] - 1.0                                                       <L 561>
        var_20 = wp::address(var_input_img, var_13);
        var_21 = wp::load(var_20);
        var_22 = wp::mul(var_19, var_21);
        var_24 = wp::sub(var_22, var_23);
        // d = 2.0 * near * far / ((far - near) * d - near - far)                                 <L 562>
        var_25 = wp::mul(var_19, var_near);
        var_26 = wp::mul(var_25, var_far);
        var_27 = wp::sub(var_far, var_near);
        var_28 = wp::mul(var_27, var_24);
        var_29 = wp::sub(var_28, var_near);
        var_30 = wp::sub(var_29, var_far);
        var_31 = wp::div(var_26, var_30);
        // output_img[tile, y, x, 0] = -d                                                         <L 563>
        var_32 = wp::neg(var_31);
        wp::array_store(var_output_img, var_0, var_15, var_1, var_4, var_32);
    }
}

extern "C" __global__ void copy_depth_frame_tiles_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_input_img,
    wp::array_t<wp::int32> var_positions,
    wp::int32 var_screen_width,
    wp::int32 var_screen_height,
    wp::int32 var_tile_height,
    wp::float32 var_near,
    wp::float32 var_far,
    wp::array_t<wp::float32> var_output_img,
    wp::array_t<wp::float32> adj_input_img,
    wp::array_t<wp::int32> adj_positions,
    wp::int32 adj_screen_width,
    wp::int32 adj_screen_height,
    wp::int32 adj_tile_height,
    wp::float32 adj_near,
    wp::float32 adj_far,
    wp::array_t<wp::float32> adj_output_img)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
    }
}



extern "C" __global__ void copy_rgb_frame_tile_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::uint8> var_input_img,
    wp::int32 var_offset_x,
    wp::int32 var_offset_y,
    wp::int32 var_screen_width,
    wp::int32 var_screen_height,
    wp::int32 var_tile_height,
    wp::array_t<wp::float32> var_output_img)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        const wp::int32 var_8 = 1;
        wp::int32 var_9;
        bool var_10;
        bool var_11;
        bool var_12;
        const wp::float32 var_13 = 0.0;
        const wp::int32 var_14 = 0;
        const wp::int32 var_15 = 2;
        const wp::int32 var_16 = 3;
        wp::int32 var_17;
        wp::int32 var_18;
        wp::uint8* var_19;
        wp::uint8 var_20;
        wp::float32 var_21;
        wp::int32 var_22;
        wp::uint8* var_23;
        wp::uint8 var_24;
        wp::float32 var_25;
        wp::int32 var_26;
        wp::uint8* var_27;
        wp::uint8 var_28;
        wp::float32 var_29;
        const wp::float32 var_30 = 255.0;
        wp::float32 var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        //---------
        // forward
        // def copy_rgb_frame_tile(                                                               <L 567>
        // tile, x, y = wp.tid()                                                                  <L 577>
        builtin_tid3d(var_0, var_1, var_2);
        // qx = x + offset_x                                                                      <L 578>
        var_3 = wp::add(var_1, var_offset_x);
        // qy = y + offset_y                                                                      <L 579>
        var_4 = wp::add(var_2, var_offset_y);
        // pixel = qy * screen_width + qx                                                         <L 580>
        var_5 = wp::mul(var_4, var_screen_width);
        var_6 = wp::add(var_5, var_3);
        // y = tile_height - y - 1                                                                <L 582>
        var_7 = wp::sub(var_tile_height, var_2);
        var_9 = wp::sub(var_7, var_8);
        // if qx >= screen_width or qy >= screen_height:                                          <L 583>
        var_10 = (var_3 >= var_screen_width);
        var_11 = (var_4 >= var_screen_height);
        var_12 = var_10 || var_11;
        if (var_12) {
            // output_img[tile, y, x, 0] = 0.0                                                    <L 584>
            wp::array_store(var_output_img, var_0, var_9, var_1, var_14, var_13);
            // output_img[tile, y, x, 1] = 0.0                                                    <L 585>
            wp::array_store(var_output_img, var_0, var_9, var_1, var_8, var_13);
            // output_img[tile, y, x, 2] = 0.0                                                    <L 586>
            wp::array_store(var_output_img, var_0, var_9, var_1, var_15, var_13);
            // return  # prevent out-of-bounds access                                             <L 587>
            return;
        }
        // pixel *= 3                                                                             <L 588>
        var_17 = wp::mul(var_6, var_16);
        // r = float(input_img[pixel + 0])                                                        <L 589>
        var_18 = wp::add(var_17, var_14);
        var_19 = wp::address(var_input_img, var_18);
        var_20 = wp::load(var_19);
        var_21 = wp::float(var_20);
        // g = float(input_img[pixel + 1])                                                        <L 590>
        var_22 = wp::add(var_17, var_8);
        var_23 = wp::address(var_input_img, var_22);
        var_24 = wp::load(var_23);
        var_25 = wp::float(var_24);
        // b = float(input_img[pixel + 2])                                                        <L 591>
        var_26 = wp::add(var_17, var_15);
        var_27 = wp::address(var_input_img, var_26);
        var_28 = wp::load(var_27);
        var_29 = wp::float(var_28);
        // output_img[tile, y, x, 0] = r / 255.0                                                  <L 592>
        var_31 = wp::div(var_21, var_30);
        wp::array_store(var_output_img, var_0, var_9, var_1, var_14, var_31);
        // output_img[tile, y, x, 1] = g / 255.0                                                  <L 593>
        var_32 = wp::div(var_25, var_30);
        wp::array_store(var_output_img, var_0, var_9, var_1, var_8, var_32);
        // output_img[tile, y, x, 2] = b / 255.0                                                  <L 594>
        var_33 = wp::div(var_29, var_30);
        wp::array_store(var_output_img, var_0, var_9, var_1, var_15, var_33);
    }
}

extern "C" __global__ void copy_rgb_frame_tile_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::uint8> var_input_img,
    wp::int32 var_offset_x,
    wp::int32 var_offset_y,
    wp::int32 var_screen_width,
    wp::int32 var_screen_height,
    wp::int32 var_tile_height,
    wp::array_t<wp::float32> var_output_img,
    wp::array_t<wp::uint8> adj_input_img,
    wp::int32 adj_offset_x,
    wp::int32 adj_offset_y,
    wp::int32 adj_screen_width,
    wp::int32 adj_screen_height,
    wp::int32 adj_tile_height,
    wp::array_t<wp::float32> adj_output_img)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
    }
}

