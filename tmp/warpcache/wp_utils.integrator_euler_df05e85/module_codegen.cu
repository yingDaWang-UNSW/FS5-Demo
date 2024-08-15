
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



// G:\My Drive\sourceCodes\fs5ydw-main\utils\integrator_euler.py:1085
static CUDA_CALLABLE wp::vec_t<3,wp::float32> eval_joint_force(
    wp::float32 var_q,
    wp::float32 var_qd,
    wp::float32 var_target,
    wp::float32 var_target_ke,
    wp::float32 var_target_kd,
    wp::float32 var_act,
    wp::float32 var_limit_lower,
    wp::float32 var_limit_upper,
    wp::float32 var_limit_ke,
    wp::float32 var_limit_kd,
    wp::vec_t<3,wp::float32> var_axis)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    bool var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    bool var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    wp::vec_t<3,wp::float32> var_21;
    //---------
    // forward
    // def eval_joint_force(                                                                  <L 1086>
    // limit_f = 0.0                                                                          <L 1099>
    // if q < limit_lower:                                                                    <L 1102>
    var_1 = (var_q < var_limit_lower);
    if (var_1) {
        // limit_f = limit_ke * (limit_lower - q) - limit_kd * min(qd, 0.0)                   <L 1103>
        var_2 = wp::sub(var_limit_lower, var_q);
        var_3 = wp::mul(var_limit_ke, var_2);
        var_4 = wp::min(var_qd, var_0);
        var_5 = wp::mul(var_limit_kd, var_4);
        var_6 = wp::sub(var_3, var_5);
    }
    var_7 = wp::select(var_1, var_0, var_6);
    // if q > limit_upper:                                                                    <L 1105>
    var_8 = (var_q > var_limit_upper);
    if (var_8) {
        // limit_f = limit_ke * (limit_upper - q) - limit_kd * max(qd, 0.0)                   <L 1106>
        var_9 = wp::sub(var_limit_upper, var_q);
        var_10 = wp::mul(var_limit_ke, var_9);
        var_11 = wp::max(var_qd, var_0);
        var_12 = wp::mul(var_limit_kd, var_11);
        var_13 = wp::sub(var_10, var_12);
    }
    var_14 = wp::select(var_8, var_7, var_13);
    // total_f = (target_ke * (q - target) + target_kd * qd + act - limit_f) * axis           <L 1109>
    var_15 = wp::sub(var_q, var_target);
    var_16 = wp::mul(var_target_ke, var_15);
    var_17 = wp::mul(var_target_kd, var_qd);
    var_18 = wp::add(var_16, var_17);
    var_19 = wp::add(var_18, var_act);
    var_20 = wp::sub(var_19, var_14);
    var_21 = wp::mul(var_20, var_axis);
    // return total_f                                                                         <L 1111>
    return var_21;
}


// G:\My Drive\sourceCodes\fs5ydw-main\utils\integrator_euler.py:1085
static CUDA_CALLABLE void adj_eval_joint_force(
    wp::float32 var_q,
    wp::float32 var_qd,
    wp::float32 var_target,
    wp::float32 var_target_ke,
    wp::float32 var_target_kd,
    wp::float32 var_act,
    wp::float32 var_limit_lower,
    wp::float32 var_limit_upper,
    wp::float32 var_limit_ke,
    wp::float32 var_limit_kd,
    wp::vec_t<3,wp::float32> var_axis,
    wp::float32 & adj_q,
    wp::float32 & adj_qd,
    wp::float32 & adj_target,
    wp::float32 & adj_target_ke,
    wp::float32 & adj_target_kd,
    wp::float32 & adj_act,
    wp::float32 & adj_limit_lower,
    wp::float32 & adj_limit_upper,
    wp::float32 & adj_limit_ke,
    wp::float32 & adj_limit_kd,
    wp::vec_t<3,wp::float32> & adj_axis,
    wp::vec_t<3,wp::float32> & adj_ret)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    bool var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    bool var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    wp::vec_t<3,wp::float32> var_21;
    //---------
    // dual vars
    wp::float32 adj_0 = {};
    bool adj_1 = {};
    wp::float32 adj_2 = {};
    wp::float32 adj_3 = {};
    wp::float32 adj_4 = {};
    wp::float32 adj_5 = {};
    wp::float32 adj_6 = {};
    wp::float32 adj_7 = {};
    bool adj_8 = {};
    wp::float32 adj_9 = {};
    wp::float32 adj_10 = {};
    wp::float32 adj_11 = {};
    wp::float32 adj_12 = {};
    wp::float32 adj_13 = {};
    wp::float32 adj_14 = {};
    wp::float32 adj_15 = {};
    wp::float32 adj_16 = {};
    wp::float32 adj_17 = {};
    wp::float32 adj_18 = {};
    wp::float32 adj_19 = {};
    wp::float32 adj_20 = {};
    wp::vec_t<3,wp::float32> adj_21 = {};
    //---------
    // forward
    // def eval_joint_force(                                                                  <L 1086>
    // limit_f = 0.0                                                                          <L 1099>
    // if q < limit_lower:                                                                    <L 1102>
    var_1 = (var_q < var_limit_lower);
    if (var_1) {
        // limit_f = limit_ke * (limit_lower - q) - limit_kd * min(qd, 0.0)                   <L 1103>
        var_2 = wp::sub(var_limit_lower, var_q);
        var_3 = wp::mul(var_limit_ke, var_2);
        var_4 = wp::min(var_qd, var_0);
        var_5 = wp::mul(var_limit_kd, var_4);
        var_6 = wp::sub(var_3, var_5);
    }
    var_7 = wp::select(var_1, var_0, var_6);
    // if q > limit_upper:                                                                    <L 1105>
    var_8 = (var_q > var_limit_upper);
    if (var_8) {
        // limit_f = limit_ke * (limit_upper - q) - limit_kd * max(qd, 0.0)                   <L 1106>
        var_9 = wp::sub(var_limit_upper, var_q);
        var_10 = wp::mul(var_limit_ke, var_9);
        var_11 = wp::max(var_qd, var_0);
        var_12 = wp::mul(var_limit_kd, var_11);
        var_13 = wp::sub(var_10, var_12);
    }
    var_14 = wp::select(var_8, var_7, var_13);
    // total_f = (target_ke * (q - target) + target_kd * qd + act - limit_f) * axis           <L 1109>
    var_15 = wp::sub(var_q, var_target);
    var_16 = wp::mul(var_target_ke, var_15);
    var_17 = wp::mul(var_target_kd, var_qd);
    var_18 = wp::add(var_16, var_17);
    var_19 = wp::add(var_18, var_act);
    var_20 = wp::sub(var_19, var_14);
    var_21 = wp::mul(var_20, var_axis);
    // return total_f                                                                         <L 1111>
    goto label0;
    //---------
    // reverse
    label0:;
    adj_21 += adj_ret;
    // adj: return total_f                                                                    <L 1111>
    wp::adj_mul(var_20, var_axis, adj_20, adj_axis, adj_21);
    wp::adj_sub(var_19, var_14, adj_19, adj_14, adj_20);
    wp::adj_add(var_18, var_act, adj_18, adj_act, adj_19);
    wp::adj_add(var_16, var_17, adj_16, adj_17, adj_18);
    wp::adj_mul(var_target_kd, var_qd, adj_target_kd, adj_qd, adj_17);
    wp::adj_mul(var_target_ke, var_15, adj_target_ke, adj_15, adj_16);
    wp::adj_sub(var_q, var_target, adj_q, adj_target, adj_15);
    // adj: total_f = (target_ke * (q - target) + target_kd * qd + act - limit_f) * axis      <L 1109>
    wp::adj_select(var_8, var_7, var_13, adj_8, adj_7, adj_13, adj_14);
    if (var_8) {
        wp::adj_sub(var_10, var_12, adj_10, adj_12, adj_13);
        wp::adj_mul(var_limit_kd, var_11, adj_limit_kd, adj_11, adj_12);
        wp::adj_max(var_qd, var_0, adj_qd, adj_0, adj_11);
        wp::adj_mul(var_limit_ke, var_9, adj_limit_ke, adj_9, adj_10);
        wp::adj_sub(var_limit_upper, var_q, adj_limit_upper, adj_q, adj_9);
        // adj: limit_f = limit_ke * (limit_upper - q) - limit_kd * max(qd, 0.0)              <L 1106>
    }
    // adj: if q > limit_upper:                                                               <L 1105>
    wp::adj_select(var_1, var_0, var_6, adj_1, adj_0, adj_6, adj_7);
    if (var_1) {
        wp::adj_sub(var_3, var_5, adj_3, adj_5, adj_6);
        wp::adj_mul(var_limit_kd, var_4, adj_limit_kd, adj_4, adj_5);
        wp::adj_min(var_qd, var_0, adj_qd, adj_0, adj_4);
        wp::adj_mul(var_limit_ke, var_2, adj_limit_ke, adj_2, adj_3);
        wp::adj_sub(var_limit_lower, var_q, adj_limit_lower, adj_q, adj_2);
        // adj: limit_f = limit_ke * (limit_lower - q) - limit_kd * min(qd, 0.0)              <L 1103>
    }
    // adj: if q < limit_lower:                                                               <L 1102>
    // adj: limit_f = 0.0                                                                     <L 1099>
    // adj: def eval_joint_force(                                                             <L 1086>
    return;
}


// G:\My Drive\sourceCodes\fs5ydw-main\utils\integrator_euler.py:1429
static CUDA_CALLABLE wp::int32 compute_muscle_force(
    wp::int32 var_i,
    wp::array_t<wp::transform_t<wp::float32>> var_body_X_s,
    wp::array_t<wp::vec_t<6,wp::float32>> var_body_v_s,
    wp::array_t<wp::vec_t<3,wp::float32>> var_body_com,
    wp::array_t<wp::int32> var_muscle_links,
    wp::array_t<wp::vec_t<3,wp::float32>> var_muscle_points,
    wp::float32 var_muscle_activation,
    wp::array_t<wp::vec_t<6,wp::float32>> var_body_f_s)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    const wp::int32 var_3 = 1;
    wp::int32 var_4;
    wp::int32* var_5;
    wp::int32 var_6;
    wp::int32 var_7;
    bool var_8;
    const wp::int32 var_9 = 0;
    wp::vec_t<3,wp::float32>* var_10;
    wp::vec_t<3,wp::float32> var_11;
    wp::vec_t<3,wp::float32> var_12;
    wp::int32 var_13;
    wp::vec_t<3,wp::float32>* var_14;
    wp::vec_t<3,wp::float32> var_15;
    wp::vec_t<3,wp::float32> var_16;
    wp::transform_t<wp::float32>* var_17;
    wp::transform_t<wp::float32> var_18;
    wp::transform_t<wp::float32> var_19;
    wp::transform_t<wp::float32>* var_20;
    wp::transform_t<wp::float32> var_21;
    wp::transform_t<wp::float32> var_22;
    wp::vec_t<3,wp::float32>* var_23;
    wp::vec_t<3,wp::float32> var_24;
    wp::vec_t<3,wp::float32> var_25;
    wp::vec_t<3,wp::float32> var_26;
    wp::vec_t<3,wp::float32>* var_27;
    wp::vec_t<3,wp::float32> var_28;
    wp::vec_t<3,wp::float32> var_29;
    wp::vec_t<3,wp::float32> var_30;
    wp::vec_t<3,wp::float32> var_31;
    wp::vec_t<3,wp::float32> var_32;
    wp::vec_t<3,wp::float32> var_33;
    wp::vec_t<3,wp::float32> var_34;
    wp::vec_t<6,wp::float32> var_35;
    wp::vec_t<6,wp::float32> var_36;
    wp::vec_t<3,wp::float32> var_37;
    wp::vec_t<6,wp::float32> var_38;
    wp::vec_t<6,wp::float32> var_39;
    //---------
    // forward
    // def compute_muscle_force(                                                              <L 1430>
    // link_0 = muscle_links[i]                                                               <L 1440>
    var_0 = wp::address(var_muscle_links, var_i);
    var_1 = wp::load(var_0);
    var_2 = wp::copy(var_1);
    // link_1 = muscle_links[i + 1]                                                           <L 1441>
    var_4 = wp::add(var_i, var_3);
    var_5 = wp::address(var_muscle_links, var_4);
    var_6 = wp::load(var_5);
    var_7 = wp::copy(var_6);
    // if link_0 == link_1:                                                                   <L 1443>
    var_8 = (var_2 == var_7);
    if (var_8) {
        // return 0                                                                           <L 1444>
        return var_9;
    }
    // r_0 = muscle_points[i]                                                                 <L 1446>
    var_10 = wp::address(var_muscle_points, var_i);
    var_11 = wp::load(var_10);
    var_12 = wp::copy(var_11);
    // r_1 = muscle_points[i + 1]                                                             <L 1447>
    var_13 = wp::add(var_i, var_3);
    var_14 = wp::address(var_muscle_points, var_13);
    var_15 = wp::load(var_14);
    var_16 = wp::copy(var_15);
    // xform_0 = body_X_s[link_0]                                                             <L 1449>
    var_17 = wp::address(var_body_X_s, var_2);
    var_18 = wp::load(var_17);
    var_19 = wp::copy(var_18);
    // xform_1 = body_X_s[link_1]                                                             <L 1450>
    var_20 = wp::address(var_body_X_s, var_7);
    var_21 = wp::load(var_20);
    var_22 = wp::copy(var_21);
    // pos_0 = wp.transform_point(xform_0, r_0 - body_com[link_0])                            <L 1452>
    var_23 = wp::address(var_body_com, var_2);
    var_24 = wp::load(var_23);
    var_25 = wp::sub(var_12, var_24);
    var_26 = wp::transform_point(var_19, var_25);
    // pos_1 = wp.transform_point(xform_1, r_1 - body_com[link_1])                            <L 1453>
    var_27 = wp::address(var_body_com, var_7);
    var_28 = wp::load(var_27);
    var_29 = wp::sub(var_16, var_28);
    var_30 = wp::transform_point(var_22, var_29);
    // n = wp.normalize(pos_1 - pos_0)                                                        <L 1455>
    var_31 = wp::sub(var_30, var_26);
    var_32 = wp::normalize(var_31);
    // f = n * muscle_activation                                                              <L 1458>
    var_33 = wp::mul(var_32, var_muscle_activation);
    // wp.atomic_sub(body_f_s, link_0, wp.spatial_vector(f, wp.cross(pos_0, f)))              <L 1460>
    var_34 = wp::cross(var_26, var_33);
    var_35 = wp::vec_t<6,wp::float32>(var_33, var_34);
    var_36 = wp::atomic_sub(var_body_f_s, var_2, var_35);
    // wp.atomic_add(body_f_s, link_1, wp.spatial_vector(f, wp.cross(pos_1, f)))              <L 1461>
    var_37 = wp::cross(var_30, var_33);
    var_38 = wp::vec_t<6,wp::float32>(var_33, var_37);
    var_39 = wp::atomic_add(var_body_f_s, var_7, var_38);
    // return 0                                                                               <L 1463>
    return var_9;
}


// G:\My Drive\sourceCodes\fs5ydw-main\utils\integrator_euler.py:1429
static CUDA_CALLABLE void adj_compute_muscle_force(
    wp::int32 var_i,
    wp::array_t<wp::transform_t<wp::float32>> var_body_X_s,
    wp::array_t<wp::vec_t<6,wp::float32>> var_body_v_s,
    wp::array_t<wp::vec_t<3,wp::float32>> var_body_com,
    wp::array_t<wp::int32> var_muscle_links,
    wp::array_t<wp::vec_t<3,wp::float32>> var_muscle_points,
    wp::float32 var_muscle_activation,
    wp::array_t<wp::vec_t<6,wp::float32>> var_body_f_s,
    wp::int32 & adj_i,
    wp::array_t<wp::transform_t<wp::float32>> & adj_body_X_s,
    wp::array_t<wp::vec_t<6,wp::float32>> & adj_body_v_s,
    wp::array_t<wp::vec_t<3,wp::float32>> & adj_body_com,
    wp::array_t<wp::int32> & adj_muscle_links,
    wp::array_t<wp::vec_t<3,wp::float32>> & adj_muscle_points,
    wp::float32 & adj_muscle_activation,
    wp::array_t<wp::vec_t<6,wp::float32>> & adj_body_f_s,
    wp::int32 & adj_ret)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    const wp::int32 var_3 = 1;
    wp::int32 var_4;
    wp::int32* var_5;
    wp::int32 var_6;
    wp::int32 var_7;
    bool var_8;
    const wp::int32 var_9 = 0;
    wp::vec_t<3,wp::float32>* var_10;
    wp::vec_t<3,wp::float32> var_11;
    wp::vec_t<3,wp::float32> var_12;
    wp::int32 var_13;
    wp::vec_t<3,wp::float32>* var_14;
    wp::vec_t<3,wp::float32> var_15;
    wp::vec_t<3,wp::float32> var_16;
    wp::transform_t<wp::float32>* var_17;
    wp::transform_t<wp::float32> var_18;
    wp::transform_t<wp::float32> var_19;
    wp::transform_t<wp::float32>* var_20;
    wp::transform_t<wp::float32> var_21;
    wp::transform_t<wp::float32> var_22;
    wp::vec_t<3,wp::float32>* var_23;
    wp::vec_t<3,wp::float32> var_24;
    wp::vec_t<3,wp::float32> var_25;
    wp::vec_t<3,wp::float32> var_26;
    wp::vec_t<3,wp::float32>* var_27;
    wp::vec_t<3,wp::float32> var_28;
    wp::vec_t<3,wp::float32> var_29;
    wp::vec_t<3,wp::float32> var_30;
    wp::vec_t<3,wp::float32> var_31;
    wp::vec_t<3,wp::float32> var_32;
    wp::vec_t<3,wp::float32> var_33;
    wp::vec_t<3,wp::float32> var_34;
    wp::vec_t<6,wp::float32> var_35;
    wp::vec_t<6,wp::float32> var_36;
    wp::vec_t<3,wp::float32> var_37;
    wp::vec_t<6,wp::float32> var_38;
    wp::vec_t<6,wp::float32> var_39;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::int32 adj_1 = {};
    wp::int32 adj_2 = {};
    wp::int32 adj_3 = {};
    wp::int32 adj_4 = {};
    wp::int32 adj_5 = {};
    wp::int32 adj_6 = {};
    wp::int32 adj_7 = {};
    bool adj_8 = {};
    wp::int32 adj_9 = {};
    wp::vec_t<3,wp::float32> adj_10 = {};
    wp::vec_t<3,wp::float32> adj_11 = {};
    wp::vec_t<3,wp::float32> adj_12 = {};
    wp::int32 adj_13 = {};
    wp::vec_t<3,wp::float32> adj_14 = {};
    wp::vec_t<3,wp::float32> adj_15 = {};
    wp::vec_t<3,wp::float32> adj_16 = {};
    wp::transform_t<wp::float32> adj_17 = {};
    wp::transform_t<wp::float32> adj_18 = {};
    wp::transform_t<wp::float32> adj_19 = {};
    wp::transform_t<wp::float32> adj_20 = {};
    wp::transform_t<wp::float32> adj_21 = {};
    wp::transform_t<wp::float32> adj_22 = {};
    wp::vec_t<3,wp::float32> adj_23 = {};
    wp::vec_t<3,wp::float32> adj_24 = {};
    wp::vec_t<3,wp::float32> adj_25 = {};
    wp::vec_t<3,wp::float32> adj_26 = {};
    wp::vec_t<3,wp::float32> adj_27 = {};
    wp::vec_t<3,wp::float32> adj_28 = {};
    wp::vec_t<3,wp::float32> adj_29 = {};
    wp::vec_t<3,wp::float32> adj_30 = {};
    wp::vec_t<3,wp::float32> adj_31 = {};
    wp::vec_t<3,wp::float32> adj_32 = {};
    wp::vec_t<3,wp::float32> adj_33 = {};
    wp::vec_t<3,wp::float32> adj_34 = {};
    wp::vec_t<6,wp::float32> adj_35 = {};
    wp::vec_t<6,wp::float32> adj_36 = {};
    wp::vec_t<3,wp::float32> adj_37 = {};
    wp::vec_t<6,wp::float32> adj_38 = {};
    wp::vec_t<6,wp::float32> adj_39 = {};
    //---------
    // forward
    // def compute_muscle_force(                                                              <L 1430>
    // link_0 = muscle_links[i]                                                               <L 1440>
    var_0 = wp::address(var_muscle_links, var_i);
    var_1 = wp::load(var_0);
    var_2 = wp::copy(var_1);
    // link_1 = muscle_links[i + 1]                                                           <L 1441>
    var_4 = wp::add(var_i, var_3);
    var_5 = wp::address(var_muscle_links, var_4);
    var_6 = wp::load(var_5);
    var_7 = wp::copy(var_6);
    // if link_0 == link_1:                                                                   <L 1443>
    var_8 = (var_2 == var_7);
    if (var_8) {
        // return 0                                                                           <L 1444>
        goto label0;
    }
    // r_0 = muscle_points[i]                                                                 <L 1446>
    var_10 = wp::address(var_muscle_points, var_i);
    var_11 = wp::load(var_10);
    var_12 = wp::copy(var_11);
    // r_1 = muscle_points[i + 1]                                                             <L 1447>
    var_13 = wp::add(var_i, var_3);
    var_14 = wp::address(var_muscle_points, var_13);
    var_15 = wp::load(var_14);
    var_16 = wp::copy(var_15);
    // xform_0 = body_X_s[link_0]                                                             <L 1449>
    var_17 = wp::address(var_body_X_s, var_2);
    var_18 = wp::load(var_17);
    var_19 = wp::copy(var_18);
    // xform_1 = body_X_s[link_1]                                                             <L 1450>
    var_20 = wp::address(var_body_X_s, var_7);
    var_21 = wp::load(var_20);
    var_22 = wp::copy(var_21);
    // pos_0 = wp.transform_point(xform_0, r_0 - body_com[link_0])                            <L 1452>
    var_23 = wp::address(var_body_com, var_2);
    var_24 = wp::load(var_23);
    var_25 = wp::sub(var_12, var_24);
    var_26 = wp::transform_point(var_19, var_25);
    // pos_1 = wp.transform_point(xform_1, r_1 - body_com[link_1])                            <L 1453>
    var_27 = wp::address(var_body_com, var_7);
    var_28 = wp::load(var_27);
    var_29 = wp::sub(var_16, var_28);
    var_30 = wp::transform_point(var_22, var_29);
    // n = wp.normalize(pos_1 - pos_0)                                                        <L 1455>
    var_31 = wp::sub(var_30, var_26);
    var_32 = wp::normalize(var_31);
    // f = n * muscle_activation                                                              <L 1458>
    var_33 = wp::mul(var_32, var_muscle_activation);
    // wp.atomic_sub(body_f_s, link_0, wp.spatial_vector(f, wp.cross(pos_0, f)))              <L 1460>
    var_34 = wp::cross(var_26, var_33);
    var_35 = wp::vec_t<6,wp::float32>(var_33, var_34);
    // var_36 = wp::atomic_sub(var_body_f_s, var_2, var_35);
    // wp.atomic_add(body_f_s, link_1, wp.spatial_vector(f, wp.cross(pos_1, f)))              <L 1461>
    var_37 = wp::cross(var_30, var_33);
    var_38 = wp::vec_t<6,wp::float32>(var_33, var_37);
    // var_39 = wp::atomic_add(var_body_f_s, var_7, var_38);
    // return 0                                                                               <L 1463>
    goto label1;
    //---------
    // reverse
    label1:;
    adj_9 += adj_ret;
    // adj: return 0                                                                          <L 1463>
    wp::adj_atomic_add(var_body_f_s, var_7, var_38, adj_body_f_s, adj_7, adj_38, adj_39);
    wp::adj_vec_t(var_33, var_37, adj_33, adj_37, adj_38);
    wp::adj_cross(var_30, var_33, adj_30, adj_33, adj_37);
    // adj: wp.atomic_add(body_f_s, link_1, wp.spatial_vector(f, wp.cross(pos_1, f)))         <L 1461>
    wp::adj_atomic_sub(var_body_f_s, var_2, var_35, adj_body_f_s, adj_2, adj_35, adj_36);
    wp::adj_vec_t(var_33, var_34, adj_33, adj_34, adj_35);
    wp::adj_cross(var_26, var_33, adj_26, adj_33, adj_34);
    // adj: wp.atomic_sub(body_f_s, link_0, wp.spatial_vector(f, wp.cross(pos_0, f)))         <L 1460>
    wp::adj_mul(var_32, var_muscle_activation, adj_32, adj_muscle_activation, adj_33);
    // adj: f = n * muscle_activation                                                         <L 1458>
    wp::adj_normalize(var_31, var_32, adj_31, adj_32);
    wp::adj_sub(var_30, var_26, adj_30, adj_26, adj_31);
    // adj: n = wp.normalize(pos_1 - pos_0)                                                   <L 1455>
    wp::adj_transform_point(var_22, var_29, adj_22, adj_29, adj_30);
    wp::adj_sub(var_16, var_28, adj_16, adj_27, adj_29);
    wp::adj_load(var_27, adj_27, adj_28);
    wp::adj_address(var_body_com, var_7, adj_body_com, adj_7, adj_27);
    // adj: pos_1 = wp.transform_point(xform_1, r_1 - body_com[link_1])                       <L 1453>
    wp::adj_transform_point(var_19, var_25, adj_19, adj_25, adj_26);
    wp::adj_sub(var_12, var_24, adj_12, adj_23, adj_25);
    wp::adj_load(var_23, adj_23, adj_24);
    wp::adj_address(var_body_com, var_2, adj_body_com, adj_2, adj_23);
    // adj: pos_0 = wp.transform_point(xform_0, r_0 - body_com[link_0])                       <L 1452>
    wp::adj_copy(var_21, adj_20, adj_22);
    wp::adj_load(var_20, adj_20, adj_21);
    wp::adj_address(var_body_X_s, var_7, adj_body_X_s, adj_7, adj_20);
    // adj: xform_1 = body_X_s[link_1]                                                        <L 1450>
    wp::adj_copy(var_18, adj_17, adj_19);
    wp::adj_load(var_17, adj_17, adj_18);
    wp::adj_address(var_body_X_s, var_2, adj_body_X_s, adj_2, adj_17);
    // adj: xform_0 = body_X_s[link_0]                                                        <L 1449>
    wp::adj_copy(var_15, adj_14, adj_16);
    wp::adj_load(var_14, adj_14, adj_15);
    wp::adj_address(var_muscle_points, var_13, adj_muscle_points, adj_13, adj_14);
    wp::adj_add(var_i, var_3, adj_i, adj_3, adj_13);
    // adj: r_1 = muscle_points[i + 1]                                                        <L 1447>
    wp::adj_copy(var_11, adj_10, adj_12);
    wp::adj_load(var_10, adj_10, adj_11);
    wp::adj_address(var_muscle_points, var_i, adj_muscle_points, adj_i, adj_10);
    // adj: r_0 = muscle_points[i]                                                            <L 1446>
    if (var_8) {
        label0:;
        adj_9 += adj_ret;
        // adj: return 0                                                                      <L 1444>
    }
    // adj: if link_0 == link_1:                                                              <L 1443>
    wp::adj_copy(var_6, adj_5, adj_7);
    wp::adj_load(var_5, adj_5, adj_6);
    wp::adj_address(var_muscle_links, var_4, adj_muscle_links, adj_4, adj_5);
    wp::adj_add(var_i, var_3, adj_i, adj_3, adj_4);
    // adj: link_1 = muscle_links[i + 1]                                                      <L 1441>
    wp::adj_copy(var_1, adj_0, adj_2);
    wp::adj_load(var_0, adj_0, adj_1);
    wp::adj_address(var_muscle_links, var_i, adj_muscle_links, adj_i, adj_0);
    // adj: link_0 = muscle_links[i]                                                          <L 1440>
    // adj: def compute_muscle_force(                                                         <L 1430>
    return;
}


// G:\My Drive\sourceCodes\fs5ydw-main\utils\collide.py:16
static CUDA_CALLABLE wp::vec_t<3,wp::float32> triangle_closest_point_barycentric(
    wp::vec_t<3,wp::float32> var_a,
    wp::vec_t<3,wp::float32> var_b,
    wp::vec_t<3,wp::float32> var_c,
    wp::vec_t<3,wp::float32> var_p)
{
    //---------
    // primal vars
    wp::vec_t<3,wp::float32> var_0;
    wp::vec_t<3,wp::float32> var_1;
    wp::vec_t<3,wp::float32> var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::float32 var_5 = 0.0;
    bool var_6;
    bool var_7;
    bool var_8;
    const wp::float32 var_9 = 1.0;
    wp::vec_t<3,wp::float32> var_10;
    wp::vec_t<3,wp::float32> var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    bool var_14;
    bool var_15;
    bool var_16;
    wp::vec_t<3,wp::float32> var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    bool var_23;
    bool var_24;
    bool var_25;
    bool var_26;
    wp::float32 var_27;
    wp::vec_t<3,wp::float32> var_28;
    wp::vec_t<3,wp::float32> var_29;
    wp::float32 var_30;
    wp::float32 var_31;
    bool var_32;
    bool var_33;
    bool var_34;
    wp::vec_t<3,wp::float32> var_35;
    wp::float32 var_36;
    wp::float32 var_37;
    wp::float32 var_38;
    wp::float32 var_39;
    wp::float32 var_40;
    bool var_41;
    bool var_42;
    bool var_43;
    bool var_44;
    wp::float32 var_45;
    wp::vec_t<3,wp::float32> var_46;
    wp::float32 var_47;
    wp::float32 var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    wp::float32 var_53;
    wp::float32 var_54;
    bool var_55;
    wp::float32 var_56;
    bool var_57;
    wp::float32 var_58;
    bool var_59;
    bool var_60;
    wp::float32 var_61;
    wp::vec_t<3,wp::float32> var_62;
    wp::float32 var_63;
    wp::float32 var_64;
    wp::float32 var_65;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    wp::float32 var_69;
    wp::vec_t<3,wp::float32> var_70;
    //---------
    // forward
    // def triangle_closest_point_barycentric(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3):       <L 17>
    // ab = b - a                                                                             <L 18>
    var_0 = wp::sub(var_b, var_a);
    // ac = c - a                                                                             <L 19>
    var_1 = wp::sub(var_c, var_a);
    // ap = p - a                                                                             <L 20>
    var_2 = wp::sub(var_p, var_a);
    // d1 = wp.dot(ab, ap)                                                                    <L 22>
    var_3 = wp::dot(var_0, var_2);
    // d2 = wp.dot(ac, ap)                                                                    <L 23>
    var_4 = wp::dot(var_1, var_2);
    // if d1 <= 0.0 and d2 <= 0.0:                                                            <L 25>
    var_6 = (var_3 <= var_5);
    var_7 = (var_4 <= var_5);
    var_8 = var_6 && var_7;
    if (var_8) {
        // return wp.vec3(1.0, 0.0, 0.0)                                                      <L 26>
        var_10 = wp::vec_t<3,wp::float32>(var_9, var_5, var_5);
        return var_10;
    }
    // bp = p - b                                                                             <L 28>
    var_11 = wp::sub(var_p, var_b);
    // d3 = wp.dot(ab, bp)                                                                    <L 29>
    var_12 = wp::dot(var_0, var_11);
    // d4 = wp.dot(ac, bp)                                                                    <L 30>
    var_13 = wp::dot(var_1, var_11);
    // if d3 >= 0.0 and d4 <= d3:                                                             <L 32>
    var_14 = (var_12 >= var_5);
    var_15 = (var_13 <= var_12);
    var_16 = var_14 && var_15;
    if (var_16) {
        // return wp.vec3(0.0, 1.0, 0.0)                                                      <L 33>
        var_17 = wp::vec_t<3,wp::float32>(var_5, var_9, var_5);
        return var_17;
    }
    // vc = d1 * d4 - d3 * d2                                                                 <L 35>
    var_18 = wp::mul(var_3, var_13);
    var_19 = wp::mul(var_12, var_4);
    var_20 = wp::sub(var_18, var_19);
    // v = d1 / (d1 - d3)                                                                     <L 36>
    var_21 = wp::sub(var_3, var_12);
    var_22 = wp::div(var_3, var_21);
    // if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:                                              <L 37>
    var_23 = (var_20 <= var_5);
    var_24 = (var_3 >= var_5);
    var_25 = (var_12 <= var_5);
    var_26 = var_23 && var_24 && var_25;
    if (var_26) {
        // return wp.vec3(1.0 - v, v, 0.0)                                                    <L 38>
        var_27 = wp::sub(var_9, var_22);
        var_28 = wp::vec_t<3,wp::float32>(var_27, var_22, var_5);
        return var_28;
    }
    // cp = p - c                                                                             <L 40>
    var_29 = wp::sub(var_p, var_c);
    // d5 = wp.dot(ab, cp)                                                                    <L 41>
    var_30 = wp::dot(var_0, var_29);
    // d6 = wp.dot(ac, cp)                                                                    <L 42>
    var_31 = wp::dot(var_1, var_29);
    // if d6 >= 0.0 and d5 <= d6:                                                             <L 44>
    var_32 = (var_31 >= var_5);
    var_33 = (var_30 <= var_31);
    var_34 = var_32 && var_33;
    if (var_34) {
        // return wp.vec3(0.0, 0.0, 1.0)                                                      <L 45>
        var_35 = wp::vec_t<3,wp::float32>(var_5, var_5, var_9);
        return var_35;
    }
    // vb = d5 * d2 - d1 * d6                                                                 <L 47>
    var_36 = wp::mul(var_30, var_4);
    var_37 = wp::mul(var_3, var_31);
    var_38 = wp::sub(var_36, var_37);
    // w = d2 / (d2 - d6)                                                                     <L 48>
    var_39 = wp::sub(var_4, var_31);
    var_40 = wp::div(var_4, var_39);
    // if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:                                              <L 49>
    var_41 = (var_38 <= var_5);
    var_42 = (var_4 >= var_5);
    var_43 = (var_31 <= var_5);
    var_44 = var_41 && var_42 && var_43;
    if (var_44) {
        // return wp.vec3(1.0 - w, 0.0, w)                                                    <L 50>
        var_45 = wp::sub(var_9, var_40);
        var_46 = wp::vec_t<3,wp::float32>(var_45, var_5, var_40);
        return var_46;
    }
    // va = d3 * d6 - d5 * d4                                                                 <L 52>
    var_47 = wp::mul(var_12, var_31);
    var_48 = wp::mul(var_30, var_13);
    var_49 = wp::sub(var_47, var_48);
    // w = (d4 - d3) / ((d4 - d3) + (d5 - d6))                                                <L 53>
    var_50 = wp::sub(var_13, var_12);
    var_51 = wp::sub(var_13, var_12);
    var_52 = wp::sub(var_30, var_31);
    var_53 = wp::add(var_51, var_52);
    var_54 = wp::div(var_50, var_53);
    // if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:                                <L 54>
    var_55 = (var_49 <= var_5);
    var_56 = wp::sub(var_13, var_12);
    var_57 = (var_56 >= var_5);
    var_58 = wp::sub(var_30, var_31);
    var_59 = (var_58 >= var_5);
    var_60 = var_55 && var_57 && var_59;
    if (var_60) {
        // return wp.vec3(0.0, w, 1.0 - w)                                                    <L 55>
        var_61 = wp::sub(var_9, var_54);
        var_62 = wp::vec_t<3,wp::float32>(var_5, var_54, var_61);
        return var_62;
    }
    // denom = 1.0 / (va + vb + vc)                                                           <L 57>
    var_63 = wp::add(var_49, var_38);
    var_64 = wp::add(var_63, var_20);
    var_65 = wp::div(var_9, var_64);
    // v = vb * denom                                                                         <L 58>
    var_66 = wp::mul(var_38, var_65);
    // w = vc * denom                                                                         <L 59>
    var_67 = wp::mul(var_20, var_65);
    // return wp.vec3(1.0 - v - w, v, w)                                                      <L 61>
    var_68 = wp::sub(var_9, var_66);
    var_69 = wp::sub(var_68, var_67);
    var_70 = wp::vec_t<3,wp::float32>(var_69, var_66, var_67);
    return var_70;
}


// G:\My Drive\sourceCodes\fs5ydw-main\utils\collide.py:16
static CUDA_CALLABLE void adj_triangle_closest_point_barycentric(
    wp::vec_t<3,wp::float32> var_a,
    wp::vec_t<3,wp::float32> var_b,
    wp::vec_t<3,wp::float32> var_c,
    wp::vec_t<3,wp::float32> var_p,
    wp::vec_t<3,wp::float32> & adj_a,
    wp::vec_t<3,wp::float32> & adj_b,
    wp::vec_t<3,wp::float32> & adj_c,
    wp::vec_t<3,wp::float32> & adj_p,
    wp::vec_t<3,wp::float32> & adj_ret)
{
    //---------
    // primal vars
    wp::vec_t<3,wp::float32> var_0;
    wp::vec_t<3,wp::float32> var_1;
    wp::vec_t<3,wp::float32> var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::float32 var_5 = 0.0;
    bool var_6;
    bool var_7;
    bool var_8;
    const wp::float32 var_9 = 1.0;
    wp::vec_t<3,wp::float32> var_10;
    wp::vec_t<3,wp::float32> var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    bool var_14;
    bool var_15;
    bool var_16;
    wp::vec_t<3,wp::float32> var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    bool var_23;
    bool var_24;
    bool var_25;
    bool var_26;
    wp::float32 var_27;
    wp::vec_t<3,wp::float32> var_28;
    wp::vec_t<3,wp::float32> var_29;
    wp::float32 var_30;
    wp::float32 var_31;
    bool var_32;
    bool var_33;
    bool var_34;
    wp::vec_t<3,wp::float32> var_35;
    wp::float32 var_36;
    wp::float32 var_37;
    wp::float32 var_38;
    wp::float32 var_39;
    wp::float32 var_40;
    bool var_41;
    bool var_42;
    bool var_43;
    bool var_44;
    wp::float32 var_45;
    wp::vec_t<3,wp::float32> var_46;
    wp::float32 var_47;
    wp::float32 var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    wp::float32 var_53;
    wp::float32 var_54;
    bool var_55;
    wp::float32 var_56;
    bool var_57;
    wp::float32 var_58;
    bool var_59;
    bool var_60;
    wp::float32 var_61;
    wp::vec_t<3,wp::float32> var_62;
    wp::float32 var_63;
    wp::float32 var_64;
    wp::float32 var_65;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    wp::float32 var_69;
    wp::vec_t<3,wp::float32> var_70;
    //---------
    // dual vars
    wp::vec_t<3,wp::float32> adj_0 = {};
    wp::vec_t<3,wp::float32> adj_1 = {};
    wp::vec_t<3,wp::float32> adj_2 = {};
    wp::float32 adj_3 = {};
    wp::float32 adj_4 = {};
    wp::float32 adj_5 = {};
    bool adj_6 = {};
    bool adj_7 = {};
    bool adj_8 = {};
    wp::float32 adj_9 = {};
    wp::vec_t<3,wp::float32> adj_10 = {};
    wp::vec_t<3,wp::float32> adj_11 = {};
    wp::float32 adj_12 = {};
    wp::float32 adj_13 = {};
    bool adj_14 = {};
    bool adj_15 = {};
    bool adj_16 = {};
    wp::vec_t<3,wp::float32> adj_17 = {};
    wp::float32 adj_18 = {};
    wp::float32 adj_19 = {};
    wp::float32 adj_20 = {};
    wp::float32 adj_21 = {};
    wp::float32 adj_22 = {};
    bool adj_23 = {};
    bool adj_24 = {};
    bool adj_25 = {};
    bool adj_26 = {};
    wp::float32 adj_27 = {};
    wp::vec_t<3,wp::float32> adj_28 = {};
    wp::vec_t<3,wp::float32> adj_29 = {};
    wp::float32 adj_30 = {};
    wp::float32 adj_31 = {};
    bool adj_32 = {};
    bool adj_33 = {};
    bool adj_34 = {};
    wp::vec_t<3,wp::float32> adj_35 = {};
    wp::float32 adj_36 = {};
    wp::float32 adj_37 = {};
    wp::float32 adj_38 = {};
    wp::float32 adj_39 = {};
    wp::float32 adj_40 = {};
    bool adj_41 = {};
    bool adj_42 = {};
    bool adj_43 = {};
    bool adj_44 = {};
    wp::float32 adj_45 = {};
    wp::vec_t<3,wp::float32> adj_46 = {};
    wp::float32 adj_47 = {};
    wp::float32 adj_48 = {};
    wp::float32 adj_49 = {};
    wp::float32 adj_50 = {};
    wp::float32 adj_51 = {};
    wp::float32 adj_52 = {};
    wp::float32 adj_53 = {};
    wp::float32 adj_54 = {};
    bool adj_55 = {};
    wp::float32 adj_56 = {};
    bool adj_57 = {};
    wp::float32 adj_58 = {};
    bool adj_59 = {};
    bool adj_60 = {};
    wp::float32 adj_61 = {};
    wp::vec_t<3,wp::float32> adj_62 = {};
    wp::float32 adj_63 = {};
    wp::float32 adj_64 = {};
    wp::float32 adj_65 = {};
    wp::float32 adj_66 = {};
    wp::float32 adj_67 = {};
    wp::float32 adj_68 = {};
    wp::float32 adj_69 = {};
    wp::vec_t<3,wp::float32> adj_70 = {};
    //---------
    // forward
    // def triangle_closest_point_barycentric(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3):       <L 17>
    // ab = b - a                                                                             <L 18>
    var_0 = wp::sub(var_b, var_a);
    // ac = c - a                                                                             <L 19>
    var_1 = wp::sub(var_c, var_a);
    // ap = p - a                                                                             <L 20>
    var_2 = wp::sub(var_p, var_a);
    // d1 = wp.dot(ab, ap)                                                                    <L 22>
    var_3 = wp::dot(var_0, var_2);
    // d2 = wp.dot(ac, ap)                                                                    <L 23>
    var_4 = wp::dot(var_1, var_2);
    // if d1 <= 0.0 and d2 <= 0.0:                                                            <L 25>
    var_6 = (var_3 <= var_5);
    var_7 = (var_4 <= var_5);
    var_8 = var_6 && var_7;
    if (var_8) {
        // return wp.vec3(1.0, 0.0, 0.0)                                                      <L 26>
        var_10 = wp::vec_t<3,wp::float32>(var_9, var_5, var_5);
        goto label0;
    }
    // bp = p - b                                                                             <L 28>
    var_11 = wp::sub(var_p, var_b);
    // d3 = wp.dot(ab, bp)                                                                    <L 29>
    var_12 = wp::dot(var_0, var_11);
    // d4 = wp.dot(ac, bp)                                                                    <L 30>
    var_13 = wp::dot(var_1, var_11);
    // if d3 >= 0.0 and d4 <= d3:                                                             <L 32>
    var_14 = (var_12 >= var_5);
    var_15 = (var_13 <= var_12);
    var_16 = var_14 && var_15;
    if (var_16) {
        // return wp.vec3(0.0, 1.0, 0.0)                                                      <L 33>
        var_17 = wp::vec_t<3,wp::float32>(var_5, var_9, var_5);
        goto label1;
    }
    // vc = d1 * d4 - d3 * d2                                                                 <L 35>
    var_18 = wp::mul(var_3, var_13);
    var_19 = wp::mul(var_12, var_4);
    var_20 = wp::sub(var_18, var_19);
    // v = d1 / (d1 - d3)                                                                     <L 36>
    var_21 = wp::sub(var_3, var_12);
    var_22 = wp::div(var_3, var_21);
    // if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:                                              <L 37>
    var_23 = (var_20 <= var_5);
    var_24 = (var_3 >= var_5);
    var_25 = (var_12 <= var_5);
    var_26 = var_23 && var_24 && var_25;
    if (var_26) {
        // return wp.vec3(1.0 - v, v, 0.0)                                                    <L 38>
        var_27 = wp::sub(var_9, var_22);
        var_28 = wp::vec_t<3,wp::float32>(var_27, var_22, var_5);
        goto label2;
    }
    // cp = p - c                                                                             <L 40>
    var_29 = wp::sub(var_p, var_c);
    // d5 = wp.dot(ab, cp)                                                                    <L 41>
    var_30 = wp::dot(var_0, var_29);
    // d6 = wp.dot(ac, cp)                                                                    <L 42>
    var_31 = wp::dot(var_1, var_29);
    // if d6 >= 0.0 and d5 <= d6:                                                             <L 44>
    var_32 = (var_31 >= var_5);
    var_33 = (var_30 <= var_31);
    var_34 = var_32 && var_33;
    if (var_34) {
        // return wp.vec3(0.0, 0.0, 1.0)                                                      <L 45>
        var_35 = wp::vec_t<3,wp::float32>(var_5, var_5, var_9);
        goto label3;
    }
    // vb = d5 * d2 - d1 * d6                                                                 <L 47>
    var_36 = wp::mul(var_30, var_4);
    var_37 = wp::mul(var_3, var_31);
    var_38 = wp::sub(var_36, var_37);
    // w = d2 / (d2 - d6)                                                                     <L 48>
    var_39 = wp::sub(var_4, var_31);
    var_40 = wp::div(var_4, var_39);
    // if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:                                              <L 49>
    var_41 = (var_38 <= var_5);
    var_42 = (var_4 >= var_5);
    var_43 = (var_31 <= var_5);
    var_44 = var_41 && var_42 && var_43;
    if (var_44) {
        // return wp.vec3(1.0 - w, 0.0, w)                                                    <L 50>
        var_45 = wp::sub(var_9, var_40);
        var_46 = wp::vec_t<3,wp::float32>(var_45, var_5, var_40);
        goto label4;
    }
    // va = d3 * d6 - d5 * d4                                                                 <L 52>
    var_47 = wp::mul(var_12, var_31);
    var_48 = wp::mul(var_30, var_13);
    var_49 = wp::sub(var_47, var_48);
    // w = (d4 - d3) / ((d4 - d3) + (d5 - d6))                                                <L 53>
    var_50 = wp::sub(var_13, var_12);
    var_51 = wp::sub(var_13, var_12);
    var_52 = wp::sub(var_30, var_31);
    var_53 = wp::add(var_51, var_52);
    var_54 = wp::div(var_50, var_53);
    // if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:                                <L 54>
    var_55 = (var_49 <= var_5);
    var_56 = wp::sub(var_13, var_12);
    var_57 = (var_56 >= var_5);
    var_58 = wp::sub(var_30, var_31);
    var_59 = (var_58 >= var_5);
    var_60 = var_55 && var_57 && var_59;
    if (var_60) {
        // return wp.vec3(0.0, w, 1.0 - w)                                                    <L 55>
        var_61 = wp::sub(var_9, var_54);
        var_62 = wp::vec_t<3,wp::float32>(var_5, var_54, var_61);
        goto label5;
    }
    // denom = 1.0 / (va + vb + vc)                                                           <L 57>
    var_63 = wp::add(var_49, var_38);
    var_64 = wp::add(var_63, var_20);
    var_65 = wp::div(var_9, var_64);
    // v = vb * denom                                                                         <L 58>
    var_66 = wp::mul(var_38, var_65);
    // w = vc * denom                                                                         <L 59>
    var_67 = wp::mul(var_20, var_65);
    // return wp.vec3(1.0 - v - w, v, w)                                                      <L 61>
    var_68 = wp::sub(var_9, var_66);
    var_69 = wp::sub(var_68, var_67);
    var_70 = wp::vec_t<3,wp::float32>(var_69, var_66, var_67);
    goto label6;
    //---------
    // reverse
    label6:;
    adj_70 += adj_ret;
    wp::adj_vec_t(var_69, var_66, var_67, adj_69, adj_66, adj_67, adj_70);
    wp::adj_sub(var_68, var_67, adj_68, adj_67, adj_69);
    wp::adj_sub(var_9, var_66, adj_9, adj_66, adj_68);
    // adj: return wp.vec3(1.0 - v - w, v, w)                                                 <L 61>
    wp::adj_mul(var_20, var_65, adj_20, adj_65, adj_67);
    // adj: w = vc * denom                                                                    <L 59>
    wp::adj_mul(var_38, var_65, adj_38, adj_65, adj_66);
    // adj: v = vb * denom                                                                    <L 58>
    wp::adj_div(var_9, var_64, var_65, adj_9, adj_64, adj_65);
    wp::adj_add(var_63, var_20, adj_63, adj_20, adj_64);
    wp::adj_add(var_49, var_38, adj_49, adj_38, adj_63);
    // adj: denom = 1.0 / (va + vb + vc)                                                      <L 57>
    if (var_60) {
        label5:;
        adj_62 += adj_ret;
        wp::adj_vec_t(var_5, var_54, var_61, adj_5, adj_54, adj_61, adj_62);
        wp::adj_sub(var_9, var_54, adj_9, adj_54, adj_61);
        // adj: return wp.vec3(0.0, w, 1.0 - w)                                               <L 55>
    }
    wp::adj_sub(var_30, var_31, adj_30, adj_31, adj_58);
    wp::adj_sub(var_13, var_12, adj_13, adj_12, adj_56);
    // adj: if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:                           <L 54>
    wp::adj_div(var_50, var_53, var_54, adj_50, adj_53, adj_54);
    wp::adj_add(var_51, var_52, adj_51, adj_52, adj_53);
    wp::adj_sub(var_30, var_31, adj_30, adj_31, adj_52);
    wp::adj_sub(var_13, var_12, adj_13, adj_12, adj_51);
    wp::adj_sub(var_13, var_12, adj_13, adj_12, adj_50);
    // adj: w = (d4 - d3) / ((d4 - d3) + (d5 - d6))                                           <L 53>
    wp::adj_sub(var_47, var_48, adj_47, adj_48, adj_49);
    wp::adj_mul(var_30, var_13, adj_30, adj_13, adj_48);
    wp::adj_mul(var_12, var_31, adj_12, adj_31, adj_47);
    // adj: va = d3 * d6 - d5 * d4                                                            <L 52>
    if (var_44) {
        label4:;
        adj_46 += adj_ret;
        wp::adj_vec_t(var_45, var_5, var_40, adj_45, adj_5, adj_40, adj_46);
        wp::adj_sub(var_9, var_40, adj_9, adj_40, adj_45);
        // adj: return wp.vec3(1.0 - w, 0.0, w)                                               <L 50>
    }
    // adj: if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:                                         <L 49>
    wp::adj_div(var_4, var_39, var_40, adj_4, adj_39, adj_40);
    wp::adj_sub(var_4, var_31, adj_4, adj_31, adj_39);
    // adj: w = d2 / (d2 - d6)                                                                <L 48>
    wp::adj_sub(var_36, var_37, adj_36, adj_37, adj_38);
    wp::adj_mul(var_3, var_31, adj_3, adj_31, adj_37);
    wp::adj_mul(var_30, var_4, adj_30, adj_4, adj_36);
    // adj: vb = d5 * d2 - d1 * d6                                                            <L 47>
    if (var_34) {
        label3:;
        adj_35 += adj_ret;
        wp::adj_vec_t(var_5, var_5, var_9, adj_5, adj_5, adj_9, adj_35);
        // adj: return wp.vec3(0.0, 0.0, 1.0)                                                 <L 45>
    }
    // adj: if d6 >= 0.0 and d5 <= d6:                                                        <L 44>
    wp::adj_dot(var_1, var_29, adj_1, adj_29, adj_31);
    // adj: d6 = wp.dot(ac, cp)                                                               <L 42>
    wp::adj_dot(var_0, var_29, adj_0, adj_29, adj_30);
    // adj: d5 = wp.dot(ab, cp)                                                               <L 41>
    wp::adj_sub(var_p, var_c, adj_p, adj_c, adj_29);
    // adj: cp = p - c                                                                        <L 40>
    if (var_26) {
        label2:;
        adj_28 += adj_ret;
        wp::adj_vec_t(var_27, var_22, var_5, adj_27, adj_22, adj_5, adj_28);
        wp::adj_sub(var_9, var_22, adj_9, adj_22, adj_27);
        // adj: return wp.vec3(1.0 - v, v, 0.0)                                               <L 38>
    }
    // adj: if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:                                         <L 37>
    wp::adj_div(var_3, var_21, var_22, adj_3, adj_21, adj_22);
    wp::adj_sub(var_3, var_12, adj_3, adj_12, adj_21);
    // adj: v = d1 / (d1 - d3)                                                                <L 36>
    wp::adj_sub(var_18, var_19, adj_18, adj_19, adj_20);
    wp::adj_mul(var_12, var_4, adj_12, adj_4, adj_19);
    wp::adj_mul(var_3, var_13, adj_3, adj_13, adj_18);
    // adj: vc = d1 * d4 - d3 * d2                                                            <L 35>
    if (var_16) {
        label1:;
        adj_17 += adj_ret;
        wp::adj_vec_t(var_5, var_9, var_5, adj_5, adj_9, adj_5, adj_17);
        // adj: return wp.vec3(0.0, 1.0, 0.0)                                                 <L 33>
    }
    // adj: if d3 >= 0.0 and d4 <= d3:                                                        <L 32>
    wp::adj_dot(var_1, var_11, adj_1, adj_11, adj_13);
    // adj: d4 = wp.dot(ac, bp)                                                               <L 30>
    wp::adj_dot(var_0, var_11, adj_0, adj_11, adj_12);
    // adj: d3 = wp.dot(ab, bp)                                                               <L 29>
    wp::adj_sub(var_p, var_b, adj_p, adj_b, adj_11);
    // adj: bp = p - b                                                                        <L 28>
    if (var_8) {
        label0:;
        adj_10 += adj_ret;
        wp::adj_vec_t(var_9, var_5, var_5, adj_9, adj_5, adj_5, adj_10);
        // adj: return wp.vec3(1.0, 0.0, 0.0)                                                 <L 26>
    }
    // adj: if d1 <= 0.0 and d2 <= 0.0:                                                       <L 25>
    wp::adj_dot(var_1, var_2, adj_1, adj_2, adj_4);
    // adj: d2 = wp.dot(ac, ap)                                                               <L 23>
    wp::adj_dot(var_0, var_2, adj_0, adj_2, adj_3);
    // adj: d1 = wp.dot(ab, ap)                                                               <L 22>
    wp::adj_sub(var_p, var_a, adj_p, adj_a, adj_2);
    // adj: ap = p - a                                                                        <L 20>
    wp::adj_sub(var_c, var_a, adj_c, adj_a, adj_1);
    // adj: ac = c - a                                                                        <L 19>
    wp::adj_sub(var_b, var_a, adj_b, adj_a, adj_0);
    // adj: ab = b - a                                                                        <L 18>
    // adj: def triangle_closest_point_barycentric(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3):  <L 17>
    return;
}


// G:\My Drive\sourceCodes\fs5ydw-main\utils\utils.py:15
static CUDA_CALLABLE wp::quat_t<wp::float32> quat_twist(
    wp::vec_t<3,wp::float32> var_axis,
    wp::quat_t<wp::float32> var_q)
{
    //---------
    // primal vars
    const wp::str var_0 = "\n    Returns the twist around an axis.\n    ";
    const wp::int32 var_1 = 0;
    wp::float32 var_2;
    const wp::int32 var_3 = 1;
    wp::float32 var_4;
    const wp::int32 var_5 = 2;
    wp::float32 var_6;
    wp::vec_t<3,wp::float32> var_7;
    wp::float32 var_8;
    wp::vec_t<3,wp::float32> var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    const wp::int32 var_13 = 3;
    wp::float32 var_14;
    wp::quat_t<wp::float32> var_15;
    wp::quat_t<wp::float32> var_16;
    //---------
    // forward
    // def quat_twist(axis: wp.vec3, q: wp.quat):                                             <L 16>
    // """                                                                                    <L 17>
    // a = wp.vec3(q[0], q[1], q[2])                                                          <L 22>
    var_2 = wp::extract(var_q, var_1);
    var_4 = wp::extract(var_q, var_3);
    var_6 = wp::extract(var_q, var_5);
    var_7 = wp::vec_t<3,wp::float32>(var_2, var_4, var_6);
    // proj = wp.dot(a, axis)                                                                 <L 23>
    var_8 = wp::dot(var_7, var_axis);
    // a = proj * axis                                                                        <L 24>
    var_9 = wp::mul(var_8, var_axis);
    // return wp.normalize(wp.quat(a[0], a[1], a[2], q[3]))                                   <L 28>
    var_10 = wp::extract(var_9, var_1);
    var_11 = wp::extract(var_9, var_3);
    var_12 = wp::extract(var_9, var_5);
    var_14 = wp::extract(var_q, var_13);
    var_15 = wp::quat_t<wp::float32>(var_10, var_11, var_12, var_14);
    var_16 = wp::normalize(var_15);
    return var_16;
}


// G:\My Drive\sourceCodes\fs5ydw-main\utils\utils.py:15
static CUDA_CALLABLE void adj_quat_twist(
    wp::vec_t<3,wp::float32> var_axis,
    wp::quat_t<wp::float32> var_q,
    wp::vec_t<3,wp::float32> & adj_axis,
    wp::quat_t<wp::float32> & adj_q,
    wp::quat_t<wp::float32> & adj_ret)
{
    //---------
    // primal vars
    const wp::str var_0 = "\n    Returns the twist around an axis.\n    ";
    const wp::int32 var_1 = 0;
    wp::float32 var_2;
    const wp::int32 var_3 = 1;
    wp::float32 var_4;
    const wp::int32 var_5 = 2;
    wp::float32 var_6;
    wp::vec_t<3,wp::float32> var_7;
    wp::float32 var_8;
    wp::vec_t<3,wp::float32> var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    const wp::int32 var_13 = 3;
    wp::float32 var_14;
    wp::quat_t<wp::float32> var_15;
    wp::quat_t<wp::float32> var_16;
    //---------
    // dual vars
    wp::str adj_0 = {};
    wp::int32 adj_1 = {};
    wp::float32 adj_2 = {};
    wp::int32 adj_3 = {};
    wp::float32 adj_4 = {};
    wp::int32 adj_5 = {};
    wp::float32 adj_6 = {};
    wp::vec_t<3,wp::float32> adj_7 = {};
    wp::float32 adj_8 = {};
    wp::vec_t<3,wp::float32> adj_9 = {};
    wp::float32 adj_10 = {};
    wp::float32 adj_11 = {};
    wp::float32 adj_12 = {};
    wp::int32 adj_13 = {};
    wp::float32 adj_14 = {};
    wp::quat_t<wp::float32> adj_15 = {};
    wp::quat_t<wp::float32> adj_16 = {};
    //---------
    // forward
    // def quat_twist(axis: wp.vec3, q: wp.quat):                                             <L 16>
    // """                                                                                    <L 17>
    // a = wp.vec3(q[0], q[1], q[2])                                                          <L 22>
    var_2 = wp::extract(var_q, var_1);
    var_4 = wp::extract(var_q, var_3);
    var_6 = wp::extract(var_q, var_5);
    var_7 = wp::vec_t<3,wp::float32>(var_2, var_4, var_6);
    // proj = wp.dot(a, axis)                                                                 <L 23>
    var_8 = wp::dot(var_7, var_axis);
    // a = proj * axis                                                                        <L 24>
    var_9 = wp::mul(var_8, var_axis);
    // return wp.normalize(wp.quat(a[0], a[1], a[2], q[3]))                                   <L 28>
    var_10 = wp::extract(var_9, var_1);
    var_11 = wp::extract(var_9, var_3);
    var_12 = wp::extract(var_9, var_5);
    var_14 = wp::extract(var_q, var_13);
    var_15 = wp::quat_t<wp::float32>(var_10, var_11, var_12, var_14);
    var_16 = wp::normalize(var_15);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_16 += adj_ret;
    wp::adj_normalize(var_15, adj_15, adj_16);
    wp::adj_quat_t(var_10, var_11, var_12, var_14, adj_10, adj_11, adj_12, adj_14, adj_15);
    wp::adj_extract(var_q, var_13, adj_q, adj_13, adj_14);
    wp::adj_extract(var_9, var_5, adj_9, adj_5, adj_12);
    wp::adj_extract(var_9, var_3, adj_9, adj_3, adj_11);
    wp::adj_extract(var_9, var_1, adj_9, adj_1, adj_10);
    // adj: return wp.normalize(wp.quat(a[0], a[1], a[2], q[3]))                              <L 28>
    wp::adj_mul(var_8, var_axis, adj_8, adj_axis, adj_9);
    // adj: a = proj * axis                                                                   <L 24>
    wp::adj_dot(var_7, var_axis, adj_7, adj_axis, adj_8);
    // adj: proj = wp.dot(a, axis)                                                            <L 23>
    wp::adj_vec_t(var_2, var_4, var_6, adj_2, adj_4, adj_6, adj_7);
    wp::adj_extract(var_q, var_5, adj_q, adj_5, adj_6);
    wp::adj_extract(var_q, var_3, adj_q, adj_3, adj_4);
    wp::adj_extract(var_q, var_1, adj_q, adj_1, adj_2);
    // adj: a = wp.vec3(q[0], q[1], q[2])                                                     <L 22>
    // adj: """                                                                               <L 17>
    // adj: def quat_twist(axis: wp.vec3, q: wp.quat):                                        <L 16>
    return;
}


// G:\My Drive\sourceCodes\fs5ydw-main\utils\utils.py:39
static CUDA_CALLABLE wp::vec_t<3,wp::float32> quat_decompose(
    wp::quat_t<wp::float32> var_q)
{
    //---------
    // primal vars
    const wp::str var_0 = "\n    Decompose a quaternion into a sequence of 3 rotations around x,y',z' respectively, i.e.: q = q_z''q_y'q_x.\n    ";
    const wp::float32 var_1 = 1.0;
    const wp::float32 var_2 = 0.0;
    wp::vec_t<3,wp::float32> var_3;
    wp::vec_t<3,wp::float32> var_4;
    wp::vec_t<3,wp::float32> var_5;
    wp::vec_t<3,wp::float32> var_6;
    wp::vec_t<3,wp::float32> var_7;
    wp::vec_t<3,wp::float32> var_8;
    wp::mat_t<3,3,wp::float32> var_9;
    const wp::int32 var_10 = 1;
    const wp::int32 var_11 = 2;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    const wp::int32 var_15 = 0;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    bool var_19;
    const wp::float32 var_20 = 1.57079632679;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::float32 var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::vec_t<3,wp::float32> var_30;
    wp::vec_t<3,wp::float32> var_31;
    //---------
    // forward
    // def quat_decompose(q: wp.quat):                                                        <L 40>
    // """                                                                                    <L 41>
    // R = wp.mat33(                                                                          <L 45>
    // wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0)),                                             <L 46>
    var_3 = wp::vec_t<3,wp::float32>(var_1, var_2, var_2);
    var_4 = wp::quat_rotate(var_q, var_3);
    // wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0)),                                             <L 47>
    var_5 = wp::vec_t<3,wp::float32>(var_2, var_1, var_2);
    var_6 = wp::quat_rotate(var_q, var_5);
    // wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0)),                                             <L 48>
    var_7 = wp::vec_t<3,wp::float32>(var_2, var_2, var_1);
    var_8 = wp::quat_rotate(var_q, var_7);
    var_9 = wp::mat_t<3,3,wp::float32>(var_4, var_6, var_8);
    // phi = wp.atan2(R[1, 2], R[2, 2])                                                       <L 52>
    var_12 = wp::extract(var_9, var_10, var_11);
    var_13 = wp::extract(var_9, var_11, var_11);
    var_14 = wp::atan2(var_12, var_13);
    // sinp = -R[0, 2]                                                                        <L 53>
    var_16 = wp::extract(var_9, var_15, var_11);
    var_17 = wp::neg(var_16);
    // if wp.abs(sinp) >= 1.0:                                                                <L 54>
    var_18 = wp::abs(var_17);
    var_19 = (var_18 >= var_1);
    if (var_19) {
        // theta = 1.57079632679 * wp.sign(sinp)                                              <L 55>
        var_21 = wp::sign(var_17);
        var_22 = wp::mul(var_20, var_21);
    }
    if (!var_19) {
        // theta = wp.asin(-R[0, 2])                                                          <L 57>
        var_23 = wp::extract(var_9, var_15, var_11);
        var_24 = wp::neg(var_23);
        var_25 = wp::asin(var_24);
    }
    var_26 = wp::select(var_19, var_25, var_22);
    // psi = wp.atan2(R[0, 1], R[0, 0])                                                       <L 58>
    var_27 = wp::extract(var_9, var_15, var_10);
    var_28 = wp::extract(var_9, var_15, var_15);
    var_29 = wp::atan2(var_27, var_28);
    // return -wp.vec3(phi, theta, psi)                                                       <L 60>
    var_30 = wp::vec_t<3,wp::float32>(var_14, var_26, var_29);
    var_31 = wp::neg(var_30);
    return var_31;
}


// G:\My Drive\sourceCodes\fs5ydw-main\utils\utils.py:39
static CUDA_CALLABLE void adj_quat_decompose(
    wp::quat_t<wp::float32> var_q,
    wp::quat_t<wp::float32> & adj_q,
    wp::vec_t<3,wp::float32> & adj_ret)
{
    //---------
    // primal vars
    const wp::str var_0 = "\n    Decompose a quaternion into a sequence of 3 rotations around x,y',z' respectively, i.e.: q = q_z''q_y'q_x.\n    ";
    const wp::float32 var_1 = 1.0;
    const wp::float32 var_2 = 0.0;
    wp::vec_t<3,wp::float32> var_3;
    wp::vec_t<3,wp::float32> var_4;
    wp::vec_t<3,wp::float32> var_5;
    wp::vec_t<3,wp::float32> var_6;
    wp::vec_t<3,wp::float32> var_7;
    wp::vec_t<3,wp::float32> var_8;
    wp::mat_t<3,3,wp::float32> var_9;
    const wp::int32 var_10 = 1;
    const wp::int32 var_11 = 2;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    const wp::int32 var_15 = 0;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    bool var_19;
    const wp::float32 var_20 = 1.57079632679;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::float32 var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::vec_t<3,wp::float32> var_30;
    wp::vec_t<3,wp::float32> var_31;
    //---------
    // dual vars
    wp::str adj_0 = {};
    wp::float32 adj_1 = {};
    wp::float32 adj_2 = {};
    wp::vec_t<3,wp::float32> adj_3 = {};
    wp::vec_t<3,wp::float32> adj_4 = {};
    wp::vec_t<3,wp::float32> adj_5 = {};
    wp::vec_t<3,wp::float32> adj_6 = {};
    wp::vec_t<3,wp::float32> adj_7 = {};
    wp::vec_t<3,wp::float32> adj_8 = {};
    wp::mat_t<3,3,wp::float32> adj_9 = {};
    wp::int32 adj_10 = {};
    wp::int32 adj_11 = {};
    wp::float32 adj_12 = {};
    wp::float32 adj_13 = {};
    wp::float32 adj_14 = {};
    wp::int32 adj_15 = {};
    wp::float32 adj_16 = {};
    wp::float32 adj_17 = {};
    wp::float32 adj_18 = {};
    bool adj_19 = {};
    wp::float32 adj_20 = {};
    wp::float32 adj_21 = {};
    wp::float32 adj_22 = {};
    wp::float32 adj_23 = {};
    wp::float32 adj_24 = {};
    wp::float32 adj_25 = {};
    wp::float32 adj_26 = {};
    wp::float32 adj_27 = {};
    wp::float32 adj_28 = {};
    wp::float32 adj_29 = {};
    wp::vec_t<3,wp::float32> adj_30 = {};
    wp::vec_t<3,wp::float32> adj_31 = {};
    //---------
    // forward
    // def quat_decompose(q: wp.quat):                                                        <L 40>
    // """                                                                                    <L 41>
    // R = wp.mat33(                                                                          <L 45>
    // wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0)),                                             <L 46>
    var_3 = wp::vec_t<3,wp::float32>(var_1, var_2, var_2);
    var_4 = wp::quat_rotate(var_q, var_3);
    // wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0)),                                             <L 47>
    var_5 = wp::vec_t<3,wp::float32>(var_2, var_1, var_2);
    var_6 = wp::quat_rotate(var_q, var_5);
    // wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0)),                                             <L 48>
    var_7 = wp::vec_t<3,wp::float32>(var_2, var_2, var_1);
    var_8 = wp::quat_rotate(var_q, var_7);
    var_9 = wp::mat_t<3,3,wp::float32>(var_4, var_6, var_8);
    // phi = wp.atan2(R[1, 2], R[2, 2])                                                       <L 52>
    var_12 = wp::extract(var_9, var_10, var_11);
    var_13 = wp::extract(var_9, var_11, var_11);
    var_14 = wp::atan2(var_12, var_13);
    // sinp = -R[0, 2]                                                                        <L 53>
    var_16 = wp::extract(var_9, var_15, var_11);
    var_17 = wp::neg(var_16);
    // if wp.abs(sinp) >= 1.0:                                                                <L 54>
    var_18 = wp::abs(var_17);
    var_19 = (var_18 >= var_1);
    if (var_19) {
        // theta = 1.57079632679 * wp.sign(sinp)                                              <L 55>
        var_21 = wp::sign(var_17);
        var_22 = wp::mul(var_20, var_21);
    }
    if (!var_19) {
        // theta = wp.asin(-R[0, 2])                                                          <L 57>
        var_23 = wp::extract(var_9, var_15, var_11);
        var_24 = wp::neg(var_23);
        var_25 = wp::asin(var_24);
    }
    var_26 = wp::select(var_19, var_25, var_22);
    // psi = wp.atan2(R[0, 1], R[0, 0])                                                       <L 58>
    var_27 = wp::extract(var_9, var_15, var_10);
    var_28 = wp::extract(var_9, var_15, var_15);
    var_29 = wp::atan2(var_27, var_28);
    // return -wp.vec3(phi, theta, psi)                                                       <L 60>
    var_30 = wp::vec_t<3,wp::float32>(var_14, var_26, var_29);
    var_31 = wp::neg(var_30);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_31 += adj_ret;
    wp::adj_neg(var_30, adj_30, adj_31);
    wp::adj_vec_t(var_14, var_26, var_29, adj_14, adj_26, adj_29, adj_30);
    // adj: return -wp.vec3(phi, theta, psi)                                                  <L 60>
    wp::adj_atan2(var_27, var_28, adj_27, adj_28, adj_29);
    wp::adj_extract(var_9, var_15, var_15, adj_9, adj_15, adj_15, adj_28);
    wp::adj_extract(var_9, var_15, var_10, adj_9, adj_15, adj_10, adj_27);
    // adj: psi = wp.atan2(R[0, 1], R[0, 0])                                                  <L 58>
    wp::adj_select(var_19, var_25, var_22, adj_19, adj_25, adj_22, adj_26);
    if (!var_19) {
        wp::adj_asin(var_24, adj_24, adj_25);
        wp::adj_neg(var_23, adj_23, adj_24);
        wp::adj_extract(var_9, var_15, var_11, adj_9, adj_15, adj_11, adj_23);
        // adj: theta = wp.asin(-R[0, 2])                                                     <L 57>
    }
    if (var_19) {
        wp::adj_mul(var_20, var_21, adj_20, adj_21, adj_22);
        wp::adj_sign(var_17, adj_17, adj_21);
        // adj: theta = 1.57079632679 * wp.sign(sinp)                                         <L 55>
    }
    wp::adj_abs(var_17, adj_17, adj_18);
    // adj: if wp.abs(sinp) >= 1.0:                                                           <L 54>
    wp::adj_neg(var_16, adj_16, adj_17);
    wp::adj_extract(var_9, var_15, var_11, adj_9, adj_15, adj_11, adj_16);
    // adj: sinp = -R[0, 2]                                                                   <L 53>
    wp::adj_atan2(var_12, var_13, adj_12, adj_13, adj_14);
    wp::adj_extract(var_9, var_11, var_11, adj_9, adj_11, adj_11, adj_13);
    wp::adj_extract(var_9, var_10, var_11, adj_9, adj_10, adj_11, adj_12);
    // adj: phi = wp.atan2(R[1, 2], R[2, 2])                                                  <L 52>
    wp::adj_mat_t(var_4, var_6, var_8, adj_4, adj_6, adj_8, adj_9);
    wp::adj_quat_rotate(var_q, var_7, adj_q, adj_7, adj_8);
    wp::adj_vec_t(var_2, var_2, var_1, adj_2, adj_2, adj_1, adj_7);
    // adj: wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0)),                                        <L 48>
    wp::adj_quat_rotate(var_q, var_5, adj_q, adj_5, adj_6);
    wp::adj_vec_t(var_2, var_1, var_2, adj_2, adj_1, adj_2, adj_5);
    // adj: wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0)),                                        <L 47>
    wp::adj_quat_rotate(var_q, var_3, adj_q, adj_3, adj_4);
    wp::adj_vec_t(var_1, var_2, var_2, adj_1, adj_2, adj_2, adj_3);
    // adj: wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0)),                                        <L 46>
    // adj: R = wp.mat33(                                                                     <L 45>
    // adj: """                                                                               <L 41>
    // adj: def quat_decompose(q: wp.quat):                                                   <L 40>
    return;
}



extern "C" __global__ void integrate_particles_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float32>> var_x,
    wp::array_t<wp::vec_t<3,wp::float32>> var_v,
    wp::array_t<wp::vec_t<3,wp::float32>> var_f,
    wp::array_t<wp::float32> var_w,
    wp::array_t<wp::uint32> var_particle_flags,
    wp::vec_t<3,wp::float32> var_gravity,
    wp::float32 var_dt,
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
        wp::vec_t<3,wp::float32> var_27;
        wp::vec_t<3,wp::float32> var_28;
        //---------
        // forward
        // def integrate_particles(                                                               <L 24>
        // tid = wp.tid()                                                                         <L 35>
        var_0 = builtin_tid1d();
        // if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:                                  <L 36>
        var_1 = wp::address(var_particle_flags, var_0);
        var_3 = wp::load(var_1);
        var_4 = wp::bit_and(var_3, var_2);
        var_6 = (var_4 == var_5);
        if (var_6) {
            // return                                                                             <L 37>
            return;
        }
        // x0 = x[tid]                                                                            <L 39>
        var_7 = wp::address(var_x, var_0);
        var_8 = wp::load(var_7);
        var_9 = wp::copy(var_8);
        // v0 = v[tid]                                                                            <L 40>
        var_10 = wp::address(var_v, var_0);
        var_11 = wp::load(var_10);
        var_12 = wp::copy(var_11);
        // f0 = f[tid]                                                                            <L 41>
        var_13 = wp::address(var_f, var_0);
        var_14 = wp::load(var_13);
        var_15 = wp::copy(var_14);
        // inv_mass = w[tid]                                                                      <L 43>
        var_16 = wp::address(var_w, var_0);
        var_17 = wp::load(var_16);
        var_18 = wp::copy(var_17);
        // v1 = v0 + (f0 * inv_mass + gravity * wp.step(0.0 - inv_mass)) * dt                     <L 46>
        var_19 = wp::mul(var_15, var_18);
        var_21 = wp::sub(var_20, var_18);
        var_22 = wp::step(var_21);
        var_23 = wp::mul(var_gravity, var_22);
        var_24 = wp::add(var_19, var_23);
        var_25 = wp::mul(var_24, var_dt);
        var_26 = wp::add(var_12, var_25);
        // x1 = x0 + v1 * dt                                                                      <L 47>
        var_27 = wp::mul(var_26, var_dt);
        var_28 = wp::add(var_9, var_27);
        // x_new[tid] = x1                                                                        <L 49>
        wp::array_store(var_x_new, var_0, var_28);
        // v_new[tid] = v1                                                                        <L 50>
        wp::array_store(var_v_new, var_0, var_26);
    }
}

extern "C" __global__ void integrate_particles_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float32>> var_x,
    wp::array_t<wp::vec_t<3,wp::float32>> var_v,
    wp::array_t<wp::vec_t<3,wp::float32>> var_f,
    wp::array_t<wp::float32> var_w,
    wp::array_t<wp::uint32> var_particle_flags,
    wp::vec_t<3,wp::float32> var_gravity,
    wp::float32 var_dt,
    wp::array_t<wp::vec_t<3,wp::float32>> var_x_new,
    wp::array_t<wp::vec_t<3,wp::float32>> var_v_new,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_x,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_v,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_f,
    wp::array_t<wp::float32> adj_w,
    wp::array_t<wp::uint32> adj_particle_flags,
    wp::vec_t<3,wp::float32> adj_gravity,
    wp::float32 adj_dt,
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
        wp::vec_t<3,wp::float32> var_27;
        wp::vec_t<3,wp::float32> var_28;
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
        wp::vec_t<3,wp::float32> adj_27 = {};
        wp::vec_t<3,wp::float32> adj_28 = {};
        //---------
        // forward
        // def integrate_particles(                                                               <L 24>
        // tid = wp.tid()                                                                         <L 35>
        var_0 = builtin_tid1d();
        // if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:                                  <L 36>
        var_1 = wp::address(var_particle_flags, var_0);
        var_3 = wp::load(var_1);
        var_4 = wp::bit_and(var_3, var_2);
        var_6 = (var_4 == var_5);
        if (var_6) {
            // return                                                                             <L 37>
            goto label0;
        }
        // x0 = x[tid]                                                                            <L 39>
        var_7 = wp::address(var_x, var_0);
        var_8 = wp::load(var_7);
        var_9 = wp::copy(var_8);
        // v0 = v[tid]                                                                            <L 40>
        var_10 = wp::address(var_v, var_0);
        var_11 = wp::load(var_10);
        var_12 = wp::copy(var_11);
        // f0 = f[tid]                                                                            <L 41>
        var_13 = wp::address(var_f, var_0);
        var_14 = wp::load(var_13);
        var_15 = wp::copy(var_14);
        // inv_mass = w[tid]                                                                      <L 43>
        var_16 = wp::address(var_w, var_0);
        var_17 = wp::load(var_16);
        var_18 = wp::copy(var_17);
        // v1 = v0 + (f0 * inv_mass + gravity * wp.step(0.0 - inv_mass)) * dt                     <L 46>
        var_19 = wp::mul(var_15, var_18);
        var_21 = wp::sub(var_20, var_18);
        var_22 = wp::step(var_21);
        var_23 = wp::mul(var_gravity, var_22);
        var_24 = wp::add(var_19, var_23);
        var_25 = wp::mul(var_24, var_dt);
        var_26 = wp::add(var_12, var_25);
        // x1 = x0 + v1 * dt                                                                      <L 47>
        var_27 = wp::mul(var_26, var_dt);
        var_28 = wp::add(var_9, var_27);
        // x_new[tid] = x1                                                                        <L 49>
        // wp::array_store(var_x_new, var_0, var_28);
        // v_new[tid] = v1                                                                        <L 50>
        // wp::array_store(var_v_new, var_0, var_26);
        //---------
        // reverse
        wp::adj_array_store(var_v_new, var_0, var_26, adj_v_new, adj_0, adj_26);
        // adj: v_new[tid] = v1                                                                   <L 50>
        wp::adj_array_store(var_x_new, var_0, var_28, adj_x_new, adj_0, adj_28);
        // adj: x_new[tid] = x1                                                                   <L 49>
        wp::adj_add(var_9, var_27, adj_9, adj_27, adj_28);
        wp::adj_mul(var_26, var_dt, adj_26, adj_dt, adj_27);
        // adj: x1 = x0 + v1 * dt                                                                 <L 47>
        wp::adj_add(var_12, var_25, adj_12, adj_25, adj_26);
        wp::adj_mul(var_24, var_dt, adj_24, adj_dt, adj_25);
        wp::adj_add(var_19, var_23, adj_19, adj_23, adj_24);
        wp::adj_mul(var_gravity, var_22, adj_gravity, adj_22, adj_23);
        wp::adj_step(var_21, adj_21, adj_22);
        wp::adj_sub(var_20, var_18, adj_20, adj_18, adj_21);
        wp::adj_mul(var_15, var_18, adj_15, adj_18, adj_19);
        // adj: v1 = v0 + (f0 * inv_mass + gravity * wp.step(0.0 - inv_mass)) * dt                <L 46>
        wp::adj_copy(var_17, adj_16, adj_18);
        wp::adj_load(var_16, adj_16, adj_17);
        wp::adj_address(var_w, var_0, adj_w, adj_0, adj_16);
        // adj: inv_mass = w[tid]                                                                 <L 43>
        wp::adj_copy(var_14, adj_13, adj_15);
        wp::adj_load(var_13, adj_13, adj_14);
        wp::adj_address(var_f, var_0, adj_f, adj_0, adj_13);
        // adj: f0 = f[tid]                                                                       <L 41>
        wp::adj_copy(var_11, adj_10, adj_12);
        wp::adj_load(var_10, adj_10, adj_11);
        wp::adj_address(var_v, var_0, adj_v, adj_0, adj_10);
        // adj: v0 = v[tid]                                                                       <L 40>
        wp::adj_copy(var_8, adj_7, adj_9);
        wp::adj_load(var_7, adj_7, adj_8);
        wp::adj_address(var_x, var_0, adj_x, adj_0, adj_7);
        // adj: x0 = x[tid]                                                                       <L 39>
        if (var_6) {
            label0:;
            // adj: return                                                                        <L 37>
        }
        wp::adj_bit_and(var_3, var_2, adj_1, adj_2, adj_4);
        wp::adj_load(var_1, adj_1, adj_3);
        wp::adj_address(var_particle_flags, var_0, adj_particle_flags, adj_0, adj_1);
        // adj: if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:                             <L 36>
        // adj: tid = wp.tid()                                                                    <L 35>
        // adj: def integrate_particles(                                                          <L 24>
        continue;
    }
}



extern "C" __global__ void integrate_bodies_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::transform_t<wp::float32>> var_body_q,
    wp::array_t<wp::vec_t<6,wp::float32>> var_body_qd,
    wp::array_t<wp::vec_t<6,wp::float32>> var_body_f,
    wp::array_t<wp::vec_t<3,wp::float32>> var_body_com,
    wp::array_t<wp::float32> var_m,
    wp::array_t<wp::mat_t<3,3,wp::float32>> var_I,
    wp::array_t<wp::float32> var_inv_m,
    wp::array_t<wp::mat_t<3,3,wp::float32>> var_inv_I,
    wp::vec_t<3,wp::float32> var_gravity,
    wp::float32 var_angular_damping,
    wp::float32 var_dt,
    wp::array_t<wp::transform_t<wp::float32>> var_body_q_new,
    wp::array_t<wp::vec_t<6,wp::float32>> var_body_qd_new)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        wp::transform_t<wp::float32>* var_1;
        wp::transform_t<wp::float32> var_2;
        wp::transform_t<wp::float32> var_3;
        wp::vec_t<6,wp::float32>* var_4;
        wp::vec_t<6,wp::float32> var_5;
        wp::vec_t<6,wp::float32> var_6;
        wp::vec_t<6,wp::float32>* var_7;
        wp::vec_t<6,wp::float32> var_8;
        wp::vec_t<6,wp::float32> var_9;
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        wp::float32* var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        wp::mat_t<3,3,wp::float32>* var_16;
        wp::mat_t<3,3,wp::float32> var_17;
        wp::mat_t<3,3,wp::float32> var_18;
        wp::mat_t<3,3,wp::float32>* var_19;
        wp::mat_t<3,3,wp::float32> var_20;
        wp::mat_t<3,3,wp::float32> var_21;
        wp::vec_t<3,wp::float32> var_22;
        wp::quat_t<wp::float32> var_23;
        wp::vec_t<3,wp::float32> var_24;
        wp::vec_t<3,wp::float32> var_25;
        wp::vec_t<3,wp::float32> var_26;
        wp::vec_t<3,wp::float32> var_27;
        wp::vec_t<3,wp::float32>* var_28;
        wp::vec_t<3,wp::float32> var_29;
        wp::vec_t<3,wp::float32> var_30;
        wp::vec_t<3,wp::float32> var_31;
        wp::vec_t<3,wp::float32> var_32;
        wp::float32 var_33;
        wp::vec_t<3,wp::float32> var_34;
        wp::vec_t<3,wp::float32> var_35;
        wp::vec_t<3,wp::float32> var_36;
        wp::vec_t<3,wp::float32> var_37;
        wp::vec_t<3,wp::float32> var_38;
        wp::vec_t<3,wp::float32> var_39;
        wp::vec_t<3,wp::float32> var_40;
        wp::vec_t<3,wp::float32> var_41;
        wp::vec_t<3,wp::float32> var_42;
        wp::vec_t<3,wp::float32> var_43;
        wp::vec_t<3,wp::float32> var_44;
        wp::vec_t<3,wp::float32> var_45;
        wp::vec_t<3,wp::float32> var_46;
        wp::vec_t<3,wp::float32> var_47;
        wp::vec_t<3,wp::float32> var_48;
        const wp::float32 var_49 = 0.0;
        wp::quat_t<wp::float32> var_50;
        wp::quat_t<wp::float32> var_51;
        const wp::float32 var_52 = 0.5;
        wp::quat_t<wp::float32> var_53;
        wp::quat_t<wp::float32> var_54;
        wp::quat_t<wp::float32> var_55;
        wp::quat_t<wp::float32> var_56;
        const wp::float32 var_57 = 1.0;
        wp::float32 var_58;
        wp::float32 var_59;
        wp::vec_t<3,wp::float32> var_60;
        wp::vec_t<3,wp::float32>* var_61;
        wp::vec_t<3,wp::float32> var_62;
        wp::vec_t<3,wp::float32> var_63;
        wp::vec_t<3,wp::float32> var_64;
        wp::transform_t<wp::float32> var_65;
        wp::vec_t<6,wp::float32> var_66;
        //---------
        // forward
        // def integrate_bodies(                                                                  <L 55>
        // tid = wp.tid()                                                                         <L 71>
        var_0 = builtin_tid1d();
        // q = body_q[tid]                                                                        <L 74>
        var_1 = wp::address(var_body_q, var_0);
        var_2 = wp::load(var_1);
        var_3 = wp::copy(var_2);
        // qd = body_qd[tid]                                                                      <L 75>
        var_4 = wp::address(var_body_qd, var_0);
        var_5 = wp::load(var_4);
        var_6 = wp::copy(var_5);
        // f = body_f[tid]                                                                        <L 76>
        var_7 = wp::address(var_body_f, var_0);
        var_8 = wp::load(var_7);
        var_9 = wp::copy(var_8);
        // mass = m[tid]                                                                          <L 79>
        var_10 = wp::address(var_m, var_0);
        var_11 = wp::load(var_10);
        var_12 = wp::copy(var_11);
        // inv_mass = inv_m[tid]  # 1 / mass                                                      <L 80>
        var_13 = wp::address(var_inv_m, var_0);
        var_14 = wp::load(var_13);
        var_15 = wp::copy(var_14);
        // inertia = I[tid]                                                                       <L 82>
        var_16 = wp::address(var_I, var_0);
        var_17 = wp::load(var_16);
        var_18 = wp::copy(var_17);
        // inv_inertia = inv_I[tid]  # inverse of 3x3 inertia matrix                              <L 83>
        var_19 = wp::address(var_inv_I, var_0);
        var_20 = wp::load(var_19);
        var_21 = wp::copy(var_20);
        // x0 = wp.transform_get_translation(q)                                                   <L 86>
        var_22 = wp::transform_get_translation(var_3);
        // r0 = wp.transform_get_rotation(q)                                                      <L 87>
        var_23 = wp::transform_get_rotation(var_3);
        // w0 = wp.spatial_top(qd)                                                                <L 90>
        var_24 = wp::spatial_top(var_6);
        // v0 = wp.spatial_bottom(qd)                                                             <L 91>
        var_25 = wp::spatial_bottom(var_6);
        // t0 = wp.spatial_top(f)                                                                 <L 94>
        var_26 = wp::spatial_top(var_9);
        // f0 = wp.spatial_bottom(f)                                                              <L 95>
        var_27 = wp::spatial_bottom(var_9);
        // x_com = x0 + wp.quat_rotate(r0, body_com[tid])                                         <L 97>
        var_28 = wp::address(var_body_com, var_0);
        var_29 = wp::load(var_28);
        var_30 = wp::quat_rotate(var_23, var_29);
        var_31 = wp::add(var_22, var_30);
        // v1 = v0 + (f0 * inv_mass + gravity * wp.nonzero(inv_mass)) * dt                        <L 100>
        var_32 = wp::mul(var_27, var_15);
        var_33 = wp::nonzero(var_15);
        var_34 = wp::mul(var_gravity, var_33);
        var_35 = wp::add(var_32, var_34);
        var_36 = wp::mul(var_35, var_dt);
        var_37 = wp::add(var_25, var_36);
        // x1 = x_com + v1 * dt                                                                   <L 101>
        var_38 = wp::mul(var_37, var_dt);
        var_39 = wp::add(var_31, var_38);
        // wb = wp.quat_rotate_inv(r0, w0)                                                        <L 104>
        var_40 = wp::quat_rotate_inv(var_23, var_24);
        // tb = wp.quat_rotate_inv(r0, t0) - wp.cross(wb, inertia * wb)  # coriolis forces        <L 105>
        var_41 = wp::quat_rotate_inv(var_23, var_26);
        var_42 = wp::mul(var_18, var_40);
        var_43 = wp::cross(var_40, var_42);
        var_44 = wp::sub(var_41, var_43);
        // w1 = wp.quat_rotate(r0, wb + inv_inertia * tb * dt)                                    <L 107>
        var_45 = wp::mul(var_21, var_44);
        var_46 = wp::mul(var_45, var_dt);
        var_47 = wp::add(var_40, var_46);
        var_48 = wp::quat_rotate(var_23, var_47);
        // r1 = wp.normalize(r0 + wp.quat(w1, 0.0) * r0 * 0.5 * dt)                               <L 108>
        var_50 = wp::quat_t<wp::float32>(var_48, var_49);
        var_51 = wp::mul(var_50, var_23);
        var_53 = wp::mul(var_51, var_52);
        var_54 = wp::mul(var_53, var_dt);
        var_55 = wp::add(var_23, var_54);
        var_56 = wp::normalize(var_55);
        // w1 *= 1.0 - angular_damping * dt                                                       <L 111>
        var_58 = wp::mul(var_angular_damping, var_dt);
        var_59 = wp::sub(var_57, var_58);
        var_60 = wp::mul(var_48, var_59);
        // body_q_new[tid] = wp.transform(x1 - wp.quat_rotate(r1, body_com[tid]), r1)             <L 113>
        var_61 = wp::address(var_body_com, var_0);
        var_62 = wp::load(var_61);
        var_63 = wp::quat_rotate(var_56, var_62);
        var_64 = wp::sub(var_39, var_63);
        var_65 = wp::transform_t<wp::float32>(var_64, var_56);
        wp::array_store(var_body_q_new, var_0, var_65);
        // body_qd_new[tid] = wp.spatial_vector(w1, v1)                                           <L 114>
        var_66 = wp::vec_t<6,wp::float32>(var_60, var_37);
        wp::array_store(var_body_qd_new, var_0, var_66);
    }
}

extern "C" __global__ void integrate_bodies_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::transform_t<wp::float32>> var_body_q,
    wp::array_t<wp::vec_t<6,wp::float32>> var_body_qd,
    wp::array_t<wp::vec_t<6,wp::float32>> var_body_f,
    wp::array_t<wp::vec_t<3,wp::float32>> var_body_com,
    wp::array_t<wp::float32> var_m,
    wp::array_t<wp::mat_t<3,3,wp::float32>> var_I,
    wp::array_t<wp::float32> var_inv_m,
    wp::array_t<wp::mat_t<3,3,wp::float32>> var_inv_I,
    wp::vec_t<3,wp::float32> var_gravity,
    wp::float32 var_angular_damping,
    wp::float32 var_dt,
    wp::array_t<wp::transform_t<wp::float32>> var_body_q_new,
    wp::array_t<wp::vec_t<6,wp::float32>> var_body_qd_new,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_q,
    wp::array_t<wp::vec_t<6,wp::float32>> adj_body_qd,
    wp::array_t<wp::vec_t<6,wp::float32>> adj_body_f,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_body_com,
    wp::array_t<wp::float32> adj_m,
    wp::array_t<wp::mat_t<3,3,wp::float32>> adj_I,
    wp::array_t<wp::float32> adj_inv_m,
    wp::array_t<wp::mat_t<3,3,wp::float32>> adj_inv_I,
    wp::vec_t<3,wp::float32> adj_gravity,
    wp::float32 adj_angular_damping,
    wp::float32 adj_dt,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_q_new,
    wp::array_t<wp::vec_t<6,wp::float32>> adj_body_qd_new)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        wp::transform_t<wp::float32>* var_1;
        wp::transform_t<wp::float32> var_2;
        wp::transform_t<wp::float32> var_3;
        wp::vec_t<6,wp::float32>* var_4;
        wp::vec_t<6,wp::float32> var_5;
        wp::vec_t<6,wp::float32> var_6;
        wp::vec_t<6,wp::float32>* var_7;
        wp::vec_t<6,wp::float32> var_8;
        wp::vec_t<6,wp::float32> var_9;
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        wp::float32* var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        wp::mat_t<3,3,wp::float32>* var_16;
        wp::mat_t<3,3,wp::float32> var_17;
        wp::mat_t<3,3,wp::float32> var_18;
        wp::mat_t<3,3,wp::float32>* var_19;
        wp::mat_t<3,3,wp::float32> var_20;
        wp::mat_t<3,3,wp::float32> var_21;
        wp::vec_t<3,wp::float32> var_22;
        wp::quat_t<wp::float32> var_23;
        wp::vec_t<3,wp::float32> var_24;
        wp::vec_t<3,wp::float32> var_25;
        wp::vec_t<3,wp::float32> var_26;
        wp::vec_t<3,wp::float32> var_27;
        wp::vec_t<3,wp::float32>* var_28;
        wp::vec_t<3,wp::float32> var_29;
        wp::vec_t<3,wp::float32> var_30;
        wp::vec_t<3,wp::float32> var_31;
        wp::vec_t<3,wp::float32> var_32;
        wp::float32 var_33;
        wp::vec_t<3,wp::float32> var_34;
        wp::vec_t<3,wp::float32> var_35;
        wp::vec_t<3,wp::float32> var_36;
        wp::vec_t<3,wp::float32> var_37;
        wp::vec_t<3,wp::float32> var_38;
        wp::vec_t<3,wp::float32> var_39;
        wp::vec_t<3,wp::float32> var_40;
        wp::vec_t<3,wp::float32> var_41;
        wp::vec_t<3,wp::float32> var_42;
        wp::vec_t<3,wp::float32> var_43;
        wp::vec_t<3,wp::float32> var_44;
        wp::vec_t<3,wp::float32> var_45;
        wp::vec_t<3,wp::float32> var_46;
        wp::vec_t<3,wp::float32> var_47;
        wp::vec_t<3,wp::float32> var_48;
        const wp::float32 var_49 = 0.0;
        wp::quat_t<wp::float32> var_50;
        wp::quat_t<wp::float32> var_51;
        const wp::float32 var_52 = 0.5;
        wp::quat_t<wp::float32> var_53;
        wp::quat_t<wp::float32> var_54;
        wp::quat_t<wp::float32> var_55;
        wp::quat_t<wp::float32> var_56;
        const wp::float32 var_57 = 1.0;
        wp::float32 var_58;
        wp::float32 var_59;
        wp::vec_t<3,wp::float32> var_60;
        wp::vec_t<3,wp::float32>* var_61;
        wp::vec_t<3,wp::float32> var_62;
        wp::vec_t<3,wp::float32> var_63;
        wp::vec_t<3,wp::float32> var_64;
        wp::transform_t<wp::float32> var_65;
        wp::vec_t<6,wp::float32> var_66;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::transform_t<wp::float32> adj_1 = {};
        wp::transform_t<wp::float32> adj_2 = {};
        wp::transform_t<wp::float32> adj_3 = {};
        wp::vec_t<6,wp::float32> adj_4 = {};
        wp::vec_t<6,wp::float32> adj_5 = {};
        wp::vec_t<6,wp::float32> adj_6 = {};
        wp::vec_t<6,wp::float32> adj_7 = {};
        wp::vec_t<6,wp::float32> adj_8 = {};
        wp::vec_t<6,wp::float32> adj_9 = {};
        wp::float32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::mat_t<3,3,wp::float32> adj_16 = {};
        wp::mat_t<3,3,wp::float32> adj_17 = {};
        wp::mat_t<3,3,wp::float32> adj_18 = {};
        wp::mat_t<3,3,wp::float32> adj_19 = {};
        wp::mat_t<3,3,wp::float32> adj_20 = {};
        wp::mat_t<3,3,wp::float32> adj_21 = {};
        wp::vec_t<3,wp::float32> adj_22 = {};
        wp::quat_t<wp::float32> adj_23 = {};
        wp::vec_t<3,wp::float32> adj_24 = {};
        wp::vec_t<3,wp::float32> adj_25 = {};
        wp::vec_t<3,wp::float32> adj_26 = {};
        wp::vec_t<3,wp::float32> adj_27 = {};
        wp::vec_t<3,wp::float32> adj_28 = {};
        wp::vec_t<3,wp::float32> adj_29 = {};
        wp::vec_t<3,wp::float32> adj_30 = {};
        wp::vec_t<3,wp::float32> adj_31 = {};
        wp::vec_t<3,wp::float32> adj_32 = {};
        wp::float32 adj_33 = {};
        wp::vec_t<3,wp::float32> adj_34 = {};
        wp::vec_t<3,wp::float32> adj_35 = {};
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
        wp::vec_t<3,wp::float32> adj_46 = {};
        wp::vec_t<3,wp::float32> adj_47 = {};
        wp::vec_t<3,wp::float32> adj_48 = {};
        wp::float32 adj_49 = {};
        wp::quat_t<wp::float32> adj_50 = {};
        wp::quat_t<wp::float32> adj_51 = {};
        wp::float32 adj_52 = {};
        wp::quat_t<wp::float32> adj_53 = {};
        wp::quat_t<wp::float32> adj_54 = {};
        wp::quat_t<wp::float32> adj_55 = {};
        wp::quat_t<wp::float32> adj_56 = {};
        wp::float32 adj_57 = {};
        wp::float32 adj_58 = {};
        wp::float32 adj_59 = {};
        wp::vec_t<3,wp::float32> adj_60 = {};
        wp::vec_t<3,wp::float32> adj_61 = {};
        wp::vec_t<3,wp::float32> adj_62 = {};
        wp::vec_t<3,wp::float32> adj_63 = {};
        wp::vec_t<3,wp::float32> adj_64 = {};
        wp::transform_t<wp::float32> adj_65 = {};
        wp::vec_t<6,wp::float32> adj_66 = {};
        //---------
        // forward
        // def integrate_bodies(                                                                  <L 55>
        // tid = wp.tid()                                                                         <L 71>
        var_0 = builtin_tid1d();
        // q = body_q[tid]                                                                        <L 74>
        var_1 = wp::address(var_body_q, var_0);
        var_2 = wp::load(var_1);
        var_3 = wp::copy(var_2);
        // qd = body_qd[tid]                                                                      <L 75>
        var_4 = wp::address(var_body_qd, var_0);
        var_5 = wp::load(var_4);
        var_6 = wp::copy(var_5);
        // f = body_f[tid]                                                                        <L 76>
        var_7 = wp::address(var_body_f, var_0);
        var_8 = wp::load(var_7);
        var_9 = wp::copy(var_8);
        // mass = m[tid]                                                                          <L 79>
        var_10 = wp::address(var_m, var_0);
        var_11 = wp::load(var_10);
        var_12 = wp::copy(var_11);
        // inv_mass = inv_m[tid]  # 1 / mass                                                      <L 80>
        var_13 = wp::address(var_inv_m, var_0);
        var_14 = wp::load(var_13);
        var_15 = wp::copy(var_14);
        // inertia = I[tid]                                                                       <L 82>
        var_16 = wp::address(var_I, var_0);
        var_17 = wp::load(var_16);
        var_18 = wp::copy(var_17);
        // inv_inertia = inv_I[tid]  # inverse of 3x3 inertia matrix                              <L 83>
        var_19 = wp::address(var_inv_I, var_0);
        var_20 = wp::load(var_19);
        var_21 = wp::copy(var_20);
        // x0 = wp.transform_get_translation(q)                                                   <L 86>
        var_22 = wp::transform_get_translation(var_3);
        // r0 = wp.transform_get_rotation(q)                                                      <L 87>
        var_23 = wp::transform_get_rotation(var_3);
        // w0 = wp.spatial_top(qd)                                                                <L 90>
        var_24 = wp::spatial_top(var_6);
        // v0 = wp.spatial_bottom(qd)                                                             <L 91>
        var_25 = wp::spatial_bottom(var_6);
        // t0 = wp.spatial_top(f)                                                                 <L 94>
        var_26 = wp::spatial_top(var_9);
        // f0 = wp.spatial_bottom(f)                                                              <L 95>
        var_27 = wp::spatial_bottom(var_9);
        // x_com = x0 + wp.quat_rotate(r0, body_com[tid])                                         <L 97>
        var_28 = wp::address(var_body_com, var_0);
        var_29 = wp::load(var_28);
        var_30 = wp::quat_rotate(var_23, var_29);
        var_31 = wp::add(var_22, var_30);
        // v1 = v0 + (f0 * inv_mass + gravity * wp.nonzero(inv_mass)) * dt                        <L 100>
        var_32 = wp::mul(var_27, var_15);
        var_33 = wp::nonzero(var_15);
        var_34 = wp::mul(var_gravity, var_33);
        var_35 = wp::add(var_32, var_34);
        var_36 = wp::mul(var_35, var_dt);
        var_37 = wp::add(var_25, var_36);
        // x1 = x_com + v1 * dt                                                                   <L 101>
        var_38 = wp::mul(var_37, var_dt);
        var_39 = wp::add(var_31, var_38);
        // wb = wp.quat_rotate_inv(r0, w0)                                                        <L 104>
        var_40 = wp::quat_rotate_inv(var_23, var_24);
        // tb = wp.quat_rotate_inv(r0, t0) - wp.cross(wb, inertia * wb)  # coriolis forces        <L 105>
        var_41 = wp::quat_rotate_inv(var_23, var_26);
        var_42 = wp::mul(var_18, var_40);
        var_43 = wp::cross(var_40, var_42);
        var_44 = wp::sub(var_41, var_43);
        // w1 = wp.quat_rotate(r0, wb + inv_inertia * tb * dt)                                    <L 107>
        var_45 = wp::mul(var_21, var_44);
        var_46 = wp::mul(var_45, var_dt);
        var_47 = wp::add(var_40, var_46);
        var_48 = wp::quat_rotate(var_23, var_47);
        // r1 = wp.normalize(r0 + wp.quat(w1, 0.0) * r0 * 0.5 * dt)                               <L 108>
        var_50 = wp::quat_t<wp::float32>(var_48, var_49);
        var_51 = wp::mul(var_50, var_23);
        var_53 = wp::mul(var_51, var_52);
        var_54 = wp::mul(var_53, var_dt);
        var_55 = wp::add(var_23, var_54);
        var_56 = wp::normalize(var_55);
        // w1 *= 1.0 - angular_damping * dt                                                       <L 111>
        var_58 = wp::mul(var_angular_damping, var_dt);
        var_59 = wp::sub(var_57, var_58);
        var_60 = wp::mul(var_48, var_59);
        // body_q_new[tid] = wp.transform(x1 - wp.quat_rotate(r1, body_com[tid]), r1)             <L 113>
        var_61 = wp::address(var_body_com, var_0);
        var_62 = wp::load(var_61);
        var_63 = wp::quat_rotate(var_56, var_62);
        var_64 = wp::sub(var_39, var_63);
        var_65 = wp::transform_t<wp::float32>(var_64, var_56);
        // wp::array_store(var_body_q_new, var_0, var_65);
        // body_qd_new[tid] = wp.spatial_vector(w1, v1)                                           <L 114>
        var_66 = wp::vec_t<6,wp::float32>(var_60, var_37);
        // wp::array_store(var_body_qd_new, var_0, var_66);
        //---------
        // reverse
        wp::adj_array_store(var_body_qd_new, var_0, var_66, adj_body_qd_new, adj_0, adj_66);
        wp::adj_vec_t(var_60, var_37, adj_60, adj_37, adj_66);
        // adj: body_qd_new[tid] = wp.spatial_vector(w1, v1)                                      <L 114>
        wp::adj_array_store(var_body_q_new, var_0, var_65, adj_body_q_new, adj_0, adj_65);
        wp::adj_transform_t(var_64, var_56, adj_64, adj_56, adj_65);
        wp::adj_sub(var_39, var_63, adj_39, adj_63, adj_64);
        wp::adj_quat_rotate(var_56, var_62, adj_56, adj_61, adj_63);
        wp::adj_load(var_61, adj_61, adj_62);
        wp::adj_address(var_body_com, var_0, adj_body_com, adj_0, adj_61);
        // adj: body_q_new[tid] = wp.transform(x1 - wp.quat_rotate(r1, body_com[tid]), r1)        <L 113>
        wp::adj_mul(var_48, var_59, adj_48, adj_59, adj_60);
        wp::adj_sub(var_57, var_58, adj_57, adj_58, adj_59);
        wp::adj_mul(var_angular_damping, var_dt, adj_angular_damping, adj_dt, adj_58);
        // adj: w1 *= 1.0 - angular_damping * dt                                                  <L 111>
        wp::adj_normalize(var_55, adj_55, adj_56);
        wp::adj_add(var_23, var_54, adj_23, adj_54, adj_55);
        wp::adj_mul(var_53, var_dt, adj_53, adj_dt, adj_54);
        wp::adj_mul(var_51, var_52, adj_51, adj_52, adj_53);
        wp::adj_mul(var_50, var_23, adj_50, adj_23, adj_51);
        wp::adj_quat_t(var_48, var_49, adj_48, adj_49, adj_50);
        // adj: r1 = wp.normalize(r0 + wp.quat(w1, 0.0) * r0 * 0.5 * dt)                          <L 108>
        wp::adj_quat_rotate(var_23, var_47, adj_23, adj_47, adj_48);
        wp::adj_add(var_40, var_46, adj_40, adj_46, adj_47);
        wp::adj_mul(var_45, var_dt, adj_45, adj_dt, adj_46);
        wp::adj_mul(var_21, var_44, adj_21, adj_44, adj_45);
        // adj: w1 = wp.quat_rotate(r0, wb + inv_inertia * tb * dt)                               <L 107>
        wp::adj_sub(var_41, var_43, adj_41, adj_43, adj_44);
        wp::adj_cross(var_40, var_42, adj_40, adj_42, adj_43);
        wp::adj_mul(var_18, var_40, adj_18, adj_40, adj_42);
        wp::adj_quat_rotate_inv(var_23, var_26, adj_23, adj_26, adj_41);
        // adj: tb = wp.quat_rotate_inv(r0, t0) - wp.cross(wb, inertia * wb)  # coriolis forces   <L 105>
        wp::adj_quat_rotate_inv(var_23, var_24, adj_23, adj_24, adj_40);
        // adj: wb = wp.quat_rotate_inv(r0, w0)                                                   <L 104>
        wp::adj_add(var_31, var_38, adj_31, adj_38, adj_39);
        wp::adj_mul(var_37, var_dt, adj_37, adj_dt, adj_38);
        // adj: x1 = x_com + v1 * dt                                                              <L 101>
        wp::adj_add(var_25, var_36, adj_25, adj_36, adj_37);
        wp::adj_mul(var_35, var_dt, adj_35, adj_dt, adj_36);
        wp::adj_add(var_32, var_34, adj_32, adj_34, adj_35);
        wp::adj_mul(var_gravity, var_33, adj_gravity, adj_33, adj_34);
        wp::adj_nonzero(var_15, adj_15, adj_33);
        wp::adj_mul(var_27, var_15, adj_27, adj_15, adj_32);
        // adj: v1 = v0 + (f0 * inv_mass + gravity * wp.nonzero(inv_mass)) * dt                   <L 100>
        wp::adj_add(var_22, var_30, adj_22, adj_30, adj_31);
        wp::adj_quat_rotate(var_23, var_29, adj_23, adj_28, adj_30);
        wp::adj_load(var_28, adj_28, adj_29);
        wp::adj_address(var_body_com, var_0, adj_body_com, adj_0, adj_28);
        // adj: x_com = x0 + wp.quat_rotate(r0, body_com[tid])                                    <L 97>
        wp::adj_spatial_bottom(var_9, adj_9, adj_27);
        // adj: f0 = wp.spatial_bottom(f)                                                         <L 95>
        wp::adj_spatial_top(var_9, adj_9, adj_26);
        // adj: t0 = wp.spatial_top(f)                                                            <L 94>
        wp::adj_spatial_bottom(var_6, adj_6, adj_25);
        // adj: v0 = wp.spatial_bottom(qd)                                                        <L 91>
        wp::adj_spatial_top(var_6, adj_6, adj_24);
        // adj: w0 = wp.spatial_top(qd)                                                           <L 90>
        wp::adj_transform_get_rotation(var_3, adj_3, adj_23);
        // adj: r0 = wp.transform_get_rotation(q)                                                 <L 87>
        wp::adj_transform_get_translation(var_3, adj_3, adj_22);
        // adj: x0 = wp.transform_get_translation(q)                                              <L 86>
        wp::adj_copy(var_20, adj_19, adj_21);
        wp::adj_load(var_19, adj_19, adj_20);
        wp::adj_address(var_inv_I, var_0, adj_inv_I, adj_0, adj_19);
        // adj: inv_inertia = inv_I[tid]  # inverse of 3x3 inertia matrix                         <L 83>
        wp::adj_copy(var_17, adj_16, adj_18);
        wp::adj_load(var_16, adj_16, adj_17);
        wp::adj_address(var_I, var_0, adj_I, adj_0, adj_16);
        // adj: inertia = I[tid]                                                                  <L 82>
        wp::adj_copy(var_14, adj_13, adj_15);
        wp::adj_load(var_13, adj_13, adj_14);
        wp::adj_address(var_inv_m, var_0, adj_inv_m, adj_0, adj_13);
        // adj: inv_mass = inv_m[tid]  # 1 / mass                                                 <L 80>
        wp::adj_copy(var_11, adj_10, adj_12);
        wp::adj_load(var_10, adj_10, adj_11);
        wp::adj_address(var_m, var_0, adj_m, adj_0, adj_10);
        // adj: mass = m[tid]                                                                     <L 79>
        wp::adj_copy(var_8, adj_7, adj_9);
        wp::adj_load(var_7, adj_7, adj_8);
        wp::adj_address(var_body_f, var_0, adj_body_f, adj_0, adj_7);
        // adj: f = body_f[tid]                                                                   <L 76>
        wp::adj_copy(var_5, adj_4, adj_6);
        wp::adj_load(var_4, adj_4, adj_5);
        wp::adj_address(var_body_qd, var_0, adj_body_qd, adj_0, adj_4);
        // adj: qd = body_qd[tid]                                                                 <L 75>
        wp::adj_copy(var_2, adj_1, adj_3);
        wp::adj_load(var_1, adj_1, adj_2);
        wp::adj_address(var_body_q, var_0, adj_body_q, adj_0, adj_1);
        // adj: q = body_q[tid]                                                                   <L 74>
        // adj: tid = wp.tid()                                                                    <L 71>
        // adj: def integrate_bodies(                                                             <L 55>
        continue;
    }
}



extern "C" __global__ void eval_springs_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float32>> var_x,
    wp::array_t<wp::vec_t<3,wp::float32>> var_v,
    wp::array_t<wp::int32> var_spring_indices,
    wp::array_t<wp::float32> var_spring_rest_lengths,
    wp::array_t<wp::float32> var_spring_stiffness,
    wp::array_t<wp::float32> var_spring_damping,
    wp::array_t<wp::vec_t<3,wp::float32>> var_f)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        const wp::int32 var_1 = 2;
        wp::int32 var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        const wp::int32 var_9 = 1;
        wp::int32 var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        wp::float32* var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32* var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::vec_t<3,wp::float32>* var_23;
        wp::vec_t<3,wp::float32> var_24;
        wp::vec_t<3,wp::float32> var_25;
        wp::vec_t<3,wp::float32>* var_26;
        wp::vec_t<3,wp::float32> var_27;
        wp::vec_t<3,wp::float32> var_28;
        wp::vec_t<3,wp::float32>* var_29;
        wp::vec_t<3,wp::float32> var_30;
        wp::vec_t<3,wp::float32> var_31;
        wp::vec_t<3,wp::float32>* var_32;
        wp::vec_t<3,wp::float32> var_33;
        wp::vec_t<3,wp::float32> var_34;
        wp::vec_t<3,wp::float32> var_35;
        wp::vec_t<3,wp::float32> var_36;
        wp::float32 var_37;
        const wp::float32 var_38 = 1.0;
        wp::float32 var_39;
        wp::vec_t<3,wp::float32> var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        wp::float32 var_45;
        wp::vec_t<3,wp::float32> var_46;
        wp::vec_t<3,wp::float32> var_47;
        wp::vec_t<3,wp::float32> var_48;
        //---------
        // forward
        // def eval_springs(                                                                      <L 118>
        // tid = wp.tid()                                                                         <L 127>
        var_0 = builtin_tid1d();
        // i = spring_indices[tid * 2 + 0]                                                        <L 129>
        var_2 = wp::mul(var_0, var_1);
        var_4 = wp::add(var_2, var_3);
        var_5 = wp::address(var_spring_indices, var_4);
        var_6 = wp::load(var_5);
        var_7 = wp::copy(var_6);
        // j = spring_indices[tid * 2 + 1]                                                        <L 130>
        var_8 = wp::mul(var_0, var_1);
        var_10 = wp::add(var_8, var_9);
        var_11 = wp::address(var_spring_indices, var_10);
        var_12 = wp::load(var_11);
        var_13 = wp::copy(var_12);
        // ke = spring_stiffness[tid]                                                             <L 132>
        var_14 = wp::address(var_spring_stiffness, var_0);
        var_15 = wp::load(var_14);
        var_16 = wp::copy(var_15);
        // kd = spring_damping[tid]                                                               <L 133>
        var_17 = wp::address(var_spring_damping, var_0);
        var_18 = wp::load(var_17);
        var_19 = wp::copy(var_18);
        // rest = spring_rest_lengths[tid]                                                        <L 134>
        var_20 = wp::address(var_spring_rest_lengths, var_0);
        var_21 = wp::load(var_20);
        var_22 = wp::copy(var_21);
        // xi = x[i]                                                                              <L 136>
        var_23 = wp::address(var_x, var_7);
        var_24 = wp::load(var_23);
        var_25 = wp::copy(var_24);
        // xj = x[j]                                                                              <L 137>
        var_26 = wp::address(var_x, var_13);
        var_27 = wp::load(var_26);
        var_28 = wp::copy(var_27);
        // vi = v[i]                                                                              <L 139>
        var_29 = wp::address(var_v, var_7);
        var_30 = wp::load(var_29);
        var_31 = wp::copy(var_30);
        // vj = v[j]                                                                              <L 140>
        var_32 = wp::address(var_v, var_13);
        var_33 = wp::load(var_32);
        var_34 = wp::copy(var_33);
        // xij = xi - xj                                                                          <L 142>
        var_35 = wp::sub(var_25, var_28);
        // vij = vi - vj                                                                          <L 143>
        var_36 = wp::sub(var_31, var_34);
        // l = wp.length(xij)                                                                     <L 145>
        var_37 = wp::length(var_35);
        // l_inv = 1.0 / l                                                                        <L 146>
        var_39 = wp::div(var_38, var_37);
        // dir = xij * l_inv                                                                      <L 149>
        var_40 = wp::mul(var_35, var_39);
        // c = l - rest                                                                           <L 151>
        var_41 = wp::sub(var_37, var_22);
        // dcdt = wp.dot(dir, vij)                                                                <L 152>
        var_42 = wp::dot(var_40, var_36);
        // fs = dir * (ke * c + kd * dcdt)                                                        <L 155>
        var_43 = wp::mul(var_16, var_41);
        var_44 = wp::mul(var_19, var_42);
        var_45 = wp::add(var_43, var_44);
        var_46 = wp::mul(var_40, var_45);
        // wp.atomic_sub(f, i, fs)                                                                <L 157>
        var_47 = wp::atomic_sub(var_f, var_7, var_46);
        // wp.atomic_add(f, j, fs)                                                                <L 158>
        var_48 = wp::atomic_add(var_f, var_13, var_46);
    }
}

extern "C" __global__ void eval_springs_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float32>> var_x,
    wp::array_t<wp::vec_t<3,wp::float32>> var_v,
    wp::array_t<wp::int32> var_spring_indices,
    wp::array_t<wp::float32> var_spring_rest_lengths,
    wp::array_t<wp::float32> var_spring_stiffness,
    wp::array_t<wp::float32> var_spring_damping,
    wp::array_t<wp::vec_t<3,wp::float32>> var_f,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_x,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_v,
    wp::array_t<wp::int32> adj_spring_indices,
    wp::array_t<wp::float32> adj_spring_rest_lengths,
    wp::array_t<wp::float32> adj_spring_stiffness,
    wp::array_t<wp::float32> adj_spring_damping,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_f)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        const wp::int32 var_1 = 2;
        wp::int32 var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        const wp::int32 var_9 = 1;
        wp::int32 var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        wp::float32* var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32* var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::vec_t<3,wp::float32>* var_23;
        wp::vec_t<3,wp::float32> var_24;
        wp::vec_t<3,wp::float32> var_25;
        wp::vec_t<3,wp::float32>* var_26;
        wp::vec_t<3,wp::float32> var_27;
        wp::vec_t<3,wp::float32> var_28;
        wp::vec_t<3,wp::float32>* var_29;
        wp::vec_t<3,wp::float32> var_30;
        wp::vec_t<3,wp::float32> var_31;
        wp::vec_t<3,wp::float32>* var_32;
        wp::vec_t<3,wp::float32> var_33;
        wp::vec_t<3,wp::float32> var_34;
        wp::vec_t<3,wp::float32> var_35;
        wp::vec_t<3,wp::float32> var_36;
        wp::float32 var_37;
        const wp::float32 var_38 = 1.0;
        wp::float32 var_39;
        wp::vec_t<3,wp::float32> var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        wp::float32 var_45;
        wp::vec_t<3,wp::float32> var_46;
        wp::vec_t<3,wp::float32> var_47;
        wp::vec_t<3,wp::float32> var_48;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::float32 adj_16 = {};
        wp::float32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::vec_t<3,wp::float32> adj_23 = {};
        wp::vec_t<3,wp::float32> adj_24 = {};
        wp::vec_t<3,wp::float32> adj_25 = {};
        wp::vec_t<3,wp::float32> adj_26 = {};
        wp::vec_t<3,wp::float32> adj_27 = {};
        wp::vec_t<3,wp::float32> adj_28 = {};
        wp::vec_t<3,wp::float32> adj_29 = {};
        wp::vec_t<3,wp::float32> adj_30 = {};
        wp::vec_t<3,wp::float32> adj_31 = {};
        wp::vec_t<3,wp::float32> adj_32 = {};
        wp::vec_t<3,wp::float32> adj_33 = {};
        wp::vec_t<3,wp::float32> adj_34 = {};
        wp::vec_t<3,wp::float32> adj_35 = {};
        wp::vec_t<3,wp::float32> adj_36 = {};
        wp::float32 adj_37 = {};
        wp::float32 adj_38 = {};
        wp::float32 adj_39 = {};
        wp::vec_t<3,wp::float32> adj_40 = {};
        wp::float32 adj_41 = {};
        wp::float32 adj_42 = {};
        wp::float32 adj_43 = {};
        wp::float32 adj_44 = {};
        wp::float32 adj_45 = {};
        wp::vec_t<3,wp::float32> adj_46 = {};
        wp::vec_t<3,wp::float32> adj_47 = {};
        wp::vec_t<3,wp::float32> adj_48 = {};
        //---------
        // forward
        // def eval_springs(                                                                      <L 118>
        // tid = wp.tid()                                                                         <L 127>
        var_0 = builtin_tid1d();
        // i = spring_indices[tid * 2 + 0]                                                        <L 129>
        var_2 = wp::mul(var_0, var_1);
        var_4 = wp::add(var_2, var_3);
        var_5 = wp::address(var_spring_indices, var_4);
        var_6 = wp::load(var_5);
        var_7 = wp::copy(var_6);
        // j = spring_indices[tid * 2 + 1]                                                        <L 130>
        var_8 = wp::mul(var_0, var_1);
        var_10 = wp::add(var_8, var_9);
        var_11 = wp::address(var_spring_indices, var_10);
        var_12 = wp::load(var_11);
        var_13 = wp::copy(var_12);
        // ke = spring_stiffness[tid]                                                             <L 132>
        var_14 = wp::address(var_spring_stiffness, var_0);
        var_15 = wp::load(var_14);
        var_16 = wp::copy(var_15);
        // kd = spring_damping[tid]                                                               <L 133>
        var_17 = wp::address(var_spring_damping, var_0);
        var_18 = wp::load(var_17);
        var_19 = wp::copy(var_18);
        // rest = spring_rest_lengths[tid]                                                        <L 134>
        var_20 = wp::address(var_spring_rest_lengths, var_0);
        var_21 = wp::load(var_20);
        var_22 = wp::copy(var_21);
        // xi = x[i]                                                                              <L 136>
        var_23 = wp::address(var_x, var_7);
        var_24 = wp::load(var_23);
        var_25 = wp::copy(var_24);
        // xj = x[j]                                                                              <L 137>
        var_26 = wp::address(var_x, var_13);
        var_27 = wp::load(var_26);
        var_28 = wp::copy(var_27);
        // vi = v[i]                                                                              <L 139>
        var_29 = wp::address(var_v, var_7);
        var_30 = wp::load(var_29);
        var_31 = wp::copy(var_30);
        // vj = v[j]                                                                              <L 140>
        var_32 = wp::address(var_v, var_13);
        var_33 = wp::load(var_32);
        var_34 = wp::copy(var_33);
        // xij = xi - xj                                                                          <L 142>
        var_35 = wp::sub(var_25, var_28);
        // vij = vi - vj                                                                          <L 143>
        var_36 = wp::sub(var_31, var_34);
        // l = wp.length(xij)                                                                     <L 145>
        var_37 = wp::length(var_35);
        // l_inv = 1.0 / l                                                                        <L 146>
        var_39 = wp::div(var_38, var_37);
        // dir = xij * l_inv                                                                      <L 149>
        var_40 = wp::mul(var_35, var_39);
        // c = l - rest                                                                           <L 151>
        var_41 = wp::sub(var_37, var_22);
        // dcdt = wp.dot(dir, vij)                                                                <L 152>
        var_42 = wp::dot(var_40, var_36);
        // fs = dir * (ke * c + kd * dcdt)                                                        <L 155>
        var_43 = wp::mul(var_16, var_41);
        var_44 = wp::mul(var_19, var_42);
        var_45 = wp::add(var_43, var_44);
        var_46 = wp::mul(var_40, var_45);
        // wp.atomic_sub(f, i, fs)                                                                <L 157>
        // var_47 = wp::atomic_sub(var_f, var_7, var_46);
        // wp.atomic_add(f, j, fs)                                                                <L 158>
        // var_48 = wp::atomic_add(var_f, var_13, var_46);
        //---------
        // reverse
        wp::adj_atomic_add(var_f, var_13, var_46, adj_f, adj_13, adj_46, adj_48);
        // adj: wp.atomic_add(f, j, fs)                                                           <L 158>
        wp::adj_atomic_sub(var_f, var_7, var_46, adj_f, adj_7, adj_46, adj_47);
        // adj: wp.atomic_sub(f, i, fs)                                                           <L 157>
        wp::adj_mul(var_40, var_45, adj_40, adj_45, adj_46);
        wp::adj_add(var_43, var_44, adj_43, adj_44, adj_45);
        wp::adj_mul(var_19, var_42, adj_19, adj_42, adj_44);
        wp::adj_mul(var_16, var_41, adj_16, adj_41, adj_43);
        // adj: fs = dir * (ke * c + kd * dcdt)                                                   <L 155>
        wp::adj_dot(var_40, var_36, adj_40, adj_36, adj_42);
        // adj: dcdt = wp.dot(dir, vij)                                                           <L 152>
        wp::adj_sub(var_37, var_22, adj_37, adj_22, adj_41);
        // adj: c = l - rest                                                                      <L 151>
        wp::adj_mul(var_35, var_39, adj_35, adj_39, adj_40);
        // adj: dir = xij * l_inv                                                                 <L 149>
        wp::adj_div(var_38, var_37, var_39, adj_38, adj_37, adj_39);
        // adj: l_inv = 1.0 / l                                                                   <L 146>
        wp::adj_length(var_35, var_37, adj_35, adj_37);
        // adj: l = wp.length(xij)                                                                <L 145>
        wp::adj_sub(var_31, var_34, adj_31, adj_34, adj_36);
        // adj: vij = vi - vj                                                                     <L 143>
        wp::adj_sub(var_25, var_28, adj_25, adj_28, adj_35);
        // adj: xij = xi - xj                                                                     <L 142>
        wp::adj_copy(var_33, adj_32, adj_34);
        wp::adj_load(var_32, adj_32, adj_33);
        wp::adj_address(var_v, var_13, adj_v, adj_13, adj_32);
        // adj: vj = v[j]                                                                         <L 140>
        wp::adj_copy(var_30, adj_29, adj_31);
        wp::adj_load(var_29, adj_29, adj_30);
        wp::adj_address(var_v, var_7, adj_v, adj_7, adj_29);
        // adj: vi = v[i]                                                                         <L 139>
        wp::adj_copy(var_27, adj_26, adj_28);
        wp::adj_load(var_26, adj_26, adj_27);
        wp::adj_address(var_x, var_13, adj_x, adj_13, adj_26);
        // adj: xj = x[j]                                                                         <L 137>
        wp::adj_copy(var_24, adj_23, adj_25);
        wp::adj_load(var_23, adj_23, adj_24);
        wp::adj_address(var_x, var_7, adj_x, adj_7, adj_23);
        // adj: xi = x[i]                                                                         <L 136>
        wp::adj_copy(var_21, adj_20, adj_22);
        wp::adj_load(var_20, adj_20, adj_21);
        wp::adj_address(var_spring_rest_lengths, var_0, adj_spring_rest_lengths, adj_0, adj_20);
        // adj: rest = spring_rest_lengths[tid]                                                   <L 134>
        wp::adj_copy(var_18, adj_17, adj_19);
        wp::adj_load(var_17, adj_17, adj_18);
        wp::adj_address(var_spring_damping, var_0, adj_spring_damping, adj_0, adj_17);
        // adj: kd = spring_damping[tid]                                                          <L 133>
        wp::adj_copy(var_15, adj_14, adj_16);
        wp::adj_load(var_14, adj_14, adj_15);
        wp::adj_address(var_spring_stiffness, var_0, adj_spring_stiffness, adj_0, adj_14);
        // adj: ke = spring_stiffness[tid]                                                        <L 132>
        wp::adj_copy(var_12, adj_11, adj_13);
        wp::adj_load(var_11, adj_11, adj_12);
        wp::adj_address(var_spring_indices, var_10, adj_spring_indices, adj_10, adj_11);
        wp::adj_add(var_8, var_9, adj_8, adj_9, adj_10);
        wp::adj_mul(var_0, var_1, adj_0, adj_1, adj_8);
        // adj: j = spring_indices[tid * 2 + 1]                                                   <L 130>
        wp::adj_copy(var_6, adj_5, adj_7);
        wp::adj_load(var_5, adj_5, adj_6);
        wp::adj_address(var_spring_indices, var_4, adj_spring_indices, adj_4, adj_5);
        wp::adj_add(var_2, var_3, adj_2, adj_3, adj_4);
        wp::adj_mul(var_0, var_1, adj_0, adj_1, adj_2);
        // adj: i = spring_indices[tid * 2 + 0]                                                   <L 129>
        // adj: tid = wp.tid()                                                                    <L 127>
        // adj: def eval_springs(                                                                 <L 118>
        continue;
    }
}



extern "C" __global__ void eval_triangles_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float32>> var_x,
    wp::array_t<wp::vec_t<3,wp::float32>> var_v,
    wp::array_t<wp::int32> var_indices,
    wp::array_t<wp::mat_t<2,2,wp::float32>> var_pose,
    wp::array_t<wp::float32> var_activation,
    wp::array_t<wp::float32> var_materials,
    wp::array_t<wp::vec_t<3,wp::float32>> var_f)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        const wp::int32 var_1 = 0;
        wp::float32* var_2;
        wp::float32 var_3;
        wp::float32 var_4;
        const wp::int32 var_5 = 1;
        wp::float32* var_6;
        wp::float32 var_7;
        wp::float32 var_8;
        const wp::int32 var_9 = 2;
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        const wp::int32 var_13 = 3;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        const wp::int32 var_17 = 4;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::int32* var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        wp::int32* var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        wp::vec_t<3,wp::float32>* var_30;
        wp::vec_t<3,wp::float32> var_31;
        wp::vec_t<3,wp::float32> var_32;
        wp::vec_t<3,wp::float32>* var_33;
        wp::vec_t<3,wp::float32> var_34;
        wp::vec_t<3,wp::float32> var_35;
        wp::vec_t<3,wp::float32>* var_36;
        wp::vec_t<3,wp::float32> var_37;
        wp::vec_t<3,wp::float32> var_38;
        wp::vec_t<3,wp::float32>* var_39;
        wp::vec_t<3,wp::float32> var_40;
        wp::vec_t<3,wp::float32> var_41;
        wp::vec_t<3,wp::float32>* var_42;
        wp::vec_t<3,wp::float32> var_43;
        wp::vec_t<3,wp::float32> var_44;
        wp::vec_t<3,wp::float32>* var_45;
        wp::vec_t<3,wp::float32> var_46;
        wp::vec_t<3,wp::float32> var_47;
        wp::vec_t<3,wp::float32> var_48;
        wp::vec_t<3,wp::float32> var_49;
        wp::vec_t<3,wp::float32> var_50;
        wp::vec_t<3,wp::float32> var_51;
        wp::mat_t<2,2,wp::float32>* var_52;
        wp::mat_t<2,2,wp::float32> var_53;
        wp::mat_t<2,2,wp::float32> var_54;
        wp::float32 var_55;
        const wp::float32 var_56 = 2.0;
        wp::float32 var_57;
        const wp::float32 var_58 = 1.0;
        wp::float32 var_59;
        wp::float32 var_60;
        wp::float32 var_61;
        wp::float32 var_62;
        wp::float32 var_63;
        wp::vec_t<3,wp::float32> var_64;
        wp::float32 var_65;
        wp::vec_t<3,wp::float32> var_66;
        wp::vec_t<3,wp::float32> var_67;
        wp::float32 var_68;
        wp::vec_t<3,wp::float32> var_69;
        wp::float32 var_70;
        wp::vec_t<3,wp::float32> var_71;
        wp::vec_t<3,wp::float32> var_72;
        wp::float32 var_73;
        wp::vec_t<3,wp::float32> var_74;
        wp::float32 var_75;
        wp::vec_t<3,wp::float32> var_76;
        wp::vec_t<3,wp::float32> var_77;
        wp::float32 var_78;
        wp::vec_t<3,wp::float32> var_79;
        wp::float32 var_80;
        wp::vec_t<3,wp::float32> var_81;
        wp::vec_t<3,wp::float32> var_82;
        wp::vec_t<3,wp::float32> var_83;
        wp::vec_t<3,wp::float32> var_84;
        wp::vec_t<3,wp::float32> var_85;
        wp::vec_t<3,wp::float32> var_86;
        wp::vec_t<3,wp::float32> var_87;
        wp::vec_t<3,wp::float32> var_88;
        wp::float32 var_89;
        wp::vec_t<3,wp::float32> var_90;
        wp::float32 var_91;
        wp::vec_t<3,wp::float32> var_92;
        wp::vec_t<3,wp::float32> var_93;
        wp::float32 var_94;
        wp::vec_t<3,wp::float32> var_95;
        wp::float32 var_96;
        wp::vec_t<3,wp::float32> var_97;
        wp::vec_t<3,wp::float32> var_98;
        wp::float32 var_99;
        wp::float32 var_100;
        wp::vec_t<3,wp::float32> var_101;
        wp::float32 var_102;
        const wp::float32 var_103 = 0.5;
        wp::float32 var_104;
        wp::float32* var_105;
        wp::float32 var_106;
        wp::float32 var_107;
        wp::float32 var_108;
        wp::float32 var_109;
        wp::float32 var_110;
        wp::vec_t<3,wp::float32> var_111;
        wp::vec_t<3,wp::float32> var_112;
        wp::vec_t<3,wp::float32> var_113;
        wp::vec_t<3,wp::float32> var_114;
        wp::vec_t<3,wp::float32> var_115;
        wp::vec_t<3,wp::float32> var_116;
        wp::vec_t<3,wp::float32> var_117;
        wp::float32 var_118;
        wp::float32 var_119;
        wp::float32 var_120;
        wp::float32 var_121;
        wp::vec_t<3,wp::float32> var_122;
        wp::float32 var_123;
        wp::float32 var_124;
        wp::float32 var_125;
        wp::float32 var_126;
        wp::vec_t<3,wp::float32> var_127;
        wp::vec_t<3,wp::float32> var_128;
        wp::float32 var_129;
        wp::vec_t<3,wp::float32> var_130;
        wp::vec_t<3,wp::float32> var_131;
        wp::vec_t<3,wp::float32> var_132;
        wp::vec_t<3,wp::float32> var_133;
        wp::vec_t<3,wp::float32> var_134;
        const wp::float32 var_135 = 0.3333;
        wp::vec_t<3,wp::float32> var_136;
        wp::vec_t<3,wp::float32> var_137;
        wp::float32 var_138;
        wp::float32 var_139;
        wp::float32 var_140;
        wp::float32 var_141;
        wp::vec_t<3,wp::float32> var_142;
        wp::float32 var_143;
        const wp::float32 var_144 = 1.57079;
        wp::float32 var_145;
        wp::float32 var_146;
        wp::float32 var_147;
        wp::float32 var_148;
        wp::vec_t<3,wp::float32> var_149;
        wp::float32 var_150;
        wp::vec_t<3,wp::float32> var_151;
        wp::vec_t<3,wp::float32> var_152;
        wp::vec_t<3,wp::float32> var_153;
        wp::vec_t<3,wp::float32> var_154;
        wp::vec_t<3,wp::float32> var_155;
        wp::vec_t<3,wp::float32> var_156;
        wp::vec_t<3,wp::float32> var_157;
        wp::vec_t<3,wp::float32> var_158;
        wp::vec_t<3,wp::float32> var_159;
        wp::vec_t<3,wp::float32> var_160;
        //---------
        // forward
        // def eval_triangles(                                                                    <L 162>
        // tid = wp.tid()                                                                         <L 171>
        var_0 = builtin_tid1d();
        // k_mu = materials[tid, 0]                                                               <L 173>
        var_2 = wp::address(var_materials, var_0, var_1);
        var_3 = wp::load(var_2);
        var_4 = wp::copy(var_3);
        // k_lambda = materials[tid, 1]                                                           <L 174>
        var_6 = wp::address(var_materials, var_0, var_5);
        var_7 = wp::load(var_6);
        var_8 = wp::copy(var_7);
        // k_damp = materials[tid, 2]                                                             <L 175>
        var_10 = wp::address(var_materials, var_0, var_9);
        var_11 = wp::load(var_10);
        var_12 = wp::copy(var_11);
        // k_drag = materials[tid, 3]                                                             <L 176>
        var_14 = wp::address(var_materials, var_0, var_13);
        var_15 = wp::load(var_14);
        var_16 = wp::copy(var_15);
        // k_lift = materials[tid, 4]                                                             <L 177>
        var_18 = wp::address(var_materials, var_0, var_17);
        var_19 = wp::load(var_18);
        var_20 = wp::copy(var_19);
        // i = indices[tid, 0]                                                                    <L 179>
        var_21 = wp::address(var_indices, var_0, var_1);
        var_22 = wp::load(var_21);
        var_23 = wp::copy(var_22);
        // j = indices[tid, 1]                                                                    <L 180>
        var_24 = wp::address(var_indices, var_0, var_5);
        var_25 = wp::load(var_24);
        var_26 = wp::copy(var_25);
        // k = indices[tid, 2]                                                                    <L 181>
        var_27 = wp::address(var_indices, var_0, var_9);
        var_28 = wp::load(var_27);
        var_29 = wp::copy(var_28);
        // x0 = x[i]  # point zero                                                                <L 183>
        var_30 = wp::address(var_x, var_23);
        var_31 = wp::load(var_30);
        var_32 = wp::copy(var_31);
        // x1 = x[j]  # point one                                                                 <L 184>
        var_33 = wp::address(var_x, var_26);
        var_34 = wp::load(var_33);
        var_35 = wp::copy(var_34);
        // x2 = x[k]  # point two                                                                 <L 185>
        var_36 = wp::address(var_x, var_29);
        var_37 = wp::load(var_36);
        var_38 = wp::copy(var_37);
        // v0 = v[i]  # vel zero                                                                  <L 187>
        var_39 = wp::address(var_v, var_23);
        var_40 = wp::load(var_39);
        var_41 = wp::copy(var_40);
        // v1 = v[j]  # vel one                                                                   <L 188>
        var_42 = wp::address(var_v, var_26);
        var_43 = wp::load(var_42);
        var_44 = wp::copy(var_43);
        // v2 = v[k]  # vel two                                                                   <L 189>
        var_45 = wp::address(var_v, var_29);
        var_46 = wp::load(var_45);
        var_47 = wp::copy(var_46);
        // x10 = x1 - x0  # barycentric coordinates (centered at p)                               <L 191>
        var_48 = wp::sub(var_35, var_32);
        // x20 = x2 - x0                                                                          <L 192>
        var_49 = wp::sub(var_38, var_32);
        // v10 = v1 - v0                                                                          <L 194>
        var_50 = wp::sub(var_44, var_41);
        // v20 = v2 - v0                                                                          <L 195>
        var_51 = wp::sub(var_47, var_41);
        // Dm = pose[tid]                                                                         <L 197>
        var_52 = wp::address(var_pose, var_0);
        var_53 = wp::load(var_52);
        var_54 = wp::copy(var_53);
        // inv_rest_area = wp.determinant(Dm) * 2.0  # 1 / det(A) = det(A^-1)                     <L 199>
        var_55 = wp::determinant(var_54);
        var_57 = wp::mul(var_55, var_56);
        // rest_area = 1.0 / inv_rest_area                                                        <L 200>
        var_59 = wp::div(var_58, var_57);
        // k_mu = k_mu * rest_area                                                                <L 203>
        var_60 = wp::mul(var_4, var_59);
        // k_lambda = k_lambda * rest_area                                                        <L 204>
        var_61 = wp::mul(var_8, var_59);
        // k_damp = k_damp * rest_area                                                            <L 205>
        var_62 = wp::mul(var_12, var_59);
        // F1 = x10 * Dm[0, 0] + x20 * Dm[1, 0]                                                   <L 208>
        var_63 = wp::extract(var_54, var_1, var_1);
        var_64 = wp::mul(var_48, var_63);
        var_65 = wp::extract(var_54, var_5, var_1);
        var_66 = wp::mul(var_49, var_65);
        var_67 = wp::add(var_64, var_66);
        // F2 = x10 * Dm[0, 1] + x20 * Dm[1, 1]                                                   <L 209>
        var_68 = wp::extract(var_54, var_1, var_5);
        var_69 = wp::mul(var_48, var_68);
        var_70 = wp::extract(var_54, var_5, var_5);
        var_71 = wp::mul(var_49, var_70);
        var_72 = wp::add(var_69, var_71);
        // dFdt1 = v10 * Dm[0, 0] + v20 * Dm[1, 0]                                                <L 212>
        var_73 = wp::extract(var_54, var_1, var_1);
        var_74 = wp::mul(var_50, var_73);
        var_75 = wp::extract(var_54, var_5, var_1);
        var_76 = wp::mul(var_51, var_75);
        var_77 = wp::add(var_74, var_76);
        // dFdt2 = v10 * Dm[0, 1] + v20 * Dm[1, 1]                                                <L 213>
        var_78 = wp::extract(var_54, var_1, var_5);
        var_79 = wp::mul(var_50, var_78);
        var_80 = wp::extract(var_54, var_5, var_5);
        var_81 = wp::mul(var_51, var_80);
        var_82 = wp::add(var_79, var_81);
        // P1 = F1 * k_mu + dFdt1 * k_damp                                                        <L 216>
        var_83 = wp::mul(var_67, var_60);
        var_84 = wp::mul(var_77, var_62);
        var_85 = wp::add(var_83, var_84);
        // P2 = F2 * k_mu + dFdt2 * k_damp                                                        <L 217>
        var_86 = wp::mul(var_72, var_60);
        var_87 = wp::mul(var_82, var_62);
        var_88 = wp::add(var_86, var_87);
        // f1 = P1 * Dm[0, 0] + P2 * Dm[0, 1]                                                     <L 254>
        var_89 = wp::extract(var_54, var_1, var_1);
        var_90 = wp::mul(var_85, var_89);
        var_91 = wp::extract(var_54, var_1, var_5);
        var_92 = wp::mul(var_88, var_91);
        var_93 = wp::add(var_90, var_92);
        // f2 = P1 * Dm[1, 0] + P2 * Dm[1, 1]                                                     <L 255>
        var_94 = wp::extract(var_54, var_5, var_1);
        var_95 = wp::mul(var_85, var_94);
        var_96 = wp::extract(var_54, var_5, var_5);
        var_97 = wp::mul(var_88, var_96);
        var_98 = wp::add(var_95, var_97);
        // alpha = 1.0 + k_mu / k_lambda                                                          <L 256>
        var_99 = wp::div(var_60, var_61);
        var_100 = wp::add(var_58, var_99);
        // n = wp.cross(x10, x20)                                                                 <L 261>
        var_101 = wp::cross(var_48, var_49);
        // area = wp.length(n) * 0.5                                                              <L 262>
        var_102 = wp::length(var_101);
        var_104 = wp::mul(var_102, var_103);
        // act = activation[tid]                                                                  <L 265>
        var_105 = wp::address(var_activation, var_0);
        var_106 = wp::load(var_105);
        var_107 = wp::copy(var_106);
        // c = area * inv_rest_area - alpha + act                                                 <L 268>
        var_108 = wp::mul(var_104, var_57);
        var_109 = wp::sub(var_108, var_100);
        var_110 = wp::add(var_109, var_107);
        // n = wp.normalize(n)                                                                    <L 271>
        var_111 = wp::normalize(var_101);
        // dcdq = wp.cross(x20, n) * inv_rest_area * 0.5                                          <L 272>
        var_112 = wp::cross(var_49, var_111);
        var_113 = wp::mul(var_112, var_57);
        var_114 = wp::mul(var_113, var_103);
        // dcdr = wp.cross(n, x10) * inv_rest_area * 0.5                                          <L 273>
        var_115 = wp::cross(var_111, var_48);
        var_116 = wp::mul(var_115, var_57);
        var_117 = wp::mul(var_116, var_103);
        // f_area = k_lambda * c                                                                  <L 275>
        var_118 = wp::mul(var_61, var_110);
        // dcdt = dot(dcdq, v1) + dot(dcdr, v2) - dot(dcdq + dcdr, v0)                            <L 280>
        var_119 = wp::dot(var_114, var_44);
        var_120 = wp::dot(var_117, var_47);
        var_121 = wp::add(var_119, var_120);
        var_122 = wp::add(var_114, var_117);
        var_123 = wp::dot(var_122, var_41);
        var_124 = wp::sub(var_121, var_123);
        // f_damp = k_damp * dcdt                                                                 <L 281>
        var_125 = wp::mul(var_62, var_124);
        // f1 = f1 + dcdq * (f_area + f_damp)                                                     <L 283>
        var_126 = wp::add(var_118, var_125);
        var_127 = wp::mul(var_114, var_126);
        var_128 = wp::add(var_93, var_127);
        // f2 = f2 + dcdr * (f_area + f_damp)                                                     <L 284>
        var_129 = wp::add(var_118, var_125);
        var_130 = wp::mul(var_117, var_129);
        var_131 = wp::add(var_98, var_130);
        // f0 = f1 + f2                                                                           <L 285>
        var_132 = wp::add(var_128, var_131);
        // vmid = (v0 + v1 + v2) * 0.3333                                                         <L 290>
        var_133 = wp::add(var_41, var_44);
        var_134 = wp::add(var_133, var_47);
        var_136 = wp::mul(var_134, var_135);
        // vdir = wp.normalize(vmid)                                                              <L 291>
        var_137 = wp::normalize(var_136);
        // f_drag = vmid * (k_drag * area * wp.abs(wp.dot(n, vmid)))                              <L 293>
        var_138 = wp::mul(var_16, var_104);
        var_139 = wp::dot(var_111, var_136);
        var_140 = wp::abs(var_139);
        var_141 = wp::mul(var_138, var_140);
        var_142 = wp::mul(var_136, var_141);
        // f_lift = n * (k_lift * area * (1.57079 - wp.acos(wp.dot(n, vdir)))) * dot(vmid, vmid)       <L 294>
        var_143 = wp::mul(var_20, var_104);
        var_145 = wp::dot(var_111, var_137);
        var_146 = wp::acos(var_145);
        var_147 = wp::sub(var_144, var_146);
        var_148 = wp::mul(var_143, var_147);
        var_149 = wp::mul(var_111, var_148);
        var_150 = wp::dot(var_136, var_136);
        var_151 = wp::mul(var_149, var_150);
        // f0 = f0 - f_drag - f_lift                                                              <L 297>
        var_152 = wp::sub(var_132, var_142);
        var_153 = wp::sub(var_152, var_151);
        // f1 = f1 + f_drag + f_lift                                                              <L 298>
        var_154 = wp::add(var_128, var_142);
        var_155 = wp::add(var_154, var_151);
        // f2 = f2 + f_drag + f_lift                                                              <L 299>
        var_156 = wp::add(var_131, var_142);
        var_157 = wp::add(var_156, var_151);
        // wp.atomic_add(f, i, f0)                                                                <L 302>
        var_158 = wp::atomic_add(var_f, var_23, var_153);
        // wp.atomic_sub(f, j, f1)                                                                <L 303>
        var_159 = wp::atomic_sub(var_f, var_26, var_155);
        // wp.atomic_sub(f, k, f2)                                                                <L 304>
        var_160 = wp::atomic_sub(var_f, var_29, var_157);
    }
}

extern "C" __global__ void eval_triangles_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float32>> var_x,
    wp::array_t<wp::vec_t<3,wp::float32>> var_v,
    wp::array_t<wp::int32> var_indices,
    wp::array_t<wp::mat_t<2,2,wp::float32>> var_pose,
    wp::array_t<wp::float32> var_activation,
    wp::array_t<wp::float32> var_materials,
    wp::array_t<wp::vec_t<3,wp::float32>> var_f,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_x,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_v,
    wp::array_t<wp::int32> adj_indices,
    wp::array_t<wp::mat_t<2,2,wp::float32>> adj_pose,
    wp::array_t<wp::float32> adj_activation,
    wp::array_t<wp::float32> adj_materials,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_f)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        const wp::int32 var_1 = 0;
        wp::float32* var_2;
        wp::float32 var_3;
        wp::float32 var_4;
        const wp::int32 var_5 = 1;
        wp::float32* var_6;
        wp::float32 var_7;
        wp::float32 var_8;
        const wp::int32 var_9 = 2;
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        const wp::int32 var_13 = 3;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        const wp::int32 var_17 = 4;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::int32* var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        wp::int32* var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        wp::vec_t<3,wp::float32>* var_30;
        wp::vec_t<3,wp::float32> var_31;
        wp::vec_t<3,wp::float32> var_32;
        wp::vec_t<3,wp::float32>* var_33;
        wp::vec_t<3,wp::float32> var_34;
        wp::vec_t<3,wp::float32> var_35;
        wp::vec_t<3,wp::float32>* var_36;
        wp::vec_t<3,wp::float32> var_37;
        wp::vec_t<3,wp::float32> var_38;
        wp::vec_t<3,wp::float32>* var_39;
        wp::vec_t<3,wp::float32> var_40;
        wp::vec_t<3,wp::float32> var_41;
        wp::vec_t<3,wp::float32>* var_42;
        wp::vec_t<3,wp::float32> var_43;
        wp::vec_t<3,wp::float32> var_44;
        wp::vec_t<3,wp::float32>* var_45;
        wp::vec_t<3,wp::float32> var_46;
        wp::vec_t<3,wp::float32> var_47;
        wp::vec_t<3,wp::float32> var_48;
        wp::vec_t<3,wp::float32> var_49;
        wp::vec_t<3,wp::float32> var_50;
        wp::vec_t<3,wp::float32> var_51;
        wp::mat_t<2,2,wp::float32>* var_52;
        wp::mat_t<2,2,wp::float32> var_53;
        wp::mat_t<2,2,wp::float32> var_54;
        wp::float32 var_55;
        const wp::float32 var_56 = 2.0;
        wp::float32 var_57;
        const wp::float32 var_58 = 1.0;
        wp::float32 var_59;
        wp::float32 var_60;
        wp::float32 var_61;
        wp::float32 var_62;
        wp::float32 var_63;
        wp::vec_t<3,wp::float32> var_64;
        wp::float32 var_65;
        wp::vec_t<3,wp::float32> var_66;
        wp::vec_t<3,wp::float32> var_67;
        wp::float32 var_68;
        wp::vec_t<3,wp::float32> var_69;
        wp::float32 var_70;
        wp::vec_t<3,wp::float32> var_71;
        wp::vec_t<3,wp::float32> var_72;
        wp::float32 var_73;
        wp::vec_t<3,wp::float32> var_74;
        wp::float32 var_75;
        wp::vec_t<3,wp::float32> var_76;
        wp::vec_t<3,wp::float32> var_77;
        wp::float32 var_78;
        wp::vec_t<3,wp::float32> var_79;
        wp::float32 var_80;
        wp::vec_t<3,wp::float32> var_81;
        wp::vec_t<3,wp::float32> var_82;
        wp::vec_t<3,wp::float32> var_83;
        wp::vec_t<3,wp::float32> var_84;
        wp::vec_t<3,wp::float32> var_85;
        wp::vec_t<3,wp::float32> var_86;
        wp::vec_t<3,wp::float32> var_87;
        wp::vec_t<3,wp::float32> var_88;
        wp::float32 var_89;
        wp::vec_t<3,wp::float32> var_90;
        wp::float32 var_91;
        wp::vec_t<3,wp::float32> var_92;
        wp::vec_t<3,wp::float32> var_93;
        wp::float32 var_94;
        wp::vec_t<3,wp::float32> var_95;
        wp::float32 var_96;
        wp::vec_t<3,wp::float32> var_97;
        wp::vec_t<3,wp::float32> var_98;
        wp::float32 var_99;
        wp::float32 var_100;
        wp::vec_t<3,wp::float32> var_101;
        wp::float32 var_102;
        const wp::float32 var_103 = 0.5;
        wp::float32 var_104;
        wp::float32* var_105;
        wp::float32 var_106;
        wp::float32 var_107;
        wp::float32 var_108;
        wp::float32 var_109;
        wp::float32 var_110;
        wp::vec_t<3,wp::float32> var_111;
        wp::vec_t<3,wp::float32> var_112;
        wp::vec_t<3,wp::float32> var_113;
        wp::vec_t<3,wp::float32> var_114;
        wp::vec_t<3,wp::float32> var_115;
        wp::vec_t<3,wp::float32> var_116;
        wp::vec_t<3,wp::float32> var_117;
        wp::float32 var_118;
        wp::float32 var_119;
        wp::float32 var_120;
        wp::float32 var_121;
        wp::vec_t<3,wp::float32> var_122;
        wp::float32 var_123;
        wp::float32 var_124;
        wp::float32 var_125;
        wp::float32 var_126;
        wp::vec_t<3,wp::float32> var_127;
        wp::vec_t<3,wp::float32> var_128;
        wp::float32 var_129;
        wp::vec_t<3,wp::float32> var_130;
        wp::vec_t<3,wp::float32> var_131;
        wp::vec_t<3,wp::float32> var_132;
        wp::vec_t<3,wp::float32> var_133;
        wp::vec_t<3,wp::float32> var_134;
        const wp::float32 var_135 = 0.3333;
        wp::vec_t<3,wp::float32> var_136;
        wp::vec_t<3,wp::float32> var_137;
        wp::float32 var_138;
        wp::float32 var_139;
        wp::float32 var_140;
        wp::float32 var_141;
        wp::vec_t<3,wp::float32> var_142;
        wp::float32 var_143;
        const wp::float32 var_144 = 1.57079;
        wp::float32 var_145;
        wp::float32 var_146;
        wp::float32 var_147;
        wp::float32 var_148;
        wp::vec_t<3,wp::float32> var_149;
        wp::float32 var_150;
        wp::vec_t<3,wp::float32> var_151;
        wp::vec_t<3,wp::float32> var_152;
        wp::vec_t<3,wp::float32> var_153;
        wp::vec_t<3,wp::float32> var_154;
        wp::vec_t<3,wp::float32> var_155;
        wp::vec_t<3,wp::float32> var_156;
        wp::vec_t<3,wp::float32> var_157;
        wp::vec_t<3,wp::float32> var_158;
        wp::vec_t<3,wp::float32> var_159;
        wp::vec_t<3,wp::float32> var_160;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::float32 adj_2 = {};
        wp::float32 adj_3 = {};
        wp::float32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::float32 adj_6 = {};
        wp::float32 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::float32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::int32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::int32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::int32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::int32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::vec_t<3,wp::float32> adj_30 = {};
        wp::vec_t<3,wp::float32> adj_31 = {};
        wp::vec_t<3,wp::float32> adj_32 = {};
        wp::vec_t<3,wp::float32> adj_33 = {};
        wp::vec_t<3,wp::float32> adj_34 = {};
        wp::vec_t<3,wp::float32> adj_35 = {};
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
        wp::vec_t<3,wp::float32> adj_46 = {};
        wp::vec_t<3,wp::float32> adj_47 = {};
        wp::vec_t<3,wp::float32> adj_48 = {};
        wp::vec_t<3,wp::float32> adj_49 = {};
        wp::vec_t<3,wp::float32> adj_50 = {};
        wp::vec_t<3,wp::float32> adj_51 = {};
        wp::mat_t<2,2,wp::float32> adj_52 = {};
        wp::mat_t<2,2,wp::float32> adj_53 = {};
        wp::mat_t<2,2,wp::float32> adj_54 = {};
        wp::float32 adj_55 = {};
        wp::float32 adj_56 = {};
        wp::float32 adj_57 = {};
        wp::float32 adj_58 = {};
        wp::float32 adj_59 = {};
        wp::float32 adj_60 = {};
        wp::float32 adj_61 = {};
        wp::float32 adj_62 = {};
        wp::float32 adj_63 = {};
        wp::vec_t<3,wp::float32> adj_64 = {};
        wp::float32 adj_65 = {};
        wp::vec_t<3,wp::float32> adj_66 = {};
        wp::vec_t<3,wp::float32> adj_67 = {};
        wp::float32 adj_68 = {};
        wp::vec_t<3,wp::float32> adj_69 = {};
        wp::float32 adj_70 = {};
        wp::vec_t<3,wp::float32> adj_71 = {};
        wp::vec_t<3,wp::float32> adj_72 = {};
        wp::float32 adj_73 = {};
        wp::vec_t<3,wp::float32> adj_74 = {};
        wp::float32 adj_75 = {};
        wp::vec_t<3,wp::float32> adj_76 = {};
        wp::vec_t<3,wp::float32> adj_77 = {};
        wp::float32 adj_78 = {};
        wp::vec_t<3,wp::float32> adj_79 = {};
        wp::float32 adj_80 = {};
        wp::vec_t<3,wp::float32> adj_81 = {};
        wp::vec_t<3,wp::float32> adj_82 = {};
        wp::vec_t<3,wp::float32> adj_83 = {};
        wp::vec_t<3,wp::float32> adj_84 = {};
        wp::vec_t<3,wp::float32> adj_85 = {};
        wp::vec_t<3,wp::float32> adj_86 = {};
        wp::vec_t<3,wp::float32> adj_87 = {};
        wp::vec_t<3,wp::float32> adj_88 = {};
        wp::float32 adj_89 = {};
        wp::vec_t<3,wp::float32> adj_90 = {};
        wp::float32 adj_91 = {};
        wp::vec_t<3,wp::float32> adj_92 = {};
        wp::vec_t<3,wp::float32> adj_93 = {};
        wp::float32 adj_94 = {};
        wp::vec_t<3,wp::float32> adj_95 = {};
        wp::float32 adj_96 = {};
        wp::vec_t<3,wp::float32> adj_97 = {};
        wp::vec_t<3,wp::float32> adj_98 = {};
        wp::float32 adj_99 = {};
        wp::float32 adj_100 = {};
        wp::vec_t<3,wp::float32> adj_101 = {};
        wp::float32 adj_102 = {};
        wp::float32 adj_103 = {};
        wp::float32 adj_104 = {};
        wp::float32 adj_105 = {};
        wp::float32 adj_106 = {};
        wp::float32 adj_107 = {};
        wp::float32 adj_108 = {};
        wp::float32 adj_109 = {};
        wp::float32 adj_110 = {};
        wp::vec_t<3,wp::float32> adj_111 = {};
        wp::vec_t<3,wp::float32> adj_112 = {};
        wp::vec_t<3,wp::float32> adj_113 = {};
        wp::vec_t<3,wp::float32> adj_114 = {};
        wp::vec_t<3,wp::float32> adj_115 = {};
        wp::vec_t<3,wp::float32> adj_116 = {};
        wp::vec_t<3,wp::float32> adj_117 = {};
        wp::float32 adj_118 = {};
        wp::float32 adj_119 = {};
        wp::float32 adj_120 = {};
        wp::float32 adj_121 = {};
        wp::vec_t<3,wp::float32> adj_122 = {};
        wp::float32 adj_123 = {};
        wp::float32 adj_124 = {};
        wp::float32 adj_125 = {};
        wp::float32 adj_126 = {};
        wp::vec_t<3,wp::float32> adj_127 = {};
        wp::vec_t<3,wp::float32> adj_128 = {};
        wp::float32 adj_129 = {};
        wp::vec_t<3,wp::float32> adj_130 = {};
        wp::vec_t<3,wp::float32> adj_131 = {};
        wp::vec_t<3,wp::float32> adj_132 = {};
        wp::vec_t<3,wp::float32> adj_133 = {};
        wp::vec_t<3,wp::float32> adj_134 = {};
        wp::float32 adj_135 = {};
        wp::vec_t<3,wp::float32> adj_136 = {};
        wp::vec_t<3,wp::float32> adj_137 = {};
        wp::float32 adj_138 = {};
        wp::float32 adj_139 = {};
        wp::float32 adj_140 = {};
        wp::float32 adj_141 = {};
        wp::vec_t<3,wp::float32> adj_142 = {};
        wp::float32 adj_143 = {};
        wp::float32 adj_144 = {};
        wp::float32 adj_145 = {};
        wp::float32 adj_146 = {};
        wp::float32 adj_147 = {};
        wp::float32 adj_148 = {};
        wp::vec_t<3,wp::float32> adj_149 = {};
        wp::float32 adj_150 = {};
        wp::vec_t<3,wp::float32> adj_151 = {};
        wp::vec_t<3,wp::float32> adj_152 = {};
        wp::vec_t<3,wp::float32> adj_153 = {};
        wp::vec_t<3,wp::float32> adj_154 = {};
        wp::vec_t<3,wp::float32> adj_155 = {};
        wp::vec_t<3,wp::float32> adj_156 = {};
        wp::vec_t<3,wp::float32> adj_157 = {};
        wp::vec_t<3,wp::float32> adj_158 = {};
        wp::vec_t<3,wp::float32> adj_159 = {};
        wp::vec_t<3,wp::float32> adj_160 = {};
        //---------
        // forward
        // def eval_triangles(                                                                    <L 162>
        // tid = wp.tid()                                                                         <L 171>
        var_0 = builtin_tid1d();
        // k_mu = materials[tid, 0]                                                               <L 173>
        var_2 = wp::address(var_materials, var_0, var_1);
        var_3 = wp::load(var_2);
        var_4 = wp::copy(var_3);
        // k_lambda = materials[tid, 1]                                                           <L 174>
        var_6 = wp::address(var_materials, var_0, var_5);
        var_7 = wp::load(var_6);
        var_8 = wp::copy(var_7);
        // k_damp = materials[tid, 2]                                                             <L 175>
        var_10 = wp::address(var_materials, var_0, var_9);
        var_11 = wp::load(var_10);
        var_12 = wp::copy(var_11);
        // k_drag = materials[tid, 3]                                                             <L 176>
        var_14 = wp::address(var_materials, var_0, var_13);
        var_15 = wp::load(var_14);
        var_16 = wp::copy(var_15);
        // k_lift = materials[tid, 4]                                                             <L 177>
        var_18 = wp::address(var_materials, var_0, var_17);
        var_19 = wp::load(var_18);
        var_20 = wp::copy(var_19);
        // i = indices[tid, 0]                                                                    <L 179>
        var_21 = wp::address(var_indices, var_0, var_1);
        var_22 = wp::load(var_21);
        var_23 = wp::copy(var_22);
        // j = indices[tid, 1]                                                                    <L 180>
        var_24 = wp::address(var_indices, var_0, var_5);
        var_25 = wp::load(var_24);
        var_26 = wp::copy(var_25);
        // k = indices[tid, 2]                                                                    <L 181>
        var_27 = wp::address(var_indices, var_0, var_9);
        var_28 = wp::load(var_27);
        var_29 = wp::copy(var_28);
        // x0 = x[i]  # point zero                                                                <L 183>
        var_30 = wp::address(var_x, var_23);
        var_31 = wp::load(var_30);
        var_32 = wp::copy(var_31);
        // x1 = x[j]  # point one                                                                 <L 184>
        var_33 = wp::address(var_x, var_26);
        var_34 = wp::load(var_33);
        var_35 = wp::copy(var_34);
        // x2 = x[k]  # point two                                                                 <L 185>
        var_36 = wp::address(var_x, var_29);
        var_37 = wp::load(var_36);
        var_38 = wp::copy(var_37);
        // v0 = v[i]  # vel zero                                                                  <L 187>
        var_39 = wp::address(var_v, var_23);
        var_40 = wp::load(var_39);
        var_41 = wp::copy(var_40);
        // v1 = v[j]  # vel one                                                                   <L 188>
        var_42 = wp::address(var_v, var_26);
        var_43 = wp::load(var_42);
        var_44 = wp::copy(var_43);
        // v2 = v[k]  # vel two                                                                   <L 189>
        var_45 = wp::address(var_v, var_29);
        var_46 = wp::load(var_45);
        var_47 = wp::copy(var_46);
        // x10 = x1 - x0  # barycentric coordinates (centered at p)                               <L 191>
        var_48 = wp::sub(var_35, var_32);
        // x20 = x2 - x0                                                                          <L 192>
        var_49 = wp::sub(var_38, var_32);
        // v10 = v1 - v0                                                                          <L 194>
        var_50 = wp::sub(var_44, var_41);
        // v20 = v2 - v0                                                                          <L 195>
        var_51 = wp::sub(var_47, var_41);
        // Dm = pose[tid]                                                                         <L 197>
        var_52 = wp::address(var_pose, var_0);
        var_53 = wp::load(var_52);
        var_54 = wp::copy(var_53);
        // inv_rest_area = wp.determinant(Dm) * 2.0  # 1 / det(A) = det(A^-1)                     <L 199>
        var_55 = wp::determinant(var_54);
        var_57 = wp::mul(var_55, var_56);
        // rest_area = 1.0 / inv_rest_area                                                        <L 200>
        var_59 = wp::div(var_58, var_57);
        // k_mu = k_mu * rest_area                                                                <L 203>
        var_60 = wp::mul(var_4, var_59);
        // k_lambda = k_lambda * rest_area                                                        <L 204>
        var_61 = wp::mul(var_8, var_59);
        // k_damp = k_damp * rest_area                                                            <L 205>
        var_62 = wp::mul(var_12, var_59);
        // F1 = x10 * Dm[0, 0] + x20 * Dm[1, 0]                                                   <L 208>
        var_63 = wp::extract(var_54, var_1, var_1);
        var_64 = wp::mul(var_48, var_63);
        var_65 = wp::extract(var_54, var_5, var_1);
        var_66 = wp::mul(var_49, var_65);
        var_67 = wp::add(var_64, var_66);
        // F2 = x10 * Dm[0, 1] + x20 * Dm[1, 1]                                                   <L 209>
        var_68 = wp::extract(var_54, var_1, var_5);
        var_69 = wp::mul(var_48, var_68);
        var_70 = wp::extract(var_54, var_5, var_5);
        var_71 = wp::mul(var_49, var_70);
        var_72 = wp::add(var_69, var_71);
        // dFdt1 = v10 * Dm[0, 0] + v20 * Dm[1, 0]                                                <L 212>
        var_73 = wp::extract(var_54, var_1, var_1);
        var_74 = wp::mul(var_50, var_73);
        var_75 = wp::extract(var_54, var_5, var_1);
        var_76 = wp::mul(var_51, var_75);
        var_77 = wp::add(var_74, var_76);
        // dFdt2 = v10 * Dm[0, 1] + v20 * Dm[1, 1]                                                <L 213>
        var_78 = wp::extract(var_54, var_1, var_5);
        var_79 = wp::mul(var_50, var_78);
        var_80 = wp::extract(var_54, var_5, var_5);
        var_81 = wp::mul(var_51, var_80);
        var_82 = wp::add(var_79, var_81);
        // P1 = F1 * k_mu + dFdt1 * k_damp                                                        <L 216>
        var_83 = wp::mul(var_67, var_60);
        var_84 = wp::mul(var_77, var_62);
        var_85 = wp::add(var_83, var_84);
        // P2 = F2 * k_mu + dFdt2 * k_damp                                                        <L 217>
        var_86 = wp::mul(var_72, var_60);
        var_87 = wp::mul(var_82, var_62);
        var_88 = wp::add(var_86, var_87);
        // f1 = P1 * Dm[0, 0] + P2 * Dm[0, 1]                                                     <L 254>
        var_89 = wp::extract(var_54, var_1, var_1);
        var_90 = wp::mul(var_85, var_89);
        var_91 = wp::extract(var_54, var_1, var_5);
        var_92 = wp::mul(var_88, var_91);
        var_93 = wp::add(var_90, var_92);
        // f2 = P1 * Dm[1, 0] + P2 * Dm[1, 1]                                                     <L 255>
        var_94 = wp::extract(var_54, var_5, var_1);
        var_95 = wp::mul(var_85, var_94);
        var_96 = wp::extract(var_54, var_5, var_5);
        var_97 = wp::mul(var_88, var_96);
        var_98 = wp::add(var_95, var_97);
        // alpha = 1.0 + k_mu / k_lambda                                                          <L 256>
        var_99 = wp::div(var_60, var_61);
        var_100 = wp::add(var_58, var_99);
        // n = wp.cross(x10, x20)                                                                 <L 261>
        var_101 = wp::cross(var_48, var_49);
        // area = wp.length(n) * 0.5                                                              <L 262>
        var_102 = wp::length(var_101);
        var_104 = wp::mul(var_102, var_103);
        // act = activation[tid]                                                                  <L 265>
        var_105 = wp::address(var_activation, var_0);
        var_106 = wp::load(var_105);
        var_107 = wp::copy(var_106);
        // c = area * inv_rest_area - alpha + act                                                 <L 268>
        var_108 = wp::mul(var_104, var_57);
        var_109 = wp::sub(var_108, var_100);
        var_110 = wp::add(var_109, var_107);
        // n = wp.normalize(n)                                                                    <L 271>
        var_111 = wp::normalize(var_101);
        // dcdq = wp.cross(x20, n) * inv_rest_area * 0.5                                          <L 272>
        var_112 = wp::cross(var_49, var_111);
        var_113 = wp::mul(var_112, var_57);
        var_114 = wp::mul(var_113, var_103);
        // dcdr = wp.cross(n, x10) * inv_rest_area * 0.5                                          <L 273>
        var_115 = wp::cross(var_111, var_48);
        var_116 = wp::mul(var_115, var_57);
        var_117 = wp::mul(var_116, var_103);
        // f_area = k_lambda * c                                                                  <L 275>
        var_118 = wp::mul(var_61, var_110);
        // dcdt = dot(dcdq, v1) + dot(dcdr, v2) - dot(dcdq + dcdr, v0)                            <L 280>
        var_119 = wp::dot(var_114, var_44);
        var_120 = wp::dot(var_117, var_47);
        var_121 = wp::add(var_119, var_120);
        var_122 = wp::add(var_114, var_117);
        var_123 = wp::dot(var_122, var_41);
        var_124 = wp::sub(var_121, var_123);
        // f_damp = k_damp * dcdt                                                                 <L 281>
        var_125 = wp::mul(var_62, var_124);
        // f1 = f1 + dcdq * (f_area + f_damp)                                                     <L 283>
        var_126 = wp::add(var_118, var_125);
        var_127 = wp::mul(var_114, var_126);
        var_128 = wp::add(var_93, var_127);
        // f2 = f2 + dcdr * (f_area + f_damp)                                                     <L 284>
        var_129 = wp::add(var_118, var_125);
        var_130 = wp::mul(var_117, var_129);
        var_131 = wp::add(var_98, var_130);
        // f0 = f1 + f2                                                                           <L 285>
        var_132 = wp::add(var_128, var_131);
        // vmid = (v0 + v1 + v2) * 0.3333                                                         <L 290>
        var_133 = wp::add(var_41, var_44);
        var_134 = wp::add(var_133, var_47);
        var_136 = wp::mul(var_134, var_135);
        // vdir = wp.normalize(vmid)                                                              <L 291>
        var_137 = wp::normalize(var_136);
        // f_drag = vmid * (k_drag * area * wp.abs(wp.dot(n, vmid)))                              <L 293>
        var_138 = wp::mul(var_16, var_104);
        var_139 = wp::dot(var_111, var_136);
        var_140 = wp::abs(var_139);
        var_141 = wp::mul(var_138, var_140);
        var_142 = wp::mul(var_136, var_141);
        // f_lift = n * (k_lift * area * (1.57079 - wp.acos(wp.dot(n, vdir)))) * dot(vmid, vmid)       <L 294>
        var_143 = wp::mul(var_20, var_104);
        var_145 = wp::dot(var_111, var_137);
        var_146 = wp::acos(var_145);
        var_147 = wp::sub(var_144, var_146);
        var_148 = wp::mul(var_143, var_147);
        var_149 = wp::mul(var_111, var_148);
        var_150 = wp::dot(var_136, var_136);
        var_151 = wp::mul(var_149, var_150);
        // f0 = f0 - f_drag - f_lift                                                              <L 297>
        var_152 = wp::sub(var_132, var_142);
        var_153 = wp::sub(var_152, var_151);
        // f1 = f1 + f_drag + f_lift                                                              <L 298>
        var_154 = wp::add(var_128, var_142);
        var_155 = wp::add(var_154, var_151);
        // f2 = f2 + f_drag + f_lift                                                              <L 299>
        var_156 = wp::add(var_131, var_142);
        var_157 = wp::add(var_156, var_151);
        // wp.atomic_add(f, i, f0)                                                                <L 302>
        // var_158 = wp::atomic_add(var_f, var_23, var_153);
        // wp.atomic_sub(f, j, f1)                                                                <L 303>
        // var_159 = wp::atomic_sub(var_f, var_26, var_155);
        // wp.atomic_sub(f, k, f2)                                                                <L 304>
        // var_160 = wp::atomic_sub(var_f, var_29, var_157);
        //---------
        // reverse
        wp::adj_atomic_sub(var_f, var_29, var_157, adj_f, adj_29, adj_157, adj_160);
        // adj: wp.atomic_sub(f, k, f2)                                                           <L 304>
        wp::adj_atomic_sub(var_f, var_26, var_155, adj_f, adj_26, adj_155, adj_159);
        // adj: wp.atomic_sub(f, j, f1)                                                           <L 303>
        wp::adj_atomic_add(var_f, var_23, var_153, adj_f, adj_23, adj_153, adj_158);
        // adj: wp.atomic_add(f, i, f0)                                                           <L 302>
        wp::adj_add(var_156, var_151, adj_156, adj_151, adj_157);
        wp::adj_add(var_131, var_142, adj_131, adj_142, adj_156);
        // adj: f2 = f2 + f_drag + f_lift                                                         <L 299>
        wp::adj_add(var_154, var_151, adj_154, adj_151, adj_155);
        wp::adj_add(var_128, var_142, adj_128, adj_142, adj_154);
        // adj: f1 = f1 + f_drag + f_lift                                                         <L 298>
        wp::adj_sub(var_152, var_151, adj_152, adj_151, adj_153);
        wp::adj_sub(var_132, var_142, adj_132, adj_142, adj_152);
        // adj: f0 = f0 - f_drag - f_lift                                                         <L 297>
        wp::adj_mul(var_149, var_150, adj_149, adj_150, adj_151);
        wp::adj_dot(var_136, var_136, adj_136, adj_136, adj_150);
        wp::adj_mul(var_111, var_148, adj_111, adj_148, adj_149);
        wp::adj_mul(var_143, var_147, adj_143, adj_147, adj_148);
        wp::adj_sub(var_144, var_146, adj_144, adj_146, adj_147);
        wp::adj_acos(var_145, adj_145, adj_146);
        wp::adj_dot(var_111, var_137, adj_111, adj_137, adj_145);
        wp::adj_mul(var_20, var_104, adj_20, adj_104, adj_143);
        // adj: f_lift = n * (k_lift * area * (1.57079 - wp.acos(wp.dot(n, vdir)))) * dot(vmid, vmid)  <L 294>
        wp::adj_mul(var_136, var_141, adj_136, adj_141, adj_142);
        wp::adj_mul(var_138, var_140, adj_138, adj_140, adj_141);
        wp::adj_abs(var_139, adj_139, adj_140);
        wp::adj_dot(var_111, var_136, adj_111, adj_136, adj_139);
        wp::adj_mul(var_16, var_104, adj_16, adj_104, adj_138);
        // adj: f_drag = vmid * (k_drag * area * wp.abs(wp.dot(n, vmid)))                         <L 293>
        wp::adj_normalize(var_136, var_137, adj_136, adj_137);
        // adj: vdir = wp.normalize(vmid)                                                         <L 291>
        wp::adj_mul(var_134, var_135, adj_134, adj_135, adj_136);
        wp::adj_add(var_133, var_47, adj_133, adj_47, adj_134);
        wp::adj_add(var_41, var_44, adj_41, adj_44, adj_133);
        // adj: vmid = (v0 + v1 + v2) * 0.3333                                                    <L 290>
        wp::adj_add(var_128, var_131, adj_128, adj_131, adj_132);
        // adj: f0 = f1 + f2                                                                      <L 285>
        wp::adj_add(var_98, var_130, adj_98, adj_130, adj_131);
        wp::adj_mul(var_117, var_129, adj_117, adj_129, adj_130);
        wp::adj_add(var_118, var_125, adj_118, adj_125, adj_129);
        // adj: f2 = f2 + dcdr * (f_area + f_damp)                                                <L 284>
        wp::adj_add(var_93, var_127, adj_93, adj_127, adj_128);
        wp::adj_mul(var_114, var_126, adj_114, adj_126, adj_127);
        wp::adj_add(var_118, var_125, adj_118, adj_125, adj_126);
        // adj: f1 = f1 + dcdq * (f_area + f_damp)                                                <L 283>
        wp::adj_mul(var_62, var_124, adj_62, adj_124, adj_125);
        // adj: f_damp = k_damp * dcdt                                                            <L 281>
        wp::adj_sub(var_121, var_123, adj_121, adj_123, adj_124);
        wp::adj_dot(var_122, var_41, adj_122, adj_41, adj_123);
        wp::adj_add(var_114, var_117, adj_114, adj_117, adj_122);
        wp::adj_add(var_119, var_120, adj_119, adj_120, adj_121);
        wp::adj_dot(var_117, var_47, adj_117, adj_47, adj_120);
        wp::adj_dot(var_114, var_44, adj_114, adj_44, adj_119);
        // adj: dcdt = dot(dcdq, v1) + dot(dcdr, v2) - dot(dcdq + dcdr, v0)                       <L 280>
        wp::adj_mul(var_61, var_110, adj_61, adj_110, adj_118);
        // adj: f_area = k_lambda * c                                                             <L 275>
        wp::adj_mul(var_116, var_103, adj_116, adj_103, adj_117);
        wp::adj_mul(var_115, var_57, adj_115, adj_57, adj_116);
        wp::adj_cross(var_111, var_48, adj_111, adj_48, adj_115);
        // adj: dcdr = wp.cross(n, x10) * inv_rest_area * 0.5                                     <L 273>
        wp::adj_mul(var_113, var_103, adj_113, adj_103, adj_114);
        wp::adj_mul(var_112, var_57, adj_112, adj_57, adj_113);
        wp::adj_cross(var_49, var_111, adj_49, adj_111, adj_112);
        // adj: dcdq = wp.cross(x20, n) * inv_rest_area * 0.5                                     <L 272>
        wp::adj_normalize(var_101, var_111, adj_101, adj_111);
        // adj: n = wp.normalize(n)                                                               <L 271>
        wp::adj_add(var_109, var_107, adj_109, adj_107, adj_110);
        wp::adj_sub(var_108, var_100, adj_108, adj_100, adj_109);
        wp::adj_mul(var_104, var_57, adj_104, adj_57, adj_108);
        // adj: c = area * inv_rest_area - alpha + act                                            <L 268>
        wp::adj_copy(var_106, adj_105, adj_107);
        wp::adj_load(var_105, adj_105, adj_106);
        wp::adj_address(var_activation, var_0, adj_activation, adj_0, adj_105);
        // adj: act = activation[tid]                                                             <L 265>
        wp::adj_mul(var_102, var_103, adj_102, adj_103, adj_104);
        wp::adj_length(var_101, var_102, adj_101, adj_102);
        // adj: area = wp.length(n) * 0.5                                                         <L 262>
        wp::adj_cross(var_48, var_49, adj_48, adj_49, adj_101);
        // adj: n = wp.cross(x10, x20)                                                            <L 261>
        wp::adj_add(var_58, var_99, adj_58, adj_99, adj_100);
        wp::adj_div(var_60, var_61, var_99, adj_60, adj_61, adj_99);
        // adj: alpha = 1.0 + k_mu / k_lambda                                                     <L 256>
        wp::adj_add(var_95, var_97, adj_95, adj_97, adj_98);
        wp::adj_mul(var_88, var_96, adj_88, adj_96, adj_97);
        wp::adj_extract(var_54, var_5, var_5, adj_54, adj_5, adj_5, adj_96);
        wp::adj_mul(var_85, var_94, adj_85, adj_94, adj_95);
        wp::adj_extract(var_54, var_5, var_1, adj_54, adj_5, adj_1, adj_94);
        // adj: f2 = P1 * Dm[1, 0] + P2 * Dm[1, 1]                                                <L 255>
        wp::adj_add(var_90, var_92, adj_90, adj_92, adj_93);
        wp::adj_mul(var_88, var_91, adj_88, adj_91, adj_92);
        wp::adj_extract(var_54, var_1, var_5, adj_54, adj_1, adj_5, adj_91);
        wp::adj_mul(var_85, var_89, adj_85, adj_89, adj_90);
        wp::adj_extract(var_54, var_1, var_1, adj_54, adj_1, adj_1, adj_89);
        // adj: f1 = P1 * Dm[0, 0] + P2 * Dm[0, 1]                                                <L 254>
        wp::adj_add(var_86, var_87, adj_86, adj_87, adj_88);
        wp::adj_mul(var_82, var_62, adj_82, adj_62, adj_87);
        wp::adj_mul(var_72, var_60, adj_72, adj_60, adj_86);
        // adj: P2 = F2 * k_mu + dFdt2 * k_damp                                                   <L 217>
        wp::adj_add(var_83, var_84, adj_83, adj_84, adj_85);
        wp::adj_mul(var_77, var_62, adj_77, adj_62, adj_84);
        wp::adj_mul(var_67, var_60, adj_67, adj_60, adj_83);
        // adj: P1 = F1 * k_mu + dFdt1 * k_damp                                                   <L 216>
        wp::adj_add(var_79, var_81, adj_79, adj_81, adj_82);
        wp::adj_mul(var_51, var_80, adj_51, adj_80, adj_81);
        wp::adj_extract(var_54, var_5, var_5, adj_54, adj_5, adj_5, adj_80);
        wp::adj_mul(var_50, var_78, adj_50, adj_78, adj_79);
        wp::adj_extract(var_54, var_1, var_5, adj_54, adj_1, adj_5, adj_78);
        // adj: dFdt2 = v10 * Dm[0, 1] + v20 * Dm[1, 1]                                           <L 213>
        wp::adj_add(var_74, var_76, adj_74, adj_76, adj_77);
        wp::adj_mul(var_51, var_75, adj_51, adj_75, adj_76);
        wp::adj_extract(var_54, var_5, var_1, adj_54, adj_5, adj_1, adj_75);
        wp::adj_mul(var_50, var_73, adj_50, adj_73, adj_74);
        wp::adj_extract(var_54, var_1, var_1, adj_54, adj_1, adj_1, adj_73);
        // adj: dFdt1 = v10 * Dm[0, 0] + v20 * Dm[1, 0]                                           <L 212>
        wp::adj_add(var_69, var_71, adj_69, adj_71, adj_72);
        wp::adj_mul(var_49, var_70, adj_49, adj_70, adj_71);
        wp::adj_extract(var_54, var_5, var_5, adj_54, adj_5, adj_5, adj_70);
        wp::adj_mul(var_48, var_68, adj_48, adj_68, adj_69);
        wp::adj_extract(var_54, var_1, var_5, adj_54, adj_1, adj_5, adj_68);
        // adj: F2 = x10 * Dm[0, 1] + x20 * Dm[1, 1]                                              <L 209>
        wp::adj_add(var_64, var_66, adj_64, adj_66, adj_67);
        wp::adj_mul(var_49, var_65, adj_49, adj_65, adj_66);
        wp::adj_extract(var_54, var_5, var_1, adj_54, adj_5, adj_1, adj_65);
        wp::adj_mul(var_48, var_63, adj_48, adj_63, adj_64);
        wp::adj_extract(var_54, var_1, var_1, adj_54, adj_1, adj_1, adj_63);
        // adj: F1 = x10 * Dm[0, 0] + x20 * Dm[1, 0]                                              <L 208>
        wp::adj_mul(var_12, var_59, adj_12, adj_59, adj_62);
        // adj: k_damp = k_damp * rest_area                                                       <L 205>
        wp::adj_mul(var_8, var_59, adj_8, adj_59, adj_61);
        // adj: k_lambda = k_lambda * rest_area                                                   <L 204>
        wp::adj_mul(var_4, var_59, adj_4, adj_59, adj_60);
        // adj: k_mu = k_mu * rest_area                                                           <L 203>
        wp::adj_div(var_58, var_57, var_59, adj_58, adj_57, adj_59);
        // adj: rest_area = 1.0 / inv_rest_area                                                   <L 200>
        wp::adj_mul(var_55, var_56, adj_55, adj_56, adj_57);
        wp::adj_determinant(var_54, adj_54, adj_55);
        // adj: inv_rest_area = wp.determinant(Dm) * 2.0  # 1 / det(A) = det(A^-1)                <L 199>
        wp::adj_copy(var_53, adj_52, adj_54);
        wp::adj_load(var_52, adj_52, adj_53);
        wp::adj_address(var_pose, var_0, adj_pose, adj_0, adj_52);
        // adj: Dm = pose[tid]                                                                    <L 197>
        wp::adj_sub(var_47, var_41, adj_47, adj_41, adj_51);
        // adj: v20 = v2 - v0                                                                     <L 195>
        wp::adj_sub(var_44, var_41, adj_44, adj_41, adj_50);
        // adj: v10 = v1 - v0                                                                     <L 194>
        wp::adj_sub(var_38, var_32, adj_38, adj_32, adj_49);
        // adj: x20 = x2 - x0                                                                     <L 192>
        wp::adj_sub(var_35, var_32, adj_35, adj_32, adj_48);
        // adj: x10 = x1 - x0  # barycentric coordinates (centered at p)                          <L 191>
        wp::adj_copy(var_46, adj_45, adj_47);
        wp::adj_load(var_45, adj_45, adj_46);
        wp::adj_address(var_v, var_29, adj_v, adj_29, adj_45);
        // adj: v2 = v[k]  # vel two                                                              <L 189>
        wp::adj_copy(var_43, adj_42, adj_44);
        wp::adj_load(var_42, adj_42, adj_43);
        wp::adj_address(var_v, var_26, adj_v, adj_26, adj_42);
        // adj: v1 = v[j]  # vel one                                                              <L 188>
        wp::adj_copy(var_40, adj_39, adj_41);
        wp::adj_load(var_39, adj_39, adj_40);
        wp::adj_address(var_v, var_23, adj_v, adj_23, adj_39);
        // adj: v0 = v[i]  # vel zero                                                             <L 187>
        wp::adj_copy(var_37, adj_36, adj_38);
        wp::adj_load(var_36, adj_36, adj_37);
        wp::adj_address(var_x, var_29, adj_x, adj_29, adj_36);
        // adj: x2 = x[k]  # point two                                                            <L 185>
        wp::adj_copy(var_34, adj_33, adj_35);
        wp::adj_load(var_33, adj_33, adj_34);
        wp::adj_address(var_x, var_26, adj_x, adj_26, adj_33);
        // adj: x1 = x[j]  # point one                                                            <L 184>
        wp::adj_copy(var_31, adj_30, adj_32);
        wp::adj_load(var_30, adj_30, adj_31);
        wp::adj_address(var_x, var_23, adj_x, adj_23, adj_30);
        // adj: x0 = x[i]  # point zero                                                           <L 183>
        wp::adj_copy(var_28, adj_27, adj_29);
        wp::adj_load(var_27, adj_27, adj_28);
        wp::adj_address(var_indices, var_0, var_9, adj_indices, adj_0, adj_9, adj_27);
        // adj: k = indices[tid, 2]                                                               <L 181>
        wp::adj_copy(var_25, adj_24, adj_26);
        wp::adj_load(var_24, adj_24, adj_25);
        wp::adj_address(var_indices, var_0, var_5, adj_indices, adj_0, adj_5, adj_24);
        // adj: j = indices[tid, 1]                                                               <L 180>
        wp::adj_copy(var_22, adj_21, adj_23);
        wp::adj_load(var_21, adj_21, adj_22);
        wp::adj_address(var_indices, var_0, var_1, adj_indices, adj_0, adj_1, adj_21);
        // adj: i = indices[tid, 0]                                                               <L 179>
        wp::adj_copy(var_19, adj_18, adj_20);
        wp::adj_load(var_18, adj_18, adj_19);
        wp::adj_address(var_materials, var_0, var_17, adj_materials, adj_0, adj_17, adj_18);
        // adj: k_lift = materials[tid, 4]                                                        <L 177>
        wp::adj_copy(var_15, adj_14, adj_16);
        wp::adj_load(var_14, adj_14, adj_15);
        wp::adj_address(var_materials, var_0, var_13, adj_materials, adj_0, adj_13, adj_14);
        // adj: k_drag = materials[tid, 3]                                                        <L 176>
        wp::adj_copy(var_11, adj_10, adj_12);
        wp::adj_load(var_10, adj_10, adj_11);
        wp::adj_address(var_materials, var_0, var_9, adj_materials, adj_0, adj_9, adj_10);
        // adj: k_damp = materials[tid, 2]                                                        <L 175>
        wp::adj_copy(var_7, adj_6, adj_8);
        wp::adj_load(var_6, adj_6, adj_7);
        wp::adj_address(var_materials, var_0, var_5, adj_materials, adj_0, adj_5, adj_6);
        // adj: k_lambda = materials[tid, 1]                                                      <L 174>
        wp::adj_copy(var_3, adj_2, adj_4);
        wp::adj_load(var_2, adj_2, adj_3);
        wp::adj_address(var_materials, var_0, var_1, adj_materials, adj_0, adj_1, adj_2);
        // adj: k_mu = materials[tid, 0]                                                          <L 173>
        // adj: tid = wp.tid()                                                                    <L 171>
        // adj: def eval_triangles(                                                               <L 162>
        continue;
    }
}



extern "C" __global__ void eval_triangles_contact_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_num_particles,
    wp::array_t<wp::vec_t<3,wp::float32>> var_x,
    wp::array_t<wp::vec_t<3,wp::float32>> var_v,
    wp::array_t<wp::int32> var_indices,
    wp::array_t<wp::mat_t<2,2,wp::float32>> var_pose,
    wp::array_t<wp::float32> var_activation,
    wp::array_t<wp::float32> var_materials,
    wp::array_t<wp::vec_t<3,wp::float32>> var_f)
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
        const wp::int32 var_3 = 0;
        wp::float32* var_4;
        wp::float32 var_5;
        wp::float32 var_6;
        const wp::int32 var_7 = 1;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        const wp::int32 var_11 = 2;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        const wp::int32 var_15 = 3;
        wp::float32* var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        const wp::int32 var_19 = 4;
        wp::float32* var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::vec_t<3,wp::float32>* var_23;
        wp::vec_t<3,wp::float32> var_24;
        wp::vec_t<3,wp::float32> var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::int32* var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        wp::int32* var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        bool var_35;
        bool var_36;
        bool var_37;
        bool var_38;
        wp::vec_t<3,wp::float32>* var_39;
        wp::vec_t<3,wp::float32> var_40;
        wp::vec_t<3,wp::float32> var_41;
        wp::vec_t<3,wp::float32>* var_42;
        wp::vec_t<3,wp::float32> var_43;
        wp::vec_t<3,wp::float32> var_44;
        wp::vec_t<3,wp::float32>* var_45;
        wp::vec_t<3,wp::float32> var_46;
        wp::vec_t<3,wp::float32> var_47;
        wp::vec_t<3,wp::float32> var_48;
        wp::float32 var_49;
        wp::vec_t<3,wp::float32> var_50;
        wp::float32 var_51;
        wp::vec_t<3,wp::float32> var_52;
        wp::vec_t<3,wp::float32> var_53;
        wp::float32 var_54;
        wp::vec_t<3,wp::float32> var_55;
        wp::vec_t<3,wp::float32> var_56;
        wp::vec_t<3,wp::float32> var_57;
        wp::float32 var_58;
        wp::vec_t<3,wp::float32> var_59;
        const wp::float32 var_60 = 0.01;
        wp::float32 var_61;
        const wp::float32 var_62 = 0.0;
        wp::float32 var_63;
        wp::vec_t<3,wp::float32> var_64;
        const wp::float32 var_65 = 100000.0;
        wp::vec_t<3,wp::float32> var_66;
        wp::vec_t<3,wp::float32> var_67;
        wp::float32 var_68;
        wp::vec_t<3,wp::float32> var_69;
        wp::vec_t<3,wp::float32> var_70;
        wp::float32 var_71;
        wp::vec_t<3,wp::float32> var_72;
        wp::vec_t<3,wp::float32> var_73;
        wp::float32 var_74;
        wp::vec_t<3,wp::float32> var_75;
        wp::vec_t<3,wp::float32> var_76;
        //---------
        // forward
        // def eval_triangles_contact(                                                            <L 356>
        // tid = wp.tid()                                                                         <L 367>
        var_0 = builtin_tid1d();
        // face_no = tid // num_particles  # which face                                           <L 368>
        var_1 = wp::floordiv(var_0, var_num_particles);
        // particle_no = tid % num_particles  # which particle                                    <L 369>
        var_2 = wp::mod(var_0, var_num_particles);
        // k_mu = materials[face_no, 0]                                                           <L 371>
        var_4 = wp::address(var_materials, var_1, var_3);
        var_5 = wp::load(var_4);
        var_6 = wp::copy(var_5);
        // k_lambda = materials[face_no, 1]                                                       <L 372>
        var_8 = wp::address(var_materials, var_1, var_7);
        var_9 = wp::load(var_8);
        var_10 = wp::copy(var_9);
        // k_damp = materials[face_no, 2]                                                         <L 373>
        var_12 = wp::address(var_materials, var_1, var_11);
        var_13 = wp::load(var_12);
        var_14 = wp::copy(var_13);
        // k_drag = materials[face_no, 3]                                                         <L 374>
        var_16 = wp::address(var_materials, var_1, var_15);
        var_17 = wp::load(var_16);
        var_18 = wp::copy(var_17);
        // k_lift = materials[face_no, 4]                                                         <L 375>
        var_20 = wp::address(var_materials, var_1, var_19);
        var_21 = wp::load(var_20);
        var_22 = wp::copy(var_21);
        // pos = x[particle_no]                                                                   <L 378>
        var_23 = wp::address(var_x, var_2);
        var_24 = wp::load(var_23);
        var_25 = wp::copy(var_24);
        // i = indices[face_no, 0]                                                                <L 380>
        var_26 = wp::address(var_indices, var_1, var_3);
        var_27 = wp::load(var_26);
        var_28 = wp::copy(var_27);
        // j = indices[face_no, 1]                                                                <L 381>
        var_29 = wp::address(var_indices, var_1, var_7);
        var_30 = wp::load(var_29);
        var_31 = wp::copy(var_30);
        // k = indices[face_no, 2]                                                                <L 382>
        var_32 = wp::address(var_indices, var_1, var_11);
        var_33 = wp::load(var_32);
        var_34 = wp::copy(var_33);
        // if i == particle_no or j == particle_no or k == particle_no:                           <L 384>
        var_35 = (var_28 == var_2);
        var_36 = (var_31 == var_2);
        var_37 = (var_34 == var_2);
        var_38 = var_35 || var_36 || var_37;
        if (var_38) {
            // return                                                                             <L 385>
            return;
        }
        // p = x[i]  # point zero                                                                 <L 387>
        var_39 = wp::address(var_x, var_28);
        var_40 = wp::load(var_39);
        var_41 = wp::copy(var_40);
        // q = x[j]  # point one                                                                  <L 388>
        var_42 = wp::address(var_x, var_31);
        var_43 = wp::load(var_42);
        var_44 = wp::copy(var_43);
        // r = x[k]  # point two                                                                  <L 389>
        var_45 = wp::address(var_x, var_34);
        var_46 = wp::load(var_45);
        var_47 = wp::copy(var_46);
        // bary = triangle_closest_point_barycentric(p, q, r, pos)                                <L 398>
        var_48 = triangle_closest_point_barycentric(var_41, var_44, var_47, var_25);
        // closest = p * bary[0] + q * bary[1] + r * bary[2]                                      <L 399>
        var_49 = wp::extract(var_48, var_3);
        var_50 = wp::mul(var_41, var_49);
        var_51 = wp::extract(var_48, var_7);
        var_52 = wp::mul(var_44, var_51);
        var_53 = wp::add(var_50, var_52);
        var_54 = wp::extract(var_48, var_11);
        var_55 = wp::mul(var_47, var_54);
        var_56 = wp::add(var_53, var_55);
        // diff = pos - closest                                                                   <L 401>
        var_57 = wp::sub(var_25, var_56);
        // dist = wp.dot(diff, diff)                                                              <L 402>
        var_58 = wp::dot(var_57, var_57);
        // n = wp.normalize(diff)                                                                 <L 403>
        var_59 = wp::normalize(var_57);
        // c = wp.min(dist - 0.01, 0.0)  # 0 unless within 0.01 of surface                        <L 404>
        var_61 = wp::sub(var_58, var_60);
        var_63 = wp::min(var_61, var_62);
        // fn = n * c * 1e5                                                                       <L 406>
        var_64 = wp::mul(var_59, var_63);
        var_66 = wp::mul(var_64, var_65);
        // wp.atomic_sub(f, particle_no, fn)                                                      <L 408>
        var_67 = wp::atomic_sub(var_f, var_2, var_66);
        // wp.atomic_add(f, i, fn * bary[0])                                                      <L 411>
        var_68 = wp::extract(var_48, var_3);
        var_69 = wp::mul(var_66, var_68);
        var_70 = wp::atomic_add(var_f, var_28, var_69);
        // wp.atomic_add(f, j, fn * bary[1])                                                      <L 412>
        var_71 = wp::extract(var_48, var_7);
        var_72 = wp::mul(var_66, var_71);
        var_73 = wp::atomic_add(var_f, var_31, var_72);
        // wp.atomic_add(f, k, fn * bary[2])                                                      <L 413>
        var_74 = wp::extract(var_48, var_11);
        var_75 = wp::mul(var_66, var_74);
        var_76 = wp::atomic_add(var_f, var_34, var_75);
    }
}

extern "C" __global__ void eval_triangles_contact_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::int32 var_num_particles,
    wp::array_t<wp::vec_t<3,wp::float32>> var_x,
    wp::array_t<wp::vec_t<3,wp::float32>> var_v,
    wp::array_t<wp::int32> var_indices,
    wp::array_t<wp::mat_t<2,2,wp::float32>> var_pose,
    wp::array_t<wp::float32> var_activation,
    wp::array_t<wp::float32> var_materials,
    wp::array_t<wp::vec_t<3,wp::float32>> var_f,
    wp::int32 adj_num_particles,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_x,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_v,
    wp::array_t<wp::int32> adj_indices,
    wp::array_t<wp::mat_t<2,2,wp::float32>> adj_pose,
    wp::array_t<wp::float32> adj_activation,
    wp::array_t<wp::float32> adj_materials,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_f)
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
        const wp::int32 var_3 = 0;
        wp::float32* var_4;
        wp::float32 var_5;
        wp::float32 var_6;
        const wp::int32 var_7 = 1;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        const wp::int32 var_11 = 2;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        const wp::int32 var_15 = 3;
        wp::float32* var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        const wp::int32 var_19 = 4;
        wp::float32* var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::vec_t<3,wp::float32>* var_23;
        wp::vec_t<3,wp::float32> var_24;
        wp::vec_t<3,wp::float32> var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::int32* var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        wp::int32* var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        bool var_35;
        bool var_36;
        bool var_37;
        bool var_38;
        wp::vec_t<3,wp::float32>* var_39;
        wp::vec_t<3,wp::float32> var_40;
        wp::vec_t<3,wp::float32> var_41;
        wp::vec_t<3,wp::float32>* var_42;
        wp::vec_t<3,wp::float32> var_43;
        wp::vec_t<3,wp::float32> var_44;
        wp::vec_t<3,wp::float32>* var_45;
        wp::vec_t<3,wp::float32> var_46;
        wp::vec_t<3,wp::float32> var_47;
        wp::vec_t<3,wp::float32> var_48;
        wp::float32 var_49;
        wp::vec_t<3,wp::float32> var_50;
        wp::float32 var_51;
        wp::vec_t<3,wp::float32> var_52;
        wp::vec_t<3,wp::float32> var_53;
        wp::float32 var_54;
        wp::vec_t<3,wp::float32> var_55;
        wp::vec_t<3,wp::float32> var_56;
        wp::vec_t<3,wp::float32> var_57;
        wp::float32 var_58;
        wp::vec_t<3,wp::float32> var_59;
        const wp::float32 var_60 = 0.01;
        wp::float32 var_61;
        const wp::float32 var_62 = 0.0;
        wp::float32 var_63;
        wp::vec_t<3,wp::float32> var_64;
        const wp::float32 var_65 = 100000.0;
        wp::vec_t<3,wp::float32> var_66;
        wp::vec_t<3,wp::float32> var_67;
        wp::float32 var_68;
        wp::vec_t<3,wp::float32> var_69;
        wp::vec_t<3,wp::float32> var_70;
        wp::float32 var_71;
        wp::vec_t<3,wp::float32> var_72;
        wp::vec_t<3,wp::float32> var_73;
        wp::float32 var_74;
        wp::vec_t<3,wp::float32> var_75;
        wp::vec_t<3,wp::float32> var_76;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::float32 adj_4 = {};
        wp::float32 adj_5 = {};
        wp::float32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::float32 adj_16 = {};
        wp::float32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::int32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::vec_t<3,wp::float32> adj_23 = {};
        wp::vec_t<3,wp::float32> adj_24 = {};
        wp::vec_t<3,wp::float32> adj_25 = {};
        wp::int32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::int32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::int32 adj_30 = {};
        wp::int32 adj_31 = {};
        wp::int32 adj_32 = {};
        wp::int32 adj_33 = {};
        wp::int32 adj_34 = {};
        bool adj_35 = {};
        bool adj_36 = {};
        bool adj_37 = {};
        bool adj_38 = {};
        wp::vec_t<3,wp::float32> adj_39 = {};
        wp::vec_t<3,wp::float32> adj_40 = {};
        wp::vec_t<3,wp::float32> adj_41 = {};
        wp::vec_t<3,wp::float32> adj_42 = {};
        wp::vec_t<3,wp::float32> adj_43 = {};
        wp::vec_t<3,wp::float32> adj_44 = {};
        wp::vec_t<3,wp::float32> adj_45 = {};
        wp::vec_t<3,wp::float32> adj_46 = {};
        wp::vec_t<3,wp::float32> adj_47 = {};
        wp::vec_t<3,wp::float32> adj_48 = {};
        wp::float32 adj_49 = {};
        wp::vec_t<3,wp::float32> adj_50 = {};
        wp::float32 adj_51 = {};
        wp::vec_t<3,wp::float32> adj_52 = {};
        wp::vec_t<3,wp::float32> adj_53 = {};
        wp::float32 adj_54 = {};
        wp::vec_t<3,wp::float32> adj_55 = {};
        wp::vec_t<3,wp::float32> adj_56 = {};
        wp::vec_t<3,wp::float32> adj_57 = {};
        wp::float32 adj_58 = {};
        wp::vec_t<3,wp::float32> adj_59 = {};
        wp::float32 adj_60 = {};
        wp::float32 adj_61 = {};
        wp::float32 adj_62 = {};
        wp::float32 adj_63 = {};
        wp::vec_t<3,wp::float32> adj_64 = {};
        wp::float32 adj_65 = {};
        wp::vec_t<3,wp::float32> adj_66 = {};
        wp::vec_t<3,wp::float32> adj_67 = {};
        wp::float32 adj_68 = {};
        wp::vec_t<3,wp::float32> adj_69 = {};
        wp::vec_t<3,wp::float32> adj_70 = {};
        wp::float32 adj_71 = {};
        wp::vec_t<3,wp::float32> adj_72 = {};
        wp::vec_t<3,wp::float32> adj_73 = {};
        wp::float32 adj_74 = {};
        wp::vec_t<3,wp::float32> adj_75 = {};
        wp::vec_t<3,wp::float32> adj_76 = {};
        //---------
        // forward
        // def eval_triangles_contact(                                                            <L 356>
        // tid = wp.tid()                                                                         <L 367>
        var_0 = builtin_tid1d();
        // face_no = tid // num_particles  # which face                                           <L 368>
        var_1 = wp::floordiv(var_0, var_num_particles);
        // particle_no = tid % num_particles  # which particle                                    <L 369>
        var_2 = wp::mod(var_0, var_num_particles);
        // k_mu = materials[face_no, 0]                                                           <L 371>
        var_4 = wp::address(var_materials, var_1, var_3);
        var_5 = wp::load(var_4);
        var_6 = wp::copy(var_5);
        // k_lambda = materials[face_no, 1]                                                       <L 372>
        var_8 = wp::address(var_materials, var_1, var_7);
        var_9 = wp::load(var_8);
        var_10 = wp::copy(var_9);
        // k_damp = materials[face_no, 2]                                                         <L 373>
        var_12 = wp::address(var_materials, var_1, var_11);
        var_13 = wp::load(var_12);
        var_14 = wp::copy(var_13);
        // k_drag = materials[face_no, 3]                                                         <L 374>
        var_16 = wp::address(var_materials, var_1, var_15);
        var_17 = wp::load(var_16);
        var_18 = wp::copy(var_17);
        // k_lift = materials[face_no, 4]                                                         <L 375>
        var_20 = wp::address(var_materials, var_1, var_19);
        var_21 = wp::load(var_20);
        var_22 = wp::copy(var_21);
        // pos = x[particle_no]                                                                   <L 378>
        var_23 = wp::address(var_x, var_2);
        var_24 = wp::load(var_23);
        var_25 = wp::copy(var_24);
        // i = indices[face_no, 0]                                                                <L 380>
        var_26 = wp::address(var_indices, var_1, var_3);
        var_27 = wp::load(var_26);
        var_28 = wp::copy(var_27);
        // j = indices[face_no, 1]                                                                <L 381>
        var_29 = wp::address(var_indices, var_1, var_7);
        var_30 = wp::load(var_29);
        var_31 = wp::copy(var_30);
        // k = indices[face_no, 2]                                                                <L 382>
        var_32 = wp::address(var_indices, var_1, var_11);
        var_33 = wp::load(var_32);
        var_34 = wp::copy(var_33);
        // if i == particle_no or j == particle_no or k == particle_no:                           <L 384>
        var_35 = (var_28 == var_2);
        var_36 = (var_31 == var_2);
        var_37 = (var_34 == var_2);
        var_38 = var_35 || var_36 || var_37;
        if (var_38) {
            // return                                                                             <L 385>
            goto label0;
        }
        // p = x[i]  # point zero                                                                 <L 387>
        var_39 = wp::address(var_x, var_28);
        var_40 = wp::load(var_39);
        var_41 = wp::copy(var_40);
        // q = x[j]  # point one                                                                  <L 388>
        var_42 = wp::address(var_x, var_31);
        var_43 = wp::load(var_42);
        var_44 = wp::copy(var_43);
        // r = x[k]  # point two                                                                  <L 389>
        var_45 = wp::address(var_x, var_34);
        var_46 = wp::load(var_45);
        var_47 = wp::copy(var_46);
        // bary = triangle_closest_point_barycentric(p, q, r, pos)                                <L 398>
        var_48 = triangle_closest_point_barycentric(var_41, var_44, var_47, var_25);
        // closest = p * bary[0] + q * bary[1] + r * bary[2]                                      <L 399>
        var_49 = wp::extract(var_48, var_3);
        var_50 = wp::mul(var_41, var_49);
        var_51 = wp::extract(var_48, var_7);
        var_52 = wp::mul(var_44, var_51);
        var_53 = wp::add(var_50, var_52);
        var_54 = wp::extract(var_48, var_11);
        var_55 = wp::mul(var_47, var_54);
        var_56 = wp::add(var_53, var_55);
        // diff = pos - closest                                                                   <L 401>
        var_57 = wp::sub(var_25, var_56);
        // dist = wp.dot(diff, diff)                                                              <L 402>
        var_58 = wp::dot(var_57, var_57);
        // n = wp.normalize(diff)                                                                 <L 403>
        var_59 = wp::normalize(var_57);
        // c = wp.min(dist - 0.01, 0.0)  # 0 unless within 0.01 of surface                        <L 404>
        var_61 = wp::sub(var_58, var_60);
        var_63 = wp::min(var_61, var_62);
        // fn = n * c * 1e5                                                                       <L 406>
        var_64 = wp::mul(var_59, var_63);
        var_66 = wp::mul(var_64, var_65);
        // wp.atomic_sub(f, particle_no, fn)                                                      <L 408>
        // var_67 = wp::atomic_sub(var_f, var_2, var_66);
        // wp.atomic_add(f, i, fn * bary[0])                                                      <L 411>
        var_68 = wp::extract(var_48, var_3);
        var_69 = wp::mul(var_66, var_68);
        // var_70 = wp::atomic_add(var_f, var_28, var_69);
        // wp.atomic_add(f, j, fn * bary[1])                                                      <L 412>
        var_71 = wp::extract(var_48, var_7);
        var_72 = wp::mul(var_66, var_71);
        // var_73 = wp::atomic_add(var_f, var_31, var_72);
        // wp.atomic_add(f, k, fn * bary[2])                                                      <L 413>
        var_74 = wp::extract(var_48, var_11);
        var_75 = wp::mul(var_66, var_74);
        // var_76 = wp::atomic_add(var_f, var_34, var_75);
        //---------
        // reverse
        wp::adj_atomic_add(var_f, var_34, var_75, adj_f, adj_34, adj_75, adj_76);
        wp::adj_mul(var_66, var_74, adj_66, adj_74, adj_75);
        wp::adj_extract(var_48, var_11, adj_48, adj_11, adj_74);
        // adj: wp.atomic_add(f, k, fn * bary[2])                                                 <L 413>
        wp::adj_atomic_add(var_f, var_31, var_72, adj_f, adj_31, adj_72, adj_73);
        wp::adj_mul(var_66, var_71, adj_66, adj_71, adj_72);
        wp::adj_extract(var_48, var_7, adj_48, adj_7, adj_71);
        // adj: wp.atomic_add(f, j, fn * bary[1])                                                 <L 412>
        wp::adj_atomic_add(var_f, var_28, var_69, adj_f, adj_28, adj_69, adj_70);
        wp::adj_mul(var_66, var_68, adj_66, adj_68, adj_69);
        wp::adj_extract(var_48, var_3, adj_48, adj_3, adj_68);
        // adj: wp.atomic_add(f, i, fn * bary[0])                                                 <L 411>
        wp::adj_atomic_sub(var_f, var_2, var_66, adj_f, adj_2, adj_66, adj_67);
        // adj: wp.atomic_sub(f, particle_no, fn)                                                 <L 408>
        wp::adj_mul(var_64, var_65, adj_64, adj_65, adj_66);
        wp::adj_mul(var_59, var_63, adj_59, adj_63, adj_64);
        // adj: fn = n * c * 1e5                                                                  <L 406>
        wp::adj_min(var_61, var_62, adj_61, adj_62, adj_63);
        wp::adj_sub(var_58, var_60, adj_58, adj_60, adj_61);
        // adj: c = wp.min(dist - 0.01, 0.0)  # 0 unless within 0.01 of surface                   <L 404>
        wp::adj_normalize(var_57, var_59, adj_57, adj_59);
        // adj: n = wp.normalize(diff)                                                            <L 403>
        wp::adj_dot(var_57, var_57, adj_57, adj_57, adj_58);
        // adj: dist = wp.dot(diff, diff)                                                         <L 402>
        wp::adj_sub(var_25, var_56, adj_25, adj_56, adj_57);
        // adj: diff = pos - closest                                                              <L 401>
        wp::adj_add(var_53, var_55, adj_53, adj_55, adj_56);
        wp::adj_mul(var_47, var_54, adj_47, adj_54, adj_55);
        wp::adj_extract(var_48, var_11, adj_48, adj_11, adj_54);
        wp::adj_add(var_50, var_52, adj_50, adj_52, adj_53);
        wp::adj_mul(var_44, var_51, adj_44, adj_51, adj_52);
        wp::adj_extract(var_48, var_7, adj_48, adj_7, adj_51);
        wp::adj_mul(var_41, var_49, adj_41, adj_49, adj_50);
        wp::adj_extract(var_48, var_3, adj_48, adj_3, adj_49);
        // adj: closest = p * bary[0] + q * bary[1] + r * bary[2]                                 <L 399>
        adj_triangle_closest_point_barycentric(var_41, var_44, var_47, var_25, adj_41, adj_44, adj_47, adj_25, adj_48);
        // adj: bary = triangle_closest_point_barycentric(p, q, r, pos)                           <L 398>
        wp::adj_copy(var_46, adj_45, adj_47);
        wp::adj_load(var_45, adj_45, adj_46);
        wp::adj_address(var_x, var_34, adj_x, adj_34, adj_45);
        // adj: r = x[k]  # point two                                                             <L 389>
        wp::adj_copy(var_43, adj_42, adj_44);
        wp::adj_load(var_42, adj_42, adj_43);
        wp::adj_address(var_x, var_31, adj_x, adj_31, adj_42);
        // adj: q = x[j]  # point one                                                             <L 388>
        wp::adj_copy(var_40, adj_39, adj_41);
        wp::adj_load(var_39, adj_39, adj_40);
        wp::adj_address(var_x, var_28, adj_x, adj_28, adj_39);
        // adj: p = x[i]  # point zero                                                            <L 387>
        if (var_38) {
            label0:;
            // adj: return                                                                        <L 385>
        }
        // adj: if i == particle_no or j == particle_no or k == particle_no:                      <L 384>
        wp::adj_copy(var_33, adj_32, adj_34);
        wp::adj_load(var_32, adj_32, adj_33);
        wp::adj_address(var_indices, var_1, var_11, adj_indices, adj_1, adj_11, adj_32);
        // adj: k = indices[face_no, 2]                                                           <L 382>
        wp::adj_copy(var_30, adj_29, adj_31);
        wp::adj_load(var_29, adj_29, adj_30);
        wp::adj_address(var_indices, var_1, var_7, adj_indices, adj_1, adj_7, adj_29);
        // adj: j = indices[face_no, 1]                                                           <L 381>
        wp::adj_copy(var_27, adj_26, adj_28);
        wp::adj_load(var_26, adj_26, adj_27);
        wp::adj_address(var_indices, var_1, var_3, adj_indices, adj_1, adj_3, adj_26);
        // adj: i = indices[face_no, 0]                                                           <L 380>
        wp::adj_copy(var_24, adj_23, adj_25);
        wp::adj_load(var_23, adj_23, adj_24);
        wp::adj_address(var_x, var_2, adj_x, adj_2, adj_23);
        // adj: pos = x[particle_no]                                                              <L 378>
        wp::adj_copy(var_21, adj_20, adj_22);
        wp::adj_load(var_20, adj_20, adj_21);
        wp::adj_address(var_materials, var_1, var_19, adj_materials, adj_1, adj_19, adj_20);
        // adj: k_lift = materials[face_no, 4]                                                    <L 375>
        wp::adj_copy(var_17, adj_16, adj_18);
        wp::adj_load(var_16, adj_16, adj_17);
        wp::adj_address(var_materials, var_1, var_15, adj_materials, adj_1, adj_15, adj_16);
        // adj: k_drag = materials[face_no, 3]                                                    <L 374>
        wp::adj_copy(var_13, adj_12, adj_14);
        wp::adj_load(var_12, adj_12, adj_13);
        wp::adj_address(var_materials, var_1, var_11, adj_materials, adj_1, adj_11, adj_12);
        // adj: k_damp = materials[face_no, 2]                                                    <L 373>
        wp::adj_copy(var_9, adj_8, adj_10);
        wp::adj_load(var_8, adj_8, adj_9);
        wp::adj_address(var_materials, var_1, var_7, adj_materials, adj_1, adj_7, adj_8);
        // adj: k_lambda = materials[face_no, 1]                                                  <L 372>
        wp::adj_copy(var_5, adj_4, adj_6);
        wp::adj_load(var_4, adj_4, adj_5);
        wp::adj_address(var_materials, var_1, var_3, adj_materials, adj_1, adj_3, adj_4);
        // adj: k_mu = materials[face_no, 0]                                                      <L 371>
        wp::adj_mod(var_0, var_num_particles, adj_0, adj_num_particles, adj_2);
        // adj: particle_no = tid % num_particles  # which particle                               <L 369>
        wp::adj_floordiv(var_0, var_num_particles, adj_0, adj_num_particles, adj_1);
        // adj: face_no = tid // num_particles  # which face                                      <L 368>
        // adj: tid = wp.tid()                                                                    <L 367>
        // adj: def eval_triangles_contact(                                                       <L 356>
        continue;
    }
}



extern "C" __global__ void eval_triangles_body_contacts_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_num_particles,
    wp::array_t<wp::vec_t<3,wp::float32>> var_x,
    wp::array_t<wp::vec_t<3,wp::float32>> var_v,
    wp::array_t<wp::int32> var_indices,
    wp::array_t<wp::vec_t<3,wp::float32>> var_body_x,
    wp::array_t<wp::quat_t<wp::float32>> var_body_r,
    wp::array_t<wp::vec_t<3,wp::float32>> var_body_v,
    wp::array_t<wp::vec_t<3,wp::float32>> var_body_w,
    wp::array_t<wp::int32> var_contact_body,
    wp::array_t<wp::vec_t<3,wp::float32>> var_contact_point,
    wp::array_t<wp::float32> var_contact_dist,
    wp::array_t<wp::int32> var_contact_mat,
    wp::array_t<wp::float32> var_materials,
    wp::array_t<wp::vec_t<3,wp::float32>> var_tri_f)
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
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::vec_t<3,wp::float32>* var_6;
        wp::vec_t<3,wp::float32> var_7;
        wp::vec_t<3,wp::float32> var_8;
        wp::float32* var_9;
        wp::float32 var_10;
        wp::float32 var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        const wp::int32 var_15 = 4;
        wp::int32 var_16;
        const wp::int32 var_17 = 0;
        wp::int32 var_18;
        wp::float32* var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::int32 var_22;
        const wp::int32 var_23 = 1;
        wp::int32 var_24;
        wp::float32* var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        wp::int32 var_28;
        const wp::int32 var_29 = 2;
        wp::int32 var_30;
        wp::float32* var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        wp::int32 var_34;
        const wp::int32 var_35 = 3;
        wp::int32 var_36;
        wp::float32* var_37;
        wp::float32 var_38;
        wp::float32 var_39;
        wp::vec_t<3,wp::float32>* var_40;
        wp::vec_t<3,wp::float32> var_41;
        wp::vec_t<3,wp::float32> var_42;
        wp::quat_t<wp::float32>* var_43;
        wp::quat_t<wp::float32> var_44;
        wp::quat_t<wp::float32> var_45;
        wp::vec_t<3,wp::float32>* var_46;
        wp::vec_t<3,wp::float32> var_47;
        wp::vec_t<3,wp::float32> var_48;
        wp::vec_t<3,wp::float32>* var_49;
        wp::vec_t<3,wp::float32> var_50;
        wp::vec_t<3,wp::float32> var_51;
        wp::vec_t<3,wp::float32> var_52;
        wp::vec_t<3,wp::float32> var_53;
        wp::vec_t<3,wp::float32> var_54;
        wp::vec_t<3,wp::float32> var_55;
        wp::vec_t<3,wp::float32> var_56;
        wp::vec_t<3,wp::float32> var_57;
        wp::vec_t<3,wp::float32> var_58;
        wp::vec_t<3,wp::float32> var_59;
        wp::int32 var_60;
        wp::int32 var_61;
        wp::int32* var_62;
        wp::int32 var_63;
        wp::int32 var_64;
        wp::int32 var_65;
        wp::int32 var_66;
        wp::int32* var_67;
        wp::int32 var_68;
        wp::int32 var_69;
        wp::int32 var_70;
        wp::int32 var_71;
        wp::int32* var_72;
        wp::int32 var_73;
        wp::int32 var_74;
        wp::vec_t<3,wp::float32>* var_75;
        wp::vec_t<3,wp::float32> var_76;
        wp::vec_t<3,wp::float32> var_77;
        wp::vec_t<3,wp::float32>* var_78;
        wp::vec_t<3,wp::float32> var_79;
        wp::vec_t<3,wp::float32> var_80;
        wp::vec_t<3,wp::float32>* var_81;
        wp::vec_t<3,wp::float32> var_82;
        wp::vec_t<3,wp::float32> var_83;
        wp::vec_t<3,wp::float32>* var_84;
        wp::vec_t<3,wp::float32> var_85;
        wp::vec_t<3,wp::float32> var_86;
        wp::vec_t<3,wp::float32>* var_87;
        wp::vec_t<3,wp::float32> var_88;
        wp::vec_t<3,wp::float32> var_89;
        wp::vec_t<3,wp::float32>* var_90;
        wp::vec_t<3,wp::float32> var_91;
        wp::vec_t<3,wp::float32> var_92;
        wp::vec_t<3,wp::float32> var_93;
        wp::float32 var_94;
        wp::vec_t<3,wp::float32> var_95;
        wp::float32 var_96;
        wp::vec_t<3,wp::float32> var_97;
        wp::vec_t<3,wp::float32> var_98;
        wp::float32 var_99;
        wp::vec_t<3,wp::float32> var_100;
        wp::vec_t<3,wp::float32> var_101;
        wp::vec_t<3,wp::float32> var_102;
        wp::float32 var_103;
        wp::vec_t<3,wp::float32> var_104;
        const wp::float32 var_105 = 0.05;
        wp::float32 var_106;
        const wp::float32 var_107 = 0.0;
        wp::float32 var_108;
        wp::float32 var_109;
        wp::float32 var_110;
        wp::vec_t<3,wp::float32> var_111;
        wp::float32 var_112;
        wp::vec_t<3,wp::float32> var_113;
        wp::vec_t<3,wp::float32> var_114;
        wp::float32 var_115;
        wp::vec_t<3,wp::float32> var_116;
        wp::vec_t<3,wp::float32> var_117;
        wp::vec_t<3,wp::float32> var_118;
        wp::float32 var_119;
        wp::vec_t<3,wp::float32> var_120;
        wp::vec_t<3,wp::float32> var_121;
        wp::float32 var_122;
        wp::float32 var_123;
        wp::float32 var_124;
        wp::float32 var_125;
        wp::float32 var_126;
        wp::float32 var_127;
        wp::float32 var_128;
        wp::float32 var_129;
        const wp::float32 var_130 = 1.0;
        wp::vec_t<3,wp::float32> var_131;
        wp::vec_t<3,wp::float32> var_132;
        wp::vec_t<3,wp::float32> var_133;
        wp::vec_t<3,wp::float32> var_134;
        wp::vec_t<3,wp::float32> var_135;
        wp::float32 var_136;
        wp::float32 var_137;
        wp::vec_t<3,wp::float32> var_138;
        wp::float32 var_139;
        wp::float32 var_140;
        wp::vec_t<3,wp::float32> var_141;
        wp::vec_t<3,wp::float32> var_142;
        wp::vec_t<3,wp::float32> var_143;
        wp::float32 var_144;
        wp::float32 var_145;
        wp::vec_t<3,wp::float32> var_146;
        wp::float32 var_147;
        wp::vec_t<3,wp::float32> var_148;
        wp::vec_t<3,wp::float32> var_149;
        wp::float32 var_150;
        wp::vec_t<3,wp::float32> var_151;
        wp::vec_t<3,wp::float32> var_152;
        wp::float32 var_153;
        wp::vec_t<3,wp::float32> var_154;
        wp::vec_t<3,wp::float32> var_155;
        wp::float32 var_156;
        wp::vec_t<3,wp::float32> var_157;
        wp::vec_t<3,wp::float32> var_158;
        //---------
        // forward
        // def eval_triangles_body_contacts(                                                      <L 417>
        // tid = wp.tid()                                                                         <L 435>
        var_0 = builtin_tid1d();
        // face_no = tid // num_particles  # which face                                           <L 437>
        var_1 = wp::floordiv(var_0, var_num_particles);
        // particle_no = tid % num_particles  # which particle                                    <L 438>
        var_2 = wp::mod(var_0, var_num_particles);
        // c_body = contact_body[particle_no]                                                     <L 442>
        var_3 = wp::address(var_contact_body, var_2);
        var_4 = wp::load(var_3);
        var_5 = wp::copy(var_4);
        // c_point = contact_point[particle_no]                                                   <L 443>
        var_6 = wp::address(var_contact_point, var_2);
        var_7 = wp::load(var_6);
        var_8 = wp::copy(var_7);
        // c_dist = contact_dist[particle_no]                                                     <L 444>
        var_9 = wp::address(var_contact_dist, var_2);
        var_10 = wp::load(var_9);
        var_11 = wp::copy(var_10);
        // c_mat = contact_mat[particle_no]                                                       <L 445>
        var_12 = wp::address(var_contact_mat, var_2);
        var_13 = wp::load(var_12);
        var_14 = wp::copy(var_13);
        // ke = materials[c_mat * 4 + 0]  # restitution coefficient                               <L 448>
        var_16 = wp::mul(var_14, var_15);
        var_18 = wp::add(var_16, var_17);
        var_19 = wp::address(var_materials, var_18);
        var_20 = wp::load(var_19);
        var_21 = wp::copy(var_20);
        // kd = materials[c_mat * 4 + 1]  # damping coefficient                                   <L 449>
        var_22 = wp::mul(var_14, var_15);
        var_24 = wp::add(var_22, var_23);
        var_25 = wp::address(var_materials, var_24);
        var_26 = wp::load(var_25);
        var_27 = wp::copy(var_26);
        // kf = materials[c_mat * 4 + 2]  # friction coefficient                                  <L 450>
        var_28 = wp::mul(var_14, var_15);
        var_30 = wp::add(var_28, var_29);
        var_31 = wp::address(var_materials, var_30);
        var_32 = wp::load(var_31);
        var_33 = wp::copy(var_32);
        // mu = materials[c_mat * 4 + 3]  # coulomb friction                                      <L 451>
        var_34 = wp::mul(var_14, var_15);
        var_36 = wp::add(var_34, var_35);
        var_37 = wp::address(var_materials, var_36);
        var_38 = wp::load(var_37);
        var_39 = wp::copy(var_38);
        // x0 = body_x[c_body]  # position of colliding body                                      <L 453>
        var_40 = wp::address(var_body_x, var_5);
        var_41 = wp::load(var_40);
        var_42 = wp::copy(var_41);
        // r0 = body_r[c_body]  # orientation of colliding body                                   <L 454>
        var_43 = wp::address(var_body_r, var_5);
        var_44 = wp::load(var_43);
        var_45 = wp::copy(var_44);
        // v0 = body_v[c_body]                                                                    <L 456>
        var_46 = wp::address(var_body_v, var_5);
        var_47 = wp::load(var_46);
        var_48 = wp::copy(var_47);
        // w0 = body_w[c_body]                                                                    <L 457>
        var_49 = wp::address(var_body_w, var_5);
        var_50 = wp::load(var_49);
        var_51 = wp::copy(var_50);
        // pos = x0 + wp.quat_rotate(r0, c_point)                                                 <L 460>
        var_52 = wp::quat_rotate(var_45, var_8);
        var_53 = wp::add(var_42, var_52);
        // r = pos - x0  # basically just c_point in the new coordinates                          <L 464>
        var_54 = wp::sub(var_53, var_42);
        // rhat = wp.normalize(r)                                                                 <L 465>
        var_55 = wp::normalize(var_54);
        // pos = pos + rhat * c_dist  # add on 'thickness' of shape, e.g.: radius of sphere/capsule       <L 466>
        var_56 = wp::mul(var_55, var_11);
        var_57 = wp::add(var_53, var_56);
        // dpdt = v0 + wp.cross(w0, r)  # this is body velocity cross offset, so it's the velocity of the contact point.       <L 469>
        var_58 = wp::cross(var_51, var_54);
        var_59 = wp::add(var_48, var_58);
        // i = indices[face_no * 3 + 0]                                                           <L 473>
        var_60 = wp::mul(var_1, var_35);
        var_61 = wp::add(var_60, var_17);
        var_62 = wp::address(var_indices, var_61);
        var_63 = wp::load(var_62);
        var_64 = wp::copy(var_63);
        // j = indices[face_no * 3 + 1]                                                           <L 474>
        var_65 = wp::mul(var_1, var_35);
        var_66 = wp::add(var_65, var_23);
        var_67 = wp::address(var_indices, var_66);
        var_68 = wp::load(var_67);
        var_69 = wp::copy(var_68);
        // k = indices[face_no * 3 + 2]                                                           <L 475>
        var_70 = wp::mul(var_1, var_35);
        var_71 = wp::add(var_70, var_29);
        var_72 = wp::address(var_indices, var_71);
        var_73 = wp::load(var_72);
        var_74 = wp::copy(var_73);
        // p = x[i]  # point zero                                                                 <L 477>
        var_75 = wp::address(var_x, var_64);
        var_76 = wp::load(var_75);
        var_77 = wp::copy(var_76);
        // q = x[j]  # point one                                                                  <L 478>
        var_78 = wp::address(var_x, var_69);
        var_79 = wp::load(var_78);
        var_80 = wp::copy(var_79);
        // r = x[k]  # point two                                                                  <L 479>
        var_81 = wp::address(var_x, var_74);
        var_82 = wp::load(var_81);
        var_83 = wp::copy(var_82);
        // vp = v[i]  # vel zero                                                                  <L 481>
        var_84 = wp::address(var_v, var_64);
        var_85 = wp::load(var_84);
        var_86 = wp::copy(var_85);
        // vq = v[j]  # vel one                                                                   <L 482>
        var_87 = wp::address(var_v, var_69);
        var_88 = wp::load(var_87);
        var_89 = wp::copy(var_88);
        // vr = v[k]  # vel two                                                                   <L 483>
        var_90 = wp::address(var_v, var_74);
        var_91 = wp::load(var_90);
        var_92 = wp::copy(var_91);
        // bary = triangle_closest_point_barycentric(p, q, r, pos)                                <L 485>
        var_93 = triangle_closest_point_barycentric(var_77, var_80, var_83, var_57);
        // closest = p * bary[0] + q * bary[1] + r * bary[2]                                      <L 486>
        var_94 = wp::extract(var_93, var_17);
        var_95 = wp::mul(var_77, var_94);
        var_96 = wp::extract(var_93, var_23);
        var_97 = wp::mul(var_80, var_96);
        var_98 = wp::add(var_95, var_97);
        var_99 = wp::extract(var_93, var_29);
        var_100 = wp::mul(var_83, var_99);
        var_101 = wp::add(var_98, var_100);
        // diff = pos - closest  # vector from tri to point                                       <L 488>
        var_102 = wp::sub(var_57, var_101);
        // dist = wp.dot(diff, diff)  # squared distance                                          <L 489>
        var_103 = wp::dot(var_102, var_102);
        // n = wp.normalize(diff)  # points into the object                                       <L 490>
        var_104 = wp::normalize(var_102);
        // c = wp.min(dist - 0.05, 0.0)  # 0 unless within 0.05 of surface                        <L 491>
        var_106 = wp::sub(var_103, var_105);
        var_108 = wp::min(var_106, var_107);
        // fn = c * ke  # normal force (restitution coefficient * how far inside for ground) (negative)       <L 497>
        var_109 = wp::mul(var_108, var_21);
        // vtri = vp * bary[0] + vq * bary[1] + vr * bary[2]  # bad approximation for centroid velocity       <L 499>
        var_110 = wp::extract(var_93, var_17);
        var_111 = wp::mul(var_86, var_110);
        var_112 = wp::extract(var_93, var_23);
        var_113 = wp::mul(var_89, var_112);
        var_114 = wp::add(var_111, var_113);
        var_115 = wp::extract(var_93, var_29);
        var_116 = wp::mul(var_92, var_115);
        var_117 = wp::add(var_114, var_116);
        // vrel = vtri - dpdt                                                                     <L 500>
        var_118 = wp::sub(var_117, var_59);
        // vn = dot(n, vrel)  # velocity component of body in negative normal direction           <L 502>
        var_119 = wp::dot(var_104, var_118);
        // vt = vrel - n * vn  # velocity component not in normal direction                       <L 503>
        var_120 = wp::mul(var_104, var_119);
        var_121 = wp::sub(var_118, var_120);
        // fd = 0.0 - wp.max(vn, 0.0) * kd * wp.step(c)  # again, negative, into the ground       <L 506>
        var_122 = wp::max(var_119, var_107);
        var_123 = wp::mul(var_122, var_27);
        var_124 = wp::step(var_108);
        var_125 = wp::mul(var_123, var_124);
        var_126 = wp::sub(var_107, var_125);
        // lower = mu * (fn + fd)                                                                 <L 512>
        var_127 = wp::add(var_109, var_126);
        var_128 = wp::mul(var_39, var_127);
        // upper = 0.0 - lower  # workaround because no unary ops yet                             <L 513>
        var_129 = wp::sub(var_107, var_128);
        // nx = cross(n, vec3(0.0, 0.0, 1.0))  # basis vectors for tangent                        <L 515>
        var_131 = wp::vec_t<3,wp::float32>(var_107, var_107, var_130);
        var_132 = wp::cross(var_104, var_131);
        // nz = cross(n, vec3(1.0, 0.0, 0.0))                                                     <L 516>
        var_133 = wp::vec_t<3,wp::float32>(var_130, var_107, var_107);
        var_134 = wp::cross(var_104, var_133);
        // vx = wp.clamp(dot(nx * kf, vt), lower, upper)                                          <L 518>
        var_135 = wp::mul(var_132, var_33);
        var_136 = wp::dot(var_135, var_121);
        var_137 = wp::clamp(var_136, var_128, var_129);
        // vz = wp.clamp(dot(nz * kf, vt), lower, upper)                                          <L 519>
        var_138 = wp::mul(var_134, var_33);
        var_139 = wp::dot(var_138, var_121);
        var_140 = wp::clamp(var_139, var_128, var_129);
        // ft = (nx * vx + nz * vz) * (0.0 - wp.step(c))  # wp.vec3(vx, 0.0, vz)*wp.step(c)       <L 521>
        var_141 = wp::mul(var_132, var_137);
        var_142 = wp::mul(var_134, var_140);
        var_143 = wp::add(var_141, var_142);
        var_144 = wp::step(var_108);
        var_145 = wp::sub(var_107, var_144);
        var_146 = wp::mul(var_143, var_145);
        // f_total = n * (fn + fd) + ft                                                           <L 526>
        var_147 = wp::add(var_109, var_126);
        var_148 = wp::mul(var_104, var_147);
        var_149 = wp::add(var_148, var_146);
        // wp.atomic_add(tri_f, i, f_total * bary[0])                                             <L 528>
        var_150 = wp::extract(var_93, var_17);
        var_151 = wp::mul(var_149, var_150);
        var_152 = wp::atomic_add(var_tri_f, var_64, var_151);
        // wp.atomic_add(tri_f, j, f_total * bary[1])                                             <L 529>
        var_153 = wp::extract(var_93, var_23);
        var_154 = wp::mul(var_149, var_153);
        var_155 = wp::atomic_add(var_tri_f, var_69, var_154);
        // wp.atomic_add(tri_f, k, f_total * bary[2])                                             <L 530>
        var_156 = wp::extract(var_93, var_29);
        var_157 = wp::mul(var_149, var_156);
        var_158 = wp::atomic_add(var_tri_f, var_74, var_157);
    }
}

extern "C" __global__ void eval_triangles_body_contacts_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::int32 var_num_particles,
    wp::array_t<wp::vec_t<3,wp::float32>> var_x,
    wp::array_t<wp::vec_t<3,wp::float32>> var_v,
    wp::array_t<wp::int32> var_indices,
    wp::array_t<wp::vec_t<3,wp::float32>> var_body_x,
    wp::array_t<wp::quat_t<wp::float32>> var_body_r,
    wp::array_t<wp::vec_t<3,wp::float32>> var_body_v,
    wp::array_t<wp::vec_t<3,wp::float32>> var_body_w,
    wp::array_t<wp::int32> var_contact_body,
    wp::array_t<wp::vec_t<3,wp::float32>> var_contact_point,
    wp::array_t<wp::float32> var_contact_dist,
    wp::array_t<wp::int32> var_contact_mat,
    wp::array_t<wp::float32> var_materials,
    wp::array_t<wp::vec_t<3,wp::float32>> var_tri_f,
    wp::int32 adj_num_particles,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_x,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_v,
    wp::array_t<wp::int32> adj_indices,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_body_x,
    wp::array_t<wp::quat_t<wp::float32>> adj_body_r,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_body_v,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_body_w,
    wp::array_t<wp::int32> adj_contact_body,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_contact_point,
    wp::array_t<wp::float32> adj_contact_dist,
    wp::array_t<wp::int32> adj_contact_mat,
    wp::array_t<wp::float32> adj_materials,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_tri_f)
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
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::vec_t<3,wp::float32>* var_6;
        wp::vec_t<3,wp::float32> var_7;
        wp::vec_t<3,wp::float32> var_8;
        wp::float32* var_9;
        wp::float32 var_10;
        wp::float32 var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        const wp::int32 var_15 = 4;
        wp::int32 var_16;
        const wp::int32 var_17 = 0;
        wp::int32 var_18;
        wp::float32* var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::int32 var_22;
        const wp::int32 var_23 = 1;
        wp::int32 var_24;
        wp::float32* var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        wp::int32 var_28;
        const wp::int32 var_29 = 2;
        wp::int32 var_30;
        wp::float32* var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        wp::int32 var_34;
        const wp::int32 var_35 = 3;
        wp::int32 var_36;
        wp::float32* var_37;
        wp::float32 var_38;
        wp::float32 var_39;
        wp::vec_t<3,wp::float32>* var_40;
        wp::vec_t<3,wp::float32> var_41;
        wp::vec_t<3,wp::float32> var_42;
        wp::quat_t<wp::float32>* var_43;
        wp::quat_t<wp::float32> var_44;
        wp::quat_t<wp::float32> var_45;
        wp::vec_t<3,wp::float32>* var_46;
        wp::vec_t<3,wp::float32> var_47;
        wp::vec_t<3,wp::float32> var_48;
        wp::vec_t<3,wp::float32>* var_49;
        wp::vec_t<3,wp::float32> var_50;
        wp::vec_t<3,wp::float32> var_51;
        wp::vec_t<3,wp::float32> var_52;
        wp::vec_t<3,wp::float32> var_53;
        wp::vec_t<3,wp::float32> var_54;
        wp::vec_t<3,wp::float32> var_55;
        wp::vec_t<3,wp::float32> var_56;
        wp::vec_t<3,wp::float32> var_57;
        wp::vec_t<3,wp::float32> var_58;
        wp::vec_t<3,wp::float32> var_59;
        wp::int32 var_60;
        wp::int32 var_61;
        wp::int32* var_62;
        wp::int32 var_63;
        wp::int32 var_64;
        wp::int32 var_65;
        wp::int32 var_66;
        wp::int32* var_67;
        wp::int32 var_68;
        wp::int32 var_69;
        wp::int32 var_70;
        wp::int32 var_71;
        wp::int32* var_72;
        wp::int32 var_73;
        wp::int32 var_74;
        wp::vec_t<3,wp::float32>* var_75;
        wp::vec_t<3,wp::float32> var_76;
        wp::vec_t<3,wp::float32> var_77;
        wp::vec_t<3,wp::float32>* var_78;
        wp::vec_t<3,wp::float32> var_79;
        wp::vec_t<3,wp::float32> var_80;
        wp::vec_t<3,wp::float32>* var_81;
        wp::vec_t<3,wp::float32> var_82;
        wp::vec_t<3,wp::float32> var_83;
        wp::vec_t<3,wp::float32>* var_84;
        wp::vec_t<3,wp::float32> var_85;
        wp::vec_t<3,wp::float32> var_86;
        wp::vec_t<3,wp::float32>* var_87;
        wp::vec_t<3,wp::float32> var_88;
        wp::vec_t<3,wp::float32> var_89;
        wp::vec_t<3,wp::float32>* var_90;
        wp::vec_t<3,wp::float32> var_91;
        wp::vec_t<3,wp::float32> var_92;
        wp::vec_t<3,wp::float32> var_93;
        wp::float32 var_94;
        wp::vec_t<3,wp::float32> var_95;
        wp::float32 var_96;
        wp::vec_t<3,wp::float32> var_97;
        wp::vec_t<3,wp::float32> var_98;
        wp::float32 var_99;
        wp::vec_t<3,wp::float32> var_100;
        wp::vec_t<3,wp::float32> var_101;
        wp::vec_t<3,wp::float32> var_102;
        wp::float32 var_103;
        wp::vec_t<3,wp::float32> var_104;
        const wp::float32 var_105 = 0.05;
        wp::float32 var_106;
        const wp::float32 var_107 = 0.0;
        wp::float32 var_108;
        wp::float32 var_109;
        wp::float32 var_110;
        wp::vec_t<3,wp::float32> var_111;
        wp::float32 var_112;
        wp::vec_t<3,wp::float32> var_113;
        wp::vec_t<3,wp::float32> var_114;
        wp::float32 var_115;
        wp::vec_t<3,wp::float32> var_116;
        wp::vec_t<3,wp::float32> var_117;
        wp::vec_t<3,wp::float32> var_118;
        wp::float32 var_119;
        wp::vec_t<3,wp::float32> var_120;
        wp::vec_t<3,wp::float32> var_121;
        wp::float32 var_122;
        wp::float32 var_123;
        wp::float32 var_124;
        wp::float32 var_125;
        wp::float32 var_126;
        wp::float32 var_127;
        wp::float32 var_128;
        wp::float32 var_129;
        const wp::float32 var_130 = 1.0;
        wp::vec_t<3,wp::float32> var_131;
        wp::vec_t<3,wp::float32> var_132;
        wp::vec_t<3,wp::float32> var_133;
        wp::vec_t<3,wp::float32> var_134;
        wp::vec_t<3,wp::float32> var_135;
        wp::float32 var_136;
        wp::float32 var_137;
        wp::vec_t<3,wp::float32> var_138;
        wp::float32 var_139;
        wp::float32 var_140;
        wp::vec_t<3,wp::float32> var_141;
        wp::vec_t<3,wp::float32> var_142;
        wp::vec_t<3,wp::float32> var_143;
        wp::float32 var_144;
        wp::float32 var_145;
        wp::vec_t<3,wp::float32> var_146;
        wp::float32 var_147;
        wp::vec_t<3,wp::float32> var_148;
        wp::vec_t<3,wp::float32> var_149;
        wp::float32 var_150;
        wp::vec_t<3,wp::float32> var_151;
        wp::vec_t<3,wp::float32> var_152;
        wp::float32 var_153;
        wp::vec_t<3,wp::float32> var_154;
        wp::vec_t<3,wp::float32> var_155;
        wp::float32 var_156;
        wp::vec_t<3,wp::float32> var_157;
        wp::vec_t<3,wp::float32> var_158;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::vec_t<3,wp::float32> adj_6 = {};
        wp::vec_t<3,wp::float32> adj_7 = {};
        wp::vec_t<3,wp::float32> adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::int32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::float32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::float32 adj_27 = {};
        wp::int32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::int32 adj_30 = {};
        wp::float32 adj_31 = {};
        wp::float32 adj_32 = {};
        wp::float32 adj_33 = {};
        wp::int32 adj_34 = {};
        wp::int32 adj_35 = {};
        wp::int32 adj_36 = {};
        wp::float32 adj_37 = {};
        wp::float32 adj_38 = {};
        wp::float32 adj_39 = {};
        wp::vec_t<3,wp::float32> adj_40 = {};
        wp::vec_t<3,wp::float32> adj_41 = {};
        wp::vec_t<3,wp::float32> adj_42 = {};
        wp::quat_t<wp::float32> adj_43 = {};
        wp::quat_t<wp::float32> adj_44 = {};
        wp::quat_t<wp::float32> adj_45 = {};
        wp::vec_t<3,wp::float32> adj_46 = {};
        wp::vec_t<3,wp::float32> adj_47 = {};
        wp::vec_t<3,wp::float32> adj_48 = {};
        wp::vec_t<3,wp::float32> adj_49 = {};
        wp::vec_t<3,wp::float32> adj_50 = {};
        wp::vec_t<3,wp::float32> adj_51 = {};
        wp::vec_t<3,wp::float32> adj_52 = {};
        wp::vec_t<3,wp::float32> adj_53 = {};
        wp::vec_t<3,wp::float32> adj_54 = {};
        wp::vec_t<3,wp::float32> adj_55 = {};
        wp::vec_t<3,wp::float32> adj_56 = {};
        wp::vec_t<3,wp::float32> adj_57 = {};
        wp::vec_t<3,wp::float32> adj_58 = {};
        wp::vec_t<3,wp::float32> adj_59 = {};
        wp::int32 adj_60 = {};
        wp::int32 adj_61 = {};
        wp::int32 adj_62 = {};
        wp::int32 adj_63 = {};
        wp::int32 adj_64 = {};
        wp::int32 adj_65 = {};
        wp::int32 adj_66 = {};
        wp::int32 adj_67 = {};
        wp::int32 adj_68 = {};
        wp::int32 adj_69 = {};
        wp::int32 adj_70 = {};
        wp::int32 adj_71 = {};
        wp::int32 adj_72 = {};
        wp::int32 adj_73 = {};
        wp::int32 adj_74 = {};
        wp::vec_t<3,wp::float32> adj_75 = {};
        wp::vec_t<3,wp::float32> adj_76 = {};
        wp::vec_t<3,wp::float32> adj_77 = {};
        wp::vec_t<3,wp::float32> adj_78 = {};
        wp::vec_t<3,wp::float32> adj_79 = {};
        wp::vec_t<3,wp::float32> adj_80 = {};
        wp::vec_t<3,wp::float32> adj_81 = {};
        wp::vec_t<3,wp::float32> adj_82 = {};
        wp::vec_t<3,wp::float32> adj_83 = {};
        wp::vec_t<3,wp::float32> adj_84 = {};
        wp::vec_t<3,wp::float32> adj_85 = {};
        wp::vec_t<3,wp::float32> adj_86 = {};
        wp::vec_t<3,wp::float32> adj_87 = {};
        wp::vec_t<3,wp::float32> adj_88 = {};
        wp::vec_t<3,wp::float32> adj_89 = {};
        wp::vec_t<3,wp::float32> adj_90 = {};
        wp::vec_t<3,wp::float32> adj_91 = {};
        wp::vec_t<3,wp::float32> adj_92 = {};
        wp::vec_t<3,wp::float32> adj_93 = {};
        wp::float32 adj_94 = {};
        wp::vec_t<3,wp::float32> adj_95 = {};
        wp::float32 adj_96 = {};
        wp::vec_t<3,wp::float32> adj_97 = {};
        wp::vec_t<3,wp::float32> adj_98 = {};
        wp::float32 adj_99 = {};
        wp::vec_t<3,wp::float32> adj_100 = {};
        wp::vec_t<3,wp::float32> adj_101 = {};
        wp::vec_t<3,wp::float32> adj_102 = {};
        wp::float32 adj_103 = {};
        wp::vec_t<3,wp::float32> adj_104 = {};
        wp::float32 adj_105 = {};
        wp::float32 adj_106 = {};
        wp::float32 adj_107 = {};
        wp::float32 adj_108 = {};
        wp::float32 adj_109 = {};
        wp::float32 adj_110 = {};
        wp::vec_t<3,wp::float32> adj_111 = {};
        wp::float32 adj_112 = {};
        wp::vec_t<3,wp::float32> adj_113 = {};
        wp::vec_t<3,wp::float32> adj_114 = {};
        wp::float32 adj_115 = {};
        wp::vec_t<3,wp::float32> adj_116 = {};
        wp::vec_t<3,wp::float32> adj_117 = {};
        wp::vec_t<3,wp::float32> adj_118 = {};
        wp::float32 adj_119 = {};
        wp::vec_t<3,wp::float32> adj_120 = {};
        wp::vec_t<3,wp::float32> adj_121 = {};
        wp::float32 adj_122 = {};
        wp::float32 adj_123 = {};
        wp::float32 adj_124 = {};
        wp::float32 adj_125 = {};
        wp::float32 adj_126 = {};
        wp::float32 adj_127 = {};
        wp::float32 adj_128 = {};
        wp::float32 adj_129 = {};
        wp::float32 adj_130 = {};
        wp::vec_t<3,wp::float32> adj_131 = {};
        wp::vec_t<3,wp::float32> adj_132 = {};
        wp::vec_t<3,wp::float32> adj_133 = {};
        wp::vec_t<3,wp::float32> adj_134 = {};
        wp::vec_t<3,wp::float32> adj_135 = {};
        wp::float32 adj_136 = {};
        wp::float32 adj_137 = {};
        wp::vec_t<3,wp::float32> adj_138 = {};
        wp::float32 adj_139 = {};
        wp::float32 adj_140 = {};
        wp::vec_t<3,wp::float32> adj_141 = {};
        wp::vec_t<3,wp::float32> adj_142 = {};
        wp::vec_t<3,wp::float32> adj_143 = {};
        wp::float32 adj_144 = {};
        wp::float32 adj_145 = {};
        wp::vec_t<3,wp::float32> adj_146 = {};
        wp::float32 adj_147 = {};
        wp::vec_t<3,wp::float32> adj_148 = {};
        wp::vec_t<3,wp::float32> adj_149 = {};
        wp::float32 adj_150 = {};
        wp::vec_t<3,wp::float32> adj_151 = {};
        wp::vec_t<3,wp::float32> adj_152 = {};
        wp::float32 adj_153 = {};
        wp::vec_t<3,wp::float32> adj_154 = {};
        wp::vec_t<3,wp::float32> adj_155 = {};
        wp::float32 adj_156 = {};
        wp::vec_t<3,wp::float32> adj_157 = {};
        wp::vec_t<3,wp::float32> adj_158 = {};
        //---------
        // forward
        // def eval_triangles_body_contacts(                                                      <L 417>
        // tid = wp.tid()                                                                         <L 435>
        var_0 = builtin_tid1d();
        // face_no = tid // num_particles  # which face                                           <L 437>
        var_1 = wp::floordiv(var_0, var_num_particles);
        // particle_no = tid % num_particles  # which particle                                    <L 438>
        var_2 = wp::mod(var_0, var_num_particles);
        // c_body = contact_body[particle_no]                                                     <L 442>
        var_3 = wp::address(var_contact_body, var_2);
        var_4 = wp::load(var_3);
        var_5 = wp::copy(var_4);
        // c_point = contact_point[particle_no]                                                   <L 443>
        var_6 = wp::address(var_contact_point, var_2);
        var_7 = wp::load(var_6);
        var_8 = wp::copy(var_7);
        // c_dist = contact_dist[particle_no]                                                     <L 444>
        var_9 = wp::address(var_contact_dist, var_2);
        var_10 = wp::load(var_9);
        var_11 = wp::copy(var_10);
        // c_mat = contact_mat[particle_no]                                                       <L 445>
        var_12 = wp::address(var_contact_mat, var_2);
        var_13 = wp::load(var_12);
        var_14 = wp::copy(var_13);
        // ke = materials[c_mat * 4 + 0]  # restitution coefficient                               <L 448>
        var_16 = wp::mul(var_14, var_15);
        var_18 = wp::add(var_16, var_17);
        var_19 = wp::address(var_materials, var_18);
        var_20 = wp::load(var_19);
        var_21 = wp::copy(var_20);
        // kd = materials[c_mat * 4 + 1]  # damping coefficient                                   <L 449>
        var_22 = wp::mul(var_14, var_15);
        var_24 = wp::add(var_22, var_23);
        var_25 = wp::address(var_materials, var_24);
        var_26 = wp::load(var_25);
        var_27 = wp::copy(var_26);
        // kf = materials[c_mat * 4 + 2]  # friction coefficient                                  <L 450>
        var_28 = wp::mul(var_14, var_15);
        var_30 = wp::add(var_28, var_29);
        var_31 = wp::address(var_materials, var_30);
        var_32 = wp::load(var_31);
        var_33 = wp::copy(var_32);
        // mu = materials[c_mat * 4 + 3]  # coulomb friction                                      <L 451>
        var_34 = wp::mul(var_14, var_15);
        var_36 = wp::add(var_34, var_35);
        var_37 = wp::address(var_materials, var_36);
        var_38 = wp::load(var_37);
        var_39 = wp::copy(var_38);
        // x0 = body_x[c_body]  # position of colliding body                                      <L 453>
        var_40 = wp::address(var_body_x, var_5);
        var_41 = wp::load(var_40);
        var_42 = wp::copy(var_41);
        // r0 = body_r[c_body]  # orientation of colliding body                                   <L 454>
        var_43 = wp::address(var_body_r, var_5);
        var_44 = wp::load(var_43);
        var_45 = wp::copy(var_44);
        // v0 = body_v[c_body]                                                                    <L 456>
        var_46 = wp::address(var_body_v, var_5);
        var_47 = wp::load(var_46);
        var_48 = wp::copy(var_47);
        // w0 = body_w[c_body]                                                                    <L 457>
        var_49 = wp::address(var_body_w, var_5);
        var_50 = wp::load(var_49);
        var_51 = wp::copy(var_50);
        // pos = x0 + wp.quat_rotate(r0, c_point)                                                 <L 460>
        var_52 = wp::quat_rotate(var_45, var_8);
        var_53 = wp::add(var_42, var_52);
        // r = pos - x0  # basically just c_point in the new coordinates                          <L 464>
        var_54 = wp::sub(var_53, var_42);
        // rhat = wp.normalize(r)                                                                 <L 465>
        var_55 = wp::normalize(var_54);
        // pos = pos + rhat * c_dist  # add on 'thickness' of shape, e.g.: radius of sphere/capsule       <L 466>
        var_56 = wp::mul(var_55, var_11);
        var_57 = wp::add(var_53, var_56);
        // dpdt = v0 + wp.cross(w0, r)  # this is body velocity cross offset, so it's the velocity of the contact point.       <L 469>
        var_58 = wp::cross(var_51, var_54);
        var_59 = wp::add(var_48, var_58);
        // i = indices[face_no * 3 + 0]                                                           <L 473>
        var_60 = wp::mul(var_1, var_35);
        var_61 = wp::add(var_60, var_17);
        var_62 = wp::address(var_indices, var_61);
        var_63 = wp::load(var_62);
        var_64 = wp::copy(var_63);
        // j = indices[face_no * 3 + 1]                                                           <L 474>
        var_65 = wp::mul(var_1, var_35);
        var_66 = wp::add(var_65, var_23);
        var_67 = wp::address(var_indices, var_66);
        var_68 = wp::load(var_67);
        var_69 = wp::copy(var_68);
        // k = indices[face_no * 3 + 2]                                                           <L 475>
        var_70 = wp::mul(var_1, var_35);
        var_71 = wp::add(var_70, var_29);
        var_72 = wp::address(var_indices, var_71);
        var_73 = wp::load(var_72);
        var_74 = wp::copy(var_73);
        // p = x[i]  # point zero                                                                 <L 477>
        var_75 = wp::address(var_x, var_64);
        var_76 = wp::load(var_75);
        var_77 = wp::copy(var_76);
        // q = x[j]  # point one                                                                  <L 478>
        var_78 = wp::address(var_x, var_69);
        var_79 = wp::load(var_78);
        var_80 = wp::copy(var_79);
        // r = x[k]  # point two                                                                  <L 479>
        var_81 = wp::address(var_x, var_74);
        var_82 = wp::load(var_81);
        var_83 = wp::copy(var_82);
        // vp = v[i]  # vel zero                                                                  <L 481>
        var_84 = wp::address(var_v, var_64);
        var_85 = wp::load(var_84);
        var_86 = wp::copy(var_85);
        // vq = v[j]  # vel one                                                                   <L 482>
        var_87 = wp::address(var_v, var_69);
        var_88 = wp::load(var_87);
        var_89 = wp::copy(var_88);
        // vr = v[k]  # vel two                                                                   <L 483>
        var_90 = wp::address(var_v, var_74);
        var_91 = wp::load(var_90);
        var_92 = wp::copy(var_91);
        // bary = triangle_closest_point_barycentric(p, q, r, pos)                                <L 485>
        var_93 = triangle_closest_point_barycentric(var_77, var_80, var_83, var_57);
        // closest = p * bary[0] + q * bary[1] + r * bary[2]                                      <L 486>
        var_94 = wp::extract(var_93, var_17);
        var_95 = wp::mul(var_77, var_94);
        var_96 = wp::extract(var_93, var_23);
        var_97 = wp::mul(var_80, var_96);
        var_98 = wp::add(var_95, var_97);
        var_99 = wp::extract(var_93, var_29);
        var_100 = wp::mul(var_83, var_99);
        var_101 = wp::add(var_98, var_100);
        // diff = pos - closest  # vector from tri to point                                       <L 488>
        var_102 = wp::sub(var_57, var_101);
        // dist = wp.dot(diff, diff)  # squared distance                                          <L 489>
        var_103 = wp::dot(var_102, var_102);
        // n = wp.normalize(diff)  # points into the object                                       <L 490>
        var_104 = wp::normalize(var_102);
        // c = wp.min(dist - 0.05, 0.0)  # 0 unless within 0.05 of surface                        <L 491>
        var_106 = wp::sub(var_103, var_105);
        var_108 = wp::min(var_106, var_107);
        // fn = c * ke  # normal force (restitution coefficient * how far inside for ground) (negative)       <L 497>
        var_109 = wp::mul(var_108, var_21);
        // vtri = vp * bary[0] + vq * bary[1] + vr * bary[2]  # bad approximation for centroid velocity       <L 499>
        var_110 = wp::extract(var_93, var_17);
        var_111 = wp::mul(var_86, var_110);
        var_112 = wp::extract(var_93, var_23);
        var_113 = wp::mul(var_89, var_112);
        var_114 = wp::add(var_111, var_113);
        var_115 = wp::extract(var_93, var_29);
        var_116 = wp::mul(var_92, var_115);
        var_117 = wp::add(var_114, var_116);
        // vrel = vtri - dpdt                                                                     <L 500>
        var_118 = wp::sub(var_117, var_59);
        // vn = dot(n, vrel)  # velocity component of body in negative normal direction           <L 502>
        var_119 = wp::dot(var_104, var_118);
        // vt = vrel - n * vn  # velocity component not in normal direction                       <L 503>
        var_120 = wp::mul(var_104, var_119);
        var_121 = wp::sub(var_118, var_120);
        // fd = 0.0 - wp.max(vn, 0.0) * kd * wp.step(c)  # again, negative, into the ground       <L 506>
        var_122 = wp::max(var_119, var_107);
        var_123 = wp::mul(var_122, var_27);
        var_124 = wp::step(var_108);
        var_125 = wp::mul(var_123, var_124);
        var_126 = wp::sub(var_107, var_125);
        // lower = mu * (fn + fd)                                                                 <L 512>
        var_127 = wp::add(var_109, var_126);
        var_128 = wp::mul(var_39, var_127);
        // upper = 0.0 - lower  # workaround because no unary ops yet                             <L 513>
        var_129 = wp::sub(var_107, var_128);
        // nx = cross(n, vec3(0.0, 0.0, 1.0))  # basis vectors for tangent                        <L 515>
        var_131 = wp::vec_t<3,wp::float32>(var_107, var_107, var_130);
        var_132 = wp::cross(var_104, var_131);
        // nz = cross(n, vec3(1.0, 0.0, 0.0))                                                     <L 516>
        var_133 = wp::vec_t<3,wp::float32>(var_130, var_107, var_107);
        var_134 = wp::cross(var_104, var_133);
        // vx = wp.clamp(dot(nx * kf, vt), lower, upper)                                          <L 518>
        var_135 = wp::mul(var_132, var_33);
        var_136 = wp::dot(var_135, var_121);
        var_137 = wp::clamp(var_136, var_128, var_129);
        // vz = wp.clamp(dot(nz * kf, vt), lower, upper)                                          <L 519>
        var_138 = wp::mul(var_134, var_33);
        var_139 = wp::dot(var_138, var_121);
        var_140 = wp::clamp(var_139, var_128, var_129);
        // ft = (nx * vx + nz * vz) * (0.0 - wp.step(c))  # wp.vec3(vx, 0.0, vz)*wp.step(c)       <L 521>
        var_141 = wp::mul(var_132, var_137);
        var_142 = wp::mul(var_134, var_140);
        var_143 = wp::add(var_141, var_142);
        var_144 = wp::step(var_108);
        var_145 = wp::sub(var_107, var_144);
        var_146 = wp::mul(var_143, var_145);
        // f_total = n * (fn + fd) + ft                                                           <L 526>
        var_147 = wp::add(var_109, var_126);
        var_148 = wp::mul(var_104, var_147);
        var_149 = wp::add(var_148, var_146);
        // wp.atomic_add(tri_f, i, f_total * bary[0])                                             <L 528>
        var_150 = wp::extract(var_93, var_17);
        var_151 = wp::mul(var_149, var_150);
        // var_152 = wp::atomic_add(var_tri_f, var_64, var_151);
        // wp.atomic_add(tri_f, j, f_total * bary[1])                                             <L 529>
        var_153 = wp::extract(var_93, var_23);
        var_154 = wp::mul(var_149, var_153);
        // var_155 = wp::atomic_add(var_tri_f, var_69, var_154);
        // wp.atomic_add(tri_f, k, f_total * bary[2])                                             <L 530>
        var_156 = wp::extract(var_93, var_29);
        var_157 = wp::mul(var_149, var_156);
        // var_158 = wp::atomic_add(var_tri_f, var_74, var_157);
        //---------
        // reverse
        wp::adj_atomic_add(var_tri_f, var_74, var_157, adj_tri_f, adj_74, adj_157, adj_158);
        wp::adj_mul(var_149, var_156, adj_149, adj_156, adj_157);
        wp::adj_extract(var_93, var_29, adj_93, adj_29, adj_156);
        // adj: wp.atomic_add(tri_f, k, f_total * bary[2])                                        <L 530>
        wp::adj_atomic_add(var_tri_f, var_69, var_154, adj_tri_f, adj_69, adj_154, adj_155);
        wp::adj_mul(var_149, var_153, adj_149, adj_153, adj_154);
        wp::adj_extract(var_93, var_23, adj_93, adj_23, adj_153);
        // adj: wp.atomic_add(tri_f, j, f_total * bary[1])                                        <L 529>
        wp::adj_atomic_add(var_tri_f, var_64, var_151, adj_tri_f, adj_64, adj_151, adj_152);
        wp::adj_mul(var_149, var_150, adj_149, adj_150, adj_151);
        wp::adj_extract(var_93, var_17, adj_93, adj_17, adj_150);
        // adj: wp.atomic_add(tri_f, i, f_total * bary[0])                                        <L 528>
        wp::adj_add(var_148, var_146, adj_148, adj_146, adj_149);
        wp::adj_mul(var_104, var_147, adj_104, adj_147, adj_148);
        wp::adj_add(var_109, var_126, adj_109, adj_126, adj_147);
        // adj: f_total = n * (fn + fd) + ft                                                      <L 526>
        wp::adj_mul(var_143, var_145, adj_143, adj_145, adj_146);
        wp::adj_sub(var_107, var_144, adj_107, adj_144, adj_145);
        wp::adj_step(var_108, adj_108, adj_144);
        wp::adj_add(var_141, var_142, adj_141, adj_142, adj_143);
        wp::adj_mul(var_134, var_140, adj_134, adj_140, adj_142);
        wp::adj_mul(var_132, var_137, adj_132, adj_137, adj_141);
        // adj: ft = (nx * vx + nz * vz) * (0.0 - wp.step(c))  # wp.vec3(vx, 0.0, vz)*wp.step(c)  <L 521>
        wp::adj_clamp(var_139, var_128, var_129, adj_139, adj_128, adj_129, adj_140);
        wp::adj_dot(var_138, var_121, adj_138, adj_121, adj_139);
        wp::adj_mul(var_134, var_33, adj_134, adj_33, adj_138);
        // adj: vz = wp.clamp(dot(nz * kf, vt), lower, upper)                                     <L 519>
        wp::adj_clamp(var_136, var_128, var_129, adj_136, adj_128, adj_129, adj_137);
        wp::adj_dot(var_135, var_121, adj_135, adj_121, adj_136);
        wp::adj_mul(var_132, var_33, adj_132, adj_33, adj_135);
        // adj: vx = wp.clamp(dot(nx * kf, vt), lower, upper)                                     <L 518>
        wp::adj_cross(var_104, var_133, adj_104, adj_133, adj_134);
        wp::adj_vec_t(var_130, var_107, var_107, adj_130, adj_107, adj_107, adj_133);
        // adj: nz = cross(n, vec3(1.0, 0.0, 0.0))                                                <L 516>
        wp::adj_cross(var_104, var_131, adj_104, adj_131, adj_132);
        wp::adj_vec_t(var_107, var_107, var_130, adj_107, adj_107, adj_130, adj_131);
        // adj: nx = cross(n, vec3(0.0, 0.0, 1.0))  # basis vectors for tangent                   <L 515>
        wp::adj_sub(var_107, var_128, adj_107, adj_128, adj_129);
        // adj: upper = 0.0 - lower  # workaround because no unary ops yet                        <L 513>
        wp::adj_mul(var_39, var_127, adj_39, adj_127, adj_128);
        wp::adj_add(var_109, var_126, adj_109, adj_126, adj_127);
        // adj: lower = mu * (fn + fd)                                                            <L 512>
        wp::adj_sub(var_107, var_125, adj_107, adj_125, adj_126);
        wp::adj_mul(var_123, var_124, adj_123, adj_124, adj_125);
        wp::adj_step(var_108, adj_108, adj_124);
        wp::adj_mul(var_122, var_27, adj_122, adj_27, adj_123);
        wp::adj_max(var_119, var_107, adj_119, adj_107, adj_122);
        // adj: fd = 0.0 - wp.max(vn, 0.0) * kd * wp.step(c)  # again, negative, into the ground  <L 506>
        wp::adj_sub(var_118, var_120, adj_118, adj_120, adj_121);
        wp::adj_mul(var_104, var_119, adj_104, adj_119, adj_120);
        // adj: vt = vrel - n * vn  # velocity component not in normal direction                  <L 503>
        wp::adj_dot(var_104, var_118, adj_104, adj_118, adj_119);
        // adj: vn = dot(n, vrel)  # velocity component of body in negative normal direction      <L 502>
        wp::adj_sub(var_117, var_59, adj_117, adj_59, adj_118);
        // adj: vrel = vtri - dpdt                                                                <L 500>
        wp::adj_add(var_114, var_116, adj_114, adj_116, adj_117);
        wp::adj_mul(var_92, var_115, adj_92, adj_115, adj_116);
        wp::adj_extract(var_93, var_29, adj_93, adj_29, adj_115);
        wp::adj_add(var_111, var_113, adj_111, adj_113, adj_114);
        wp::adj_mul(var_89, var_112, adj_89, adj_112, adj_113);
        wp::adj_extract(var_93, var_23, adj_93, adj_23, adj_112);
        wp::adj_mul(var_86, var_110, adj_86, adj_110, adj_111);
        wp::adj_extract(var_93, var_17, adj_93, adj_17, adj_110);
        // adj: vtri = vp * bary[0] + vq * bary[1] + vr * bary[2]  # bad approximation for centroid velocity  <L 499>
        wp::adj_mul(var_108, var_21, adj_108, adj_21, adj_109);
        // adj: fn = c * ke  # normal force (restitution coefficient * how far inside for ground) (negative)  <L 497>
        wp::adj_min(var_106, var_107, adj_106, adj_107, adj_108);
        wp::adj_sub(var_103, var_105, adj_103, adj_105, adj_106);
        // adj: c = wp.min(dist - 0.05, 0.0)  # 0 unless within 0.05 of surface                   <L 491>
        wp::adj_normalize(var_102, var_104, adj_102, adj_104);
        // adj: n = wp.normalize(diff)  # points into the object                                  <L 490>
        wp::adj_dot(var_102, var_102, adj_102, adj_102, adj_103);
        // adj: dist = wp.dot(diff, diff)  # squared distance                                     <L 489>
        wp::adj_sub(var_57, var_101, adj_57, adj_101, adj_102);
        // adj: diff = pos - closest  # vector from tri to point                                  <L 488>
        wp::adj_add(var_98, var_100, adj_98, adj_100, adj_101);
        wp::adj_mul(var_83, var_99, adj_83, adj_99, adj_100);
        wp::adj_extract(var_93, var_29, adj_93, adj_29, adj_99);
        wp::adj_add(var_95, var_97, adj_95, adj_97, adj_98);
        wp::adj_mul(var_80, var_96, adj_80, adj_96, adj_97);
        wp::adj_extract(var_93, var_23, adj_93, adj_23, adj_96);
        wp::adj_mul(var_77, var_94, adj_77, adj_94, adj_95);
        wp::adj_extract(var_93, var_17, adj_93, adj_17, adj_94);
        // adj: closest = p * bary[0] + q * bary[1] + r * bary[2]                                 <L 486>
        adj_triangle_closest_point_barycentric(var_77, var_80, var_83, var_57, adj_77, adj_80, adj_83, adj_57, adj_93);
        // adj: bary = triangle_closest_point_barycentric(p, q, r, pos)                           <L 485>
        wp::adj_copy(var_91, adj_90, adj_92);
        wp::adj_load(var_90, adj_90, adj_91);
        wp::adj_address(var_v, var_74, adj_v, adj_74, adj_90);
        // adj: vr = v[k]  # vel two                                                              <L 483>
        wp::adj_copy(var_88, adj_87, adj_89);
        wp::adj_load(var_87, adj_87, adj_88);
        wp::adj_address(var_v, var_69, adj_v, adj_69, adj_87);
        // adj: vq = v[j]  # vel one                                                              <L 482>
        wp::adj_copy(var_85, adj_84, adj_86);
        wp::adj_load(var_84, adj_84, adj_85);
        wp::adj_address(var_v, var_64, adj_v, adj_64, adj_84);
        // adj: vp = v[i]  # vel zero                                                             <L 481>
        wp::adj_copy(var_82, adj_81, adj_83);
        wp::adj_load(var_81, adj_81, adj_82);
        wp::adj_address(var_x, var_74, adj_x, adj_74, adj_81);
        // adj: r = x[k]  # point two                                                             <L 479>
        wp::adj_copy(var_79, adj_78, adj_80);
        wp::adj_load(var_78, adj_78, adj_79);
        wp::adj_address(var_x, var_69, adj_x, adj_69, adj_78);
        // adj: q = x[j]  # point one                                                             <L 478>
        wp::adj_copy(var_76, adj_75, adj_77);
        wp::adj_load(var_75, adj_75, adj_76);
        wp::adj_address(var_x, var_64, adj_x, adj_64, adj_75);
        // adj: p = x[i]  # point zero                                                            <L 477>
        wp::adj_copy(var_73, adj_72, adj_74);
        wp::adj_load(var_72, adj_72, adj_73);
        wp::adj_address(var_indices, var_71, adj_indices, adj_71, adj_72);
        wp::adj_add(var_70, var_29, adj_70, adj_29, adj_71);
        wp::adj_mul(var_1, var_35, adj_1, adj_35, adj_70);
        // adj: k = indices[face_no * 3 + 2]                                                      <L 475>
        wp::adj_copy(var_68, adj_67, adj_69);
        wp::adj_load(var_67, adj_67, adj_68);
        wp::adj_address(var_indices, var_66, adj_indices, adj_66, adj_67);
        wp::adj_add(var_65, var_23, adj_65, adj_23, adj_66);
        wp::adj_mul(var_1, var_35, adj_1, adj_35, adj_65);
        // adj: j = indices[face_no * 3 + 1]                                                      <L 474>
        wp::adj_copy(var_63, adj_62, adj_64);
        wp::adj_load(var_62, adj_62, adj_63);
        wp::adj_address(var_indices, var_61, adj_indices, adj_61, adj_62);
        wp::adj_add(var_60, var_17, adj_60, adj_17, adj_61);
        wp::adj_mul(var_1, var_35, adj_1, adj_35, adj_60);
        // adj: i = indices[face_no * 3 + 0]                                                      <L 473>
        wp::adj_add(var_48, var_58, adj_48, adj_58, adj_59);
        wp::adj_cross(var_51, var_54, adj_51, adj_54, adj_58);
        // adj: dpdt = v0 + wp.cross(w0, r)  # this is body velocity cross offset, so it's the velocity of the contact point.  <L 469>
        wp::adj_add(var_53, var_56, adj_53, adj_56, adj_57);
        wp::adj_mul(var_55, var_11, adj_55, adj_11, adj_56);
        // adj: pos = pos + rhat * c_dist  # add on 'thickness' of shape, e.g.: radius of sphere/capsule  <L 466>
        wp::adj_normalize(var_54, var_55, adj_54, adj_55);
        // adj: rhat = wp.normalize(r)                                                            <L 465>
        wp::adj_sub(var_53, var_42, adj_53, adj_42, adj_54);
        // adj: r = pos - x0  # basically just c_point in the new coordinates                     <L 464>
        wp::adj_add(var_42, var_52, adj_42, adj_52, adj_53);
        wp::adj_quat_rotate(var_45, var_8, adj_45, adj_8, adj_52);
        // adj: pos = x0 + wp.quat_rotate(r0, c_point)                                            <L 460>
        wp::adj_copy(var_50, adj_49, adj_51);
        wp::adj_load(var_49, adj_49, adj_50);
        wp::adj_address(var_body_w, var_5, adj_body_w, adj_5, adj_49);
        // adj: w0 = body_w[c_body]                                                               <L 457>
        wp::adj_copy(var_47, adj_46, adj_48);
        wp::adj_load(var_46, adj_46, adj_47);
        wp::adj_address(var_body_v, var_5, adj_body_v, adj_5, adj_46);
        // adj: v0 = body_v[c_body]                                                               <L 456>
        wp::adj_copy(var_44, adj_43, adj_45);
        wp::adj_load(var_43, adj_43, adj_44);
        wp::adj_address(var_body_r, var_5, adj_body_r, adj_5, adj_43);
        // adj: r0 = body_r[c_body]  # orientation of colliding body                              <L 454>
        wp::adj_copy(var_41, adj_40, adj_42);
        wp::adj_load(var_40, adj_40, adj_41);
        wp::adj_address(var_body_x, var_5, adj_body_x, adj_5, adj_40);
        // adj: x0 = body_x[c_body]  # position of colliding body                                 <L 453>
        wp::adj_copy(var_38, adj_37, adj_39);
        wp::adj_load(var_37, adj_37, adj_38);
        wp::adj_address(var_materials, var_36, adj_materials, adj_36, adj_37);
        wp::adj_add(var_34, var_35, adj_34, adj_35, adj_36);
        wp::adj_mul(var_14, var_15, adj_14, adj_15, adj_34);
        // adj: mu = materials[c_mat * 4 + 3]  # coulomb friction                                 <L 451>
        wp::adj_copy(var_32, adj_31, adj_33);
        wp::adj_load(var_31, adj_31, adj_32);
        wp::adj_address(var_materials, var_30, adj_materials, adj_30, adj_31);
        wp::adj_add(var_28, var_29, adj_28, adj_29, adj_30);
        wp::adj_mul(var_14, var_15, adj_14, adj_15, adj_28);
        // adj: kf = materials[c_mat * 4 + 2]  # friction coefficient                             <L 450>
        wp::adj_copy(var_26, adj_25, adj_27);
        wp::adj_load(var_25, adj_25, adj_26);
        wp::adj_address(var_materials, var_24, adj_materials, adj_24, adj_25);
        wp::adj_add(var_22, var_23, adj_22, adj_23, adj_24);
        wp::adj_mul(var_14, var_15, adj_14, adj_15, adj_22);
        // adj: kd = materials[c_mat * 4 + 1]  # damping coefficient                              <L 449>
        wp::adj_copy(var_20, adj_19, adj_21);
        wp::adj_load(var_19, adj_19, adj_20);
        wp::adj_address(var_materials, var_18, adj_materials, adj_18, adj_19);
        wp::adj_add(var_16, var_17, adj_16, adj_17, adj_18);
        wp::adj_mul(var_14, var_15, adj_14, adj_15, adj_16);
        // adj: ke = materials[c_mat * 4 + 0]  # restitution coefficient                          <L 448>
        wp::adj_copy(var_13, adj_12, adj_14);
        wp::adj_load(var_12, adj_12, adj_13);
        wp::adj_address(var_contact_mat, var_2, adj_contact_mat, adj_2, adj_12);
        // adj: c_mat = contact_mat[particle_no]                                                  <L 445>
        wp::adj_copy(var_10, adj_9, adj_11);
        wp::adj_load(var_9, adj_9, adj_10);
        wp::adj_address(var_contact_dist, var_2, adj_contact_dist, adj_2, adj_9);
        // adj: c_dist = contact_dist[particle_no]                                                <L 444>
        wp::adj_copy(var_7, adj_6, adj_8);
        wp::adj_load(var_6, adj_6, adj_7);
        wp::adj_address(var_contact_point, var_2, adj_contact_point, adj_2, adj_6);
        // adj: c_point = contact_point[particle_no]                                              <L 443>
        wp::adj_copy(var_4, adj_3, adj_5);
        wp::adj_load(var_3, adj_3, adj_4);
        wp::adj_address(var_contact_body, var_2, adj_contact_body, adj_2, adj_3);
        // adj: c_body = contact_body[particle_no]                                                <L 442>
        wp::adj_mod(var_0, var_num_particles, adj_0, adj_num_particles, adj_2);
        // adj: particle_no = tid % num_particles  # which particle                               <L 438>
        wp::adj_floordiv(var_0, var_num_particles, adj_0, adj_num_particles, adj_1);
        // adj: face_no = tid // num_particles  # which face                                      <L 437>
        // adj: tid = wp.tid()                                                                    <L 435>
        // adj: def eval_triangles_body_contacts(                                                 <L 417>
        continue;
    }
}



extern "C" __global__ void eval_bending_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float32>> var_x,
    wp::array_t<wp::vec_t<3,wp::float32>> var_v,
    wp::array_t<wp::int32> var_indices,
    wp::array_t<wp::float32> var_rest,
    wp::array_t<wp::float32> var_bending_properties,
    wp::array_t<wp::vec_t<3,wp::float32>> var_f)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        const wp::int32 var_1 = 0;
        wp::float32* var_2;
        wp::float32 var_3;
        wp::float32 var_4;
        const wp::int32 var_5 = 1;
        wp::float32* var_6;
        wp::float32 var_7;
        wp::float32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        const wp::int32 var_15 = 2;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        const wp::int32 var_19 = 3;
        wp::int32* var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        wp::float32* var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        wp::vec_t<3,wp::float32>* var_26;
        wp::vec_t<3,wp::float32> var_27;
        wp::vec_t<3,wp::float32> var_28;
        wp::vec_t<3,wp::float32>* var_29;
        wp::vec_t<3,wp::float32> var_30;
        wp::vec_t<3,wp::float32> var_31;
        wp::vec_t<3,wp::float32>* var_32;
        wp::vec_t<3,wp::float32> var_33;
        wp::vec_t<3,wp::float32> var_34;
        wp::vec_t<3,wp::float32>* var_35;
        wp::vec_t<3,wp::float32> var_36;
        wp::vec_t<3,wp::float32> var_37;
        wp::vec_t<3,wp::float32>* var_38;
        wp::vec_t<3,wp::float32> var_39;
        wp::vec_t<3,wp::float32> var_40;
        wp::vec_t<3,wp::float32>* var_41;
        wp::vec_t<3,wp::float32> var_42;
        wp::vec_t<3,wp::float32> var_43;
        wp::vec_t<3,wp::float32>* var_44;
        wp::vec_t<3,wp::float32> var_45;
        wp::vec_t<3,wp::float32> var_46;
        wp::vec_t<3,wp::float32>* var_47;
        wp::vec_t<3,wp::float32> var_48;
        wp::vec_t<3,wp::float32> var_49;
        wp::vec_t<3,wp::float32> var_50;
        wp::vec_t<3,wp::float32> var_51;
        wp::vec_t<3,wp::float32> var_52;
        wp::vec_t<3,wp::float32> var_53;
        wp::vec_t<3,wp::float32> var_54;
        wp::vec_t<3,wp::float32> var_55;
        wp::float32 var_56;
        wp::float32 var_57;
        const wp::float32 var_58 = 1e-06;
        bool var_59;
        bool var_60;
        bool var_61;
        const wp::float32 var_62 = 1.0;
        wp::float32 var_63;
        wp::float32 var_64;
        wp::float32 var_65;
        wp::float32 var_66;
        wp::float32 var_67;
        wp::vec_t<3,wp::float32> var_68;
        wp::vec_t<3,wp::float32> var_69;
        wp::vec_t<3,wp::float32> var_70;
        wp::vec_t<3,wp::float32> var_71;
        wp::vec_t<3,wp::float32> var_72;
        wp::vec_t<3,wp::float32> var_73;
        wp::float32 var_74;
        wp::vec_t<3,wp::float32> var_75;
        wp::float32 var_76;
        wp::float32 var_77;
        wp::float32 var_78;
        wp::float32 var_79;
        wp::vec_t<3,wp::float32> var_80;
        wp::vec_t<3,wp::float32> var_81;
        wp::vec_t<3,wp::float32> var_82;
        wp::float32 var_83;
        wp::vec_t<3,wp::float32> var_84;
        wp::vec_t<3,wp::float32> var_85;
        wp::float32 var_86;
        wp::vec_t<3,wp::float32> var_87;
        wp::vec_t<3,wp::float32> var_88;
        wp::vec_t<3,wp::float32> var_89;
        wp::float32 var_90;
        wp::vec_t<3,wp::float32> var_91;
        wp::vec_t<3,wp::float32> var_92;
        wp::float32 var_93;
        wp::vec_t<3,wp::float32> var_94;
        wp::vec_t<3,wp::float32> var_95;
        wp::float32 var_96;
        wp::float32 var_97;
        wp::float32 var_98;
        wp::float32 var_99;
        wp::float32 var_100;
        wp::float32 var_101;
        wp::float32 var_102;
        wp::float32 var_103;
        wp::float32 var_104;
        wp::float32 var_105;
        const wp::float32 var_106 = 0.0;
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
        wp::vec_t<3,wp::float32> var_117;
        //---------
        // forward
        // def eval_bending(                                                                      <L 534>
        // tid = wp.tid()                                                                         <L 542>
        var_0 = builtin_tid1d();
        // ke = bending_properties[tid, 0]                                                        <L 543>
        var_2 = wp::address(var_bending_properties, var_0, var_1);
        var_3 = wp::load(var_2);
        var_4 = wp::copy(var_3);
        // kd = bending_properties[tid, 1]                                                        <L 544>
        var_6 = wp::address(var_bending_properties, var_0, var_5);
        var_7 = wp::load(var_6);
        var_8 = wp::copy(var_7);
        // i = indices[tid, 0]                                                                    <L 546>
        var_9 = wp::address(var_indices, var_0, var_1);
        var_10 = wp::load(var_9);
        var_11 = wp::copy(var_10);
        // j = indices[tid, 1]                                                                    <L 547>
        var_12 = wp::address(var_indices, var_0, var_5);
        var_13 = wp::load(var_12);
        var_14 = wp::copy(var_13);
        // k = indices[tid, 2]                                                                    <L 548>
        var_16 = wp::address(var_indices, var_0, var_15);
        var_17 = wp::load(var_16);
        var_18 = wp::copy(var_17);
        // l = indices[tid, 3]                                                                    <L 549>
        var_20 = wp::address(var_indices, var_0, var_19);
        var_21 = wp::load(var_20);
        var_22 = wp::copy(var_21);
        // rest_angle = rest[tid]                                                                 <L 551>
        var_23 = wp::address(var_rest, var_0);
        var_24 = wp::load(var_23);
        var_25 = wp::copy(var_24);
        // x1 = x[i]                                                                              <L 553>
        var_26 = wp::address(var_x, var_11);
        var_27 = wp::load(var_26);
        var_28 = wp::copy(var_27);
        // x2 = x[j]                                                                              <L 554>
        var_29 = wp::address(var_x, var_14);
        var_30 = wp::load(var_29);
        var_31 = wp::copy(var_30);
        // x3 = x[k]                                                                              <L 555>
        var_32 = wp::address(var_x, var_18);
        var_33 = wp::load(var_32);
        var_34 = wp::copy(var_33);
        // x4 = x[l]                                                                              <L 556>
        var_35 = wp::address(var_x, var_22);
        var_36 = wp::load(var_35);
        var_37 = wp::copy(var_36);
        // v1 = v[i]                                                                              <L 558>
        var_38 = wp::address(var_v, var_11);
        var_39 = wp::load(var_38);
        var_40 = wp::copy(var_39);
        // v2 = v[j]                                                                              <L 559>
        var_41 = wp::address(var_v, var_14);
        var_42 = wp::load(var_41);
        var_43 = wp::copy(var_42);
        // v3 = v[k]                                                                              <L 560>
        var_44 = wp::address(var_v, var_18);
        var_45 = wp::load(var_44);
        var_46 = wp::copy(var_45);
        // v4 = v[l]                                                                              <L 561>
        var_47 = wp::address(var_v, var_22);
        var_48 = wp::load(var_47);
        var_49 = wp::copy(var_48);
        // n1 = wp.cross(x3 - x1, x4 - x1)  # normal to face 1                                    <L 563>
        var_50 = wp::sub(var_34, var_28);
        var_51 = wp::sub(var_37, var_28);
        var_52 = wp::cross(var_50, var_51);
        // n2 = wp.cross(x4 - x2, x3 - x2)  # normal to face 2                                    <L 564>
        var_53 = wp::sub(var_37, var_31);
        var_54 = wp::sub(var_34, var_31);
        var_55 = wp::cross(var_53, var_54);
        // n1_length = wp.length(n1)                                                              <L 566>
        var_56 = wp::length(var_52);
        // n2_length = wp.length(n2)                                                              <L 567>
        var_57 = wp::length(var_55);
        // if n1_length < 1.0e-6 or n2_length < 1.0e-6:                                           <L 569>
        var_59 = (var_56 < var_58);
        var_60 = (var_57 < var_58);
        var_61 = var_59 || var_60;
        if (var_61) {
            // return                                                                             <L 570>
            return;
        }
        // rcp_n1 = 1.0 / n1_length                                                               <L 572>
        var_63 = wp::div(var_62, var_56);
        // rcp_n2 = 1.0 / n2_length                                                               <L 573>
        var_64 = wp::div(var_62, var_57);
        // cos_theta = wp.dot(n1, n2) * rcp_n1 * rcp_n2                                           <L 575>
        var_65 = wp::dot(var_52, var_55);
        var_66 = wp::mul(var_65, var_63);
        var_67 = wp::mul(var_66, var_64);
        // n1 = n1 * rcp_n1 * rcp_n1                                                              <L 577>
        var_68 = wp::mul(var_52, var_63);
        var_69 = wp::mul(var_68, var_63);
        // n2 = n2 * rcp_n2 * rcp_n2                                                              <L 578>
        var_70 = wp::mul(var_55, var_64);
        var_71 = wp::mul(var_70, var_64);
        // e = x4 - x3                                                                            <L 580>
        var_72 = wp::sub(var_37, var_34);
        // e_hat = wp.normalize(e)                                                                <L 581>
        var_73 = wp::normalize(var_72);
        // e_length = wp.length(e)                                                                <L 582>
        var_74 = wp::length(var_72);
        // s = wp.sign(wp.dot(wp.cross(n2, n1), e_hat))                                           <L 584>
        var_75 = wp::cross(var_71, var_69);
        var_76 = wp::dot(var_75, var_73);
        var_77 = wp::sign(var_76);
        // angle = wp.acos(cos_theta) * s                                                         <L 585>
        var_78 = wp::acos(var_67);
        var_79 = wp::mul(var_78, var_77);
        // d1 = n1 * e_length                                                                     <L 587>
        var_80 = wp::mul(var_69, var_74);
        // d2 = n2 * e_length                                                                     <L 588>
        var_81 = wp::mul(var_71, var_74);
        // d3 = n1 * wp.dot(x1 - x4, e_hat) + n2 * wp.dot(x2 - x4, e_hat)                         <L 589>
        var_82 = wp::sub(var_28, var_37);
        var_83 = wp::dot(var_82, var_73);
        var_84 = wp::mul(var_69, var_83);
        var_85 = wp::sub(var_31, var_37);
        var_86 = wp::dot(var_85, var_73);
        var_87 = wp::mul(var_71, var_86);
        var_88 = wp::add(var_84, var_87);
        // d4 = n1 * wp.dot(x3 - x1, e_hat) + n2 * wp.dot(x3 - x2, e_hat)                         <L 590>
        var_89 = wp::sub(var_34, var_28);
        var_90 = wp::dot(var_89, var_73);
        var_91 = wp::mul(var_69, var_90);
        var_92 = wp::sub(var_34, var_31);
        var_93 = wp::dot(var_92, var_73);
        var_94 = wp::mul(var_71, var_93);
        var_95 = wp::add(var_91, var_94);
        // f_elastic = ke * (angle - rest_angle)                                                  <L 593>
        var_96 = wp::sub(var_79, var_25);
        var_97 = wp::mul(var_4, var_96);
        // f_damp = kd * (wp.dot(d1, v1) + wp.dot(d2, v2) + wp.dot(d3, v3) + wp.dot(d4, v4))       <L 596>
        var_98 = wp::dot(var_80, var_40);
        var_99 = wp::dot(var_81, var_43);
        var_100 = wp::add(var_98, var_99);
        var_101 = wp::dot(var_88, var_46);
        var_102 = wp::add(var_100, var_101);
        var_103 = wp::dot(var_95, var_49);
        var_104 = wp::add(var_102, var_103);
        var_105 = wp::mul(var_8, var_104);
        // f_total = 0.0 - e_length * (f_elastic + f_damp)                                        <L 599>
        var_107 = wp::add(var_97, var_105);
        var_108 = wp::mul(var_74, var_107);
        var_109 = wp::sub(var_106, var_108);
        // wp.atomic_add(f, i, d1 * f_total)                                                      <L 601>
        var_110 = wp::mul(var_80, var_109);
        var_111 = wp::atomic_add(var_f, var_11, var_110);
        // wp.atomic_add(f, j, d2 * f_total)                                                      <L 602>
        var_112 = wp::mul(var_81, var_109);
        var_113 = wp::atomic_add(var_f, var_14, var_112);
        // wp.atomic_add(f, k, d3 * f_total)                                                      <L 603>
        var_114 = wp::mul(var_88, var_109);
        var_115 = wp::atomic_add(var_f, var_18, var_114);
        // wp.atomic_add(f, l, d4 * f_total)                                                      <L 604>
        var_116 = wp::mul(var_95, var_109);
        var_117 = wp::atomic_add(var_f, var_22, var_116);
    }
}

extern "C" __global__ void eval_bending_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float32>> var_x,
    wp::array_t<wp::vec_t<3,wp::float32>> var_v,
    wp::array_t<wp::int32> var_indices,
    wp::array_t<wp::float32> var_rest,
    wp::array_t<wp::float32> var_bending_properties,
    wp::array_t<wp::vec_t<3,wp::float32>> var_f,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_x,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_v,
    wp::array_t<wp::int32> adj_indices,
    wp::array_t<wp::float32> adj_rest,
    wp::array_t<wp::float32> adj_bending_properties,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_f)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        const wp::int32 var_1 = 0;
        wp::float32* var_2;
        wp::float32 var_3;
        wp::float32 var_4;
        const wp::int32 var_5 = 1;
        wp::float32* var_6;
        wp::float32 var_7;
        wp::float32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        const wp::int32 var_15 = 2;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        const wp::int32 var_19 = 3;
        wp::int32* var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        wp::float32* var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        wp::vec_t<3,wp::float32>* var_26;
        wp::vec_t<3,wp::float32> var_27;
        wp::vec_t<3,wp::float32> var_28;
        wp::vec_t<3,wp::float32>* var_29;
        wp::vec_t<3,wp::float32> var_30;
        wp::vec_t<3,wp::float32> var_31;
        wp::vec_t<3,wp::float32>* var_32;
        wp::vec_t<3,wp::float32> var_33;
        wp::vec_t<3,wp::float32> var_34;
        wp::vec_t<3,wp::float32>* var_35;
        wp::vec_t<3,wp::float32> var_36;
        wp::vec_t<3,wp::float32> var_37;
        wp::vec_t<3,wp::float32>* var_38;
        wp::vec_t<3,wp::float32> var_39;
        wp::vec_t<3,wp::float32> var_40;
        wp::vec_t<3,wp::float32>* var_41;
        wp::vec_t<3,wp::float32> var_42;
        wp::vec_t<3,wp::float32> var_43;
        wp::vec_t<3,wp::float32>* var_44;
        wp::vec_t<3,wp::float32> var_45;
        wp::vec_t<3,wp::float32> var_46;
        wp::vec_t<3,wp::float32>* var_47;
        wp::vec_t<3,wp::float32> var_48;
        wp::vec_t<3,wp::float32> var_49;
        wp::vec_t<3,wp::float32> var_50;
        wp::vec_t<3,wp::float32> var_51;
        wp::vec_t<3,wp::float32> var_52;
        wp::vec_t<3,wp::float32> var_53;
        wp::vec_t<3,wp::float32> var_54;
        wp::vec_t<3,wp::float32> var_55;
        wp::float32 var_56;
        wp::float32 var_57;
        const wp::float32 var_58 = 1e-06;
        bool var_59;
        bool var_60;
        bool var_61;
        const wp::float32 var_62 = 1.0;
        wp::float32 var_63;
        wp::float32 var_64;
        wp::float32 var_65;
        wp::float32 var_66;
        wp::float32 var_67;
        wp::vec_t<3,wp::float32> var_68;
        wp::vec_t<3,wp::float32> var_69;
        wp::vec_t<3,wp::float32> var_70;
        wp::vec_t<3,wp::float32> var_71;
        wp::vec_t<3,wp::float32> var_72;
        wp::vec_t<3,wp::float32> var_73;
        wp::float32 var_74;
        wp::vec_t<3,wp::float32> var_75;
        wp::float32 var_76;
        wp::float32 var_77;
        wp::float32 var_78;
        wp::float32 var_79;
        wp::vec_t<3,wp::float32> var_80;
        wp::vec_t<3,wp::float32> var_81;
        wp::vec_t<3,wp::float32> var_82;
        wp::float32 var_83;
        wp::vec_t<3,wp::float32> var_84;
        wp::vec_t<3,wp::float32> var_85;
        wp::float32 var_86;
        wp::vec_t<3,wp::float32> var_87;
        wp::vec_t<3,wp::float32> var_88;
        wp::vec_t<3,wp::float32> var_89;
        wp::float32 var_90;
        wp::vec_t<3,wp::float32> var_91;
        wp::vec_t<3,wp::float32> var_92;
        wp::float32 var_93;
        wp::vec_t<3,wp::float32> var_94;
        wp::vec_t<3,wp::float32> var_95;
        wp::float32 var_96;
        wp::float32 var_97;
        wp::float32 var_98;
        wp::float32 var_99;
        wp::float32 var_100;
        wp::float32 var_101;
        wp::float32 var_102;
        wp::float32 var_103;
        wp::float32 var_104;
        wp::float32 var_105;
        const wp::float32 var_106 = 0.0;
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
        wp::vec_t<3,wp::float32> var_117;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::float32 adj_2 = {};
        wp::float32 adj_3 = {};
        wp::float32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::float32 adj_6 = {};
        wp::float32 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        wp::int32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::int32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::float32 adj_24 = {};
        wp::float32 adj_25 = {};
        wp::vec_t<3,wp::float32> adj_26 = {};
        wp::vec_t<3,wp::float32> adj_27 = {};
        wp::vec_t<3,wp::float32> adj_28 = {};
        wp::vec_t<3,wp::float32> adj_29 = {};
        wp::vec_t<3,wp::float32> adj_30 = {};
        wp::vec_t<3,wp::float32> adj_31 = {};
        wp::vec_t<3,wp::float32> adj_32 = {};
        wp::vec_t<3,wp::float32> adj_33 = {};
        wp::vec_t<3,wp::float32> adj_34 = {};
        wp::vec_t<3,wp::float32> adj_35 = {};
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
        wp::vec_t<3,wp::float32> adj_46 = {};
        wp::vec_t<3,wp::float32> adj_47 = {};
        wp::vec_t<3,wp::float32> adj_48 = {};
        wp::vec_t<3,wp::float32> adj_49 = {};
        wp::vec_t<3,wp::float32> adj_50 = {};
        wp::vec_t<3,wp::float32> adj_51 = {};
        wp::vec_t<3,wp::float32> adj_52 = {};
        wp::vec_t<3,wp::float32> adj_53 = {};
        wp::vec_t<3,wp::float32> adj_54 = {};
        wp::vec_t<3,wp::float32> adj_55 = {};
        wp::float32 adj_56 = {};
        wp::float32 adj_57 = {};
        wp::float32 adj_58 = {};
        bool adj_59 = {};
        bool adj_60 = {};
        bool adj_61 = {};
        wp::float32 adj_62 = {};
        wp::float32 adj_63 = {};
        wp::float32 adj_64 = {};
        wp::float32 adj_65 = {};
        wp::float32 adj_66 = {};
        wp::float32 adj_67 = {};
        wp::vec_t<3,wp::float32> adj_68 = {};
        wp::vec_t<3,wp::float32> adj_69 = {};
        wp::vec_t<3,wp::float32> adj_70 = {};
        wp::vec_t<3,wp::float32> adj_71 = {};
        wp::vec_t<3,wp::float32> adj_72 = {};
        wp::vec_t<3,wp::float32> adj_73 = {};
        wp::float32 adj_74 = {};
        wp::vec_t<3,wp::float32> adj_75 = {};
        wp::float32 adj_76 = {};
        wp::float32 adj_77 = {};
        wp::float32 adj_78 = {};
        wp::float32 adj_79 = {};
        wp::vec_t<3,wp::float32> adj_80 = {};
        wp::vec_t<3,wp::float32> adj_81 = {};
        wp::vec_t<3,wp::float32> adj_82 = {};
        wp::float32 adj_83 = {};
        wp::vec_t<3,wp::float32> adj_84 = {};
        wp::vec_t<3,wp::float32> adj_85 = {};
        wp::float32 adj_86 = {};
        wp::vec_t<3,wp::float32> adj_87 = {};
        wp::vec_t<3,wp::float32> adj_88 = {};
        wp::vec_t<3,wp::float32> adj_89 = {};
        wp::float32 adj_90 = {};
        wp::vec_t<3,wp::float32> adj_91 = {};
        wp::vec_t<3,wp::float32> adj_92 = {};
        wp::float32 adj_93 = {};
        wp::vec_t<3,wp::float32> adj_94 = {};
        wp::vec_t<3,wp::float32> adj_95 = {};
        wp::float32 adj_96 = {};
        wp::float32 adj_97 = {};
        wp::float32 adj_98 = {};
        wp::float32 adj_99 = {};
        wp::float32 adj_100 = {};
        wp::float32 adj_101 = {};
        wp::float32 adj_102 = {};
        wp::float32 adj_103 = {};
        wp::float32 adj_104 = {};
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
        wp::vec_t<3,wp::float32> adj_117 = {};
        //---------
        // forward
        // def eval_bending(                                                                      <L 534>
        // tid = wp.tid()                                                                         <L 542>
        var_0 = builtin_tid1d();
        // ke = bending_properties[tid, 0]                                                        <L 543>
        var_2 = wp::address(var_bending_properties, var_0, var_1);
        var_3 = wp::load(var_2);
        var_4 = wp::copy(var_3);
        // kd = bending_properties[tid, 1]                                                        <L 544>
        var_6 = wp::address(var_bending_properties, var_0, var_5);
        var_7 = wp::load(var_6);
        var_8 = wp::copy(var_7);
        // i = indices[tid, 0]                                                                    <L 546>
        var_9 = wp::address(var_indices, var_0, var_1);
        var_10 = wp::load(var_9);
        var_11 = wp::copy(var_10);
        // j = indices[tid, 1]                                                                    <L 547>
        var_12 = wp::address(var_indices, var_0, var_5);
        var_13 = wp::load(var_12);
        var_14 = wp::copy(var_13);
        // k = indices[tid, 2]                                                                    <L 548>
        var_16 = wp::address(var_indices, var_0, var_15);
        var_17 = wp::load(var_16);
        var_18 = wp::copy(var_17);
        // l = indices[tid, 3]                                                                    <L 549>
        var_20 = wp::address(var_indices, var_0, var_19);
        var_21 = wp::load(var_20);
        var_22 = wp::copy(var_21);
        // rest_angle = rest[tid]                                                                 <L 551>
        var_23 = wp::address(var_rest, var_0);
        var_24 = wp::load(var_23);
        var_25 = wp::copy(var_24);
        // x1 = x[i]                                                                              <L 553>
        var_26 = wp::address(var_x, var_11);
        var_27 = wp::load(var_26);
        var_28 = wp::copy(var_27);
        // x2 = x[j]                                                                              <L 554>
        var_29 = wp::address(var_x, var_14);
        var_30 = wp::load(var_29);
        var_31 = wp::copy(var_30);
        // x3 = x[k]                                                                              <L 555>
        var_32 = wp::address(var_x, var_18);
        var_33 = wp::load(var_32);
        var_34 = wp::copy(var_33);
        // x4 = x[l]                                                                              <L 556>
        var_35 = wp::address(var_x, var_22);
        var_36 = wp::load(var_35);
        var_37 = wp::copy(var_36);
        // v1 = v[i]                                                                              <L 558>
        var_38 = wp::address(var_v, var_11);
        var_39 = wp::load(var_38);
        var_40 = wp::copy(var_39);
        // v2 = v[j]                                                                              <L 559>
        var_41 = wp::address(var_v, var_14);
        var_42 = wp::load(var_41);
        var_43 = wp::copy(var_42);
        // v3 = v[k]                                                                              <L 560>
        var_44 = wp::address(var_v, var_18);
        var_45 = wp::load(var_44);
        var_46 = wp::copy(var_45);
        // v4 = v[l]                                                                              <L 561>
        var_47 = wp::address(var_v, var_22);
        var_48 = wp::load(var_47);
        var_49 = wp::copy(var_48);
        // n1 = wp.cross(x3 - x1, x4 - x1)  # normal to face 1                                    <L 563>
        var_50 = wp::sub(var_34, var_28);
        var_51 = wp::sub(var_37, var_28);
        var_52 = wp::cross(var_50, var_51);
        // n2 = wp.cross(x4 - x2, x3 - x2)  # normal to face 2                                    <L 564>
        var_53 = wp::sub(var_37, var_31);
        var_54 = wp::sub(var_34, var_31);
        var_55 = wp::cross(var_53, var_54);
        // n1_length = wp.length(n1)                                                              <L 566>
        var_56 = wp::length(var_52);
        // n2_length = wp.length(n2)                                                              <L 567>
        var_57 = wp::length(var_55);
        // if n1_length < 1.0e-6 or n2_length < 1.0e-6:                                           <L 569>
        var_59 = (var_56 < var_58);
        var_60 = (var_57 < var_58);
        var_61 = var_59 || var_60;
        if (var_61) {
            // return                                                                             <L 570>
            goto label0;
        }
        // rcp_n1 = 1.0 / n1_length                                                               <L 572>
        var_63 = wp::div(var_62, var_56);
        // rcp_n2 = 1.0 / n2_length                                                               <L 573>
        var_64 = wp::div(var_62, var_57);
        // cos_theta = wp.dot(n1, n2) * rcp_n1 * rcp_n2                                           <L 575>
        var_65 = wp::dot(var_52, var_55);
        var_66 = wp::mul(var_65, var_63);
        var_67 = wp::mul(var_66, var_64);
        // n1 = n1 * rcp_n1 * rcp_n1                                                              <L 577>
        var_68 = wp::mul(var_52, var_63);
        var_69 = wp::mul(var_68, var_63);
        // n2 = n2 * rcp_n2 * rcp_n2                                                              <L 578>
        var_70 = wp::mul(var_55, var_64);
        var_71 = wp::mul(var_70, var_64);
        // e = x4 - x3                                                                            <L 580>
        var_72 = wp::sub(var_37, var_34);
        // e_hat = wp.normalize(e)                                                                <L 581>
        var_73 = wp::normalize(var_72);
        // e_length = wp.length(e)                                                                <L 582>
        var_74 = wp::length(var_72);
        // s = wp.sign(wp.dot(wp.cross(n2, n1), e_hat))                                           <L 584>
        var_75 = wp::cross(var_71, var_69);
        var_76 = wp::dot(var_75, var_73);
        var_77 = wp::sign(var_76);
        // angle = wp.acos(cos_theta) * s                                                         <L 585>
        var_78 = wp::acos(var_67);
        var_79 = wp::mul(var_78, var_77);
        // d1 = n1 * e_length                                                                     <L 587>
        var_80 = wp::mul(var_69, var_74);
        // d2 = n2 * e_length                                                                     <L 588>
        var_81 = wp::mul(var_71, var_74);
        // d3 = n1 * wp.dot(x1 - x4, e_hat) + n2 * wp.dot(x2 - x4, e_hat)                         <L 589>
        var_82 = wp::sub(var_28, var_37);
        var_83 = wp::dot(var_82, var_73);
        var_84 = wp::mul(var_69, var_83);
        var_85 = wp::sub(var_31, var_37);
        var_86 = wp::dot(var_85, var_73);
        var_87 = wp::mul(var_71, var_86);
        var_88 = wp::add(var_84, var_87);
        // d4 = n1 * wp.dot(x3 - x1, e_hat) + n2 * wp.dot(x3 - x2, e_hat)                         <L 590>
        var_89 = wp::sub(var_34, var_28);
        var_90 = wp::dot(var_89, var_73);
        var_91 = wp::mul(var_69, var_90);
        var_92 = wp::sub(var_34, var_31);
        var_93 = wp::dot(var_92, var_73);
        var_94 = wp::mul(var_71, var_93);
        var_95 = wp::add(var_91, var_94);
        // f_elastic = ke * (angle - rest_angle)                                                  <L 593>
        var_96 = wp::sub(var_79, var_25);
        var_97 = wp::mul(var_4, var_96);
        // f_damp = kd * (wp.dot(d1, v1) + wp.dot(d2, v2) + wp.dot(d3, v3) + wp.dot(d4, v4))       <L 596>
        var_98 = wp::dot(var_80, var_40);
        var_99 = wp::dot(var_81, var_43);
        var_100 = wp::add(var_98, var_99);
        var_101 = wp::dot(var_88, var_46);
        var_102 = wp::add(var_100, var_101);
        var_103 = wp::dot(var_95, var_49);
        var_104 = wp::add(var_102, var_103);
        var_105 = wp::mul(var_8, var_104);
        // f_total = 0.0 - e_length * (f_elastic + f_damp)                                        <L 599>
        var_107 = wp::add(var_97, var_105);
        var_108 = wp::mul(var_74, var_107);
        var_109 = wp::sub(var_106, var_108);
        // wp.atomic_add(f, i, d1 * f_total)                                                      <L 601>
        var_110 = wp::mul(var_80, var_109);
        // var_111 = wp::atomic_add(var_f, var_11, var_110);
        // wp.atomic_add(f, j, d2 * f_total)                                                      <L 602>
        var_112 = wp::mul(var_81, var_109);
        // var_113 = wp::atomic_add(var_f, var_14, var_112);
        // wp.atomic_add(f, k, d3 * f_total)                                                      <L 603>
        var_114 = wp::mul(var_88, var_109);
        // var_115 = wp::atomic_add(var_f, var_18, var_114);
        // wp.atomic_add(f, l, d4 * f_total)                                                      <L 604>
        var_116 = wp::mul(var_95, var_109);
        // var_117 = wp::atomic_add(var_f, var_22, var_116);
        //---------
        // reverse
        wp::adj_atomic_add(var_f, var_22, var_116, adj_f, adj_22, adj_116, adj_117);
        wp::adj_mul(var_95, var_109, adj_95, adj_109, adj_116);
        // adj: wp.atomic_add(f, l, d4 * f_total)                                                 <L 604>
        wp::adj_atomic_add(var_f, var_18, var_114, adj_f, adj_18, adj_114, adj_115);
        wp::adj_mul(var_88, var_109, adj_88, adj_109, adj_114);
        // adj: wp.atomic_add(f, k, d3 * f_total)                                                 <L 603>
        wp::adj_atomic_add(var_f, var_14, var_112, adj_f, adj_14, adj_112, adj_113);
        wp::adj_mul(var_81, var_109, adj_81, adj_109, adj_112);
        // adj: wp.atomic_add(f, j, d2 * f_total)                                                 <L 602>
        wp::adj_atomic_add(var_f, var_11, var_110, adj_f, adj_11, adj_110, adj_111);
        wp::adj_mul(var_80, var_109, adj_80, adj_109, adj_110);
        // adj: wp.atomic_add(f, i, d1 * f_total)                                                 <L 601>
        wp::adj_sub(var_106, var_108, adj_106, adj_108, adj_109);
        wp::adj_mul(var_74, var_107, adj_74, adj_107, adj_108);
        wp::adj_add(var_97, var_105, adj_97, adj_105, adj_107);
        // adj: f_total = 0.0 - e_length * (f_elastic + f_damp)                                   <L 599>
        wp::adj_mul(var_8, var_104, adj_8, adj_104, adj_105);
        wp::adj_add(var_102, var_103, adj_102, adj_103, adj_104);
        wp::adj_dot(var_95, var_49, adj_95, adj_49, adj_103);
        wp::adj_add(var_100, var_101, adj_100, adj_101, adj_102);
        wp::adj_dot(var_88, var_46, adj_88, adj_46, adj_101);
        wp::adj_add(var_98, var_99, adj_98, adj_99, adj_100);
        wp::adj_dot(var_81, var_43, adj_81, adj_43, adj_99);
        wp::adj_dot(var_80, var_40, adj_80, adj_40, adj_98);
        // adj: f_damp = kd * (wp.dot(d1, v1) + wp.dot(d2, v2) + wp.dot(d3, v3) + wp.dot(d4, v4))  <L 596>
        wp::adj_mul(var_4, var_96, adj_4, adj_96, adj_97);
        wp::adj_sub(var_79, var_25, adj_79, adj_25, adj_96);
        // adj: f_elastic = ke * (angle - rest_angle)                                             <L 593>
        wp::adj_add(var_91, var_94, adj_91, adj_94, adj_95);
        wp::adj_mul(var_71, var_93, adj_71, adj_93, adj_94);
        wp::adj_dot(var_92, var_73, adj_92, adj_73, adj_93);
        wp::adj_sub(var_34, var_31, adj_34, adj_31, adj_92);
        wp::adj_mul(var_69, var_90, adj_69, adj_90, adj_91);
        wp::adj_dot(var_89, var_73, adj_89, adj_73, adj_90);
        wp::adj_sub(var_34, var_28, adj_34, adj_28, adj_89);
        // adj: d4 = n1 * wp.dot(x3 - x1, e_hat) + n2 * wp.dot(x3 - x2, e_hat)                    <L 590>
        wp::adj_add(var_84, var_87, adj_84, adj_87, adj_88);
        wp::adj_mul(var_71, var_86, adj_71, adj_86, adj_87);
        wp::adj_dot(var_85, var_73, adj_85, adj_73, adj_86);
        wp::adj_sub(var_31, var_37, adj_31, adj_37, adj_85);
        wp::adj_mul(var_69, var_83, adj_69, adj_83, adj_84);
        wp::adj_dot(var_82, var_73, adj_82, adj_73, adj_83);
        wp::adj_sub(var_28, var_37, adj_28, adj_37, adj_82);
        // adj: d3 = n1 * wp.dot(x1 - x4, e_hat) + n2 * wp.dot(x2 - x4, e_hat)                    <L 589>
        wp::adj_mul(var_71, var_74, adj_71, adj_74, adj_81);
        // adj: d2 = n2 * e_length                                                                <L 588>
        wp::adj_mul(var_69, var_74, adj_69, adj_74, adj_80);
        // adj: d1 = n1 * e_length                                                                <L 587>
        wp::adj_mul(var_78, var_77, adj_78, adj_77, adj_79);
        wp::adj_acos(var_67, adj_67, adj_78);
        // adj: angle = wp.acos(cos_theta) * s                                                    <L 585>
        wp::adj_sign(var_76, adj_76, adj_77);
        wp::adj_dot(var_75, var_73, adj_75, adj_73, adj_76);
        wp::adj_cross(var_71, var_69, adj_71, adj_69, adj_75);
        // adj: s = wp.sign(wp.dot(wp.cross(n2, n1), e_hat))                                      <L 584>
        wp::adj_length(var_72, var_74, adj_72, adj_74);
        // adj: e_length = wp.length(e)                                                           <L 582>
        wp::adj_normalize(var_72, var_73, adj_72, adj_73);
        // adj: e_hat = wp.normalize(e)                                                           <L 581>
        wp::adj_sub(var_37, var_34, adj_37, adj_34, adj_72);
        // adj: e = x4 - x3                                                                       <L 580>
        wp::adj_mul(var_70, var_64, adj_70, adj_64, adj_71);
        wp::adj_mul(var_55, var_64, adj_55, adj_64, adj_70);
        // adj: n2 = n2 * rcp_n2 * rcp_n2                                                         <L 578>
        wp::adj_mul(var_68, var_63, adj_68, adj_63, adj_69);
        wp::adj_mul(var_52, var_63, adj_52, adj_63, adj_68);
        // adj: n1 = n1 * rcp_n1 * rcp_n1                                                         <L 577>
        wp::adj_mul(var_66, var_64, adj_66, adj_64, adj_67);
        wp::adj_mul(var_65, var_63, adj_65, adj_63, adj_66);
        wp::adj_dot(var_52, var_55, adj_52, adj_55, adj_65);
        // adj: cos_theta = wp.dot(n1, n2) * rcp_n1 * rcp_n2                                      <L 575>
        wp::adj_div(var_62, var_57, var_64, adj_62, adj_57, adj_64);
        // adj: rcp_n2 = 1.0 / n2_length                                                          <L 573>
        wp::adj_div(var_62, var_56, var_63, adj_62, adj_56, adj_63);
        // adj: rcp_n1 = 1.0 / n1_length                                                          <L 572>
        if (var_61) {
            label0:;
            // adj: return                                                                        <L 570>
        }
        // adj: if n1_length < 1.0e-6 or n2_length < 1.0e-6:                                      <L 569>
        wp::adj_length(var_55, var_57, adj_55, adj_57);
        // adj: n2_length = wp.length(n2)                                                         <L 567>
        wp::adj_length(var_52, var_56, adj_52, adj_56);
        // adj: n1_length = wp.length(n1)                                                         <L 566>
        wp::adj_cross(var_53, var_54, adj_53, adj_54, adj_55);
        wp::adj_sub(var_34, var_31, adj_34, adj_31, adj_54);
        wp::adj_sub(var_37, var_31, adj_37, adj_31, adj_53);
        // adj: n2 = wp.cross(x4 - x2, x3 - x2)  # normal to face 2                               <L 564>
        wp::adj_cross(var_50, var_51, adj_50, adj_51, adj_52);
        wp::adj_sub(var_37, var_28, adj_37, adj_28, adj_51);
        wp::adj_sub(var_34, var_28, adj_34, adj_28, adj_50);
        // adj: n1 = wp.cross(x3 - x1, x4 - x1)  # normal to face 1                               <L 563>
        wp::adj_copy(var_48, adj_47, adj_49);
        wp::adj_load(var_47, adj_47, adj_48);
        wp::adj_address(var_v, var_22, adj_v, adj_22, adj_47);
        // adj: v4 = v[l]                                                                         <L 561>
        wp::adj_copy(var_45, adj_44, adj_46);
        wp::adj_load(var_44, adj_44, adj_45);
        wp::adj_address(var_v, var_18, adj_v, adj_18, adj_44);
        // adj: v3 = v[k]                                                                         <L 560>
        wp::adj_copy(var_42, adj_41, adj_43);
        wp::adj_load(var_41, adj_41, adj_42);
        wp::adj_address(var_v, var_14, adj_v, adj_14, adj_41);
        // adj: v2 = v[j]                                                                         <L 559>
        wp::adj_copy(var_39, adj_38, adj_40);
        wp::adj_load(var_38, adj_38, adj_39);
        wp::adj_address(var_v, var_11, adj_v, adj_11, adj_38);
        // adj: v1 = v[i]                                                                         <L 558>
        wp::adj_copy(var_36, adj_35, adj_37);
        wp::adj_load(var_35, adj_35, adj_36);
        wp::adj_address(var_x, var_22, adj_x, adj_22, adj_35);
        // adj: x4 = x[l]                                                                         <L 556>
        wp::adj_copy(var_33, adj_32, adj_34);
        wp::adj_load(var_32, adj_32, adj_33);
        wp::adj_address(var_x, var_18, adj_x, adj_18, adj_32);
        // adj: x3 = x[k]                                                                         <L 555>
        wp::adj_copy(var_30, adj_29, adj_31);
        wp::adj_load(var_29, adj_29, adj_30);
        wp::adj_address(var_x, var_14, adj_x, adj_14, adj_29);
        // adj: x2 = x[j]                                                                         <L 554>
        wp::adj_copy(var_27, adj_26, adj_28);
        wp::adj_load(var_26, adj_26, adj_27);
        wp::adj_address(var_x, var_11, adj_x, adj_11, adj_26);
        // adj: x1 = x[i]                                                                         <L 553>
        wp::adj_copy(var_24, adj_23, adj_25);
        wp::adj_load(var_23, adj_23, adj_24);
        wp::adj_address(var_rest, var_0, adj_rest, adj_0, adj_23);
        // adj: rest_angle = rest[tid]                                                            <L 551>
        wp::adj_copy(var_21, adj_20, adj_22);
        wp::adj_load(var_20, adj_20, adj_21);
        wp::adj_address(var_indices, var_0, var_19, adj_indices, adj_0, adj_19, adj_20);
        // adj: l = indices[tid, 3]                                                               <L 549>
        wp::adj_copy(var_17, adj_16, adj_18);
        wp::adj_load(var_16, adj_16, adj_17);
        wp::adj_address(var_indices, var_0, var_15, adj_indices, adj_0, adj_15, adj_16);
        // adj: k = indices[tid, 2]                                                               <L 548>
        wp::adj_copy(var_13, adj_12, adj_14);
        wp::adj_load(var_12, adj_12, adj_13);
        wp::adj_address(var_indices, var_0, var_5, adj_indices, adj_0, adj_5, adj_12);
        // adj: j = indices[tid, 1]                                                               <L 547>
        wp::adj_copy(var_10, adj_9, adj_11);
        wp::adj_load(var_9, adj_9, adj_10);
        wp::adj_address(var_indices, var_0, var_1, adj_indices, adj_0, adj_1, adj_9);
        // adj: i = indices[tid, 0]                                                               <L 546>
        wp::adj_copy(var_7, adj_6, adj_8);
        wp::adj_load(var_6, adj_6, adj_7);
        wp::adj_address(var_bending_properties, var_0, var_5, adj_bending_properties, adj_0, adj_5, adj_6);
        // adj: kd = bending_properties[tid, 1]                                                   <L 544>
        wp::adj_copy(var_3, adj_2, adj_4);
        wp::adj_load(var_2, adj_2, adj_3);
        wp::adj_address(var_bending_properties, var_0, var_1, adj_bending_properties, adj_0, adj_1, adj_2);
        // adj: ke = bending_properties[tid, 0]                                                   <L 543>
        // adj: tid = wp.tid()                                                                    <L 542>
        // adj: def eval_bending(                                                                 <L 534>
        continue;
    }
}



extern "C" __global__ void eval_tetrahedra_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float32>> var_x,
    wp::array_t<wp::vec_t<3,wp::float32>> var_v,
    wp::array_t<wp::int32> var_indices,
    wp::array_t<wp::mat_t<3,3,wp::float32>> var_pose,
    wp::array_t<wp::float32> var_activation,
    wp::array_t<wp::float32> var_materials,
    wp::array_t<wp::vec_t<3,wp::float32>> var_f)
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
        const wp::int32 var_13 = 3;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::float32* var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32* var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::float32* var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        wp::float32* var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::vec_t<3,wp::float32>* var_29;
        wp::vec_t<3,wp::float32> var_30;
        wp::vec_t<3,wp::float32> var_31;
        wp::vec_t<3,wp::float32>* var_32;
        wp::vec_t<3,wp::float32> var_33;
        wp::vec_t<3,wp::float32> var_34;
        wp::vec_t<3,wp::float32>* var_35;
        wp::vec_t<3,wp::float32> var_36;
        wp::vec_t<3,wp::float32> var_37;
        wp::vec_t<3,wp::float32>* var_38;
        wp::vec_t<3,wp::float32> var_39;
        wp::vec_t<3,wp::float32> var_40;
        wp::vec_t<3,wp::float32>* var_41;
        wp::vec_t<3,wp::float32> var_42;
        wp::vec_t<3,wp::float32> var_43;
        wp::vec_t<3,wp::float32>* var_44;
        wp::vec_t<3,wp::float32> var_45;
        wp::vec_t<3,wp::float32> var_46;
        wp::vec_t<3,wp::float32>* var_47;
        wp::vec_t<3,wp::float32> var_48;
        wp::vec_t<3,wp::float32> var_49;
        wp::vec_t<3,wp::float32>* var_50;
        wp::vec_t<3,wp::float32> var_51;
        wp::vec_t<3,wp::float32> var_52;
        wp::vec_t<3,wp::float32> var_53;
        wp::vec_t<3,wp::float32> var_54;
        wp::vec_t<3,wp::float32> var_55;
        wp::vec_t<3,wp::float32> var_56;
        wp::vec_t<3,wp::float32> var_57;
        wp::vec_t<3,wp::float32> var_58;
        wp::mat_t<3,3,wp::float32> var_59;
        wp::mat_t<3,3,wp::float32>* var_60;
        wp::mat_t<3,3,wp::float32> var_61;
        wp::mat_t<3,3,wp::float32> var_62;
        wp::float32 var_63;
        const wp::float32 var_64 = 6.0;
        wp::float32 var_65;
        const wp::float32 var_66 = 1.0;
        wp::float32 var_67;
        wp::float32 var_68;
        wp::float32 var_69;
        const wp::float32 var_70 = 4.0;
        wp::float32 var_71;
        wp::float32 var_72;
        wp::float32 var_73;
        wp::float32 var_74;
        wp::float32 var_75;
        wp::float32 var_76;
        wp::mat_t<3,3,wp::float32> var_77;
        wp::mat_t<3,3,wp::float32> var_78;
        wp::mat_t<3,3,wp::float32> var_79;
        wp::float32 var_80;
        wp::float32 var_81;
        wp::float32 var_82;
        wp::vec_t<3,wp::float32> var_83;
        wp::float32 var_84;
        wp::float32 var_85;
        wp::float32 var_86;
        wp::vec_t<3,wp::float32> var_87;
        wp::float32 var_88;
        wp::float32 var_89;
        wp::float32 var_90;
        wp::vec_t<3,wp::float32> var_91;
        wp::float32 var_92;
        wp::float32 var_93;
        wp::float32 var_94;
        wp::float32 var_95;
        wp::float32 var_96;
        wp::mat_t<3,3,wp::float32> var_97;
        wp::float32 var_98;
        wp::float32 var_99;
        wp::float32 var_100;
        wp::mat_t<3,3,wp::float32> var_101;
        wp::mat_t<3,3,wp::float32> var_102;
        wp::mat_t<3,3,wp::float32> var_103;
        wp::mat_t<3,3,wp::float32> var_104;
        wp::mat_t<3,3,wp::float32> var_105;
        wp::float32 var_106;
        wp::float32 var_107;
        wp::float32 var_108;
        wp::vec_t<3,wp::float32> var_109;
        wp::float32 var_110;
        wp::float32 var_111;
        wp::float32 var_112;
        wp::vec_t<3,wp::float32> var_113;
        wp::float32 var_114;
        wp::float32 var_115;
        wp::float32 var_116;
        wp::vec_t<3,wp::float32> var_117;
        wp::float32 var_118;
        wp::float32 var_119;
        wp::vec_t<3,wp::float32> var_120;
        wp::vec_t<3,wp::float32> var_121;
        wp::vec_t<3,wp::float32> var_122;
        wp::vec_t<3,wp::float32> var_123;
        wp::vec_t<3,wp::float32> var_124;
        wp::vec_t<3,wp::float32> var_125;
        wp::float32 var_126;
        wp::float32 var_127;
        wp::float32 var_128;
        wp::float32 var_129;
        wp::float32 var_130;
        wp::float32 var_131;
        wp::float32 var_132;
        wp::float32 var_133;
        wp::float32 var_134;
        wp::float32 var_135;
        wp::vec_t<3,wp::float32> var_136;
        wp::vec_t<3,wp::float32> var_137;
        wp::vec_t<3,wp::float32> var_138;
        wp::vec_t<3,wp::float32> var_139;
        wp::vec_t<3,wp::float32> var_140;
        wp::vec_t<3,wp::float32> var_141;
        wp::vec_t<3,wp::float32> var_142;
        wp::vec_t<3,wp::float32> var_143;
        const wp::float32 var_144 = 0.0;
        wp::float32 var_145;
        wp::vec_t<3,wp::float32> var_146;
        wp::vec_t<3,wp::float32> var_147;
        wp::vec_t<3,wp::float32> var_148;
        wp::vec_t<3,wp::float32> var_149;
        wp::vec_t<3,wp::float32> var_150;
        //---------
        // forward
        // def eval_tetrahedra(                                                                   <L 608>
        // tid = wp.tid()                                                                         <L 617>
        var_0 = builtin_tid1d();
        // i = indices[tid, 0]                                                                    <L 619>
        var_2 = wp::address(var_indices, var_0, var_1);
        var_3 = wp::load(var_2);
        var_4 = wp::copy(var_3);
        // j = indices[tid, 1]                                                                    <L 620>
        var_6 = wp::address(var_indices, var_0, var_5);
        var_7 = wp::load(var_6);
        var_8 = wp::copy(var_7);
        // k = indices[tid, 2]                                                                    <L 621>
        var_10 = wp::address(var_indices, var_0, var_9);
        var_11 = wp::load(var_10);
        var_12 = wp::copy(var_11);
        // l = indices[tid, 3]                                                                    <L 622>
        var_14 = wp::address(var_indices, var_0, var_13);
        var_15 = wp::load(var_14);
        var_16 = wp::copy(var_15);
        // act = activation[tid]                                                                  <L 624>
        var_17 = wp::address(var_activation, var_0);
        var_18 = wp::load(var_17);
        var_19 = wp::copy(var_18);
        // k_mu = materials[tid, 0]                                                               <L 626>
        var_20 = wp::address(var_materials, var_0, var_1);
        var_21 = wp::load(var_20);
        var_22 = wp::copy(var_21);
        // k_lambda = materials[tid, 1]                                                           <L 627>
        var_23 = wp::address(var_materials, var_0, var_5);
        var_24 = wp::load(var_23);
        var_25 = wp::copy(var_24);
        // k_damp = materials[tid, 2]                                                             <L 628>
        var_26 = wp::address(var_materials, var_0, var_9);
        var_27 = wp::load(var_26);
        var_28 = wp::copy(var_27);
        // x0 = x[i]                                                                              <L 630>
        var_29 = wp::address(var_x, var_4);
        var_30 = wp::load(var_29);
        var_31 = wp::copy(var_30);
        // x1 = x[j]                                                                              <L 631>
        var_32 = wp::address(var_x, var_8);
        var_33 = wp::load(var_32);
        var_34 = wp::copy(var_33);
        // x2 = x[k]                                                                              <L 632>
        var_35 = wp::address(var_x, var_12);
        var_36 = wp::load(var_35);
        var_37 = wp::copy(var_36);
        // x3 = x[l]                                                                              <L 633>
        var_38 = wp::address(var_x, var_16);
        var_39 = wp::load(var_38);
        var_40 = wp::copy(var_39);
        // v0 = v[i]                                                                              <L 635>
        var_41 = wp::address(var_v, var_4);
        var_42 = wp::load(var_41);
        var_43 = wp::copy(var_42);
        // v1 = v[j]                                                                              <L 636>
        var_44 = wp::address(var_v, var_8);
        var_45 = wp::load(var_44);
        var_46 = wp::copy(var_45);
        // v2 = v[k]                                                                              <L 637>
        var_47 = wp::address(var_v, var_12);
        var_48 = wp::load(var_47);
        var_49 = wp::copy(var_48);
        // v3 = v[l]                                                                              <L 638>
        var_50 = wp::address(var_v, var_16);
        var_51 = wp::load(var_50);
        var_52 = wp::copy(var_51);
        // x10 = x1 - x0                                                                          <L 640>
        var_53 = wp::sub(var_34, var_31);
        // x20 = x2 - x0                                                                          <L 641>
        var_54 = wp::sub(var_37, var_31);
        // x30 = x3 - x0                                                                          <L 642>
        var_55 = wp::sub(var_40, var_31);
        // v10 = v1 - v0                                                                          <L 644>
        var_56 = wp::sub(var_46, var_43);
        // v20 = v2 - v0                                                                          <L 645>
        var_57 = wp::sub(var_49, var_43);
        // v30 = v3 - v0                                                                          <L 646>
        var_58 = wp::sub(var_52, var_43);
        // Ds = wp.mat33(x10, x20, x30)                                                           <L 648>
        var_59 = wp::mat_t<3,3,wp::float32>(var_53, var_54, var_55);
        // Dm = pose[tid]                                                                         <L 649>
        var_60 = wp::address(var_pose, var_0);
        var_61 = wp::load(var_60);
        var_62 = wp::copy(var_61);
        // inv_rest_volume = wp.determinant(Dm) * 6.0                                             <L 651>
        var_63 = wp::determinant(var_62);
        var_65 = wp::mul(var_63, var_64);
        // rest_volume = 1.0 / inv_rest_volume                                                    <L 652>
        var_67 = wp::div(var_66, var_65);
        // alpha = 1.0 + k_mu / k_lambda - k_mu / (4.0 * k_lambda)                                <L 654>
        var_68 = wp::div(var_22, var_25);
        var_69 = wp::add(var_66, var_68);
        var_71 = wp::mul(var_70, var_25);
        var_72 = wp::div(var_22, var_71);
        var_73 = wp::sub(var_69, var_72);
        // k_mu = k_mu * rest_volume                                                              <L 657>
        var_74 = wp::mul(var_22, var_67);
        // k_lambda = k_lambda * rest_volume                                                      <L 658>
        var_75 = wp::mul(var_25, var_67);
        // k_damp = k_damp * rest_volume                                                          <L 659>
        var_76 = wp::mul(var_28, var_67);
        // F = Ds * Dm                                                                            <L 662>
        var_77 = wp::mul(var_59, var_62);
        // dFdt = wp.mat33(v10, v20, v30) * Dm                                                    <L 663>
        var_78 = wp::mat_t<3,3,wp::float32>(var_56, var_57, var_58);
        var_79 = wp::mul(var_78, var_62);
        // col1 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])                                              <L 665>
        var_80 = wp::extract(var_77, var_1, var_1);
        var_81 = wp::extract(var_77, var_5, var_1);
        var_82 = wp::extract(var_77, var_9, var_1);
        var_83 = wp::vec_t<3,wp::float32>(var_80, var_81, var_82);
        // col2 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])                                              <L 666>
        var_84 = wp::extract(var_77, var_1, var_5);
        var_85 = wp::extract(var_77, var_5, var_5);
        var_86 = wp::extract(var_77, var_9, var_5);
        var_87 = wp::vec_t<3,wp::float32>(var_84, var_85, var_86);
        // col3 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])                                              <L 667>
        var_88 = wp::extract(var_77, var_1, var_9);
        var_89 = wp::extract(var_77, var_5, var_9);
        var_90 = wp::extract(var_77, var_9, var_9);
        var_91 = wp::vec_t<3,wp::float32>(var_88, var_89, var_90);
        // Ic = dot(col1, col1) + dot(col2, col2) + dot(col3, col3)                               <L 672>
        var_92 = wp::dot(var_83, var_83);
        var_93 = wp::dot(var_87, var_87);
        var_94 = wp::add(var_92, var_93);
        var_95 = wp::dot(var_91, var_91);
        var_96 = wp::add(var_94, var_95);
        // P = F * k_mu * (1.0 - 1.0 / (Ic + 1.0)) + dFdt * k_damp                                <L 675>
        var_97 = wp::mul(var_77, var_74);
        var_98 = wp::add(var_96, var_66);
        var_99 = wp::div(var_66, var_98);
        var_100 = wp::sub(var_66, var_99);
        var_101 = wp::mul(var_97, var_100);
        var_102 = wp::mul(var_79, var_76);
        var_103 = wp::add(var_101, var_102);
        // H = P * wp.transpose(Dm)                                                               <L 676>
        var_104 = wp::transpose(var_62);
        var_105 = wp::mul(var_103, var_104);
        // f1 = wp.vec3(H[0, 0], H[1, 0], H[2, 0])                                                <L 678>
        var_106 = wp::extract(var_105, var_1, var_1);
        var_107 = wp::extract(var_105, var_5, var_1);
        var_108 = wp::extract(var_105, var_9, var_1);
        var_109 = wp::vec_t<3,wp::float32>(var_106, var_107, var_108);
        // f2 = wp.vec3(H[0, 1], H[1, 1], H[2, 1])                                                <L 679>
        var_110 = wp::extract(var_105, var_1, var_5);
        var_111 = wp::extract(var_105, var_5, var_5);
        var_112 = wp::extract(var_105, var_9, var_5);
        var_113 = wp::vec_t<3,wp::float32>(var_110, var_111, var_112);
        // f3 = wp.vec3(H[0, 2], H[1, 2], H[2, 2])                                                <L 680>
        var_114 = wp::extract(var_105, var_1, var_9);
        var_115 = wp::extract(var_105, var_5, var_9);
        var_116 = wp::extract(var_105, var_9, var_9);
        var_117 = wp::vec_t<3,wp::float32>(var_114, var_115, var_116);
        // J = wp.determinant(F)                                                                  <L 761>
        var_118 = wp::determinant(var_77);
        // s = inv_rest_volume / 6.0                                                              <L 764>
        var_119 = wp::div(var_65, var_64);
        // dJdx1 = wp.cross(x20, x30) * s                                                         <L 765>
        var_120 = wp::cross(var_54, var_55);
        var_121 = wp::mul(var_120, var_119);
        // dJdx2 = wp.cross(x30, x10) * s                                                         <L 766>
        var_122 = wp::cross(var_55, var_53);
        var_123 = wp::mul(var_122, var_119);
        // dJdx3 = wp.cross(x10, x20) * s                                                         <L 767>
        var_124 = wp::cross(var_53, var_54);
        var_125 = wp::mul(var_124, var_119);
        // f_volume = (J - alpha + act) * k_lambda                                                <L 769>
        var_126 = wp::sub(var_118, var_73);
        var_127 = wp::add(var_126, var_19);
        var_128 = wp::mul(var_127, var_75);
        // f_damp = (wp.dot(dJdx1, v1) + wp.dot(dJdx2, v2) + wp.dot(dJdx3, v3)) * k_damp          <L 770>
        var_129 = wp::dot(var_121, var_46);
        var_130 = wp::dot(var_123, var_49);
        var_131 = wp::add(var_129, var_130);
        var_132 = wp::dot(var_125, var_52);
        var_133 = wp::add(var_131, var_132);
        var_134 = wp::mul(var_133, var_76);
        // f_total = f_volume + f_damp                                                            <L 772>
        var_135 = wp::add(var_128, var_134);
        // f1 = f1 + dJdx1 * f_total                                                              <L 774>
        var_136 = wp::mul(var_121, var_135);
        var_137 = wp::add(var_109, var_136);
        // f2 = f2 + dJdx2 * f_total                                                              <L 775>
        var_138 = wp::mul(var_123, var_135);
        var_139 = wp::add(var_113, var_138);
        // f3 = f3 + dJdx3 * f_total                                                              <L 776>
        var_140 = wp::mul(var_125, var_135);
        var_141 = wp::add(var_117, var_140);
        // f0 = (f1 + f2 + f3) * (0.0 - 1.0)                                                      <L 777>
        var_142 = wp::add(var_137, var_139);
        var_143 = wp::add(var_142, var_141);
        var_145 = wp::sub(var_144, var_66);
        var_146 = wp::mul(var_143, var_145);
        // wp.atomic_sub(f, i, f0)                                                                <L 780>
        var_147 = wp::atomic_sub(var_f, var_4, var_146);
        // wp.atomic_sub(f, j, f1)                                                                <L 781>
        var_148 = wp::atomic_sub(var_f, var_8, var_137);
        // wp.atomic_sub(f, k, f2)                                                                <L 782>
        var_149 = wp::atomic_sub(var_f, var_12, var_139);
        // wp.atomic_sub(f, l, f3)                                                                <L 783>
        var_150 = wp::atomic_sub(var_f, var_16, var_141);
    }
}

extern "C" __global__ void eval_tetrahedra_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float32>> var_x,
    wp::array_t<wp::vec_t<3,wp::float32>> var_v,
    wp::array_t<wp::int32> var_indices,
    wp::array_t<wp::mat_t<3,3,wp::float32>> var_pose,
    wp::array_t<wp::float32> var_activation,
    wp::array_t<wp::float32> var_materials,
    wp::array_t<wp::vec_t<3,wp::float32>> var_f,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_x,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_v,
    wp::array_t<wp::int32> adj_indices,
    wp::array_t<wp::mat_t<3,3,wp::float32>> adj_pose,
    wp::array_t<wp::float32> adj_activation,
    wp::array_t<wp::float32> adj_materials,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_f)
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
        const wp::int32 var_13 = 3;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::float32* var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32* var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::float32* var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        wp::float32* var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::vec_t<3,wp::float32>* var_29;
        wp::vec_t<3,wp::float32> var_30;
        wp::vec_t<3,wp::float32> var_31;
        wp::vec_t<3,wp::float32>* var_32;
        wp::vec_t<3,wp::float32> var_33;
        wp::vec_t<3,wp::float32> var_34;
        wp::vec_t<3,wp::float32>* var_35;
        wp::vec_t<3,wp::float32> var_36;
        wp::vec_t<3,wp::float32> var_37;
        wp::vec_t<3,wp::float32>* var_38;
        wp::vec_t<3,wp::float32> var_39;
        wp::vec_t<3,wp::float32> var_40;
        wp::vec_t<3,wp::float32>* var_41;
        wp::vec_t<3,wp::float32> var_42;
        wp::vec_t<3,wp::float32> var_43;
        wp::vec_t<3,wp::float32>* var_44;
        wp::vec_t<3,wp::float32> var_45;
        wp::vec_t<3,wp::float32> var_46;
        wp::vec_t<3,wp::float32>* var_47;
        wp::vec_t<3,wp::float32> var_48;
        wp::vec_t<3,wp::float32> var_49;
        wp::vec_t<3,wp::float32>* var_50;
        wp::vec_t<3,wp::float32> var_51;
        wp::vec_t<3,wp::float32> var_52;
        wp::vec_t<3,wp::float32> var_53;
        wp::vec_t<3,wp::float32> var_54;
        wp::vec_t<3,wp::float32> var_55;
        wp::vec_t<3,wp::float32> var_56;
        wp::vec_t<3,wp::float32> var_57;
        wp::vec_t<3,wp::float32> var_58;
        wp::mat_t<3,3,wp::float32> var_59;
        wp::mat_t<3,3,wp::float32>* var_60;
        wp::mat_t<3,3,wp::float32> var_61;
        wp::mat_t<3,3,wp::float32> var_62;
        wp::float32 var_63;
        const wp::float32 var_64 = 6.0;
        wp::float32 var_65;
        const wp::float32 var_66 = 1.0;
        wp::float32 var_67;
        wp::float32 var_68;
        wp::float32 var_69;
        const wp::float32 var_70 = 4.0;
        wp::float32 var_71;
        wp::float32 var_72;
        wp::float32 var_73;
        wp::float32 var_74;
        wp::float32 var_75;
        wp::float32 var_76;
        wp::mat_t<3,3,wp::float32> var_77;
        wp::mat_t<3,3,wp::float32> var_78;
        wp::mat_t<3,3,wp::float32> var_79;
        wp::float32 var_80;
        wp::float32 var_81;
        wp::float32 var_82;
        wp::vec_t<3,wp::float32> var_83;
        wp::float32 var_84;
        wp::float32 var_85;
        wp::float32 var_86;
        wp::vec_t<3,wp::float32> var_87;
        wp::float32 var_88;
        wp::float32 var_89;
        wp::float32 var_90;
        wp::vec_t<3,wp::float32> var_91;
        wp::float32 var_92;
        wp::float32 var_93;
        wp::float32 var_94;
        wp::float32 var_95;
        wp::float32 var_96;
        wp::mat_t<3,3,wp::float32> var_97;
        wp::float32 var_98;
        wp::float32 var_99;
        wp::float32 var_100;
        wp::mat_t<3,3,wp::float32> var_101;
        wp::mat_t<3,3,wp::float32> var_102;
        wp::mat_t<3,3,wp::float32> var_103;
        wp::mat_t<3,3,wp::float32> var_104;
        wp::mat_t<3,3,wp::float32> var_105;
        wp::float32 var_106;
        wp::float32 var_107;
        wp::float32 var_108;
        wp::vec_t<3,wp::float32> var_109;
        wp::float32 var_110;
        wp::float32 var_111;
        wp::float32 var_112;
        wp::vec_t<3,wp::float32> var_113;
        wp::float32 var_114;
        wp::float32 var_115;
        wp::float32 var_116;
        wp::vec_t<3,wp::float32> var_117;
        wp::float32 var_118;
        wp::float32 var_119;
        wp::vec_t<3,wp::float32> var_120;
        wp::vec_t<3,wp::float32> var_121;
        wp::vec_t<3,wp::float32> var_122;
        wp::vec_t<3,wp::float32> var_123;
        wp::vec_t<3,wp::float32> var_124;
        wp::vec_t<3,wp::float32> var_125;
        wp::float32 var_126;
        wp::float32 var_127;
        wp::float32 var_128;
        wp::float32 var_129;
        wp::float32 var_130;
        wp::float32 var_131;
        wp::float32 var_132;
        wp::float32 var_133;
        wp::float32 var_134;
        wp::float32 var_135;
        wp::vec_t<3,wp::float32> var_136;
        wp::vec_t<3,wp::float32> var_137;
        wp::vec_t<3,wp::float32> var_138;
        wp::vec_t<3,wp::float32> var_139;
        wp::vec_t<3,wp::float32> var_140;
        wp::vec_t<3,wp::float32> var_141;
        wp::vec_t<3,wp::float32> var_142;
        wp::vec_t<3,wp::float32> var_143;
        const wp::float32 var_144 = 0.0;
        wp::float32 var_145;
        wp::vec_t<3,wp::float32> var_146;
        wp::vec_t<3,wp::float32> var_147;
        wp::vec_t<3,wp::float32> var_148;
        wp::vec_t<3,wp::float32> var_149;
        wp::vec_t<3,wp::float32> var_150;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::float32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::float32 adj_24 = {};
        wp::float32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::float32 adj_27 = {};
        wp::float32 adj_28 = {};
        wp::vec_t<3,wp::float32> adj_29 = {};
        wp::vec_t<3,wp::float32> adj_30 = {};
        wp::vec_t<3,wp::float32> adj_31 = {};
        wp::vec_t<3,wp::float32> adj_32 = {};
        wp::vec_t<3,wp::float32> adj_33 = {};
        wp::vec_t<3,wp::float32> adj_34 = {};
        wp::vec_t<3,wp::float32> adj_35 = {};
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
        wp::vec_t<3,wp::float32> adj_46 = {};
        wp::vec_t<3,wp::float32> adj_47 = {};
        wp::vec_t<3,wp::float32> adj_48 = {};
        wp::vec_t<3,wp::float32> adj_49 = {};
        wp::vec_t<3,wp::float32> adj_50 = {};
        wp::vec_t<3,wp::float32> adj_51 = {};
        wp::vec_t<3,wp::float32> adj_52 = {};
        wp::vec_t<3,wp::float32> adj_53 = {};
        wp::vec_t<3,wp::float32> adj_54 = {};
        wp::vec_t<3,wp::float32> adj_55 = {};
        wp::vec_t<3,wp::float32> adj_56 = {};
        wp::vec_t<3,wp::float32> adj_57 = {};
        wp::vec_t<3,wp::float32> adj_58 = {};
        wp::mat_t<3,3,wp::float32> adj_59 = {};
        wp::mat_t<3,3,wp::float32> adj_60 = {};
        wp::mat_t<3,3,wp::float32> adj_61 = {};
        wp::mat_t<3,3,wp::float32> adj_62 = {};
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
        wp::mat_t<3,3,wp::float32> adj_77 = {};
        wp::mat_t<3,3,wp::float32> adj_78 = {};
        wp::mat_t<3,3,wp::float32> adj_79 = {};
        wp::float32 adj_80 = {};
        wp::float32 adj_81 = {};
        wp::float32 adj_82 = {};
        wp::vec_t<3,wp::float32> adj_83 = {};
        wp::float32 adj_84 = {};
        wp::float32 adj_85 = {};
        wp::float32 adj_86 = {};
        wp::vec_t<3,wp::float32> adj_87 = {};
        wp::float32 adj_88 = {};
        wp::float32 adj_89 = {};
        wp::float32 adj_90 = {};
        wp::vec_t<3,wp::float32> adj_91 = {};
        wp::float32 adj_92 = {};
        wp::float32 adj_93 = {};
        wp::float32 adj_94 = {};
        wp::float32 adj_95 = {};
        wp::float32 adj_96 = {};
        wp::mat_t<3,3,wp::float32> adj_97 = {};
        wp::float32 adj_98 = {};
        wp::float32 adj_99 = {};
        wp::float32 adj_100 = {};
        wp::mat_t<3,3,wp::float32> adj_101 = {};
        wp::mat_t<3,3,wp::float32> adj_102 = {};
        wp::mat_t<3,3,wp::float32> adj_103 = {};
        wp::mat_t<3,3,wp::float32> adj_104 = {};
        wp::mat_t<3,3,wp::float32> adj_105 = {};
        wp::float32 adj_106 = {};
        wp::float32 adj_107 = {};
        wp::float32 adj_108 = {};
        wp::vec_t<3,wp::float32> adj_109 = {};
        wp::float32 adj_110 = {};
        wp::float32 adj_111 = {};
        wp::float32 adj_112 = {};
        wp::vec_t<3,wp::float32> adj_113 = {};
        wp::float32 adj_114 = {};
        wp::float32 adj_115 = {};
        wp::float32 adj_116 = {};
        wp::vec_t<3,wp::float32> adj_117 = {};
        wp::float32 adj_118 = {};
        wp::float32 adj_119 = {};
        wp::vec_t<3,wp::float32> adj_120 = {};
        wp::vec_t<3,wp::float32> adj_121 = {};
        wp::vec_t<3,wp::float32> adj_122 = {};
        wp::vec_t<3,wp::float32> adj_123 = {};
        wp::vec_t<3,wp::float32> adj_124 = {};
        wp::vec_t<3,wp::float32> adj_125 = {};
        wp::float32 adj_126 = {};
        wp::float32 adj_127 = {};
        wp::float32 adj_128 = {};
        wp::float32 adj_129 = {};
        wp::float32 adj_130 = {};
        wp::float32 adj_131 = {};
        wp::float32 adj_132 = {};
        wp::float32 adj_133 = {};
        wp::float32 adj_134 = {};
        wp::float32 adj_135 = {};
        wp::vec_t<3,wp::float32> adj_136 = {};
        wp::vec_t<3,wp::float32> adj_137 = {};
        wp::vec_t<3,wp::float32> adj_138 = {};
        wp::vec_t<3,wp::float32> adj_139 = {};
        wp::vec_t<3,wp::float32> adj_140 = {};
        wp::vec_t<3,wp::float32> adj_141 = {};
        wp::vec_t<3,wp::float32> adj_142 = {};
        wp::vec_t<3,wp::float32> adj_143 = {};
        wp::float32 adj_144 = {};
        wp::float32 adj_145 = {};
        wp::vec_t<3,wp::float32> adj_146 = {};
        wp::vec_t<3,wp::float32> adj_147 = {};
        wp::vec_t<3,wp::float32> adj_148 = {};
        wp::vec_t<3,wp::float32> adj_149 = {};
        wp::vec_t<3,wp::float32> adj_150 = {};
        //---------
        // forward
        // def eval_tetrahedra(                                                                   <L 608>
        // tid = wp.tid()                                                                         <L 617>
        var_0 = builtin_tid1d();
        // i = indices[tid, 0]                                                                    <L 619>
        var_2 = wp::address(var_indices, var_0, var_1);
        var_3 = wp::load(var_2);
        var_4 = wp::copy(var_3);
        // j = indices[tid, 1]                                                                    <L 620>
        var_6 = wp::address(var_indices, var_0, var_5);
        var_7 = wp::load(var_6);
        var_8 = wp::copy(var_7);
        // k = indices[tid, 2]                                                                    <L 621>
        var_10 = wp::address(var_indices, var_0, var_9);
        var_11 = wp::load(var_10);
        var_12 = wp::copy(var_11);
        // l = indices[tid, 3]                                                                    <L 622>
        var_14 = wp::address(var_indices, var_0, var_13);
        var_15 = wp::load(var_14);
        var_16 = wp::copy(var_15);
        // act = activation[tid]                                                                  <L 624>
        var_17 = wp::address(var_activation, var_0);
        var_18 = wp::load(var_17);
        var_19 = wp::copy(var_18);
        // k_mu = materials[tid, 0]                                                               <L 626>
        var_20 = wp::address(var_materials, var_0, var_1);
        var_21 = wp::load(var_20);
        var_22 = wp::copy(var_21);
        // k_lambda = materials[tid, 1]                                                           <L 627>
        var_23 = wp::address(var_materials, var_0, var_5);
        var_24 = wp::load(var_23);
        var_25 = wp::copy(var_24);
        // k_damp = materials[tid, 2]                                                             <L 628>
        var_26 = wp::address(var_materials, var_0, var_9);
        var_27 = wp::load(var_26);
        var_28 = wp::copy(var_27);
        // x0 = x[i]                                                                              <L 630>
        var_29 = wp::address(var_x, var_4);
        var_30 = wp::load(var_29);
        var_31 = wp::copy(var_30);
        // x1 = x[j]                                                                              <L 631>
        var_32 = wp::address(var_x, var_8);
        var_33 = wp::load(var_32);
        var_34 = wp::copy(var_33);
        // x2 = x[k]                                                                              <L 632>
        var_35 = wp::address(var_x, var_12);
        var_36 = wp::load(var_35);
        var_37 = wp::copy(var_36);
        // x3 = x[l]                                                                              <L 633>
        var_38 = wp::address(var_x, var_16);
        var_39 = wp::load(var_38);
        var_40 = wp::copy(var_39);
        // v0 = v[i]                                                                              <L 635>
        var_41 = wp::address(var_v, var_4);
        var_42 = wp::load(var_41);
        var_43 = wp::copy(var_42);
        // v1 = v[j]                                                                              <L 636>
        var_44 = wp::address(var_v, var_8);
        var_45 = wp::load(var_44);
        var_46 = wp::copy(var_45);
        // v2 = v[k]                                                                              <L 637>
        var_47 = wp::address(var_v, var_12);
        var_48 = wp::load(var_47);
        var_49 = wp::copy(var_48);
        // v3 = v[l]                                                                              <L 638>
        var_50 = wp::address(var_v, var_16);
        var_51 = wp::load(var_50);
        var_52 = wp::copy(var_51);
        // x10 = x1 - x0                                                                          <L 640>
        var_53 = wp::sub(var_34, var_31);
        // x20 = x2 - x0                                                                          <L 641>
        var_54 = wp::sub(var_37, var_31);
        // x30 = x3 - x0                                                                          <L 642>
        var_55 = wp::sub(var_40, var_31);
        // v10 = v1 - v0                                                                          <L 644>
        var_56 = wp::sub(var_46, var_43);
        // v20 = v2 - v0                                                                          <L 645>
        var_57 = wp::sub(var_49, var_43);
        // v30 = v3 - v0                                                                          <L 646>
        var_58 = wp::sub(var_52, var_43);
        // Ds = wp.mat33(x10, x20, x30)                                                           <L 648>
        var_59 = wp::mat_t<3,3,wp::float32>(var_53, var_54, var_55);
        // Dm = pose[tid]                                                                         <L 649>
        var_60 = wp::address(var_pose, var_0);
        var_61 = wp::load(var_60);
        var_62 = wp::copy(var_61);
        // inv_rest_volume = wp.determinant(Dm) * 6.0                                             <L 651>
        var_63 = wp::determinant(var_62);
        var_65 = wp::mul(var_63, var_64);
        // rest_volume = 1.0 / inv_rest_volume                                                    <L 652>
        var_67 = wp::div(var_66, var_65);
        // alpha = 1.0 + k_mu / k_lambda - k_mu / (4.0 * k_lambda)                                <L 654>
        var_68 = wp::div(var_22, var_25);
        var_69 = wp::add(var_66, var_68);
        var_71 = wp::mul(var_70, var_25);
        var_72 = wp::div(var_22, var_71);
        var_73 = wp::sub(var_69, var_72);
        // k_mu = k_mu * rest_volume                                                              <L 657>
        var_74 = wp::mul(var_22, var_67);
        // k_lambda = k_lambda * rest_volume                                                      <L 658>
        var_75 = wp::mul(var_25, var_67);
        // k_damp = k_damp * rest_volume                                                          <L 659>
        var_76 = wp::mul(var_28, var_67);
        // F = Ds * Dm                                                                            <L 662>
        var_77 = wp::mul(var_59, var_62);
        // dFdt = wp.mat33(v10, v20, v30) * Dm                                                    <L 663>
        var_78 = wp::mat_t<3,3,wp::float32>(var_56, var_57, var_58);
        var_79 = wp::mul(var_78, var_62);
        // col1 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])                                              <L 665>
        var_80 = wp::extract(var_77, var_1, var_1);
        var_81 = wp::extract(var_77, var_5, var_1);
        var_82 = wp::extract(var_77, var_9, var_1);
        var_83 = wp::vec_t<3,wp::float32>(var_80, var_81, var_82);
        // col2 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])                                              <L 666>
        var_84 = wp::extract(var_77, var_1, var_5);
        var_85 = wp::extract(var_77, var_5, var_5);
        var_86 = wp::extract(var_77, var_9, var_5);
        var_87 = wp::vec_t<3,wp::float32>(var_84, var_85, var_86);
        // col3 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])                                              <L 667>
        var_88 = wp::extract(var_77, var_1, var_9);
        var_89 = wp::extract(var_77, var_5, var_9);
        var_90 = wp::extract(var_77, var_9, var_9);
        var_91 = wp::vec_t<3,wp::float32>(var_88, var_89, var_90);
        // Ic = dot(col1, col1) + dot(col2, col2) + dot(col3, col3)                               <L 672>
        var_92 = wp::dot(var_83, var_83);
        var_93 = wp::dot(var_87, var_87);
        var_94 = wp::add(var_92, var_93);
        var_95 = wp::dot(var_91, var_91);
        var_96 = wp::add(var_94, var_95);
        // P = F * k_mu * (1.0 - 1.0 / (Ic + 1.0)) + dFdt * k_damp                                <L 675>
        var_97 = wp::mul(var_77, var_74);
        var_98 = wp::add(var_96, var_66);
        var_99 = wp::div(var_66, var_98);
        var_100 = wp::sub(var_66, var_99);
        var_101 = wp::mul(var_97, var_100);
        var_102 = wp::mul(var_79, var_76);
        var_103 = wp::add(var_101, var_102);
        // H = P * wp.transpose(Dm)                                                               <L 676>
        var_104 = wp::transpose(var_62);
        var_105 = wp::mul(var_103, var_104);
        // f1 = wp.vec3(H[0, 0], H[1, 0], H[2, 0])                                                <L 678>
        var_106 = wp::extract(var_105, var_1, var_1);
        var_107 = wp::extract(var_105, var_5, var_1);
        var_108 = wp::extract(var_105, var_9, var_1);
        var_109 = wp::vec_t<3,wp::float32>(var_106, var_107, var_108);
        // f2 = wp.vec3(H[0, 1], H[1, 1], H[2, 1])                                                <L 679>
        var_110 = wp::extract(var_105, var_1, var_5);
        var_111 = wp::extract(var_105, var_5, var_5);
        var_112 = wp::extract(var_105, var_9, var_5);
        var_113 = wp::vec_t<3,wp::float32>(var_110, var_111, var_112);
        // f3 = wp.vec3(H[0, 2], H[1, 2], H[2, 2])                                                <L 680>
        var_114 = wp::extract(var_105, var_1, var_9);
        var_115 = wp::extract(var_105, var_5, var_9);
        var_116 = wp::extract(var_105, var_9, var_9);
        var_117 = wp::vec_t<3,wp::float32>(var_114, var_115, var_116);
        // J = wp.determinant(F)                                                                  <L 761>
        var_118 = wp::determinant(var_77);
        // s = inv_rest_volume / 6.0                                                              <L 764>
        var_119 = wp::div(var_65, var_64);
        // dJdx1 = wp.cross(x20, x30) * s                                                         <L 765>
        var_120 = wp::cross(var_54, var_55);
        var_121 = wp::mul(var_120, var_119);
        // dJdx2 = wp.cross(x30, x10) * s                                                         <L 766>
        var_122 = wp::cross(var_55, var_53);
        var_123 = wp::mul(var_122, var_119);
        // dJdx3 = wp.cross(x10, x20) * s                                                         <L 767>
        var_124 = wp::cross(var_53, var_54);
        var_125 = wp::mul(var_124, var_119);
        // f_volume = (J - alpha + act) * k_lambda                                                <L 769>
        var_126 = wp::sub(var_118, var_73);
        var_127 = wp::add(var_126, var_19);
        var_128 = wp::mul(var_127, var_75);
        // f_damp = (wp.dot(dJdx1, v1) + wp.dot(dJdx2, v2) + wp.dot(dJdx3, v3)) * k_damp          <L 770>
        var_129 = wp::dot(var_121, var_46);
        var_130 = wp::dot(var_123, var_49);
        var_131 = wp::add(var_129, var_130);
        var_132 = wp::dot(var_125, var_52);
        var_133 = wp::add(var_131, var_132);
        var_134 = wp::mul(var_133, var_76);
        // f_total = f_volume + f_damp                                                            <L 772>
        var_135 = wp::add(var_128, var_134);
        // f1 = f1 + dJdx1 * f_total                                                              <L 774>
        var_136 = wp::mul(var_121, var_135);
        var_137 = wp::add(var_109, var_136);
        // f2 = f2 + dJdx2 * f_total                                                              <L 775>
        var_138 = wp::mul(var_123, var_135);
        var_139 = wp::add(var_113, var_138);
        // f3 = f3 + dJdx3 * f_total                                                              <L 776>
        var_140 = wp::mul(var_125, var_135);
        var_141 = wp::add(var_117, var_140);
        // f0 = (f1 + f2 + f3) * (0.0 - 1.0)                                                      <L 777>
        var_142 = wp::add(var_137, var_139);
        var_143 = wp::add(var_142, var_141);
        var_145 = wp::sub(var_144, var_66);
        var_146 = wp::mul(var_143, var_145);
        // wp.atomic_sub(f, i, f0)                                                                <L 780>
        // var_147 = wp::atomic_sub(var_f, var_4, var_146);
        // wp.atomic_sub(f, j, f1)                                                                <L 781>
        // var_148 = wp::atomic_sub(var_f, var_8, var_137);
        // wp.atomic_sub(f, k, f2)                                                                <L 782>
        // var_149 = wp::atomic_sub(var_f, var_12, var_139);
        // wp.atomic_sub(f, l, f3)                                                                <L 783>
        // var_150 = wp::atomic_sub(var_f, var_16, var_141);
        //---------
        // reverse
        wp::adj_atomic_sub(var_f, var_16, var_141, adj_f, adj_16, adj_141, adj_150);
        // adj: wp.atomic_sub(f, l, f3)                                                           <L 783>
        wp::adj_atomic_sub(var_f, var_12, var_139, adj_f, adj_12, adj_139, adj_149);
        // adj: wp.atomic_sub(f, k, f2)                                                           <L 782>
        wp::adj_atomic_sub(var_f, var_8, var_137, adj_f, adj_8, adj_137, adj_148);
        // adj: wp.atomic_sub(f, j, f1)                                                           <L 781>
        wp::adj_atomic_sub(var_f, var_4, var_146, adj_f, adj_4, adj_146, adj_147);
        // adj: wp.atomic_sub(f, i, f0)                                                           <L 780>
        wp::adj_mul(var_143, var_145, adj_143, adj_145, adj_146);
        wp::adj_sub(var_144, var_66, adj_144, adj_66, adj_145);
        wp::adj_add(var_142, var_141, adj_142, adj_141, adj_143);
        wp::adj_add(var_137, var_139, adj_137, adj_139, adj_142);
        // adj: f0 = (f1 + f2 + f3) * (0.0 - 1.0)                                                 <L 777>
        wp::adj_add(var_117, var_140, adj_117, adj_140, adj_141);
        wp::adj_mul(var_125, var_135, adj_125, adj_135, adj_140);
        // adj: f3 = f3 + dJdx3 * f_total                                                         <L 776>
        wp::adj_add(var_113, var_138, adj_113, adj_138, adj_139);
        wp::adj_mul(var_123, var_135, adj_123, adj_135, adj_138);
        // adj: f2 = f2 + dJdx2 * f_total                                                         <L 775>
        wp::adj_add(var_109, var_136, adj_109, adj_136, adj_137);
        wp::adj_mul(var_121, var_135, adj_121, adj_135, adj_136);
        // adj: f1 = f1 + dJdx1 * f_total                                                         <L 774>
        wp::adj_add(var_128, var_134, adj_128, adj_134, adj_135);
        // adj: f_total = f_volume + f_damp                                                       <L 772>
        wp::adj_mul(var_133, var_76, adj_133, adj_76, adj_134);
        wp::adj_add(var_131, var_132, adj_131, adj_132, adj_133);
        wp::adj_dot(var_125, var_52, adj_125, adj_52, adj_132);
        wp::adj_add(var_129, var_130, adj_129, adj_130, adj_131);
        wp::adj_dot(var_123, var_49, adj_123, adj_49, adj_130);
        wp::adj_dot(var_121, var_46, adj_121, adj_46, adj_129);
        // adj: f_damp = (wp.dot(dJdx1, v1) + wp.dot(dJdx2, v2) + wp.dot(dJdx3, v3)) * k_damp     <L 770>
        wp::adj_mul(var_127, var_75, adj_127, adj_75, adj_128);
        wp::adj_add(var_126, var_19, adj_126, adj_19, adj_127);
        wp::adj_sub(var_118, var_73, adj_118, adj_73, adj_126);
        // adj: f_volume = (J - alpha + act) * k_lambda                                           <L 769>
        wp::adj_mul(var_124, var_119, adj_124, adj_119, adj_125);
        wp::adj_cross(var_53, var_54, adj_53, adj_54, adj_124);
        // adj: dJdx3 = wp.cross(x10, x20) * s                                                    <L 767>
        wp::adj_mul(var_122, var_119, adj_122, adj_119, adj_123);
        wp::adj_cross(var_55, var_53, adj_55, adj_53, adj_122);
        // adj: dJdx2 = wp.cross(x30, x10) * s                                                    <L 766>
        wp::adj_mul(var_120, var_119, adj_120, adj_119, adj_121);
        wp::adj_cross(var_54, var_55, adj_54, adj_55, adj_120);
        // adj: dJdx1 = wp.cross(x20, x30) * s                                                    <L 765>
        wp::adj_div(var_65, var_64, var_119, adj_65, adj_64, adj_119);
        // adj: s = inv_rest_volume / 6.0                                                         <L 764>
        wp::adj_determinant(var_77, adj_77, adj_118);
        // adj: J = wp.determinant(F)                                                             <L 761>
        wp::adj_vec_t(var_114, var_115, var_116, adj_114, adj_115, adj_116, adj_117);
        wp::adj_extract(var_105, var_9, var_9, adj_105, adj_9, adj_9, adj_116);
        wp::adj_extract(var_105, var_5, var_9, adj_105, adj_5, adj_9, adj_115);
        wp::adj_extract(var_105, var_1, var_9, adj_105, adj_1, adj_9, adj_114);
        // adj: f3 = wp.vec3(H[0, 2], H[1, 2], H[2, 2])                                           <L 680>
        wp::adj_vec_t(var_110, var_111, var_112, adj_110, adj_111, adj_112, adj_113);
        wp::adj_extract(var_105, var_9, var_5, adj_105, adj_9, adj_5, adj_112);
        wp::adj_extract(var_105, var_5, var_5, adj_105, adj_5, adj_5, adj_111);
        wp::adj_extract(var_105, var_1, var_5, adj_105, adj_1, adj_5, adj_110);
        // adj: f2 = wp.vec3(H[0, 1], H[1, 1], H[2, 1])                                           <L 679>
        wp::adj_vec_t(var_106, var_107, var_108, adj_106, adj_107, adj_108, adj_109);
        wp::adj_extract(var_105, var_9, var_1, adj_105, adj_9, adj_1, adj_108);
        wp::adj_extract(var_105, var_5, var_1, adj_105, adj_5, adj_1, adj_107);
        wp::adj_extract(var_105, var_1, var_1, adj_105, adj_1, adj_1, adj_106);
        // adj: f1 = wp.vec3(H[0, 0], H[1, 0], H[2, 0])                                           <L 678>
        wp::adj_mul(var_103, var_104, adj_103, adj_104, adj_105);
        wp::adj_transpose(var_62, adj_62, adj_104);
        // adj: H = P * wp.transpose(Dm)                                                          <L 676>
        wp::adj_add(var_101, var_102, adj_101, adj_102, adj_103);
        wp::adj_mul(var_79, var_76, adj_79, adj_76, adj_102);
        wp::adj_mul(var_97, var_100, adj_97, adj_100, adj_101);
        wp::adj_sub(var_66, var_99, adj_66, adj_99, adj_100);
        wp::adj_div(var_66, var_98, var_99, adj_66, adj_98, adj_99);
        wp::adj_add(var_96, var_66, adj_96, adj_66, adj_98);
        wp::adj_mul(var_77, var_74, adj_77, adj_74, adj_97);
        // adj: P = F * k_mu * (1.0 - 1.0 / (Ic + 1.0)) + dFdt * k_damp                           <L 675>
        wp::adj_add(var_94, var_95, adj_94, adj_95, adj_96);
        wp::adj_dot(var_91, var_91, adj_91, adj_91, adj_95);
        wp::adj_add(var_92, var_93, adj_92, adj_93, adj_94);
        wp::adj_dot(var_87, var_87, adj_87, adj_87, adj_93);
        wp::adj_dot(var_83, var_83, adj_83, adj_83, adj_92);
        // adj: Ic = dot(col1, col1) + dot(col2, col2) + dot(col3, col3)                          <L 672>
        wp::adj_vec_t(var_88, var_89, var_90, adj_88, adj_89, adj_90, adj_91);
        wp::adj_extract(var_77, var_9, var_9, adj_77, adj_9, adj_9, adj_90);
        wp::adj_extract(var_77, var_5, var_9, adj_77, adj_5, adj_9, adj_89);
        wp::adj_extract(var_77, var_1, var_9, adj_77, adj_1, adj_9, adj_88);
        // adj: col3 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])                                         <L 667>
        wp::adj_vec_t(var_84, var_85, var_86, adj_84, adj_85, adj_86, adj_87);
        wp::adj_extract(var_77, var_9, var_5, adj_77, adj_9, adj_5, adj_86);
        wp::adj_extract(var_77, var_5, var_5, adj_77, adj_5, adj_5, adj_85);
        wp::adj_extract(var_77, var_1, var_5, adj_77, adj_1, adj_5, adj_84);
        // adj: col2 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])                                         <L 666>
        wp::adj_vec_t(var_80, var_81, var_82, adj_80, adj_81, adj_82, adj_83);
        wp::adj_extract(var_77, var_9, var_1, adj_77, adj_9, adj_1, adj_82);
        wp::adj_extract(var_77, var_5, var_1, adj_77, adj_5, adj_1, adj_81);
        wp::adj_extract(var_77, var_1, var_1, adj_77, adj_1, adj_1, adj_80);
        // adj: col1 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])                                         <L 665>
        wp::adj_mul(var_78, var_62, adj_78, adj_62, adj_79);
        wp::adj_mat_t(var_56, var_57, var_58, adj_56, adj_57, adj_58, adj_78);
        // adj: dFdt = wp.mat33(v10, v20, v30) * Dm                                               <L 663>
        wp::adj_mul(var_59, var_62, adj_59, adj_62, adj_77);
        // adj: F = Ds * Dm                                                                       <L 662>
        wp::adj_mul(var_28, var_67, adj_28, adj_67, adj_76);
        // adj: k_damp = k_damp * rest_volume                                                     <L 659>
        wp::adj_mul(var_25, var_67, adj_25, adj_67, adj_75);
        // adj: k_lambda = k_lambda * rest_volume                                                 <L 658>
        wp::adj_mul(var_22, var_67, adj_22, adj_67, adj_74);
        // adj: k_mu = k_mu * rest_volume                                                         <L 657>
        wp::adj_sub(var_69, var_72, adj_69, adj_72, adj_73);
        wp::adj_div(var_22, var_71, var_72, adj_22, adj_71, adj_72);
        wp::adj_mul(var_70, var_25, adj_70, adj_25, adj_71);
        wp::adj_add(var_66, var_68, adj_66, adj_68, adj_69);
        wp::adj_div(var_22, var_25, var_68, adj_22, adj_25, adj_68);
        // adj: alpha = 1.0 + k_mu / k_lambda - k_mu / (4.0 * k_lambda)                           <L 654>
        wp::adj_div(var_66, var_65, var_67, adj_66, adj_65, adj_67);
        // adj: rest_volume = 1.0 / inv_rest_volume                                               <L 652>
        wp::adj_mul(var_63, var_64, adj_63, adj_64, adj_65);
        wp::adj_determinant(var_62, adj_62, adj_63);
        // adj: inv_rest_volume = wp.determinant(Dm) * 6.0                                        <L 651>
        wp::adj_copy(var_61, adj_60, adj_62);
        wp::adj_load(var_60, adj_60, adj_61);
        wp::adj_address(var_pose, var_0, adj_pose, adj_0, adj_60);
        // adj: Dm = pose[tid]                                                                    <L 649>
        wp::adj_mat_t(var_53, var_54, var_55, adj_53, adj_54, adj_55, adj_59);
        // adj: Ds = wp.mat33(x10, x20, x30)                                                      <L 648>
        wp::adj_sub(var_52, var_43, adj_52, adj_43, adj_58);
        // adj: v30 = v3 - v0                                                                     <L 646>
        wp::adj_sub(var_49, var_43, adj_49, adj_43, adj_57);
        // adj: v20 = v2 - v0                                                                     <L 645>
        wp::adj_sub(var_46, var_43, adj_46, adj_43, adj_56);
        // adj: v10 = v1 - v0                                                                     <L 644>
        wp::adj_sub(var_40, var_31, adj_40, adj_31, adj_55);
        // adj: x30 = x3 - x0                                                                     <L 642>
        wp::adj_sub(var_37, var_31, adj_37, adj_31, adj_54);
        // adj: x20 = x2 - x0                                                                     <L 641>
        wp::adj_sub(var_34, var_31, adj_34, adj_31, adj_53);
        // adj: x10 = x1 - x0                                                                     <L 640>
        wp::adj_copy(var_51, adj_50, adj_52);
        wp::adj_load(var_50, adj_50, adj_51);
        wp::adj_address(var_v, var_16, adj_v, adj_16, adj_50);
        // adj: v3 = v[l]                                                                         <L 638>
        wp::adj_copy(var_48, adj_47, adj_49);
        wp::adj_load(var_47, adj_47, adj_48);
        wp::adj_address(var_v, var_12, adj_v, adj_12, adj_47);
        // adj: v2 = v[k]                                                                         <L 637>
        wp::adj_copy(var_45, adj_44, adj_46);
        wp::adj_load(var_44, adj_44, adj_45);
        wp::adj_address(var_v, var_8, adj_v, adj_8, adj_44);
        // adj: v1 = v[j]                                                                         <L 636>
        wp::adj_copy(var_42, adj_41, adj_43);
        wp::adj_load(var_41, adj_41, adj_42);
        wp::adj_address(var_v, var_4, adj_v, adj_4, adj_41);
        // adj: v0 = v[i]                                                                         <L 635>
        wp::adj_copy(var_39, adj_38, adj_40);
        wp::adj_load(var_38, adj_38, adj_39);
        wp::adj_address(var_x, var_16, adj_x, adj_16, adj_38);
        // adj: x3 = x[l]                                                                         <L 633>
        wp::adj_copy(var_36, adj_35, adj_37);
        wp::adj_load(var_35, adj_35, adj_36);
        wp::adj_address(var_x, var_12, adj_x, adj_12, adj_35);
        // adj: x2 = x[k]                                                                         <L 632>
        wp::adj_copy(var_33, adj_32, adj_34);
        wp::adj_load(var_32, adj_32, adj_33);
        wp::adj_address(var_x, var_8, adj_x, adj_8, adj_32);
        // adj: x1 = x[j]                                                                         <L 631>
        wp::adj_copy(var_30, adj_29, adj_31);
        wp::adj_load(var_29, adj_29, adj_30);
        wp::adj_address(var_x, var_4, adj_x, adj_4, adj_29);
        // adj: x0 = x[i]                                                                         <L 630>
        wp::adj_copy(var_27, adj_26, adj_28);
        wp::adj_load(var_26, adj_26, adj_27);
        wp::adj_address(var_materials, var_0, var_9, adj_materials, adj_0, adj_9, adj_26);
        // adj: k_damp = materials[tid, 2]                                                        <L 628>
        wp::adj_copy(var_24, adj_23, adj_25);
        wp::adj_load(var_23, adj_23, adj_24);
        wp::adj_address(var_materials, var_0, var_5, adj_materials, adj_0, adj_5, adj_23);
        // adj: k_lambda = materials[tid, 1]                                                      <L 627>
        wp::adj_copy(var_21, adj_20, adj_22);
        wp::adj_load(var_20, adj_20, adj_21);
        wp::adj_address(var_materials, var_0, var_1, adj_materials, adj_0, adj_1, adj_20);
        // adj: k_mu = materials[tid, 0]                                                          <L 626>
        wp::adj_copy(var_18, adj_17, adj_19);
        wp::adj_load(var_17, adj_17, adj_18);
        wp::adj_address(var_activation, var_0, adj_activation, adj_0, adj_17);
        // adj: act = activation[tid]                                                             <L 624>
        wp::adj_copy(var_15, adj_14, adj_16);
        wp::adj_load(var_14, adj_14, adj_15);
        wp::adj_address(var_indices, var_0, var_13, adj_indices, adj_0, adj_13, adj_14);
        // adj: l = indices[tid, 3]                                                               <L 622>
        wp::adj_copy(var_11, adj_10, adj_12);
        wp::adj_load(var_10, adj_10, adj_11);
        wp::adj_address(var_indices, var_0, var_9, adj_indices, adj_0, adj_9, adj_10);
        // adj: k = indices[tid, 2]                                                               <L 621>
        wp::adj_copy(var_7, adj_6, adj_8);
        wp::adj_load(var_6, adj_6, adj_7);
        wp::adj_address(var_indices, var_0, var_5, adj_indices, adj_0, adj_5, adj_6);
        // adj: j = indices[tid, 1]                                                               <L 620>
        wp::adj_copy(var_3, adj_2, adj_4);
        wp::adj_load(var_2, adj_2, adj_3);
        wp::adj_address(var_indices, var_0, var_1, adj_indices, adj_0, adj_1, adj_2);
        // adj: i = indices[tid, 0]                                                               <L 619>
        // adj: tid = wp.tid()                                                                    <L 617>
        // adj: def eval_tetrahedra(                                                              <L 608>
        continue;
    }
}



extern "C" __global__ void eval_particle_ground_contacts_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_x,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_v,
    wp::array_t<wp::float32> var_particle_radius,
    wp::array_t<wp::uint32> var_particle_flags,
    wp::float32 var_ke,
    wp::float32 var_kd,
    wp::float32 var_kf,
    wp::float32 var_mu,
    wp::array_t<wp::float32> var_ground,
    wp::array_t<wp::vec_t<3,wp::float32>> var_f)
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
        wp::float32* var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        wp::float32* var_16;
        const wp::int32 var_17 = 1;
        wp::float32* var_18;
        const wp::int32 var_19 = 2;
        wp::float32* var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::vec_t<3,wp::float32> var_24;
        wp::float32 var_25;
        const wp::int32 var_26 = 3;
        wp::float32* var_27;
        wp::float32 var_28;
        wp::float32 var_29;
        wp::float32 var_30;
        const wp::float32 var_31 = 0.0;
        wp::float32 var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        bool var_35;
        wp::float32 var_36;
        wp::float32 var_37;
        wp::float32 var_38;
        wp::vec_t<3,wp::float32> var_39;
        wp::vec_t<3,wp::float32> var_40;
        wp::float32 var_41;
        bool var_42;
        wp::vec_t<3,wp::float32> var_43;
        wp::vec_t<3,wp::float32> var_44;
        wp::float32 var_45;
        wp::float32 var_46;
        wp::float32 var_47;
        wp::float32 var_48;
        wp::vec_t<3,wp::float32>* var_49;
        wp::vec_t<3,wp::float32> var_50;
        wp::vec_t<3,wp::float32> var_51;
        wp::vec_t<3,wp::float32> var_52;
        wp::vec_t<3,wp::float32> var_53;
        wp::vec_t<3,wp::float32> var_54;
        //---------
        // forward
        // def eval_particle_ground_contacts(                                                     <L 787>
        // tid = wp.tid()                                                                         <L 800>
        var_0 = builtin_tid1d();
        // if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:                                  <L 801>
        var_1 = wp::address(var_particle_flags, var_0);
        var_3 = wp::load(var_1);
        var_4 = wp::bit_and(var_3, var_2);
        var_6 = (var_4 == var_5);
        if (var_6) {
            // return                                                                             <L 802>
            return;
        }
        // x = particle_x[tid]                                                                    <L 804>
        var_7 = wp::address(var_particle_x, var_0);
        var_8 = wp::load(var_7);
        var_9 = wp::copy(var_8);
        // v = particle_v[tid]                                                                    <L 805>
        var_10 = wp::address(var_particle_v, var_0);
        var_11 = wp::load(var_10);
        var_12 = wp::copy(var_11);
        // radius = particle_radius[tid]                                                          <L 806>
        var_13 = wp::address(var_particle_radius, var_0);
        var_14 = wp::load(var_13);
        var_15 = wp::copy(var_14);
        // n = wp.vec3(ground[0], ground[1], ground[2])                                           <L 808>
        var_16 = wp::address(var_ground, var_5);
        var_18 = wp::address(var_ground, var_17);
        var_20 = wp::address(var_ground, var_19);
        var_21 = wp::load(var_16);
        var_22 = wp::load(var_18);
        var_23 = wp::load(var_20);
        var_24 = wp::vec_t<3,wp::float32>(var_21, var_22, var_23);
        // c = wp.min(wp.dot(n, x) + ground[3] - radius, 0.0)                                     <L 809>
        var_25 = wp::dot(var_24, var_9);
        var_27 = wp::address(var_ground, var_26);
        var_28 = wp::load(var_27);
        var_29 = wp::add(var_25, var_28);
        var_30 = wp::sub(var_29, var_15);
        var_32 = wp::min(var_30, var_31);
        // vn = wp.dot(n, v)                                                                      <L 811>
        var_33 = wp::dot(var_24, var_12);
        // jn = c * ke                                                                            <L 812>
        var_34 = wp::mul(var_32, var_ke);
        // if c >= 0.0:                                                                           <L 814>
        var_35 = (var_32 >= var_31);
        if (var_35) {
            // return                                                                             <L 815>
            return;
        }
        // jd = min(vn, 0.0) * kd                                                                 <L 817>
        var_36 = wp::min(var_33, var_31);
        var_37 = wp::mul(var_36, var_kd);
        // fn = jn + jd                                                                           <L 820>
        var_38 = wp::add(var_34, var_37);
        // vt = v - n * vn                                                                        <L 823>
        var_39 = wp::mul(var_24, var_33);
        var_40 = wp::sub(var_12, var_39);
        // vs = wp.length(vt)                                                                     <L 824>
        var_41 = wp::length(var_40);
        // if vs > 0.0:                                                                           <L 826>
        var_42 = (var_41 > var_31);
        if (var_42) {
            // vt = vt / vs                                                                       <L 827>
            var_43 = wp::div(var_40, var_41);
        }
        var_44 = wp::select(var_42, var_40, var_43);
        // ft = wp.min(vs * kf, mu * wp.abs(fn))                                                  <L 830>
        var_45 = wp::mul(var_41, var_kf);
        var_46 = wp::abs(var_38);
        var_47 = wp::mul(var_mu, var_46);
        var_48 = wp::min(var_45, var_47);
        // f[tid] = f[tid] - n * fn - vt * ft                                                     <L 833>
        var_49 = wp::address(var_f, var_0);
        var_50 = wp::mul(var_24, var_38);
        var_51 = wp::load(var_49);
        var_52 = wp::sub(var_51, var_50);
        var_53 = wp::mul(var_44, var_48);
        var_54 = wp::sub(var_52, var_53);
        wp::array_store(var_f, var_0, var_54);
    }
}

extern "C" __global__ void eval_particle_ground_contacts_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_x,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_v,
    wp::array_t<wp::float32> var_particle_radius,
    wp::array_t<wp::uint32> var_particle_flags,
    wp::float32 var_ke,
    wp::float32 var_kd,
    wp::float32 var_kf,
    wp::float32 var_mu,
    wp::array_t<wp::float32> var_ground,
    wp::array_t<wp::vec_t<3,wp::float32>> var_f,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_particle_x,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_particle_v,
    wp::array_t<wp::float32> adj_particle_radius,
    wp::array_t<wp::uint32> adj_particle_flags,
    wp::float32 adj_ke,
    wp::float32 adj_kd,
    wp::float32 adj_kf,
    wp::float32 adj_mu,
    wp::array_t<wp::float32> adj_ground,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_f)
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
        wp::float32* var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        wp::float32* var_16;
        const wp::int32 var_17 = 1;
        wp::float32* var_18;
        const wp::int32 var_19 = 2;
        wp::float32* var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::vec_t<3,wp::float32> var_24;
        wp::float32 var_25;
        const wp::int32 var_26 = 3;
        wp::float32* var_27;
        wp::float32 var_28;
        wp::float32 var_29;
        wp::float32 var_30;
        const wp::float32 var_31 = 0.0;
        wp::float32 var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        bool var_35;
        wp::float32 var_36;
        wp::float32 var_37;
        wp::float32 var_38;
        wp::vec_t<3,wp::float32> var_39;
        wp::vec_t<3,wp::float32> var_40;
        wp::float32 var_41;
        bool var_42;
        wp::vec_t<3,wp::float32> var_43;
        wp::vec_t<3,wp::float32> var_44;
        wp::float32 var_45;
        wp::float32 var_46;
        wp::float32 var_47;
        wp::float32 var_48;
        wp::vec_t<3,wp::float32>* var_49;
        wp::vec_t<3,wp::float32> var_50;
        wp::vec_t<3,wp::float32> var_51;
        wp::vec_t<3,wp::float32> var_52;
        wp::vec_t<3,wp::float32> var_53;
        wp::vec_t<3,wp::float32> var_54;
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
        wp::float32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::float32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::int32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::vec_t<3,wp::float32> adj_24 = {};
        wp::float32 adj_25 = {};
        wp::int32 adj_26 = {};
        wp::float32 adj_27 = {};
        wp::float32 adj_28 = {};
        wp::float32 adj_29 = {};
        wp::float32 adj_30 = {};
        wp::float32 adj_31 = {};
        wp::float32 adj_32 = {};
        wp::float32 adj_33 = {};
        wp::float32 adj_34 = {};
        bool adj_35 = {};
        wp::float32 adj_36 = {};
        wp::float32 adj_37 = {};
        wp::float32 adj_38 = {};
        wp::vec_t<3,wp::float32> adj_39 = {};
        wp::vec_t<3,wp::float32> adj_40 = {};
        wp::float32 adj_41 = {};
        bool adj_42 = {};
        wp::vec_t<3,wp::float32> adj_43 = {};
        wp::vec_t<3,wp::float32> adj_44 = {};
        wp::float32 adj_45 = {};
        wp::float32 adj_46 = {};
        wp::float32 adj_47 = {};
        wp::float32 adj_48 = {};
        wp::vec_t<3,wp::float32> adj_49 = {};
        wp::vec_t<3,wp::float32> adj_50 = {};
        wp::vec_t<3,wp::float32> adj_51 = {};
        wp::vec_t<3,wp::float32> adj_52 = {};
        wp::vec_t<3,wp::float32> adj_53 = {};
        wp::vec_t<3,wp::float32> adj_54 = {};
        //---------
        // forward
        // def eval_particle_ground_contacts(                                                     <L 787>
        // tid = wp.tid()                                                                         <L 800>
        var_0 = builtin_tid1d();
        // if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:                                  <L 801>
        var_1 = wp::address(var_particle_flags, var_0);
        var_3 = wp::load(var_1);
        var_4 = wp::bit_and(var_3, var_2);
        var_6 = (var_4 == var_5);
        if (var_6) {
            // return                                                                             <L 802>
            goto label0;
        }
        // x = particle_x[tid]                                                                    <L 804>
        var_7 = wp::address(var_particle_x, var_0);
        var_8 = wp::load(var_7);
        var_9 = wp::copy(var_8);
        // v = particle_v[tid]                                                                    <L 805>
        var_10 = wp::address(var_particle_v, var_0);
        var_11 = wp::load(var_10);
        var_12 = wp::copy(var_11);
        // radius = particle_radius[tid]                                                          <L 806>
        var_13 = wp::address(var_particle_radius, var_0);
        var_14 = wp::load(var_13);
        var_15 = wp::copy(var_14);
        // n = wp.vec3(ground[0], ground[1], ground[2])                                           <L 808>
        var_16 = wp::address(var_ground, var_5);
        var_18 = wp::address(var_ground, var_17);
        var_20 = wp::address(var_ground, var_19);
        var_21 = wp::load(var_16);
        var_22 = wp::load(var_18);
        var_23 = wp::load(var_20);
        var_24 = wp::vec_t<3,wp::float32>(var_21, var_22, var_23);
        // c = wp.min(wp.dot(n, x) + ground[3] - radius, 0.0)                                     <L 809>
        var_25 = wp::dot(var_24, var_9);
        var_27 = wp::address(var_ground, var_26);
        var_28 = wp::load(var_27);
        var_29 = wp::add(var_25, var_28);
        var_30 = wp::sub(var_29, var_15);
        var_32 = wp::min(var_30, var_31);
        // vn = wp.dot(n, v)                                                                      <L 811>
        var_33 = wp::dot(var_24, var_12);
        // jn = c * ke                                                                            <L 812>
        var_34 = wp::mul(var_32, var_ke);
        // if c >= 0.0:                                                                           <L 814>
        var_35 = (var_32 >= var_31);
        if (var_35) {
            // return                                                                             <L 815>
            goto label1;
        }
        // jd = min(vn, 0.0) * kd                                                                 <L 817>
        var_36 = wp::min(var_33, var_31);
        var_37 = wp::mul(var_36, var_kd);
        // fn = jn + jd                                                                           <L 820>
        var_38 = wp::add(var_34, var_37);
        // vt = v - n * vn                                                                        <L 823>
        var_39 = wp::mul(var_24, var_33);
        var_40 = wp::sub(var_12, var_39);
        // vs = wp.length(vt)                                                                     <L 824>
        var_41 = wp::length(var_40);
        // if vs > 0.0:                                                                           <L 826>
        var_42 = (var_41 > var_31);
        if (var_42) {
            // vt = vt / vs                                                                       <L 827>
            var_43 = wp::div(var_40, var_41);
        }
        var_44 = wp::select(var_42, var_40, var_43);
        // ft = wp.min(vs * kf, mu * wp.abs(fn))                                                  <L 830>
        var_45 = wp::mul(var_41, var_kf);
        var_46 = wp::abs(var_38);
        var_47 = wp::mul(var_mu, var_46);
        var_48 = wp::min(var_45, var_47);
        // f[tid] = f[tid] - n * fn - vt * ft                                                     <L 833>
        var_49 = wp::address(var_f, var_0);
        var_50 = wp::mul(var_24, var_38);
        var_51 = wp::load(var_49);
        var_52 = wp::sub(var_51, var_50);
        var_53 = wp::mul(var_44, var_48);
        var_54 = wp::sub(var_52, var_53);
        // wp::array_store(var_f, var_0, var_54);
        //---------
        // reverse
        wp::adj_array_store(var_f, var_0, var_54, adj_f, adj_0, adj_54);
        wp::adj_sub(var_52, var_53, adj_52, adj_53, adj_54);
        wp::adj_mul(var_44, var_48, adj_44, adj_48, adj_53);
        wp::adj_sub(var_51, var_50, adj_49, adj_50, adj_52);
        wp::adj_load(var_49, adj_49, adj_51);
        wp::adj_mul(var_24, var_38, adj_24, adj_38, adj_50);
        wp::adj_address(var_f, var_0, adj_f, adj_0, adj_49);
        // adj: f[tid] = f[tid] - n * fn - vt * ft                                                <L 833>
        wp::adj_min(var_45, var_47, adj_45, adj_47, adj_48);
        wp::adj_mul(var_mu, var_46, adj_mu, adj_46, adj_47);
        wp::adj_abs(var_38, adj_38, adj_46);
        wp::adj_mul(var_41, var_kf, adj_41, adj_kf, adj_45);
        // adj: ft = wp.min(vs * kf, mu * wp.abs(fn))                                             <L 830>
        wp::adj_select(var_42, var_40, var_43, adj_42, adj_40, adj_43, adj_44);
        if (var_42) {
            wp::adj_div(var_40, var_41, adj_40, adj_41, adj_43);
            // adj: vt = vt / vs                                                                  <L 827>
        }
        // adj: if vs > 0.0:                                                                      <L 826>
        wp::adj_length(var_40, var_41, adj_40, adj_41);
        // adj: vs = wp.length(vt)                                                                <L 824>
        wp::adj_sub(var_12, var_39, adj_12, adj_39, adj_40);
        wp::adj_mul(var_24, var_33, adj_24, adj_33, adj_39);
        // adj: vt = v - n * vn                                                                   <L 823>
        wp::adj_add(var_34, var_37, adj_34, adj_37, adj_38);
        // adj: fn = jn + jd                                                                      <L 820>
        wp::adj_mul(var_36, var_kd, adj_36, adj_kd, adj_37);
        wp::adj_min(var_33, var_31, adj_33, adj_31, adj_36);
        // adj: jd = min(vn, 0.0) * kd                                                            <L 817>
        if (var_35) {
            label1:;
            // adj: return                                                                        <L 815>
        }
        // adj: if c >= 0.0:                                                                      <L 814>
        wp::adj_mul(var_32, var_ke, adj_32, adj_ke, adj_34);
        // adj: jn = c * ke                                                                       <L 812>
        wp::adj_dot(var_24, var_12, adj_24, adj_12, adj_33);
        // adj: vn = wp.dot(n, v)                                                                 <L 811>
        wp::adj_min(var_30, var_31, adj_30, adj_31, adj_32);
        wp::adj_sub(var_29, var_15, adj_29, adj_15, adj_30);
        wp::adj_add(var_25, var_28, adj_25, adj_27, adj_29);
        wp::adj_load(var_27, adj_27, adj_28);
        wp::adj_address(var_ground, var_26, adj_ground, adj_26, adj_27);
        wp::adj_dot(var_24, var_9, adj_24, adj_9, adj_25);
        // adj: c = wp.min(wp.dot(n, x) + ground[3] - radius, 0.0)                                <L 809>
        wp::adj_vec_t(var_21, var_22, var_23, adj_16, adj_18, adj_20, adj_24);
        wp::adj_load(var_20, adj_20, adj_23);
        wp::adj_load(var_18, adj_18, adj_22);
        wp::adj_load(var_16, adj_16, adj_21);
        wp::adj_address(var_ground, var_19, adj_ground, adj_19, adj_20);
        wp::adj_address(var_ground, var_17, adj_ground, adj_17, adj_18);
        wp::adj_address(var_ground, var_5, adj_ground, adj_5, adj_16);
        // adj: n = wp.vec3(ground[0], ground[1], ground[2])                                      <L 808>
        wp::adj_copy(var_14, adj_13, adj_15);
        wp::adj_load(var_13, adj_13, adj_14);
        wp::adj_address(var_particle_radius, var_0, adj_particle_radius, adj_0, adj_13);
        // adj: radius = particle_radius[tid]                                                     <L 806>
        wp::adj_copy(var_11, adj_10, adj_12);
        wp::adj_load(var_10, adj_10, adj_11);
        wp::adj_address(var_particle_v, var_0, adj_particle_v, adj_0, adj_10);
        // adj: v = particle_v[tid]                                                               <L 805>
        wp::adj_copy(var_8, adj_7, adj_9);
        wp::adj_load(var_7, adj_7, adj_8);
        wp::adj_address(var_particle_x, var_0, adj_particle_x, adj_0, adj_7);
        // adj: x = particle_x[tid]                                                               <L 804>
        if (var_6) {
            label0:;
            // adj: return                                                                        <L 802>
        }
        wp::adj_bit_and(var_3, var_2, adj_1, adj_2, adj_4);
        wp::adj_load(var_1, adj_1, adj_3);
        wp::adj_address(var_particle_flags, var_0, adj_particle_flags, adj_0, adj_1);
        // adj: if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:                             <L 801>
        // adj: tid = wp.tid()                                                                    <L 800>
        // adj: def eval_particle_ground_contacts(                                                <L 787>
        continue;
    }
}



extern "C" __global__ void eval_particle_contacts_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_x,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_v,
    wp::array_t<wp::float32> var_particle_radius,
    wp::array_t<wp::uint32> var_particle_flags,
    wp::array_t<wp::transform_t<wp::float32>> var_body_q,
    wp::array_t<wp::vec_t<6,wp::float32>> var_body_qd,
    wp::array_t<wp::vec_t<3,wp::float32>> var_body_com,
    wp::array_t<wp::int32> var_shape_body,
    ModelShapeMaterials var_shape_materials,
    wp::float32 var_particle_ke,
    wp::float32 var_particle_kd,
    wp::float32 var_particle_kf,
    wp::float32 var_particle_mu,
    wp::float32 var_particle_ka,
    wp::array_t<wp::int32> var_contact_count,
    wp::array_t<wp::int32> var_contact_particle,
    wp::array_t<wp::int32> var_contact_shape,
    wp::array_t<wp::vec_t<3,wp::float32>> var_contact_body_pos,
    wp::array_t<wp::vec_t<3,wp::float32>> var_contact_body_vel,
    wp::array_t<wp::vec_t<3,wp::float32>> var_contact_normal,
    wp::int32 var_contact_max,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_f,
    wp::array_t<wp::vec_t<6,wp::float32>> var_body_f)
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
        wp::vec_t<6,wp::float32> var_28;
        bool var_29;
        wp::transform_t<wp::float32>* var_30;
        wp::transform_t<wp::float32> var_31;
        wp::transform_t<wp::float32> var_32;
        wp::vec_t<3,wp::float32>* var_33;
        wp::vec_t<3,wp::float32> var_34;
        wp::vec_t<3,wp::float32> var_35;
        wp::vec_t<6,wp::float32>* var_36;
        wp::vec_t<6,wp::float32> var_37;
        wp::vec_t<6,wp::float32> var_38;
        wp::transform_t<wp::float32> var_39;
        wp::vec_t<3,wp::float32> var_40;
        wp::vec_t<6,wp::float32> var_41;
        wp::vec_t<3,wp::float32>* var_42;
        wp::vec_t<3,wp::float32> var_43;
        wp::vec_t<3,wp::float32> var_44;
        wp::vec_t<3,wp::float32> var_45;
        wp::vec_t<3,wp::float32> var_46;
        wp::vec_t<3,wp::float32>* var_47;
        wp::vec_t<3,wp::float32> var_48;
        wp::vec_t<3,wp::float32> var_49;
        wp::vec_t<3,wp::float32> var_50;
        wp::float32 var_51;
        wp::float32* var_52;
        wp::float32 var_53;
        wp::float32 var_54;
        bool var_55;
        const wp::float32 var_56 = 0.5;
        wp::array_t<wp::float32>* var_57;
        wp::array_t<wp::float32> var_58;
        wp::float32* var_59;
        wp::float32 var_60;
        wp::float32 var_61;
        wp::float32 var_62;
        wp::array_t<wp::float32>* var_63;
        wp::array_t<wp::float32> var_64;
        wp::float32* var_65;
        wp::float32 var_66;
        wp::float32 var_67;
        wp::float32 var_68;
        wp::array_t<wp::float32>* var_69;
        wp::array_t<wp::float32> var_70;
        wp::float32* var_71;
        wp::float32 var_72;
        wp::float32 var_73;
        wp::float32 var_74;
        wp::array_t<wp::float32>* var_75;
        wp::array_t<wp::float32> var_76;
        wp::float32* var_77;
        wp::float32 var_78;
        wp::float32 var_79;
        wp::float32 var_80;
        wp::vec_t<3,wp::float32> var_81;
        wp::vec_t<3,wp::float32> var_82;
        wp::vec_t<3,wp::float32> var_83;
        wp::vec_t<3,wp::float32> var_84;
        wp::vec_t<3,wp::float32>* var_85;
        wp::vec_t<3,wp::float32> var_86;
        wp::vec_t<3,wp::float32> var_87;
        wp::vec_t<3,wp::float32> var_88;
        wp::vec_t<3,wp::float32> var_89;
        wp::float32 var_90;
        wp::vec_t<3,wp::float32> var_91;
        wp::vec_t<3,wp::float32> var_92;
        wp::vec_t<3,wp::float32> var_93;
        wp::vec_t<3,wp::float32> var_94;
        const wp::float32 var_95 = 0.0;
        wp::float32 var_96;
        wp::vec_t<3,wp::float32> var_97;
        wp::vec_t<3,wp::float32> var_98;
        wp::vec_t<3,wp::float32> var_99;
        wp::float32 var_100;
        wp::float32 var_101;
        wp::float32 var_102;
        wp::float32 var_103;
        wp::float32 var_104;
        wp::float32 var_105;
        wp::vec_t<3,wp::float32> var_106;
        wp::vec_t<3,wp::float32> var_107;
        wp::vec_t<3,wp::float32> var_108;
        wp::vec_t<3,wp::float32> var_109;
        wp::vec_t<3,wp::float32> var_110;
        bool var_111;
        wp::vec_t<6,wp::float32> var_112;
        wp::vec_t<6,wp::float32> var_113;
        //---------
        // forward
        // def eval_particle_contacts(                                                            <L 837>
        // tid = wp.tid()                                                                         <L 863>
        var_0 = builtin_tid1d();
        // count = min(contact_max, contact_count[0])                                             <L 865>
        var_2 = wp::address(var_contact_count, var_1);
        var_3 = wp::load(var_2);
        var_4 = wp::min(var_contact_max, var_3);
        // if tid >= count:                                                                       <L 866>
        var_5 = (var_0 >= var_4);
        if (var_5) {
            // return                                                                             <L 867>
            return;
        }
        // shape_index = contact_shape[tid]                                                       <L 869>
        var_6 = wp::address(var_contact_shape, var_0);
        var_7 = wp::load(var_6);
        var_8 = wp::copy(var_7);
        // body_index = shape_body[shape_index]                                                   <L 870>
        var_9 = wp::address(var_shape_body, var_8);
        var_10 = wp::load(var_9);
        var_11 = wp::copy(var_10);
        // particle_index = contact_particle[tid]                                                 <L 871>
        var_12 = wp::address(var_contact_particle, var_0);
        var_13 = wp::load(var_12);
        var_14 = wp::copy(var_13);
        // if (particle_flags[particle_index] & PARTICLE_FLAG_ACTIVE) == 0:                       <L 872>
        var_15 = wp::address(var_particle_flags, var_14);
        var_17 = wp::load(var_15);
        var_18 = wp::bit_and(var_17, var_16);
        var_19 = (var_18 == var_1);
        if (var_19) {
            // return                                                                             <L 873>
            return;
        }
        // px = particle_x[particle_index]                                                        <L 875>
        var_20 = wp::address(var_particle_x, var_14);
        var_21 = wp::load(var_20);
        var_22 = wp::copy(var_21);
        // pv = particle_v[particle_index]                                                        <L 876>
        var_23 = wp::address(var_particle_v, var_14);
        var_24 = wp::load(var_23);
        var_25 = wp::copy(var_24);
        // X_wb = wp.transform_identity()                                                         <L 878>
        var_26 = wp::transform_identity<wp::float32>();
        // X_com = wp.vec3()                                                                      <L 879>
        var_27 = wp::vec_t<3,wp::float32>();
        // body_v_s = wp.spatial_vector()                                                         <L 880>
        var_28 = wp::vec_t<6,wp::float32>();
        // if body_index >= 0:                                                                    <L 882>
        var_29 = (var_11 >= var_1);
        if (var_29) {
            // X_wb = body_q[body_index]                                                          <L 883>
            var_30 = wp::address(var_body_q, var_11);
            var_31 = wp::load(var_30);
            var_32 = wp::copy(var_31);
            // X_com = body_com[body_index]                                                       <L 884>
            var_33 = wp::address(var_body_com, var_11);
            var_34 = wp::load(var_33);
            var_35 = wp::copy(var_34);
            // body_v_s = body_qd[body_index]                                                     <L 885>
            var_36 = wp::address(var_body_qd, var_11);
            var_37 = wp::load(var_36);
            var_38 = wp::copy(var_37);
        }
        var_39 = wp::select(var_29, var_26, var_32);
        var_40 = wp::select(var_29, var_27, var_35);
        var_41 = wp::select(var_29, var_28, var_38);
        // bx = wp.transform_point(X_wb, contact_body_pos[tid])                                   <L 888>
        var_42 = wp::address(var_contact_body_pos, var_0);
        var_43 = wp::load(var_42);
        var_44 = wp::transform_point(var_39, var_43);
        // r = bx - wp.transform_point(X_wb, X_com)                                               <L 889>
        var_45 = wp::transform_point(var_39, var_40);
        var_46 = wp::sub(var_44, var_45);
        // n = contact_normal[tid]                                                                <L 891>
        var_47 = wp::address(var_contact_normal, var_0);
        var_48 = wp::load(var_47);
        var_49 = wp::copy(var_48);
        // c = wp.dot(n, px - bx) - particle_radius[tid]                                          <L 892>
        var_50 = wp::sub(var_22, var_44);
        var_51 = wp::dot(var_49, var_50);
        var_52 = wp::address(var_particle_radius, var_0);
        var_53 = wp::load(var_52);
        var_54 = wp::sub(var_51, var_53);
        // if c > particle_ka:                                                                    <L 894>
        var_55 = (var_54 > var_particle_ka);
        if (var_55) {
            // return                                                                             <L 895>
            return;
        }
        // ke = 0.5 * (particle_ke + shape_materials.ke[shape_index])                             <L 898>
        var_57 = &(var_shape_materials.ke);
        var_58 = wp::load(var_57);
        var_59 = wp::address(var_58, var_8);
        var_60 = wp::load(var_59);
        var_61 = wp::add(var_particle_ke, var_60);
        var_62 = wp::mul(var_56, var_61);
        // kd = 0.5 * (particle_kd + shape_materials.kd[shape_index])                             <L 899>
        var_63 = &(var_shape_materials.kd);
        var_64 = wp::load(var_63);
        var_65 = wp::address(var_64, var_8);
        var_66 = wp::load(var_65);
        var_67 = wp::add(var_particle_kd, var_66);
        var_68 = wp::mul(var_56, var_67);
        // kf = 0.5 * (particle_kf + shape_materials.kf[shape_index])                             <L 900>
        var_69 = &(var_shape_materials.kf);
        var_70 = wp::load(var_69);
        var_71 = wp::address(var_70, var_8);
        var_72 = wp::load(var_71);
        var_73 = wp::add(var_particle_kf, var_72);
        var_74 = wp::mul(var_56, var_73);
        // mu = 0.5 * (particle_mu + shape_materials.mu[shape_index])                             <L 901>
        var_75 = &(var_shape_materials.mu);
        var_76 = wp::load(var_75);
        var_77 = wp::address(var_76, var_8);
        var_78 = wp::load(var_77);
        var_79 = wp::add(var_particle_mu, var_78);
        var_80 = wp::mul(var_56, var_79);
        // body_w = wp.spatial_top(body_v_s)                                                      <L 903>
        var_81 = wp::spatial_top(var_41);
        // body_v = wp.spatial_bottom(body_v_s)                                                   <L 904>
        var_82 = wp::spatial_bottom(var_41);
        // bv = body_v + wp.cross(body_w, r) + wp.transform_vector(X_wb, contact_body_vel[tid])       <L 907>
        var_83 = wp::cross(var_81, var_46);
        var_84 = wp::add(var_82, var_83);
        var_85 = wp::address(var_contact_body_vel, var_0);
        var_86 = wp::load(var_85);
        var_87 = wp::transform_vector(var_39, var_86);
        var_88 = wp::add(var_84, var_87);
        // v = pv - bv                                                                            <L 910>
        var_89 = wp::sub(var_25, var_88);
        // vn = wp.dot(n, v)                                                                      <L 913>
        var_90 = wp::dot(var_49, var_89);
        // vt = v - n * vn                                                                        <L 914>
        var_91 = wp::mul(var_49, var_90);
        var_92 = wp::sub(var_89, var_91);
        // fn = n * c * ke                                                                        <L 917>
        var_93 = wp::mul(var_49, var_54);
        var_94 = wp::mul(var_93, var_62);
        // fd = n * wp.min(vn, 0.0) * kd                                                          <L 920>
        var_96 = wp::min(var_90, var_95);
        var_97 = wp::mul(var_49, var_96);
        var_98 = wp::mul(var_97, var_68);
        // ft = wp.normalize(vt) * wp.min(kf * wp.length(vt), abs(mu * c * ke))                   <L 935>
        var_99 = wp::normalize(var_92);
        var_100 = wp::length(var_92);
        var_101 = wp::mul(var_74, var_100);
        var_102 = wp::mul(var_80, var_54);
        var_103 = wp::mul(var_102, var_62);
        var_104 = wp::abs(var_103);
        var_105 = wp::min(var_101, var_104);
        var_106 = wp::mul(var_99, var_105);
        // f_total = fn + (fd + ft)                                                               <L 937>
        var_107 = wp::add(var_98, var_106);
        var_108 = wp::add(var_94, var_107);
        // t_total = wp.cross(r, f_total)                                                         <L 938>
        var_109 = wp::cross(var_46, var_108);
        // wp.atomic_sub(particle_f, particle_index, f_total)                                     <L 940>
        var_110 = wp::atomic_sub(var_particle_f, var_14, var_108);
        // if body_index >= 0:                                                                    <L 942>
        var_111 = (var_11 >= var_1);
        if (var_111) {
            // wp.atomic_add(body_f, body_index, wp.spatial_vector(t_total, f_total))             <L 943>
            var_112 = wp::vec_t<6,wp::float32>(var_109, var_108);
            var_113 = wp::atomic_add(var_body_f, var_11, var_112);
        }
    }
}

extern "C" __global__ void eval_particle_contacts_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_x,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_v,
    wp::array_t<wp::float32> var_particle_radius,
    wp::array_t<wp::uint32> var_particle_flags,
    wp::array_t<wp::transform_t<wp::float32>> var_body_q,
    wp::array_t<wp::vec_t<6,wp::float32>> var_body_qd,
    wp::array_t<wp::vec_t<3,wp::float32>> var_body_com,
    wp::array_t<wp::int32> var_shape_body,
    ModelShapeMaterials var_shape_materials,
    wp::float32 var_particle_ke,
    wp::float32 var_particle_kd,
    wp::float32 var_particle_kf,
    wp::float32 var_particle_mu,
    wp::float32 var_particle_ka,
    wp::array_t<wp::int32> var_contact_count,
    wp::array_t<wp::int32> var_contact_particle,
    wp::array_t<wp::int32> var_contact_shape,
    wp::array_t<wp::vec_t<3,wp::float32>> var_contact_body_pos,
    wp::array_t<wp::vec_t<3,wp::float32>> var_contact_body_vel,
    wp::array_t<wp::vec_t<3,wp::float32>> var_contact_normal,
    wp::int32 var_contact_max,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_f,
    wp::array_t<wp::vec_t<6,wp::float32>> var_body_f,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_particle_x,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_particle_v,
    wp::array_t<wp::float32> adj_particle_radius,
    wp::array_t<wp::uint32> adj_particle_flags,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_q,
    wp::array_t<wp::vec_t<6,wp::float32>> adj_body_qd,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_body_com,
    wp::array_t<wp::int32> adj_shape_body,
    ModelShapeMaterials adj_shape_materials,
    wp::float32 adj_particle_ke,
    wp::float32 adj_particle_kd,
    wp::float32 adj_particle_kf,
    wp::float32 adj_particle_mu,
    wp::float32 adj_particle_ka,
    wp::array_t<wp::int32> adj_contact_count,
    wp::array_t<wp::int32> adj_contact_particle,
    wp::array_t<wp::int32> adj_contact_shape,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_contact_body_pos,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_contact_body_vel,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_contact_normal,
    wp::int32 adj_contact_max,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_particle_f,
    wp::array_t<wp::vec_t<6,wp::float32>> adj_body_f)
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
        wp::vec_t<6,wp::float32> var_28;
        bool var_29;
        wp::transform_t<wp::float32>* var_30;
        wp::transform_t<wp::float32> var_31;
        wp::transform_t<wp::float32> var_32;
        wp::vec_t<3,wp::float32>* var_33;
        wp::vec_t<3,wp::float32> var_34;
        wp::vec_t<3,wp::float32> var_35;
        wp::vec_t<6,wp::float32>* var_36;
        wp::vec_t<6,wp::float32> var_37;
        wp::vec_t<6,wp::float32> var_38;
        wp::transform_t<wp::float32> var_39;
        wp::vec_t<3,wp::float32> var_40;
        wp::vec_t<6,wp::float32> var_41;
        wp::vec_t<3,wp::float32>* var_42;
        wp::vec_t<3,wp::float32> var_43;
        wp::vec_t<3,wp::float32> var_44;
        wp::vec_t<3,wp::float32> var_45;
        wp::vec_t<3,wp::float32> var_46;
        wp::vec_t<3,wp::float32>* var_47;
        wp::vec_t<3,wp::float32> var_48;
        wp::vec_t<3,wp::float32> var_49;
        wp::vec_t<3,wp::float32> var_50;
        wp::float32 var_51;
        wp::float32* var_52;
        wp::float32 var_53;
        wp::float32 var_54;
        bool var_55;
        const wp::float32 var_56 = 0.5;
        wp::array_t<wp::float32>* var_57;
        wp::array_t<wp::float32> var_58;
        wp::float32* var_59;
        wp::float32 var_60;
        wp::float32 var_61;
        wp::float32 var_62;
        wp::array_t<wp::float32>* var_63;
        wp::array_t<wp::float32> var_64;
        wp::float32* var_65;
        wp::float32 var_66;
        wp::float32 var_67;
        wp::float32 var_68;
        wp::array_t<wp::float32>* var_69;
        wp::array_t<wp::float32> var_70;
        wp::float32* var_71;
        wp::float32 var_72;
        wp::float32 var_73;
        wp::float32 var_74;
        wp::array_t<wp::float32>* var_75;
        wp::array_t<wp::float32> var_76;
        wp::float32* var_77;
        wp::float32 var_78;
        wp::float32 var_79;
        wp::float32 var_80;
        wp::vec_t<3,wp::float32> var_81;
        wp::vec_t<3,wp::float32> var_82;
        wp::vec_t<3,wp::float32> var_83;
        wp::vec_t<3,wp::float32> var_84;
        wp::vec_t<3,wp::float32>* var_85;
        wp::vec_t<3,wp::float32> var_86;
        wp::vec_t<3,wp::float32> var_87;
        wp::vec_t<3,wp::float32> var_88;
        wp::vec_t<3,wp::float32> var_89;
        wp::float32 var_90;
        wp::vec_t<3,wp::float32> var_91;
        wp::vec_t<3,wp::float32> var_92;
        wp::vec_t<3,wp::float32> var_93;
        wp::vec_t<3,wp::float32> var_94;
        const wp::float32 var_95 = 0.0;
        wp::float32 var_96;
        wp::vec_t<3,wp::float32> var_97;
        wp::vec_t<3,wp::float32> var_98;
        wp::vec_t<3,wp::float32> var_99;
        wp::float32 var_100;
        wp::float32 var_101;
        wp::float32 var_102;
        wp::float32 var_103;
        wp::float32 var_104;
        wp::float32 var_105;
        wp::vec_t<3,wp::float32> var_106;
        wp::vec_t<3,wp::float32> var_107;
        wp::vec_t<3,wp::float32> var_108;
        wp::vec_t<3,wp::float32> var_109;
        wp::vec_t<3,wp::float32> var_110;
        bool var_111;
        wp::vec_t<6,wp::float32> var_112;
        wp::vec_t<6,wp::float32> var_113;
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
        wp::vec_t<6,wp::float32> adj_28 = {};
        bool adj_29 = {};
        wp::transform_t<wp::float32> adj_30 = {};
        wp::transform_t<wp::float32> adj_31 = {};
        wp::transform_t<wp::float32> adj_32 = {};
        wp::vec_t<3,wp::float32> adj_33 = {};
        wp::vec_t<3,wp::float32> adj_34 = {};
        wp::vec_t<3,wp::float32> adj_35 = {};
        wp::vec_t<6,wp::float32> adj_36 = {};
        wp::vec_t<6,wp::float32> adj_37 = {};
        wp::vec_t<6,wp::float32> adj_38 = {};
        wp::transform_t<wp::float32> adj_39 = {};
        wp::vec_t<3,wp::float32> adj_40 = {};
        wp::vec_t<6,wp::float32> adj_41 = {};
        wp::vec_t<3,wp::float32> adj_42 = {};
        wp::vec_t<3,wp::float32> adj_43 = {};
        wp::vec_t<3,wp::float32> adj_44 = {};
        wp::vec_t<3,wp::float32> adj_45 = {};
        wp::vec_t<3,wp::float32> adj_46 = {};
        wp::vec_t<3,wp::float32> adj_47 = {};
        wp::vec_t<3,wp::float32> adj_48 = {};
        wp::vec_t<3,wp::float32> adj_49 = {};
        wp::vec_t<3,wp::float32> adj_50 = {};
        wp::float32 adj_51 = {};
        wp::float32 adj_52 = {};
        wp::float32 adj_53 = {};
        wp::float32 adj_54 = {};
        bool adj_55 = {};
        wp::float32 adj_56 = {};
        wp::array_t<wp::float32> adj_57 = {};
        wp::array_t<wp::float32> adj_58 = {};
        wp::float32 adj_59 = {};
        wp::float32 adj_60 = {};
        wp::float32 adj_61 = {};
        wp::float32 adj_62 = {};
        wp::array_t<wp::float32> adj_63 = {};
        wp::array_t<wp::float32> adj_64 = {};
        wp::float32 adj_65 = {};
        wp::float32 adj_66 = {};
        wp::float32 adj_67 = {};
        wp::float32 adj_68 = {};
        wp::array_t<wp::float32> adj_69 = {};
        wp::array_t<wp::float32> adj_70 = {};
        wp::float32 adj_71 = {};
        wp::float32 adj_72 = {};
        wp::float32 adj_73 = {};
        wp::float32 adj_74 = {};
        wp::array_t<wp::float32> adj_75 = {};
        wp::array_t<wp::float32> adj_76 = {};
        wp::float32 adj_77 = {};
        wp::float32 adj_78 = {};
        wp::float32 adj_79 = {};
        wp::float32 adj_80 = {};
        wp::vec_t<3,wp::float32> adj_81 = {};
        wp::vec_t<3,wp::float32> adj_82 = {};
        wp::vec_t<3,wp::float32> adj_83 = {};
        wp::vec_t<3,wp::float32> adj_84 = {};
        wp::vec_t<3,wp::float32> adj_85 = {};
        wp::vec_t<3,wp::float32> adj_86 = {};
        wp::vec_t<3,wp::float32> adj_87 = {};
        wp::vec_t<3,wp::float32> adj_88 = {};
        wp::vec_t<3,wp::float32> adj_89 = {};
        wp::float32 adj_90 = {};
        wp::vec_t<3,wp::float32> adj_91 = {};
        wp::vec_t<3,wp::float32> adj_92 = {};
        wp::vec_t<3,wp::float32> adj_93 = {};
        wp::vec_t<3,wp::float32> adj_94 = {};
        wp::float32 adj_95 = {};
        wp::float32 adj_96 = {};
        wp::vec_t<3,wp::float32> adj_97 = {};
        wp::vec_t<3,wp::float32> adj_98 = {};
        wp::vec_t<3,wp::float32> adj_99 = {};
        wp::float32 adj_100 = {};
        wp::float32 adj_101 = {};
        wp::float32 adj_102 = {};
        wp::float32 adj_103 = {};
        wp::float32 adj_104 = {};
        wp::float32 adj_105 = {};
        wp::vec_t<3,wp::float32> adj_106 = {};
        wp::vec_t<3,wp::float32> adj_107 = {};
        wp::vec_t<3,wp::float32> adj_108 = {};
        wp::vec_t<3,wp::float32> adj_109 = {};
        wp::vec_t<3,wp::float32> adj_110 = {};
        bool adj_111 = {};
        wp::vec_t<6,wp::float32> adj_112 = {};
        wp::vec_t<6,wp::float32> adj_113 = {};
        //---------
        // forward
        // def eval_particle_contacts(                                                            <L 837>
        // tid = wp.tid()                                                                         <L 863>
        var_0 = builtin_tid1d();
        // count = min(contact_max, contact_count[0])                                             <L 865>
        var_2 = wp::address(var_contact_count, var_1);
        var_3 = wp::load(var_2);
        var_4 = wp::min(var_contact_max, var_3);
        // if tid >= count:                                                                       <L 866>
        var_5 = (var_0 >= var_4);
        if (var_5) {
            // return                                                                             <L 867>
            goto label0;
        }
        // shape_index = contact_shape[tid]                                                       <L 869>
        var_6 = wp::address(var_contact_shape, var_0);
        var_7 = wp::load(var_6);
        var_8 = wp::copy(var_7);
        // body_index = shape_body[shape_index]                                                   <L 870>
        var_9 = wp::address(var_shape_body, var_8);
        var_10 = wp::load(var_9);
        var_11 = wp::copy(var_10);
        // particle_index = contact_particle[tid]                                                 <L 871>
        var_12 = wp::address(var_contact_particle, var_0);
        var_13 = wp::load(var_12);
        var_14 = wp::copy(var_13);
        // if (particle_flags[particle_index] & PARTICLE_FLAG_ACTIVE) == 0:                       <L 872>
        var_15 = wp::address(var_particle_flags, var_14);
        var_17 = wp::load(var_15);
        var_18 = wp::bit_and(var_17, var_16);
        var_19 = (var_18 == var_1);
        if (var_19) {
            // return                                                                             <L 873>
            goto label1;
        }
        // px = particle_x[particle_index]                                                        <L 875>
        var_20 = wp::address(var_particle_x, var_14);
        var_21 = wp::load(var_20);
        var_22 = wp::copy(var_21);
        // pv = particle_v[particle_index]                                                        <L 876>
        var_23 = wp::address(var_particle_v, var_14);
        var_24 = wp::load(var_23);
        var_25 = wp::copy(var_24);
        // X_wb = wp.transform_identity()                                                         <L 878>
        var_26 = wp::transform_identity<wp::float32>();
        // X_com = wp.vec3()                                                                      <L 879>
        var_27 = wp::vec_t<3,wp::float32>();
        // body_v_s = wp.spatial_vector()                                                         <L 880>
        var_28 = wp::vec_t<6,wp::float32>();
        // if body_index >= 0:                                                                    <L 882>
        var_29 = (var_11 >= var_1);
        if (var_29) {
            // X_wb = body_q[body_index]                                                          <L 883>
            var_30 = wp::address(var_body_q, var_11);
            var_31 = wp::load(var_30);
            var_32 = wp::copy(var_31);
            // X_com = body_com[body_index]                                                       <L 884>
            var_33 = wp::address(var_body_com, var_11);
            var_34 = wp::load(var_33);
            var_35 = wp::copy(var_34);
            // body_v_s = body_qd[body_index]                                                     <L 885>
            var_36 = wp::address(var_body_qd, var_11);
            var_37 = wp::load(var_36);
            var_38 = wp::copy(var_37);
        }
        var_39 = wp::select(var_29, var_26, var_32);
        var_40 = wp::select(var_29, var_27, var_35);
        var_41 = wp::select(var_29, var_28, var_38);
        // bx = wp.transform_point(X_wb, contact_body_pos[tid])                                   <L 888>
        var_42 = wp::address(var_contact_body_pos, var_0);
        var_43 = wp::load(var_42);
        var_44 = wp::transform_point(var_39, var_43);
        // r = bx - wp.transform_point(X_wb, X_com)                                               <L 889>
        var_45 = wp::transform_point(var_39, var_40);
        var_46 = wp::sub(var_44, var_45);
        // n = contact_normal[tid]                                                                <L 891>
        var_47 = wp::address(var_contact_normal, var_0);
        var_48 = wp::load(var_47);
        var_49 = wp::copy(var_48);
        // c = wp.dot(n, px - bx) - particle_radius[tid]                                          <L 892>
        var_50 = wp::sub(var_22, var_44);
        var_51 = wp::dot(var_49, var_50);
        var_52 = wp::address(var_particle_radius, var_0);
        var_53 = wp::load(var_52);
        var_54 = wp::sub(var_51, var_53);
        // if c > particle_ka:                                                                    <L 894>
        var_55 = (var_54 > var_particle_ka);
        if (var_55) {
            // return                                                                             <L 895>
            goto label2;
        }
        // ke = 0.5 * (particle_ke + shape_materials.ke[shape_index])                             <L 898>
        var_57 = &(var_shape_materials.ke);
        var_58 = wp::load(var_57);
        var_59 = wp::address(var_58, var_8);
        var_60 = wp::load(var_59);
        var_61 = wp::add(var_particle_ke, var_60);
        var_62 = wp::mul(var_56, var_61);
        // kd = 0.5 * (particle_kd + shape_materials.kd[shape_index])                             <L 899>
        var_63 = &(var_shape_materials.kd);
        var_64 = wp::load(var_63);
        var_65 = wp::address(var_64, var_8);
        var_66 = wp::load(var_65);
        var_67 = wp::add(var_particle_kd, var_66);
        var_68 = wp::mul(var_56, var_67);
        // kf = 0.5 * (particle_kf + shape_materials.kf[shape_index])                             <L 900>
        var_69 = &(var_shape_materials.kf);
        var_70 = wp::load(var_69);
        var_71 = wp::address(var_70, var_8);
        var_72 = wp::load(var_71);
        var_73 = wp::add(var_particle_kf, var_72);
        var_74 = wp::mul(var_56, var_73);
        // mu = 0.5 * (particle_mu + shape_materials.mu[shape_index])                             <L 901>
        var_75 = &(var_shape_materials.mu);
        var_76 = wp::load(var_75);
        var_77 = wp::address(var_76, var_8);
        var_78 = wp::load(var_77);
        var_79 = wp::add(var_particle_mu, var_78);
        var_80 = wp::mul(var_56, var_79);
        // body_w = wp.spatial_top(body_v_s)                                                      <L 903>
        var_81 = wp::spatial_top(var_41);
        // body_v = wp.spatial_bottom(body_v_s)                                                   <L 904>
        var_82 = wp::spatial_bottom(var_41);
        // bv = body_v + wp.cross(body_w, r) + wp.transform_vector(X_wb, contact_body_vel[tid])       <L 907>
        var_83 = wp::cross(var_81, var_46);
        var_84 = wp::add(var_82, var_83);
        var_85 = wp::address(var_contact_body_vel, var_0);
        var_86 = wp::load(var_85);
        var_87 = wp::transform_vector(var_39, var_86);
        var_88 = wp::add(var_84, var_87);
        // v = pv - bv                                                                            <L 910>
        var_89 = wp::sub(var_25, var_88);
        // vn = wp.dot(n, v)                                                                      <L 913>
        var_90 = wp::dot(var_49, var_89);
        // vt = v - n * vn                                                                        <L 914>
        var_91 = wp::mul(var_49, var_90);
        var_92 = wp::sub(var_89, var_91);
        // fn = n * c * ke                                                                        <L 917>
        var_93 = wp::mul(var_49, var_54);
        var_94 = wp::mul(var_93, var_62);
        // fd = n * wp.min(vn, 0.0) * kd                                                          <L 920>
        var_96 = wp::min(var_90, var_95);
        var_97 = wp::mul(var_49, var_96);
        var_98 = wp::mul(var_97, var_68);
        // ft = wp.normalize(vt) * wp.min(kf * wp.length(vt), abs(mu * c * ke))                   <L 935>
        var_99 = wp::normalize(var_92);
        var_100 = wp::length(var_92);
        var_101 = wp::mul(var_74, var_100);
        var_102 = wp::mul(var_80, var_54);
        var_103 = wp::mul(var_102, var_62);
        var_104 = wp::abs(var_103);
        var_105 = wp::min(var_101, var_104);
        var_106 = wp::mul(var_99, var_105);
        // f_total = fn + (fd + ft)                                                               <L 937>
        var_107 = wp::add(var_98, var_106);
        var_108 = wp::add(var_94, var_107);
        // t_total = wp.cross(r, f_total)                                                         <L 938>
        var_109 = wp::cross(var_46, var_108);
        // wp.atomic_sub(particle_f, particle_index, f_total)                                     <L 940>
        // var_110 = wp::atomic_sub(var_particle_f, var_14, var_108);
        // if body_index >= 0:                                                                    <L 942>
        var_111 = (var_11 >= var_1);
        if (var_111) {
            // wp.atomic_add(body_f, body_index, wp.spatial_vector(t_total, f_total))             <L 943>
            var_112 = wp::vec_t<6,wp::float32>(var_109, var_108);
            // var_113 = wp::atomic_add(var_body_f, var_11, var_112);
        }
        //---------
        // reverse
        if (var_111) {
            wp::adj_atomic_add(var_body_f, var_11, var_112, adj_body_f, adj_11, adj_112, adj_113);
            wp::adj_vec_t(var_109, var_108, adj_109, adj_108, adj_112);
            // adj: wp.atomic_add(body_f, body_index, wp.spatial_vector(t_total, f_total))        <L 943>
        }
        // adj: if body_index >= 0:                                                               <L 942>
        wp::adj_atomic_sub(var_particle_f, var_14, var_108, adj_particle_f, adj_14, adj_108, adj_110);
        // adj: wp.atomic_sub(particle_f, particle_index, f_total)                                <L 940>
        wp::adj_cross(var_46, var_108, adj_46, adj_108, adj_109);
        // adj: t_total = wp.cross(r, f_total)                                                    <L 938>
        wp::adj_add(var_94, var_107, adj_94, adj_107, adj_108);
        wp::adj_add(var_98, var_106, adj_98, adj_106, adj_107);
        // adj: f_total = fn + (fd + ft)                                                          <L 937>
        wp::adj_mul(var_99, var_105, adj_99, adj_105, adj_106);
        wp::adj_min(var_101, var_104, adj_101, adj_104, adj_105);
        wp::adj_abs(var_103, adj_103, adj_104);
        wp::adj_mul(var_102, var_62, adj_102, adj_62, adj_103);
        wp::adj_mul(var_80, var_54, adj_80, adj_54, adj_102);
        wp::adj_mul(var_74, var_100, adj_74, adj_100, adj_101);
        wp::adj_length(var_92, var_100, adj_92, adj_100);
        wp::adj_normalize(var_92, var_99, adj_92, adj_99);
        // adj: ft = wp.normalize(vt) * wp.min(kf * wp.length(vt), abs(mu * c * ke))              <L 935>
        wp::adj_mul(var_97, var_68, adj_97, adj_68, adj_98);
        wp::adj_mul(var_49, var_96, adj_49, adj_96, adj_97);
        wp::adj_min(var_90, var_95, adj_90, adj_95, adj_96);
        // adj: fd = n * wp.min(vn, 0.0) * kd                                                     <L 920>
        wp::adj_mul(var_93, var_62, adj_93, adj_62, adj_94);
        wp::adj_mul(var_49, var_54, adj_49, adj_54, adj_93);
        // adj: fn = n * c * ke                                                                   <L 917>
        wp::adj_sub(var_89, var_91, adj_89, adj_91, adj_92);
        wp::adj_mul(var_49, var_90, adj_49, adj_90, adj_91);
        // adj: vt = v - n * vn                                                                   <L 914>
        wp::adj_dot(var_49, var_89, adj_49, adj_89, adj_90);
        // adj: vn = wp.dot(n, v)                                                                 <L 913>
        wp::adj_sub(var_25, var_88, adj_25, adj_88, adj_89);
        // adj: v = pv - bv                                                                       <L 910>
        wp::adj_add(var_84, var_87, adj_84, adj_87, adj_88);
        wp::adj_transform_vector(var_39, var_86, adj_39, adj_85, adj_87);
        wp::adj_load(var_85, adj_85, adj_86);
        wp::adj_address(var_contact_body_vel, var_0, adj_contact_body_vel, adj_0, adj_85);
        wp::adj_add(var_82, var_83, adj_82, adj_83, adj_84);
        wp::adj_cross(var_81, var_46, adj_81, adj_46, adj_83);
        // adj: bv = body_v + wp.cross(body_w, r) + wp.transform_vector(X_wb, contact_body_vel[tid])  <L 907>
        wp::adj_spatial_bottom(var_41, adj_41, adj_82);
        // adj: body_v = wp.spatial_bottom(body_v_s)                                              <L 904>
        wp::adj_spatial_top(var_41, adj_41, adj_81);
        // adj: body_w = wp.spatial_top(body_v_s)                                                 <L 903>
        wp::adj_mul(var_56, var_79, adj_56, adj_79, adj_80);
        wp::adj_add(var_particle_mu, var_78, adj_particle_mu, adj_77, adj_79);
        wp::adj_load(var_77, adj_77, adj_78);
        wp::adj_address(var_76, var_8, adj_75, adj_8, adj_77);
        wp::adj_load(var_75, adj_75, adj_76);
        adj_shape_materials.mu = adj_75;
        // adj: mu = 0.5 * (particle_mu + shape_materials.mu[shape_index])                        <L 901>
        wp::adj_mul(var_56, var_73, adj_56, adj_73, adj_74);
        wp::adj_add(var_particle_kf, var_72, adj_particle_kf, adj_71, adj_73);
        wp::adj_load(var_71, adj_71, adj_72);
        wp::adj_address(var_70, var_8, adj_69, adj_8, adj_71);
        wp::adj_load(var_69, adj_69, adj_70);
        adj_shape_materials.kf = adj_69;
        // adj: kf = 0.5 * (particle_kf + shape_materials.kf[shape_index])                        <L 900>
        wp::adj_mul(var_56, var_67, adj_56, adj_67, adj_68);
        wp::adj_add(var_particle_kd, var_66, adj_particle_kd, adj_65, adj_67);
        wp::adj_load(var_65, adj_65, adj_66);
        wp::adj_address(var_64, var_8, adj_63, adj_8, adj_65);
        wp::adj_load(var_63, adj_63, adj_64);
        adj_shape_materials.kd = adj_63;
        // adj: kd = 0.5 * (particle_kd + shape_materials.kd[shape_index])                        <L 899>
        wp::adj_mul(var_56, var_61, adj_56, adj_61, adj_62);
        wp::adj_add(var_particle_ke, var_60, adj_particle_ke, adj_59, adj_61);
        wp::adj_load(var_59, adj_59, adj_60);
        wp::adj_address(var_58, var_8, adj_57, adj_8, adj_59);
        wp::adj_load(var_57, adj_57, adj_58);
        adj_shape_materials.ke = adj_57;
        // adj: ke = 0.5 * (particle_ke + shape_materials.ke[shape_index])                        <L 898>
        if (var_55) {
            label2:;
            // adj: return                                                                        <L 895>
        }
        // adj: if c > particle_ka:                                                               <L 894>
        wp::adj_sub(var_51, var_53, adj_51, adj_52, adj_54);
        wp::adj_load(var_52, adj_52, adj_53);
        wp::adj_address(var_particle_radius, var_0, adj_particle_radius, adj_0, adj_52);
        wp::adj_dot(var_49, var_50, adj_49, adj_50, adj_51);
        wp::adj_sub(var_22, var_44, adj_22, adj_44, adj_50);
        // adj: c = wp.dot(n, px - bx) - particle_radius[tid]                                     <L 892>
        wp::adj_copy(var_48, adj_47, adj_49);
        wp::adj_load(var_47, adj_47, adj_48);
        wp::adj_address(var_contact_normal, var_0, adj_contact_normal, adj_0, adj_47);
        // adj: n = contact_normal[tid]                                                           <L 891>
        wp::adj_sub(var_44, var_45, adj_44, adj_45, adj_46);
        wp::adj_transform_point(var_39, var_40, adj_39, adj_40, adj_45);
        // adj: r = bx - wp.transform_point(X_wb, X_com)                                          <L 889>
        wp::adj_transform_point(var_39, var_43, adj_39, adj_42, adj_44);
        wp::adj_load(var_42, adj_42, adj_43);
        wp::adj_address(var_contact_body_pos, var_0, adj_contact_body_pos, adj_0, adj_42);
        // adj: bx = wp.transform_point(X_wb, contact_body_pos[tid])                              <L 888>
        wp::adj_select(var_29, var_28, var_38, adj_29, adj_28, adj_38, adj_41);
        wp::adj_select(var_29, var_27, var_35, adj_29, adj_27, adj_35, adj_40);
        wp::adj_select(var_29, var_26, var_32, adj_29, adj_26, adj_32, adj_39);
        if (var_29) {
            wp::adj_copy(var_37, adj_36, adj_38);
            wp::adj_load(var_36, adj_36, adj_37);
            wp::adj_address(var_body_qd, var_11, adj_body_qd, adj_11, adj_36);
            // adj: body_v_s = body_qd[body_index]                                                <L 885>
            wp::adj_copy(var_34, adj_33, adj_35);
            wp::adj_load(var_33, adj_33, adj_34);
            wp::adj_address(var_body_com, var_11, adj_body_com, adj_11, adj_33);
            // adj: X_com = body_com[body_index]                                                  <L 884>
            wp::adj_copy(var_31, adj_30, adj_32);
            wp::adj_load(var_30, adj_30, adj_31);
            wp::adj_address(var_body_q, var_11, adj_body_q, adj_11, adj_30);
            // adj: X_wb = body_q[body_index]                                                     <L 883>
        }
        // adj: if body_index >= 0:                                                               <L 882>
        // adj: body_v_s = wp.spatial_vector()                                                    <L 880>
        // adj: X_com = wp.vec3()                                                                 <L 879>
        // adj: X_wb = wp.transform_identity()                                                    <L 878>
        wp::adj_copy(var_24, adj_23, adj_25);
        wp::adj_load(var_23, adj_23, adj_24);
        wp::adj_address(var_particle_v, var_14, adj_particle_v, adj_14, adj_23);
        // adj: pv = particle_v[particle_index]                                                   <L 876>
        wp::adj_copy(var_21, adj_20, adj_22);
        wp::adj_load(var_20, adj_20, adj_21);
        wp::adj_address(var_particle_x, var_14, adj_particle_x, adj_14, adj_20);
        // adj: px = particle_x[particle_index]                                                   <L 875>
        if (var_19) {
            label1:;
            // adj: return                                                                        <L 873>
        }
        wp::adj_bit_and(var_17, var_16, adj_15, adj_16, adj_18);
        wp::adj_load(var_15, adj_15, adj_17);
        wp::adj_address(var_particle_flags, var_14, adj_particle_flags, adj_14, adj_15);
        // adj: if (particle_flags[particle_index] & PARTICLE_FLAG_ACTIVE) == 0:                  <L 872>
        wp::adj_copy(var_13, adj_12, adj_14);
        wp::adj_load(var_12, adj_12, adj_13);
        wp::adj_address(var_contact_particle, var_0, adj_contact_particle, adj_0, adj_12);
        // adj: particle_index = contact_particle[tid]                                            <L 871>
        wp::adj_copy(var_10, adj_9, adj_11);
        wp::adj_load(var_9, adj_9, adj_10);
        wp::adj_address(var_shape_body, var_8, adj_shape_body, adj_8, adj_9);
        // adj: body_index = shape_body[shape_index]                                              <L 870>
        wp::adj_copy(var_7, adj_6, adj_8);
        wp::adj_load(var_6, adj_6, adj_7);
        wp::adj_address(var_contact_shape, var_0, adj_contact_shape, adj_0, adj_6);
        // adj: shape_index = contact_shape[tid]                                                  <L 869>
        if (var_5) {
            label0:;
            // adj: return                                                                        <L 867>
        }
        // adj: if tid >= count:                                                                  <L 866>
        wp::adj_min(var_contact_max, var_3, adj_contact_max, adj_2, adj_4);
        wp::adj_load(var_2, adj_2, adj_3);
        wp::adj_address(var_contact_count, var_1, adj_contact_count, adj_1, adj_2);
        // adj: count = min(contact_max, contact_count[0])                                        <L 865>
        // adj: tid = wp.tid()                                                                    <L 863>
        // adj: def eval_particle_contacts(                                                       <L 837>
        continue;
    }
}



extern "C" __global__ void eval_rigid_contacts_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::transform_t<wp::float32>> var_body_q,
    wp::array_t<wp::vec_t<6,wp::float32>> var_body_qd,
    wp::array_t<wp::vec_t<3,wp::float32>> var_body_com,
    ModelShapeMaterials var_shape_materials,
    ModelShapeGeometry var_geo,
    wp::array_t<wp::int32> var_contact_count,
    wp::array_t<wp::int32> var_contact_body0,
    wp::array_t<wp::int32> var_contact_body1,
    wp::array_t<wp::vec_t<3,wp::float32>> var_contact_point0,
    wp::array_t<wp::vec_t<3,wp::float32>> var_contact_point1,
    wp::array_t<wp::vec_t<3,wp::float32>> var_contact_normal,
    wp::array_t<wp::int32> var_contact_shape0,
    wp::array_t<wp::int32> var_contact_shape1,
    wp::array_t<wp::vec_t<6,wp::float32>> var_body_f)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32* var_1;
        wp::int32* var_2;
        bool var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        const wp::int32 var_6 = 0;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        bool var_10;
        const wp::float32 var_11 = 0.0;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        bool var_18;
        const wp::int32 var_19 = 1;
        wp::int32 var_20;
        wp::array_t<wp::float32>* var_21;
        wp::array_t<wp::float32> var_22;
        wp::float32* var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        wp::array_t<wp::float32>* var_26;
        wp::array_t<wp::float32> var_27;
        wp::float32* var_28;
        wp::float32 var_29;
        wp::float32 var_30;
        wp::array_t<wp::float32>* var_31;
        wp::array_t<wp::float32> var_32;
        wp::float32* var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        wp::array_t<wp::float32>* var_36;
        wp::array_t<wp::float32> var_37;
        wp::float32* var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        wp::array_t<wp::float32>* var_41;
        wp::array_t<wp::float32> var_42;
        wp::float32* var_43;
        wp::float32 var_44;
        wp::float32 var_45;
        wp::float32 var_46;
        wp::float32 var_47;
        wp::float32 var_48;
        wp::float32 var_49;
        wp::int32 var_50;
        wp::float32 var_51;
        bool var_52;
        wp::int32 var_53;
        wp::array_t<wp::float32>* var_54;
        wp::array_t<wp::float32> var_55;
        wp::float32* var_56;
        wp::float32 var_57;
        wp::float32 var_58;
        wp::array_t<wp::float32>* var_59;
        wp::array_t<wp::float32> var_60;
        wp::float32* var_61;
        wp::float32 var_62;
        wp::float32 var_63;
        wp::array_t<wp::float32>* var_64;
        wp::array_t<wp::float32> var_65;
        wp::float32* var_66;
        wp::float32 var_67;
        wp::float32 var_68;
        wp::array_t<wp::float32>* var_69;
        wp::array_t<wp::float32> var_70;
        wp::float32* var_71;
        wp::float32 var_72;
        wp::float32 var_73;
        wp::array_t<wp::float32>* var_74;
        wp::array_t<wp::float32> var_75;
        wp::float32* var_76;
        wp::float32 var_77;
        wp::float32 var_78;
        wp::float32 var_79;
        wp::float32 var_80;
        wp::float32 var_81;
        wp::float32 var_82;
        wp::int32 var_83;
        wp::float32 var_84;
        bool var_85;
        wp::float32 var_86;
        wp::float32 var_87;
        wp::float32 var_88;
        wp::float32 var_89;
        wp::float32 var_90;
        wp::float32 var_91;
        wp::float32 var_92;
        wp::float32 var_93;
        wp::float32 var_94;
        wp::float32 var_95;
        wp::float32 var_96;
        wp::float32 var_97;
        wp::int32* var_98;
        wp::int32 var_99;
        wp::int32 var_100;
        wp::int32* var_101;
        wp::int32 var_102;
        wp::int32 var_103;
        wp::vec_t<3,wp::float32>* var_104;
        wp::vec_t<3,wp::float32> var_105;
        wp::vec_t<3,wp::float32> var_106;
        wp::vec_t<3,wp::float32>* var_107;
        wp::vec_t<3,wp::float32> var_108;
        wp::vec_t<3,wp::float32> var_109;
        wp::vec_t<3,wp::float32>* var_110;
        wp::vec_t<3,wp::float32> var_111;
        wp::vec_t<3,wp::float32> var_112;
        bool var_113;
        wp::transform_t<wp::float32>* var_114;
        wp::transform_t<wp::float32> var_115;
        wp::transform_t<wp::float32> var_116;
        wp::vec_t<3,wp::float32>* var_117;
        wp::vec_t<3,wp::float32> var_118;
        wp::vec_t<3,wp::float32> var_119;
        wp::vec_t<3,wp::float32> var_120;
        wp::vec_t<3,wp::float32> var_121;
        wp::vec_t<3,wp::float32> var_122;
        wp::vec_t<3,wp::float32> var_123;
        wp::vec_t<3,wp::float32> var_124;
        wp::vec_t<3,wp::float32> var_125;
        bool var_126;
        wp::transform_t<wp::float32>* var_127;
        wp::transform_t<wp::float32> var_128;
        wp::transform_t<wp::float32> var_129;
        wp::vec_t<3,wp::float32>* var_130;
        wp::vec_t<3,wp::float32> var_131;
        wp::vec_t<3,wp::float32> var_132;
        wp::vec_t<3,wp::float32> var_133;
        wp::vec_t<3,wp::float32> var_134;
        wp::vec_t<3,wp::float32> var_135;
        wp::vec_t<3,wp::float32> var_136;
        wp::vec_t<3,wp::float32> var_137;
        wp::vec_t<3,wp::float32> var_138;
        wp::vec_t<3,wp::float32> var_139;
        wp::float32 var_140;
        bool var_141;
        wp::vec_t<3,wp::float32> var_142;
        wp::vec_t<3,wp::float32> var_143;
        bool var_144;
        wp::vec_t<6,wp::float32>* var_145;
        wp::vec_t<6,wp::float32> var_146;
        wp::vec_t<6,wp::float32> var_147;
        wp::vec_t<3,wp::float32> var_148;
        wp::vec_t<3,wp::float32> var_149;
        wp::vec_t<3,wp::float32> var_150;
        wp::vec_t<3,wp::float32> var_151;
        wp::vec_t<3,wp::float32> var_152;
        bool var_153;
        wp::vec_t<6,wp::float32>* var_154;
        wp::vec_t<6,wp::float32> var_155;
        wp::vec_t<6,wp::float32> var_156;
        wp::vec_t<3,wp::float32> var_157;
        wp::vec_t<3,wp::float32> var_158;
        wp::vec_t<3,wp::float32> var_159;
        wp::vec_t<3,wp::float32> var_160;
        wp::vec_t<3,wp::float32> var_161;
        wp::vec_t<3,wp::float32> var_162;
        wp::float32 var_163;
        wp::vec_t<3,wp::float32> var_164;
        wp::vec_t<3,wp::float32> var_165;
        wp::float32 var_166;
        wp::float32 var_167;
        wp::float32 var_168;
        wp::float32 var_169;
        wp::float32 var_170;
        wp::vec_t<3,wp::float32> var_171;
        wp::float32 var_172;
        wp::float32 var_173;
        wp::float32 var_174;
        wp::float32 var_175;
        wp::float32 var_176;
        wp::float32 var_177;
        wp::vec_t<3,wp::float32> var_178;
        wp::float32 var_179;
        wp::vec_t<3,wp::float32> var_180;
        wp::vec_t<3,wp::float32> var_181;
        bool var_182;
        wp::vec_t<3,wp::float32> var_183;
        wp::vec_t<6,wp::float32> var_184;
        wp::vec_t<6,wp::float32> var_185;
        bool var_186;
        wp::vec_t<3,wp::float32> var_187;
        wp::vec_t<6,wp::float32> var_188;
        wp::vec_t<6,wp::float32> var_189;
        //---------
        // forward
        // def eval_rigid_contacts(                                                               <L 947>
        // tid = wp.tid()                                                                         <L 964>
        var_0 = builtin_tid1d();
        // if contact_shape0[tid] == contact_shape1[tid]:                                         <L 965>
        var_1 = wp::address(var_contact_shape0, var_0);
        var_2 = wp::address(var_contact_shape1, var_0);
        var_4 = wp::load(var_1);
        var_5 = wp::load(var_2);
        var_3 = (var_4 == var_5);
        if (var_3) {
            // return                                                                             <L 966>
            return;
        }
        // count = contact_count[0]                                                               <L 968>
        var_7 = wp::address(var_contact_count, var_6);
        var_8 = wp::load(var_7);
        var_9 = wp::copy(var_8);
        // if tid >= count:                                                                       <L 969>
        var_10 = (var_0 >= var_9);
        if (var_10) {
            // return                                                                             <L 970>
            return;
        }
        // ke = 0.0  # restitution coefficient                                                    <L 973>
        // kd = 0.0  # damping coefficient                                                        <L 974>
        // kf = 0.0  # friction coefficient                                                       <L 975>
        // mu = 0.0  # coulomb friction                                                           <L 976>
        // mat_nonzero = 0                                                                        <L 977>
        // thickness_a = 0.0                                                                      <L 978>
        // thickness_b = 0.0                                                                      <L 979>
        // shape_a = contact_shape0[tid]                                                          <L 980>
        var_12 = wp::address(var_contact_shape0, var_0);
        var_13 = wp::load(var_12);
        var_14 = wp::copy(var_13);
        // shape_b = contact_shape1[tid]                                                          <L 981>
        var_15 = wp::address(var_contact_shape1, var_0);
        var_16 = wp::load(var_15);
        var_17 = wp::copy(var_16);
        // if shape_a >= 0:                                                                       <L 982>
        var_18 = (var_14 >= var_6);
        if (var_18) {
            // mat_nonzero += 1                                                                   <L 983>
            var_20 = wp::add(var_6, var_19);
            // ke += shape_materials.ke[shape_a]                                                  <L 984>
            var_21 = &(var_shape_materials.ke);
            var_22 = wp::load(var_21);
            var_23 = wp::address(var_22, var_14);
            var_24 = wp::load(var_23);
            var_25 = wp::add(var_11, var_24);
            // kd += shape_materials.kd[shape_a]                                                  <L 985>
            var_26 = &(var_shape_materials.kd);
            var_27 = wp::load(var_26);
            var_28 = wp::address(var_27, var_14);
            var_29 = wp::load(var_28);
            var_30 = wp::add(var_11, var_29);
            // kf += shape_materials.kf[shape_a]                                                  <L 986>
            var_31 = &(var_shape_materials.kf);
            var_32 = wp::load(var_31);
            var_33 = wp::address(var_32, var_14);
            var_34 = wp::load(var_33);
            var_35 = wp::add(var_11, var_34);
            // mu += shape_materials.mu[shape_a]                                                  <L 987>
            var_36 = &(var_shape_materials.mu);
            var_37 = wp::load(var_36);
            var_38 = wp::address(var_37, var_14);
            var_39 = wp::load(var_38);
            var_40 = wp::add(var_11, var_39);
            // thickness_a = geo.thickness[shape_a]                                               <L 988>
            var_41 = &(var_geo.thickness);
            var_42 = wp::load(var_41);
            var_43 = wp::address(var_42, var_14);
            var_44 = wp::load(var_43);
            var_45 = wp::copy(var_44);
        }
        var_46 = wp::select(var_18, var_11, var_25);
        var_47 = wp::select(var_18, var_11, var_30);
        var_48 = wp::select(var_18, var_11, var_35);
        var_49 = wp::select(var_18, var_11, var_40);
        var_50 = wp::select(var_18, var_6, var_20);
        var_51 = wp::select(var_18, var_11, var_45);
        // if shape_b >= 0:                                                                       <L 989>
        var_52 = (var_17 >= var_6);
        if (var_52) {
            // mat_nonzero += 1                                                                   <L 990>
            var_53 = wp::add(var_50, var_19);
            // ke += shape_materials.ke[shape_b]                                                  <L 991>
            var_54 = &(var_shape_materials.ke);
            var_55 = wp::load(var_54);
            var_56 = wp::address(var_55, var_17);
            var_57 = wp::load(var_56);
            var_58 = wp::add(var_46, var_57);
            // kd += shape_materials.kd[shape_b]                                                  <L 992>
            var_59 = &(var_shape_materials.kd);
            var_60 = wp::load(var_59);
            var_61 = wp::address(var_60, var_17);
            var_62 = wp::load(var_61);
            var_63 = wp::add(var_47, var_62);
            // kf += shape_materials.kf[shape_b]                                                  <L 993>
            var_64 = &(var_shape_materials.kf);
            var_65 = wp::load(var_64);
            var_66 = wp::address(var_65, var_17);
            var_67 = wp::load(var_66);
            var_68 = wp::add(var_48, var_67);
            // mu += shape_materials.mu[shape_b]                                                  <L 994>
            var_69 = &(var_shape_materials.mu);
            var_70 = wp::load(var_69);
            var_71 = wp::address(var_70, var_17);
            var_72 = wp::load(var_71);
            var_73 = wp::add(var_49, var_72);
            // thickness_b = geo.thickness[shape_b]                                               <L 995>
            var_74 = &(var_geo.thickness);
            var_75 = wp::load(var_74);
            var_76 = wp::address(var_75, var_17);
            var_77 = wp::load(var_76);
            var_78 = wp::copy(var_77);
        }
        var_79 = wp::select(var_52, var_46, var_58);
        var_80 = wp::select(var_52, var_47, var_63);
        var_81 = wp::select(var_52, var_48, var_68);
        var_82 = wp::select(var_52, var_49, var_73);
        var_83 = wp::select(var_52, var_50, var_53);
        var_84 = wp::select(var_52, var_11, var_78);
        // if mat_nonzero > 0:                                                                    <L 996>
        var_85 = (var_83 > var_6);
        if (var_85) {
            // ke = ke / float(mat_nonzero)                                                       <L 997>
            var_86 = wp::float(var_83);
            var_87 = wp::div(var_79, var_86);
            // kd = kd / float(mat_nonzero)                                                       <L 998>
            var_88 = wp::float(var_83);
            var_89 = wp::div(var_80, var_88);
            // kf = kf / float(mat_nonzero)                                                       <L 999>
            var_90 = wp::float(var_83);
            var_91 = wp::div(var_81, var_90);
            // mu = mu / float(mat_nonzero)                                                       <L 1000>
            var_92 = wp::float(var_83);
            var_93 = wp::div(var_82, var_92);
        }
        var_94 = wp::select(var_85, var_79, var_87);
        var_95 = wp::select(var_85, var_80, var_89);
        var_96 = wp::select(var_85, var_81, var_91);
        var_97 = wp::select(var_85, var_82, var_93);
        // body_a = contact_body0[tid]                                                            <L 1002>
        var_98 = wp::address(var_contact_body0, var_0);
        var_99 = wp::load(var_98);
        var_100 = wp::copy(var_99);
        // body_b = contact_body1[tid]                                                            <L 1003>
        var_101 = wp::address(var_contact_body1, var_0);
        var_102 = wp::load(var_101);
        var_103 = wp::copy(var_102);
        // n = contact_normal[tid]                                                                <L 1006>
        var_104 = wp::address(var_contact_normal, var_0);
        var_105 = wp::load(var_104);
        var_106 = wp::copy(var_105);
        // bx_a = contact_point0[tid]                                                             <L 1007>
        var_107 = wp::address(var_contact_point0, var_0);
        var_108 = wp::load(var_107);
        var_109 = wp::copy(var_108);
        // bx_b = contact_point1[tid]                                                             <L 1008>
        var_110 = wp::address(var_contact_point1, var_0);
        var_111 = wp::load(var_110);
        var_112 = wp::copy(var_111);
        // if body_a >= 0:                                                                        <L 1009>
        var_113 = (var_100 >= var_6);
        if (var_113) {
            // X_wb_a = body_q[body_a]                                                            <L 1010>
            var_114 = wp::address(var_body_q, var_100);
            var_115 = wp::load(var_114);
            var_116 = wp::copy(var_115);
            // X_com_a = body_com[body_a]                                                         <L 1011>
            var_117 = wp::address(var_body_com, var_100);
            var_118 = wp::load(var_117);
            var_119 = wp::copy(var_118);
            // bx_a = wp.transform_point(X_wb_a, bx_a) - thickness_a * n                          <L 1012>
            var_120 = wp::transform_point(var_116, var_109);
            var_121 = wp::mul(var_51, var_106);
            var_122 = wp::sub(var_120, var_121);
            // r_a = bx_a - wp.transform_point(X_wb_a, X_com_a)                                   <L 1013>
            var_123 = wp::transform_point(var_116, var_119);
            var_124 = wp::sub(var_122, var_123);
        }
        var_125 = wp::select(var_113, var_109, var_122);
        // if body_b >= 0:                                                                        <L 1015>
        var_126 = (var_103 >= var_6);
        if (var_126) {
            // X_wb_b = body_q[body_b]                                                            <L 1016>
            var_127 = wp::address(var_body_q, var_103);
            var_128 = wp::load(var_127);
            var_129 = wp::copy(var_128);
            // X_com_b = body_com[body_b]                                                         <L 1017>
            var_130 = wp::address(var_body_com, var_103);
            var_131 = wp::load(var_130);
            var_132 = wp::copy(var_131);
            // bx_b = wp.transform_point(X_wb_b, bx_b) + thickness_b * n                          <L 1018>
            var_133 = wp::transform_point(var_129, var_112);
            var_134 = wp::mul(var_84, var_106);
            var_135 = wp::add(var_133, var_134);
            // r_b = bx_b - wp.transform_point(X_wb_b, X_com_b)                                   <L 1019>
            var_136 = wp::transform_point(var_129, var_132);
            var_137 = wp::sub(var_135, var_136);
        }
        var_138 = wp::select(var_126, var_112, var_135);
        // d = wp.dot(n, bx_a - bx_b)                                                             <L 1021>
        var_139 = wp::sub(var_125, var_138);
        var_140 = wp::dot(var_106, var_139);
        // if d >= 0.0:                                                                           <L 1023>
        var_141 = (var_140 >= var_11);
        if (var_141) {
            // return                                                                             <L 1024>
            return;
        }
        // bv_a = wp.vec3(0.0)                                                                    <L 1027>
        var_142 = wp::vec_t<3,wp::float32>(var_11);
        // bv_b = wp.vec3(0.0)                                                                    <L 1028>
        var_143 = wp::vec_t<3,wp::float32>(var_11);
        // if body_a >= 0:                                                                        <L 1029>
        var_144 = (var_100 >= var_6);
        if (var_144) {
            // body_v_s_a = body_qd[body_a]                                                       <L 1030>
            var_145 = wp::address(var_body_qd, var_100);
            var_146 = wp::load(var_145);
            var_147 = wp::copy(var_146);
            // body_w_a = wp.spatial_top(body_v_s_a)                                              <L 1031>
            var_148 = wp::spatial_top(var_147);
            // body_v_a = wp.spatial_bottom(body_v_s_a)                                           <L 1032>
            var_149 = wp::spatial_bottom(var_147);
            // bv_a = body_v_a + wp.cross(body_w_a, r_a)                                          <L 1033>
            var_150 = wp::cross(var_148, var_124);
            var_151 = wp::add(var_149, var_150);
        }
        var_152 = wp::select(var_144, var_142, var_151);
        // if body_b >= 0:                                                                        <L 1035>
        var_153 = (var_103 >= var_6);
        if (var_153) {
            // body_v_s_b = body_qd[body_b]                                                       <L 1036>
            var_154 = wp::address(var_body_qd, var_103);
            var_155 = wp::load(var_154);
            var_156 = wp::copy(var_155);
            // body_w_b = wp.spatial_top(body_v_s_b)                                              <L 1037>
            var_157 = wp::spatial_top(var_156);
            // body_v_b = wp.spatial_bottom(body_v_s_b)                                           <L 1038>
            var_158 = wp::spatial_bottom(var_156);
            // bv_b = body_v_b + wp.cross(body_w_b, r_b)                                          <L 1039>
            var_159 = wp::cross(var_157, var_137);
            var_160 = wp::add(var_158, var_159);
        }
        var_161 = wp::select(var_153, var_143, var_160);
        // v = bv_a - bv_b                                                                        <L 1042>
        var_162 = wp::sub(var_152, var_161);
        // vn = wp.dot(n, v)                                                                      <L 1047>
        var_163 = wp::dot(var_106, var_162);
        // vt = v - n * vn                                                                        <L 1048>
        var_164 = wp::mul(var_106, var_163);
        var_165 = wp::sub(var_162, var_164);
        // fn = d * ke                                                                            <L 1051>
        var_166 = wp::mul(var_140, var_94);
        // fd = wp.min(vn, 0.0) * kd * wp.step(d)                                                 <L 1054>
        var_167 = wp::min(var_163, var_11);
        var_168 = wp::mul(var_167, var_95);
        var_169 = wp::step(var_140);
        var_170 = wp::mul(var_168, var_169);
        // ft = wp.normalize(vt) * wp.min(kf * wp.length(vt), 0.0 - mu * (fn + fd))               <L 1070>
        var_171 = wp::normalize(var_165);
        var_172 = wp::length(var_165);
        var_173 = wp::mul(var_96, var_172);
        var_174 = wp::add(var_166, var_170);
        var_175 = wp::mul(var_97, var_174);
        var_176 = wp::sub(var_11, var_175);
        var_177 = wp::min(var_173, var_176);
        var_178 = wp::mul(var_171, var_177);
        // f_total = n * (fn + fd) + ft                                                           <L 1073>
        var_179 = wp::add(var_166, var_170);
        var_180 = wp::mul(var_106, var_179);
        var_181 = wp::add(var_180, var_178);
        // if body_a >= 0:                                                                        <L 1079>
        var_182 = (var_100 >= var_6);
        if (var_182) {
            // wp.atomic_sub(body_f, body_a, wp.spatial_vector(wp.cross(r_a, f_total), f_total))       <L 1080>
            var_183 = wp::cross(var_124, var_181);
            var_184 = wp::vec_t<6,wp::float32>(var_183, var_181);
            var_185 = wp::atomic_sub(var_body_f, var_100, var_184);
        }
        // if body_b >= 0:                                                                        <L 1081>
        var_186 = (var_103 >= var_6);
        if (var_186) {
            // wp.atomic_add(body_f, body_b, wp.spatial_vector(wp.cross(r_b, f_total), f_total))       <L 1082>
            var_187 = wp::cross(var_137, var_181);
            var_188 = wp::vec_t<6,wp::float32>(var_187, var_181);
            var_189 = wp::atomic_add(var_body_f, var_103, var_188);
        }
    }
}

extern "C" __global__ void eval_rigid_contacts_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::transform_t<wp::float32>> var_body_q,
    wp::array_t<wp::vec_t<6,wp::float32>> var_body_qd,
    wp::array_t<wp::vec_t<3,wp::float32>> var_body_com,
    ModelShapeMaterials var_shape_materials,
    ModelShapeGeometry var_geo,
    wp::array_t<wp::int32> var_contact_count,
    wp::array_t<wp::int32> var_contact_body0,
    wp::array_t<wp::int32> var_contact_body1,
    wp::array_t<wp::vec_t<3,wp::float32>> var_contact_point0,
    wp::array_t<wp::vec_t<3,wp::float32>> var_contact_point1,
    wp::array_t<wp::vec_t<3,wp::float32>> var_contact_normal,
    wp::array_t<wp::int32> var_contact_shape0,
    wp::array_t<wp::int32> var_contact_shape1,
    wp::array_t<wp::vec_t<6,wp::float32>> var_body_f,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_q,
    wp::array_t<wp::vec_t<6,wp::float32>> adj_body_qd,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_body_com,
    ModelShapeMaterials adj_shape_materials,
    ModelShapeGeometry adj_geo,
    wp::array_t<wp::int32> adj_contact_count,
    wp::array_t<wp::int32> adj_contact_body0,
    wp::array_t<wp::int32> adj_contact_body1,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_contact_point0,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_contact_point1,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_contact_normal,
    wp::array_t<wp::int32> adj_contact_shape0,
    wp::array_t<wp::int32> adj_contact_shape1,
    wp::array_t<wp::vec_t<6,wp::float32>> adj_body_f)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32* var_1;
        wp::int32* var_2;
        bool var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        const wp::int32 var_6 = 0;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        bool var_10;
        const wp::float32 var_11 = 0.0;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        bool var_18;
        const wp::int32 var_19 = 1;
        wp::int32 var_20;
        wp::array_t<wp::float32>* var_21;
        wp::array_t<wp::float32> var_22;
        wp::float32* var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        wp::array_t<wp::float32>* var_26;
        wp::array_t<wp::float32> var_27;
        wp::float32* var_28;
        wp::float32 var_29;
        wp::float32 var_30;
        wp::array_t<wp::float32>* var_31;
        wp::array_t<wp::float32> var_32;
        wp::float32* var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        wp::array_t<wp::float32>* var_36;
        wp::array_t<wp::float32> var_37;
        wp::float32* var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        wp::array_t<wp::float32>* var_41;
        wp::array_t<wp::float32> var_42;
        wp::float32* var_43;
        wp::float32 var_44;
        wp::float32 var_45;
        wp::float32 var_46;
        wp::float32 var_47;
        wp::float32 var_48;
        wp::float32 var_49;
        wp::int32 var_50;
        wp::float32 var_51;
        bool var_52;
        wp::int32 var_53;
        wp::array_t<wp::float32>* var_54;
        wp::array_t<wp::float32> var_55;
        wp::float32* var_56;
        wp::float32 var_57;
        wp::float32 var_58;
        wp::array_t<wp::float32>* var_59;
        wp::array_t<wp::float32> var_60;
        wp::float32* var_61;
        wp::float32 var_62;
        wp::float32 var_63;
        wp::array_t<wp::float32>* var_64;
        wp::array_t<wp::float32> var_65;
        wp::float32* var_66;
        wp::float32 var_67;
        wp::float32 var_68;
        wp::array_t<wp::float32>* var_69;
        wp::array_t<wp::float32> var_70;
        wp::float32* var_71;
        wp::float32 var_72;
        wp::float32 var_73;
        wp::array_t<wp::float32>* var_74;
        wp::array_t<wp::float32> var_75;
        wp::float32* var_76;
        wp::float32 var_77;
        wp::float32 var_78;
        wp::float32 var_79;
        wp::float32 var_80;
        wp::float32 var_81;
        wp::float32 var_82;
        wp::int32 var_83;
        wp::float32 var_84;
        bool var_85;
        wp::float32 var_86;
        wp::float32 var_87;
        wp::float32 var_88;
        wp::float32 var_89;
        wp::float32 var_90;
        wp::float32 var_91;
        wp::float32 var_92;
        wp::float32 var_93;
        wp::float32 var_94;
        wp::float32 var_95;
        wp::float32 var_96;
        wp::float32 var_97;
        wp::int32* var_98;
        wp::int32 var_99;
        wp::int32 var_100;
        wp::int32* var_101;
        wp::int32 var_102;
        wp::int32 var_103;
        wp::vec_t<3,wp::float32>* var_104;
        wp::vec_t<3,wp::float32> var_105;
        wp::vec_t<3,wp::float32> var_106;
        wp::vec_t<3,wp::float32>* var_107;
        wp::vec_t<3,wp::float32> var_108;
        wp::vec_t<3,wp::float32> var_109;
        wp::vec_t<3,wp::float32>* var_110;
        wp::vec_t<3,wp::float32> var_111;
        wp::vec_t<3,wp::float32> var_112;
        bool var_113;
        wp::transform_t<wp::float32>* var_114;
        wp::transform_t<wp::float32> var_115;
        wp::transform_t<wp::float32> var_116;
        wp::vec_t<3,wp::float32>* var_117;
        wp::vec_t<3,wp::float32> var_118;
        wp::vec_t<3,wp::float32> var_119;
        wp::vec_t<3,wp::float32> var_120;
        wp::vec_t<3,wp::float32> var_121;
        wp::vec_t<3,wp::float32> var_122;
        wp::vec_t<3,wp::float32> var_123;
        wp::vec_t<3,wp::float32> var_124;
        wp::vec_t<3,wp::float32> var_125;
        bool var_126;
        wp::transform_t<wp::float32>* var_127;
        wp::transform_t<wp::float32> var_128;
        wp::transform_t<wp::float32> var_129;
        wp::vec_t<3,wp::float32>* var_130;
        wp::vec_t<3,wp::float32> var_131;
        wp::vec_t<3,wp::float32> var_132;
        wp::vec_t<3,wp::float32> var_133;
        wp::vec_t<3,wp::float32> var_134;
        wp::vec_t<3,wp::float32> var_135;
        wp::vec_t<3,wp::float32> var_136;
        wp::vec_t<3,wp::float32> var_137;
        wp::vec_t<3,wp::float32> var_138;
        wp::vec_t<3,wp::float32> var_139;
        wp::float32 var_140;
        bool var_141;
        wp::vec_t<3,wp::float32> var_142;
        wp::vec_t<3,wp::float32> var_143;
        bool var_144;
        wp::vec_t<6,wp::float32>* var_145;
        wp::vec_t<6,wp::float32> var_146;
        wp::vec_t<6,wp::float32> var_147;
        wp::vec_t<3,wp::float32> var_148;
        wp::vec_t<3,wp::float32> var_149;
        wp::vec_t<3,wp::float32> var_150;
        wp::vec_t<3,wp::float32> var_151;
        wp::vec_t<3,wp::float32> var_152;
        bool var_153;
        wp::vec_t<6,wp::float32>* var_154;
        wp::vec_t<6,wp::float32> var_155;
        wp::vec_t<6,wp::float32> var_156;
        wp::vec_t<3,wp::float32> var_157;
        wp::vec_t<3,wp::float32> var_158;
        wp::vec_t<3,wp::float32> var_159;
        wp::vec_t<3,wp::float32> var_160;
        wp::vec_t<3,wp::float32> var_161;
        wp::vec_t<3,wp::float32> var_162;
        wp::float32 var_163;
        wp::vec_t<3,wp::float32> var_164;
        wp::vec_t<3,wp::float32> var_165;
        wp::float32 var_166;
        wp::float32 var_167;
        wp::float32 var_168;
        wp::float32 var_169;
        wp::float32 var_170;
        wp::vec_t<3,wp::float32> var_171;
        wp::float32 var_172;
        wp::float32 var_173;
        wp::float32 var_174;
        wp::float32 var_175;
        wp::float32 var_176;
        wp::float32 var_177;
        wp::vec_t<3,wp::float32> var_178;
        wp::float32 var_179;
        wp::vec_t<3,wp::float32> var_180;
        wp::vec_t<3,wp::float32> var_181;
        bool var_182;
        wp::vec_t<3,wp::float32> var_183;
        wp::vec_t<6,wp::float32> var_184;
        wp::vec_t<6,wp::float32> var_185;
        bool var_186;
        wp::vec_t<3,wp::float32> var_187;
        wp::vec_t<6,wp::float32> var_188;
        wp::vec_t<6,wp::float32> var_189;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        bool adj_10 = {};
        wp::float32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        bool adj_18 = {};
        wp::int32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::array_t<wp::float32> adj_21 = {};
        wp::array_t<wp::float32> adj_22 = {};
        wp::float32 adj_23 = {};
        wp::float32 adj_24 = {};
        wp::float32 adj_25 = {};
        wp::array_t<wp::float32> adj_26 = {};
        wp::array_t<wp::float32> adj_27 = {};
        wp::float32 adj_28 = {};
        wp::float32 adj_29 = {};
        wp::float32 adj_30 = {};
        wp::array_t<wp::float32> adj_31 = {};
        wp::array_t<wp::float32> adj_32 = {};
        wp::float32 adj_33 = {};
        wp::float32 adj_34 = {};
        wp::float32 adj_35 = {};
        wp::array_t<wp::float32> adj_36 = {};
        wp::array_t<wp::float32> adj_37 = {};
        wp::float32 adj_38 = {};
        wp::float32 adj_39 = {};
        wp::float32 adj_40 = {};
        wp::array_t<wp::float32> adj_41 = {};
        wp::array_t<wp::float32> adj_42 = {};
        wp::float32 adj_43 = {};
        wp::float32 adj_44 = {};
        wp::float32 adj_45 = {};
        wp::float32 adj_46 = {};
        wp::float32 adj_47 = {};
        wp::float32 adj_48 = {};
        wp::float32 adj_49 = {};
        wp::int32 adj_50 = {};
        wp::float32 adj_51 = {};
        bool adj_52 = {};
        wp::int32 adj_53 = {};
        wp::array_t<wp::float32> adj_54 = {};
        wp::array_t<wp::float32> adj_55 = {};
        wp::float32 adj_56 = {};
        wp::float32 adj_57 = {};
        wp::float32 adj_58 = {};
        wp::array_t<wp::float32> adj_59 = {};
        wp::array_t<wp::float32> adj_60 = {};
        wp::float32 adj_61 = {};
        wp::float32 adj_62 = {};
        wp::float32 adj_63 = {};
        wp::array_t<wp::float32> adj_64 = {};
        wp::array_t<wp::float32> adj_65 = {};
        wp::float32 adj_66 = {};
        wp::float32 adj_67 = {};
        wp::float32 adj_68 = {};
        wp::array_t<wp::float32> adj_69 = {};
        wp::array_t<wp::float32> adj_70 = {};
        wp::float32 adj_71 = {};
        wp::float32 adj_72 = {};
        wp::float32 adj_73 = {};
        wp::array_t<wp::float32> adj_74 = {};
        wp::array_t<wp::float32> adj_75 = {};
        wp::float32 adj_76 = {};
        wp::float32 adj_77 = {};
        wp::float32 adj_78 = {};
        wp::float32 adj_79 = {};
        wp::float32 adj_80 = {};
        wp::float32 adj_81 = {};
        wp::float32 adj_82 = {};
        wp::int32 adj_83 = {};
        wp::float32 adj_84 = {};
        bool adj_85 = {};
        wp::float32 adj_86 = {};
        wp::float32 adj_87 = {};
        wp::float32 adj_88 = {};
        wp::float32 adj_89 = {};
        wp::float32 adj_90 = {};
        wp::float32 adj_91 = {};
        wp::float32 adj_92 = {};
        wp::float32 adj_93 = {};
        wp::float32 adj_94 = {};
        wp::float32 adj_95 = {};
        wp::float32 adj_96 = {};
        wp::float32 adj_97 = {};
        wp::int32 adj_98 = {};
        wp::int32 adj_99 = {};
        wp::int32 adj_100 = {};
        wp::int32 adj_101 = {};
        wp::int32 adj_102 = {};
        wp::int32 adj_103 = {};
        wp::vec_t<3,wp::float32> adj_104 = {};
        wp::vec_t<3,wp::float32> adj_105 = {};
        wp::vec_t<3,wp::float32> adj_106 = {};
        wp::vec_t<3,wp::float32> adj_107 = {};
        wp::vec_t<3,wp::float32> adj_108 = {};
        wp::vec_t<3,wp::float32> adj_109 = {};
        wp::vec_t<3,wp::float32> adj_110 = {};
        wp::vec_t<3,wp::float32> adj_111 = {};
        wp::vec_t<3,wp::float32> adj_112 = {};
        bool adj_113 = {};
        wp::transform_t<wp::float32> adj_114 = {};
        wp::transform_t<wp::float32> adj_115 = {};
        wp::transform_t<wp::float32> adj_116 = {};
        wp::vec_t<3,wp::float32> adj_117 = {};
        wp::vec_t<3,wp::float32> adj_118 = {};
        wp::vec_t<3,wp::float32> adj_119 = {};
        wp::vec_t<3,wp::float32> adj_120 = {};
        wp::vec_t<3,wp::float32> adj_121 = {};
        wp::vec_t<3,wp::float32> adj_122 = {};
        wp::vec_t<3,wp::float32> adj_123 = {};
        wp::vec_t<3,wp::float32> adj_124 = {};
        wp::vec_t<3,wp::float32> adj_125 = {};
        bool adj_126 = {};
        wp::transform_t<wp::float32> adj_127 = {};
        wp::transform_t<wp::float32> adj_128 = {};
        wp::transform_t<wp::float32> adj_129 = {};
        wp::vec_t<3,wp::float32> adj_130 = {};
        wp::vec_t<3,wp::float32> adj_131 = {};
        wp::vec_t<3,wp::float32> adj_132 = {};
        wp::vec_t<3,wp::float32> adj_133 = {};
        wp::vec_t<3,wp::float32> adj_134 = {};
        wp::vec_t<3,wp::float32> adj_135 = {};
        wp::vec_t<3,wp::float32> adj_136 = {};
        wp::vec_t<3,wp::float32> adj_137 = {};
        wp::vec_t<3,wp::float32> adj_138 = {};
        wp::vec_t<3,wp::float32> adj_139 = {};
        wp::float32 adj_140 = {};
        bool adj_141 = {};
        wp::vec_t<3,wp::float32> adj_142 = {};
        wp::vec_t<3,wp::float32> adj_143 = {};
        bool adj_144 = {};
        wp::vec_t<6,wp::float32> adj_145 = {};
        wp::vec_t<6,wp::float32> adj_146 = {};
        wp::vec_t<6,wp::float32> adj_147 = {};
        wp::vec_t<3,wp::float32> adj_148 = {};
        wp::vec_t<3,wp::float32> adj_149 = {};
        wp::vec_t<3,wp::float32> adj_150 = {};
        wp::vec_t<3,wp::float32> adj_151 = {};
        wp::vec_t<3,wp::float32> adj_152 = {};
        bool adj_153 = {};
        wp::vec_t<6,wp::float32> adj_154 = {};
        wp::vec_t<6,wp::float32> adj_155 = {};
        wp::vec_t<6,wp::float32> adj_156 = {};
        wp::vec_t<3,wp::float32> adj_157 = {};
        wp::vec_t<3,wp::float32> adj_158 = {};
        wp::vec_t<3,wp::float32> adj_159 = {};
        wp::vec_t<3,wp::float32> adj_160 = {};
        wp::vec_t<3,wp::float32> adj_161 = {};
        wp::vec_t<3,wp::float32> adj_162 = {};
        wp::float32 adj_163 = {};
        wp::vec_t<3,wp::float32> adj_164 = {};
        wp::vec_t<3,wp::float32> adj_165 = {};
        wp::float32 adj_166 = {};
        wp::float32 adj_167 = {};
        wp::float32 adj_168 = {};
        wp::float32 adj_169 = {};
        wp::float32 adj_170 = {};
        wp::vec_t<3,wp::float32> adj_171 = {};
        wp::float32 adj_172 = {};
        wp::float32 adj_173 = {};
        wp::float32 adj_174 = {};
        wp::float32 adj_175 = {};
        wp::float32 adj_176 = {};
        wp::float32 adj_177 = {};
        wp::vec_t<3,wp::float32> adj_178 = {};
        wp::float32 adj_179 = {};
        wp::vec_t<3,wp::float32> adj_180 = {};
        wp::vec_t<3,wp::float32> adj_181 = {};
        bool adj_182 = {};
        wp::vec_t<3,wp::float32> adj_183 = {};
        wp::vec_t<6,wp::float32> adj_184 = {};
        wp::vec_t<6,wp::float32> adj_185 = {};
        bool adj_186 = {};
        wp::vec_t<3,wp::float32> adj_187 = {};
        wp::vec_t<6,wp::float32> adj_188 = {};
        wp::vec_t<6,wp::float32> adj_189 = {};
        //---------
        // forward
        // def eval_rigid_contacts(                                                               <L 947>
        // tid = wp.tid()                                                                         <L 964>
        var_0 = builtin_tid1d();
        // if contact_shape0[tid] == contact_shape1[tid]:                                         <L 965>
        var_1 = wp::address(var_contact_shape0, var_0);
        var_2 = wp::address(var_contact_shape1, var_0);
        var_4 = wp::load(var_1);
        var_5 = wp::load(var_2);
        var_3 = (var_4 == var_5);
        if (var_3) {
            // return                                                                             <L 966>
            goto label0;
        }
        // count = contact_count[0]                                                               <L 968>
        var_7 = wp::address(var_contact_count, var_6);
        var_8 = wp::load(var_7);
        var_9 = wp::copy(var_8);
        // if tid >= count:                                                                       <L 969>
        var_10 = (var_0 >= var_9);
        if (var_10) {
            // return                                                                             <L 970>
            goto label1;
        }
        // ke = 0.0  # restitution coefficient                                                    <L 973>
        // kd = 0.0  # damping coefficient                                                        <L 974>
        // kf = 0.0  # friction coefficient                                                       <L 975>
        // mu = 0.0  # coulomb friction                                                           <L 976>
        // mat_nonzero = 0                                                                        <L 977>
        // thickness_a = 0.0                                                                      <L 978>
        // thickness_b = 0.0                                                                      <L 979>
        // shape_a = contact_shape0[tid]                                                          <L 980>
        var_12 = wp::address(var_contact_shape0, var_0);
        var_13 = wp::load(var_12);
        var_14 = wp::copy(var_13);
        // shape_b = contact_shape1[tid]                                                          <L 981>
        var_15 = wp::address(var_contact_shape1, var_0);
        var_16 = wp::load(var_15);
        var_17 = wp::copy(var_16);
        // if shape_a >= 0:                                                                       <L 982>
        var_18 = (var_14 >= var_6);
        if (var_18) {
            // mat_nonzero += 1                                                                   <L 983>
            var_20 = wp::add(var_6, var_19);
            // ke += shape_materials.ke[shape_a]                                                  <L 984>
            var_21 = &(var_shape_materials.ke);
            var_22 = wp::load(var_21);
            var_23 = wp::address(var_22, var_14);
            var_24 = wp::load(var_23);
            var_25 = wp::add(var_11, var_24);
            // kd += shape_materials.kd[shape_a]                                                  <L 985>
            var_26 = &(var_shape_materials.kd);
            var_27 = wp::load(var_26);
            var_28 = wp::address(var_27, var_14);
            var_29 = wp::load(var_28);
            var_30 = wp::add(var_11, var_29);
            // kf += shape_materials.kf[shape_a]                                                  <L 986>
            var_31 = &(var_shape_materials.kf);
            var_32 = wp::load(var_31);
            var_33 = wp::address(var_32, var_14);
            var_34 = wp::load(var_33);
            var_35 = wp::add(var_11, var_34);
            // mu += shape_materials.mu[shape_a]                                                  <L 987>
            var_36 = &(var_shape_materials.mu);
            var_37 = wp::load(var_36);
            var_38 = wp::address(var_37, var_14);
            var_39 = wp::load(var_38);
            var_40 = wp::add(var_11, var_39);
            // thickness_a = geo.thickness[shape_a]                                               <L 988>
            var_41 = &(var_geo.thickness);
            var_42 = wp::load(var_41);
            var_43 = wp::address(var_42, var_14);
            var_44 = wp::load(var_43);
            var_45 = wp::copy(var_44);
        }
        var_46 = wp::select(var_18, var_11, var_25);
        var_47 = wp::select(var_18, var_11, var_30);
        var_48 = wp::select(var_18, var_11, var_35);
        var_49 = wp::select(var_18, var_11, var_40);
        var_50 = wp::select(var_18, var_6, var_20);
        var_51 = wp::select(var_18, var_11, var_45);
        // if shape_b >= 0:                                                                       <L 989>
        var_52 = (var_17 >= var_6);
        if (var_52) {
            // mat_nonzero += 1                                                                   <L 990>
            var_53 = wp::add(var_50, var_19);
            // ke += shape_materials.ke[shape_b]                                                  <L 991>
            var_54 = &(var_shape_materials.ke);
            var_55 = wp::load(var_54);
            var_56 = wp::address(var_55, var_17);
            var_57 = wp::load(var_56);
            var_58 = wp::add(var_46, var_57);
            // kd += shape_materials.kd[shape_b]                                                  <L 992>
            var_59 = &(var_shape_materials.kd);
            var_60 = wp::load(var_59);
            var_61 = wp::address(var_60, var_17);
            var_62 = wp::load(var_61);
            var_63 = wp::add(var_47, var_62);
            // kf += shape_materials.kf[shape_b]                                                  <L 993>
            var_64 = &(var_shape_materials.kf);
            var_65 = wp::load(var_64);
            var_66 = wp::address(var_65, var_17);
            var_67 = wp::load(var_66);
            var_68 = wp::add(var_48, var_67);
            // mu += shape_materials.mu[shape_b]                                                  <L 994>
            var_69 = &(var_shape_materials.mu);
            var_70 = wp::load(var_69);
            var_71 = wp::address(var_70, var_17);
            var_72 = wp::load(var_71);
            var_73 = wp::add(var_49, var_72);
            // thickness_b = geo.thickness[shape_b]                                               <L 995>
            var_74 = &(var_geo.thickness);
            var_75 = wp::load(var_74);
            var_76 = wp::address(var_75, var_17);
            var_77 = wp::load(var_76);
            var_78 = wp::copy(var_77);
        }
        var_79 = wp::select(var_52, var_46, var_58);
        var_80 = wp::select(var_52, var_47, var_63);
        var_81 = wp::select(var_52, var_48, var_68);
        var_82 = wp::select(var_52, var_49, var_73);
        var_83 = wp::select(var_52, var_50, var_53);
        var_84 = wp::select(var_52, var_11, var_78);
        // if mat_nonzero > 0:                                                                    <L 996>
        var_85 = (var_83 > var_6);
        if (var_85) {
            // ke = ke / float(mat_nonzero)                                                       <L 997>
            var_86 = wp::float(var_83);
            var_87 = wp::div(var_79, var_86);
            // kd = kd / float(mat_nonzero)                                                       <L 998>
            var_88 = wp::float(var_83);
            var_89 = wp::div(var_80, var_88);
            // kf = kf / float(mat_nonzero)                                                       <L 999>
            var_90 = wp::float(var_83);
            var_91 = wp::div(var_81, var_90);
            // mu = mu / float(mat_nonzero)                                                       <L 1000>
            var_92 = wp::float(var_83);
            var_93 = wp::div(var_82, var_92);
        }
        var_94 = wp::select(var_85, var_79, var_87);
        var_95 = wp::select(var_85, var_80, var_89);
        var_96 = wp::select(var_85, var_81, var_91);
        var_97 = wp::select(var_85, var_82, var_93);
        // body_a = contact_body0[tid]                                                            <L 1002>
        var_98 = wp::address(var_contact_body0, var_0);
        var_99 = wp::load(var_98);
        var_100 = wp::copy(var_99);
        // body_b = contact_body1[tid]                                                            <L 1003>
        var_101 = wp::address(var_contact_body1, var_0);
        var_102 = wp::load(var_101);
        var_103 = wp::copy(var_102);
        // n = contact_normal[tid]                                                                <L 1006>
        var_104 = wp::address(var_contact_normal, var_0);
        var_105 = wp::load(var_104);
        var_106 = wp::copy(var_105);
        // bx_a = contact_point0[tid]                                                             <L 1007>
        var_107 = wp::address(var_contact_point0, var_0);
        var_108 = wp::load(var_107);
        var_109 = wp::copy(var_108);
        // bx_b = contact_point1[tid]                                                             <L 1008>
        var_110 = wp::address(var_contact_point1, var_0);
        var_111 = wp::load(var_110);
        var_112 = wp::copy(var_111);
        // if body_a >= 0:                                                                        <L 1009>
        var_113 = (var_100 >= var_6);
        if (var_113) {
            // X_wb_a = body_q[body_a]                                                            <L 1010>
            var_114 = wp::address(var_body_q, var_100);
            var_115 = wp::load(var_114);
            var_116 = wp::copy(var_115);
            // X_com_a = body_com[body_a]                                                         <L 1011>
            var_117 = wp::address(var_body_com, var_100);
            var_118 = wp::load(var_117);
            var_119 = wp::copy(var_118);
            // bx_a = wp.transform_point(X_wb_a, bx_a) - thickness_a * n                          <L 1012>
            var_120 = wp::transform_point(var_116, var_109);
            var_121 = wp::mul(var_51, var_106);
            var_122 = wp::sub(var_120, var_121);
            // r_a = bx_a - wp.transform_point(X_wb_a, X_com_a)                                   <L 1013>
            var_123 = wp::transform_point(var_116, var_119);
            var_124 = wp::sub(var_122, var_123);
        }
        var_125 = wp::select(var_113, var_109, var_122);
        // if body_b >= 0:                                                                        <L 1015>
        var_126 = (var_103 >= var_6);
        if (var_126) {
            // X_wb_b = body_q[body_b]                                                            <L 1016>
            var_127 = wp::address(var_body_q, var_103);
            var_128 = wp::load(var_127);
            var_129 = wp::copy(var_128);
            // X_com_b = body_com[body_b]                                                         <L 1017>
            var_130 = wp::address(var_body_com, var_103);
            var_131 = wp::load(var_130);
            var_132 = wp::copy(var_131);
            // bx_b = wp.transform_point(X_wb_b, bx_b) + thickness_b * n                          <L 1018>
            var_133 = wp::transform_point(var_129, var_112);
            var_134 = wp::mul(var_84, var_106);
            var_135 = wp::add(var_133, var_134);
            // r_b = bx_b - wp.transform_point(X_wb_b, X_com_b)                                   <L 1019>
            var_136 = wp::transform_point(var_129, var_132);
            var_137 = wp::sub(var_135, var_136);
        }
        var_138 = wp::select(var_126, var_112, var_135);
        // d = wp.dot(n, bx_a - bx_b)                                                             <L 1021>
        var_139 = wp::sub(var_125, var_138);
        var_140 = wp::dot(var_106, var_139);
        // if d >= 0.0:                                                                           <L 1023>
        var_141 = (var_140 >= var_11);
        if (var_141) {
            // return                                                                             <L 1024>
            goto label2;
        }
        // bv_a = wp.vec3(0.0)                                                                    <L 1027>
        var_142 = wp::vec_t<3,wp::float32>(var_11);
        // bv_b = wp.vec3(0.0)                                                                    <L 1028>
        var_143 = wp::vec_t<3,wp::float32>(var_11);
        // if body_a >= 0:                                                                        <L 1029>
        var_144 = (var_100 >= var_6);
        if (var_144) {
            // body_v_s_a = body_qd[body_a]                                                       <L 1030>
            var_145 = wp::address(var_body_qd, var_100);
            var_146 = wp::load(var_145);
            var_147 = wp::copy(var_146);
            // body_w_a = wp.spatial_top(body_v_s_a)                                              <L 1031>
            var_148 = wp::spatial_top(var_147);
            // body_v_a = wp.spatial_bottom(body_v_s_a)                                           <L 1032>
            var_149 = wp::spatial_bottom(var_147);
            // bv_a = body_v_a + wp.cross(body_w_a, r_a)                                          <L 1033>
            var_150 = wp::cross(var_148, var_124);
            var_151 = wp::add(var_149, var_150);
        }
        var_152 = wp::select(var_144, var_142, var_151);
        // if body_b >= 0:                                                                        <L 1035>
        var_153 = (var_103 >= var_6);
        if (var_153) {
            // body_v_s_b = body_qd[body_b]                                                       <L 1036>
            var_154 = wp::address(var_body_qd, var_103);
            var_155 = wp::load(var_154);
            var_156 = wp::copy(var_155);
            // body_w_b = wp.spatial_top(body_v_s_b)                                              <L 1037>
            var_157 = wp::spatial_top(var_156);
            // body_v_b = wp.spatial_bottom(body_v_s_b)                                           <L 1038>
            var_158 = wp::spatial_bottom(var_156);
            // bv_b = body_v_b + wp.cross(body_w_b, r_b)                                          <L 1039>
            var_159 = wp::cross(var_157, var_137);
            var_160 = wp::add(var_158, var_159);
        }
        var_161 = wp::select(var_153, var_143, var_160);
        // v = bv_a - bv_b                                                                        <L 1042>
        var_162 = wp::sub(var_152, var_161);
        // vn = wp.dot(n, v)                                                                      <L 1047>
        var_163 = wp::dot(var_106, var_162);
        // vt = v - n * vn                                                                        <L 1048>
        var_164 = wp::mul(var_106, var_163);
        var_165 = wp::sub(var_162, var_164);
        // fn = d * ke                                                                            <L 1051>
        var_166 = wp::mul(var_140, var_94);
        // fd = wp.min(vn, 0.0) * kd * wp.step(d)                                                 <L 1054>
        var_167 = wp::min(var_163, var_11);
        var_168 = wp::mul(var_167, var_95);
        var_169 = wp::step(var_140);
        var_170 = wp::mul(var_168, var_169);
        // ft = wp.normalize(vt) * wp.min(kf * wp.length(vt), 0.0 - mu * (fn + fd))               <L 1070>
        var_171 = wp::normalize(var_165);
        var_172 = wp::length(var_165);
        var_173 = wp::mul(var_96, var_172);
        var_174 = wp::add(var_166, var_170);
        var_175 = wp::mul(var_97, var_174);
        var_176 = wp::sub(var_11, var_175);
        var_177 = wp::min(var_173, var_176);
        var_178 = wp::mul(var_171, var_177);
        // f_total = n * (fn + fd) + ft                                                           <L 1073>
        var_179 = wp::add(var_166, var_170);
        var_180 = wp::mul(var_106, var_179);
        var_181 = wp::add(var_180, var_178);
        // if body_a >= 0:                                                                        <L 1079>
        var_182 = (var_100 >= var_6);
        if (var_182) {
            // wp.atomic_sub(body_f, body_a, wp.spatial_vector(wp.cross(r_a, f_total), f_total))       <L 1080>
            var_183 = wp::cross(var_124, var_181);
            var_184 = wp::vec_t<6,wp::float32>(var_183, var_181);
            // var_185 = wp::atomic_sub(var_body_f, var_100, var_184);
        }
        // if body_b >= 0:                                                                        <L 1081>
        var_186 = (var_103 >= var_6);
        if (var_186) {
            // wp.atomic_add(body_f, body_b, wp.spatial_vector(wp.cross(r_b, f_total), f_total))       <L 1082>
            var_187 = wp::cross(var_137, var_181);
            var_188 = wp::vec_t<6,wp::float32>(var_187, var_181);
            // var_189 = wp::atomic_add(var_body_f, var_103, var_188);
        }
        //---------
        // reverse
        if (var_186) {
            wp::adj_atomic_add(var_body_f, var_103, var_188, adj_body_f, adj_103, adj_188, adj_189);
            wp::adj_vec_t(var_187, var_181, adj_187, adj_181, adj_188);
            wp::adj_cross(var_137, var_181, adj_137, adj_181, adj_187);
            // adj: wp.atomic_add(body_f, body_b, wp.spatial_vector(wp.cross(r_b, f_total), f_total))  <L 1082>
        }
        // adj: if body_b >= 0:                                                                   <L 1081>
        if (var_182) {
            wp::adj_atomic_sub(var_body_f, var_100, var_184, adj_body_f, adj_100, adj_184, adj_185);
            wp::adj_vec_t(var_183, var_181, adj_183, adj_181, adj_184);
            wp::adj_cross(var_124, var_181, adj_124, adj_181, adj_183);
            // adj: wp.atomic_sub(body_f, body_a, wp.spatial_vector(wp.cross(r_a, f_total), f_total))  <L 1080>
        }
        // adj: if body_a >= 0:                                                                   <L 1079>
        wp::adj_add(var_180, var_178, adj_180, adj_178, adj_181);
        wp::adj_mul(var_106, var_179, adj_106, adj_179, adj_180);
        wp::adj_add(var_166, var_170, adj_166, adj_170, adj_179);
        // adj: f_total = n * (fn + fd) + ft                                                      <L 1073>
        wp::adj_mul(var_171, var_177, adj_171, adj_177, adj_178);
        wp::adj_min(var_173, var_176, adj_173, adj_176, adj_177);
        wp::adj_sub(var_11, var_175, adj_11, adj_175, adj_176);
        wp::adj_mul(var_97, var_174, adj_97, adj_174, adj_175);
        wp::adj_add(var_166, var_170, adj_166, adj_170, adj_174);
        wp::adj_mul(var_96, var_172, adj_96, adj_172, adj_173);
        wp::adj_length(var_165, var_172, adj_165, adj_172);
        wp::adj_normalize(var_165, var_171, adj_165, adj_171);
        // adj: ft = wp.normalize(vt) * wp.min(kf * wp.length(vt), 0.0 - mu * (fn + fd))          <L 1070>
        wp::adj_mul(var_168, var_169, adj_168, adj_169, adj_170);
        wp::adj_step(var_140, adj_140, adj_169);
        wp::adj_mul(var_167, var_95, adj_167, adj_95, adj_168);
        wp::adj_min(var_163, var_11, adj_163, adj_11, adj_167);
        // adj: fd = wp.min(vn, 0.0) * kd * wp.step(d)                                            <L 1054>
        wp::adj_mul(var_140, var_94, adj_140, adj_94, adj_166);
        // adj: fn = d * ke                                                                       <L 1051>
        wp::adj_sub(var_162, var_164, adj_162, adj_164, adj_165);
        wp::adj_mul(var_106, var_163, adj_106, adj_163, adj_164);
        // adj: vt = v - n * vn                                                                   <L 1048>
        wp::adj_dot(var_106, var_162, adj_106, adj_162, adj_163);
        // adj: vn = wp.dot(n, v)                                                                 <L 1047>
        wp::adj_sub(var_152, var_161, adj_152, adj_161, adj_162);
        // adj: v = bv_a - bv_b                                                                   <L 1042>
        wp::adj_select(var_153, var_143, var_160, adj_153, adj_143, adj_160, adj_161);
        if (var_153) {
            wp::adj_add(var_158, var_159, adj_158, adj_159, adj_160);
            wp::adj_cross(var_157, var_137, adj_157, adj_137, adj_159);
            // adj: bv_b = body_v_b + wp.cross(body_w_b, r_b)                                     <L 1039>
            wp::adj_spatial_bottom(var_156, adj_156, adj_158);
            // adj: body_v_b = wp.spatial_bottom(body_v_s_b)                                      <L 1038>
            wp::adj_spatial_top(var_156, adj_156, adj_157);
            // adj: body_w_b = wp.spatial_top(body_v_s_b)                                         <L 1037>
            wp::adj_copy(var_155, adj_154, adj_156);
            wp::adj_load(var_154, adj_154, adj_155);
            wp::adj_address(var_body_qd, var_103, adj_body_qd, adj_103, adj_154);
            // adj: body_v_s_b = body_qd[body_b]                                                  <L 1036>
        }
        // adj: if body_b >= 0:                                                                   <L 1035>
        wp::adj_select(var_144, var_142, var_151, adj_144, adj_142, adj_151, adj_152);
        if (var_144) {
            wp::adj_add(var_149, var_150, adj_149, adj_150, adj_151);
            wp::adj_cross(var_148, var_124, adj_148, adj_124, adj_150);
            // adj: bv_a = body_v_a + wp.cross(body_w_a, r_a)                                     <L 1033>
            wp::adj_spatial_bottom(var_147, adj_147, adj_149);
            // adj: body_v_a = wp.spatial_bottom(body_v_s_a)                                      <L 1032>
            wp::adj_spatial_top(var_147, adj_147, adj_148);
            // adj: body_w_a = wp.spatial_top(body_v_s_a)                                         <L 1031>
            wp::adj_copy(var_146, adj_145, adj_147);
            wp::adj_load(var_145, adj_145, adj_146);
            wp::adj_address(var_body_qd, var_100, adj_body_qd, adj_100, adj_145);
            // adj: body_v_s_a = body_qd[body_a]                                                  <L 1030>
        }
        // adj: if body_a >= 0:                                                                   <L 1029>
        wp::adj_vec_t(var_11, adj_11, adj_143);
        // adj: bv_b = wp.vec3(0.0)                                                               <L 1028>
        wp::adj_vec_t(var_11, adj_11, adj_142);
        // adj: bv_a = wp.vec3(0.0)                                                               <L 1027>
        if (var_141) {
            label2:;
            // adj: return                                                                        <L 1024>
        }
        // adj: if d >= 0.0:                                                                      <L 1023>
        wp::adj_dot(var_106, var_139, adj_106, adj_139, adj_140);
        wp::adj_sub(var_125, var_138, adj_125, adj_138, adj_139);
        // adj: d = wp.dot(n, bx_a - bx_b)                                                        <L 1021>
        wp::adj_select(var_126, var_112, var_135, adj_126, adj_112, adj_135, adj_138);
        if (var_126) {
            wp::adj_sub(var_135, var_136, adj_135, adj_136, adj_137);
            wp::adj_transform_point(var_129, var_132, adj_129, adj_132, adj_136);
            // adj: r_b = bx_b - wp.transform_point(X_wb_b, X_com_b)                              <L 1019>
            wp::adj_add(var_133, var_134, adj_133, adj_134, adj_135);
            wp::adj_mul(var_84, var_106, adj_84, adj_106, adj_134);
            wp::adj_transform_point(var_129, var_112, adj_129, adj_112, adj_133);
            // adj: bx_b = wp.transform_point(X_wb_b, bx_b) + thickness_b * n                     <L 1018>
            wp::adj_copy(var_131, adj_130, adj_132);
            wp::adj_load(var_130, adj_130, adj_131);
            wp::adj_address(var_body_com, var_103, adj_body_com, adj_103, adj_130);
            // adj: X_com_b = body_com[body_b]                                                    <L 1017>
            wp::adj_copy(var_128, adj_127, adj_129);
            wp::adj_load(var_127, adj_127, adj_128);
            wp::adj_address(var_body_q, var_103, adj_body_q, adj_103, adj_127);
            // adj: X_wb_b = body_q[body_b]                                                       <L 1016>
        }
        // adj: if body_b >= 0:                                                                   <L 1015>
        wp::adj_select(var_113, var_109, var_122, adj_113, adj_109, adj_122, adj_125);
        if (var_113) {
            wp::adj_sub(var_122, var_123, adj_122, adj_123, adj_124);
            wp::adj_transform_point(var_116, var_119, adj_116, adj_119, adj_123);
            // adj: r_a = bx_a - wp.transform_point(X_wb_a, X_com_a)                              <L 1013>
            wp::adj_sub(var_120, var_121, adj_120, adj_121, adj_122);
            wp::adj_mul(var_51, var_106, adj_51, adj_106, adj_121);
            wp::adj_transform_point(var_116, var_109, adj_116, adj_109, adj_120);
            // adj: bx_a = wp.transform_point(X_wb_a, bx_a) - thickness_a * n                     <L 1012>
            wp::adj_copy(var_118, adj_117, adj_119);
            wp::adj_load(var_117, adj_117, adj_118);
            wp::adj_address(var_body_com, var_100, adj_body_com, adj_100, adj_117);
            // adj: X_com_a = body_com[body_a]                                                    <L 1011>
            wp::adj_copy(var_115, adj_114, adj_116);
            wp::adj_load(var_114, adj_114, adj_115);
            wp::adj_address(var_body_q, var_100, adj_body_q, adj_100, adj_114);
            // adj: X_wb_a = body_q[body_a]                                                       <L 1010>
        }
        // adj: if body_a >= 0:                                                                   <L 1009>
        wp::adj_copy(var_111, adj_110, adj_112);
        wp::adj_load(var_110, adj_110, adj_111);
        wp::adj_address(var_contact_point1, var_0, adj_contact_point1, adj_0, adj_110);
        // adj: bx_b = contact_point1[tid]                                                        <L 1008>
        wp::adj_copy(var_108, adj_107, adj_109);
        wp::adj_load(var_107, adj_107, adj_108);
        wp::adj_address(var_contact_point0, var_0, adj_contact_point0, adj_0, adj_107);
        // adj: bx_a = contact_point0[tid]                                                        <L 1007>
        wp::adj_copy(var_105, adj_104, adj_106);
        wp::adj_load(var_104, adj_104, adj_105);
        wp::adj_address(var_contact_normal, var_0, adj_contact_normal, adj_0, adj_104);
        // adj: n = contact_normal[tid]                                                           <L 1006>
        wp::adj_copy(var_102, adj_101, adj_103);
        wp::adj_load(var_101, adj_101, adj_102);
        wp::adj_address(var_contact_body1, var_0, adj_contact_body1, adj_0, adj_101);
        // adj: body_b = contact_body1[tid]                                                       <L 1003>
        wp::adj_copy(var_99, adj_98, adj_100);
        wp::adj_load(var_98, adj_98, adj_99);
        wp::adj_address(var_contact_body0, var_0, adj_contact_body0, adj_0, adj_98);
        // adj: body_a = contact_body0[tid]                                                       <L 1002>
        wp::adj_select(var_85, var_82, var_93, adj_85, adj_82, adj_93, adj_97);
        wp::adj_select(var_85, var_81, var_91, adj_85, adj_81, adj_91, adj_96);
        wp::adj_select(var_85, var_80, var_89, adj_85, adj_80, adj_89, adj_95);
        wp::adj_select(var_85, var_79, var_87, adj_85, adj_79, adj_87, adj_94);
        if (var_85) {
            wp::adj_div(var_82, var_92, var_93, adj_82, adj_92, adj_93);
            wp::adj_float(var_83, adj_83, adj_92);
            // adj: mu = mu / float(mat_nonzero)                                                  <L 1000>
            wp::adj_div(var_81, var_90, var_91, adj_81, adj_90, adj_91);
            wp::adj_float(var_83, adj_83, adj_90);
            // adj: kf = kf / float(mat_nonzero)                                                  <L 999>
            wp::adj_div(var_80, var_88, var_89, adj_80, adj_88, adj_89);
            wp::adj_float(var_83, adj_83, adj_88);
            // adj: kd = kd / float(mat_nonzero)                                                  <L 998>
            wp::adj_div(var_79, var_86, var_87, adj_79, adj_86, adj_87);
            wp::adj_float(var_83, adj_83, adj_86);
            // adj: ke = ke / float(mat_nonzero)                                                  <L 997>
        }
        // adj: if mat_nonzero > 0:                                                               <L 996>
        wp::adj_select(var_52, var_11, var_78, adj_52, adj_11, adj_78, adj_84);
        wp::adj_select(var_52, var_50, var_53, adj_52, adj_50, adj_53, adj_83);
        wp::adj_select(var_52, var_49, var_73, adj_52, adj_49, adj_73, adj_82);
        wp::adj_select(var_52, var_48, var_68, adj_52, adj_48, adj_68, adj_81);
        wp::adj_select(var_52, var_47, var_63, adj_52, adj_47, adj_63, adj_80);
        wp::adj_select(var_52, var_46, var_58, adj_52, adj_46, adj_58, adj_79);
        if (var_52) {
            wp::adj_copy(var_77, adj_76, adj_78);
            wp::adj_load(var_76, adj_76, adj_77);
            wp::adj_address(var_75, var_17, adj_74, adj_17, adj_76);
            wp::adj_load(var_74, adj_74, adj_75);
            adj_geo.thickness = adj_74;
            // adj: thickness_b = geo.thickness[shape_b]                                          <L 995>
            wp::adj_add(var_49, var_72, adj_49, adj_71, adj_73);
            wp::adj_load(var_71, adj_71, adj_72);
            wp::adj_address(var_70, var_17, adj_69, adj_17, adj_71);
            wp::adj_load(var_69, adj_69, adj_70);
            adj_shape_materials.mu = adj_69;
            // adj: mu += shape_materials.mu[shape_b]                                             <L 994>
            wp::adj_add(var_48, var_67, adj_48, adj_66, adj_68);
            wp::adj_load(var_66, adj_66, adj_67);
            wp::adj_address(var_65, var_17, adj_64, adj_17, adj_66);
            wp::adj_load(var_64, adj_64, adj_65);
            adj_shape_materials.kf = adj_64;
            // adj: kf += shape_materials.kf[shape_b]                                             <L 993>
            wp::adj_add(var_47, var_62, adj_47, adj_61, adj_63);
            wp::adj_load(var_61, adj_61, adj_62);
            wp::adj_address(var_60, var_17, adj_59, adj_17, adj_61);
            wp::adj_load(var_59, adj_59, adj_60);
            adj_shape_materials.kd = adj_59;
            // adj: kd += shape_materials.kd[shape_b]                                             <L 992>
            wp::adj_add(var_46, var_57, adj_46, adj_56, adj_58);
            wp::adj_load(var_56, adj_56, adj_57);
            wp::adj_address(var_55, var_17, adj_54, adj_17, adj_56);
            wp::adj_load(var_54, adj_54, adj_55);
            adj_shape_materials.ke = adj_54;
            // adj: ke += shape_materials.ke[shape_b]                                             <L 991>
            wp::adj_add(var_50, var_19, adj_50, adj_19, adj_53);
            // adj: mat_nonzero += 1                                                              <L 990>
        }
        // adj: if shape_b >= 0:                                                                  <L 989>
        wp::adj_select(var_18, var_11, var_45, adj_18, adj_11, adj_45, adj_51);
        wp::adj_select(var_18, var_6, var_20, adj_18, adj_6, adj_20, adj_50);
        wp::adj_select(var_18, var_11, var_40, adj_18, adj_11, adj_40, adj_49);
        wp::adj_select(var_18, var_11, var_35, adj_18, adj_11, adj_35, adj_48);
        wp::adj_select(var_18, var_11, var_30, adj_18, adj_11, adj_30, adj_47);
        wp::adj_select(var_18, var_11, var_25, adj_18, adj_11, adj_25, adj_46);
        if (var_18) {
            wp::adj_copy(var_44, adj_43, adj_45);
            wp::adj_load(var_43, adj_43, adj_44);
            wp::adj_address(var_42, var_14, adj_41, adj_14, adj_43);
            wp::adj_load(var_41, adj_41, adj_42);
            adj_geo.thickness = adj_41;
            // adj: thickness_a = geo.thickness[shape_a]                                          <L 988>
            wp::adj_add(var_11, var_39, adj_11, adj_38, adj_40);
            wp::adj_load(var_38, adj_38, adj_39);
            wp::adj_address(var_37, var_14, adj_36, adj_14, adj_38);
            wp::adj_load(var_36, adj_36, adj_37);
            adj_shape_materials.mu = adj_36;
            // adj: mu += shape_materials.mu[shape_a]                                             <L 987>
            wp::adj_add(var_11, var_34, adj_11, adj_33, adj_35);
            wp::adj_load(var_33, adj_33, adj_34);
            wp::adj_address(var_32, var_14, adj_31, adj_14, adj_33);
            wp::adj_load(var_31, adj_31, adj_32);
            adj_shape_materials.kf = adj_31;
            // adj: kf += shape_materials.kf[shape_a]                                             <L 986>
            wp::adj_add(var_11, var_29, adj_11, adj_28, adj_30);
            wp::adj_load(var_28, adj_28, adj_29);
            wp::adj_address(var_27, var_14, adj_26, adj_14, adj_28);
            wp::adj_load(var_26, adj_26, adj_27);
            adj_shape_materials.kd = adj_26;
            // adj: kd += shape_materials.kd[shape_a]                                             <L 985>
            wp::adj_add(var_11, var_24, adj_11, adj_23, adj_25);
            wp::adj_load(var_23, adj_23, adj_24);
            wp::adj_address(var_22, var_14, adj_21, adj_14, adj_23);
            wp::adj_load(var_21, adj_21, adj_22);
            adj_shape_materials.ke = adj_21;
            // adj: ke += shape_materials.ke[shape_a]                                             <L 984>
            wp::adj_add(var_6, var_19, adj_6, adj_19, adj_20);
            // adj: mat_nonzero += 1                                                              <L 983>
        }
        // adj: if shape_a >= 0:                                                                  <L 982>
        wp::adj_copy(var_16, adj_15, adj_17);
        wp::adj_load(var_15, adj_15, adj_16);
        wp::adj_address(var_contact_shape1, var_0, adj_contact_shape1, adj_0, adj_15);
        // adj: shape_b = contact_shape1[tid]                                                     <L 981>
        wp::adj_copy(var_13, adj_12, adj_14);
        wp::adj_load(var_12, adj_12, adj_13);
        wp::adj_address(var_contact_shape0, var_0, adj_contact_shape0, adj_0, adj_12);
        // adj: shape_a = contact_shape0[tid]                                                     <L 980>
        // adj: thickness_b = 0.0                                                                 <L 979>
        // adj: thickness_a = 0.0                                                                 <L 978>
        // adj: mat_nonzero = 0                                                                   <L 977>
        // adj: mu = 0.0  # coulomb friction                                                      <L 976>
        // adj: kf = 0.0  # friction coefficient                                                  <L 975>
        // adj: kd = 0.0  # damping coefficient                                                   <L 974>
        // adj: ke = 0.0  # restitution coefficient                                               <L 973>
        if (var_10) {
            label1:;
            // adj: return                                                                        <L 970>
        }
        // adj: if tid >= count:                                                                  <L 969>
        wp::adj_copy(var_8, adj_7, adj_9);
        wp::adj_load(var_7, adj_7, adj_8);
        wp::adj_address(var_contact_count, var_6, adj_contact_count, adj_6, adj_7);
        // adj: count = contact_count[0]                                                          <L 968>
        if (var_3) {
            label0:;
            // adj: return                                                                        <L 966>
        }
        wp::adj_load(var_2, adj_2, adj_5);
        wp::adj_load(var_1, adj_1, adj_4);
        wp::adj_address(var_contact_shape1, var_0, adj_contact_shape1, adj_0, adj_2);
        wp::adj_address(var_contact_shape0, var_0, adj_contact_shape0, adj_0, adj_1);
        // adj: if contact_shape0[tid] == contact_shape1[tid]:                                    <L 965>
        // adj: tid = wp.tid()                                                                    <L 964>
        // adj: def eval_rigid_contacts(                                                          <L 947>
        continue;
    }
}



extern "C" __global__ void eval_body_joints_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::transform_t<wp::float32>> var_body_q,
    wp::array_t<wp::vec_t<6,wp::float32>> var_body_qd,
    wp::array_t<wp::vec_t<3,wp::float32>> var_body_com,
    wp::array_t<wp::int32> var_joint_q_start,
    wp::array_t<wp::int32> var_joint_qd_start,
    wp::array_t<wp::int32> var_joint_type,
    wp::array_t<wp::int32> var_joint_enabled,
    wp::array_t<wp::int32> var_joint_child,
    wp::array_t<wp::int32> var_joint_parent,
    wp::array_t<wp::transform_t<wp::float32>> var_joint_X_p,
    wp::array_t<wp::transform_t<wp::float32>> var_joint_X_c,
    wp::array_t<wp::vec_t<3,wp::float32>> var_joint_axis,
    wp::array_t<wp::int32> var_joint_axis_start,
    wp::array_t<wp::int32> var_joint_axis_dim,
    wp::array_t<wp::float32> var_joint_target,
    wp::array_t<wp::float32> var_joint_act,
    wp::array_t<wp::float32> var_joint_target_ke,
    wp::array_t<wp::float32> var_joint_target_kd,
    wp::array_t<wp::float32> var_joint_limit_lower,
    wp::array_t<wp::float32> var_joint_limit_upper,
    wp::array_t<wp::float32> var_joint_limit_ke,
    wp::array_t<wp::float32> var_joint_limit_kd,
    wp::float32 var_joint_attach_ke,
    wp::float32 var_joint_attach_kd,
    wp::array_t<wp::vec_t<6,wp::float32>> var_body_f)
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
        wp::int32* var_4;
        const wp::int32 var_5 = 0;
        bool var_6;
        wp::int32 var_7;
        const wp::int32 var_8 = 4;
        bool var_9;
        bool var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::transform_t<wp::float32>* var_17;
        wp::transform_t<wp::float32> var_18;
        wp::transform_t<wp::float32> var_19;
        wp::transform_t<wp::float32>* var_20;
        wp::transform_t<wp::float32> var_21;
        wp::transform_t<wp::float32> var_22;
        wp::transform_t<wp::float32> var_23;
        wp::vec_t<3,wp::float32> var_24;
        wp::vec_t<3,wp::float32> var_25;
        wp::vec_t<3,wp::float32> var_26;
        bool var_27;
        wp::transform_t<wp::float32>* var_28;
        wp::transform_t<wp::float32> var_29;
        wp::transform_t<wp::float32> var_30;
        wp::vec_t<3,wp::float32> var_31;
        wp::transform_t<wp::float32>* var_32;
        wp::vec_t<3,wp::float32>* var_33;
        wp::transform_t<wp::float32> var_34;
        wp::vec_t<3,wp::float32> var_35;
        wp::vec_t<3,wp::float32> var_36;
        wp::vec_t<3,wp::float32> var_37;
        wp::vec_t<6,wp::float32>* var_38;
        wp::vec_t<6,wp::float32> var_39;
        wp::vec_t<6,wp::float32> var_40;
        wp::vec_t<3,wp::float32> var_41;
        wp::vec_t<3,wp::float32> var_42;
        wp::vec_t<3,wp::float32> var_43;
        wp::vec_t<3,wp::float32> var_44;
        wp::transform_t<wp::float32> var_45;
        wp::vec_t<3,wp::float32> var_46;
        wp::vec_t<3,wp::float32> var_47;
        wp::vec_t<3,wp::float32> var_48;
        wp::transform_t<wp::float32>* var_49;
        wp::transform_t<wp::float32> var_50;
        wp::transform_t<wp::float32> var_51;
        wp::vec_t<3,wp::float32> var_52;
        wp::transform_t<wp::float32>* var_53;
        wp::vec_t<3,wp::float32>* var_54;
        wp::transform_t<wp::float32> var_55;
        wp::vec_t<3,wp::float32> var_56;
        wp::vec_t<3,wp::float32> var_57;
        wp::vec_t<3,wp::float32> var_58;
        wp::vec_t<6,wp::float32>* var_59;
        wp::vec_t<6,wp::float32> var_60;
        wp::vec_t<6,wp::float32> var_61;
        wp::vec_t<3,wp::float32> var_62;
        wp::vec_t<3,wp::float32> var_63;
        wp::vec_t<3,wp::float32> var_64;
        wp::vec_t<3,wp::float32> var_65;
        wp::int32* var_66;
        wp::int32 var_67;
        wp::int32 var_68;
        wp::int32* var_69;
        wp::int32 var_70;
        wp::int32 var_71;
        wp::int32* var_72;
        wp::int32 var_73;
        wp::int32 var_74;
        wp::float32* var_75;
        wp::float32 var_76;
        wp::float32 var_77;
        wp::float32* var_78;
        wp::float32 var_79;
        wp::float32 var_80;
        wp::float32* var_81;
        wp::float32 var_82;
        wp::float32 var_83;
        wp::float32* var_84;
        wp::float32 var_85;
        wp::float32 var_86;
        wp::float32* var_87;
        wp::float32 var_88;
        wp::float32 var_89;
        wp::float32* var_90;
        wp::float32 var_91;
        wp::float32 var_92;
        wp::float32* var_93;
        wp::float32 var_94;
        wp::float32 var_95;
        wp::float32* var_96;
        wp::float32 var_97;
        wp::float32 var_98;
        wp::vec_t<3,wp::float32> var_99;
        wp::vec_t<3,wp::float32> var_100;
        wp::quat_t<wp::float32> var_101;
        wp::quat_t<wp::float32> var_102;
        wp::vec_t<3,wp::float32> var_103;
        wp::quat_t<wp::float32> var_104;
        wp::quat_t<wp::float32> var_105;
        wp::vec_t<3,wp::float32> var_106;
        wp::vec_t<3,wp::float32> var_107;
        wp::vec_t<3,wp::float32> var_108;
        wp::vec_t<3,wp::float32> var_109;
        const wp::float32 var_110 = 0.01;
        const wp::int32 var_111 = 3;
        bool var_112;
        wp::float32 var_113;
        const wp::int32 var_114 = 1;
        wp::float32 var_115;
        const wp::int32 var_116 = 2;
        wp::float32 var_117;
        wp::vec_t<3,wp::float32> var_118;
        wp::vec_t<3,wp::float32> var_119;
        const wp::int32 var_120 = 3;
        wp::float32 var_121;
        wp::float32 var_122;
        wp::vec_t<3,wp::float32> var_123;
        const wp::float32 var_124 = 2.0;
        wp::vec_t<3,wp::float32> var_125;
        wp::vec_t<3,wp::float32> var_126;
        wp::vec_t<3,wp::float32> var_127;
        wp::vec_t<3,wp::float32> var_128;
        wp::vec_t<3,wp::float32> var_129;
        wp::vec_t<3,wp::float32> var_130;
        wp::vec_t<3,wp::float32> var_131;
        wp::vec_t<3,wp::float32> var_132;
        wp::vec_t<3,wp::float32> var_133;
        wp::vec_t<3,wp::float32> var_134;
        wp::vec_t<3,wp::float32> var_135;
        wp::vec_t<3,wp::float32> var_136;
        wp::vec_t<3,wp::float32> var_137;
        const wp::int32 var_138 = 0;
        bool var_139;
        wp::vec_t<3,wp::float32>* var_140;
        wp::vec_t<3,wp::float32> var_141;
        wp::vec_t<3,wp::float32> var_142;
        wp::vec_t<3,wp::float32> var_143;
        wp::float32 var_144;
        wp::float32 var_145;
        wp::vec_t<3,wp::float32> var_146;
        wp::float32 var_147;
        wp::float32 var_148;
        wp::float32 var_149;
        wp::vec_t<3,wp::float32> var_150;
        wp::vec_t<3,wp::float32> var_151;
        wp::float32 var_152;
        wp::float32 var_153;
        wp::vec_t<3,wp::float32> var_154;
        wp::vec_t<3,wp::float32> var_155;
        wp::vec_t<3,wp::float32> var_156;
        wp::vec_t<3,wp::float32> var_157;
        wp::vec_t<3,wp::float32> var_158;
        wp::vec_t<3,wp::float32> var_159;
        wp::vec_t<3,wp::float32> var_160;
        wp::vec_t<3,wp::float32> var_161;
        wp::vec_t<3,wp::float32> var_162;
        wp::vec_t<3,wp::float32> var_163;
        wp::vec_t<3,wp::float32> var_164;
        wp::vec_t<3,wp::float32> var_165;
        wp::vec_t<3,wp::float32> var_166;
        wp::vec_t<3,wp::float32> var_167;
        wp::vec_t<3,wp::float32> var_168;
        wp::vec_t<3,wp::float32> var_169;
        wp::vec_t<3,wp::float32> var_170;
        wp::vec_t<3,wp::float32> var_171;
        wp::vec_t<3,wp::float32> var_172;
        const wp::int32 var_173 = 1;
        bool var_174;
        wp::vec_t<3,wp::float32>* var_175;
        wp::vec_t<3,wp::float32> var_176;
        wp::vec_t<3,wp::float32> var_177;
        wp::vec_t<3,wp::float32> var_178;
        wp::vec_t<3,wp::float32> var_179;
        wp::quat_t<wp::float32> var_180;
        wp::float32 var_181;
        wp::float32 var_182;
        wp::float32 var_183;
        wp::float32 var_184;
        wp::float32 var_185;
        wp::float32 var_186;
        wp::vec_t<3,wp::float32> var_187;
        wp::float32 var_188;
        wp::float32 var_189;
        wp::float32 var_190;
        wp::float32 var_191;
        wp::vec_t<3,wp::float32> var_192;
        wp::vec_t<3,wp::float32> var_193;
        wp::vec_t<3,wp::float32> var_194;
        wp::vec_t<3,wp::float32> var_195;
        wp::vec_t<3,wp::float32> var_196;
        wp::vec_t<3,wp::float32> var_197;
        wp::vec_t<3,wp::float32> var_198;
        wp::vec_t<3,wp::float32> var_199;
        wp::vec_t<3,wp::float32> var_200;
        wp::vec_t<3,wp::float32> var_201;
        wp::vec_t<3,wp::float32> var_202;
        wp::vec_t<3,wp::float32> var_203;
        wp::vec_t<3,wp::float32> var_204;
        wp::vec_t<3,wp::float32> var_205;
        wp::vec_t<3,wp::float32> var_206;
        wp::vec_t<3,wp::float32> var_207;
        wp::vec_t<3,wp::float32> var_208;
        wp::float32 var_209;
        wp::float32 var_210;
        const wp::int32 var_211 = 2;
        bool var_212;
        wp::float32 var_213;
        wp::float32 var_214;
        wp::float32 var_215;
        wp::vec_t<3,wp::float32> var_216;
        wp::vec_t<3,wp::float32> var_217;
        wp::float32 var_218;
        wp::float32 var_219;
        wp::vec_t<3,wp::float32> var_220;
        wp::vec_t<3,wp::float32> var_221;
        wp::vec_t<3,wp::float32> var_222;
        wp::vec_t<3,wp::float32> var_223;
        wp::vec_t<3,wp::float32> var_224;
        wp::vec_t<3,wp::float32> var_225;
        wp::vec_t<3,wp::float32> var_226;
        wp::vec_t<3,wp::float32> var_227;
        wp::vec_t<3,wp::float32> var_228;
        wp::vec_t<3,wp::float32> var_229;
        wp::vec_t<3,wp::float32> var_230;
        wp::vec_t<3,wp::float32> var_231;
        wp::vec_t<3,wp::float32> var_232;
        wp::vec_t<3,wp::float32> var_233;
        const wp::int32 var_234 = 5;
        bool var_235;
        wp::quat_t<wp::float32> var_236;
        wp::quat_t<wp::float32> var_237;
        wp::vec_t<3,wp::float32> var_238;
        const wp::float32 var_239 = 1.0;
        const wp::float32 var_240 = 0.0;
        wp::vec_t<3,wp::float32> var_241;
        wp::float32 var_242;
        wp::quat_t<wp::float32> var_243;
        wp::vec_t<3,wp::float32> var_244;
        wp::vec_t<3,wp::float32> var_245;
        wp::float32 var_246;
        wp::quat_t<wp::float32> var_247;
        wp::quat_t<wp::float32> var_248;
        wp::vec_t<3,wp::float32> var_249;
        wp::vec_t<3,wp::float32> var_250;
        wp::float32 var_251;
        wp::quat_t<wp::float32> var_252;
        wp::quat_t<wp::float32> var_253;
        wp::vec_t<3,wp::float32> var_254;
        wp::vec_t<3,wp::float32> var_255;
        wp::vec_t<3,wp::float32> var_256;
        wp::vec_t<3,wp::float32> var_257;
        wp::float32 var_258;
        wp::float32 var_259;
        wp::int32 var_260;
        wp::float32* var_261;
        wp::int32 var_262;
        wp::float32* var_263;
        wp::int32 var_264;
        wp::float32* var_265;
        wp::int32 var_266;
        wp::float32* var_267;
        wp::int32 var_268;
        wp::float32* var_269;
        wp::int32 var_270;
        wp::float32* var_271;
        wp::int32 var_272;
        wp::float32* var_273;
        wp::int32 var_274;
        wp::float32* var_275;
        wp::float32 var_276;
        wp::float32 var_277;
        wp::float32 var_278;
        wp::float32 var_279;
        wp::float32 var_280;
        wp::float32 var_281;
        wp::float32 var_282;
        wp::float32 var_283;
        wp::vec_t<3,wp::float32> var_284;
        wp::vec_t<3,wp::float32> var_285;
        wp::float32 var_286;
        wp::float32 var_287;
        wp::int32 var_288;
        wp::float32* var_289;
        wp::int32 var_290;
        wp::float32* var_291;
        wp::int32 var_292;
        wp::float32* var_293;
        wp::int32 var_294;
        wp::float32* var_295;
        wp::int32 var_296;
        wp::float32* var_297;
        wp::int32 var_298;
        wp::float32* var_299;
        wp::int32 var_300;
        wp::float32* var_301;
        wp::int32 var_302;
        wp::float32* var_303;
        wp::float32 var_304;
        wp::float32 var_305;
        wp::float32 var_306;
        wp::float32 var_307;
        wp::float32 var_308;
        wp::float32 var_309;
        wp::float32 var_310;
        wp::float32 var_311;
        wp::vec_t<3,wp::float32> var_312;
        wp::vec_t<3,wp::float32> var_313;
        wp::float32 var_314;
        wp::float32 var_315;
        wp::int32 var_316;
        wp::float32* var_317;
        wp::int32 var_318;
        wp::float32* var_319;
        wp::int32 var_320;
        wp::float32* var_321;
        wp::int32 var_322;
        wp::float32* var_323;
        wp::int32 var_324;
        wp::float32* var_325;
        wp::int32 var_326;
        wp::float32* var_327;
        wp::int32 var_328;
        wp::float32* var_329;
        wp::int32 var_330;
        wp::float32* var_331;
        wp::float32 var_332;
        wp::float32 var_333;
        wp::float32 var_334;
        wp::float32 var_335;
        wp::float32 var_336;
        wp::float32 var_337;
        wp::float32 var_338;
        wp::float32 var_339;
        wp::vec_t<3,wp::float32> var_340;
        wp::vec_t<3,wp::float32> var_341;
        wp::vec_t<3,wp::float32> var_342;
        wp::vec_t<3,wp::float32> var_343;
        wp::vec_t<3,wp::float32> var_344;
        wp::vec_t<3,wp::float32> var_345;
        wp::vec_t<3,wp::float32> var_346;
        wp::vec_t<3,wp::float32> var_347;
        const wp::int32 var_348 = 6;
        bool var_349;
        wp::quat_t<wp::float32> var_350;
        wp::quat_t<wp::float32> var_351;
        wp::vec_t<3,wp::float32> var_352;
        wp::vec_t<3,wp::float32> var_353;
        wp::float32 var_354;
        wp::quat_t<wp::float32> var_355;
        wp::vec_t<3,wp::float32> var_356;
        wp::vec_t<3,wp::float32> var_357;
        wp::float32 var_358;
        wp::quat_t<wp::float32> var_359;
        wp::quat_t<wp::float32> var_360;
        wp::vec_t<3,wp::float32> var_361;
        wp::vec_t<3,wp::float32> var_362;
        wp::float32 var_363;
        wp::quat_t<wp::float32> var_364;
        wp::quat_t<wp::float32> var_365;
        wp::vec_t<3,wp::float32> var_366;
        wp::vec_t<3,wp::float32> var_367;
        wp::vec_t<3,wp::float32> var_368;
        wp::vec_t<3,wp::float32> var_369;
        wp::float32 var_370;
        wp::float32 var_371;
        wp::int32 var_372;
        wp::float32* var_373;
        wp::int32 var_374;
        wp::float32* var_375;
        wp::int32 var_376;
        wp::float32* var_377;
        wp::int32 var_378;
        wp::float32* var_379;
        wp::int32 var_380;
        wp::float32* var_381;
        wp::int32 var_382;
        wp::float32* var_383;
        wp::int32 var_384;
        wp::float32* var_385;
        wp::int32 var_386;
        wp::float32* var_387;
        wp::float32 var_388;
        wp::float32 var_389;
        wp::float32 var_390;
        wp::float32 var_391;
        wp::float32 var_392;
        wp::float32 var_393;
        wp::float32 var_394;
        wp::float32 var_395;
        wp::vec_t<3,wp::float32> var_396;
        wp::vec_t<3,wp::float32> var_397;
        wp::float32 var_398;
        wp::float32 var_399;
        wp::int32 var_400;
        wp::float32* var_401;
        wp::int32 var_402;
        wp::float32* var_403;
        wp::int32 var_404;
        wp::float32* var_405;
        wp::int32 var_406;
        wp::float32* var_407;
        wp::int32 var_408;
        wp::float32* var_409;
        wp::int32 var_410;
        wp::float32* var_411;
        wp::int32 var_412;
        wp::float32* var_413;
        wp::int32 var_414;
        wp::float32* var_415;
        wp::float32 var_416;
        wp::float32 var_417;
        wp::float32 var_418;
        wp::float32 var_419;
        wp::float32 var_420;
        wp::float32 var_421;
        wp::float32 var_422;
        wp::float32 var_423;
        wp::vec_t<3,wp::float32> var_424;
        wp::vec_t<3,wp::float32> var_425;
        wp::float32 var_426;
        wp::float32 var_427;
        wp::float32 var_428;
        wp::vec_t<3,wp::float32> var_429;
        wp::vec_t<3,wp::float32> var_430;
        wp::vec_t<3,wp::float32> var_431;
        wp::vec_t<3,wp::float32> var_432;
        wp::vec_t<3,wp::float32> var_433;
        wp::vec_t<3,wp::float32> var_434;
        wp::vec_t<3,wp::float32> var_435;
        wp::vec_t<3,wp::float32> var_436;
        wp::quat_t<wp::float32> var_437;
        wp::vec_t<3,wp::float32> var_438;
        wp::vec_t<3,wp::float32> var_439;
        wp::quat_t<wp::float32> var_440;
        wp::vec_t<3,wp::float32> var_441;
        wp::quat_t<wp::float32> var_442;
        wp::vec_t<3,wp::float32> var_443;
        wp::quat_t<wp::float32> var_444;
        wp::quat_t<wp::float32> var_445;
        bool var_446;
        wp::vec_t<3,wp::float32> var_447;
        wp::vec_t<3,wp::float32> var_448;
        wp::vec_t<6,wp::float32> var_449;
        wp::vec_t<6,wp::float32> var_450;
        wp::vec_t<3,wp::float32> var_451;
        wp::vec_t<3,wp::float32> var_452;
        wp::vec_t<6,wp::float32> var_453;
        wp::vec_t<6,wp::float32> var_454;
        //---------
        // forward
        // def eval_body_joints(                                                                  <L 1115>
        // tid = wp.tid()                                                                         <L 1142>
        var_0 = builtin_tid1d();
        // type = joint_type[tid]                                                                 <L 1143>
        var_1 = wp::address(var_joint_type, var_0);
        var_2 = wp::load(var_1);
        var_3 = wp::copy(var_2);
        // if joint_enabled[tid] == 0 or type == fs5Model.JOINT_FREE:                             <L 1146>
        var_4 = wp::address(var_joint_enabled, var_0);
        var_7 = wp::load(var_4);
        var_6 = (var_7 == var_5);
        var_9 = (var_3 == var_8);
        var_10 = var_6 || var_9;
        if (var_10) {
            // return                                                                             <L 1147>
            return;
        }
        // c_child = joint_child[tid]                                                             <L 1149>
        var_11 = wp::address(var_joint_child, var_0);
        var_12 = wp::load(var_11);
        var_13 = wp::copy(var_12);
        // c_parent = joint_parent[tid]                                                           <L 1150>
        var_14 = wp::address(var_joint_parent, var_0);
        var_15 = wp::load(var_14);
        var_16 = wp::copy(var_15);
        // X_pj = joint_X_p[tid]                                                                  <L 1152>
        var_17 = wp::address(var_joint_X_p, var_0);
        var_18 = wp::load(var_17);
        var_19 = wp::copy(var_18);
        // X_cj = joint_X_c[tid]                                                                  <L 1153>
        var_20 = wp::address(var_joint_X_c, var_0);
        var_21 = wp::load(var_20);
        var_22 = wp::copy(var_21);
        // X_wp = X_pj                                                                            <L 1155>
        var_23 = wp::copy(var_19);
        // r_p = wp.vec3()                                                                        <L 1156>
        var_24 = wp::vec_t<3,wp::float32>();
        // w_p = wp.vec3()                                                                        <L 1157>
        var_25 = wp::vec_t<3,wp::float32>();
        // v_p = wp.vec3()                                                                        <L 1158>
        var_26 = wp::vec_t<3,wp::float32>();
        // if c_parent >= 0:                                                                      <L 1161>
        var_27 = (var_16 >= var_5);
        if (var_27) {
            // X_wp = body_q[c_parent] * X_wp                                                     <L 1162>
            var_28 = wp::address(var_body_q, var_16);
            var_29 = wp::load(var_28);
            var_30 = wp::mul(var_29, var_23);
            // r_p = wp.transform_get_translation(X_wp) - wp.transform_point(body_q[c_parent], body_com[c_parent])       <L 1163>
            var_31 = wp::transform_get_translation(var_30);
            var_32 = wp::address(var_body_q, var_16);
            var_33 = wp::address(var_body_com, var_16);
            var_34 = wp::load(var_32);
            var_35 = wp::load(var_33);
            var_36 = wp::transform_point(var_34, var_35);
            var_37 = wp::sub(var_31, var_36);
            // twist_p = body_qd[c_parent]                                                        <L 1165>
            var_38 = wp::address(var_body_qd, var_16);
            var_39 = wp::load(var_38);
            var_40 = wp::copy(var_39);
            // w_p = wp.spatial_top(twist_p)                                                      <L 1167>
            var_41 = wp::spatial_top(var_40);
            // v_p = wp.spatial_bottom(twist_p) + wp.cross(w_p, r_p)                              <L 1168>
            var_42 = wp::spatial_bottom(var_40);
            var_43 = wp::cross(var_41, var_37);
            var_44 = wp::add(var_42, var_43);
        }
        var_45 = wp::select(var_27, var_23, var_30);
        var_46 = wp::select(var_27, var_24, var_37);
        var_47 = wp::select(var_27, var_25, var_41);
        var_48 = wp::select(var_27, var_26, var_44);
        // X_wc = body_q[c_child] * X_cj                                                          <L 1171>
        var_49 = wp::address(var_body_q, var_13);
        var_50 = wp::load(var_49);
        var_51 = wp::mul(var_50, var_22);
        // r_c = wp.transform_get_translation(X_wc) - wp.transform_point(body_q[c_child], body_com[c_child])       <L 1172>
        var_52 = wp::transform_get_translation(var_51);
        var_53 = wp::address(var_body_q, var_13);
        var_54 = wp::address(var_body_com, var_13);
        var_55 = wp::load(var_53);
        var_56 = wp::load(var_54);
        var_57 = wp::transform_point(var_55, var_56);
        var_58 = wp::sub(var_52, var_57);
        // twist_c = body_qd[c_child]                                                             <L 1174>
        var_59 = wp::address(var_body_qd, var_13);
        var_60 = wp::load(var_59);
        var_61 = wp::copy(var_60);
        // w_c = wp.spatial_top(twist_c)                                                          <L 1176>
        var_62 = wp::spatial_top(var_61);
        // v_c = wp.spatial_bottom(twist_c) + wp.cross(w_c, r_c)                                  <L 1177>
        var_63 = wp::spatial_bottom(var_61);
        var_64 = wp::cross(var_62, var_58);
        var_65 = wp::add(var_63, var_64);
        // q_start = joint_q_start[tid]                                                           <L 1180>
        var_66 = wp::address(var_joint_q_start, var_0);
        var_67 = wp::load(var_66);
        var_68 = wp::copy(var_67);
        // qd_start = joint_qd_start[tid]                                                         <L 1181>
        var_69 = wp::address(var_joint_qd_start, var_0);
        var_70 = wp::load(var_69);
        var_71 = wp::copy(var_70);
        // axis_start = joint_axis_start[tid]                                                     <L 1182>
        var_72 = wp::address(var_joint_axis_start, var_0);
        var_73 = wp::load(var_72);
        var_74 = wp::copy(var_73);
        // target = joint_target[axis_start]                                                      <L 1184>
        var_75 = wp::address(var_joint_target, var_74);
        var_76 = wp::load(var_75);
        var_77 = wp::copy(var_76);
        // target_ke = joint_target_ke[axis_start]                                                <L 1185>
        var_78 = wp::address(var_joint_target_ke, var_74);
        var_79 = wp::load(var_78);
        var_80 = wp::copy(var_79);
        // target_kd = joint_target_kd[axis_start]                                                <L 1186>
        var_81 = wp::address(var_joint_target_kd, var_74);
        var_82 = wp::load(var_81);
        var_83 = wp::copy(var_82);
        // limit_ke = joint_limit_ke[axis_start]                                                  <L 1187>
        var_84 = wp::address(var_joint_limit_ke, var_74);
        var_85 = wp::load(var_84);
        var_86 = wp::copy(var_85);
        // limit_kd = joint_limit_kd[axis_start]                                                  <L 1188>
        var_87 = wp::address(var_joint_limit_kd, var_74);
        var_88 = wp::load(var_87);
        var_89 = wp::copy(var_88);
        // limit_lower = joint_limit_lower[axis_start]                                            <L 1189>
        var_90 = wp::address(var_joint_limit_lower, var_74);
        var_91 = wp::load(var_90);
        var_92 = wp::copy(var_91);
        // limit_upper = joint_limit_upper[axis_start]                                            <L 1190>
        var_93 = wp::address(var_joint_limit_upper, var_74);
        var_94 = wp::load(var_93);
        var_95 = wp::copy(var_94);
        // act = joint_act[qd_start]                                                              <L 1192>
        var_96 = wp::address(var_joint_act, var_71);
        var_97 = wp::load(var_96);
        var_98 = wp::copy(var_97);
        // x_p = wp.transform_get_translation(X_wp)                                               <L 1194>
        var_99 = wp::transform_get_translation(var_45);
        // x_c = wp.transform_get_translation(X_wc)                                               <L 1195>
        var_100 = wp::transform_get_translation(var_51);
        // q_p = wp.transform_get_rotation(X_wp)                                                  <L 1197>
        var_101 = wp::transform_get_rotation(var_45);
        // q_c = wp.transform_get_rotation(X_wc)                                                  <L 1198>
        var_102 = wp::transform_get_rotation(var_51);
        // x_err = x_c - x_p                                                                      <L 1201>
        var_103 = wp::sub(var_100, var_99);
        // r_err = wp.quat_inverse(q_p) * q_c                                                     <L 1202>
        var_104 = wp::quat_inverse(var_101);
        var_105 = wp::mul(var_104, var_102);
        // v_err = v_c - v_p                                                                      <L 1203>
        var_106 = wp::sub(var_65, var_48);
        // w_err = w_c - w_p                                                                      <L 1204>
        var_107 = wp::sub(var_62, var_47);
        // t_total = wp.vec3()                                                                    <L 1207>
        var_108 = wp::vec_t<3,wp::float32>();
        // f_total = wp.vec3()                                                                    <L 1208>
        var_109 = wp::vec_t<3,wp::float32>();
        // angular_damping_scale = 0.01                                                           <L 1211>
        // if type == fs5Model.JOINT_FIXED:                                                       <L 1213>
        var_112 = (var_3 == var_111);
        if (var_112) {
            // ang_err = wp.normalize(wp.vec3(r_err[0], r_err[1], r_err[2])) * wp.acos(r_err[3]) * 2.0       <L 1214>
            var_113 = wp::extract(var_105, var_5);
            var_115 = wp::extract(var_105, var_114);
            var_117 = wp::extract(var_105, var_116);
            var_118 = wp::vec_t<3,wp::float32>(var_113, var_115, var_117);
            var_119 = wp::normalize(var_118);
            var_121 = wp::extract(var_105, var_120);
            var_122 = wp::acos(var_121);
            var_123 = wp::mul(var_119, var_122);
            var_125 = wp::mul(var_123, var_124);
            // f_total += x_err * joint_attach_ke + v_err * joint_attach_kd                       <L 1216>
            var_126 = wp::mul(var_103, var_joint_attach_ke);
            var_127 = wp::mul(var_106, var_joint_attach_kd);
            var_128 = wp::add(var_126, var_127);
            var_129 = wp::add(var_109, var_128);
            // t_total += (                                                                       <L 1217>
            // wp.transform_vector(X_wp, ang_err) * joint_attach_ke + w_err * joint_attach_kd * angular_damping_scale       <L 1218>
            var_130 = wp::transform_vector(var_45, var_125);
            var_131 = wp::mul(var_130, var_joint_attach_ke);
            var_132 = wp::mul(var_107, var_joint_attach_kd);
            var_133 = wp::mul(var_132, var_110);
            var_134 = wp::add(var_131, var_133);
            var_135 = wp::add(var_108, var_134);
        }
        var_136 = wp::select(var_112, var_108, var_135);
        var_137 = wp::select(var_112, var_109, var_129);
        // if type == fs5Model.JOINT_PRISMATIC:                                                   <L 1221>
        var_139 = (var_3 == var_138);
        if (var_139) {
            // axis = joint_axis[axis_start]                                                      <L 1222>
            var_140 = wp::address(var_joint_axis, var_74);
            var_141 = wp::load(var_140);
            var_142 = wp::copy(var_141);
            // axis_p = wp.transform_vector(X_wp, axis)                                           <L 1225>
            var_143 = wp::transform_vector(var_45, var_142);
            // q = wp.dot(x_err, axis_p)                                                          <L 1228>
            var_144 = wp::dot(var_103, var_143);
            // qd = wp.dot(v_err, axis_p)                                                         <L 1229>
            var_145 = wp::dot(var_106, var_143);
            // f_total = eval_joint_force(                                                        <L 1231>
            // q, qd, target, target_ke, target_kd, act, limit_lower, limit_upper, limit_ke, limit_kd, axis_p       <L 1232>
            var_146 = eval_joint_force(var_144, var_145, var_77, var_80, var_83, var_98, var_92, var_95, var_86, var_89, var_143);
            // ang_err = wp.normalize(wp.vec3(r_err[0], r_err[1], r_err[2])) * wp.acos(r_err[3]) * 2.0       <L 1236>
            var_147 = wp::extract(var_105, var_5);
            var_148 = wp::extract(var_105, var_114);
            var_149 = wp::extract(var_105, var_116);
            var_150 = wp::vec_t<3,wp::float32>(var_147, var_148, var_149);
            var_151 = wp::normalize(var_150);
            var_152 = wp::extract(var_105, var_120);
            var_153 = wp::acos(var_152);
            var_154 = wp::mul(var_151, var_153);
            var_155 = wp::mul(var_154, var_124);
            // f_total += (x_err - q * axis_p) * joint_attach_ke + (v_err - qd * axis_p) * joint_attach_kd       <L 1239>
            var_156 = wp::mul(var_144, var_143);
            var_157 = wp::sub(var_103, var_156);
            var_158 = wp::mul(var_157, var_joint_attach_ke);
            var_159 = wp::mul(var_145, var_143);
            var_160 = wp::sub(var_106, var_159);
            var_161 = wp::mul(var_160, var_joint_attach_kd);
            var_162 = wp::add(var_158, var_161);
            var_163 = wp::add(var_146, var_162);
            // t_total += (                                                                       <L 1240>
            // wp.transform_vector(X_wp, ang_err) * joint_attach_ke + w_err * joint_attach_kd * angular_damping_scale       <L 1241>
            var_164 = wp::transform_vector(var_45, var_155);
            var_165 = wp::mul(var_164, var_joint_attach_ke);
            var_166 = wp::mul(var_107, var_joint_attach_kd);
            var_167 = wp::mul(var_166, var_110);
            var_168 = wp::add(var_165, var_167);
            var_169 = wp::add(var_136, var_168);
        }
        var_170 = wp::select(var_139, var_136, var_169);
        var_171 = wp::select(var_139, var_137, var_163);
        var_172 = wp::select(var_139, var_125, var_155);
        // if type == fs5Model.JOINT_REVOLUTE:                                                    <L 1244>
        var_174 = (var_3 == var_173);
        if (var_174) {
            // axis = joint_axis[axis_start]                                                      <L 1245>
            var_175 = wp::address(var_joint_axis, var_74);
            var_176 = wp::load(var_175);
            var_177 = wp::copy(var_176);
            // axis_p = wp.transform_vector(X_wp, axis)                                           <L 1247>
            var_178 = wp::transform_vector(var_45, var_177);
            // axis_c = wp.transform_vector(X_wc, axis)                                           <L 1248>
            var_179 = wp::transform_vector(var_51, var_177);
            // twist = quat_twist(axis, r_err)                                                    <L 1251>
            var_180 = quat_twist(var_177, var_105);
            // q = wp.acos(twist[3]) * 2.0 * wp.sign(wp.dot(axis, wp.vec3(twist[0], twist[1], twist[2])))       <L 1253>
            var_181 = wp::extract(var_180, var_120);
            var_182 = wp::acos(var_181);
            var_183 = wp::mul(var_182, var_124);
            var_184 = wp::extract(var_180, var_5);
            var_185 = wp::extract(var_180, var_114);
            var_186 = wp::extract(var_180, var_116);
            var_187 = wp::vec_t<3,wp::float32>(var_184, var_185, var_186);
            var_188 = wp::dot(var_177, var_187);
            var_189 = wp::sign(var_188);
            var_190 = wp::mul(var_183, var_189);
            // qd = wp.dot(w_err, axis_p)                                                         <L 1254>
            var_191 = wp::dot(var_107, var_178);
            // t_total = eval_joint_force(                                                        <L 1256>
            // q, qd, target, target_ke, target_kd, act, limit_lower, limit_upper, limit_ke, limit_kd, axis_p       <L 1257>
            var_192 = eval_joint_force(var_190, var_191, var_77, var_80, var_83, var_98, var_92, var_95, var_86, var_89, var_178);
            // swing_err = wp.cross(axis_p, axis_c)                                               <L 1261>
            var_193 = wp::cross(var_178, var_179);
            // f_total += x_err * joint_attach_ke + v_err * joint_attach_kd                       <L 1263>
            var_194 = wp::mul(var_103, var_joint_attach_ke);
            var_195 = wp::mul(var_106, var_joint_attach_kd);
            var_196 = wp::add(var_194, var_195);
            var_197 = wp::add(var_171, var_196);
            // t_total += swing_err * joint_attach_ke + (w_err - qd * axis_p) * joint_attach_kd * angular_damping_scale       <L 1264>
            var_198 = wp::mul(var_193, var_joint_attach_ke);
            var_199 = wp::mul(var_191, var_178);
            var_200 = wp::sub(var_107, var_199);
            var_201 = wp::mul(var_200, var_joint_attach_kd);
            var_202 = wp::mul(var_201, var_110);
            var_203 = wp::add(var_198, var_202);
            var_204 = wp::add(var_192, var_203);
        }
        var_205 = wp::select(var_174, var_170, var_204);
        var_206 = wp::select(var_174, var_171, var_197);
        var_207 = wp::select(var_174, var_142, var_177);
        var_208 = wp::select(var_174, var_143, var_178);
        var_209 = wp::select(var_174, var_144, var_190);
        var_210 = wp::select(var_174, var_145, var_191);
        // if type == fs5Model.JOINT_BALL:                                                        <L 1266>
        var_212 = (var_3 == var_211);
        if (var_212) {
            // ang_err = wp.normalize(wp.vec3(r_err[0], r_err[1], r_err[2])) * wp.acos(r_err[3]) * 2.0       <L 1267>
            var_213 = wp::extract(var_105, var_5);
            var_214 = wp::extract(var_105, var_114);
            var_215 = wp::extract(var_105, var_116);
            var_216 = wp::vec_t<3,wp::float32>(var_213, var_214, var_215);
            var_217 = wp::normalize(var_216);
            var_218 = wp::extract(var_105, var_120);
            var_219 = wp::acos(var_218);
            var_220 = wp::mul(var_217, var_219);
            var_221 = wp::mul(var_220, var_124);
            // t_total += target_kd * w_err + target_ke * wp.transform_vector(X_wp, ang_err)       <L 1270>
            var_222 = wp::mul(var_83, var_107);
            var_223 = wp::transform_vector(var_45, var_221);
            var_224 = wp::mul(var_80, var_223);
            var_225 = wp::add(var_222, var_224);
            var_226 = wp::add(var_205, var_225);
            // f_total += x_err * joint_attach_ke + v_err * joint_attach_kd                       <L 1271>
            var_227 = wp::mul(var_103, var_joint_attach_ke);
            var_228 = wp::mul(var_106, var_joint_attach_kd);
            var_229 = wp::add(var_227, var_228);
            var_230 = wp::add(var_206, var_229);
        }
        var_231 = wp::select(var_212, var_205, var_226);
        var_232 = wp::select(var_212, var_206, var_230);
        var_233 = wp::select(var_212, var_172, var_221);
        // if type == fs5Model.JOINT_COMPOUND:                                                    <L 1273>
        var_235 = (var_3 == var_234);
        if (var_235) {
            // q_pc = wp.quat_inverse(q_p) * q_c                                                  <L 1274>
            var_236 = wp::quat_inverse(var_101);
            var_237 = wp::mul(var_236, var_102);
            // angles = quat_decompose(q_pc)                                                      <L 1277>
            var_238 = quat_decompose(var_237);
            // axis_0 = wp.vec3(1.0, 0.0, 0.0)                                                    <L 1280>
            var_241 = wp::vec_t<3,wp::float32>(var_239, var_240, var_240);
            // q_0 = wp.quat_from_axis_angle(axis_0, angles[0])                                   <L 1281>
            var_242 = wp::extract(var_238, var_5);
            var_243 = wp::quat_from_axis_angle(var_241, var_242);
            // axis_1 = wp.quat_rotate(q_0, wp.vec3(0.0, 1.0, 0.0))                               <L 1283>
            var_244 = wp::vec_t<3,wp::float32>(var_240, var_239, var_240);
            var_245 = wp::quat_rotate(var_243, var_244);
            // q_1 = wp.quat_from_axis_angle(axis_1, angles[1])                                   <L 1284>
            var_246 = wp::extract(var_238, var_114);
            var_247 = wp::quat_from_axis_angle(var_245, var_246);
            // axis_2 = wp.quat_rotate(q_1 * q_0, wp.vec3(0.0, 0.0, 1.0))                         <L 1286>
            var_248 = wp::mul(var_247, var_243);
            var_249 = wp::vec_t<3,wp::float32>(var_240, var_240, var_239);
            var_250 = wp::quat_rotate(var_248, var_249);
            // q_2 = wp.quat_from_axis_angle(axis_2, angles[2])                                   <L 1287>
            var_251 = wp::extract(var_238, var_116);
            var_252 = wp::quat_from_axis_angle(var_250, var_251);
            // q_w = q_p                                                                          <L 1289>
            var_253 = wp::copy(var_101);
            // axis_0 = wp.transform_vector(X_wp, axis_0)                                         <L 1291>
            var_254 = wp::transform_vector(var_45, var_241);
            // axis_1 = wp.transform_vector(X_wp, axis_1)                                         <L 1292>
            var_255 = wp::transform_vector(var_45, var_245);
            // axis_2 = wp.transform_vector(X_wp, axis_2)                                         <L 1293>
            var_256 = wp::transform_vector(var_45, var_250);
            // t_total = wp.vec3()                                                                <L 1296>
            var_257 = wp::vec_t<3,wp::float32>();
            // t_total += eval_joint_force(                                                       <L 1302>
            // angles[0],                                                                         <L 1303>
            var_258 = wp::extract(var_238, var_5);
            // wp.dot(axis_0, w_err),                                                             <L 1304>
            var_259 = wp::dot(var_254, var_107);
            // joint_target[axis_start + 0],                                                      <L 1305>
            var_260 = wp::add(var_74, var_5);
            var_261 = wp::address(var_joint_target, var_260);
            // joint_target_ke[axis_start + 0],                                                   <L 1306>
            var_262 = wp::add(var_74, var_5);
            var_263 = wp::address(var_joint_target_ke, var_262);
            // joint_target_kd[axis_start + 0],                                                   <L 1307>
            var_264 = wp::add(var_74, var_5);
            var_265 = wp::address(var_joint_target_kd, var_264);
            // joint_act[axis_start + 0],                                                         <L 1308>
            var_266 = wp::add(var_74, var_5);
            var_267 = wp::address(var_joint_act, var_266);
            // joint_limit_lower[axis_start + 0],                                                 <L 1309>
            var_268 = wp::add(var_74, var_5);
            var_269 = wp::address(var_joint_limit_lower, var_268);
            // joint_limit_upper[axis_start + 0],                                                 <L 1310>
            var_270 = wp::add(var_74, var_5);
            var_271 = wp::address(var_joint_limit_upper, var_270);
            // joint_limit_ke[axis_start + 0],                                                    <L 1311>
            var_272 = wp::add(var_74, var_5);
            var_273 = wp::address(var_joint_limit_ke, var_272);
            // joint_limit_kd[axis_start + 0],                                                    <L 1312>
            var_274 = wp::add(var_74, var_5);
            var_275 = wp::address(var_joint_limit_kd, var_274);
            // axis_0,                                                                            <L 1313>
            var_276 = wp::load(var_261);
            var_277 = wp::load(var_263);
            var_278 = wp::load(var_265);
            var_279 = wp::load(var_267);
            var_280 = wp::load(var_269);
            var_281 = wp::load(var_271);
            var_282 = wp::load(var_273);
            var_283 = wp::load(var_275);
            var_284 = eval_joint_force(var_258, var_259, var_276, var_277, var_278, var_279, var_280, var_281, var_282, var_283, var_254);
            var_285 = wp::add(var_257, var_284);
            // t_total += eval_joint_force(                                                       <L 1315>
            // angles[1],                                                                         <L 1316>
            var_286 = wp::extract(var_238, var_114);
            // wp.dot(axis_1, w_err),                                                             <L 1317>
            var_287 = wp::dot(var_255, var_107);
            // joint_target[axis_start + 1],                                                      <L 1318>
            var_288 = wp::add(var_74, var_114);
            var_289 = wp::address(var_joint_target, var_288);
            // joint_target_ke[axis_start + 1],                                                   <L 1319>
            var_290 = wp::add(var_74, var_114);
            var_291 = wp::address(var_joint_target_ke, var_290);
            // joint_target_kd[axis_start + 1],                                                   <L 1320>
            var_292 = wp::add(var_74, var_114);
            var_293 = wp::address(var_joint_target_kd, var_292);
            // joint_act[axis_start + 1],                                                         <L 1321>
            var_294 = wp::add(var_74, var_114);
            var_295 = wp::address(var_joint_act, var_294);
            // joint_limit_lower[axis_start + 1],                                                 <L 1322>
            var_296 = wp::add(var_74, var_114);
            var_297 = wp::address(var_joint_limit_lower, var_296);
            // joint_limit_upper[axis_start + 1],                                                 <L 1323>
            var_298 = wp::add(var_74, var_114);
            var_299 = wp::address(var_joint_limit_upper, var_298);
            // joint_limit_ke[axis_start + 1],                                                    <L 1324>
            var_300 = wp::add(var_74, var_114);
            var_301 = wp::address(var_joint_limit_ke, var_300);
            // joint_limit_kd[axis_start + 1],                                                    <L 1325>
            var_302 = wp::add(var_74, var_114);
            var_303 = wp::address(var_joint_limit_kd, var_302);
            // axis_1,                                                                            <L 1326>
            var_304 = wp::load(var_289);
            var_305 = wp::load(var_291);
            var_306 = wp::load(var_293);
            var_307 = wp::load(var_295);
            var_308 = wp::load(var_297);
            var_309 = wp::load(var_299);
            var_310 = wp::load(var_301);
            var_311 = wp::load(var_303);
            var_312 = eval_joint_force(var_286, var_287, var_304, var_305, var_306, var_307, var_308, var_309, var_310, var_311, var_255);
            var_313 = wp::add(var_285, var_312);
            // t_total += eval_joint_force(                                                       <L 1328>
            // angles[2],                                                                         <L 1329>
            var_314 = wp::extract(var_238, var_116);
            // wp.dot(axis_2, w_err),                                                             <L 1330>
            var_315 = wp::dot(var_256, var_107);
            // joint_target[axis_start + 2],                                                      <L 1331>
            var_316 = wp::add(var_74, var_116);
            var_317 = wp::address(var_joint_target, var_316);
            // joint_target_ke[axis_start + 2],                                                   <L 1332>
            var_318 = wp::add(var_74, var_116);
            var_319 = wp::address(var_joint_target_ke, var_318);
            // joint_target_kd[axis_start + 2],                                                   <L 1333>
            var_320 = wp::add(var_74, var_116);
            var_321 = wp::address(var_joint_target_kd, var_320);
            // joint_act[axis_start + 2],                                                         <L 1334>
            var_322 = wp::add(var_74, var_116);
            var_323 = wp::address(var_joint_act, var_322);
            // joint_limit_lower[axis_start + 2],                                                 <L 1335>
            var_324 = wp::add(var_74, var_116);
            var_325 = wp::address(var_joint_limit_lower, var_324);
            // joint_limit_upper[axis_start + 2],                                                 <L 1336>
            var_326 = wp::add(var_74, var_116);
            var_327 = wp::address(var_joint_limit_upper, var_326);
            // joint_limit_ke[axis_start + 2],                                                    <L 1337>
            var_328 = wp::add(var_74, var_116);
            var_329 = wp::address(var_joint_limit_ke, var_328);
            // joint_limit_kd[axis_start + 2],                                                    <L 1338>
            var_330 = wp::add(var_74, var_116);
            var_331 = wp::address(var_joint_limit_kd, var_330);
            // axis_2,                                                                            <L 1339>
            var_332 = wp::load(var_317);
            var_333 = wp::load(var_319);
            var_334 = wp::load(var_321);
            var_335 = wp::load(var_323);
            var_336 = wp::load(var_325);
            var_337 = wp::load(var_327);
            var_338 = wp::load(var_329);
            var_339 = wp::load(var_331);
            var_340 = eval_joint_force(var_314, var_315, var_332, var_333, var_334, var_335, var_336, var_337, var_338, var_339, var_256);
            var_341 = wp::add(var_313, var_340);
            // f_total += x_err * joint_attach_ke + v_err * joint_attach_kd                       <L 1342>
            var_342 = wp::mul(var_103, var_joint_attach_ke);
            var_343 = wp::mul(var_106, var_joint_attach_kd);
            var_344 = wp::add(var_342, var_343);
            var_345 = wp::add(var_232, var_344);
        }
        var_346 = wp::select(var_235, var_231, var_341);
        var_347 = wp::select(var_235, var_232, var_345);
        // if type == fs5Model.JOINT_UNIVERSAL:                                                   <L 1344>
        var_349 = (var_3 == var_348);
        if (var_349) {
            // q_pc = wp.quat_inverse(q_p) * q_c                                                  <L 1345>
            var_350 = wp::quat_inverse(var_101);
            var_351 = wp::mul(var_350, var_102);
            // angles = quat_decompose(q_pc)                                                      <L 1348>
            var_352 = quat_decompose(var_351);
            // axis_0 = wp.vec3(1.0, 0.0, 0.0)                                                    <L 1351>
            var_353 = wp::vec_t<3,wp::float32>(var_239, var_240, var_240);
            // q_0 = wp.quat_from_axis_angle(axis_0, angles[0])                                   <L 1352>
            var_354 = wp::extract(var_352, var_5);
            var_355 = wp::quat_from_axis_angle(var_353, var_354);
            // axis_1 = wp.quat_rotate(q_0, wp.vec3(0.0, 1.0, 0.0))                               <L 1354>
            var_356 = wp::vec_t<3,wp::float32>(var_240, var_239, var_240);
            var_357 = wp::quat_rotate(var_355, var_356);
            // q_1 = wp.quat_from_axis_angle(axis_1, angles[1])                                   <L 1355>
            var_358 = wp::extract(var_352, var_114);
            var_359 = wp::quat_from_axis_angle(var_357, var_358);
            // axis_2 = wp.quat_rotate(q_1 * q_0, wp.vec3(0.0, 0.0, 1.0))                         <L 1357>
            var_360 = wp::mul(var_359, var_355);
            var_361 = wp::vec_t<3,wp::float32>(var_240, var_240, var_239);
            var_362 = wp::quat_rotate(var_360, var_361);
            // q_2 = wp.quat_from_axis_angle(axis_2, angles[2])                                   <L 1358>
            var_363 = wp::extract(var_352, var_116);
            var_364 = wp::quat_from_axis_angle(var_362, var_363);
            // q_w = q_p                                                                          <L 1360>
            var_365 = wp::copy(var_101);
            // axis_0 = wp.transform_vector(X_wp, axis_0)                                         <L 1362>
            var_366 = wp::transform_vector(var_45, var_353);
            // axis_1 = wp.transform_vector(X_wp, axis_1)                                         <L 1363>
            var_367 = wp::transform_vector(var_45, var_357);
            // axis_2 = wp.transform_vector(X_wp, axis_2)                                         <L 1364>
            var_368 = wp::transform_vector(var_45, var_362);
            // t_total = wp.vec3()                                                                <L 1367>
            var_369 = wp::vec_t<3,wp::float32>();
            // t_total += eval_joint_force(                                                       <L 1378>
            // angles[0],                                                                         <L 1379>
            var_370 = wp::extract(var_352, var_5);
            // wp.dot(axis_0, w_err),                                                             <L 1380>
            var_371 = wp::dot(var_366, var_107);
            // joint_target[axis_start + 0],                                                      <L 1381>
            var_372 = wp::add(var_74, var_5);
            var_373 = wp::address(var_joint_target, var_372);
            // joint_target_ke[axis_start + 0],                                                   <L 1382>
            var_374 = wp::add(var_74, var_5);
            var_375 = wp::address(var_joint_target_ke, var_374);
            // joint_target_kd[axis_start + 0],                                                   <L 1383>
            var_376 = wp::add(var_74, var_5);
            var_377 = wp::address(var_joint_target_kd, var_376);
            // joint_act[axis_start + 0],                                                         <L 1384>
            var_378 = wp::add(var_74, var_5);
            var_379 = wp::address(var_joint_act, var_378);
            // joint_limit_lower[axis_start + 0],                                                 <L 1385>
            var_380 = wp::add(var_74, var_5);
            var_381 = wp::address(var_joint_limit_lower, var_380);
            // joint_limit_upper[axis_start + 0],                                                 <L 1386>
            var_382 = wp::add(var_74, var_5);
            var_383 = wp::address(var_joint_limit_upper, var_382);
            // joint_limit_ke[axis_start + 0],                                                    <L 1387>
            var_384 = wp::add(var_74, var_5);
            var_385 = wp::address(var_joint_limit_ke, var_384);
            // joint_limit_kd[axis_start + 0],                                                    <L 1388>
            var_386 = wp::add(var_74, var_5);
            var_387 = wp::address(var_joint_limit_kd, var_386);
            // axis_0,                                                                            <L 1389>
            var_388 = wp::load(var_373);
            var_389 = wp::load(var_375);
            var_390 = wp::load(var_377);
            var_391 = wp::load(var_379);
            var_392 = wp::load(var_381);
            var_393 = wp::load(var_383);
            var_394 = wp::load(var_385);
            var_395 = wp::load(var_387);
            var_396 = eval_joint_force(var_370, var_371, var_388, var_389, var_390, var_391, var_392, var_393, var_394, var_395, var_366);
            var_397 = wp::add(var_369, var_396);
            // t_total += eval_joint_force(                                                       <L 1391>
            // angles[1],                                                                         <L 1392>
            var_398 = wp::extract(var_352, var_114);
            // wp.dot(axis_1, w_err),                                                             <L 1393>
            var_399 = wp::dot(var_367, var_107);
            // joint_target[axis_start + 1],                                                      <L 1394>
            var_400 = wp::add(var_74, var_114);
            var_401 = wp::address(var_joint_target, var_400);
            // joint_target_ke[axis_start + 1],                                                   <L 1395>
            var_402 = wp::add(var_74, var_114);
            var_403 = wp::address(var_joint_target_ke, var_402);
            // joint_target_kd[axis_start + 1],                                                   <L 1396>
            var_404 = wp::add(var_74, var_114);
            var_405 = wp::address(var_joint_target_kd, var_404);
            // joint_act[axis_start + 1],                                                         <L 1397>
            var_406 = wp::add(var_74, var_114);
            var_407 = wp::address(var_joint_act, var_406);
            // joint_limit_lower[axis_start + 1],                                                 <L 1398>
            var_408 = wp::add(var_74, var_114);
            var_409 = wp::address(var_joint_limit_lower, var_408);
            // joint_limit_upper[axis_start + 1],                                                 <L 1399>
            var_410 = wp::add(var_74, var_114);
            var_411 = wp::address(var_joint_limit_upper, var_410);
            // joint_limit_ke[axis_start + 1],                                                    <L 1400>
            var_412 = wp::add(var_74, var_114);
            var_413 = wp::address(var_joint_limit_ke, var_412);
            // joint_limit_kd[axis_start + 1],                                                    <L 1401>
            var_414 = wp::add(var_74, var_114);
            var_415 = wp::address(var_joint_limit_kd, var_414);
            // axis_1,                                                                            <L 1402>
            var_416 = wp::load(var_401);
            var_417 = wp::load(var_403);
            var_418 = wp::load(var_405);
            var_419 = wp::load(var_407);
            var_420 = wp::load(var_409);
            var_421 = wp::load(var_411);
            var_422 = wp::load(var_413);
            var_423 = wp::load(var_415);
            var_424 = eval_joint_force(var_398, var_399, var_416, var_417, var_418, var_419, var_420, var_421, var_422, var_423, var_367);
            var_425 = wp::add(var_397, var_424);
            // t_total += eval_joint_force(                                                       <L 1406>
            // angles[2],                                                                         <L 1407>
            var_426 = wp::extract(var_352, var_116);
            // wp.dot(axis_2, w_err),                                                             <L 1408>
            var_427 = wp::dot(var_368, var_107);
            // 0.0,                                                                               <L 1409>
            // joint_attach_ke,                                                                   <L 1410>
            // joint_attach_kd * angular_damping_scale,                                           <L 1411>
            var_428 = wp::mul(var_joint_attach_kd, var_110);
            // 0.0,                                                                               <L 1412>
            // 0.0,                                                                               <L 1413>
            // 0.0,                                                                               <L 1414>
            // 0.0,                                                                               <L 1415>
            // 0.0,                                                                               <L 1416>
            // axis_2,                                                                            <L 1417>
            var_429 = eval_joint_force(var_426, var_427, var_240, var_joint_attach_ke, var_428, var_240, var_240, var_240, var_240, var_240, var_368);
            var_430 = wp::add(var_425, var_429);
            // f_total += x_err * joint_attach_ke + v_err * joint_attach_kd                       <L 1420>
            var_431 = wp::mul(var_103, var_joint_attach_ke);
            var_432 = wp::mul(var_106, var_joint_attach_kd);
            var_433 = wp::add(var_431, var_432);
            var_434 = wp::add(var_347, var_433);
        }
        var_435 = wp::select(var_349, var_346, var_430);
        var_436 = wp::select(var_349, var_347, var_434);
        var_437 = wp::select(var_349, var_237, var_351);
        var_438 = wp::select(var_349, var_238, var_352);
        var_439 = wp::select(var_349, var_254, var_366);
        var_440 = wp::select(var_349, var_243, var_355);
        var_441 = wp::select(var_349, var_255, var_367);
        var_442 = wp::select(var_349, var_247, var_359);
        var_443 = wp::select(var_349, var_256, var_368);
        var_444 = wp::select(var_349, var_252, var_364);
        var_445 = wp::select(var_349, var_253, var_365);
        // if c_parent >= 0:                                                                      <L 1423>
        var_446 = (var_16 >= var_5);
        if (var_446) {
            // wp.atomic_add(body_f, c_parent, wp.spatial_vector(t_total + wp.cross(r_p, f_total), f_total))       <L 1424>
            var_447 = wp::cross(var_46, var_436);
            var_448 = wp::add(var_435, var_447);
            var_449 = wp::vec_t<6,wp::float32>(var_448, var_436);
            var_450 = wp::atomic_add(var_body_f, var_16, var_449);
        }
        // wp.atomic_sub(body_f, c_child, wp.spatial_vector(t_total + wp.cross(r_c, f_total), f_total))       <L 1426>
        var_451 = wp::cross(var_58, var_436);
        var_452 = wp::add(var_435, var_451);
        var_453 = wp::vec_t<6,wp::float32>(var_452, var_436);
        var_454 = wp::atomic_sub(var_body_f, var_13, var_453);
    }
}

extern "C" __global__ void eval_body_joints_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::transform_t<wp::float32>> var_body_q,
    wp::array_t<wp::vec_t<6,wp::float32>> var_body_qd,
    wp::array_t<wp::vec_t<3,wp::float32>> var_body_com,
    wp::array_t<wp::int32> var_joint_q_start,
    wp::array_t<wp::int32> var_joint_qd_start,
    wp::array_t<wp::int32> var_joint_type,
    wp::array_t<wp::int32> var_joint_enabled,
    wp::array_t<wp::int32> var_joint_child,
    wp::array_t<wp::int32> var_joint_parent,
    wp::array_t<wp::transform_t<wp::float32>> var_joint_X_p,
    wp::array_t<wp::transform_t<wp::float32>> var_joint_X_c,
    wp::array_t<wp::vec_t<3,wp::float32>> var_joint_axis,
    wp::array_t<wp::int32> var_joint_axis_start,
    wp::array_t<wp::int32> var_joint_axis_dim,
    wp::array_t<wp::float32> var_joint_target,
    wp::array_t<wp::float32> var_joint_act,
    wp::array_t<wp::float32> var_joint_target_ke,
    wp::array_t<wp::float32> var_joint_target_kd,
    wp::array_t<wp::float32> var_joint_limit_lower,
    wp::array_t<wp::float32> var_joint_limit_upper,
    wp::array_t<wp::float32> var_joint_limit_ke,
    wp::array_t<wp::float32> var_joint_limit_kd,
    wp::float32 var_joint_attach_ke,
    wp::float32 var_joint_attach_kd,
    wp::array_t<wp::vec_t<6,wp::float32>> var_body_f,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_q,
    wp::array_t<wp::vec_t<6,wp::float32>> adj_body_qd,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_body_com,
    wp::array_t<wp::int32> adj_joint_q_start,
    wp::array_t<wp::int32> adj_joint_qd_start,
    wp::array_t<wp::int32> adj_joint_type,
    wp::array_t<wp::int32> adj_joint_enabled,
    wp::array_t<wp::int32> adj_joint_child,
    wp::array_t<wp::int32> adj_joint_parent,
    wp::array_t<wp::transform_t<wp::float32>> adj_joint_X_p,
    wp::array_t<wp::transform_t<wp::float32>> adj_joint_X_c,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_joint_axis,
    wp::array_t<wp::int32> adj_joint_axis_start,
    wp::array_t<wp::int32> adj_joint_axis_dim,
    wp::array_t<wp::float32> adj_joint_target,
    wp::array_t<wp::float32> adj_joint_act,
    wp::array_t<wp::float32> adj_joint_target_ke,
    wp::array_t<wp::float32> adj_joint_target_kd,
    wp::array_t<wp::float32> adj_joint_limit_lower,
    wp::array_t<wp::float32> adj_joint_limit_upper,
    wp::array_t<wp::float32> adj_joint_limit_ke,
    wp::array_t<wp::float32> adj_joint_limit_kd,
    wp::float32 adj_joint_attach_ke,
    wp::float32 adj_joint_attach_kd,
    wp::array_t<wp::vec_t<6,wp::float32>> adj_body_f)
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
        wp::int32* var_4;
        const wp::int32 var_5 = 0;
        bool var_6;
        wp::int32 var_7;
        const wp::int32 var_8 = 4;
        bool var_9;
        bool var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::transform_t<wp::float32>* var_17;
        wp::transform_t<wp::float32> var_18;
        wp::transform_t<wp::float32> var_19;
        wp::transform_t<wp::float32>* var_20;
        wp::transform_t<wp::float32> var_21;
        wp::transform_t<wp::float32> var_22;
        wp::transform_t<wp::float32> var_23;
        wp::vec_t<3,wp::float32> var_24;
        wp::vec_t<3,wp::float32> var_25;
        wp::vec_t<3,wp::float32> var_26;
        bool var_27;
        wp::transform_t<wp::float32>* var_28;
        wp::transform_t<wp::float32> var_29;
        wp::transform_t<wp::float32> var_30;
        wp::vec_t<3,wp::float32> var_31;
        wp::transform_t<wp::float32>* var_32;
        wp::vec_t<3,wp::float32>* var_33;
        wp::transform_t<wp::float32> var_34;
        wp::vec_t<3,wp::float32> var_35;
        wp::vec_t<3,wp::float32> var_36;
        wp::vec_t<3,wp::float32> var_37;
        wp::vec_t<6,wp::float32>* var_38;
        wp::vec_t<6,wp::float32> var_39;
        wp::vec_t<6,wp::float32> var_40;
        wp::vec_t<3,wp::float32> var_41;
        wp::vec_t<3,wp::float32> var_42;
        wp::vec_t<3,wp::float32> var_43;
        wp::vec_t<3,wp::float32> var_44;
        wp::transform_t<wp::float32> var_45;
        wp::vec_t<3,wp::float32> var_46;
        wp::vec_t<3,wp::float32> var_47;
        wp::vec_t<3,wp::float32> var_48;
        wp::transform_t<wp::float32>* var_49;
        wp::transform_t<wp::float32> var_50;
        wp::transform_t<wp::float32> var_51;
        wp::vec_t<3,wp::float32> var_52;
        wp::transform_t<wp::float32>* var_53;
        wp::vec_t<3,wp::float32>* var_54;
        wp::transform_t<wp::float32> var_55;
        wp::vec_t<3,wp::float32> var_56;
        wp::vec_t<3,wp::float32> var_57;
        wp::vec_t<3,wp::float32> var_58;
        wp::vec_t<6,wp::float32>* var_59;
        wp::vec_t<6,wp::float32> var_60;
        wp::vec_t<6,wp::float32> var_61;
        wp::vec_t<3,wp::float32> var_62;
        wp::vec_t<3,wp::float32> var_63;
        wp::vec_t<3,wp::float32> var_64;
        wp::vec_t<3,wp::float32> var_65;
        wp::int32* var_66;
        wp::int32 var_67;
        wp::int32 var_68;
        wp::int32* var_69;
        wp::int32 var_70;
        wp::int32 var_71;
        wp::int32* var_72;
        wp::int32 var_73;
        wp::int32 var_74;
        wp::float32* var_75;
        wp::float32 var_76;
        wp::float32 var_77;
        wp::float32* var_78;
        wp::float32 var_79;
        wp::float32 var_80;
        wp::float32* var_81;
        wp::float32 var_82;
        wp::float32 var_83;
        wp::float32* var_84;
        wp::float32 var_85;
        wp::float32 var_86;
        wp::float32* var_87;
        wp::float32 var_88;
        wp::float32 var_89;
        wp::float32* var_90;
        wp::float32 var_91;
        wp::float32 var_92;
        wp::float32* var_93;
        wp::float32 var_94;
        wp::float32 var_95;
        wp::float32* var_96;
        wp::float32 var_97;
        wp::float32 var_98;
        wp::vec_t<3,wp::float32> var_99;
        wp::vec_t<3,wp::float32> var_100;
        wp::quat_t<wp::float32> var_101;
        wp::quat_t<wp::float32> var_102;
        wp::vec_t<3,wp::float32> var_103;
        wp::quat_t<wp::float32> var_104;
        wp::quat_t<wp::float32> var_105;
        wp::vec_t<3,wp::float32> var_106;
        wp::vec_t<3,wp::float32> var_107;
        wp::vec_t<3,wp::float32> var_108;
        wp::vec_t<3,wp::float32> var_109;
        const wp::float32 var_110 = 0.01;
        const wp::int32 var_111 = 3;
        bool var_112;
        wp::float32 var_113;
        const wp::int32 var_114 = 1;
        wp::float32 var_115;
        const wp::int32 var_116 = 2;
        wp::float32 var_117;
        wp::vec_t<3,wp::float32> var_118;
        wp::vec_t<3,wp::float32> var_119;
        const wp::int32 var_120 = 3;
        wp::float32 var_121;
        wp::float32 var_122;
        wp::vec_t<3,wp::float32> var_123;
        const wp::float32 var_124 = 2.0;
        wp::vec_t<3,wp::float32> var_125;
        wp::vec_t<3,wp::float32> var_126;
        wp::vec_t<3,wp::float32> var_127;
        wp::vec_t<3,wp::float32> var_128;
        wp::vec_t<3,wp::float32> var_129;
        wp::vec_t<3,wp::float32> var_130;
        wp::vec_t<3,wp::float32> var_131;
        wp::vec_t<3,wp::float32> var_132;
        wp::vec_t<3,wp::float32> var_133;
        wp::vec_t<3,wp::float32> var_134;
        wp::vec_t<3,wp::float32> var_135;
        wp::vec_t<3,wp::float32> var_136;
        wp::vec_t<3,wp::float32> var_137;
        const wp::int32 var_138 = 0;
        bool var_139;
        wp::vec_t<3,wp::float32>* var_140;
        wp::vec_t<3,wp::float32> var_141;
        wp::vec_t<3,wp::float32> var_142;
        wp::vec_t<3,wp::float32> var_143;
        wp::float32 var_144;
        wp::float32 var_145;
        wp::vec_t<3,wp::float32> var_146;
        wp::float32 var_147;
        wp::float32 var_148;
        wp::float32 var_149;
        wp::vec_t<3,wp::float32> var_150;
        wp::vec_t<3,wp::float32> var_151;
        wp::float32 var_152;
        wp::float32 var_153;
        wp::vec_t<3,wp::float32> var_154;
        wp::vec_t<3,wp::float32> var_155;
        wp::vec_t<3,wp::float32> var_156;
        wp::vec_t<3,wp::float32> var_157;
        wp::vec_t<3,wp::float32> var_158;
        wp::vec_t<3,wp::float32> var_159;
        wp::vec_t<3,wp::float32> var_160;
        wp::vec_t<3,wp::float32> var_161;
        wp::vec_t<3,wp::float32> var_162;
        wp::vec_t<3,wp::float32> var_163;
        wp::vec_t<3,wp::float32> var_164;
        wp::vec_t<3,wp::float32> var_165;
        wp::vec_t<3,wp::float32> var_166;
        wp::vec_t<3,wp::float32> var_167;
        wp::vec_t<3,wp::float32> var_168;
        wp::vec_t<3,wp::float32> var_169;
        wp::vec_t<3,wp::float32> var_170;
        wp::vec_t<3,wp::float32> var_171;
        wp::vec_t<3,wp::float32> var_172;
        const wp::int32 var_173 = 1;
        bool var_174;
        wp::vec_t<3,wp::float32>* var_175;
        wp::vec_t<3,wp::float32> var_176;
        wp::vec_t<3,wp::float32> var_177;
        wp::vec_t<3,wp::float32> var_178;
        wp::vec_t<3,wp::float32> var_179;
        wp::quat_t<wp::float32> var_180;
        wp::float32 var_181;
        wp::float32 var_182;
        wp::float32 var_183;
        wp::float32 var_184;
        wp::float32 var_185;
        wp::float32 var_186;
        wp::vec_t<3,wp::float32> var_187;
        wp::float32 var_188;
        wp::float32 var_189;
        wp::float32 var_190;
        wp::float32 var_191;
        wp::vec_t<3,wp::float32> var_192;
        wp::vec_t<3,wp::float32> var_193;
        wp::vec_t<3,wp::float32> var_194;
        wp::vec_t<3,wp::float32> var_195;
        wp::vec_t<3,wp::float32> var_196;
        wp::vec_t<3,wp::float32> var_197;
        wp::vec_t<3,wp::float32> var_198;
        wp::vec_t<3,wp::float32> var_199;
        wp::vec_t<3,wp::float32> var_200;
        wp::vec_t<3,wp::float32> var_201;
        wp::vec_t<3,wp::float32> var_202;
        wp::vec_t<3,wp::float32> var_203;
        wp::vec_t<3,wp::float32> var_204;
        wp::vec_t<3,wp::float32> var_205;
        wp::vec_t<3,wp::float32> var_206;
        wp::vec_t<3,wp::float32> var_207;
        wp::vec_t<3,wp::float32> var_208;
        wp::float32 var_209;
        wp::float32 var_210;
        const wp::int32 var_211 = 2;
        bool var_212;
        wp::float32 var_213;
        wp::float32 var_214;
        wp::float32 var_215;
        wp::vec_t<3,wp::float32> var_216;
        wp::vec_t<3,wp::float32> var_217;
        wp::float32 var_218;
        wp::float32 var_219;
        wp::vec_t<3,wp::float32> var_220;
        wp::vec_t<3,wp::float32> var_221;
        wp::vec_t<3,wp::float32> var_222;
        wp::vec_t<3,wp::float32> var_223;
        wp::vec_t<3,wp::float32> var_224;
        wp::vec_t<3,wp::float32> var_225;
        wp::vec_t<3,wp::float32> var_226;
        wp::vec_t<3,wp::float32> var_227;
        wp::vec_t<3,wp::float32> var_228;
        wp::vec_t<3,wp::float32> var_229;
        wp::vec_t<3,wp::float32> var_230;
        wp::vec_t<3,wp::float32> var_231;
        wp::vec_t<3,wp::float32> var_232;
        wp::vec_t<3,wp::float32> var_233;
        const wp::int32 var_234 = 5;
        bool var_235;
        wp::quat_t<wp::float32> var_236;
        wp::quat_t<wp::float32> var_237;
        wp::vec_t<3,wp::float32> var_238;
        const wp::float32 var_239 = 1.0;
        const wp::float32 var_240 = 0.0;
        wp::vec_t<3,wp::float32> var_241;
        wp::float32 var_242;
        wp::quat_t<wp::float32> var_243;
        wp::vec_t<3,wp::float32> var_244;
        wp::vec_t<3,wp::float32> var_245;
        wp::float32 var_246;
        wp::quat_t<wp::float32> var_247;
        wp::quat_t<wp::float32> var_248;
        wp::vec_t<3,wp::float32> var_249;
        wp::vec_t<3,wp::float32> var_250;
        wp::float32 var_251;
        wp::quat_t<wp::float32> var_252;
        wp::quat_t<wp::float32> var_253;
        wp::vec_t<3,wp::float32> var_254;
        wp::vec_t<3,wp::float32> var_255;
        wp::vec_t<3,wp::float32> var_256;
        wp::vec_t<3,wp::float32> var_257;
        wp::float32 var_258;
        wp::float32 var_259;
        wp::int32 var_260;
        wp::float32* var_261;
        wp::int32 var_262;
        wp::float32* var_263;
        wp::int32 var_264;
        wp::float32* var_265;
        wp::int32 var_266;
        wp::float32* var_267;
        wp::int32 var_268;
        wp::float32* var_269;
        wp::int32 var_270;
        wp::float32* var_271;
        wp::int32 var_272;
        wp::float32* var_273;
        wp::int32 var_274;
        wp::float32* var_275;
        wp::float32 var_276;
        wp::float32 var_277;
        wp::float32 var_278;
        wp::float32 var_279;
        wp::float32 var_280;
        wp::float32 var_281;
        wp::float32 var_282;
        wp::float32 var_283;
        wp::vec_t<3,wp::float32> var_284;
        wp::vec_t<3,wp::float32> var_285;
        wp::float32 var_286;
        wp::float32 var_287;
        wp::int32 var_288;
        wp::float32* var_289;
        wp::int32 var_290;
        wp::float32* var_291;
        wp::int32 var_292;
        wp::float32* var_293;
        wp::int32 var_294;
        wp::float32* var_295;
        wp::int32 var_296;
        wp::float32* var_297;
        wp::int32 var_298;
        wp::float32* var_299;
        wp::int32 var_300;
        wp::float32* var_301;
        wp::int32 var_302;
        wp::float32* var_303;
        wp::float32 var_304;
        wp::float32 var_305;
        wp::float32 var_306;
        wp::float32 var_307;
        wp::float32 var_308;
        wp::float32 var_309;
        wp::float32 var_310;
        wp::float32 var_311;
        wp::vec_t<3,wp::float32> var_312;
        wp::vec_t<3,wp::float32> var_313;
        wp::float32 var_314;
        wp::float32 var_315;
        wp::int32 var_316;
        wp::float32* var_317;
        wp::int32 var_318;
        wp::float32* var_319;
        wp::int32 var_320;
        wp::float32* var_321;
        wp::int32 var_322;
        wp::float32* var_323;
        wp::int32 var_324;
        wp::float32* var_325;
        wp::int32 var_326;
        wp::float32* var_327;
        wp::int32 var_328;
        wp::float32* var_329;
        wp::int32 var_330;
        wp::float32* var_331;
        wp::float32 var_332;
        wp::float32 var_333;
        wp::float32 var_334;
        wp::float32 var_335;
        wp::float32 var_336;
        wp::float32 var_337;
        wp::float32 var_338;
        wp::float32 var_339;
        wp::vec_t<3,wp::float32> var_340;
        wp::vec_t<3,wp::float32> var_341;
        wp::vec_t<3,wp::float32> var_342;
        wp::vec_t<3,wp::float32> var_343;
        wp::vec_t<3,wp::float32> var_344;
        wp::vec_t<3,wp::float32> var_345;
        wp::vec_t<3,wp::float32> var_346;
        wp::vec_t<3,wp::float32> var_347;
        const wp::int32 var_348 = 6;
        bool var_349;
        wp::quat_t<wp::float32> var_350;
        wp::quat_t<wp::float32> var_351;
        wp::vec_t<3,wp::float32> var_352;
        wp::vec_t<3,wp::float32> var_353;
        wp::float32 var_354;
        wp::quat_t<wp::float32> var_355;
        wp::vec_t<3,wp::float32> var_356;
        wp::vec_t<3,wp::float32> var_357;
        wp::float32 var_358;
        wp::quat_t<wp::float32> var_359;
        wp::quat_t<wp::float32> var_360;
        wp::vec_t<3,wp::float32> var_361;
        wp::vec_t<3,wp::float32> var_362;
        wp::float32 var_363;
        wp::quat_t<wp::float32> var_364;
        wp::quat_t<wp::float32> var_365;
        wp::vec_t<3,wp::float32> var_366;
        wp::vec_t<3,wp::float32> var_367;
        wp::vec_t<3,wp::float32> var_368;
        wp::vec_t<3,wp::float32> var_369;
        wp::float32 var_370;
        wp::float32 var_371;
        wp::int32 var_372;
        wp::float32* var_373;
        wp::int32 var_374;
        wp::float32* var_375;
        wp::int32 var_376;
        wp::float32* var_377;
        wp::int32 var_378;
        wp::float32* var_379;
        wp::int32 var_380;
        wp::float32* var_381;
        wp::int32 var_382;
        wp::float32* var_383;
        wp::int32 var_384;
        wp::float32* var_385;
        wp::int32 var_386;
        wp::float32* var_387;
        wp::float32 var_388;
        wp::float32 var_389;
        wp::float32 var_390;
        wp::float32 var_391;
        wp::float32 var_392;
        wp::float32 var_393;
        wp::float32 var_394;
        wp::float32 var_395;
        wp::vec_t<3,wp::float32> var_396;
        wp::vec_t<3,wp::float32> var_397;
        wp::float32 var_398;
        wp::float32 var_399;
        wp::int32 var_400;
        wp::float32* var_401;
        wp::int32 var_402;
        wp::float32* var_403;
        wp::int32 var_404;
        wp::float32* var_405;
        wp::int32 var_406;
        wp::float32* var_407;
        wp::int32 var_408;
        wp::float32* var_409;
        wp::int32 var_410;
        wp::float32* var_411;
        wp::int32 var_412;
        wp::float32* var_413;
        wp::int32 var_414;
        wp::float32* var_415;
        wp::float32 var_416;
        wp::float32 var_417;
        wp::float32 var_418;
        wp::float32 var_419;
        wp::float32 var_420;
        wp::float32 var_421;
        wp::float32 var_422;
        wp::float32 var_423;
        wp::vec_t<3,wp::float32> var_424;
        wp::vec_t<3,wp::float32> var_425;
        wp::float32 var_426;
        wp::float32 var_427;
        wp::float32 var_428;
        wp::vec_t<3,wp::float32> var_429;
        wp::vec_t<3,wp::float32> var_430;
        wp::vec_t<3,wp::float32> var_431;
        wp::vec_t<3,wp::float32> var_432;
        wp::vec_t<3,wp::float32> var_433;
        wp::vec_t<3,wp::float32> var_434;
        wp::vec_t<3,wp::float32> var_435;
        wp::vec_t<3,wp::float32> var_436;
        wp::quat_t<wp::float32> var_437;
        wp::vec_t<3,wp::float32> var_438;
        wp::vec_t<3,wp::float32> var_439;
        wp::quat_t<wp::float32> var_440;
        wp::vec_t<3,wp::float32> var_441;
        wp::quat_t<wp::float32> var_442;
        wp::vec_t<3,wp::float32> var_443;
        wp::quat_t<wp::float32> var_444;
        wp::quat_t<wp::float32> var_445;
        bool var_446;
        wp::vec_t<3,wp::float32> var_447;
        wp::vec_t<3,wp::float32> var_448;
        wp::vec_t<6,wp::float32> var_449;
        wp::vec_t<6,wp::float32> var_450;
        wp::vec_t<3,wp::float32> var_451;
        wp::vec_t<3,wp::float32> var_452;
        wp::vec_t<6,wp::float32> var_453;
        wp::vec_t<6,wp::float32> var_454;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        bool adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        bool adj_9 = {};
        bool adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::transform_t<wp::float32> adj_17 = {};
        wp::transform_t<wp::float32> adj_18 = {};
        wp::transform_t<wp::float32> adj_19 = {};
        wp::transform_t<wp::float32> adj_20 = {};
        wp::transform_t<wp::float32> adj_21 = {};
        wp::transform_t<wp::float32> adj_22 = {};
        wp::transform_t<wp::float32> adj_23 = {};
        wp::vec_t<3,wp::float32> adj_24 = {};
        wp::vec_t<3,wp::float32> adj_25 = {};
        wp::vec_t<3,wp::float32> adj_26 = {};
        bool adj_27 = {};
        wp::transform_t<wp::float32> adj_28 = {};
        wp::transform_t<wp::float32> adj_29 = {};
        wp::transform_t<wp::float32> adj_30 = {};
        wp::vec_t<3,wp::float32> adj_31 = {};
        wp::transform_t<wp::float32> adj_32 = {};
        wp::vec_t<3,wp::float32> adj_33 = {};
        wp::transform_t<wp::float32> adj_34 = {};
        wp::vec_t<3,wp::float32> adj_35 = {};
        wp::vec_t<3,wp::float32> adj_36 = {};
        wp::vec_t<3,wp::float32> adj_37 = {};
        wp::vec_t<6,wp::float32> adj_38 = {};
        wp::vec_t<6,wp::float32> adj_39 = {};
        wp::vec_t<6,wp::float32> adj_40 = {};
        wp::vec_t<3,wp::float32> adj_41 = {};
        wp::vec_t<3,wp::float32> adj_42 = {};
        wp::vec_t<3,wp::float32> adj_43 = {};
        wp::vec_t<3,wp::float32> adj_44 = {};
        wp::transform_t<wp::float32> adj_45 = {};
        wp::vec_t<3,wp::float32> adj_46 = {};
        wp::vec_t<3,wp::float32> adj_47 = {};
        wp::vec_t<3,wp::float32> adj_48 = {};
        wp::transform_t<wp::float32> adj_49 = {};
        wp::transform_t<wp::float32> adj_50 = {};
        wp::transform_t<wp::float32> adj_51 = {};
        wp::vec_t<3,wp::float32> adj_52 = {};
        wp::transform_t<wp::float32> adj_53 = {};
        wp::vec_t<3,wp::float32> adj_54 = {};
        wp::transform_t<wp::float32> adj_55 = {};
        wp::vec_t<3,wp::float32> adj_56 = {};
        wp::vec_t<3,wp::float32> adj_57 = {};
        wp::vec_t<3,wp::float32> adj_58 = {};
        wp::vec_t<6,wp::float32> adj_59 = {};
        wp::vec_t<6,wp::float32> adj_60 = {};
        wp::vec_t<6,wp::float32> adj_61 = {};
        wp::vec_t<3,wp::float32> adj_62 = {};
        wp::vec_t<3,wp::float32> adj_63 = {};
        wp::vec_t<3,wp::float32> adj_64 = {};
        wp::vec_t<3,wp::float32> adj_65 = {};
        wp::int32 adj_66 = {};
        wp::int32 adj_67 = {};
        wp::int32 adj_68 = {};
        wp::int32 adj_69 = {};
        wp::int32 adj_70 = {};
        wp::int32 adj_71 = {};
        wp::int32 adj_72 = {};
        wp::int32 adj_73 = {};
        wp::int32 adj_74 = {};
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
        wp::float32 adj_95 = {};
        wp::float32 adj_96 = {};
        wp::float32 adj_97 = {};
        wp::float32 adj_98 = {};
        wp::vec_t<3,wp::float32> adj_99 = {};
        wp::vec_t<3,wp::float32> adj_100 = {};
        wp::quat_t<wp::float32> adj_101 = {};
        wp::quat_t<wp::float32> adj_102 = {};
        wp::vec_t<3,wp::float32> adj_103 = {};
        wp::quat_t<wp::float32> adj_104 = {};
        wp::quat_t<wp::float32> adj_105 = {};
        wp::vec_t<3,wp::float32> adj_106 = {};
        wp::vec_t<3,wp::float32> adj_107 = {};
        wp::vec_t<3,wp::float32> adj_108 = {};
        wp::vec_t<3,wp::float32> adj_109 = {};
        wp::float32 adj_110 = {};
        wp::int32 adj_111 = {};
        bool adj_112 = {};
        wp::float32 adj_113 = {};
        wp::int32 adj_114 = {};
        wp::float32 adj_115 = {};
        wp::int32 adj_116 = {};
        wp::float32 adj_117 = {};
        wp::vec_t<3,wp::float32> adj_118 = {};
        wp::vec_t<3,wp::float32> adj_119 = {};
        wp::int32 adj_120 = {};
        wp::float32 adj_121 = {};
        wp::float32 adj_122 = {};
        wp::vec_t<3,wp::float32> adj_123 = {};
        wp::float32 adj_124 = {};
        wp::vec_t<3,wp::float32> adj_125 = {};
        wp::vec_t<3,wp::float32> adj_126 = {};
        wp::vec_t<3,wp::float32> adj_127 = {};
        wp::vec_t<3,wp::float32> adj_128 = {};
        wp::vec_t<3,wp::float32> adj_129 = {};
        wp::vec_t<3,wp::float32> adj_130 = {};
        wp::vec_t<3,wp::float32> adj_131 = {};
        wp::vec_t<3,wp::float32> adj_132 = {};
        wp::vec_t<3,wp::float32> adj_133 = {};
        wp::vec_t<3,wp::float32> adj_134 = {};
        wp::vec_t<3,wp::float32> adj_135 = {};
        wp::vec_t<3,wp::float32> adj_136 = {};
        wp::vec_t<3,wp::float32> adj_137 = {};
        wp::int32 adj_138 = {};
        bool adj_139 = {};
        wp::vec_t<3,wp::float32> adj_140 = {};
        wp::vec_t<3,wp::float32> adj_141 = {};
        wp::vec_t<3,wp::float32> adj_142 = {};
        wp::vec_t<3,wp::float32> adj_143 = {};
        wp::float32 adj_144 = {};
        wp::float32 adj_145 = {};
        wp::vec_t<3,wp::float32> adj_146 = {};
        wp::float32 adj_147 = {};
        wp::float32 adj_148 = {};
        wp::float32 adj_149 = {};
        wp::vec_t<3,wp::float32> adj_150 = {};
        wp::vec_t<3,wp::float32> adj_151 = {};
        wp::float32 adj_152 = {};
        wp::float32 adj_153 = {};
        wp::vec_t<3,wp::float32> adj_154 = {};
        wp::vec_t<3,wp::float32> adj_155 = {};
        wp::vec_t<3,wp::float32> adj_156 = {};
        wp::vec_t<3,wp::float32> adj_157 = {};
        wp::vec_t<3,wp::float32> adj_158 = {};
        wp::vec_t<3,wp::float32> adj_159 = {};
        wp::vec_t<3,wp::float32> adj_160 = {};
        wp::vec_t<3,wp::float32> adj_161 = {};
        wp::vec_t<3,wp::float32> adj_162 = {};
        wp::vec_t<3,wp::float32> adj_163 = {};
        wp::vec_t<3,wp::float32> adj_164 = {};
        wp::vec_t<3,wp::float32> adj_165 = {};
        wp::vec_t<3,wp::float32> adj_166 = {};
        wp::vec_t<3,wp::float32> adj_167 = {};
        wp::vec_t<3,wp::float32> adj_168 = {};
        wp::vec_t<3,wp::float32> adj_169 = {};
        wp::vec_t<3,wp::float32> adj_170 = {};
        wp::vec_t<3,wp::float32> adj_171 = {};
        wp::vec_t<3,wp::float32> adj_172 = {};
        wp::int32 adj_173 = {};
        bool adj_174 = {};
        wp::vec_t<3,wp::float32> adj_175 = {};
        wp::vec_t<3,wp::float32> adj_176 = {};
        wp::vec_t<3,wp::float32> adj_177 = {};
        wp::vec_t<3,wp::float32> adj_178 = {};
        wp::vec_t<3,wp::float32> adj_179 = {};
        wp::quat_t<wp::float32> adj_180 = {};
        wp::float32 adj_181 = {};
        wp::float32 adj_182 = {};
        wp::float32 adj_183 = {};
        wp::float32 adj_184 = {};
        wp::float32 adj_185 = {};
        wp::float32 adj_186 = {};
        wp::vec_t<3,wp::float32> adj_187 = {};
        wp::float32 adj_188 = {};
        wp::float32 adj_189 = {};
        wp::float32 adj_190 = {};
        wp::float32 adj_191 = {};
        wp::vec_t<3,wp::float32> adj_192 = {};
        wp::vec_t<3,wp::float32> adj_193 = {};
        wp::vec_t<3,wp::float32> adj_194 = {};
        wp::vec_t<3,wp::float32> adj_195 = {};
        wp::vec_t<3,wp::float32> adj_196 = {};
        wp::vec_t<3,wp::float32> adj_197 = {};
        wp::vec_t<3,wp::float32> adj_198 = {};
        wp::vec_t<3,wp::float32> adj_199 = {};
        wp::vec_t<3,wp::float32> adj_200 = {};
        wp::vec_t<3,wp::float32> adj_201 = {};
        wp::vec_t<3,wp::float32> adj_202 = {};
        wp::vec_t<3,wp::float32> adj_203 = {};
        wp::vec_t<3,wp::float32> adj_204 = {};
        wp::vec_t<3,wp::float32> adj_205 = {};
        wp::vec_t<3,wp::float32> adj_206 = {};
        wp::vec_t<3,wp::float32> adj_207 = {};
        wp::vec_t<3,wp::float32> adj_208 = {};
        wp::float32 adj_209 = {};
        wp::float32 adj_210 = {};
        wp::int32 adj_211 = {};
        bool adj_212 = {};
        wp::float32 adj_213 = {};
        wp::float32 adj_214 = {};
        wp::float32 adj_215 = {};
        wp::vec_t<3,wp::float32> adj_216 = {};
        wp::vec_t<3,wp::float32> adj_217 = {};
        wp::float32 adj_218 = {};
        wp::float32 adj_219 = {};
        wp::vec_t<3,wp::float32> adj_220 = {};
        wp::vec_t<3,wp::float32> adj_221 = {};
        wp::vec_t<3,wp::float32> adj_222 = {};
        wp::vec_t<3,wp::float32> adj_223 = {};
        wp::vec_t<3,wp::float32> adj_224 = {};
        wp::vec_t<3,wp::float32> adj_225 = {};
        wp::vec_t<3,wp::float32> adj_226 = {};
        wp::vec_t<3,wp::float32> adj_227 = {};
        wp::vec_t<3,wp::float32> adj_228 = {};
        wp::vec_t<3,wp::float32> adj_229 = {};
        wp::vec_t<3,wp::float32> adj_230 = {};
        wp::vec_t<3,wp::float32> adj_231 = {};
        wp::vec_t<3,wp::float32> adj_232 = {};
        wp::vec_t<3,wp::float32> adj_233 = {};
        wp::int32 adj_234 = {};
        bool adj_235 = {};
        wp::quat_t<wp::float32> adj_236 = {};
        wp::quat_t<wp::float32> adj_237 = {};
        wp::vec_t<3,wp::float32> adj_238 = {};
        wp::float32 adj_239 = {};
        wp::float32 adj_240 = {};
        wp::vec_t<3,wp::float32> adj_241 = {};
        wp::float32 adj_242 = {};
        wp::quat_t<wp::float32> adj_243 = {};
        wp::vec_t<3,wp::float32> adj_244 = {};
        wp::vec_t<3,wp::float32> adj_245 = {};
        wp::float32 adj_246 = {};
        wp::quat_t<wp::float32> adj_247 = {};
        wp::quat_t<wp::float32> adj_248 = {};
        wp::vec_t<3,wp::float32> adj_249 = {};
        wp::vec_t<3,wp::float32> adj_250 = {};
        wp::float32 adj_251 = {};
        wp::quat_t<wp::float32> adj_252 = {};
        wp::quat_t<wp::float32> adj_253 = {};
        wp::vec_t<3,wp::float32> adj_254 = {};
        wp::vec_t<3,wp::float32> adj_255 = {};
        wp::vec_t<3,wp::float32> adj_256 = {};
        wp::vec_t<3,wp::float32> adj_257 = {};
        wp::float32 adj_258 = {};
        wp::float32 adj_259 = {};
        wp::int32 adj_260 = {};
        wp::float32 adj_261 = {};
        wp::int32 adj_262 = {};
        wp::float32 adj_263 = {};
        wp::int32 adj_264 = {};
        wp::float32 adj_265 = {};
        wp::int32 adj_266 = {};
        wp::float32 adj_267 = {};
        wp::int32 adj_268 = {};
        wp::float32 adj_269 = {};
        wp::int32 adj_270 = {};
        wp::float32 adj_271 = {};
        wp::int32 adj_272 = {};
        wp::float32 adj_273 = {};
        wp::int32 adj_274 = {};
        wp::float32 adj_275 = {};
        wp::float32 adj_276 = {};
        wp::float32 adj_277 = {};
        wp::float32 adj_278 = {};
        wp::float32 adj_279 = {};
        wp::float32 adj_280 = {};
        wp::float32 adj_281 = {};
        wp::float32 adj_282 = {};
        wp::float32 adj_283 = {};
        wp::vec_t<3,wp::float32> adj_284 = {};
        wp::vec_t<3,wp::float32> adj_285 = {};
        wp::float32 adj_286 = {};
        wp::float32 adj_287 = {};
        wp::int32 adj_288 = {};
        wp::float32 adj_289 = {};
        wp::int32 adj_290 = {};
        wp::float32 adj_291 = {};
        wp::int32 adj_292 = {};
        wp::float32 adj_293 = {};
        wp::int32 adj_294 = {};
        wp::float32 adj_295 = {};
        wp::int32 adj_296 = {};
        wp::float32 adj_297 = {};
        wp::int32 adj_298 = {};
        wp::float32 adj_299 = {};
        wp::int32 adj_300 = {};
        wp::float32 adj_301 = {};
        wp::int32 adj_302 = {};
        wp::float32 adj_303 = {};
        wp::float32 adj_304 = {};
        wp::float32 adj_305 = {};
        wp::float32 adj_306 = {};
        wp::float32 adj_307 = {};
        wp::float32 adj_308 = {};
        wp::float32 adj_309 = {};
        wp::float32 adj_310 = {};
        wp::float32 adj_311 = {};
        wp::vec_t<3,wp::float32> adj_312 = {};
        wp::vec_t<3,wp::float32> adj_313 = {};
        wp::float32 adj_314 = {};
        wp::float32 adj_315 = {};
        wp::int32 adj_316 = {};
        wp::float32 adj_317 = {};
        wp::int32 adj_318 = {};
        wp::float32 adj_319 = {};
        wp::int32 adj_320 = {};
        wp::float32 adj_321 = {};
        wp::int32 adj_322 = {};
        wp::float32 adj_323 = {};
        wp::int32 adj_324 = {};
        wp::float32 adj_325 = {};
        wp::int32 adj_326 = {};
        wp::float32 adj_327 = {};
        wp::int32 adj_328 = {};
        wp::float32 adj_329 = {};
        wp::int32 adj_330 = {};
        wp::float32 adj_331 = {};
        wp::float32 adj_332 = {};
        wp::float32 adj_333 = {};
        wp::float32 adj_334 = {};
        wp::float32 adj_335 = {};
        wp::float32 adj_336 = {};
        wp::float32 adj_337 = {};
        wp::float32 adj_338 = {};
        wp::float32 adj_339 = {};
        wp::vec_t<3,wp::float32> adj_340 = {};
        wp::vec_t<3,wp::float32> adj_341 = {};
        wp::vec_t<3,wp::float32> adj_342 = {};
        wp::vec_t<3,wp::float32> adj_343 = {};
        wp::vec_t<3,wp::float32> adj_344 = {};
        wp::vec_t<3,wp::float32> adj_345 = {};
        wp::vec_t<3,wp::float32> adj_346 = {};
        wp::vec_t<3,wp::float32> adj_347 = {};
        wp::int32 adj_348 = {};
        bool adj_349 = {};
        wp::quat_t<wp::float32> adj_350 = {};
        wp::quat_t<wp::float32> adj_351 = {};
        wp::vec_t<3,wp::float32> adj_352 = {};
        wp::vec_t<3,wp::float32> adj_353 = {};
        wp::float32 adj_354 = {};
        wp::quat_t<wp::float32> adj_355 = {};
        wp::vec_t<3,wp::float32> adj_356 = {};
        wp::vec_t<3,wp::float32> adj_357 = {};
        wp::float32 adj_358 = {};
        wp::quat_t<wp::float32> adj_359 = {};
        wp::quat_t<wp::float32> adj_360 = {};
        wp::vec_t<3,wp::float32> adj_361 = {};
        wp::vec_t<3,wp::float32> adj_362 = {};
        wp::float32 adj_363 = {};
        wp::quat_t<wp::float32> adj_364 = {};
        wp::quat_t<wp::float32> adj_365 = {};
        wp::vec_t<3,wp::float32> adj_366 = {};
        wp::vec_t<3,wp::float32> adj_367 = {};
        wp::vec_t<3,wp::float32> adj_368 = {};
        wp::vec_t<3,wp::float32> adj_369 = {};
        wp::float32 adj_370 = {};
        wp::float32 adj_371 = {};
        wp::int32 adj_372 = {};
        wp::float32 adj_373 = {};
        wp::int32 adj_374 = {};
        wp::float32 adj_375 = {};
        wp::int32 adj_376 = {};
        wp::float32 adj_377 = {};
        wp::int32 adj_378 = {};
        wp::float32 adj_379 = {};
        wp::int32 adj_380 = {};
        wp::float32 adj_381 = {};
        wp::int32 adj_382 = {};
        wp::float32 adj_383 = {};
        wp::int32 adj_384 = {};
        wp::float32 adj_385 = {};
        wp::int32 adj_386 = {};
        wp::float32 adj_387 = {};
        wp::float32 adj_388 = {};
        wp::float32 adj_389 = {};
        wp::float32 adj_390 = {};
        wp::float32 adj_391 = {};
        wp::float32 adj_392 = {};
        wp::float32 adj_393 = {};
        wp::float32 adj_394 = {};
        wp::float32 adj_395 = {};
        wp::vec_t<3,wp::float32> adj_396 = {};
        wp::vec_t<3,wp::float32> adj_397 = {};
        wp::float32 adj_398 = {};
        wp::float32 adj_399 = {};
        wp::int32 adj_400 = {};
        wp::float32 adj_401 = {};
        wp::int32 adj_402 = {};
        wp::float32 adj_403 = {};
        wp::int32 adj_404 = {};
        wp::float32 adj_405 = {};
        wp::int32 adj_406 = {};
        wp::float32 adj_407 = {};
        wp::int32 adj_408 = {};
        wp::float32 adj_409 = {};
        wp::int32 adj_410 = {};
        wp::float32 adj_411 = {};
        wp::int32 adj_412 = {};
        wp::float32 adj_413 = {};
        wp::int32 adj_414 = {};
        wp::float32 adj_415 = {};
        wp::float32 adj_416 = {};
        wp::float32 adj_417 = {};
        wp::float32 adj_418 = {};
        wp::float32 adj_419 = {};
        wp::float32 adj_420 = {};
        wp::float32 adj_421 = {};
        wp::float32 adj_422 = {};
        wp::float32 adj_423 = {};
        wp::vec_t<3,wp::float32> adj_424 = {};
        wp::vec_t<3,wp::float32> adj_425 = {};
        wp::float32 adj_426 = {};
        wp::float32 adj_427 = {};
        wp::float32 adj_428 = {};
        wp::vec_t<3,wp::float32> adj_429 = {};
        wp::vec_t<3,wp::float32> adj_430 = {};
        wp::vec_t<3,wp::float32> adj_431 = {};
        wp::vec_t<3,wp::float32> adj_432 = {};
        wp::vec_t<3,wp::float32> adj_433 = {};
        wp::vec_t<3,wp::float32> adj_434 = {};
        wp::vec_t<3,wp::float32> adj_435 = {};
        wp::vec_t<3,wp::float32> adj_436 = {};
        wp::quat_t<wp::float32> adj_437 = {};
        wp::vec_t<3,wp::float32> adj_438 = {};
        wp::vec_t<3,wp::float32> adj_439 = {};
        wp::quat_t<wp::float32> adj_440 = {};
        wp::vec_t<3,wp::float32> adj_441 = {};
        wp::quat_t<wp::float32> adj_442 = {};
        wp::vec_t<3,wp::float32> adj_443 = {};
        wp::quat_t<wp::float32> adj_444 = {};
        wp::quat_t<wp::float32> adj_445 = {};
        bool adj_446 = {};
        wp::vec_t<3,wp::float32> adj_447 = {};
        wp::vec_t<3,wp::float32> adj_448 = {};
        wp::vec_t<6,wp::float32> adj_449 = {};
        wp::vec_t<6,wp::float32> adj_450 = {};
        wp::vec_t<3,wp::float32> adj_451 = {};
        wp::vec_t<3,wp::float32> adj_452 = {};
        wp::vec_t<6,wp::float32> adj_453 = {};
        wp::vec_t<6,wp::float32> adj_454 = {};
        //---------
        // forward
        // def eval_body_joints(                                                                  <L 1115>
        // tid = wp.tid()                                                                         <L 1142>
        var_0 = builtin_tid1d();
        // type = joint_type[tid]                                                                 <L 1143>
        var_1 = wp::address(var_joint_type, var_0);
        var_2 = wp::load(var_1);
        var_3 = wp::copy(var_2);
        // if joint_enabled[tid] == 0 or type == fs5Model.JOINT_FREE:                             <L 1146>
        var_4 = wp::address(var_joint_enabled, var_0);
        var_7 = wp::load(var_4);
        var_6 = (var_7 == var_5);
        var_9 = (var_3 == var_8);
        var_10 = var_6 || var_9;
        if (var_10) {
            // return                                                                             <L 1147>
            goto label0;
        }
        // c_child = joint_child[tid]                                                             <L 1149>
        var_11 = wp::address(var_joint_child, var_0);
        var_12 = wp::load(var_11);
        var_13 = wp::copy(var_12);
        // c_parent = joint_parent[tid]                                                           <L 1150>
        var_14 = wp::address(var_joint_parent, var_0);
        var_15 = wp::load(var_14);
        var_16 = wp::copy(var_15);
        // X_pj = joint_X_p[tid]                                                                  <L 1152>
        var_17 = wp::address(var_joint_X_p, var_0);
        var_18 = wp::load(var_17);
        var_19 = wp::copy(var_18);
        // X_cj = joint_X_c[tid]                                                                  <L 1153>
        var_20 = wp::address(var_joint_X_c, var_0);
        var_21 = wp::load(var_20);
        var_22 = wp::copy(var_21);
        // X_wp = X_pj                                                                            <L 1155>
        var_23 = wp::copy(var_19);
        // r_p = wp.vec3()                                                                        <L 1156>
        var_24 = wp::vec_t<3,wp::float32>();
        // w_p = wp.vec3()                                                                        <L 1157>
        var_25 = wp::vec_t<3,wp::float32>();
        // v_p = wp.vec3()                                                                        <L 1158>
        var_26 = wp::vec_t<3,wp::float32>();
        // if c_parent >= 0:                                                                      <L 1161>
        var_27 = (var_16 >= var_5);
        if (var_27) {
            // X_wp = body_q[c_parent] * X_wp                                                     <L 1162>
            var_28 = wp::address(var_body_q, var_16);
            var_29 = wp::load(var_28);
            var_30 = wp::mul(var_29, var_23);
            // r_p = wp.transform_get_translation(X_wp) - wp.transform_point(body_q[c_parent], body_com[c_parent])       <L 1163>
            var_31 = wp::transform_get_translation(var_30);
            var_32 = wp::address(var_body_q, var_16);
            var_33 = wp::address(var_body_com, var_16);
            var_34 = wp::load(var_32);
            var_35 = wp::load(var_33);
            var_36 = wp::transform_point(var_34, var_35);
            var_37 = wp::sub(var_31, var_36);
            // twist_p = body_qd[c_parent]                                                        <L 1165>
            var_38 = wp::address(var_body_qd, var_16);
            var_39 = wp::load(var_38);
            var_40 = wp::copy(var_39);
            // w_p = wp.spatial_top(twist_p)                                                      <L 1167>
            var_41 = wp::spatial_top(var_40);
            // v_p = wp.spatial_bottom(twist_p) + wp.cross(w_p, r_p)                              <L 1168>
            var_42 = wp::spatial_bottom(var_40);
            var_43 = wp::cross(var_41, var_37);
            var_44 = wp::add(var_42, var_43);
        }
        var_45 = wp::select(var_27, var_23, var_30);
        var_46 = wp::select(var_27, var_24, var_37);
        var_47 = wp::select(var_27, var_25, var_41);
        var_48 = wp::select(var_27, var_26, var_44);
        // X_wc = body_q[c_child] * X_cj                                                          <L 1171>
        var_49 = wp::address(var_body_q, var_13);
        var_50 = wp::load(var_49);
        var_51 = wp::mul(var_50, var_22);
        // r_c = wp.transform_get_translation(X_wc) - wp.transform_point(body_q[c_child], body_com[c_child])       <L 1172>
        var_52 = wp::transform_get_translation(var_51);
        var_53 = wp::address(var_body_q, var_13);
        var_54 = wp::address(var_body_com, var_13);
        var_55 = wp::load(var_53);
        var_56 = wp::load(var_54);
        var_57 = wp::transform_point(var_55, var_56);
        var_58 = wp::sub(var_52, var_57);
        // twist_c = body_qd[c_child]                                                             <L 1174>
        var_59 = wp::address(var_body_qd, var_13);
        var_60 = wp::load(var_59);
        var_61 = wp::copy(var_60);
        // w_c = wp.spatial_top(twist_c)                                                          <L 1176>
        var_62 = wp::spatial_top(var_61);
        // v_c = wp.spatial_bottom(twist_c) + wp.cross(w_c, r_c)                                  <L 1177>
        var_63 = wp::spatial_bottom(var_61);
        var_64 = wp::cross(var_62, var_58);
        var_65 = wp::add(var_63, var_64);
        // q_start = joint_q_start[tid]                                                           <L 1180>
        var_66 = wp::address(var_joint_q_start, var_0);
        var_67 = wp::load(var_66);
        var_68 = wp::copy(var_67);
        // qd_start = joint_qd_start[tid]                                                         <L 1181>
        var_69 = wp::address(var_joint_qd_start, var_0);
        var_70 = wp::load(var_69);
        var_71 = wp::copy(var_70);
        // axis_start = joint_axis_start[tid]                                                     <L 1182>
        var_72 = wp::address(var_joint_axis_start, var_0);
        var_73 = wp::load(var_72);
        var_74 = wp::copy(var_73);
        // target = joint_target[axis_start]                                                      <L 1184>
        var_75 = wp::address(var_joint_target, var_74);
        var_76 = wp::load(var_75);
        var_77 = wp::copy(var_76);
        // target_ke = joint_target_ke[axis_start]                                                <L 1185>
        var_78 = wp::address(var_joint_target_ke, var_74);
        var_79 = wp::load(var_78);
        var_80 = wp::copy(var_79);
        // target_kd = joint_target_kd[axis_start]                                                <L 1186>
        var_81 = wp::address(var_joint_target_kd, var_74);
        var_82 = wp::load(var_81);
        var_83 = wp::copy(var_82);
        // limit_ke = joint_limit_ke[axis_start]                                                  <L 1187>
        var_84 = wp::address(var_joint_limit_ke, var_74);
        var_85 = wp::load(var_84);
        var_86 = wp::copy(var_85);
        // limit_kd = joint_limit_kd[axis_start]                                                  <L 1188>
        var_87 = wp::address(var_joint_limit_kd, var_74);
        var_88 = wp::load(var_87);
        var_89 = wp::copy(var_88);
        // limit_lower = joint_limit_lower[axis_start]                                            <L 1189>
        var_90 = wp::address(var_joint_limit_lower, var_74);
        var_91 = wp::load(var_90);
        var_92 = wp::copy(var_91);
        // limit_upper = joint_limit_upper[axis_start]                                            <L 1190>
        var_93 = wp::address(var_joint_limit_upper, var_74);
        var_94 = wp::load(var_93);
        var_95 = wp::copy(var_94);
        // act = joint_act[qd_start]                                                              <L 1192>
        var_96 = wp::address(var_joint_act, var_71);
        var_97 = wp::load(var_96);
        var_98 = wp::copy(var_97);
        // x_p = wp.transform_get_translation(X_wp)                                               <L 1194>
        var_99 = wp::transform_get_translation(var_45);
        // x_c = wp.transform_get_translation(X_wc)                                               <L 1195>
        var_100 = wp::transform_get_translation(var_51);
        // q_p = wp.transform_get_rotation(X_wp)                                                  <L 1197>
        var_101 = wp::transform_get_rotation(var_45);
        // q_c = wp.transform_get_rotation(X_wc)                                                  <L 1198>
        var_102 = wp::transform_get_rotation(var_51);
        // x_err = x_c - x_p                                                                      <L 1201>
        var_103 = wp::sub(var_100, var_99);
        // r_err = wp.quat_inverse(q_p) * q_c                                                     <L 1202>
        var_104 = wp::quat_inverse(var_101);
        var_105 = wp::mul(var_104, var_102);
        // v_err = v_c - v_p                                                                      <L 1203>
        var_106 = wp::sub(var_65, var_48);
        // w_err = w_c - w_p                                                                      <L 1204>
        var_107 = wp::sub(var_62, var_47);
        // t_total = wp.vec3()                                                                    <L 1207>
        var_108 = wp::vec_t<3,wp::float32>();
        // f_total = wp.vec3()                                                                    <L 1208>
        var_109 = wp::vec_t<3,wp::float32>();
        // angular_damping_scale = 0.01                                                           <L 1211>
        // if type == fs5Model.JOINT_FIXED:                                                       <L 1213>
        var_112 = (var_3 == var_111);
        if (var_112) {
            // ang_err = wp.normalize(wp.vec3(r_err[0], r_err[1], r_err[2])) * wp.acos(r_err[3]) * 2.0       <L 1214>
            var_113 = wp::extract(var_105, var_5);
            var_115 = wp::extract(var_105, var_114);
            var_117 = wp::extract(var_105, var_116);
            var_118 = wp::vec_t<3,wp::float32>(var_113, var_115, var_117);
            var_119 = wp::normalize(var_118);
            var_121 = wp::extract(var_105, var_120);
            var_122 = wp::acos(var_121);
            var_123 = wp::mul(var_119, var_122);
            var_125 = wp::mul(var_123, var_124);
            // f_total += x_err * joint_attach_ke + v_err * joint_attach_kd                       <L 1216>
            var_126 = wp::mul(var_103, var_joint_attach_ke);
            var_127 = wp::mul(var_106, var_joint_attach_kd);
            var_128 = wp::add(var_126, var_127);
            var_129 = wp::add(var_109, var_128);
            // t_total += (                                                                       <L 1217>
            // wp.transform_vector(X_wp, ang_err) * joint_attach_ke + w_err * joint_attach_kd * angular_damping_scale       <L 1218>
            var_130 = wp::transform_vector(var_45, var_125);
            var_131 = wp::mul(var_130, var_joint_attach_ke);
            var_132 = wp::mul(var_107, var_joint_attach_kd);
            var_133 = wp::mul(var_132, var_110);
            var_134 = wp::add(var_131, var_133);
            var_135 = wp::add(var_108, var_134);
        }
        var_136 = wp::select(var_112, var_108, var_135);
        var_137 = wp::select(var_112, var_109, var_129);
        // if type == fs5Model.JOINT_PRISMATIC:                                                   <L 1221>
        var_139 = (var_3 == var_138);
        if (var_139) {
            // axis = joint_axis[axis_start]                                                      <L 1222>
            var_140 = wp::address(var_joint_axis, var_74);
            var_141 = wp::load(var_140);
            var_142 = wp::copy(var_141);
            // axis_p = wp.transform_vector(X_wp, axis)                                           <L 1225>
            var_143 = wp::transform_vector(var_45, var_142);
            // q = wp.dot(x_err, axis_p)                                                          <L 1228>
            var_144 = wp::dot(var_103, var_143);
            // qd = wp.dot(v_err, axis_p)                                                         <L 1229>
            var_145 = wp::dot(var_106, var_143);
            // f_total = eval_joint_force(                                                        <L 1231>
            // q, qd, target, target_ke, target_kd, act, limit_lower, limit_upper, limit_ke, limit_kd, axis_p       <L 1232>
            var_146 = eval_joint_force(var_144, var_145, var_77, var_80, var_83, var_98, var_92, var_95, var_86, var_89, var_143);
            // ang_err = wp.normalize(wp.vec3(r_err[0], r_err[1], r_err[2])) * wp.acos(r_err[3]) * 2.0       <L 1236>
            var_147 = wp::extract(var_105, var_5);
            var_148 = wp::extract(var_105, var_114);
            var_149 = wp::extract(var_105, var_116);
            var_150 = wp::vec_t<3,wp::float32>(var_147, var_148, var_149);
            var_151 = wp::normalize(var_150);
            var_152 = wp::extract(var_105, var_120);
            var_153 = wp::acos(var_152);
            var_154 = wp::mul(var_151, var_153);
            var_155 = wp::mul(var_154, var_124);
            // f_total += (x_err - q * axis_p) * joint_attach_ke + (v_err - qd * axis_p) * joint_attach_kd       <L 1239>
            var_156 = wp::mul(var_144, var_143);
            var_157 = wp::sub(var_103, var_156);
            var_158 = wp::mul(var_157, var_joint_attach_ke);
            var_159 = wp::mul(var_145, var_143);
            var_160 = wp::sub(var_106, var_159);
            var_161 = wp::mul(var_160, var_joint_attach_kd);
            var_162 = wp::add(var_158, var_161);
            var_163 = wp::add(var_146, var_162);
            // t_total += (                                                                       <L 1240>
            // wp.transform_vector(X_wp, ang_err) * joint_attach_ke + w_err * joint_attach_kd * angular_damping_scale       <L 1241>
            var_164 = wp::transform_vector(var_45, var_155);
            var_165 = wp::mul(var_164, var_joint_attach_ke);
            var_166 = wp::mul(var_107, var_joint_attach_kd);
            var_167 = wp::mul(var_166, var_110);
            var_168 = wp::add(var_165, var_167);
            var_169 = wp::add(var_136, var_168);
        }
        var_170 = wp::select(var_139, var_136, var_169);
        var_171 = wp::select(var_139, var_137, var_163);
        var_172 = wp::select(var_139, var_125, var_155);
        // if type == fs5Model.JOINT_REVOLUTE:                                                    <L 1244>
        var_174 = (var_3 == var_173);
        if (var_174) {
            // axis = joint_axis[axis_start]                                                      <L 1245>
            var_175 = wp::address(var_joint_axis, var_74);
            var_176 = wp::load(var_175);
            var_177 = wp::copy(var_176);
            // axis_p = wp.transform_vector(X_wp, axis)                                           <L 1247>
            var_178 = wp::transform_vector(var_45, var_177);
            // axis_c = wp.transform_vector(X_wc, axis)                                           <L 1248>
            var_179 = wp::transform_vector(var_51, var_177);
            // twist = quat_twist(axis, r_err)                                                    <L 1251>
            var_180 = quat_twist(var_177, var_105);
            // q = wp.acos(twist[3]) * 2.0 * wp.sign(wp.dot(axis, wp.vec3(twist[0], twist[1], twist[2])))       <L 1253>
            var_181 = wp::extract(var_180, var_120);
            var_182 = wp::acos(var_181);
            var_183 = wp::mul(var_182, var_124);
            var_184 = wp::extract(var_180, var_5);
            var_185 = wp::extract(var_180, var_114);
            var_186 = wp::extract(var_180, var_116);
            var_187 = wp::vec_t<3,wp::float32>(var_184, var_185, var_186);
            var_188 = wp::dot(var_177, var_187);
            var_189 = wp::sign(var_188);
            var_190 = wp::mul(var_183, var_189);
            // qd = wp.dot(w_err, axis_p)                                                         <L 1254>
            var_191 = wp::dot(var_107, var_178);
            // t_total = eval_joint_force(                                                        <L 1256>
            // q, qd, target, target_ke, target_kd, act, limit_lower, limit_upper, limit_ke, limit_kd, axis_p       <L 1257>
            var_192 = eval_joint_force(var_190, var_191, var_77, var_80, var_83, var_98, var_92, var_95, var_86, var_89, var_178);
            // swing_err = wp.cross(axis_p, axis_c)                                               <L 1261>
            var_193 = wp::cross(var_178, var_179);
            // f_total += x_err * joint_attach_ke + v_err * joint_attach_kd                       <L 1263>
            var_194 = wp::mul(var_103, var_joint_attach_ke);
            var_195 = wp::mul(var_106, var_joint_attach_kd);
            var_196 = wp::add(var_194, var_195);
            var_197 = wp::add(var_171, var_196);
            // t_total += swing_err * joint_attach_ke + (w_err - qd * axis_p) * joint_attach_kd * angular_damping_scale       <L 1264>
            var_198 = wp::mul(var_193, var_joint_attach_ke);
            var_199 = wp::mul(var_191, var_178);
            var_200 = wp::sub(var_107, var_199);
            var_201 = wp::mul(var_200, var_joint_attach_kd);
            var_202 = wp::mul(var_201, var_110);
            var_203 = wp::add(var_198, var_202);
            var_204 = wp::add(var_192, var_203);
        }
        var_205 = wp::select(var_174, var_170, var_204);
        var_206 = wp::select(var_174, var_171, var_197);
        var_207 = wp::select(var_174, var_142, var_177);
        var_208 = wp::select(var_174, var_143, var_178);
        var_209 = wp::select(var_174, var_144, var_190);
        var_210 = wp::select(var_174, var_145, var_191);
        // if type == fs5Model.JOINT_BALL:                                                        <L 1266>
        var_212 = (var_3 == var_211);
        if (var_212) {
            // ang_err = wp.normalize(wp.vec3(r_err[0], r_err[1], r_err[2])) * wp.acos(r_err[3]) * 2.0       <L 1267>
            var_213 = wp::extract(var_105, var_5);
            var_214 = wp::extract(var_105, var_114);
            var_215 = wp::extract(var_105, var_116);
            var_216 = wp::vec_t<3,wp::float32>(var_213, var_214, var_215);
            var_217 = wp::normalize(var_216);
            var_218 = wp::extract(var_105, var_120);
            var_219 = wp::acos(var_218);
            var_220 = wp::mul(var_217, var_219);
            var_221 = wp::mul(var_220, var_124);
            // t_total += target_kd * w_err + target_ke * wp.transform_vector(X_wp, ang_err)       <L 1270>
            var_222 = wp::mul(var_83, var_107);
            var_223 = wp::transform_vector(var_45, var_221);
            var_224 = wp::mul(var_80, var_223);
            var_225 = wp::add(var_222, var_224);
            var_226 = wp::add(var_205, var_225);
            // f_total += x_err * joint_attach_ke + v_err * joint_attach_kd                       <L 1271>
            var_227 = wp::mul(var_103, var_joint_attach_ke);
            var_228 = wp::mul(var_106, var_joint_attach_kd);
            var_229 = wp::add(var_227, var_228);
            var_230 = wp::add(var_206, var_229);
        }
        var_231 = wp::select(var_212, var_205, var_226);
        var_232 = wp::select(var_212, var_206, var_230);
        var_233 = wp::select(var_212, var_172, var_221);
        // if type == fs5Model.JOINT_COMPOUND:                                                    <L 1273>
        var_235 = (var_3 == var_234);
        if (var_235) {
            // q_pc = wp.quat_inverse(q_p) * q_c                                                  <L 1274>
            var_236 = wp::quat_inverse(var_101);
            var_237 = wp::mul(var_236, var_102);
            // angles = quat_decompose(q_pc)                                                      <L 1277>
            var_238 = quat_decompose(var_237);
            // axis_0 = wp.vec3(1.0, 0.0, 0.0)                                                    <L 1280>
            var_241 = wp::vec_t<3,wp::float32>(var_239, var_240, var_240);
            // q_0 = wp.quat_from_axis_angle(axis_0, angles[0])                                   <L 1281>
            var_242 = wp::extract(var_238, var_5);
            var_243 = wp::quat_from_axis_angle(var_241, var_242);
            // axis_1 = wp.quat_rotate(q_0, wp.vec3(0.0, 1.0, 0.0))                               <L 1283>
            var_244 = wp::vec_t<3,wp::float32>(var_240, var_239, var_240);
            var_245 = wp::quat_rotate(var_243, var_244);
            // q_1 = wp.quat_from_axis_angle(axis_1, angles[1])                                   <L 1284>
            var_246 = wp::extract(var_238, var_114);
            var_247 = wp::quat_from_axis_angle(var_245, var_246);
            // axis_2 = wp.quat_rotate(q_1 * q_0, wp.vec3(0.0, 0.0, 1.0))                         <L 1286>
            var_248 = wp::mul(var_247, var_243);
            var_249 = wp::vec_t<3,wp::float32>(var_240, var_240, var_239);
            var_250 = wp::quat_rotate(var_248, var_249);
            // q_2 = wp.quat_from_axis_angle(axis_2, angles[2])                                   <L 1287>
            var_251 = wp::extract(var_238, var_116);
            var_252 = wp::quat_from_axis_angle(var_250, var_251);
            // q_w = q_p                                                                          <L 1289>
            var_253 = wp::copy(var_101);
            // axis_0 = wp.transform_vector(X_wp, axis_0)                                         <L 1291>
            var_254 = wp::transform_vector(var_45, var_241);
            // axis_1 = wp.transform_vector(X_wp, axis_1)                                         <L 1292>
            var_255 = wp::transform_vector(var_45, var_245);
            // axis_2 = wp.transform_vector(X_wp, axis_2)                                         <L 1293>
            var_256 = wp::transform_vector(var_45, var_250);
            // t_total = wp.vec3()                                                                <L 1296>
            var_257 = wp::vec_t<3,wp::float32>();
            // t_total += eval_joint_force(                                                       <L 1302>
            // angles[0],                                                                         <L 1303>
            var_258 = wp::extract(var_238, var_5);
            // wp.dot(axis_0, w_err),                                                             <L 1304>
            var_259 = wp::dot(var_254, var_107);
            // joint_target[axis_start + 0],                                                      <L 1305>
            var_260 = wp::add(var_74, var_5);
            var_261 = wp::address(var_joint_target, var_260);
            // joint_target_ke[axis_start + 0],                                                   <L 1306>
            var_262 = wp::add(var_74, var_5);
            var_263 = wp::address(var_joint_target_ke, var_262);
            // joint_target_kd[axis_start + 0],                                                   <L 1307>
            var_264 = wp::add(var_74, var_5);
            var_265 = wp::address(var_joint_target_kd, var_264);
            // joint_act[axis_start + 0],                                                         <L 1308>
            var_266 = wp::add(var_74, var_5);
            var_267 = wp::address(var_joint_act, var_266);
            // joint_limit_lower[axis_start + 0],                                                 <L 1309>
            var_268 = wp::add(var_74, var_5);
            var_269 = wp::address(var_joint_limit_lower, var_268);
            // joint_limit_upper[axis_start + 0],                                                 <L 1310>
            var_270 = wp::add(var_74, var_5);
            var_271 = wp::address(var_joint_limit_upper, var_270);
            // joint_limit_ke[axis_start + 0],                                                    <L 1311>
            var_272 = wp::add(var_74, var_5);
            var_273 = wp::address(var_joint_limit_ke, var_272);
            // joint_limit_kd[axis_start + 0],                                                    <L 1312>
            var_274 = wp::add(var_74, var_5);
            var_275 = wp::address(var_joint_limit_kd, var_274);
            // axis_0,                                                                            <L 1313>
            var_276 = wp::load(var_261);
            var_277 = wp::load(var_263);
            var_278 = wp::load(var_265);
            var_279 = wp::load(var_267);
            var_280 = wp::load(var_269);
            var_281 = wp::load(var_271);
            var_282 = wp::load(var_273);
            var_283 = wp::load(var_275);
            var_284 = eval_joint_force(var_258, var_259, var_276, var_277, var_278, var_279, var_280, var_281, var_282, var_283, var_254);
            var_285 = wp::add(var_257, var_284);
            // t_total += eval_joint_force(                                                       <L 1315>
            // angles[1],                                                                         <L 1316>
            var_286 = wp::extract(var_238, var_114);
            // wp.dot(axis_1, w_err),                                                             <L 1317>
            var_287 = wp::dot(var_255, var_107);
            // joint_target[axis_start + 1],                                                      <L 1318>
            var_288 = wp::add(var_74, var_114);
            var_289 = wp::address(var_joint_target, var_288);
            // joint_target_ke[axis_start + 1],                                                   <L 1319>
            var_290 = wp::add(var_74, var_114);
            var_291 = wp::address(var_joint_target_ke, var_290);
            // joint_target_kd[axis_start + 1],                                                   <L 1320>
            var_292 = wp::add(var_74, var_114);
            var_293 = wp::address(var_joint_target_kd, var_292);
            // joint_act[axis_start + 1],                                                         <L 1321>
            var_294 = wp::add(var_74, var_114);
            var_295 = wp::address(var_joint_act, var_294);
            // joint_limit_lower[axis_start + 1],                                                 <L 1322>
            var_296 = wp::add(var_74, var_114);
            var_297 = wp::address(var_joint_limit_lower, var_296);
            // joint_limit_upper[axis_start + 1],                                                 <L 1323>
            var_298 = wp::add(var_74, var_114);
            var_299 = wp::address(var_joint_limit_upper, var_298);
            // joint_limit_ke[axis_start + 1],                                                    <L 1324>
            var_300 = wp::add(var_74, var_114);
            var_301 = wp::address(var_joint_limit_ke, var_300);
            // joint_limit_kd[axis_start + 1],                                                    <L 1325>
            var_302 = wp::add(var_74, var_114);
            var_303 = wp::address(var_joint_limit_kd, var_302);
            // axis_1,                                                                            <L 1326>
            var_304 = wp::load(var_289);
            var_305 = wp::load(var_291);
            var_306 = wp::load(var_293);
            var_307 = wp::load(var_295);
            var_308 = wp::load(var_297);
            var_309 = wp::load(var_299);
            var_310 = wp::load(var_301);
            var_311 = wp::load(var_303);
            var_312 = eval_joint_force(var_286, var_287, var_304, var_305, var_306, var_307, var_308, var_309, var_310, var_311, var_255);
            var_313 = wp::add(var_285, var_312);
            // t_total += eval_joint_force(                                                       <L 1328>
            // angles[2],                                                                         <L 1329>
            var_314 = wp::extract(var_238, var_116);
            // wp.dot(axis_2, w_err),                                                             <L 1330>
            var_315 = wp::dot(var_256, var_107);
            // joint_target[axis_start + 2],                                                      <L 1331>
            var_316 = wp::add(var_74, var_116);
            var_317 = wp::address(var_joint_target, var_316);
            // joint_target_ke[axis_start + 2],                                                   <L 1332>
            var_318 = wp::add(var_74, var_116);
            var_319 = wp::address(var_joint_target_ke, var_318);
            // joint_target_kd[axis_start + 2],                                                   <L 1333>
            var_320 = wp::add(var_74, var_116);
            var_321 = wp::address(var_joint_target_kd, var_320);
            // joint_act[axis_start + 2],                                                         <L 1334>
            var_322 = wp::add(var_74, var_116);
            var_323 = wp::address(var_joint_act, var_322);
            // joint_limit_lower[axis_start + 2],                                                 <L 1335>
            var_324 = wp::add(var_74, var_116);
            var_325 = wp::address(var_joint_limit_lower, var_324);
            // joint_limit_upper[axis_start + 2],                                                 <L 1336>
            var_326 = wp::add(var_74, var_116);
            var_327 = wp::address(var_joint_limit_upper, var_326);
            // joint_limit_ke[axis_start + 2],                                                    <L 1337>
            var_328 = wp::add(var_74, var_116);
            var_329 = wp::address(var_joint_limit_ke, var_328);
            // joint_limit_kd[axis_start + 2],                                                    <L 1338>
            var_330 = wp::add(var_74, var_116);
            var_331 = wp::address(var_joint_limit_kd, var_330);
            // axis_2,                                                                            <L 1339>
            var_332 = wp::load(var_317);
            var_333 = wp::load(var_319);
            var_334 = wp::load(var_321);
            var_335 = wp::load(var_323);
            var_336 = wp::load(var_325);
            var_337 = wp::load(var_327);
            var_338 = wp::load(var_329);
            var_339 = wp::load(var_331);
            var_340 = eval_joint_force(var_314, var_315, var_332, var_333, var_334, var_335, var_336, var_337, var_338, var_339, var_256);
            var_341 = wp::add(var_313, var_340);
            // f_total += x_err * joint_attach_ke + v_err * joint_attach_kd                       <L 1342>
            var_342 = wp::mul(var_103, var_joint_attach_ke);
            var_343 = wp::mul(var_106, var_joint_attach_kd);
            var_344 = wp::add(var_342, var_343);
            var_345 = wp::add(var_232, var_344);
        }
        var_346 = wp::select(var_235, var_231, var_341);
        var_347 = wp::select(var_235, var_232, var_345);
        // if type == fs5Model.JOINT_UNIVERSAL:                                                   <L 1344>
        var_349 = (var_3 == var_348);
        if (var_349) {
            // q_pc = wp.quat_inverse(q_p) * q_c                                                  <L 1345>
            var_350 = wp::quat_inverse(var_101);
            var_351 = wp::mul(var_350, var_102);
            // angles = quat_decompose(q_pc)                                                      <L 1348>
            var_352 = quat_decompose(var_351);
            // axis_0 = wp.vec3(1.0, 0.0, 0.0)                                                    <L 1351>
            var_353 = wp::vec_t<3,wp::float32>(var_239, var_240, var_240);
            // q_0 = wp.quat_from_axis_angle(axis_0, angles[0])                                   <L 1352>
            var_354 = wp::extract(var_352, var_5);
            var_355 = wp::quat_from_axis_angle(var_353, var_354);
            // axis_1 = wp.quat_rotate(q_0, wp.vec3(0.0, 1.0, 0.0))                               <L 1354>
            var_356 = wp::vec_t<3,wp::float32>(var_240, var_239, var_240);
            var_357 = wp::quat_rotate(var_355, var_356);
            // q_1 = wp.quat_from_axis_angle(axis_1, angles[1])                                   <L 1355>
            var_358 = wp::extract(var_352, var_114);
            var_359 = wp::quat_from_axis_angle(var_357, var_358);
            // axis_2 = wp.quat_rotate(q_1 * q_0, wp.vec3(0.0, 0.0, 1.0))                         <L 1357>
            var_360 = wp::mul(var_359, var_355);
            var_361 = wp::vec_t<3,wp::float32>(var_240, var_240, var_239);
            var_362 = wp::quat_rotate(var_360, var_361);
            // q_2 = wp.quat_from_axis_angle(axis_2, angles[2])                                   <L 1358>
            var_363 = wp::extract(var_352, var_116);
            var_364 = wp::quat_from_axis_angle(var_362, var_363);
            // q_w = q_p                                                                          <L 1360>
            var_365 = wp::copy(var_101);
            // axis_0 = wp.transform_vector(X_wp, axis_0)                                         <L 1362>
            var_366 = wp::transform_vector(var_45, var_353);
            // axis_1 = wp.transform_vector(X_wp, axis_1)                                         <L 1363>
            var_367 = wp::transform_vector(var_45, var_357);
            // axis_2 = wp.transform_vector(X_wp, axis_2)                                         <L 1364>
            var_368 = wp::transform_vector(var_45, var_362);
            // t_total = wp.vec3()                                                                <L 1367>
            var_369 = wp::vec_t<3,wp::float32>();
            // t_total += eval_joint_force(                                                       <L 1378>
            // angles[0],                                                                         <L 1379>
            var_370 = wp::extract(var_352, var_5);
            // wp.dot(axis_0, w_err),                                                             <L 1380>
            var_371 = wp::dot(var_366, var_107);
            // joint_target[axis_start + 0],                                                      <L 1381>
            var_372 = wp::add(var_74, var_5);
            var_373 = wp::address(var_joint_target, var_372);
            // joint_target_ke[axis_start + 0],                                                   <L 1382>
            var_374 = wp::add(var_74, var_5);
            var_375 = wp::address(var_joint_target_ke, var_374);
            // joint_target_kd[axis_start + 0],                                                   <L 1383>
            var_376 = wp::add(var_74, var_5);
            var_377 = wp::address(var_joint_target_kd, var_376);
            // joint_act[axis_start + 0],                                                         <L 1384>
            var_378 = wp::add(var_74, var_5);
            var_379 = wp::address(var_joint_act, var_378);
            // joint_limit_lower[axis_start + 0],                                                 <L 1385>
            var_380 = wp::add(var_74, var_5);
            var_381 = wp::address(var_joint_limit_lower, var_380);
            // joint_limit_upper[axis_start + 0],                                                 <L 1386>
            var_382 = wp::add(var_74, var_5);
            var_383 = wp::address(var_joint_limit_upper, var_382);
            // joint_limit_ke[axis_start + 0],                                                    <L 1387>
            var_384 = wp::add(var_74, var_5);
            var_385 = wp::address(var_joint_limit_ke, var_384);
            // joint_limit_kd[axis_start + 0],                                                    <L 1388>
            var_386 = wp::add(var_74, var_5);
            var_387 = wp::address(var_joint_limit_kd, var_386);
            // axis_0,                                                                            <L 1389>
            var_388 = wp::load(var_373);
            var_389 = wp::load(var_375);
            var_390 = wp::load(var_377);
            var_391 = wp::load(var_379);
            var_392 = wp::load(var_381);
            var_393 = wp::load(var_383);
            var_394 = wp::load(var_385);
            var_395 = wp::load(var_387);
            var_396 = eval_joint_force(var_370, var_371, var_388, var_389, var_390, var_391, var_392, var_393, var_394, var_395, var_366);
            var_397 = wp::add(var_369, var_396);
            // t_total += eval_joint_force(                                                       <L 1391>
            // angles[1],                                                                         <L 1392>
            var_398 = wp::extract(var_352, var_114);
            // wp.dot(axis_1, w_err),                                                             <L 1393>
            var_399 = wp::dot(var_367, var_107);
            // joint_target[axis_start + 1],                                                      <L 1394>
            var_400 = wp::add(var_74, var_114);
            var_401 = wp::address(var_joint_target, var_400);
            // joint_target_ke[axis_start + 1],                                                   <L 1395>
            var_402 = wp::add(var_74, var_114);
            var_403 = wp::address(var_joint_target_ke, var_402);
            // joint_target_kd[axis_start + 1],                                                   <L 1396>
            var_404 = wp::add(var_74, var_114);
            var_405 = wp::address(var_joint_target_kd, var_404);
            // joint_act[axis_start + 1],                                                         <L 1397>
            var_406 = wp::add(var_74, var_114);
            var_407 = wp::address(var_joint_act, var_406);
            // joint_limit_lower[axis_start + 1],                                                 <L 1398>
            var_408 = wp::add(var_74, var_114);
            var_409 = wp::address(var_joint_limit_lower, var_408);
            // joint_limit_upper[axis_start + 1],                                                 <L 1399>
            var_410 = wp::add(var_74, var_114);
            var_411 = wp::address(var_joint_limit_upper, var_410);
            // joint_limit_ke[axis_start + 1],                                                    <L 1400>
            var_412 = wp::add(var_74, var_114);
            var_413 = wp::address(var_joint_limit_ke, var_412);
            // joint_limit_kd[axis_start + 1],                                                    <L 1401>
            var_414 = wp::add(var_74, var_114);
            var_415 = wp::address(var_joint_limit_kd, var_414);
            // axis_1,                                                                            <L 1402>
            var_416 = wp::load(var_401);
            var_417 = wp::load(var_403);
            var_418 = wp::load(var_405);
            var_419 = wp::load(var_407);
            var_420 = wp::load(var_409);
            var_421 = wp::load(var_411);
            var_422 = wp::load(var_413);
            var_423 = wp::load(var_415);
            var_424 = eval_joint_force(var_398, var_399, var_416, var_417, var_418, var_419, var_420, var_421, var_422, var_423, var_367);
            var_425 = wp::add(var_397, var_424);
            // t_total += eval_joint_force(                                                       <L 1406>
            // angles[2],                                                                         <L 1407>
            var_426 = wp::extract(var_352, var_116);
            // wp.dot(axis_2, w_err),                                                             <L 1408>
            var_427 = wp::dot(var_368, var_107);
            // 0.0,                                                                               <L 1409>
            // joint_attach_ke,                                                                   <L 1410>
            // joint_attach_kd * angular_damping_scale,                                           <L 1411>
            var_428 = wp::mul(var_joint_attach_kd, var_110);
            // 0.0,                                                                               <L 1412>
            // 0.0,                                                                               <L 1413>
            // 0.0,                                                                               <L 1414>
            // 0.0,                                                                               <L 1415>
            // 0.0,                                                                               <L 1416>
            // axis_2,                                                                            <L 1417>
            var_429 = eval_joint_force(var_426, var_427, var_240, var_joint_attach_ke, var_428, var_240, var_240, var_240, var_240, var_240, var_368);
            var_430 = wp::add(var_425, var_429);
            // f_total += x_err * joint_attach_ke + v_err * joint_attach_kd                       <L 1420>
            var_431 = wp::mul(var_103, var_joint_attach_ke);
            var_432 = wp::mul(var_106, var_joint_attach_kd);
            var_433 = wp::add(var_431, var_432);
            var_434 = wp::add(var_347, var_433);
        }
        var_435 = wp::select(var_349, var_346, var_430);
        var_436 = wp::select(var_349, var_347, var_434);
        var_437 = wp::select(var_349, var_237, var_351);
        var_438 = wp::select(var_349, var_238, var_352);
        var_439 = wp::select(var_349, var_254, var_366);
        var_440 = wp::select(var_349, var_243, var_355);
        var_441 = wp::select(var_349, var_255, var_367);
        var_442 = wp::select(var_349, var_247, var_359);
        var_443 = wp::select(var_349, var_256, var_368);
        var_444 = wp::select(var_349, var_252, var_364);
        var_445 = wp::select(var_349, var_253, var_365);
        // if c_parent >= 0:                                                                      <L 1423>
        var_446 = (var_16 >= var_5);
        if (var_446) {
            // wp.atomic_add(body_f, c_parent, wp.spatial_vector(t_total + wp.cross(r_p, f_total), f_total))       <L 1424>
            var_447 = wp::cross(var_46, var_436);
            var_448 = wp::add(var_435, var_447);
            var_449 = wp::vec_t<6,wp::float32>(var_448, var_436);
            // var_450 = wp::atomic_add(var_body_f, var_16, var_449);
        }
        // wp.atomic_sub(body_f, c_child, wp.spatial_vector(t_total + wp.cross(r_c, f_total), f_total))       <L 1426>
        var_451 = wp::cross(var_58, var_436);
        var_452 = wp::add(var_435, var_451);
        var_453 = wp::vec_t<6,wp::float32>(var_452, var_436);
        // var_454 = wp::atomic_sub(var_body_f, var_13, var_453);
        //---------
        // reverse
        wp::adj_atomic_sub(var_body_f, var_13, var_453, adj_body_f, adj_13, adj_453, adj_454);
        wp::adj_vec_t(var_452, var_436, adj_452, adj_436, adj_453);
        wp::adj_add(var_435, var_451, adj_435, adj_451, adj_452);
        wp::adj_cross(var_58, var_436, adj_58, adj_436, adj_451);
        // adj: wp.atomic_sub(body_f, c_child, wp.spatial_vector(t_total + wp.cross(r_c, f_total), f_total))  <L 1426>
        if (var_446) {
            wp::adj_atomic_add(var_body_f, var_16, var_449, adj_body_f, adj_16, adj_449, adj_450);
            wp::adj_vec_t(var_448, var_436, adj_448, adj_436, adj_449);
            wp::adj_add(var_435, var_447, adj_435, adj_447, adj_448);
            wp::adj_cross(var_46, var_436, adj_46, adj_436, adj_447);
            // adj: wp.atomic_add(body_f, c_parent, wp.spatial_vector(t_total + wp.cross(r_p, f_total), f_total))  <L 1424>
        }
        // adj: if c_parent >= 0:                                                                 <L 1423>
        wp::adj_select(var_349, var_253, var_365, adj_349, adj_253, adj_365, adj_445);
        wp::adj_select(var_349, var_252, var_364, adj_349, adj_252, adj_364, adj_444);
        wp::adj_select(var_349, var_256, var_368, adj_349, adj_256, adj_368, adj_443);
        wp::adj_select(var_349, var_247, var_359, adj_349, adj_247, adj_359, adj_442);
        wp::adj_select(var_349, var_255, var_367, adj_349, adj_255, adj_367, adj_441);
        wp::adj_select(var_349, var_243, var_355, adj_349, adj_243, adj_355, adj_440);
        wp::adj_select(var_349, var_254, var_366, adj_349, adj_254, adj_366, adj_439);
        wp::adj_select(var_349, var_238, var_352, adj_349, adj_238, adj_352, adj_438);
        wp::adj_select(var_349, var_237, var_351, adj_349, adj_237, adj_351, adj_437);
        wp::adj_select(var_349, var_347, var_434, adj_349, adj_347, adj_434, adj_436);
        wp::adj_select(var_349, var_346, var_430, adj_349, adj_346, adj_430, adj_435);
        if (var_349) {
            wp::adj_add(var_347, var_433, adj_347, adj_433, adj_434);
            wp::adj_add(var_431, var_432, adj_431, adj_432, adj_433);
            wp::adj_mul(var_106, var_joint_attach_kd, adj_106, adj_joint_attach_kd, adj_432);
            wp::adj_mul(var_103, var_joint_attach_ke, adj_103, adj_joint_attach_ke, adj_431);
            // adj: f_total += x_err * joint_attach_ke + v_err * joint_attach_kd                  <L 1420>
            wp::adj_add(var_425, var_429, adj_425, adj_429, adj_430);
            adj_eval_joint_force(var_426, var_427, var_240, var_joint_attach_ke, var_428, var_240, var_240, var_240, var_240, var_240, var_368, adj_426, adj_427, adj_240, adj_joint_attach_ke, adj_428, adj_240, adj_240, adj_240, adj_240, adj_240, adj_368, adj_429);
            // adj: axis_2,                                                                       <L 1417>
            // adj: 0.0,                                                                          <L 1416>
            // adj: 0.0,                                                                          <L 1415>
            // adj: 0.0,                                                                          <L 1414>
            // adj: 0.0,                                                                          <L 1413>
            // adj: 0.0,                                                                          <L 1412>
            wp::adj_mul(var_joint_attach_kd, var_110, adj_joint_attach_kd, adj_110, adj_428);
            // adj: joint_attach_kd * angular_damping_scale,                                      <L 1411>
            // adj: joint_attach_ke,                                                              <L 1410>
            // adj: 0.0,                                                                          <L 1409>
            wp::adj_dot(var_368, var_107, adj_368, adj_107, adj_427);
            // adj: wp.dot(axis_2, w_err),                                                        <L 1408>
            wp::adj_extract(var_352, var_116, adj_352, adj_116, adj_426);
            // adj: angles[2],                                                                    <L 1407>
            // adj: t_total += eval_joint_force(                                                  <L 1406>
            wp::adj_add(var_397, var_424, adj_397, adj_424, adj_425);
            adj_eval_joint_force(var_398, var_399, var_416, var_417, var_418, var_419, var_420, var_421, var_422, var_423, var_367, adj_398, adj_399, adj_401, adj_403, adj_405, adj_407, adj_409, adj_411, adj_413, adj_415, adj_367, adj_424);
            wp::adj_load(var_415, adj_415, adj_423);
            wp::adj_load(var_413, adj_413, adj_422);
            wp::adj_load(var_411, adj_411, adj_421);
            wp::adj_load(var_409, adj_409, adj_420);
            wp::adj_load(var_407, adj_407, adj_419);
            wp::adj_load(var_405, adj_405, adj_418);
            wp::adj_load(var_403, adj_403, adj_417);
            wp::adj_load(var_401, adj_401, adj_416);
            // adj: axis_1,                                                                       <L 1402>
            wp::adj_address(var_joint_limit_kd, var_414, adj_joint_limit_kd, adj_414, adj_415);
            wp::adj_add(var_74, var_114, adj_74, adj_114, adj_414);
            // adj: joint_limit_kd[axis_start + 1],                                               <L 1401>
            wp::adj_address(var_joint_limit_ke, var_412, adj_joint_limit_ke, adj_412, adj_413);
            wp::adj_add(var_74, var_114, adj_74, adj_114, adj_412);
            // adj: joint_limit_ke[axis_start + 1],                                               <L 1400>
            wp::adj_address(var_joint_limit_upper, var_410, adj_joint_limit_upper, adj_410, adj_411);
            wp::adj_add(var_74, var_114, adj_74, adj_114, adj_410);
            // adj: joint_limit_upper[axis_start + 1],                                            <L 1399>
            wp::adj_address(var_joint_limit_lower, var_408, adj_joint_limit_lower, adj_408, adj_409);
            wp::adj_add(var_74, var_114, adj_74, adj_114, adj_408);
            // adj: joint_limit_lower[axis_start + 1],                                            <L 1398>
            wp::adj_address(var_joint_act, var_406, adj_joint_act, adj_406, adj_407);
            wp::adj_add(var_74, var_114, adj_74, adj_114, adj_406);
            // adj: joint_act[axis_start + 1],                                                    <L 1397>
            wp::adj_address(var_joint_target_kd, var_404, adj_joint_target_kd, adj_404, adj_405);
            wp::adj_add(var_74, var_114, adj_74, adj_114, adj_404);
            // adj: joint_target_kd[axis_start + 1],                                              <L 1396>
            wp::adj_address(var_joint_target_ke, var_402, adj_joint_target_ke, adj_402, adj_403);
            wp::adj_add(var_74, var_114, adj_74, adj_114, adj_402);
            // adj: joint_target_ke[axis_start + 1],                                              <L 1395>
            wp::adj_address(var_joint_target, var_400, adj_joint_target, adj_400, adj_401);
            wp::adj_add(var_74, var_114, adj_74, adj_114, adj_400);
            // adj: joint_target[axis_start + 1],                                                 <L 1394>
            wp::adj_dot(var_367, var_107, adj_367, adj_107, adj_399);
            // adj: wp.dot(axis_1, w_err),                                                        <L 1393>
            wp::adj_extract(var_352, var_114, adj_352, adj_114, adj_398);
            // adj: angles[1],                                                                    <L 1392>
            // adj: t_total += eval_joint_force(                                                  <L 1391>
            wp::adj_add(var_369, var_396, adj_369, adj_396, adj_397);
            adj_eval_joint_force(var_370, var_371, var_388, var_389, var_390, var_391, var_392, var_393, var_394, var_395, var_366, adj_370, adj_371, adj_373, adj_375, adj_377, adj_379, adj_381, adj_383, adj_385, adj_387, adj_366, adj_396);
            wp::adj_load(var_387, adj_387, adj_395);
            wp::adj_load(var_385, adj_385, adj_394);
            wp::adj_load(var_383, adj_383, adj_393);
            wp::adj_load(var_381, adj_381, adj_392);
            wp::adj_load(var_379, adj_379, adj_391);
            wp::adj_load(var_377, adj_377, adj_390);
            wp::adj_load(var_375, adj_375, adj_389);
            wp::adj_load(var_373, adj_373, adj_388);
            // adj: axis_0,                                                                       <L 1389>
            wp::adj_address(var_joint_limit_kd, var_386, adj_joint_limit_kd, adj_386, adj_387);
            wp::adj_add(var_74, var_5, adj_74, adj_5, adj_386);
            // adj: joint_limit_kd[axis_start + 0],                                               <L 1388>
            wp::adj_address(var_joint_limit_ke, var_384, adj_joint_limit_ke, adj_384, adj_385);
            wp::adj_add(var_74, var_5, adj_74, adj_5, adj_384);
            // adj: joint_limit_ke[axis_start + 0],                                               <L 1387>
            wp::adj_address(var_joint_limit_upper, var_382, adj_joint_limit_upper, adj_382, adj_383);
            wp::adj_add(var_74, var_5, adj_74, adj_5, adj_382);
            // adj: joint_limit_upper[axis_start + 0],                                            <L 1386>
            wp::adj_address(var_joint_limit_lower, var_380, adj_joint_limit_lower, adj_380, adj_381);
            wp::adj_add(var_74, var_5, adj_74, adj_5, adj_380);
            // adj: joint_limit_lower[axis_start + 0],                                            <L 1385>
            wp::adj_address(var_joint_act, var_378, adj_joint_act, adj_378, adj_379);
            wp::adj_add(var_74, var_5, adj_74, adj_5, adj_378);
            // adj: joint_act[axis_start + 0],                                                    <L 1384>
            wp::adj_address(var_joint_target_kd, var_376, adj_joint_target_kd, adj_376, adj_377);
            wp::adj_add(var_74, var_5, adj_74, adj_5, adj_376);
            // adj: joint_target_kd[axis_start + 0],                                              <L 1383>
            wp::adj_address(var_joint_target_ke, var_374, adj_joint_target_ke, adj_374, adj_375);
            wp::adj_add(var_74, var_5, adj_74, adj_5, adj_374);
            // adj: joint_target_ke[axis_start + 0],                                              <L 1382>
            wp::adj_address(var_joint_target, var_372, adj_joint_target, adj_372, adj_373);
            wp::adj_add(var_74, var_5, adj_74, adj_5, adj_372);
            // adj: joint_target[axis_start + 0],                                                 <L 1381>
            wp::adj_dot(var_366, var_107, adj_366, adj_107, adj_371);
            // adj: wp.dot(axis_0, w_err),                                                        <L 1380>
            wp::adj_extract(var_352, var_5, adj_352, adj_5, adj_370);
            // adj: angles[0],                                                                    <L 1379>
            // adj: t_total += eval_joint_force(                                                  <L 1378>
            // adj: t_total = wp.vec3()                                                           <L 1367>
            wp::adj_transform_vector(var_45, var_362, adj_45, adj_362, adj_368);
            // adj: axis_2 = wp.transform_vector(X_wp, axis_2)                                    <L 1364>
            wp::adj_transform_vector(var_45, var_357, adj_45, adj_357, adj_367);
            // adj: axis_1 = wp.transform_vector(X_wp, axis_1)                                    <L 1363>
            wp::adj_transform_vector(var_45, var_353, adj_45, adj_353, adj_366);
            // adj: axis_0 = wp.transform_vector(X_wp, axis_0)                                    <L 1362>
            wp::adj_copy(var_101, adj_101, adj_365);
            // adj: q_w = q_p                                                                     <L 1360>
            wp::adj_quat_from_axis_angle(var_362, var_363, adj_362, adj_363, adj_364);
            wp::adj_extract(var_352, var_116, adj_352, adj_116, adj_363);
            // adj: q_2 = wp.quat_from_axis_angle(axis_2, angles[2])                              <L 1358>
            wp::adj_quat_rotate(var_360, var_361, adj_360, adj_361, adj_362);
            wp::adj_vec_t(var_240, var_240, var_239, adj_240, adj_240, adj_239, adj_361);
            wp::adj_mul(var_359, var_355, adj_359, adj_355, adj_360);
            // adj: axis_2 = wp.quat_rotate(q_1 * q_0, wp.vec3(0.0, 0.0, 1.0))                    <L 1357>
            wp::adj_quat_from_axis_angle(var_357, var_358, adj_357, adj_358, adj_359);
            wp::adj_extract(var_352, var_114, adj_352, adj_114, adj_358);
            // adj: q_1 = wp.quat_from_axis_angle(axis_1, angles[1])                              <L 1355>
            wp::adj_quat_rotate(var_355, var_356, adj_355, adj_356, adj_357);
            wp::adj_vec_t(var_240, var_239, var_240, adj_240, adj_239, adj_240, adj_356);
            // adj: axis_1 = wp.quat_rotate(q_0, wp.vec3(0.0, 1.0, 0.0))                          <L 1354>
            wp::adj_quat_from_axis_angle(var_353, var_354, adj_353, adj_354, adj_355);
            wp::adj_extract(var_352, var_5, adj_352, adj_5, adj_354);
            // adj: q_0 = wp.quat_from_axis_angle(axis_0, angles[0])                              <L 1352>
            wp::adj_vec_t(var_239, var_240, var_240, adj_239, adj_240, adj_240, adj_353);
            // adj: axis_0 = wp.vec3(1.0, 0.0, 0.0)                                               <L 1351>
            adj_quat_decompose(var_351, adj_351, adj_352);
            // adj: angles = quat_decompose(q_pc)                                                 <L 1348>
            wp::adj_mul(var_350, var_102, adj_350, adj_102, adj_351);
            wp::adj_quat_inverse(var_101, adj_101, adj_350);
            // adj: q_pc = wp.quat_inverse(q_p) * q_c                                             <L 1345>
        }
        // adj: if type == fs5Model.JOINT_UNIVERSAL:                                              <L 1344>
        wp::adj_select(var_235, var_232, var_345, adj_235, adj_232, adj_345, adj_347);
        wp::adj_select(var_235, var_231, var_341, adj_235, adj_231, adj_341, adj_346);
        if (var_235) {
            wp::adj_add(var_232, var_344, adj_232, adj_344, adj_345);
            wp::adj_add(var_342, var_343, adj_342, adj_343, adj_344);
            wp::adj_mul(var_106, var_joint_attach_kd, adj_106, adj_joint_attach_kd, adj_343);
            wp::adj_mul(var_103, var_joint_attach_ke, adj_103, adj_joint_attach_ke, adj_342);
            // adj: f_total += x_err * joint_attach_ke + v_err * joint_attach_kd                  <L 1342>
            wp::adj_add(var_313, var_340, adj_313, adj_340, adj_341);
            adj_eval_joint_force(var_314, var_315, var_332, var_333, var_334, var_335, var_336, var_337, var_338, var_339, var_256, adj_314, adj_315, adj_317, adj_319, adj_321, adj_323, adj_325, adj_327, adj_329, adj_331, adj_256, adj_340);
            wp::adj_load(var_331, adj_331, adj_339);
            wp::adj_load(var_329, adj_329, adj_338);
            wp::adj_load(var_327, adj_327, adj_337);
            wp::adj_load(var_325, adj_325, adj_336);
            wp::adj_load(var_323, adj_323, adj_335);
            wp::adj_load(var_321, adj_321, adj_334);
            wp::adj_load(var_319, adj_319, adj_333);
            wp::adj_load(var_317, adj_317, adj_332);
            // adj: axis_2,                                                                       <L 1339>
            wp::adj_address(var_joint_limit_kd, var_330, adj_joint_limit_kd, adj_330, adj_331);
            wp::adj_add(var_74, var_116, adj_74, adj_116, adj_330);
            // adj: joint_limit_kd[axis_start + 2],                                               <L 1338>
            wp::adj_address(var_joint_limit_ke, var_328, adj_joint_limit_ke, adj_328, adj_329);
            wp::adj_add(var_74, var_116, adj_74, adj_116, adj_328);
            // adj: joint_limit_ke[axis_start + 2],                                               <L 1337>
            wp::adj_address(var_joint_limit_upper, var_326, adj_joint_limit_upper, adj_326, adj_327);
            wp::adj_add(var_74, var_116, adj_74, adj_116, adj_326);
            // adj: joint_limit_upper[axis_start + 2],                                            <L 1336>
            wp::adj_address(var_joint_limit_lower, var_324, adj_joint_limit_lower, adj_324, adj_325);
            wp::adj_add(var_74, var_116, adj_74, adj_116, adj_324);
            // adj: joint_limit_lower[axis_start + 2],                                            <L 1335>
            wp::adj_address(var_joint_act, var_322, adj_joint_act, adj_322, adj_323);
            wp::adj_add(var_74, var_116, adj_74, adj_116, adj_322);
            // adj: joint_act[axis_start + 2],                                                    <L 1334>
            wp::adj_address(var_joint_target_kd, var_320, adj_joint_target_kd, adj_320, adj_321);
            wp::adj_add(var_74, var_116, adj_74, adj_116, adj_320);
            // adj: joint_target_kd[axis_start + 2],                                              <L 1333>
            wp::adj_address(var_joint_target_ke, var_318, adj_joint_target_ke, adj_318, adj_319);
            wp::adj_add(var_74, var_116, adj_74, adj_116, adj_318);
            // adj: joint_target_ke[axis_start + 2],                                              <L 1332>
            wp::adj_address(var_joint_target, var_316, adj_joint_target, adj_316, adj_317);
            wp::adj_add(var_74, var_116, adj_74, adj_116, adj_316);
            // adj: joint_target[axis_start + 2],                                                 <L 1331>
            wp::adj_dot(var_256, var_107, adj_256, adj_107, adj_315);
            // adj: wp.dot(axis_2, w_err),                                                        <L 1330>
            wp::adj_extract(var_238, var_116, adj_238, adj_116, adj_314);
            // adj: angles[2],                                                                    <L 1329>
            // adj: t_total += eval_joint_force(                                                  <L 1328>
            wp::adj_add(var_285, var_312, adj_285, adj_312, adj_313);
            adj_eval_joint_force(var_286, var_287, var_304, var_305, var_306, var_307, var_308, var_309, var_310, var_311, var_255, adj_286, adj_287, adj_289, adj_291, adj_293, adj_295, adj_297, adj_299, adj_301, adj_303, adj_255, adj_312);
            wp::adj_load(var_303, adj_303, adj_311);
            wp::adj_load(var_301, adj_301, adj_310);
            wp::adj_load(var_299, adj_299, adj_309);
            wp::adj_load(var_297, adj_297, adj_308);
            wp::adj_load(var_295, adj_295, adj_307);
            wp::adj_load(var_293, adj_293, adj_306);
            wp::adj_load(var_291, adj_291, adj_305);
            wp::adj_load(var_289, adj_289, adj_304);
            // adj: axis_1,                                                                       <L 1326>
            wp::adj_address(var_joint_limit_kd, var_302, adj_joint_limit_kd, adj_302, adj_303);
            wp::adj_add(var_74, var_114, adj_74, adj_114, adj_302);
            // adj: joint_limit_kd[axis_start + 1],                                               <L 1325>
            wp::adj_address(var_joint_limit_ke, var_300, adj_joint_limit_ke, adj_300, adj_301);
            wp::adj_add(var_74, var_114, adj_74, adj_114, adj_300);
            // adj: joint_limit_ke[axis_start + 1],                                               <L 1324>
            wp::adj_address(var_joint_limit_upper, var_298, adj_joint_limit_upper, adj_298, adj_299);
            wp::adj_add(var_74, var_114, adj_74, adj_114, adj_298);
            // adj: joint_limit_upper[axis_start + 1],                                            <L 1323>
            wp::adj_address(var_joint_limit_lower, var_296, adj_joint_limit_lower, adj_296, adj_297);
            wp::adj_add(var_74, var_114, adj_74, adj_114, adj_296);
            // adj: joint_limit_lower[axis_start + 1],                                            <L 1322>
            wp::adj_address(var_joint_act, var_294, adj_joint_act, adj_294, adj_295);
            wp::adj_add(var_74, var_114, adj_74, adj_114, adj_294);
            // adj: joint_act[axis_start + 1],                                                    <L 1321>
            wp::adj_address(var_joint_target_kd, var_292, adj_joint_target_kd, adj_292, adj_293);
            wp::adj_add(var_74, var_114, adj_74, adj_114, adj_292);
            // adj: joint_target_kd[axis_start + 1],                                              <L 1320>
            wp::adj_address(var_joint_target_ke, var_290, adj_joint_target_ke, adj_290, adj_291);
            wp::adj_add(var_74, var_114, adj_74, adj_114, adj_290);
            // adj: joint_target_ke[axis_start + 1],                                              <L 1319>
            wp::adj_address(var_joint_target, var_288, adj_joint_target, adj_288, adj_289);
            wp::adj_add(var_74, var_114, adj_74, adj_114, adj_288);
            // adj: joint_target[axis_start + 1],                                                 <L 1318>
            wp::adj_dot(var_255, var_107, adj_255, adj_107, adj_287);
            // adj: wp.dot(axis_1, w_err),                                                        <L 1317>
            wp::adj_extract(var_238, var_114, adj_238, adj_114, adj_286);
            // adj: angles[1],                                                                    <L 1316>
            // adj: t_total += eval_joint_force(                                                  <L 1315>
            wp::adj_add(var_257, var_284, adj_257, adj_284, adj_285);
            adj_eval_joint_force(var_258, var_259, var_276, var_277, var_278, var_279, var_280, var_281, var_282, var_283, var_254, adj_258, adj_259, adj_261, adj_263, adj_265, adj_267, adj_269, adj_271, adj_273, adj_275, adj_254, adj_284);
            wp::adj_load(var_275, adj_275, adj_283);
            wp::adj_load(var_273, adj_273, adj_282);
            wp::adj_load(var_271, adj_271, adj_281);
            wp::adj_load(var_269, adj_269, adj_280);
            wp::adj_load(var_267, adj_267, adj_279);
            wp::adj_load(var_265, adj_265, adj_278);
            wp::adj_load(var_263, adj_263, adj_277);
            wp::adj_load(var_261, adj_261, adj_276);
            // adj: axis_0,                                                                       <L 1313>
            wp::adj_address(var_joint_limit_kd, var_274, adj_joint_limit_kd, adj_274, adj_275);
            wp::adj_add(var_74, var_5, adj_74, adj_5, adj_274);
            // adj: joint_limit_kd[axis_start + 0],                                               <L 1312>
            wp::adj_address(var_joint_limit_ke, var_272, adj_joint_limit_ke, adj_272, adj_273);
            wp::adj_add(var_74, var_5, adj_74, adj_5, adj_272);
            // adj: joint_limit_ke[axis_start + 0],                                               <L 1311>
            wp::adj_address(var_joint_limit_upper, var_270, adj_joint_limit_upper, adj_270, adj_271);
            wp::adj_add(var_74, var_5, adj_74, adj_5, adj_270);
            // adj: joint_limit_upper[axis_start + 0],                                            <L 1310>
            wp::adj_address(var_joint_limit_lower, var_268, adj_joint_limit_lower, adj_268, adj_269);
            wp::adj_add(var_74, var_5, adj_74, adj_5, adj_268);
            // adj: joint_limit_lower[axis_start + 0],                                            <L 1309>
            wp::adj_address(var_joint_act, var_266, adj_joint_act, adj_266, adj_267);
            wp::adj_add(var_74, var_5, adj_74, adj_5, adj_266);
            // adj: joint_act[axis_start + 0],                                                    <L 1308>
            wp::adj_address(var_joint_target_kd, var_264, adj_joint_target_kd, adj_264, adj_265);
            wp::adj_add(var_74, var_5, adj_74, adj_5, adj_264);
            // adj: joint_target_kd[axis_start + 0],                                              <L 1307>
            wp::adj_address(var_joint_target_ke, var_262, adj_joint_target_ke, adj_262, adj_263);
            wp::adj_add(var_74, var_5, adj_74, adj_5, adj_262);
            // adj: joint_target_ke[axis_start + 0],                                              <L 1306>
            wp::adj_address(var_joint_target, var_260, adj_joint_target, adj_260, adj_261);
            wp::adj_add(var_74, var_5, adj_74, adj_5, adj_260);
            // adj: joint_target[axis_start + 0],                                                 <L 1305>
            wp::adj_dot(var_254, var_107, adj_254, adj_107, adj_259);
            // adj: wp.dot(axis_0, w_err),                                                        <L 1304>
            wp::adj_extract(var_238, var_5, adj_238, adj_5, adj_258);
            // adj: angles[0],                                                                    <L 1303>
            // adj: t_total += eval_joint_force(                                                  <L 1302>
            // adj: t_total = wp.vec3()                                                           <L 1296>
            wp::adj_transform_vector(var_45, var_250, adj_45, adj_250, adj_256);
            // adj: axis_2 = wp.transform_vector(X_wp, axis_2)                                    <L 1293>
            wp::adj_transform_vector(var_45, var_245, adj_45, adj_245, adj_255);
            // adj: axis_1 = wp.transform_vector(X_wp, axis_1)                                    <L 1292>
            wp::adj_transform_vector(var_45, var_241, adj_45, adj_241, adj_254);
            // adj: axis_0 = wp.transform_vector(X_wp, axis_0)                                    <L 1291>
            wp::adj_copy(var_101, adj_101, adj_253);
            // adj: q_w = q_p                                                                     <L 1289>
            wp::adj_quat_from_axis_angle(var_250, var_251, adj_250, adj_251, adj_252);
            wp::adj_extract(var_238, var_116, adj_238, adj_116, adj_251);
            // adj: q_2 = wp.quat_from_axis_angle(axis_2, angles[2])                              <L 1287>
            wp::adj_quat_rotate(var_248, var_249, adj_248, adj_249, adj_250);
            wp::adj_vec_t(var_240, var_240, var_239, adj_240, adj_240, adj_239, adj_249);
            wp::adj_mul(var_247, var_243, adj_247, adj_243, adj_248);
            // adj: axis_2 = wp.quat_rotate(q_1 * q_0, wp.vec3(0.0, 0.0, 1.0))                    <L 1286>
            wp::adj_quat_from_axis_angle(var_245, var_246, adj_245, adj_246, adj_247);
            wp::adj_extract(var_238, var_114, adj_238, adj_114, adj_246);
            // adj: q_1 = wp.quat_from_axis_angle(axis_1, angles[1])                              <L 1284>
            wp::adj_quat_rotate(var_243, var_244, adj_243, adj_244, adj_245);
            wp::adj_vec_t(var_240, var_239, var_240, adj_240, adj_239, adj_240, adj_244);
            // adj: axis_1 = wp.quat_rotate(q_0, wp.vec3(0.0, 1.0, 0.0))                          <L 1283>
            wp::adj_quat_from_axis_angle(var_241, var_242, adj_241, adj_242, adj_243);
            wp::adj_extract(var_238, var_5, adj_238, adj_5, adj_242);
            // adj: q_0 = wp.quat_from_axis_angle(axis_0, angles[0])                              <L 1281>
            wp::adj_vec_t(var_239, var_240, var_240, adj_239, adj_240, adj_240, adj_241);
            // adj: axis_0 = wp.vec3(1.0, 0.0, 0.0)                                               <L 1280>
            adj_quat_decompose(var_237, adj_237, adj_238);
            // adj: angles = quat_decompose(q_pc)                                                 <L 1277>
            wp::adj_mul(var_236, var_102, adj_236, adj_102, adj_237);
            wp::adj_quat_inverse(var_101, adj_101, adj_236);
            // adj: q_pc = wp.quat_inverse(q_p) * q_c                                             <L 1274>
        }
        // adj: if type == fs5Model.JOINT_COMPOUND:                                               <L 1273>
        wp::adj_select(var_212, var_172, var_221, adj_212, adj_172, adj_221, adj_233);
        wp::adj_select(var_212, var_206, var_230, adj_212, adj_206, adj_230, adj_232);
        wp::adj_select(var_212, var_205, var_226, adj_212, adj_205, adj_226, adj_231);
        if (var_212) {
            wp::adj_add(var_206, var_229, adj_206, adj_229, adj_230);
            wp::adj_add(var_227, var_228, adj_227, adj_228, adj_229);
            wp::adj_mul(var_106, var_joint_attach_kd, adj_106, adj_joint_attach_kd, adj_228);
            wp::adj_mul(var_103, var_joint_attach_ke, adj_103, adj_joint_attach_ke, adj_227);
            // adj: f_total += x_err * joint_attach_ke + v_err * joint_attach_kd                  <L 1271>
            wp::adj_add(var_205, var_225, adj_205, adj_225, adj_226);
            wp::adj_add(var_222, var_224, adj_222, adj_224, adj_225);
            wp::adj_mul(var_80, var_223, adj_80, adj_223, adj_224);
            wp::adj_transform_vector(var_45, var_221, adj_45, adj_221, adj_223);
            wp::adj_mul(var_83, var_107, adj_83, adj_107, adj_222);
            // adj: t_total += target_kd * w_err + target_ke * wp.transform_vector(X_wp, ang_err)  <L 1270>
            wp::adj_mul(var_220, var_124, adj_220, adj_124, adj_221);
            wp::adj_mul(var_217, var_219, adj_217, adj_219, adj_220);
            wp::adj_acos(var_218, adj_218, adj_219);
            wp::adj_extract(var_105, var_120, adj_105, adj_120, adj_218);
            wp::adj_normalize(var_216, var_217, adj_216, adj_217);
            wp::adj_vec_t(var_213, var_214, var_215, adj_213, adj_214, adj_215, adj_216);
            wp::adj_extract(var_105, var_116, adj_105, adj_116, adj_215);
            wp::adj_extract(var_105, var_114, adj_105, adj_114, adj_214);
            wp::adj_extract(var_105, var_5, adj_105, adj_5, adj_213);
            // adj: ang_err = wp.normalize(wp.vec3(r_err[0], r_err[1], r_err[2])) * wp.acos(r_err[3]) * 2.0  <L 1267>
        }
        // adj: if type == fs5Model.JOINT_BALL:                                                   <L 1266>
        wp::adj_select(var_174, var_145, var_191, adj_174, adj_145, adj_191, adj_210);
        wp::adj_select(var_174, var_144, var_190, adj_174, adj_144, adj_190, adj_209);
        wp::adj_select(var_174, var_143, var_178, adj_174, adj_143, adj_178, adj_208);
        wp::adj_select(var_174, var_142, var_177, adj_174, adj_142, adj_177, adj_207);
        wp::adj_select(var_174, var_171, var_197, adj_174, adj_171, adj_197, adj_206);
        wp::adj_select(var_174, var_170, var_204, adj_174, adj_170, adj_204, adj_205);
        if (var_174) {
            wp::adj_add(var_192, var_203, adj_192, adj_203, adj_204);
            wp::adj_add(var_198, var_202, adj_198, adj_202, adj_203);
            wp::adj_mul(var_201, var_110, adj_201, adj_110, adj_202);
            wp::adj_mul(var_200, var_joint_attach_kd, adj_200, adj_joint_attach_kd, adj_201);
            wp::adj_sub(var_107, var_199, adj_107, adj_199, adj_200);
            wp::adj_mul(var_191, var_178, adj_191, adj_178, adj_199);
            wp::adj_mul(var_193, var_joint_attach_ke, adj_193, adj_joint_attach_ke, adj_198);
            // adj: t_total += swing_err * joint_attach_ke + (w_err - qd * axis_p) * joint_attach_kd * angular_damping_scale  <L 1264>
            wp::adj_add(var_171, var_196, adj_171, adj_196, adj_197);
            wp::adj_add(var_194, var_195, adj_194, adj_195, adj_196);
            wp::adj_mul(var_106, var_joint_attach_kd, adj_106, adj_joint_attach_kd, adj_195);
            wp::adj_mul(var_103, var_joint_attach_ke, adj_103, adj_joint_attach_ke, adj_194);
            // adj: f_total += x_err * joint_attach_ke + v_err * joint_attach_kd                  <L 1263>
            wp::adj_cross(var_178, var_179, adj_178, adj_179, adj_193);
            // adj: swing_err = wp.cross(axis_p, axis_c)                                          <L 1261>
            adj_eval_joint_force(var_190, var_191, var_77, var_80, var_83, var_98, var_92, var_95, var_86, var_89, var_178, adj_190, adj_191, adj_77, adj_80, adj_83, adj_98, adj_92, adj_95, adj_86, adj_89, adj_178, adj_192);
            // adj: q, qd, target, target_ke, target_kd, act, limit_lower, limit_upper, limit_ke, limit_kd, axis_p  <L 1257>
            // adj: t_total = eval_joint_force(                                                   <L 1256>
            wp::adj_dot(var_107, var_178, adj_107, adj_178, adj_191);
            // adj: qd = wp.dot(w_err, axis_p)                                                    <L 1254>
            wp::adj_mul(var_183, var_189, adj_183, adj_189, adj_190);
            wp::adj_sign(var_188, adj_188, adj_189);
            wp::adj_dot(var_177, var_187, adj_177, adj_187, adj_188);
            wp::adj_vec_t(var_184, var_185, var_186, adj_184, adj_185, adj_186, adj_187);
            wp::adj_extract(var_180, var_116, adj_180, adj_116, adj_186);
            wp::adj_extract(var_180, var_114, adj_180, adj_114, adj_185);
            wp::adj_extract(var_180, var_5, adj_180, adj_5, adj_184);
            wp::adj_mul(var_182, var_124, adj_182, adj_124, adj_183);
            wp::adj_acos(var_181, adj_181, adj_182);
            wp::adj_extract(var_180, var_120, adj_180, adj_120, adj_181);
            // adj: q = wp.acos(twist[3]) * 2.0 * wp.sign(wp.dot(axis, wp.vec3(twist[0], twist[1], twist[2])))  <L 1253>
            adj_quat_twist(var_177, var_105, adj_177, adj_105, adj_180);
            // adj: twist = quat_twist(axis, r_err)                                               <L 1251>
            wp::adj_transform_vector(var_51, var_177, adj_51, adj_177, adj_179);
            // adj: axis_c = wp.transform_vector(X_wc, axis)                                      <L 1248>
            wp::adj_transform_vector(var_45, var_177, adj_45, adj_177, adj_178);
            // adj: axis_p = wp.transform_vector(X_wp, axis)                                      <L 1247>
            wp::adj_copy(var_176, adj_175, adj_177);
            wp::adj_load(var_175, adj_175, adj_176);
            wp::adj_address(var_joint_axis, var_74, adj_joint_axis, adj_74, adj_175);
            // adj: axis = joint_axis[axis_start]                                                 <L 1245>
        }
        // adj: if type == fs5Model.JOINT_REVOLUTE:                                               <L 1244>
        wp::adj_select(var_139, var_125, var_155, adj_139, adj_125, adj_155, adj_172);
        wp::adj_select(var_139, var_137, var_163, adj_139, adj_137, adj_163, adj_171);
        wp::adj_select(var_139, var_136, var_169, adj_139, adj_136, adj_169, adj_170);
        if (var_139) {
            wp::adj_add(var_136, var_168, adj_136, adj_168, adj_169);
            wp::adj_add(var_165, var_167, adj_165, adj_167, adj_168);
            wp::adj_mul(var_166, var_110, adj_166, adj_110, adj_167);
            wp::adj_mul(var_107, var_joint_attach_kd, adj_107, adj_joint_attach_kd, adj_166);
            wp::adj_mul(var_164, var_joint_attach_ke, adj_164, adj_joint_attach_ke, adj_165);
            wp::adj_transform_vector(var_45, var_155, adj_45, adj_155, adj_164);
            // adj: wp.transform_vector(X_wp, ang_err) * joint_attach_ke + w_err * joint_attach_kd * angular_damping_scale  <L 1241>
            // adj: t_total += (                                                                  <L 1240>
            wp::adj_add(var_146, var_162, adj_146, adj_162, adj_163);
            wp::adj_add(var_158, var_161, adj_158, adj_161, adj_162);
            wp::adj_mul(var_160, var_joint_attach_kd, adj_160, adj_joint_attach_kd, adj_161);
            wp::adj_sub(var_106, var_159, adj_106, adj_159, adj_160);
            wp::adj_mul(var_145, var_143, adj_145, adj_143, adj_159);
            wp::adj_mul(var_157, var_joint_attach_ke, adj_157, adj_joint_attach_ke, adj_158);
            wp::adj_sub(var_103, var_156, adj_103, adj_156, adj_157);
            wp::adj_mul(var_144, var_143, adj_144, adj_143, adj_156);
            // adj: f_total += (x_err - q * axis_p) * joint_attach_ke + (v_err - qd * axis_p) * joint_attach_kd  <L 1239>
            wp::adj_mul(var_154, var_124, adj_154, adj_124, adj_155);
            wp::adj_mul(var_151, var_153, adj_151, adj_153, adj_154);
            wp::adj_acos(var_152, adj_152, adj_153);
            wp::adj_extract(var_105, var_120, adj_105, adj_120, adj_152);
            wp::adj_normalize(var_150, var_151, adj_150, adj_151);
            wp::adj_vec_t(var_147, var_148, var_149, adj_147, adj_148, adj_149, adj_150);
            wp::adj_extract(var_105, var_116, adj_105, adj_116, adj_149);
            wp::adj_extract(var_105, var_114, adj_105, adj_114, adj_148);
            wp::adj_extract(var_105, var_5, adj_105, adj_5, adj_147);
            // adj: ang_err = wp.normalize(wp.vec3(r_err[0], r_err[1], r_err[2])) * wp.acos(r_err[3]) * 2.0  <L 1236>
            adj_eval_joint_force(var_144, var_145, var_77, var_80, var_83, var_98, var_92, var_95, var_86, var_89, var_143, adj_144, adj_145, adj_77, adj_80, adj_83, adj_98, adj_92, adj_95, adj_86, adj_89, adj_143, adj_146);
            // adj: q, qd, target, target_ke, target_kd, act, limit_lower, limit_upper, limit_ke, limit_kd, axis_p  <L 1232>
            // adj: f_total = eval_joint_force(                                                   <L 1231>
            wp::adj_dot(var_106, var_143, adj_106, adj_143, adj_145);
            // adj: qd = wp.dot(v_err, axis_p)                                                    <L 1229>
            wp::adj_dot(var_103, var_143, adj_103, adj_143, adj_144);
            // adj: q = wp.dot(x_err, axis_p)                                                     <L 1228>
            wp::adj_transform_vector(var_45, var_142, adj_45, adj_142, adj_143);
            // adj: axis_p = wp.transform_vector(X_wp, axis)                                      <L 1225>
            wp::adj_copy(var_141, adj_140, adj_142);
            wp::adj_load(var_140, adj_140, adj_141);
            wp::adj_address(var_joint_axis, var_74, adj_joint_axis, adj_74, adj_140);
            // adj: axis = joint_axis[axis_start]                                                 <L 1222>
        }
        // adj: if type == fs5Model.JOINT_PRISMATIC:                                              <L 1221>
        wp::adj_select(var_112, var_109, var_129, adj_112, adj_109, adj_129, adj_137);
        wp::adj_select(var_112, var_108, var_135, adj_112, adj_108, adj_135, adj_136);
        if (var_112) {
            wp::adj_add(var_108, var_134, adj_108, adj_134, adj_135);
            wp::adj_add(var_131, var_133, adj_131, adj_133, adj_134);
            wp::adj_mul(var_132, var_110, adj_132, adj_110, adj_133);
            wp::adj_mul(var_107, var_joint_attach_kd, adj_107, adj_joint_attach_kd, adj_132);
            wp::adj_mul(var_130, var_joint_attach_ke, adj_130, adj_joint_attach_ke, adj_131);
            wp::adj_transform_vector(var_45, var_125, adj_45, adj_125, adj_130);
            // adj: wp.transform_vector(X_wp, ang_err) * joint_attach_ke + w_err * joint_attach_kd * angular_damping_scale  <L 1218>
            // adj: t_total += (                                                                  <L 1217>
            wp::adj_add(var_109, var_128, adj_109, adj_128, adj_129);
            wp::adj_add(var_126, var_127, adj_126, adj_127, adj_128);
            wp::adj_mul(var_106, var_joint_attach_kd, adj_106, adj_joint_attach_kd, adj_127);
            wp::adj_mul(var_103, var_joint_attach_ke, adj_103, adj_joint_attach_ke, adj_126);
            // adj: f_total += x_err * joint_attach_ke + v_err * joint_attach_kd                  <L 1216>
            wp::adj_mul(var_123, var_124, adj_123, adj_124, adj_125);
            wp::adj_mul(var_119, var_122, adj_119, adj_122, adj_123);
            wp::adj_acos(var_121, adj_121, adj_122);
            wp::adj_extract(var_105, var_120, adj_105, adj_120, adj_121);
            wp::adj_normalize(var_118, var_119, adj_118, adj_119);
            wp::adj_vec_t(var_113, var_115, var_117, adj_113, adj_115, adj_117, adj_118);
            wp::adj_extract(var_105, var_116, adj_105, adj_116, adj_117);
            wp::adj_extract(var_105, var_114, adj_105, adj_114, adj_115);
            wp::adj_extract(var_105, var_5, adj_105, adj_5, adj_113);
            // adj: ang_err = wp.normalize(wp.vec3(r_err[0], r_err[1], r_err[2])) * wp.acos(r_err[3]) * 2.0  <L 1214>
        }
        // adj: if type == fs5Model.JOINT_FIXED:                                                  <L 1213>
        // adj: angular_damping_scale = 0.01                                                      <L 1211>
        // adj: f_total = wp.vec3()                                                               <L 1208>
        // adj: t_total = wp.vec3()                                                               <L 1207>
        wp::adj_sub(var_62, var_47, adj_62, adj_47, adj_107);
        // adj: w_err = w_c - w_p                                                                 <L 1204>
        wp::adj_sub(var_65, var_48, adj_65, adj_48, adj_106);
        // adj: v_err = v_c - v_p                                                                 <L 1203>
        wp::adj_mul(var_104, var_102, adj_104, adj_102, adj_105);
        wp::adj_quat_inverse(var_101, adj_101, adj_104);
        // adj: r_err = wp.quat_inverse(q_p) * q_c                                                <L 1202>
        wp::adj_sub(var_100, var_99, adj_100, adj_99, adj_103);
        // adj: x_err = x_c - x_p                                                                 <L 1201>
        wp::adj_transform_get_rotation(var_51, adj_51, adj_102);
        // adj: q_c = wp.transform_get_rotation(X_wc)                                             <L 1198>
        wp::adj_transform_get_rotation(var_45, adj_45, adj_101);
        // adj: q_p = wp.transform_get_rotation(X_wp)                                             <L 1197>
        wp::adj_transform_get_translation(var_51, adj_51, adj_100);
        // adj: x_c = wp.transform_get_translation(X_wc)                                          <L 1195>
        wp::adj_transform_get_translation(var_45, adj_45, adj_99);
        // adj: x_p = wp.transform_get_translation(X_wp)                                          <L 1194>
        wp::adj_copy(var_97, adj_96, adj_98);
        wp::adj_load(var_96, adj_96, adj_97);
        wp::adj_address(var_joint_act, var_71, adj_joint_act, adj_71, adj_96);
        // adj: act = joint_act[qd_start]                                                         <L 1192>
        wp::adj_copy(var_94, adj_93, adj_95);
        wp::adj_load(var_93, adj_93, adj_94);
        wp::adj_address(var_joint_limit_upper, var_74, adj_joint_limit_upper, adj_74, adj_93);
        // adj: limit_upper = joint_limit_upper[axis_start]                                       <L 1190>
        wp::adj_copy(var_91, adj_90, adj_92);
        wp::adj_load(var_90, adj_90, adj_91);
        wp::adj_address(var_joint_limit_lower, var_74, adj_joint_limit_lower, adj_74, adj_90);
        // adj: limit_lower = joint_limit_lower[axis_start]                                       <L 1189>
        wp::adj_copy(var_88, adj_87, adj_89);
        wp::adj_load(var_87, adj_87, adj_88);
        wp::adj_address(var_joint_limit_kd, var_74, adj_joint_limit_kd, adj_74, adj_87);
        // adj: limit_kd = joint_limit_kd[axis_start]                                             <L 1188>
        wp::adj_copy(var_85, adj_84, adj_86);
        wp::adj_load(var_84, adj_84, adj_85);
        wp::adj_address(var_joint_limit_ke, var_74, adj_joint_limit_ke, adj_74, adj_84);
        // adj: limit_ke = joint_limit_ke[axis_start]                                             <L 1187>
        wp::adj_copy(var_82, adj_81, adj_83);
        wp::adj_load(var_81, adj_81, adj_82);
        wp::adj_address(var_joint_target_kd, var_74, adj_joint_target_kd, adj_74, adj_81);
        // adj: target_kd = joint_target_kd[axis_start]                                           <L 1186>
        wp::adj_copy(var_79, adj_78, adj_80);
        wp::adj_load(var_78, adj_78, adj_79);
        wp::adj_address(var_joint_target_ke, var_74, adj_joint_target_ke, adj_74, adj_78);
        // adj: target_ke = joint_target_ke[axis_start]                                           <L 1185>
        wp::adj_copy(var_76, adj_75, adj_77);
        wp::adj_load(var_75, adj_75, adj_76);
        wp::adj_address(var_joint_target, var_74, adj_joint_target, adj_74, adj_75);
        // adj: target = joint_target[axis_start]                                                 <L 1184>
        wp::adj_copy(var_73, adj_72, adj_74);
        wp::adj_load(var_72, adj_72, adj_73);
        wp::adj_address(var_joint_axis_start, var_0, adj_joint_axis_start, adj_0, adj_72);
        // adj: axis_start = joint_axis_start[tid]                                                <L 1182>
        wp::adj_copy(var_70, adj_69, adj_71);
        wp::adj_load(var_69, adj_69, adj_70);
        wp::adj_address(var_joint_qd_start, var_0, adj_joint_qd_start, adj_0, adj_69);
        // adj: qd_start = joint_qd_start[tid]                                                    <L 1181>
        wp::adj_copy(var_67, adj_66, adj_68);
        wp::adj_load(var_66, adj_66, adj_67);
        wp::adj_address(var_joint_q_start, var_0, adj_joint_q_start, adj_0, adj_66);
        // adj: q_start = joint_q_start[tid]                                                      <L 1180>
        wp::adj_add(var_63, var_64, adj_63, adj_64, adj_65);
        wp::adj_cross(var_62, var_58, adj_62, adj_58, adj_64);
        wp::adj_spatial_bottom(var_61, adj_61, adj_63);
        // adj: v_c = wp.spatial_bottom(twist_c) + wp.cross(w_c, r_c)                             <L 1177>
        wp::adj_spatial_top(var_61, adj_61, adj_62);
        // adj: w_c = wp.spatial_top(twist_c)                                                     <L 1176>
        wp::adj_copy(var_60, adj_59, adj_61);
        wp::adj_load(var_59, adj_59, adj_60);
        wp::adj_address(var_body_qd, var_13, adj_body_qd, adj_13, adj_59);
        // adj: twist_c = body_qd[c_child]                                                        <L 1174>
        wp::adj_sub(var_52, var_57, adj_52, adj_57, adj_58);
        wp::adj_transform_point(var_55, var_56, adj_53, adj_54, adj_57);
        wp::adj_load(var_54, adj_54, adj_56);
        wp::adj_load(var_53, adj_53, adj_55);
        wp::adj_address(var_body_com, var_13, adj_body_com, adj_13, adj_54);
        wp::adj_address(var_body_q, var_13, adj_body_q, adj_13, adj_53);
        wp::adj_transform_get_translation(var_51, adj_51, adj_52);
        // adj: r_c = wp.transform_get_translation(X_wc) - wp.transform_point(body_q[c_child], body_com[c_child])  <L 1172>
        wp::adj_mul(var_50, var_22, adj_49, adj_22, adj_51);
        wp::adj_load(var_49, adj_49, adj_50);
        wp::adj_address(var_body_q, var_13, adj_body_q, adj_13, adj_49);
        // adj: X_wc = body_q[c_child] * X_cj                                                     <L 1171>
        wp::adj_select(var_27, var_26, var_44, adj_27, adj_26, adj_44, adj_48);
        wp::adj_select(var_27, var_25, var_41, adj_27, adj_25, adj_41, adj_47);
        wp::adj_select(var_27, var_24, var_37, adj_27, adj_24, adj_37, adj_46);
        wp::adj_select(var_27, var_23, var_30, adj_27, adj_23, adj_30, adj_45);
        if (var_27) {
            wp::adj_add(var_42, var_43, adj_42, adj_43, adj_44);
            wp::adj_cross(var_41, var_37, adj_41, adj_37, adj_43);
            wp::adj_spatial_bottom(var_40, adj_40, adj_42);
            // adj: v_p = wp.spatial_bottom(twist_p) + wp.cross(w_p, r_p)                         <L 1168>
            wp::adj_spatial_top(var_40, adj_40, adj_41);
            // adj: w_p = wp.spatial_top(twist_p)                                                 <L 1167>
            wp::adj_copy(var_39, adj_38, adj_40);
            wp::adj_load(var_38, adj_38, adj_39);
            wp::adj_address(var_body_qd, var_16, adj_body_qd, adj_16, adj_38);
            // adj: twist_p = body_qd[c_parent]                                                   <L 1165>
            wp::adj_sub(var_31, var_36, adj_31, adj_36, adj_37);
            wp::adj_transform_point(var_34, var_35, adj_32, adj_33, adj_36);
            wp::adj_load(var_33, adj_33, adj_35);
            wp::adj_load(var_32, adj_32, adj_34);
            wp::adj_address(var_body_com, var_16, adj_body_com, adj_16, adj_33);
            wp::adj_address(var_body_q, var_16, adj_body_q, adj_16, adj_32);
            wp::adj_transform_get_translation(var_30, adj_30, adj_31);
            // adj: r_p = wp.transform_get_translation(X_wp) - wp.transform_point(body_q[c_parent], body_com[c_parent])  <L 1163>
            wp::adj_mul(var_29, var_23, adj_28, adj_23, adj_30);
            wp::adj_load(var_28, adj_28, adj_29);
            wp::adj_address(var_body_q, var_16, adj_body_q, adj_16, adj_28);
            // adj: X_wp = body_q[c_parent] * X_wp                                                <L 1162>
        }
        // adj: if c_parent >= 0:                                                                 <L 1161>
        // adj: v_p = wp.vec3()                                                                   <L 1158>
        // adj: w_p = wp.vec3()                                                                   <L 1157>
        // adj: r_p = wp.vec3()                                                                   <L 1156>
        wp::adj_copy(var_19, adj_19, adj_23);
        // adj: X_wp = X_pj                                                                       <L 1155>
        wp::adj_copy(var_21, adj_20, adj_22);
        wp::adj_load(var_20, adj_20, adj_21);
        wp::adj_address(var_joint_X_c, var_0, adj_joint_X_c, adj_0, adj_20);
        // adj: X_cj = joint_X_c[tid]                                                             <L 1153>
        wp::adj_copy(var_18, adj_17, adj_19);
        wp::adj_load(var_17, adj_17, adj_18);
        wp::adj_address(var_joint_X_p, var_0, adj_joint_X_p, adj_0, adj_17);
        // adj: X_pj = joint_X_p[tid]                                                             <L 1152>
        wp::adj_copy(var_15, adj_14, adj_16);
        wp::adj_load(var_14, adj_14, adj_15);
        wp::adj_address(var_joint_parent, var_0, adj_joint_parent, adj_0, adj_14);
        // adj: c_parent = joint_parent[tid]                                                      <L 1150>
        wp::adj_copy(var_12, adj_11, adj_13);
        wp::adj_load(var_11, adj_11, adj_12);
        wp::adj_address(var_joint_child, var_0, adj_joint_child, adj_0, adj_11);
        // adj: c_child = joint_child[tid]                                                        <L 1149>
        if (var_10) {
            label0:;
            // adj: return                                                                        <L 1147>
        }
        wp::adj_load(var_4, adj_4, adj_7);
        wp::adj_address(var_joint_enabled, var_0, adj_joint_enabled, adj_0, adj_4);
        // adj: if joint_enabled[tid] == 0 or type == fs5Model.JOINT_FREE:                        <L 1146>
        wp::adj_copy(var_2, adj_1, adj_3);
        wp::adj_load(var_1, adj_1, adj_2);
        wp::adj_address(var_joint_type, var_0, adj_joint_type, adj_0, adj_1);
        // adj: type = joint_type[tid]                                                            <L 1143>
        // adj: tid = wp.tid()                                                                    <L 1142>
        // adj: def eval_body_joints(                                                             <L 1115>
        continue;
    }
}



extern "C" __global__ void eval_muscles_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::transform_t<wp::float32>> var_body_X_s,
    wp::array_t<wp::vec_t<6,wp::float32>> var_body_v_s,
    wp::array_t<wp::vec_t<3,wp::float32>> var_body_com,
    wp::array_t<wp::int32> var_muscle_start,
    wp::array_t<wp::float32> var_muscle_params,
    wp::array_t<wp::int32> var_muscle_links,
    wp::array_t<wp::vec_t<3,wp::float32>> var_muscle_points,
    wp::array_t<wp::float32> var_muscle_activation,
    wp::array_t<wp::vec_t<6,wp::float32>> var_body_f_s)
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
        const wp::int32 var_4 = 1;
        wp::int32 var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::float32* var_9;
        wp::float32 var_10;
        wp::float32 var_11;
        wp::range_t var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        //---------
        // forward
        // def eval_muscles(                                                                      <L 1467>
        // tid = wp.tid()                                                                         <L 1479>
        var_0 = builtin_tid1d();
        // m_start = muscle_start[tid]                                                            <L 1481>
        var_1 = wp::address(var_muscle_start, var_0);
        var_2 = wp::load(var_1);
        var_3 = wp::copy(var_2);
        // m_end = muscle_start[tid + 1] - 1                                                      <L 1482>
        var_5 = wp::add(var_0, var_4);
        var_6 = wp::address(var_muscle_start, var_5);
        var_7 = wp::load(var_6);
        var_8 = wp::sub(var_7, var_4);
        // activation = muscle_activation[tid]                                                    <L 1484>
        var_9 = wp::address(var_muscle_activation, var_0);
        var_10 = wp::load(var_9);
        var_11 = wp::copy(var_10);
        // for i in range(m_start, m_end):                                                        <L 1486>
        var_12 = wp::range(var_3, var_8);
        start_for_0:;
            if (iter_cmp(var_12) == 0) goto end_for_0;
            var_13 = wp::iter_next(var_12);
            // compute_muscle_force(i, body_X_s, body_v_s, body_com, muscle_links, muscle_points, activation, body_f_s)       <L 1487>
            var_14 = compute_muscle_force(var_13, var_body_X_s, var_body_v_s, var_body_com, var_muscle_links, var_muscle_points, var_11, var_body_f_s);
            goto start_for_0;
        end_for_0:;
    }
}

extern "C" __global__ void eval_muscles_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::transform_t<wp::float32>> var_body_X_s,
    wp::array_t<wp::vec_t<6,wp::float32>> var_body_v_s,
    wp::array_t<wp::vec_t<3,wp::float32>> var_body_com,
    wp::array_t<wp::int32> var_muscle_start,
    wp::array_t<wp::float32> var_muscle_params,
    wp::array_t<wp::int32> var_muscle_links,
    wp::array_t<wp::vec_t<3,wp::float32>> var_muscle_points,
    wp::array_t<wp::float32> var_muscle_activation,
    wp::array_t<wp::vec_t<6,wp::float32>> var_body_f_s,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_X_s,
    wp::array_t<wp::vec_t<6,wp::float32>> adj_body_v_s,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_body_com,
    wp::array_t<wp::int32> adj_muscle_start,
    wp::array_t<wp::float32> adj_muscle_params,
    wp::array_t<wp::int32> adj_muscle_links,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_muscle_points,
    wp::array_t<wp::float32> adj_muscle_activation,
    wp::array_t<wp::vec_t<6,wp::float32>> adj_body_f_s)
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
        const wp::int32 var_4 = 1;
        wp::int32 var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::float32* var_9;
        wp::float32 var_10;
        wp::float32 var_11;
        wp::range_t var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::range_t adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        //---------
        // forward
        // def eval_muscles(                                                                      <L 1467>
        // tid = wp.tid()                                                                         <L 1479>
        var_0 = builtin_tid1d();
        // m_start = muscle_start[tid]                                                            <L 1481>
        var_1 = wp::address(var_muscle_start, var_0);
        var_2 = wp::load(var_1);
        var_3 = wp::copy(var_2);
        // m_end = muscle_start[tid + 1] - 1                                                      <L 1482>
        var_5 = wp::add(var_0, var_4);
        var_6 = wp::address(var_muscle_start, var_5);
        var_7 = wp::load(var_6);
        var_8 = wp::sub(var_7, var_4);
        // activation = muscle_activation[tid]                                                    <L 1484>
        var_9 = wp::address(var_muscle_activation, var_0);
        var_10 = wp::load(var_9);
        var_11 = wp::copy(var_10);
        // for i in range(m_start, m_end):                                                        <L 1486>
        var_12 = wp::range(var_3, var_8);
        //---------
        // reverse
        var_12 = wp::iter_reverse(var_12);
        start_for_0:;
            if (iter_cmp(var_12) == 0) goto end_for_0;
            var_13 = wp::iter_next(var_12);
        	adj_14 = {};
            // compute_muscle_force(i, body_X_s, body_v_s, body_com, muscle_links, muscle_points, activation, body_f_s)       <L 1487>
            var_14 = compute_muscle_force(var_13, var_body_X_s, var_body_v_s, var_body_com, var_muscle_links, var_muscle_points, var_11, var_body_f_s);
            adj_compute_muscle_force(var_13, var_body_X_s, var_body_v_s, var_body_com, var_muscle_links, var_muscle_points, var_11, var_body_f_s, adj_13, adj_body_X_s, adj_body_v_s, adj_body_com, adj_muscle_links, adj_muscle_points, adj_11, adj_body_f_s, adj_14);
            // adj: compute_muscle_force(i, body_X_s, body_v_s, body_com, muscle_links, muscle_points, activation, body_f_s)  <L 1487>
        	goto start_for_0;
        end_for_0:;
        wp::adj_range(var_3, var_8, adj_3, adj_8, adj_12);
        // adj: for i in range(m_start, m_end):                                                   <L 1486>
        wp::adj_copy(var_10, adj_9, adj_11);
        wp::adj_load(var_9, adj_9, adj_10);
        wp::adj_address(var_muscle_activation, var_0, adj_muscle_activation, adj_0, adj_9);
        // adj: activation = muscle_activation[tid]                                               <L 1484>
        wp::adj_sub(var_7, var_4, adj_6, adj_4, adj_8);
        wp::adj_load(var_6, adj_6, adj_7);
        wp::adj_address(var_muscle_start, var_5, adj_muscle_start, adj_5, adj_6);
        wp::adj_add(var_0, var_4, adj_0, adj_4, adj_5);
        // adj: m_end = muscle_start[tid + 1] - 1                                                 <L 1482>
        wp::adj_copy(var_2, adj_1, adj_3);
        wp::adj_load(var_1, adj_1, adj_2);
        wp::adj_address(var_muscle_start, var_0, adj_muscle_start, adj_0, adj_1);
        // adj: m_start = muscle_start[tid]                                                       <L 1481>
        // adj: tid = wp.tid()                                                                    <L 1479>
        // adj: def eval_muscles(                                                                 <L 1467>
        continue;
    }
}



extern "C" __global__ void compute_particle_residual_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_qd_0,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_qd_1,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_f,
    wp::array_t<wp::float32> var_particle_m,
    wp::array_t<wp::uint32> var_particle_flags,
    wp::vec_t<3,wp::float32> var_gravity,
    wp::float32 var_dt,
    wp::array_t<wp::vec_t<3,wp::float32>> var_residual)
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
        wp::vec_t<3,wp::float32>* var_10;
        wp::vec_t<3,wp::float32> var_11;
        wp::vec_t<3,wp::float32> var_12;
        wp::vec_t<3,wp::float32>* var_13;
        wp::vec_t<3,wp::float32> var_14;
        wp::vec_t<3,wp::float32> var_15;
        wp::vec_t<3,wp::float32>* var_16;
        wp::vec_t<3,wp::float32> var_17;
        wp::vec_t<3,wp::float32> var_18;
        wp::vec_t<3,wp::float32> var_19;
        const wp::float32 var_20 = 0.0;
        bool var_21;
        wp::vec_t<3,wp::float32> var_22;
        wp::vec_t<3,wp::float32> var_23;
        wp::vec_t<3,wp::float32> var_24;
        wp::vec_t<3,wp::float32> var_25;
        wp::vec_t<3,wp::float32> var_26;
        wp::vec_t<3,wp::float32> var_27;
        wp::vec_t<3,wp::float32> var_28;
        wp::vec_t<3,wp::float32> var_29;
        //---------
        // forward
        // def compute_particle_residual(                                                         <L 1837>
        // tid = wp.tid()                                                                         <L 1847>
        var_0 = builtin_tid1d();
        // if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:                                  <L 1848>
        var_1 = wp::address(var_particle_flags, var_0);
        var_3 = wp::load(var_1);
        var_4 = wp::bit_and(var_3, var_2);
        var_6 = (var_4 == var_5);
        if (var_6) {
            // return                                                                             <L 1849>
            return;
        }
        // m = particle_m[tid]                                                                    <L 1851>
        var_7 = wp::address(var_particle_m, var_0);
        var_8 = wp::load(var_7);
        var_9 = wp::copy(var_8);
        // v1 = particle_qd_1[tid]                                                                <L 1852>
        var_10 = wp::address(var_particle_qd_1, var_0);
        var_11 = wp::load(var_10);
        var_12 = wp::copy(var_11);
        // v0 = particle_qd_0[tid]                                                                <L 1853>
        var_13 = wp::address(var_particle_qd_0, var_0);
        var_14 = wp::load(var_13);
        var_15 = wp::copy(var_14);
        // f = particle_f[tid]                                                                    <L 1854>
        var_16 = wp::address(var_particle_f, var_0);
        var_17 = wp::load(var_16);
        var_18 = wp::copy(var_17);
        // err = wp.vec3()                                                                        <L 1856>
        var_19 = wp::vec_t<3,wp::float32>();
        // if m > 0.0:                                                                            <L 1858>
        var_21 = (var_9 > var_20);
        if (var_21) {
            // err = (v1 - v0) * m - f * dt - gravity * dt * m                                    <L 1859>
            var_22 = wp::sub(var_12, var_15);
            var_23 = wp::mul(var_22, var_9);
            var_24 = wp::mul(var_18, var_dt);
            var_25 = wp::sub(var_23, var_24);
            var_26 = wp::mul(var_gravity, var_dt);
            var_27 = wp::mul(var_26, var_9);
            var_28 = wp::sub(var_25, var_27);
        }
        var_29 = wp::select(var_21, var_19, var_28);
        // residual[tid] = err                                                                    <L 1861>
        wp::array_store(var_residual, var_0, var_29);
    }
}

extern "C" __global__ void compute_particle_residual_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_qd_0,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_qd_1,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_f,
    wp::array_t<wp::float32> var_particle_m,
    wp::array_t<wp::uint32> var_particle_flags,
    wp::vec_t<3,wp::float32> var_gravity,
    wp::float32 var_dt,
    wp::array_t<wp::vec_t<3,wp::float32>> var_residual,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_particle_qd_0,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_particle_qd_1,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_particle_f,
    wp::array_t<wp::float32> adj_particle_m,
    wp::array_t<wp::uint32> adj_particle_flags,
    wp::vec_t<3,wp::float32> adj_gravity,
    wp::float32 adj_dt,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_residual)
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
        wp::vec_t<3,wp::float32>* var_10;
        wp::vec_t<3,wp::float32> var_11;
        wp::vec_t<3,wp::float32> var_12;
        wp::vec_t<3,wp::float32>* var_13;
        wp::vec_t<3,wp::float32> var_14;
        wp::vec_t<3,wp::float32> var_15;
        wp::vec_t<3,wp::float32>* var_16;
        wp::vec_t<3,wp::float32> var_17;
        wp::vec_t<3,wp::float32> var_18;
        wp::vec_t<3,wp::float32> var_19;
        const wp::float32 var_20 = 0.0;
        bool var_21;
        wp::vec_t<3,wp::float32> var_22;
        wp::vec_t<3,wp::float32> var_23;
        wp::vec_t<3,wp::float32> var_24;
        wp::vec_t<3,wp::float32> var_25;
        wp::vec_t<3,wp::float32> var_26;
        wp::vec_t<3,wp::float32> var_27;
        wp::vec_t<3,wp::float32> var_28;
        wp::vec_t<3,wp::float32> var_29;
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
        wp::vec_t<3,wp::float32> adj_10 = {};
        wp::vec_t<3,wp::float32> adj_11 = {};
        wp::vec_t<3,wp::float32> adj_12 = {};
        wp::vec_t<3,wp::float32> adj_13 = {};
        wp::vec_t<3,wp::float32> adj_14 = {};
        wp::vec_t<3,wp::float32> adj_15 = {};
        wp::vec_t<3,wp::float32> adj_16 = {};
        wp::vec_t<3,wp::float32> adj_17 = {};
        wp::vec_t<3,wp::float32> adj_18 = {};
        wp::vec_t<3,wp::float32> adj_19 = {};
        wp::float32 adj_20 = {};
        bool adj_21 = {};
        wp::vec_t<3,wp::float32> adj_22 = {};
        wp::vec_t<3,wp::float32> adj_23 = {};
        wp::vec_t<3,wp::float32> adj_24 = {};
        wp::vec_t<3,wp::float32> adj_25 = {};
        wp::vec_t<3,wp::float32> adj_26 = {};
        wp::vec_t<3,wp::float32> adj_27 = {};
        wp::vec_t<3,wp::float32> adj_28 = {};
        wp::vec_t<3,wp::float32> adj_29 = {};
        //---------
        // forward
        // def compute_particle_residual(                                                         <L 1837>
        // tid = wp.tid()                                                                         <L 1847>
        var_0 = builtin_tid1d();
        // if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:                                  <L 1848>
        var_1 = wp::address(var_particle_flags, var_0);
        var_3 = wp::load(var_1);
        var_4 = wp::bit_and(var_3, var_2);
        var_6 = (var_4 == var_5);
        if (var_6) {
            // return                                                                             <L 1849>
            goto label0;
        }
        // m = particle_m[tid]                                                                    <L 1851>
        var_7 = wp::address(var_particle_m, var_0);
        var_8 = wp::load(var_7);
        var_9 = wp::copy(var_8);
        // v1 = particle_qd_1[tid]                                                                <L 1852>
        var_10 = wp::address(var_particle_qd_1, var_0);
        var_11 = wp::load(var_10);
        var_12 = wp::copy(var_11);
        // v0 = particle_qd_0[tid]                                                                <L 1853>
        var_13 = wp::address(var_particle_qd_0, var_0);
        var_14 = wp::load(var_13);
        var_15 = wp::copy(var_14);
        // f = particle_f[tid]                                                                    <L 1854>
        var_16 = wp::address(var_particle_f, var_0);
        var_17 = wp::load(var_16);
        var_18 = wp::copy(var_17);
        // err = wp.vec3()                                                                        <L 1856>
        var_19 = wp::vec_t<3,wp::float32>();
        // if m > 0.0:                                                                            <L 1858>
        var_21 = (var_9 > var_20);
        if (var_21) {
            // err = (v1 - v0) * m - f * dt - gravity * dt * m                                    <L 1859>
            var_22 = wp::sub(var_12, var_15);
            var_23 = wp::mul(var_22, var_9);
            var_24 = wp::mul(var_18, var_dt);
            var_25 = wp::sub(var_23, var_24);
            var_26 = wp::mul(var_gravity, var_dt);
            var_27 = wp::mul(var_26, var_9);
            var_28 = wp::sub(var_25, var_27);
        }
        var_29 = wp::select(var_21, var_19, var_28);
        // residual[tid] = err                                                                    <L 1861>
        // wp::array_store(var_residual, var_0, var_29);
        //---------
        // reverse
        wp::adj_array_store(var_residual, var_0, var_29, adj_residual, adj_0, adj_29);
        // adj: residual[tid] = err                                                               <L 1861>
        wp::adj_select(var_21, var_19, var_28, adj_21, adj_19, adj_28, adj_29);
        if (var_21) {
            wp::adj_sub(var_25, var_27, adj_25, adj_27, adj_28);
            wp::adj_mul(var_26, var_9, adj_26, adj_9, adj_27);
            wp::adj_mul(var_gravity, var_dt, adj_gravity, adj_dt, adj_26);
            wp::adj_sub(var_23, var_24, adj_23, adj_24, adj_25);
            wp::adj_mul(var_18, var_dt, adj_18, adj_dt, adj_24);
            wp::adj_mul(var_22, var_9, adj_22, adj_9, adj_23);
            wp::adj_sub(var_12, var_15, adj_12, adj_15, adj_22);
            // adj: err = (v1 - v0) * m - f * dt - gravity * dt * m                               <L 1859>
        }
        // adj: if m > 0.0:                                                                       <L 1858>
        // adj: err = wp.vec3()                                                                   <L 1856>
        wp::adj_copy(var_17, adj_16, adj_18);
        wp::adj_load(var_16, adj_16, adj_17);
        wp::adj_address(var_particle_f, var_0, adj_particle_f, adj_0, adj_16);
        // adj: f = particle_f[tid]                                                               <L 1854>
        wp::adj_copy(var_14, adj_13, adj_15);
        wp::adj_load(var_13, adj_13, adj_14);
        wp::adj_address(var_particle_qd_0, var_0, adj_particle_qd_0, adj_0, adj_13);
        // adj: v0 = particle_qd_0[tid]                                                           <L 1853>
        wp::adj_copy(var_11, adj_10, adj_12);
        wp::adj_load(var_10, adj_10, adj_11);
        wp::adj_address(var_particle_qd_1, var_0, adj_particle_qd_1, adj_0, adj_10);
        // adj: v1 = particle_qd_1[tid]                                                           <L 1852>
        wp::adj_copy(var_8, adj_7, adj_9);
        wp::adj_load(var_7, adj_7, adj_8);
        wp::adj_address(var_particle_m, var_0, adj_particle_m, adj_0, adj_7);
        // adj: m = particle_m[tid]                                                               <L 1851>
        if (var_6) {
            label0:;
            // adj: return                                                                        <L 1849>
        }
        wp::adj_bit_and(var_3, var_2, adj_1, adj_2, adj_4);
        wp::adj_load(var_1, adj_1, adj_3);
        wp::adj_address(var_particle_flags, var_0, adj_particle_flags, adj_0, adj_1);
        // adj: if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:                             <L 1848>
        // adj: tid = wp.tid()                                                                    <L 1847>
        // adj: def compute_particle_residual(                                                    <L 1837>
        continue;
    }
}



extern "C" __global__ void update_particle_position_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_q_0,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_q_1,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_qd_1,
    wp::array_t<wp::vec_t<3,wp::float32>> var_x,
    wp::array_t<wp::uint32> var_particle_flags,
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
        wp::vec_t<3,wp::float32> var_8;
        wp::vec_t<3,wp::float32> var_9;
        wp::vec_t<3,wp::float32>* var_10;
        wp::vec_t<3,wp::float32> var_11;
        wp::vec_t<3,wp::float32> var_12;
        wp::vec_t<3,wp::float32> var_13;
        wp::vec_t<3,wp::float32> var_14;
        //---------
        // forward
        // def update_particle_position(                                                          <L 1865>
        // tid = wp.tid()                                                                         <L 1873>
        var_0 = builtin_tid1d();
        // if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:                                  <L 1874>
        var_1 = wp::address(var_particle_flags, var_0);
        var_3 = wp::load(var_1);
        var_4 = wp::bit_and(var_3, var_2);
        var_6 = (var_4 == var_5);
        if (var_6) {
            // return                                                                             <L 1875>
            return;
        }
        // qd_1 = x[tid]                                                                          <L 1877>
        var_7 = wp::address(var_x, var_0);
        var_8 = wp::load(var_7);
        var_9 = wp::copy(var_8);
        // q_0 = particle_q_0[tid]                                                                <L 1879>
        var_10 = wp::address(var_particle_q_0, var_0);
        var_11 = wp::load(var_10);
        var_12 = wp::copy(var_11);
        // q_1 = q_0 + qd_1 * dt                                                                  <L 1880>
        var_13 = wp::mul(var_9, var_dt);
        var_14 = wp::add(var_12, var_13);
        // particle_q_1[tid] = q_1                                                                <L 1882>
        wp::array_store(var_particle_q_1, var_0, var_14);
        // particle_qd_1[tid] = qd_1                                                              <L 1883>
        wp::array_store(var_particle_qd_1, var_0, var_9);
    }
}

extern "C" __global__ void update_particle_position_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_q_0,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_q_1,
    wp::array_t<wp::vec_t<3,wp::float32>> var_particle_qd_1,
    wp::array_t<wp::vec_t<3,wp::float32>> var_x,
    wp::array_t<wp::uint32> var_particle_flags,
    wp::float32 var_dt,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_particle_q_0,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_particle_q_1,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_particle_qd_1,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_x,
    wp::array_t<wp::uint32> adj_particle_flags,
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
        wp::vec_t<3,wp::float32> var_8;
        wp::vec_t<3,wp::float32> var_9;
        wp::vec_t<3,wp::float32>* var_10;
        wp::vec_t<3,wp::float32> var_11;
        wp::vec_t<3,wp::float32> var_12;
        wp::vec_t<3,wp::float32> var_13;
        wp::vec_t<3,wp::float32> var_14;
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
        //---------
        // forward
        // def update_particle_position(                                                          <L 1865>
        // tid = wp.tid()                                                                         <L 1873>
        var_0 = builtin_tid1d();
        // if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:                                  <L 1874>
        var_1 = wp::address(var_particle_flags, var_0);
        var_3 = wp::load(var_1);
        var_4 = wp::bit_and(var_3, var_2);
        var_6 = (var_4 == var_5);
        if (var_6) {
            // return                                                                             <L 1875>
            goto label0;
        }
        // qd_1 = x[tid]                                                                          <L 1877>
        var_7 = wp::address(var_x, var_0);
        var_8 = wp::load(var_7);
        var_9 = wp::copy(var_8);
        // q_0 = particle_q_0[tid]                                                                <L 1879>
        var_10 = wp::address(var_particle_q_0, var_0);
        var_11 = wp::load(var_10);
        var_12 = wp::copy(var_11);
        // q_1 = q_0 + qd_1 * dt                                                                  <L 1880>
        var_13 = wp::mul(var_9, var_dt);
        var_14 = wp::add(var_12, var_13);
        // particle_q_1[tid] = q_1                                                                <L 1882>
        // wp::array_store(var_particle_q_1, var_0, var_14);
        // particle_qd_1[tid] = qd_1                                                              <L 1883>
        // wp::array_store(var_particle_qd_1, var_0, var_9);
        //---------
        // reverse
        wp::adj_array_store(var_particle_qd_1, var_0, var_9, adj_particle_qd_1, adj_0, adj_9);
        // adj: particle_qd_1[tid] = qd_1                                                         <L 1883>
        wp::adj_array_store(var_particle_q_1, var_0, var_14, adj_particle_q_1, adj_0, adj_14);
        // adj: particle_q_1[tid] = q_1                                                           <L 1882>
        wp::adj_add(var_12, var_13, adj_12, adj_13, adj_14);
        wp::adj_mul(var_9, var_dt, adj_9, adj_dt, adj_13);
        // adj: q_1 = q_0 + qd_1 * dt                                                             <L 1880>
        wp::adj_copy(var_11, adj_10, adj_12);
        wp::adj_load(var_10, adj_10, adj_11);
        wp::adj_address(var_particle_q_0, var_0, adj_particle_q_0, adj_0, adj_10);
        // adj: q_0 = particle_q_0[tid]                                                           <L 1879>
        wp::adj_copy(var_8, adj_7, adj_9);
        wp::adj_load(var_7, adj_7, adj_8);
        wp::adj_address(var_x, var_0, adj_x, adj_0, adj_7);
        // adj: qd_1 = x[tid]                                                                     <L 1877>
        if (var_6) {
            label0:;
            // adj: return                                                                        <L 1875>
        }
        wp::adj_bit_and(var_3, var_2, adj_1, adj_2, adj_4);
        wp::adj_load(var_1, adj_1, adj_3);
        wp::adj_address(var_particle_flags, var_0, adj_particle_flags, adj_0, adj_1);
        // adj: if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:                             <L 1874>
        // adj: tid = wp.tid()                                                                    <L 1873>
        // adj: def update_particle_position(                                                     <L 1865>
        continue;
    }
}

