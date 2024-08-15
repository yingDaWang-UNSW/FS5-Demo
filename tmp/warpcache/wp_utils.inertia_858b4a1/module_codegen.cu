
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


// G:\My Drive\sourceCodes\fs5ydw-main\utils\inertia.py:19
static CUDA_CALLABLE wp::float32 triangle_inertia(
    wp::vec_t<3,wp::float32> var_p,
    wp::vec_t<3,wp::float32> var_q,
    wp::vec_t<3,wp::float32> var_r,
    wp::float32 var_density,
    wp::vec_t<3,wp::float32> var_com,
    wp::array_t<wp::float32> var_mass,
    wp::array_t<wp::mat_t<3,3,wp::float32>> var_inertia)
{
    //---------
    // primal vars
    wp::vec_t<3,wp::float32> var_0;
    wp::vec_t<3,wp::float32> var_1;
    wp::vec_t<3,wp::float32> var_2;
    const wp::int32 var_3 = 0;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    const wp::int32 var_11 = 2;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::mat_t<3,3,wp::float32> var_15;
    wp::float32 var_16;
    const wp::float32 var_17 = 6.0;
    wp::float32 var_18;
    const wp::float32 var_19 = 4.0;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    const wp::float32 var_23 = 5.0;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::vec_t<3,wp::float32> var_26;
    wp::vec_t<3,wp::float32> var_27;
    wp::vec_t<3,wp::float32> var_28;
    wp::vec_t<3,wp::float32> var_29;
    wp::vec_t<3,wp::float32> var_30;
    wp::vec_t<3,wp::float32> var_31;
    wp::vec_t<3,wp::float32> var_32;
    wp::vec_t<3,wp::float32> var_33;
    wp::vec_t<3,wp::float32> var_34;
    wp::vec_t<3,wp::float32> var_35;
    wp::vec_t<3,wp::float32> var_36;
    wp::vec_t<3,wp::float32> var_37;
    wp::vec_t<3,wp::float32> var_38;
    wp::vec_t<3,wp::float32> var_39;
    wp::vec_t<3,wp::float32> var_40;
    wp::vec_t<3,wp::float32> var_41;
    wp::vec_t<3,wp::float32> var_42;
    const wp::float32 var_43 = 1.0;
    const wp::float32 var_44 = 0.0;
    wp::mat_t<3,3,wp::float32> var_45;
    wp::float32 var_46;
    wp::mat_t<3,3,wp::float32> var_47;
    wp::mat_t<3,3,wp::float32> var_48;
    wp::mat_t<3,3,wp::float32> var_49;
    wp::float32 var_50;
    wp::mat_t<3,3,wp::float32> var_51;
    wp::mat_t<3,3,wp::float32> var_52;
    wp::mat_t<3,3,wp::float32> var_53;
    wp::mat_t<3,3,wp::float32> var_54;
    wp::float32 var_55;
    wp::mat_t<3,3,wp::float32> var_56;
    wp::mat_t<3,3,wp::float32> var_57;
    wp::mat_t<3,3,wp::float32> var_58;
    wp::mat_t<3,3,wp::float32> var_59;
    wp::float32 var_60;
    wp::mat_t<3,3,wp::float32> var_61;
    wp::mat_t<3,3,wp::float32> var_62;
    wp::mat_t<3,3,wp::float32> var_63;
    wp::mat_t<3,3,wp::float32> var_64;
    wp::float32 var_65;
    wp::mat_t<3,3,wp::float32> var_66;
    wp::mat_t<3,3,wp::float32> var_67;
    //---------
    // forward
    // def triangle_inertia(                                                                  <L 20>
    // pcom = p - com                                                                         <L 30>
    var_0 = wp::sub(var_p, var_com);
    // qcom = q - com                                                                         <L 31>
    var_1 = wp::sub(var_q, var_com);
    // rcom = r - com                                                                         <L 32>
    var_2 = wp::sub(var_r, var_com);
    // Dm = wp.mat33(pcom[0], qcom[0], rcom[0], pcom[1], qcom[1], rcom[1], pcom[2], qcom[2], rcom[2])       <L 34>
    var_4 = wp::extract(var_0, var_3);
    var_5 = wp::extract(var_1, var_3);
    var_6 = wp::extract(var_2, var_3);
    var_8 = wp::extract(var_0, var_7);
    var_9 = wp::extract(var_1, var_7);
    var_10 = wp::extract(var_2, var_7);
    var_12 = wp::extract(var_0, var_11);
    var_13 = wp::extract(var_1, var_11);
    var_14 = wp::extract(var_2, var_11);
    var_15 = wp::mat_t<3,3,wp::float32>(var_4, var_5, var_6, var_8, var_9, var_10, var_12, var_13, var_14);
    // volume = wp.determinant(Dm) / 6.0                                                      <L 36>
    var_16 = wp::determinant(var_15);
    var_18 = wp::div(var_16, var_17);
    // wp.atomic_add(mass, 0, 4.0 * density * volume)                                         <L 39>
    var_20 = wp::mul(var_19, var_density);
    var_21 = wp::mul(var_20, var_18);
    var_22 = wp::atomic_add(var_mass, var_3, var_21);
    // alpha = wp.sqrt(5.0) / 5.0                                                             <L 41>
    var_24 = wp::sqrt(var_23);
    var_25 = wp::div(var_24, var_23);
    // mid = (com + p + q + r) / 4.0                                                          <L 42>
    var_26 = wp::add(var_com, var_p);
    var_27 = wp::add(var_26, var_q);
    var_28 = wp::add(var_27, var_r);
    var_29 = wp::div(var_28, var_19);
    // off_mid = mid - com                                                                    <L 43>
    var_30 = wp::sub(var_29, var_com);
    // d0 = alpha * (p - mid) + off_mid                                                       <L 46>
    var_31 = wp::sub(var_p, var_29);
    var_32 = wp::mul(var_25, var_31);
    var_33 = wp::add(var_32, var_30);
    // d1 = alpha * (q - mid) + off_mid                                                       <L 47>
    var_34 = wp::sub(var_q, var_29);
    var_35 = wp::mul(var_25, var_34);
    var_36 = wp::add(var_35, var_30);
    // d2 = alpha * (r - mid) + off_mid                                                       <L 48>
    var_37 = wp::sub(var_r, var_29);
    var_38 = wp::mul(var_25, var_37);
    var_39 = wp::add(var_38, var_30);
    // d3 = alpha * (com - mid) + off_mid                                                     <L 49>
    var_40 = wp::sub(var_com, var_29);
    var_41 = wp::mul(var_25, var_40);
    var_42 = wp::add(var_41, var_30);
    // identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)                       <L 52>
    var_45 = wp::mat_t<3,3,wp::float32>(var_43, var_44, var_44, var_44, var_43, var_44, var_44, var_44, var_43);
    // I = wp.dot(d0, d0) * identity - wp.outer(d0, d0)                                       <L 53>
    var_46 = wp::dot(var_33, var_33);
    var_47 = wp::mul(var_46, var_45);
    var_48 = wp::outer(var_33, var_33);
    var_49 = wp::sub(var_47, var_48);
    // I += wp.dot(d1, d1) * identity - wp.outer(d1, d1)                                      <L 54>
    var_50 = wp::dot(var_36, var_36);
    var_51 = wp::mul(var_50, var_45);
    var_52 = wp::outer(var_36, var_36);
    var_53 = wp::sub(var_51, var_52);
    var_54 = wp::add(var_49, var_53);
    // I += wp.dot(d2, d2) * identity - wp.outer(d2, d2)                                      <L 55>
    var_55 = wp::dot(var_39, var_39);
    var_56 = wp::mul(var_55, var_45);
    var_57 = wp::outer(var_39, var_39);
    var_58 = wp::sub(var_56, var_57);
    var_59 = wp::add(var_54, var_58);
    // I += wp.dot(d3, d3) * identity - wp.outer(d3, d3)                                      <L 56>
    var_60 = wp::dot(var_42, var_42);
    var_61 = wp::mul(var_60, var_45);
    var_62 = wp::outer(var_42, var_42);
    var_63 = wp::sub(var_61, var_62);
    var_64 = wp::add(var_59, var_63);
    // wp.atomic_add(inertia, 0, (density * volume) * I)                                      <L 58>
    var_65 = wp::mul(var_density, var_18);
    var_66 = wp::mul(var_65, var_64);
    var_67 = wp::atomic_add(var_inertia, var_3, var_66);
    // return volume                                                                          <L 60>
    return var_18;
}


// G:\My Drive\sourceCodes\fs5ydw-main\utils\inertia.py:19
static CUDA_CALLABLE void adj_triangle_inertia(
    wp::vec_t<3,wp::float32> var_p,
    wp::vec_t<3,wp::float32> var_q,
    wp::vec_t<3,wp::float32> var_r,
    wp::float32 var_density,
    wp::vec_t<3,wp::float32> var_com,
    wp::array_t<wp::float32> var_mass,
    wp::array_t<wp::mat_t<3,3,wp::float32>> var_inertia,
    wp::vec_t<3,wp::float32> & adj_p,
    wp::vec_t<3,wp::float32> & adj_q,
    wp::vec_t<3,wp::float32> & adj_r,
    wp::float32 & adj_density,
    wp::vec_t<3,wp::float32> & adj_com,
    wp::array_t<wp::float32> & adj_mass,
    wp::array_t<wp::mat_t<3,3,wp::float32>> & adj_inertia,
    wp::float32 & adj_ret)
{
    //---------
    // primal vars
    wp::vec_t<3,wp::float32> var_0;
    wp::vec_t<3,wp::float32> var_1;
    wp::vec_t<3,wp::float32> var_2;
    const wp::int32 var_3 = 0;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    const wp::int32 var_11 = 2;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::mat_t<3,3,wp::float32> var_15;
    wp::float32 var_16;
    const wp::float32 var_17 = 6.0;
    wp::float32 var_18;
    const wp::float32 var_19 = 4.0;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    const wp::float32 var_23 = 5.0;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::vec_t<3,wp::float32> var_26;
    wp::vec_t<3,wp::float32> var_27;
    wp::vec_t<3,wp::float32> var_28;
    wp::vec_t<3,wp::float32> var_29;
    wp::vec_t<3,wp::float32> var_30;
    wp::vec_t<3,wp::float32> var_31;
    wp::vec_t<3,wp::float32> var_32;
    wp::vec_t<3,wp::float32> var_33;
    wp::vec_t<3,wp::float32> var_34;
    wp::vec_t<3,wp::float32> var_35;
    wp::vec_t<3,wp::float32> var_36;
    wp::vec_t<3,wp::float32> var_37;
    wp::vec_t<3,wp::float32> var_38;
    wp::vec_t<3,wp::float32> var_39;
    wp::vec_t<3,wp::float32> var_40;
    wp::vec_t<3,wp::float32> var_41;
    wp::vec_t<3,wp::float32> var_42;
    const wp::float32 var_43 = 1.0;
    const wp::float32 var_44 = 0.0;
    wp::mat_t<3,3,wp::float32> var_45;
    wp::float32 var_46;
    wp::mat_t<3,3,wp::float32> var_47;
    wp::mat_t<3,3,wp::float32> var_48;
    wp::mat_t<3,3,wp::float32> var_49;
    wp::float32 var_50;
    wp::mat_t<3,3,wp::float32> var_51;
    wp::mat_t<3,3,wp::float32> var_52;
    wp::mat_t<3,3,wp::float32> var_53;
    wp::mat_t<3,3,wp::float32> var_54;
    wp::float32 var_55;
    wp::mat_t<3,3,wp::float32> var_56;
    wp::mat_t<3,3,wp::float32> var_57;
    wp::mat_t<3,3,wp::float32> var_58;
    wp::mat_t<3,3,wp::float32> var_59;
    wp::float32 var_60;
    wp::mat_t<3,3,wp::float32> var_61;
    wp::mat_t<3,3,wp::float32> var_62;
    wp::mat_t<3,3,wp::float32> var_63;
    wp::mat_t<3,3,wp::float32> var_64;
    wp::float32 var_65;
    wp::mat_t<3,3,wp::float32> var_66;
    wp::mat_t<3,3,wp::float32> var_67;
    //---------
    // dual vars
    wp::vec_t<3,wp::float32> adj_0 = {};
    wp::vec_t<3,wp::float32> adj_1 = {};
    wp::vec_t<3,wp::float32> adj_2 = {};
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
    wp::mat_t<3,3,wp::float32> adj_15 = {};
    wp::float32 adj_16 = {};
    wp::float32 adj_17 = {};
    wp::float32 adj_18 = {};
    wp::float32 adj_19 = {};
    wp::float32 adj_20 = {};
    wp::float32 adj_21 = {};
    wp::float32 adj_22 = {};
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
    wp::float32 adj_43 = {};
    wp::float32 adj_44 = {};
    wp::mat_t<3,3,wp::float32> adj_45 = {};
    wp::float32 adj_46 = {};
    wp::mat_t<3,3,wp::float32> adj_47 = {};
    wp::mat_t<3,3,wp::float32> adj_48 = {};
    wp::mat_t<3,3,wp::float32> adj_49 = {};
    wp::float32 adj_50 = {};
    wp::mat_t<3,3,wp::float32> adj_51 = {};
    wp::mat_t<3,3,wp::float32> adj_52 = {};
    wp::mat_t<3,3,wp::float32> adj_53 = {};
    wp::mat_t<3,3,wp::float32> adj_54 = {};
    wp::float32 adj_55 = {};
    wp::mat_t<3,3,wp::float32> adj_56 = {};
    wp::mat_t<3,3,wp::float32> adj_57 = {};
    wp::mat_t<3,3,wp::float32> adj_58 = {};
    wp::mat_t<3,3,wp::float32> adj_59 = {};
    wp::float32 adj_60 = {};
    wp::mat_t<3,3,wp::float32> adj_61 = {};
    wp::mat_t<3,3,wp::float32> adj_62 = {};
    wp::mat_t<3,3,wp::float32> adj_63 = {};
    wp::mat_t<3,3,wp::float32> adj_64 = {};
    wp::float32 adj_65 = {};
    wp::mat_t<3,3,wp::float32> adj_66 = {};
    wp::mat_t<3,3,wp::float32> adj_67 = {};
    //---------
    // forward
    // def triangle_inertia(                                                                  <L 20>
    // pcom = p - com                                                                         <L 30>
    var_0 = wp::sub(var_p, var_com);
    // qcom = q - com                                                                         <L 31>
    var_1 = wp::sub(var_q, var_com);
    // rcom = r - com                                                                         <L 32>
    var_2 = wp::sub(var_r, var_com);
    // Dm = wp.mat33(pcom[0], qcom[0], rcom[0], pcom[1], qcom[1], rcom[1], pcom[2], qcom[2], rcom[2])       <L 34>
    var_4 = wp::extract(var_0, var_3);
    var_5 = wp::extract(var_1, var_3);
    var_6 = wp::extract(var_2, var_3);
    var_8 = wp::extract(var_0, var_7);
    var_9 = wp::extract(var_1, var_7);
    var_10 = wp::extract(var_2, var_7);
    var_12 = wp::extract(var_0, var_11);
    var_13 = wp::extract(var_1, var_11);
    var_14 = wp::extract(var_2, var_11);
    var_15 = wp::mat_t<3,3,wp::float32>(var_4, var_5, var_6, var_8, var_9, var_10, var_12, var_13, var_14);
    // volume = wp.determinant(Dm) / 6.0                                                      <L 36>
    var_16 = wp::determinant(var_15);
    var_18 = wp::div(var_16, var_17);
    // wp.atomic_add(mass, 0, 4.0 * density * volume)                                         <L 39>
    var_20 = wp::mul(var_19, var_density);
    var_21 = wp::mul(var_20, var_18);
    // var_22 = wp::atomic_add(var_mass, var_3, var_21);
    // alpha = wp.sqrt(5.0) / 5.0                                                             <L 41>
    var_24 = wp::sqrt(var_23);
    var_25 = wp::div(var_24, var_23);
    // mid = (com + p + q + r) / 4.0                                                          <L 42>
    var_26 = wp::add(var_com, var_p);
    var_27 = wp::add(var_26, var_q);
    var_28 = wp::add(var_27, var_r);
    var_29 = wp::div(var_28, var_19);
    // off_mid = mid - com                                                                    <L 43>
    var_30 = wp::sub(var_29, var_com);
    // d0 = alpha * (p - mid) + off_mid                                                       <L 46>
    var_31 = wp::sub(var_p, var_29);
    var_32 = wp::mul(var_25, var_31);
    var_33 = wp::add(var_32, var_30);
    // d1 = alpha * (q - mid) + off_mid                                                       <L 47>
    var_34 = wp::sub(var_q, var_29);
    var_35 = wp::mul(var_25, var_34);
    var_36 = wp::add(var_35, var_30);
    // d2 = alpha * (r - mid) + off_mid                                                       <L 48>
    var_37 = wp::sub(var_r, var_29);
    var_38 = wp::mul(var_25, var_37);
    var_39 = wp::add(var_38, var_30);
    // d3 = alpha * (com - mid) + off_mid                                                     <L 49>
    var_40 = wp::sub(var_com, var_29);
    var_41 = wp::mul(var_25, var_40);
    var_42 = wp::add(var_41, var_30);
    // identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)                       <L 52>
    var_45 = wp::mat_t<3,3,wp::float32>(var_43, var_44, var_44, var_44, var_43, var_44, var_44, var_44, var_43);
    // I = wp.dot(d0, d0) * identity - wp.outer(d0, d0)                                       <L 53>
    var_46 = wp::dot(var_33, var_33);
    var_47 = wp::mul(var_46, var_45);
    var_48 = wp::outer(var_33, var_33);
    var_49 = wp::sub(var_47, var_48);
    // I += wp.dot(d1, d1) * identity - wp.outer(d1, d1)                                      <L 54>
    var_50 = wp::dot(var_36, var_36);
    var_51 = wp::mul(var_50, var_45);
    var_52 = wp::outer(var_36, var_36);
    var_53 = wp::sub(var_51, var_52);
    var_54 = wp::add(var_49, var_53);
    // I += wp.dot(d2, d2) * identity - wp.outer(d2, d2)                                      <L 55>
    var_55 = wp::dot(var_39, var_39);
    var_56 = wp::mul(var_55, var_45);
    var_57 = wp::outer(var_39, var_39);
    var_58 = wp::sub(var_56, var_57);
    var_59 = wp::add(var_54, var_58);
    // I += wp.dot(d3, d3) * identity - wp.outer(d3, d3)                                      <L 56>
    var_60 = wp::dot(var_42, var_42);
    var_61 = wp::mul(var_60, var_45);
    var_62 = wp::outer(var_42, var_42);
    var_63 = wp::sub(var_61, var_62);
    var_64 = wp::add(var_59, var_63);
    // wp.atomic_add(inertia, 0, (density * volume) * I)                                      <L 58>
    var_65 = wp::mul(var_density, var_18);
    var_66 = wp::mul(var_65, var_64);
    // var_67 = wp::atomic_add(var_inertia, var_3, var_66);
    // return volume                                                                          <L 60>
    goto label0;
    //---------
    // reverse
    label0:;
    adj_18 += adj_ret;
    // adj: return volume                                                                     <L 60>
    wp::adj_atomic_add(var_inertia, var_3, var_66, adj_inertia, adj_3, adj_66, adj_67);
    wp::adj_mul(var_65, var_64, adj_65, adj_64, adj_66);
    wp::adj_mul(var_density, var_18, adj_density, adj_18, adj_65);
    // adj: wp.atomic_add(inertia, 0, (density * volume) * I)                                 <L 58>
    wp::adj_add(var_59, var_63, adj_59, adj_63, adj_64);
    wp::adj_sub(var_61, var_62, adj_61, adj_62, adj_63);
    wp::adj_outer(var_42, var_42, adj_42, adj_42, adj_62);
    wp::adj_mul(var_60, var_45, adj_60, adj_45, adj_61);
    wp::adj_dot(var_42, var_42, adj_42, adj_42, adj_60);
    // adj: I += wp.dot(d3, d3) * identity - wp.outer(d3, d3)                                 <L 56>
    wp::adj_add(var_54, var_58, adj_54, adj_58, adj_59);
    wp::adj_sub(var_56, var_57, adj_56, adj_57, adj_58);
    wp::adj_outer(var_39, var_39, adj_39, adj_39, adj_57);
    wp::adj_mul(var_55, var_45, adj_55, adj_45, adj_56);
    wp::adj_dot(var_39, var_39, adj_39, adj_39, adj_55);
    // adj: I += wp.dot(d2, d2) * identity - wp.outer(d2, d2)                                 <L 55>
    wp::adj_add(var_49, var_53, adj_49, adj_53, adj_54);
    wp::adj_sub(var_51, var_52, adj_51, adj_52, adj_53);
    wp::adj_outer(var_36, var_36, adj_36, adj_36, adj_52);
    wp::adj_mul(var_50, var_45, adj_50, adj_45, adj_51);
    wp::adj_dot(var_36, var_36, adj_36, adj_36, adj_50);
    // adj: I += wp.dot(d1, d1) * identity - wp.outer(d1, d1)                                 <L 54>
    wp::adj_sub(var_47, var_48, adj_47, adj_48, adj_49);
    wp::adj_outer(var_33, var_33, adj_33, adj_33, adj_48);
    wp::adj_mul(var_46, var_45, adj_46, adj_45, adj_47);
    wp::adj_dot(var_33, var_33, adj_33, adj_33, adj_46);
    // adj: I = wp.dot(d0, d0) * identity - wp.outer(d0, d0)                                  <L 53>
    wp::adj_mat_t(var_43, var_44, var_44, var_44, var_43, var_44, var_44, var_44, var_43, adj_43, adj_44, adj_44, adj_44, adj_43, adj_44, adj_44, adj_44, adj_43, adj_45);
    // adj: identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)                  <L 52>
    wp::adj_add(var_41, var_30, adj_41, adj_30, adj_42);
    wp::adj_mul(var_25, var_40, adj_25, adj_40, adj_41);
    wp::adj_sub(var_com, var_29, adj_com, adj_29, adj_40);
    // adj: d3 = alpha * (com - mid) + off_mid                                                <L 49>
    wp::adj_add(var_38, var_30, adj_38, adj_30, adj_39);
    wp::adj_mul(var_25, var_37, adj_25, adj_37, adj_38);
    wp::adj_sub(var_r, var_29, adj_r, adj_29, adj_37);
    // adj: d2 = alpha * (r - mid) + off_mid                                                  <L 48>
    wp::adj_add(var_35, var_30, adj_35, adj_30, adj_36);
    wp::adj_mul(var_25, var_34, adj_25, adj_34, adj_35);
    wp::adj_sub(var_q, var_29, adj_q, adj_29, adj_34);
    // adj: d1 = alpha * (q - mid) + off_mid                                                  <L 47>
    wp::adj_add(var_32, var_30, adj_32, adj_30, adj_33);
    wp::adj_mul(var_25, var_31, adj_25, adj_31, adj_32);
    wp::adj_sub(var_p, var_29, adj_p, adj_29, adj_31);
    // adj: d0 = alpha * (p - mid) + off_mid                                                  <L 46>
    wp::adj_sub(var_29, var_com, adj_29, adj_com, adj_30);
    // adj: off_mid = mid - com                                                               <L 43>
    wp::adj_div(var_28, var_19, adj_28, adj_19, adj_29);
    wp::adj_add(var_27, var_r, adj_27, adj_r, adj_28);
    wp::adj_add(var_26, var_q, adj_26, adj_q, adj_27);
    wp::adj_add(var_com, var_p, adj_com, adj_p, adj_26);
    // adj: mid = (com + p + q + r) / 4.0                                                     <L 42>
    wp::adj_div(var_24, var_23, var_25, adj_24, adj_23, adj_25);
    wp::adj_sqrt(var_23, var_24, adj_23, adj_24);
    // adj: alpha = wp.sqrt(5.0) / 5.0                                                        <L 41>
    wp::adj_atomic_add(var_mass, var_3, var_21, adj_mass, adj_3, adj_21, adj_22);
    wp::adj_mul(var_20, var_18, adj_20, adj_18, adj_21);
    wp::adj_mul(var_19, var_density, adj_19, adj_density, adj_20);
    // adj: wp.atomic_add(mass, 0, 4.0 * density * volume)                                    <L 39>
    wp::adj_div(var_16, var_17, var_18, adj_16, adj_17, adj_18);
    wp::adj_determinant(var_15, adj_15, adj_16);
    // adj: volume = wp.determinant(Dm) / 6.0                                                 <L 36>
    wp::adj_mat_t(var_4, var_5, var_6, var_8, var_9, var_10, var_12, var_13, var_14, adj_4, adj_5, adj_6, adj_8, adj_9, adj_10, adj_12, adj_13, adj_14, adj_15);
    wp::adj_extract(var_2, var_11, adj_2, adj_11, adj_14);
    wp::adj_extract(var_1, var_11, adj_1, adj_11, adj_13);
    wp::adj_extract(var_0, var_11, adj_0, adj_11, adj_12);
    wp::adj_extract(var_2, var_7, adj_2, adj_7, adj_10);
    wp::adj_extract(var_1, var_7, adj_1, adj_7, adj_9);
    wp::adj_extract(var_0, var_7, adj_0, adj_7, adj_8);
    wp::adj_extract(var_2, var_3, adj_2, adj_3, adj_6);
    wp::adj_extract(var_1, var_3, adj_1, adj_3, adj_5);
    wp::adj_extract(var_0, var_3, adj_0, adj_3, adj_4);
    // adj: Dm = wp.mat33(pcom[0], qcom[0], rcom[0], pcom[1], qcom[1], rcom[1], pcom[2], qcom[2], rcom[2])  <L 34>
    wp::adj_sub(var_r, var_com, adj_r, adj_com, adj_2);
    // adj: rcom = r - com                                                                    <L 32>
    wp::adj_sub(var_q, var_com, adj_q, adj_com, adj_1);
    // adj: qcom = q - com                                                                    <L 31>
    wp::adj_sub(var_p, var_com, adj_p, adj_com, adj_0);
    // adj: pcom = p - com                                                                    <L 30>
    // adj: def triangle_inertia(                                                             <L 20>
    return;
}



extern "C" __global__ void compute_solid_mesh_inertia_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::vec_t<3,wp::float32> var_com,
    wp::float32 var_weight,
    wp::array_t<wp::int32> var_indices,
    wp::array_t<wp::vec_t<3,wp::float32>> var_vertices,
    wp::array_t<wp::float32> var_mass,
    wp::array_t<wp::mat_t<3,3,wp::float32>> var_inertia,
    wp::array_t<wp::float32> var_volume)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        const wp::int32 var_1 = 3;
        wp::int32 var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::vec_t<3,wp::float32>* var_7;
        wp::vec_t<3,wp::float32> var_8;
        wp::vec_t<3,wp::float32> var_9;
        wp::int32 var_10;
        const wp::int32 var_11 = 1;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::vec_t<3,wp::float32>* var_15;
        wp::vec_t<3,wp::float32> var_16;
        wp::vec_t<3,wp::float32> var_17;
        wp::int32 var_18;
        const wp::int32 var_19 = 2;
        wp::int32 var_20;
        wp::int32* var_21;
        wp::int32 var_22;
        wp::vec_t<3,wp::float32>* var_23;
        wp::vec_t<3,wp::float32> var_24;
        wp::vec_t<3,wp::float32> var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        //---------
        // forward
        // def compute_solid_mesh_inertia(                                                        <L 64>
        // i = wp.tid()                                                                           <L 75>
        var_0 = builtin_tid1d();
        // p = vertices[indices[i * 3 + 0]]                                                       <L 77>
        var_2 = wp::mul(var_0, var_1);
        var_4 = wp::add(var_2, var_3);
        var_5 = wp::address(var_indices, var_4);
        var_6 = wp::load(var_5);
        var_7 = wp::address(var_vertices, var_6);
        var_8 = wp::load(var_7);
        var_9 = wp::copy(var_8);
        // q = vertices[indices[i * 3 + 1]]                                                       <L 78>
        var_10 = wp::mul(var_0, var_1);
        var_12 = wp::add(var_10, var_11);
        var_13 = wp::address(var_indices, var_12);
        var_14 = wp::load(var_13);
        var_15 = wp::address(var_vertices, var_14);
        var_16 = wp::load(var_15);
        var_17 = wp::copy(var_16);
        // r = vertices[indices[i * 3 + 2]]                                                       <L 79>
        var_18 = wp::mul(var_0, var_1);
        var_20 = wp::add(var_18, var_19);
        var_21 = wp::address(var_indices, var_20);
        var_22 = wp::load(var_21);
        var_23 = wp::address(var_vertices, var_22);
        var_24 = wp::load(var_23);
        var_25 = wp::copy(var_24);
        // vol = triangle_inertia(p, q, r, weight, com, mass, inertia)                            <L 81>
        var_26 = triangle_inertia(var_9, var_17, var_25, var_weight, var_com, var_mass, var_inertia);
        // wp.atomic_add(volume, 0, vol)                                                          <L 82>
        var_27 = wp::atomic_add(var_volume, var_3, var_26);
    }
}

extern "C" __global__ void compute_solid_mesh_inertia_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::vec_t<3,wp::float32> var_com,
    wp::float32 var_weight,
    wp::array_t<wp::int32> var_indices,
    wp::array_t<wp::vec_t<3,wp::float32>> var_vertices,
    wp::array_t<wp::float32> var_mass,
    wp::array_t<wp::mat_t<3,3,wp::float32>> var_inertia,
    wp::array_t<wp::float32> var_volume,
    wp::vec_t<3,wp::float32> adj_com,
    wp::float32 adj_weight,
    wp::array_t<wp::int32> adj_indices,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_vertices,
    wp::array_t<wp::float32> adj_mass,
    wp::array_t<wp::mat_t<3,3,wp::float32>> adj_inertia,
    wp::array_t<wp::float32> adj_volume)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        const wp::int32 var_1 = 3;
        wp::int32 var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::vec_t<3,wp::float32>* var_7;
        wp::vec_t<3,wp::float32> var_8;
        wp::vec_t<3,wp::float32> var_9;
        wp::int32 var_10;
        const wp::int32 var_11 = 1;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::vec_t<3,wp::float32>* var_15;
        wp::vec_t<3,wp::float32> var_16;
        wp::vec_t<3,wp::float32> var_17;
        wp::int32 var_18;
        const wp::int32 var_19 = 2;
        wp::int32 var_20;
        wp::int32* var_21;
        wp::int32 var_22;
        wp::vec_t<3,wp::float32>* var_23;
        wp::vec_t<3,wp::float32> var_24;
        wp::vec_t<3,wp::float32> var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::vec_t<3,wp::float32> adj_7 = {};
        wp::vec_t<3,wp::float32> adj_8 = {};
        wp::vec_t<3,wp::float32> adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::vec_t<3,wp::float32> adj_15 = {};
        wp::vec_t<3,wp::float32> adj_16 = {};
        wp::vec_t<3,wp::float32> adj_17 = {};
        wp::int32 adj_18 = {};
        wp::int32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::int32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::vec_t<3,wp::float32> adj_23 = {};
        wp::vec_t<3,wp::float32> adj_24 = {};
        wp::vec_t<3,wp::float32> adj_25 = {};
        wp::float32 adj_26 = {};
        wp::float32 adj_27 = {};
        //---------
        // forward
        // def compute_solid_mesh_inertia(                                                        <L 64>
        // i = wp.tid()                                                                           <L 75>
        var_0 = builtin_tid1d();
        // p = vertices[indices[i * 3 + 0]]                                                       <L 77>
        var_2 = wp::mul(var_0, var_1);
        var_4 = wp::add(var_2, var_3);
        var_5 = wp::address(var_indices, var_4);
        var_6 = wp::load(var_5);
        var_7 = wp::address(var_vertices, var_6);
        var_8 = wp::load(var_7);
        var_9 = wp::copy(var_8);
        // q = vertices[indices[i * 3 + 1]]                                                       <L 78>
        var_10 = wp::mul(var_0, var_1);
        var_12 = wp::add(var_10, var_11);
        var_13 = wp::address(var_indices, var_12);
        var_14 = wp::load(var_13);
        var_15 = wp::address(var_vertices, var_14);
        var_16 = wp::load(var_15);
        var_17 = wp::copy(var_16);
        // r = vertices[indices[i * 3 + 2]]                                                       <L 79>
        var_18 = wp::mul(var_0, var_1);
        var_20 = wp::add(var_18, var_19);
        var_21 = wp::address(var_indices, var_20);
        var_22 = wp::load(var_21);
        var_23 = wp::address(var_vertices, var_22);
        var_24 = wp::load(var_23);
        var_25 = wp::copy(var_24);
        // vol = triangle_inertia(p, q, r, weight, com, mass, inertia)                            <L 81>
        var_26 = triangle_inertia(var_9, var_17, var_25, var_weight, var_com, var_mass, var_inertia);
        // wp.atomic_add(volume, 0, vol)                                                          <L 82>
        // var_27 = wp::atomic_add(var_volume, var_3, var_26);
        //---------
        // reverse
        wp::adj_atomic_add(var_volume, var_3, var_26, adj_volume, adj_3, adj_26, adj_27);
        // adj: wp.atomic_add(volume, 0, vol)                                                     <L 82>
        adj_triangle_inertia(var_9, var_17, var_25, var_weight, var_com, var_mass, var_inertia, adj_9, adj_17, adj_25, adj_weight, adj_com, adj_mass, adj_inertia, adj_26);
        // adj: vol = triangle_inertia(p, q, r, weight, com, mass, inertia)                       <L 81>
        wp::adj_copy(var_24, adj_23, adj_25);
        wp::adj_load(var_23, adj_23, adj_24);
        wp::adj_address(var_vertices, var_22, adj_vertices, adj_21, adj_23);
        wp::adj_load(var_21, adj_21, adj_22);
        wp::adj_address(var_indices, var_20, adj_indices, adj_20, adj_21);
        wp::adj_add(var_18, var_19, adj_18, adj_19, adj_20);
        wp::adj_mul(var_0, var_1, adj_0, adj_1, adj_18);
        // adj: r = vertices[indices[i * 3 + 2]]                                                  <L 79>
        wp::adj_copy(var_16, adj_15, adj_17);
        wp::adj_load(var_15, adj_15, adj_16);
        wp::adj_address(var_vertices, var_14, adj_vertices, adj_13, adj_15);
        wp::adj_load(var_13, adj_13, adj_14);
        wp::adj_address(var_indices, var_12, adj_indices, adj_12, adj_13);
        wp::adj_add(var_10, var_11, adj_10, adj_11, adj_12);
        wp::adj_mul(var_0, var_1, adj_0, adj_1, adj_10);
        // adj: q = vertices[indices[i * 3 + 1]]                                                  <L 78>
        wp::adj_copy(var_8, adj_7, adj_9);
        wp::adj_load(var_7, adj_7, adj_8);
        wp::adj_address(var_vertices, var_6, adj_vertices, adj_5, adj_7);
        wp::adj_load(var_5, adj_5, adj_6);
        wp::adj_address(var_indices, var_4, adj_indices, adj_4, adj_5);
        wp::adj_add(var_2, var_3, adj_2, adj_3, adj_4);
        wp::adj_mul(var_0, var_1, adj_0, adj_1, adj_2);
        // adj: p = vertices[indices[i * 3 + 0]]                                                  <L 77>
        // adj: i = wp.tid()                                                                      <L 75>
        // adj: def compute_solid_mesh_inertia(                                                   <L 64>
        continue;
    }
}



extern "C" __global__ void compute_hollow_mesh_inertia_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::vec_t<3,wp::float32> var_com,
    wp::float32 var_density,
    wp::array_t<wp::int32> var_indices,
    wp::array_t<wp::vec_t<3,wp::float32>> var_vertices,
    wp::array_t<wp::float32> var_thickness,
    wp::array_t<wp::float32> var_mass,
    wp::array_t<wp::mat_t<3,3,wp::float32>> var_inertia,
    wp::array_t<wp::float32> var_volume)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        const wp::int32 var_1 = 3;
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
        wp::int32 var_14;
        const wp::int32 var_15 = 2;
        wp::int32 var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::vec_t<3,wp::float32>* var_20;
        wp::vec_t<3,wp::float32> var_21;
        wp::vec_t<3,wp::float32> var_22;
        wp::vec_t<3,wp::float32>* var_23;
        wp::vec_t<3,wp::float32> var_24;
        wp::vec_t<3,wp::float32> var_25;
        wp::vec_t<3,wp::float32>* var_26;
        wp::vec_t<3,wp::float32> var_27;
        wp::vec_t<3,wp::float32> var_28;
        wp::vec_t<3,wp::float32> var_29;
        wp::vec_t<3,wp::float32> var_30;
        wp::vec_t<3,wp::float32> var_31;
        wp::vec_t<3,wp::float32> var_32;
        wp::vec_t<3,wp::float32> var_33;
        wp::float32* var_34;
        wp::float32 var_35;
        wp::vec_t<3,wp::float32> var_36;
        wp::float32* var_37;
        wp::float32 var_38;
        wp::vec_t<3,wp::float32> var_39;
        wp::float32* var_40;
        wp::float32 var_41;
        wp::vec_t<3,wp::float32> var_42;
        wp::vec_t<3,wp::float32> var_43;
        wp::vec_t<3,wp::float32> var_44;
        wp::vec_t<3,wp::float32> var_45;
        wp::vec_t<3,wp::float32> var_46;
        wp::vec_t<3,wp::float32> var_47;
        wp::vec_t<3,wp::float32> var_48;
        wp::float32 var_49;
        wp::float32 var_50;
        wp::float32 var_51;
        wp::float32 var_52;
        wp::float32 var_53;
        wp::float32 var_54;
        wp::float32 var_55;
        wp::float32 var_56;
        wp::vec_t<3,wp::float32> var_57;
        wp::vec_t<3,wp::float32> var_58;
        wp::vec_t<3,wp::float32> var_59;
        wp::float32 var_60;
        const wp::float32 var_61 = 0.5;
        wp::float32 var_62;
        wp::float32* var_63;
        wp::float32* var_64;
        wp::float32 var_65;
        wp::float32 var_66;
        wp::float32 var_67;
        wp::float32* var_68;
        wp::float32 var_69;
        wp::float32 var_70;
        wp::float32 var_71;
        const wp::float32 var_72 = 3.0;
        wp::float32 var_73;
        wp::float32 var_74;
        //---------
        // forward
        // def compute_hollow_mesh_inertia(                                                       <L 86>
        // tid = wp.tid()                                                                         <L 98>
        var_0 = builtin_tid1d();
        // i = indices[tid * 3 + 0]                                                               <L 99>
        var_2 = wp::mul(var_0, var_1);
        var_4 = wp::add(var_2, var_3);
        var_5 = wp::address(var_indices, var_4);
        var_6 = wp::load(var_5);
        var_7 = wp::copy(var_6);
        // j = indices[tid * 3 + 1]                                                               <L 100>
        var_8 = wp::mul(var_0, var_1);
        var_10 = wp::add(var_8, var_9);
        var_11 = wp::address(var_indices, var_10);
        var_12 = wp::load(var_11);
        var_13 = wp::copy(var_12);
        // k = indices[tid * 3 + 2]                                                               <L 101>
        var_14 = wp::mul(var_0, var_1);
        var_16 = wp::add(var_14, var_15);
        var_17 = wp::address(var_indices, var_16);
        var_18 = wp::load(var_17);
        var_19 = wp::copy(var_18);
        // vi = vertices[i]                                                                       <L 103>
        var_20 = wp::address(var_vertices, var_7);
        var_21 = wp::load(var_20);
        var_22 = wp::copy(var_21);
        // vj = vertices[j]                                                                       <L 104>
        var_23 = wp::address(var_vertices, var_13);
        var_24 = wp::load(var_23);
        var_25 = wp::copy(var_24);
        // vk = vertices[k]                                                                       <L 105>
        var_26 = wp::address(var_vertices, var_19);
        var_27 = wp::load(var_26);
        var_28 = wp::copy(var_27);
        // normal = -wp.normalize(wp.cross(vj - vi, vk - vi))                                     <L 107>
        var_29 = wp::sub(var_25, var_22);
        var_30 = wp::sub(var_28, var_22);
        var_31 = wp::cross(var_29, var_30);
        var_32 = wp::normalize(var_31);
        var_33 = wp::neg(var_32);
        // ti = normal * thickness[i]                                                             <L 108>
        var_34 = wp::address(var_thickness, var_7);
        var_35 = wp::load(var_34);
        var_36 = wp::mul(var_33, var_35);
        // tj = normal * thickness[j]                                                             <L 109>
        var_37 = wp::address(var_thickness, var_13);
        var_38 = wp::load(var_37);
        var_39 = wp::mul(var_33, var_38);
        // tk = normal * thickness[k]                                                             <L 110>
        var_40 = wp::address(var_thickness, var_19);
        var_41 = wp::load(var_40);
        var_42 = wp::mul(var_33, var_41);
        // vi0 = vi - ti                                                                          <L 113>
        var_43 = wp::sub(var_22, var_36);
        // vi1 = vi + ti                                                                          <L 114>
        var_44 = wp::add(var_22, var_36);
        // vj0 = vj - tj                                                                          <L 115>
        var_45 = wp::sub(var_25, var_39);
        // vj1 = vj + tj                                                                          <L 116>
        var_46 = wp::add(var_25, var_39);
        // vk0 = vk - tk                                                                          <L 117>
        var_47 = wp::sub(var_28, var_42);
        // vk1 = vk + tk                                                                          <L 118>
        var_48 = wp::add(var_28, var_42);
        // triangle_inertia(vi0, vj0, vk0, density, com, mass, inertia)                           <L 120>
        var_49 = triangle_inertia(var_43, var_45, var_47, var_density, var_com, var_mass, var_inertia);
        // triangle_inertia(vj0, vk1, vk0, density, com, mass, inertia)                           <L 121>
        var_50 = triangle_inertia(var_45, var_48, var_47, var_density, var_com, var_mass, var_inertia);
        // triangle_inertia(vj0, vj1, vk1, density, com, mass, inertia)                           <L 122>
        var_51 = triangle_inertia(var_45, var_46, var_48, var_density, var_com, var_mass, var_inertia);
        // triangle_inertia(vj0, vi1, vj1, density, com, mass, inertia)                           <L 123>
        var_52 = triangle_inertia(var_45, var_44, var_46, var_density, var_com, var_mass, var_inertia);
        // triangle_inertia(vj0, vi0, vi1, density, com, mass, inertia)                           <L 124>
        var_53 = triangle_inertia(var_45, var_43, var_44, var_density, var_com, var_mass, var_inertia);
        // triangle_inertia(vj1, vi1, vk1, density, com, mass, inertia)                           <L 125>
        var_54 = triangle_inertia(var_46, var_44, var_48, var_density, var_com, var_mass, var_inertia);
        // triangle_inertia(vi1, vi0, vk0, density, com, mass, inertia)                           <L 126>
        var_55 = triangle_inertia(var_44, var_43, var_47, var_density, var_com, var_mass, var_inertia);
        // triangle_inertia(vi1, vk0, vk1, density, com, mass, inertia)                           <L 127>
        var_56 = triangle_inertia(var_44, var_47, var_48, var_density, var_com, var_mass, var_inertia);
        // a = wp.length(wp.cross(vj - vi, vk - vi)) * 0.5                                        <L 130>
        var_57 = wp::sub(var_25, var_22);
        var_58 = wp::sub(var_28, var_22);
        var_59 = wp::cross(var_57, var_58);
        var_60 = wp::length(var_59);
        var_62 = wp::mul(var_60, var_61);
        // vol = a * (thickness[i] + thickness[j] + thickness[k]) / 3.0                           <L 131>
        var_63 = wp::address(var_thickness, var_7);
        var_64 = wp::address(var_thickness, var_13);
        var_65 = wp::load(var_63);
        var_66 = wp::load(var_64);
        var_67 = wp::add(var_65, var_66);
        var_68 = wp::address(var_thickness, var_19);
        var_69 = wp::load(var_68);
        var_70 = wp::add(var_67, var_69);
        var_71 = wp::mul(var_62, var_70);
        var_73 = wp::div(var_71, var_72);
        // wp.atomic_add(volume, 0, vol)                                                          <L 132>
        var_74 = wp::atomic_add(var_volume, var_3, var_73);
    }
}

extern "C" __global__ void compute_hollow_mesh_inertia_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::vec_t<3,wp::float32> var_com,
    wp::float32 var_density,
    wp::array_t<wp::int32> var_indices,
    wp::array_t<wp::vec_t<3,wp::float32>> var_vertices,
    wp::array_t<wp::float32> var_thickness,
    wp::array_t<wp::float32> var_mass,
    wp::array_t<wp::mat_t<3,3,wp::float32>> var_inertia,
    wp::array_t<wp::float32> var_volume,
    wp::vec_t<3,wp::float32> adj_com,
    wp::float32 adj_density,
    wp::array_t<wp::int32> adj_indices,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_vertices,
    wp::array_t<wp::float32> adj_thickness,
    wp::array_t<wp::float32> adj_mass,
    wp::array_t<wp::mat_t<3,3,wp::float32>> adj_inertia,
    wp::array_t<wp::float32> adj_volume)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        const wp::int32 var_1 = 3;
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
        wp::int32 var_14;
        const wp::int32 var_15 = 2;
        wp::int32 var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::vec_t<3,wp::float32>* var_20;
        wp::vec_t<3,wp::float32> var_21;
        wp::vec_t<3,wp::float32> var_22;
        wp::vec_t<3,wp::float32>* var_23;
        wp::vec_t<3,wp::float32> var_24;
        wp::vec_t<3,wp::float32> var_25;
        wp::vec_t<3,wp::float32>* var_26;
        wp::vec_t<3,wp::float32> var_27;
        wp::vec_t<3,wp::float32> var_28;
        wp::vec_t<3,wp::float32> var_29;
        wp::vec_t<3,wp::float32> var_30;
        wp::vec_t<3,wp::float32> var_31;
        wp::vec_t<3,wp::float32> var_32;
        wp::vec_t<3,wp::float32> var_33;
        wp::float32* var_34;
        wp::float32 var_35;
        wp::vec_t<3,wp::float32> var_36;
        wp::float32* var_37;
        wp::float32 var_38;
        wp::vec_t<3,wp::float32> var_39;
        wp::float32* var_40;
        wp::float32 var_41;
        wp::vec_t<3,wp::float32> var_42;
        wp::vec_t<3,wp::float32> var_43;
        wp::vec_t<3,wp::float32> var_44;
        wp::vec_t<3,wp::float32> var_45;
        wp::vec_t<3,wp::float32> var_46;
        wp::vec_t<3,wp::float32> var_47;
        wp::vec_t<3,wp::float32> var_48;
        wp::float32 var_49;
        wp::float32 var_50;
        wp::float32 var_51;
        wp::float32 var_52;
        wp::float32 var_53;
        wp::float32 var_54;
        wp::float32 var_55;
        wp::float32 var_56;
        wp::vec_t<3,wp::float32> var_57;
        wp::vec_t<3,wp::float32> var_58;
        wp::vec_t<3,wp::float32> var_59;
        wp::float32 var_60;
        const wp::float32 var_61 = 0.5;
        wp::float32 var_62;
        wp::float32* var_63;
        wp::float32* var_64;
        wp::float32 var_65;
        wp::float32 var_66;
        wp::float32 var_67;
        wp::float32* var_68;
        wp::float32 var_69;
        wp::float32 var_70;
        wp::float32 var_71;
        const wp::float32 var_72 = 3.0;
        wp::float32 var_73;
        wp::float32 var_74;
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
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        wp::int32 adj_19 = {};
        wp::vec_t<3,wp::float32> adj_20 = {};
        wp::vec_t<3,wp::float32> adj_21 = {};
        wp::vec_t<3,wp::float32> adj_22 = {};
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
        wp::float32 adj_34 = {};
        wp::float32 adj_35 = {};
        wp::vec_t<3,wp::float32> adj_36 = {};
        wp::float32 adj_37 = {};
        wp::float32 adj_38 = {};
        wp::vec_t<3,wp::float32> adj_39 = {};
        wp::float32 adj_40 = {};
        wp::float32 adj_41 = {};
        wp::vec_t<3,wp::float32> adj_42 = {};
        wp::vec_t<3,wp::float32> adj_43 = {};
        wp::vec_t<3,wp::float32> adj_44 = {};
        wp::vec_t<3,wp::float32> adj_45 = {};
        wp::vec_t<3,wp::float32> adj_46 = {};
        wp::vec_t<3,wp::float32> adj_47 = {};
        wp::vec_t<3,wp::float32> adj_48 = {};
        wp::float32 adj_49 = {};
        wp::float32 adj_50 = {};
        wp::float32 adj_51 = {};
        wp::float32 adj_52 = {};
        wp::float32 adj_53 = {};
        wp::float32 adj_54 = {};
        wp::float32 adj_55 = {};
        wp::float32 adj_56 = {};
        wp::vec_t<3,wp::float32> adj_57 = {};
        wp::vec_t<3,wp::float32> adj_58 = {};
        wp::vec_t<3,wp::float32> adj_59 = {};
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
        //---------
        // forward
        // def compute_hollow_mesh_inertia(                                                       <L 86>
        // tid = wp.tid()                                                                         <L 98>
        var_0 = builtin_tid1d();
        // i = indices[tid * 3 + 0]                                                               <L 99>
        var_2 = wp::mul(var_0, var_1);
        var_4 = wp::add(var_2, var_3);
        var_5 = wp::address(var_indices, var_4);
        var_6 = wp::load(var_5);
        var_7 = wp::copy(var_6);
        // j = indices[tid * 3 + 1]                                                               <L 100>
        var_8 = wp::mul(var_0, var_1);
        var_10 = wp::add(var_8, var_9);
        var_11 = wp::address(var_indices, var_10);
        var_12 = wp::load(var_11);
        var_13 = wp::copy(var_12);
        // k = indices[tid * 3 + 2]                                                               <L 101>
        var_14 = wp::mul(var_0, var_1);
        var_16 = wp::add(var_14, var_15);
        var_17 = wp::address(var_indices, var_16);
        var_18 = wp::load(var_17);
        var_19 = wp::copy(var_18);
        // vi = vertices[i]                                                                       <L 103>
        var_20 = wp::address(var_vertices, var_7);
        var_21 = wp::load(var_20);
        var_22 = wp::copy(var_21);
        // vj = vertices[j]                                                                       <L 104>
        var_23 = wp::address(var_vertices, var_13);
        var_24 = wp::load(var_23);
        var_25 = wp::copy(var_24);
        // vk = vertices[k]                                                                       <L 105>
        var_26 = wp::address(var_vertices, var_19);
        var_27 = wp::load(var_26);
        var_28 = wp::copy(var_27);
        // normal = -wp.normalize(wp.cross(vj - vi, vk - vi))                                     <L 107>
        var_29 = wp::sub(var_25, var_22);
        var_30 = wp::sub(var_28, var_22);
        var_31 = wp::cross(var_29, var_30);
        var_32 = wp::normalize(var_31);
        var_33 = wp::neg(var_32);
        // ti = normal * thickness[i]                                                             <L 108>
        var_34 = wp::address(var_thickness, var_7);
        var_35 = wp::load(var_34);
        var_36 = wp::mul(var_33, var_35);
        // tj = normal * thickness[j]                                                             <L 109>
        var_37 = wp::address(var_thickness, var_13);
        var_38 = wp::load(var_37);
        var_39 = wp::mul(var_33, var_38);
        // tk = normal * thickness[k]                                                             <L 110>
        var_40 = wp::address(var_thickness, var_19);
        var_41 = wp::load(var_40);
        var_42 = wp::mul(var_33, var_41);
        // vi0 = vi - ti                                                                          <L 113>
        var_43 = wp::sub(var_22, var_36);
        // vi1 = vi + ti                                                                          <L 114>
        var_44 = wp::add(var_22, var_36);
        // vj0 = vj - tj                                                                          <L 115>
        var_45 = wp::sub(var_25, var_39);
        // vj1 = vj + tj                                                                          <L 116>
        var_46 = wp::add(var_25, var_39);
        // vk0 = vk - tk                                                                          <L 117>
        var_47 = wp::sub(var_28, var_42);
        // vk1 = vk + tk                                                                          <L 118>
        var_48 = wp::add(var_28, var_42);
        // triangle_inertia(vi0, vj0, vk0, density, com, mass, inertia)                           <L 120>
        var_49 = triangle_inertia(var_43, var_45, var_47, var_density, var_com, var_mass, var_inertia);
        // triangle_inertia(vj0, vk1, vk0, density, com, mass, inertia)                           <L 121>
        var_50 = triangle_inertia(var_45, var_48, var_47, var_density, var_com, var_mass, var_inertia);
        // triangle_inertia(vj0, vj1, vk1, density, com, mass, inertia)                           <L 122>
        var_51 = triangle_inertia(var_45, var_46, var_48, var_density, var_com, var_mass, var_inertia);
        // triangle_inertia(vj0, vi1, vj1, density, com, mass, inertia)                           <L 123>
        var_52 = triangle_inertia(var_45, var_44, var_46, var_density, var_com, var_mass, var_inertia);
        // triangle_inertia(vj0, vi0, vi1, density, com, mass, inertia)                           <L 124>
        var_53 = triangle_inertia(var_45, var_43, var_44, var_density, var_com, var_mass, var_inertia);
        // triangle_inertia(vj1, vi1, vk1, density, com, mass, inertia)                           <L 125>
        var_54 = triangle_inertia(var_46, var_44, var_48, var_density, var_com, var_mass, var_inertia);
        // triangle_inertia(vi1, vi0, vk0, density, com, mass, inertia)                           <L 126>
        var_55 = triangle_inertia(var_44, var_43, var_47, var_density, var_com, var_mass, var_inertia);
        // triangle_inertia(vi1, vk0, vk1, density, com, mass, inertia)                           <L 127>
        var_56 = triangle_inertia(var_44, var_47, var_48, var_density, var_com, var_mass, var_inertia);
        // a = wp.length(wp.cross(vj - vi, vk - vi)) * 0.5                                        <L 130>
        var_57 = wp::sub(var_25, var_22);
        var_58 = wp::sub(var_28, var_22);
        var_59 = wp::cross(var_57, var_58);
        var_60 = wp::length(var_59);
        var_62 = wp::mul(var_60, var_61);
        // vol = a * (thickness[i] + thickness[j] + thickness[k]) / 3.0                           <L 131>
        var_63 = wp::address(var_thickness, var_7);
        var_64 = wp::address(var_thickness, var_13);
        var_65 = wp::load(var_63);
        var_66 = wp::load(var_64);
        var_67 = wp::add(var_65, var_66);
        var_68 = wp::address(var_thickness, var_19);
        var_69 = wp::load(var_68);
        var_70 = wp::add(var_67, var_69);
        var_71 = wp::mul(var_62, var_70);
        var_73 = wp::div(var_71, var_72);
        // wp.atomic_add(volume, 0, vol)                                                          <L 132>
        // var_74 = wp::atomic_add(var_volume, var_3, var_73);
        //---------
        // reverse
        wp::adj_atomic_add(var_volume, var_3, var_73, adj_volume, adj_3, adj_73, adj_74);
        // adj: wp.atomic_add(volume, 0, vol)                                                     <L 132>
        wp::adj_div(var_71, var_72, var_73, adj_71, adj_72, adj_73);
        wp::adj_mul(var_62, var_70, adj_62, adj_70, adj_71);
        wp::adj_add(var_67, var_69, adj_67, adj_68, adj_70);
        wp::adj_load(var_68, adj_68, adj_69);
        wp::adj_address(var_thickness, var_19, adj_thickness, adj_19, adj_68);
        wp::adj_add(var_65, var_66, adj_63, adj_64, adj_67);
        wp::adj_load(var_64, adj_64, adj_66);
        wp::adj_load(var_63, adj_63, adj_65);
        wp::adj_address(var_thickness, var_13, adj_thickness, adj_13, adj_64);
        wp::adj_address(var_thickness, var_7, adj_thickness, adj_7, adj_63);
        // adj: vol = a * (thickness[i] + thickness[j] + thickness[k]) / 3.0                      <L 131>
        wp::adj_mul(var_60, var_61, adj_60, adj_61, adj_62);
        wp::adj_length(var_59, var_60, adj_59, adj_60);
        wp::adj_cross(var_57, var_58, adj_57, adj_58, adj_59);
        wp::adj_sub(var_28, var_22, adj_28, adj_22, adj_58);
        wp::adj_sub(var_25, var_22, adj_25, adj_22, adj_57);
        // adj: a = wp.length(wp.cross(vj - vi, vk - vi)) * 0.5                                   <L 130>
        adj_triangle_inertia(var_44, var_47, var_48, var_density, var_com, var_mass, var_inertia, adj_44, adj_47, adj_48, adj_density, adj_com, adj_mass, adj_inertia, adj_56);
        // adj: triangle_inertia(vi1, vk0, vk1, density, com, mass, inertia)                      <L 127>
        adj_triangle_inertia(var_44, var_43, var_47, var_density, var_com, var_mass, var_inertia, adj_44, adj_43, adj_47, adj_density, adj_com, adj_mass, adj_inertia, adj_55);
        // adj: triangle_inertia(vi1, vi0, vk0, density, com, mass, inertia)                      <L 126>
        adj_triangle_inertia(var_46, var_44, var_48, var_density, var_com, var_mass, var_inertia, adj_46, adj_44, adj_48, adj_density, adj_com, adj_mass, adj_inertia, adj_54);
        // adj: triangle_inertia(vj1, vi1, vk1, density, com, mass, inertia)                      <L 125>
        adj_triangle_inertia(var_45, var_43, var_44, var_density, var_com, var_mass, var_inertia, adj_45, adj_43, adj_44, adj_density, adj_com, adj_mass, adj_inertia, adj_53);
        // adj: triangle_inertia(vj0, vi0, vi1, density, com, mass, inertia)                      <L 124>
        adj_triangle_inertia(var_45, var_44, var_46, var_density, var_com, var_mass, var_inertia, adj_45, adj_44, adj_46, adj_density, adj_com, adj_mass, adj_inertia, adj_52);
        // adj: triangle_inertia(vj0, vi1, vj1, density, com, mass, inertia)                      <L 123>
        adj_triangle_inertia(var_45, var_46, var_48, var_density, var_com, var_mass, var_inertia, adj_45, adj_46, adj_48, adj_density, adj_com, adj_mass, adj_inertia, adj_51);
        // adj: triangle_inertia(vj0, vj1, vk1, density, com, mass, inertia)                      <L 122>
        adj_triangle_inertia(var_45, var_48, var_47, var_density, var_com, var_mass, var_inertia, adj_45, adj_48, adj_47, adj_density, adj_com, adj_mass, adj_inertia, adj_50);
        // adj: triangle_inertia(vj0, vk1, vk0, density, com, mass, inertia)                      <L 121>
        adj_triangle_inertia(var_43, var_45, var_47, var_density, var_com, var_mass, var_inertia, adj_43, adj_45, adj_47, adj_density, adj_com, adj_mass, adj_inertia, adj_49);
        // adj: triangle_inertia(vi0, vj0, vk0, density, com, mass, inertia)                      <L 120>
        wp::adj_add(var_28, var_42, adj_28, adj_42, adj_48);
        // adj: vk1 = vk + tk                                                                     <L 118>
        wp::adj_sub(var_28, var_42, adj_28, adj_42, adj_47);
        // adj: vk0 = vk - tk                                                                     <L 117>
        wp::adj_add(var_25, var_39, adj_25, adj_39, adj_46);
        // adj: vj1 = vj + tj                                                                     <L 116>
        wp::adj_sub(var_25, var_39, adj_25, adj_39, adj_45);
        // adj: vj0 = vj - tj                                                                     <L 115>
        wp::adj_add(var_22, var_36, adj_22, adj_36, adj_44);
        // adj: vi1 = vi + ti                                                                     <L 114>
        wp::adj_sub(var_22, var_36, adj_22, adj_36, adj_43);
        // adj: vi0 = vi - ti                                                                     <L 113>
        wp::adj_mul(var_33, var_41, adj_33, adj_40, adj_42);
        wp::adj_load(var_40, adj_40, adj_41);
        wp::adj_address(var_thickness, var_19, adj_thickness, adj_19, adj_40);
        // adj: tk = normal * thickness[k]                                                        <L 110>
        wp::adj_mul(var_33, var_38, adj_33, adj_37, adj_39);
        wp::adj_load(var_37, adj_37, adj_38);
        wp::adj_address(var_thickness, var_13, adj_thickness, adj_13, adj_37);
        // adj: tj = normal * thickness[j]                                                        <L 109>
        wp::adj_mul(var_33, var_35, adj_33, adj_34, adj_36);
        wp::adj_load(var_34, adj_34, adj_35);
        wp::adj_address(var_thickness, var_7, adj_thickness, adj_7, adj_34);
        // adj: ti = normal * thickness[i]                                                        <L 108>
        wp::adj_neg(var_32, adj_32, adj_33);
        wp::adj_normalize(var_31, var_32, adj_31, adj_32);
        wp::adj_cross(var_29, var_30, adj_29, adj_30, adj_31);
        wp::adj_sub(var_28, var_22, adj_28, adj_22, adj_30);
        wp::adj_sub(var_25, var_22, adj_25, adj_22, adj_29);
        // adj: normal = -wp.normalize(wp.cross(vj - vi, vk - vi))                                <L 107>
        wp::adj_copy(var_27, adj_26, adj_28);
        wp::adj_load(var_26, adj_26, adj_27);
        wp::adj_address(var_vertices, var_19, adj_vertices, adj_19, adj_26);
        // adj: vk = vertices[k]                                                                  <L 105>
        wp::adj_copy(var_24, adj_23, adj_25);
        wp::adj_load(var_23, adj_23, adj_24);
        wp::adj_address(var_vertices, var_13, adj_vertices, adj_13, adj_23);
        // adj: vj = vertices[j]                                                                  <L 104>
        wp::adj_copy(var_21, adj_20, adj_22);
        wp::adj_load(var_20, adj_20, adj_21);
        wp::adj_address(var_vertices, var_7, adj_vertices, adj_7, adj_20);
        // adj: vi = vertices[i]                                                                  <L 103>
        wp::adj_copy(var_18, adj_17, adj_19);
        wp::adj_load(var_17, adj_17, adj_18);
        wp::adj_address(var_indices, var_16, adj_indices, adj_16, adj_17);
        wp::adj_add(var_14, var_15, adj_14, adj_15, adj_16);
        wp::adj_mul(var_0, var_1, adj_0, adj_1, adj_14);
        // adj: k = indices[tid * 3 + 2]                                                          <L 101>
        wp::adj_copy(var_12, adj_11, adj_13);
        wp::adj_load(var_11, adj_11, adj_12);
        wp::adj_address(var_indices, var_10, adj_indices, adj_10, adj_11);
        wp::adj_add(var_8, var_9, adj_8, adj_9, adj_10);
        wp::adj_mul(var_0, var_1, adj_0, adj_1, adj_8);
        // adj: j = indices[tid * 3 + 1]                                                          <L 100>
        wp::adj_copy(var_6, adj_5, adj_7);
        wp::adj_load(var_5, adj_5, adj_6);
        wp::adj_address(var_indices, var_4, adj_indices, adj_4, adj_5);
        wp::adj_add(var_2, var_3, adj_2, adj_3, adj_4);
        wp::adj_mul(var_0, var_1, adj_0, adj_1, adj_2);
        // adj: i = indices[tid * 3 + 0]                                                          <L 99>
        // adj: tid = wp.tid()                                                                    <L 98>
        // adj: def compute_hollow_mesh_inertia(                                                  <L 86>
        continue;
    }
}

