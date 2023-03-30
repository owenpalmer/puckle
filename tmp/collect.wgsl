// --------------------------------
// @module: Definitions
// --------------------------------
let PI: f32 = 3.1415927;
fn interpolate_1_1(x: f32, x0: f32, x1: f32, y0: f32, y1: f32) -> f32 {
            let p = (x - x0) / (x1 - x0);
            return mix(y0, y1, f32(p));
          }
fn interpolate_1_3(x: f32, x0: f32, x1: f32, y0: vec3<f32>, y1: vec3<f32>) -> vec3<f32> {
            let p = (x - x0) / (x1 - x0);
            return mix(y0, y1, vec3<f32>(p));
          }
fn interpolate_2_2(x: vec2<f32>, x0: vec2<f32>, x1: vec2<f32>, y0: vec2<f32>, y1: vec2<f32>) -> vec2<f32> {
            let p = (x - x0) / (x1 - x0);
            return mix(y0, y1, vec2<f32>(p));
          }
fn interpolate_3_3(x: vec3<f32>, x0: vec3<f32>, x1: vec3<f32>, y0: vec3<f32>, y1: vec3<f32>) -> vec3<f32> {
            let p = (x - x0) / (x1 - x0);
            return mix(y0, y1, vec3<f32>(p));
          }
fn interpolate_4_4(x: vec4<f32>, x0: vec4<f32>, x1: vec4<f32>, y0: vec4<f32>, y1: vec4<f32>) -> vec4<f32> {
            let p = (x - x0) / (x1 - x0);
            return mix(y0, y1, vec4<f32>(p));
          }
fn interpolate_clamped_1_1(x: f32, x0: f32, x1: f32, y0: f32, y1: f32) -> f32 {
            let p = clamp((x - x0) / (x1 - x0), f32(0.), f32(1.));
            return mix(y0, y1, f32(p));
          }
fn interpolate_clamped_1_2(x: f32, x0: f32, x1: f32, y0: vec2<f32>, y1: vec2<f32>) -> vec2<f32> {
            let p = clamp((x - x0) / (x1 - x0), f32(0.), f32(1.));
            return mix(y0, y1, vec2<f32>(p));
          }
fn interpolate_clamped_1_3(x: f32, x0: f32, x1: f32, y0: vec3<f32>, y1: vec3<f32>) -> vec3<f32> {
            let p = clamp((x - x0) / (x1 - x0), f32(0.), f32(1.));
            return mix(y0, y1, vec3<f32>(p));
          }
fn interpolate_clamped_2_2(x: vec2<f32>, x0: vec2<f32>, x1: vec2<f32>, y0: vec2<f32>, y1: vec2<f32>) -> vec2<f32> {
            let p = clamp((x - x0) / (x1 - x0), vec2<f32>(0.), vec2<f32>(1.));
            return mix(y0, y1, vec2<f32>(p));
          }
fn interpolate_clamped_3_3(x: vec3<f32>, x0: vec3<f32>, x1: vec3<f32>, y0: vec3<f32>, y1: vec3<f32>) -> vec3<f32> {
            let p = clamp((x - x0) / (x1 - x0), vec3<f32>(0.), vec3<f32>(1.));
            return mix(y0, y1, vec3<f32>(p));
          }
fn interpolate_clamped_4_4(x: vec4<f32>, x0: vec4<f32>, x1: vec4<f32>, y0: vec4<f32>, y1: vec4<f32>) -> vec4<f32> {
            let p = clamp((x - x0) / (x1 - x0), vec4<f32>(0.), vec4<f32>(1.));
            return mix(y0, y1, vec4<f32>(p));
          }
// From https://github.com/glslify/glsl-inverse/blob/master/index.glsl
fn inverse(m: mat4x4<f32>) -> mat4x4<f32> {
    let x = m[0];
    let y = m[1];
    let z = m[2];
    let w = m[3];

    let a00 = x.x; let a01 = x.y; let a02 = x.z; let a03 = x.w;
    let a10 = y.x; let a11 = y.y; let a12 = y.z; let a13 = y.w;
    let a20 = z.x; let a21 = z.y; let a22 = z.z; let a23 = z.w;
    let a30 = w.x; let a31 = w.y; let a32 = w.z; let a33 = w.w;

    let b00 = a00 * a11 - a01 * a10;
    let b01 = a00 * a12 - a02 * a10;
    let b02 = a00 * a13 - a03 * a10;
    let b03 = a01 * a12 - a02 * a11;
    let b04 = a01 * a13 - a03 * a11;
    let b05 = a02 * a13 - a03 * a12;
    let b06 = a20 * a31 - a21 * a30;
    let b07 = a20 * a32 - a22 * a30;
    let b08 = a20 * a33 - a23 * a30;
    let b09 = a21 * a32 - a22 * a31;
    let b10 = a21 * a33 - a23 * a31;
    let b11 = a22 * a33 - a23 * a32;

    let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;

    return mat4x4<f32>(
        vec4<f32>(
            a11 * b11 - a12 * b10 + a13 * b09,
            a02 * b10 - a01 * b11 - a03 * b09,
            a31 * b05 - a32 * b04 + a33 * b03,
            a22 * b04 - a21 * b05 - a23 * b03,
        ),
        vec4<f32>(
            a12 * b08 - a10 * b11 - a13 * b07,
            a00 * b11 - a02 * b08 + a03 * b07,
            a32 * b02 - a30 * b05 - a33 * b01,
            a20 * b05 - a22 * b02 + a23 * b01,
        ),
        vec4<f32>(
            a10 * b10 - a11 * b08 + a13 * b06,
            a01 * b08 - a00 * b10 - a03 * b06,
            a30 * b04 - a31 * b02 + a33 * b00,
            a21 * b02 - a20 * b04 - a23 * b00,
        ),
        vec4<f32>(
            a11 * b07 - a10 * b09 - a12 * b06,
            a00 * b09 - a01 * b07 + a02 * b06,
            a31 * b01 - a30 * b03 - a32 * b00,
            a20 * b03 - a21 * b01 + a22 * b00
        )
    ) * (1. / det);
}

fn f32_to_color(v: f32) -> vec3<f32> {
    if (v == 0.) {
        return vec3<f32>(1., 1., 1.);
    } else if (v <= 1.) {
        return vec3<f32>(v, 0., 0.);
    } else if (v <= 2.) {
        return vec3<f32>(0., v - 1., 0.);
    } else if (v <= 3.) {
        return vec3<f32>(0., 0., v - 2.);
    } else if (v <= 4.) {
        return vec3<f32>(v - 3., 0., v - 3.);
    } else if (v <= 5.) {
        return vec3<f32>(0., v - 4., v - 4.);
    } else if (v <= 6.) {
        return vec3<f32>(v - 5., v - 5., 0.);
    } else if (v <= 7.) {
        return vec3<f32>(v - 6., v - 6., v - 6.);
    } else {
        return vec3<f32>(0.5, 0.5, 0.5);
    }
}

fn u32_to_color(v: u32) -> vec3<f32> {
    if (v == 0u) {
        return vec3<f32>(1., 1., 1.);
    } else if (v == 1u) {
        return vec3<f32>(1., 0., 0.);
    } else if (v == 2u) {
        return vec3<f32>(0., 1., 0.);
    } else if (v == 3u) {
        return vec3<f32>(0., 0., 1.);
    } else if (v == 4u) {
        return vec3<f32>(1., 0., 1.);
    } else if (v == 5u) {
        return vec3<f32>(0., 1., 1.);
    } else if (v == 6u) {
        return vec3<f32>(1., 1., 0.);
    } else {
        return vec3<f32>(0.5, 0.5, 0.5);
    }
}

fn from_linear_to_srgb(linear_rgb: vec3<f32>) -> vec3<f32> {
    return 1.055*pow(linear_rgb, vec3<f32>(1.0 / 2.4) ) - 0.055;
}
fn from_srgb_to_linear(srgb: vec3<f32>) -> vec3<f32> {
    return pow((srgb + vec3<f32>(0.055))/vec3<f32>(1.055), vec3<f32>(2.4));
}

struct F32Buffer { data: array<f32>, };
struct U32Buffer { data: array<u32>, };
struct I32Buffer { data: array<i32>, };
struct Vec2Buffer { data: array<vec2<f32>>, };
struct Vec3Buffer { data: array<vec3<f32>>, }; // Note: this is stride 16 so needs to be fed Vec4s
struct Vec4Buffer { data: array<vec4<f32>>, };
struct UVec2Buffer { data: array<vec2<u32>>, };
struct UVec3Buffer { data: array<vec3<u32>>, }; // Note: this is stride 16 so needs to be fed UVec4s
struct UVec4Buffer { data: array<vec4<u32>>, };
struct Mat4x4Buffer { data: array<mat4x4<f32>>, };

fn quat_from_mat3(mat3: mat3x3<f32>) -> vec4<f32> {
    // From: https://github.com/bitshifter/glam-rs/blob/main/src/f32/scalar/quat.rs#L182
    let m00 = mat3[0][0];
    let m01 = mat3[0][1];
    let m02 = mat3[0][2];

    let m10 = mat3[1][0];
    let m11 = mat3[1][1];
    let m12 = mat3[1][2];

    let m20 = mat3[2][0];
    let m21 = mat3[2][1];
    let m22 = mat3[2][2];
    if m22 <= 0.0 {
        // x^2 + y^2 >= z^2 + w^2
        let dif10 = m11 - m00;
        let omm22 = 1.0 - m22;
        if dif10 <= 0.0 {
            // x^2 >= y^2
            let four_xsq = omm22 - dif10;
            let inv4x = 0.5 * inverseSqrt(four_xsq);
            return vec4<f32>(
                four_xsq * inv4x,
                (m01 + m10) * inv4x,
                (m02 + m20) * inv4x,
                (m12 - m21) * inv4x,
            );
        } else {
            // y^2 >= x^2
            let four_ysq = omm22 + dif10;
            let inv4y = 0.5 * inverseSqrt(four_ysq);
            return vec4<f32>(
                (m01 + m10) * inv4y,
                four_ysq * inv4y,
                (m12 + m21) * inv4y,
                (m20 - m02) * inv4y,
            );
        }
    } else {
        // z^2 + w^2 >= x^2 + y^2
        let sum10 = m11 + m00;
        let opm22 = 1.0 + m22;
        if sum10 <= 0.0 {
            // z^2 >= w^2
            let four_zsq = opm22 - sum10;
            let inv4z = 0.5 * inverseSqrt(four_zsq);
            return vec4<f32>(
                (m02 + m20) * inv4z,
                (m12 + m21) * inv4z,
                four_zsq * inv4z,
                (m01 - m10) * inv4z,
            );
        } else {
            // w^2 >= z^2
            let four_wsq = opm22 + sum10;
            let inv4w = 0.5 * inverseSqrt(four_wsq);
            return vec4<f32>(
                (m12 - m21) * inv4w,
                (m20 - m02) * inv4w,
                (m01 - m10) * inv4w,
                four_wsq * inv4w,
            );
        }
    }
}
fn mat3_from_quat(quat: vec4<f32>) -> mat3x3<f32> {
    let x2 = quat.x + quat.x;
    let y2 = quat.y + quat.y;
    let z2 = quat.z + quat.z;
    let xx = quat.x * x2;
    let xy = quat.x * y2;
    let xz = quat.x * z2;
    let yy = quat.y * y2;
    let yz = quat.y * z2;
    let zz = quat.z * z2;
    let wx = quat.w * x2;
    let wy = quat.w * y2;
    let wz = quat.w * z2;

    return mat3x3<f32>(
        vec3<f32>(1.0 - (yy + zz), xy + wz, xz - wy),
        vec3<f32>(xy - wz, 1.0 - (xx + zz), yz + wx),
        vec3<f32>(xz + wy, yz - wx, 1.0 - (xx + yy))
    );
}


// --------------------------------
// @module: MeshBufferTypes
// --------------------------------

struct MeshMetadata {
    position_offset: u32,
    normal_offset: u32,
    tangent_offset: u32,
    texcoord0_offset: u32,
    joint_offset: u32,
    weight_offset: u32,
    index_offset: u32,

    index_count: u32,
};


struct MeshMetadatas {
    data: array<MeshMetadata>,
};


// --------------------------------
// @module: Resources
// --------------------------------

@group(0)
@binding(0)
var<storage> mesh_metadatas: MeshMetadatas;

@group(0)
@binding(1)
var<storage> mesh_position: Vec3Buffer;
@group(0)
@binding(2)
var<storage> mesh_normal: Vec3Buffer;
@group(0)
@binding(3)
var<storage> mesh_tangent: Vec3Buffer;
@group(0)
@binding(4)
var<storage> mesh_texcoord0: Vec2Buffer;
@group(0)
@binding(5)
var<storage> mesh_joint: UVec4Buffer;
@group(0)
@binding(6)
var<storage> mesh_weight: Vec4Buffer;

fn get_raw_mesh_position(vertex_index: u32) -> vec3<f32> {
    return mesh_position.data[vertex_index];
}

fn get_raw_mesh_uv(vertex_index: u32) -> vec2<f32> {
    return mesh_texcoord0.data[vertex_index];
}

fn get_mesh_position(mesh_id: u32, vertex_index: u32) -> vec3<f32> {
    return mesh_position.data[mesh_metadatas.data[mesh_id].position_offset + vertex_index];
}
fn get_mesh_normal(mesh_id: u32, vertex_index: u32) -> vec3<f32> {
    return mesh_normal.data[mesh_metadatas.data[mesh_id].normal_offset + vertex_index];
}
fn get_mesh_tangent(mesh_id: u32, vertex_index: u32) -> vec3<f32> {
    return mesh_tangent.data[mesh_metadatas.data[mesh_id].tangent_offset + vertex_index];
}
fn get_mesh_texcoord0(mesh_id: u32, vertex_index: u32) -> vec2<f32> {
    return mesh_texcoord0.data[mesh_metadatas.data[mesh_id].texcoord0_offset + vertex_index];
}
fn get_mesh_joint(mesh_id: u32, vertex_index: u32) -> vec4<u32> {
    return mesh_joint.data[mesh_metadatas.data[mesh_id].joint_offset + vertex_index];
}
fn get_mesh_weight(mesh_id: u32, vertex_index: u32) -> vec4<f32> {
    return mesh_weight.data[mesh_metadatas.data[mesh_id].weight_offset + vertex_index];
}

@group(0)
@binding(7)
var<storage> skins: Mat4x4Buffer;


// --------------------------------
// @module: GpuWorld
// --------------------------------

struct EntityLayoutBuffer { data: array<i32>, };
@group(1)
@binding(0)
var<storage> entity_layout: EntityLayoutBuffer;


struct EntityMat4Buffer { data: array<mat4x4<f32>> };

@group(1)
@binding(1)
var<storage> entity_Mat4_data: EntityMat4Buffer;

fn get_entity_component_offset_Mat4(component_index: u32, entity_loc: vec2<u32>) -> i32 {
    let archetypes = u32(entity_layout.data[0]);
    let layout_offset = 1u + (0u + component_index) * archetypes;
    return entity_layout.data[layout_offset + entity_loc.x];
}

fn get_entity_data_Mat4(component_index: u32, entity_loc: vec2<u32>) -> mat4x4<f32> {
    return entity_Mat4_data.data[u32(get_entity_component_offset_Mat4(component_index, entity_loc)) + entity_loc.y];
}
fn get_entity_data_or_Mat4(component_index: u32, entity_loc: vec2<u32>, default_value: mat4x4<f32>) -> mat4x4<f32> {
    let loc = get_entity_component_offset_Mat4(component_index, entity_loc);
    if (loc >= 0) {
        return entity_Mat4_data.data[u32(loc) + entity_loc.y];
    } else {
        return default_value;
    }
}




fn get_entity_mesh_to_world(entity_loc: vec2<u32>) -> mat4x4<f32> {
    return get_entity_data_Mat4(0u, entity_loc);
}

fn get_entity_mesh_to_world_or(entity_loc: vec2<u32>, default_value: mat4x4<f32>) -> mat4x4<f32> {
    return get_entity_data_or_Mat4(0u, entity_loc, default_value);
}

fn has_entity_mesh_to_world(entity_loc: vec2<u32>) -> bool {
    return get_entity_component_offset_Mat4(0u, entity_loc) >= 0;
}


fn get_entity_renderer_cameras_visible(entity_loc: vec2<u32>) -> mat4x4<f32> {
    return get_entity_data_Mat4(1u, entity_loc);
}

fn get_entity_renderer_cameras_visible_or(entity_loc: vec2<u32>, default_value: mat4x4<f32>) -> mat4x4<f32> {
    return get_entity_data_or_Mat4(1u, entity_loc, default_value);
}

fn has_entity_renderer_cameras_visible(entity_loc: vec2<u32>) -> bool {
    return get_entity_component_offset_Mat4(1u, entity_loc) >= 0;
}


fn get_entity_lod_cutoffs(entity_loc: vec2<u32>) -> mat4x4<f32> {
    return get_entity_data_Mat4(2u, entity_loc);
}

fn get_entity_lod_cutoffs_or(entity_loc: vec2<u32>, default_value: mat4x4<f32>) -> mat4x4<f32> {
    return get_entity_data_or_Mat4(2u, entity_loc, default_value);
}

fn has_entity_lod_cutoffs(entity_loc: vec2<u32>) -> bool {
    return get_entity_component_offset_Mat4(2u, entity_loc) >= 0;
}




struct EntityVec4Buffer { data: array<vec4<f32>> };

@group(1)
@binding(2)
var<storage> entity_Vec4_data: EntityVec4Buffer;

fn get_entity_component_offset_Vec4(component_index: u32, entity_loc: vec2<u32>) -> i32 {
    let archetypes = u32(entity_layout.data[0]);
    let layout_offset = 1u + (3u + component_index) * archetypes;
    return entity_layout.data[layout_offset + entity_loc.x];
}

fn get_entity_data_Vec4(component_index: u32, entity_loc: vec2<u32>) -> vec4<f32> {
    return entity_Vec4_data.data[u32(get_entity_component_offset_Vec4(component_index, entity_loc)) + entity_loc.y];
}
fn get_entity_data_or_Vec4(component_index: u32, entity_loc: vec2<u32>, default_value: vec4<f32>) -> vec4<f32> {
    let loc = get_entity_component_offset_Vec4(component_index, entity_loc);
    if (loc >= 0) {
        return entity_Vec4_data.data[u32(loc) + entity_loc.y];
    } else {
        return default_value;
    }
}




fn get_entity_world_bounding_sphere(entity_loc: vec2<u32>) -> vec4<f32> {
    return get_entity_data_Vec4(0u, entity_loc);
}

fn get_entity_world_bounding_sphere_or(entity_loc: vec2<u32>, default_value: vec4<f32>) -> vec4<f32> {
    return get_entity_data_or_Vec4(0u, entity_loc, default_value);
}

fn has_entity_world_bounding_sphere(entity_loc: vec2<u32>) -> bool {
    return get_entity_component_offset_Vec4(0u, entity_loc) >= 0;
}


fn get_entity_visibility_from(entity_loc: vec2<u32>) -> vec4<f32> {
    return get_entity_data_Vec4(1u, entity_loc);
}

fn get_entity_visibility_from_or(entity_loc: vec2<u32>, default_value: vec4<f32>) -> vec4<f32> {
    return get_entity_data_or_Vec4(1u, entity_loc, default_value);
}

fn has_entity_visibility_from(entity_loc: vec2<u32>) -> bool {
    return get_entity_component_offset_Vec4(1u, entity_loc) >= 0;
}


fn get_entity_color(entity_loc: vec2<u32>) -> vec4<f32> {
    return get_entity_data_Vec4(2u, entity_loc);
}

fn get_entity_color_or(entity_loc: vec2<u32>, default_value: vec4<f32>) -> vec4<f32> {
    return get_entity_data_or_Vec4(2u, entity_loc, default_value);
}

fn has_entity_color(entity_loc: vec2<u32>) -> bool {
    return get_entity_component_offset_Vec4(2u, entity_loc) >= 0;
}


fn get_entity_outline(entity_loc: vec2<u32>) -> vec4<f32> {
    return get_entity_data_Vec4(3u, entity_loc);
}

fn get_entity_outline_or(entity_loc: vec2<u32>, default_value: vec4<f32>) -> vec4<f32> {
    return get_entity_data_or_Vec4(3u, entity_loc, default_value);
}

fn has_entity_outline(entity_loc: vec2<u32>) -> bool {
    return get_entity_component_offset_Vec4(3u, entity_loc) >= 0;
}


fn get_entity_gpu_lod(entity_loc: vec2<u32>) -> vec4<f32> {
    return get_entity_data_Vec4(4u, entity_loc);
}

fn get_entity_gpu_lod_or(entity_loc: vec2<u32>, default_value: vec4<f32>) -> vec4<f32> {
    return get_entity_data_or_Vec4(4u, entity_loc, default_value);
}

fn has_entity_gpu_lod(entity_loc: vec2<u32>) -> bool {
    return get_entity_component_offset_Vec4(4u, entity_loc) >= 0;
}


fn get_entity_skin(entity_loc: vec2<u32>) -> vec4<f32> {
    return get_entity_data_Vec4(5u, entity_loc);
}

fn get_entity_skin_or(entity_loc: vec2<u32>, default_value: vec4<f32>) -> vec4<f32> {
    return get_entity_data_or_Vec4(5u, entity_loc, default_value);
}

fn has_entity_skin(entity_loc: vec2<u32>) -> bool {
    return get_entity_component_offset_Vec4(5u, entity_loc) >= 0;
}


fn get_entity_ui_size(entity_loc: vec2<u32>) -> vec4<f32> {
    return get_entity_data_Vec4(6u, entity_loc);
}

fn get_entity_ui_size_or(entity_loc: vec2<u32>, default_value: vec4<f32>) -> vec4<f32> {
    return get_entity_data_or_Vec4(6u, entity_loc, default_value);
}

fn has_entity_ui_size(entity_loc: vec2<u32>) -> bool {
    return get_entity_component_offset_Vec4(6u, entity_loc) >= 0;
}




struct EntityUVec4Array20Buffer { data: array<array<vec4<u32>, 20>> };

@group(1)
@binding(3)
var<storage> entity_UVec4Array20_data: EntityUVec4Array20Buffer;

fn get_entity_component_offset_UVec4Array20(component_index: u32, entity_loc: vec2<u32>) -> i32 {
    let archetypes = u32(entity_layout.data[0]);
    let layout_offset = 1u + (10u + component_index) * archetypes;
    return entity_layout.data[layout_offset + entity_loc.x];
}

fn get_entity_data_UVec4Array20(component_index: u32, entity_loc: vec2<u32>) -> array<vec4<u32>, 20> {
    return entity_UVec4Array20_data.data[u32(get_entity_component_offset_UVec4Array20(component_index, entity_loc)) + entity_loc.y];
}
fn get_entity_data_or_UVec4Array20(component_index: u32, entity_loc: vec2<u32>, default_value: array<vec4<u32>, 20>) -> array<vec4<u32>, 20> {
    let loc = get_entity_component_offset_UVec4Array20(component_index, entity_loc);
    if (loc >= 0) {
        return entity_UVec4Array20_data.data[u32(loc) + entity_loc.y];
    } else {
        return default_value;
    }
}




fn get_entity_primitives(entity_loc: vec2<u32>) -> array<vec4<u32>, 20> {
    return get_entity_data_UVec4Array20(0u, entity_loc);
}

fn get_entity_primitives_or(entity_loc: vec2<u32>, default_value: array<vec4<u32>, 20>) -> array<vec4<u32>, 20> {
    return get_entity_data_or_UVec4Array20(0u, entity_loc, default_value);
}

fn has_entity_primitives(entity_loc: vec2<u32>) -> bool {
    return get_entity_component_offset_UVec4Array20(0u, entity_loc) >= 0;
}





// --------------------------------
// @module: RendererCollect
// --------------------------------

struct Params {
    camera: u32,
};
@group(2)
@binding(0)
var<uniform> params: Params;

struct CollectPrimitive {
    entity_loc: vec2<u32>,
    primitive_index: u32,
    material_index: u32,
};
struct CollectPrimitives { data: array<CollectPrimitive>, };
@group(2)
@binding(1)
var<storage> input_primitives: CollectPrimitives;

struct DrawIndexedIndirect {
    vertex_count: u32,
    instance_count: u32,
    base_index: u32,
    vertex_offset: i32,
    base_instance: u32,
};

struct Commands {
    data: array<DrawIndexedIndirect>,
};
@group(2)
@binding(2)
var<storage, read_write> output_commands: Commands;

struct AtomicU32Buffer { data: array<atomic<u32>>, };
@group(2)
@binding(3)
var<storage, read_write> output_counts: AtomicU32Buffer;

@group(2)
@binding(4)
var<storage> material_layouts: UVec2Buffer;


fn is_visible(entity_loc: vec2<u32>, primitive_lod: u32) -> bool {

    var visibility_from: vec2<u32> = entity_loc;
    if (has_entity_visibility_from(entity_loc)) {
        let visibility_from_raw = get_entity_visibility_from(entity_loc);
        // reinterpret floats as u32
        visibility_from = bitcast<vec4<u32>>(visibility_from_raw).xy;
    }

    // let entity_lod = u32(get_entity_gpu_lod_or(visibility_from, 0.0).x);
    let flod = get_entity_gpu_lod_or(visibility_from, vec4<f32>(0.0));
    let entity_lod = u32(flod.x);

    if (entity_lod != primitive_lod) {
        return false;
    }
    if (has_entity_renderer_cameras_visible(visibility_from)) {
        var cameras = get_entity_renderer_cameras_visible(visibility_from);
        let camera_i = params.camera >> 2u;
        let camera_j = params.camera & 3u;

        return cameras[camera_i][camera_j] > 0.0;
    } else {
        return true;
    }
}

@compute
@workgroup_size(32)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let chunk = 256u * 32u;
    let index = global_id.y * chunk + global_id.x;

    if (index >= arrayLength(&input_primitives.data)) {
        return;
    }


    let primitive = input_primitives.data[index];
    let material_layout = material_layouts.data[primitive.material_index];
    if (index < material_layout.x || index >= material_layout.x + material_layout.y) {
        return;
    }

    var entity_primitives = get_entity_primitives(primitive.entity_loc);
    let entity_primitive = entity_primitives[primitive.primitive_index];
    let primitive_lod = entity_primitive.y;

    if (is_visible(primitive.entity_loc, primitive_lod)) {
        let out_offset = atomicAdd(&output_counts.data[primitive.material_index], 1u);
        let out_index = material_layout.x + out_offset;
        let mesh_index = entity_primitive.x;
        let mesh = mesh_metadatas.data[mesh_index];
        output_commands.data[out_index].vertex_count = mesh.index_count;
        output_commands.data[out_index].instance_count = 1u;
        output_commands.data[out_index].base_index = mesh.index_offset;
        output_commands.data[out_index].base_instance = index;
    }
}
