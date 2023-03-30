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
// @module: Globals
// --------------------------------

@group(1)
@binding(0)
var default_sampler: sampler;

struct ForwardGlobalParams {
    projection_view: mat4x4<f32>,
    inv_projection_view: mat4x4<f32>,
    camera_position: vec4<f32>,
    camera_forward: vec3<f32>,
    camera_far: f32,
    sun_direction: vec4<f32>,
    sun_diffuse: vec4<f32>,
    sun_ambient: vec4<f32>,
    fog_color: vec4<f32>,
    forward_camera_position: vec4<f32>,
    fog: i32,
    time: f32,
    fog_height_falloff: f32,
    fog_density: f32,

    debug_metallic_roughness: f32,
    debug_normals: f32,
    debug_shading: f32,
};

struct ShadowCamera {
    viewproj: mat4x4<f32>,
    far: f32,
    near: f32,
};

@group(1)
@binding(1)
var<uniform> global_params: ForwardGlobalParams;

struct ShadowCameras {
    cameras: array<ShadowCamera>,
};

@group(1)
@binding(2)
var<storage> shadow_cameras: ShadowCameras;

@group(1)
@binding(3)
var shadow_sampler: sampler_comparison;
@group(1)
@binding(4)
var shadow_texture: texture_depth_2d_array;

@group(1)
@binding(5)
var solids_screen_color: texture_2d<f32>;

@group(1)
@binding(6)
var solids_screen_depth: texture_depth_2d;

@group(1)
@binding(7)
var solids_screen_normal_quat: texture_2d<f32>;

fn inside(v: vec3<f32>) -> bool {
    return v.x > -1. && v.x < 1. && v.y > -1. && v.y < 1. && v.z > 0. && v.z < 1.;
}

fn fetch_shadow_cascade(cascade: i32, homogeneous_coords: vec3<f32>) -> f32 {
    let light_local = homogeneous_coords.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    return textureSampleCompareLevel(shadow_texture, shadow_sampler, light_local, cascade, homogeneous_coords.z + 0.0001);
}

fn get_shadow_cascade(world_position: vec4<f32>) -> i32 {
    for (var i: i32=0; i < 5; i = i + 1) {
        let cam = shadow_cameras.cameras[i];
        let p = cam.viewproj * world_position;
        if (inside(p.xyz / p.w)) {
            return i;
        }
    }
    return 0;
}

fn fetch_shadow(light_angle: f32, world_position: vec4<f32>) -> f32 {
    for (var i: i32=0; i < 5; i = i + 1) {
        // The texel size is in world coordinates, transform to depth buffer by
        // dividing by the depth of the camera
        let cam = shadow_cameras.cameras[i].viewproj * world_position;
        let p = cam.xyz / cam.w;
        if (inside(p)) {
            return fetch_shadow_cascade(i, p);
        }
    }
    return 1.;
}

fn screen_pixel_to_uv(pixel_position: vec2<f32>, screen_size: vec2<f32>) -> vec2<f32> {
    return pixel_position / screen_size;
}

fn screen_uv_to_pixel(uv: vec2<f32>, screen_size: vec2<f32>) -> vec2<f32> {
    return uv * screen_size;
}

fn screen_uv_to_ndc(uv: vec2<f32>) -> vec3<f32> {
    return vec3<f32>(uv.x * 2. - 1., -(uv.y * 2. - 1.), 0.);
}

fn screen_ndc_to_uv(ndc: vec3<f32>) -> vec2<f32> {
    return vec2<f32>((ndc.x + 1.) / 2., (-ndc.y + 1.) / 2.);
}

fn screen_pixel_to_ndc(pixel_position: vec2<f32>, screen_size: vec2<f32>) -> vec3<f32> {
    return screen_uv_to_ndc(screen_pixel_to_uv(pixel_position, screen_size));
}

fn screen_ndc_to_pixel(ndc: vec3<f32>, screen_size: vec2<f32>) -> vec2<f32> {
    return screen_uv_to_pixel(screen_ndc_to_uv(ndc), screen_size);
}

fn project_point(transform: mat4x4<f32>, position: vec3<f32>) -> vec3<f32> {
    let p = transform * vec4<f32>(position, 1.);
    return p.xyz / p.w;
}

fn get_solids_screen_depth(screen_ndc: vec3<f32>) -> f32 {
    let screen_tc = screen_ndc_to_uv(screen_ndc);
    // return textureSampleLevel(solids_screen_depth, default_sampler, screen_tc, 0.);
    return textureSample(solids_screen_depth, default_sampler, screen_tc);
}

fn get_solids_screen_color(screen_ndc: vec3<f32>) -> vec3<f32> {
    let screen_tc = screen_ndc_to_uv(screen_ndc);
    return textureSample(solids_screen_color, default_sampler, screen_tc).rgb;
}

fn get_solids_screen_normal_quat(screen_ndc: vec3<f32>) -> vec4<f32> {
    let screen_tc = screen_ndc_to_uv(screen_ndc);
    return textureSample(solids_screen_normal_quat, default_sampler, screen_tc);
}

struct MaterialInput {
    position: vec4<f32>,
    texcoord: vec2<f32>,
    world_position: vec3<f32>,
    normal: vec3<f32>,
    normal_matrix: mat3x3<f32>,
    instance_index: u32,
    entity_loc: vec2<u32>,
    local_position: vec3<f32>,
};

struct MaterialOutput {
    base_color: vec3<f32>,
    emissive_factor: vec3<f32>,
    opacity: f32,
    alpha_cutoff: f32,
    shading: f32,
    normal: vec3<f32>,
    metallic: f32,
    roughness: f32,
};

struct MainFsOut {
    @location(0) color: vec4<f32>,
    @location(1) normal: vec4<f32>,
}

fn apply_fog(color: vec3<f32>, camera_pos: vec3<f32>, world_pos: vec3<f32>) -> vec3<f32> {
    // From https://developer.amd.com/wordpress/media/2012/10/Wenzel-Real-time_Atmospheric_Effects_in_Games.pdf
    let camera_to_world_pos = world_pos - camera_pos;
    let vol_fog_height_density_at_viewer = exp( -global_params.fog_height_falloff * camera_pos.z );

    var fog_int = length(camera_to_world_pos) * vol_fog_height_density_at_viewer;
    let slope_threashold = 0.01;
    if (abs(camera_to_world_pos.z) > slope_threashold) {
        let t = global_params.fog_height_falloff * camera_to_world_pos.z;
        fog_int = fog_int * ( 1.0 - exp( -t ) ) / t;
    }
    let fog_amount = 1. - exp( -global_params.fog_density * fog_int );
    return mix(color, global_params.fog_color.rgb, clamp(fog_amount, 0., 1.));
}

fn fresnel(ndoth: f32, f0: vec3<f32>) -> vec3<f32> {
    let v = clamp(1.0 - ndoth, 0.0, 1.0);
    return f0 + (1.0 - f0) * pow(v, 5.0);
}

// Section: PBR

fn distribution_ggx(normal: vec3<f32>, h: vec3<f32>, roughness: f32) -> f32 {
    // A squared roughness looks more correct based on observation by Disney and
    // Unreal
    let a = roughness * roughness;
    let a2 = a * a;

    let ndoth = max(dot(normal, h), 0.0);
    let ndoth2 = ndoth * ndoth;

    let numerator =a2;
    let denom = ndoth2 * (a2 - 1.0) + 1.0;

    let denom2 = PI * denom * denom;
    return numerator / denom2;
}

fn geometry_schlick_ggx(ndotv: f32, k: f32) -> f32 {
    let numerator = ndotv;
    let denom = ndotv * (1.0 - k) + k;

    return numerator / denom;
}

fn geometry_smith(normal: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32)
-> f32 {
    // See prior comment about roughness squaring
    let a = (roughness * roughness) + 1.0;

    // Direct mapping
    // (r+1)^2 / 8.0
    let k = (a * a) / 8.0;

    let ndotv = max(dot(normal, v), 0.0);
    let ndotl = max(dot(normal, l), 0.0);

    return
          geometry_schlick_ggx(ndotv, k)
        * geometry_schlick_ggx(ndotl, k);
}

fn shading(material: MaterialOutput, world_position: vec4<f32>) -> vec4<f32> {
    if (global_params.debug_shading > 0.0) {
      return vec4(material.base_color.rgb, material.opacity);
    }

    let v = normalize(global_params.camera_position.xyz - world_position.xyz);

    let l = normalize(global_params.sun_direction.xyz);
    let h = normalize(v + l);

    let albedo = material.base_color.rgb;

    let metallic = material.metallic;
    let roughness = material.roughness;
    let normal = material.normal;

    // Interpolate the normal incidence.
    //
    // I.e; the reflected light rays when viewed straight ahead.
    //
    // For dielectric materials, such as wood, ceramic, plastics, etc, the
    // reflected light is not tinted and reflected by an average of (0.04, 0.04,
    // .0.4) of the
    // incoming ray.
    //
    // This value approaches (1,1,1) at a perpendicular incidence.
    //
    // Metallic materials tint the reflected light, and this is perceived as the
    // metals gloss.
    //
    // This tint is approximated to the albedo when the metallic factor is high
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let f = fresnel(max(dot(h,v), 0.0), f0);

    // Approximate microfacet alignment againt the halfway view direction
    let ndf = distribution_ggx(normal, h, roughness);
    // Approximate geometry occlusion and shadowing caused by microfacet induced
    // roughness
    let g = geometry_smith(normal, v, l, roughness);

    let ndotl = max(dot(normal, l), 0.0);
    let ndotv = max(dot(normal, v), 0.0);

    /// dgf / (4 n*v n*l)
    let numerator = ndf * g * f;
    let denom =
        max(4.0
        * ndotv
        * ndotl, 0.001);


    // Use fresnell scattering as a reflection coefficient
    let ks = f;

    // The diffuse/refracted coefficient is the opposite of the reflected
    // factor.
    //
    // If the material is metallic, the reflacted light is completely absorbed
    // and does not exit the material.
    //
    // This causes metallic objects to have no diffuse light
    let kd = (vec3<f32>(1.0) - ks) * (1.0 - metallic);

    let lambert = kd * albedo / PI;
    // Cook-torrance specular reflection
    let specular = ks * (ndf * g * f) / denom;

    let radiance = global_params.sun_diffuse.rgb;

    let in_shadow = fetch_shadow(ndotl, world_position);

    let direct = (lambert + specular) * radiance * ndotl * in_shadow;

    let indirect = albedo * global_params.sun_ambient.rgb;

    let lum = direct + indirect;

    var color = mix(material.base_color.rgb, lum, material.shading) + material.emissive_factor;

    color = mix(color, vec3(metallic, roughness, 0.0), global_params.debug_metallic_roughness);
    color = mix(color, normal, global_params.debug_normals);

    if (global_params.fog != 0) {
        color = apply_fog(color, global_params.camera_position.xyz, world_position.xyz);
    }

    // let color = vec3<f32>(roughness, metallic, 0.0);
    // color = color + u32_to_color(u32(get_shadow_cascade(world_position))) * 0.2;

    // let color = vec3<f32>(max(dot(material.normal, l), 0.0), 0.0, 0.0);

    return vec4<f32>(color,  material.opacity);

}

struct FSOutput {
    @location(0) color: vec4<f32>,
    @location(1) outline: vec4<f32>,
};


// --------------------------------
// @module: GpuWorld
// --------------------------------

struct EntityLayoutBuffer { data: array<i32>, };
@group(2)
@binding(0)
var<storage> entity_layout: EntityLayoutBuffer;


struct EntityMat4Buffer { data: array<mat4x4<f32>> };

@group(2)
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

@group(2)
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

@group(2)
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
// @module: Common
// --------------------------------


@group(3)
@binding(0)
var<storage> primitives: UVec4Buffer;

struct ModelToWorld {
    local: vec4<f32>,
    pos: vec4<f32>,
    normal: vec3<f32>,
    tangent: vec3<f32>,
}

/// Transform a vertex from model space to world space by applying
// joint matrices (if applicable) and transformation matrices
fn model_to_world(loc: vec2<u32>, mesh_index: u32, vertex_index: u32) -> ModelToWorld {
    let model = get_entity_mesh_to_world(loc);
    let pos = vec4<f32>(get_mesh_position(mesh_index, vertex_index), 1.0);
    let normal = vec4<f32>(get_mesh_normal(mesh_index, vertex_index), 0.0);
    let tangent = vec4<f32>(get_mesh_tangent(mesh_index, vertex_index), 0.0);

    if (has_entity_skin(loc)) {
        let joint = get_mesh_joint(mesh_index, vertex_index);
        let weight = get_mesh_weight(mesh_index, vertex_index);
        let skin_offset = u32(get_entity_skin(loc).x);

        let ltw_x: mat4x4<f32> = skins.data[skin_offset + joint.x];
        let ltw_y: mat4x4<f32> = skins.data[skin_offset + joint.y];
        let ltw_z: mat4x4<f32> = skins.data[skin_offset + joint.z];
        let ltw_w: mat4x4<f32> = skins.data[skin_offset + joint.w];

        var total_pos     = vec4<f32>(0.0);
        var total_norm    = vec4<f32>(0.0);
        var total_tangent = vec4<f32>(0.0);

        // Normalize the weights
        let mesh_weight = weight / dot(weight, vec4<f32>(1.0));

        total_pos = total_pos + (ltw_x * pos) * mesh_weight.x;
        total_pos = total_pos + (ltw_y * pos) * mesh_weight.y;
        total_pos = total_pos + (ltw_z * pos) * mesh_weight.z;
        total_pos = total_pos + (ltw_w * pos) * mesh_weight.w;

        total_pos.w = 1.0;

        total_norm = total_norm + (ltw_x * normal) * mesh_weight.x;
        total_norm = total_norm + (ltw_y * normal) * mesh_weight.y;
        total_norm = total_norm + (ltw_z * normal) * mesh_weight.z;
        total_norm = total_norm + (ltw_w * normal) * mesh_weight.w;


        total_tangent = total_tangent + (ltw_x * tangent) * mesh_weight.x;
        total_tangent = total_tangent + (ltw_y * tangent) * mesh_weight.y;
        total_tangent = total_tangent + (ltw_z * tangent) * mesh_weight.z;
        total_tangent = total_tangent + (ltw_w * tangent) * mesh_weight.w;

        return ModelToWorld(total_pos, model * total_pos, normalize((model * total_norm).xyz), normalize((model * normalize(tangent)).xyz));
    } else {
        return ModelToWorld(pos, model * pos, normalize((model * normal).xyz), normalize((model * tangent).xyz));
    }
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
// @module: RectMaterial
// --------------------------------

struct RectMaterialParams {
    background: vec4<f32>,
    border_color: vec4<f32>,
    border_radius: vec4<f32>,
    border_thickness: f32,
}
@group(4)
@binding(0)
var<uniform> rect_params: RectMaterialParams;

fn get_corner_from_tc(tc: vec2<f32>) -> u32 {
    var corner = 0u;
    if (tc.x >= 0.5) {
        corner += 1u;
    }
    if (tc.y >= 0.5) {
        corner += 2u;
    }
    return corner;
}

fn get_material(in: MaterialInput) -> MaterialOutput {
    var out: MaterialOutput;
    out.roughness = 0.4;
    out.metallic = 0.5;
    let size = get_entity_ui_size(in.entity_loc).xy;
    let p = (0.5 - abs(in.texcoord - 0.5)) * size; // Normalized to top left
    let corner = get_corner_from_tc(in.texcoord);
    let border_radius = rect_params.border_radius[corner];
    let d = distance(vec2(border_radius), p);

    let entity_color = get_entity_color_or(in.entity_loc, vec4<f32>(1., 1., 1., 1.));
    let border_color = rect_params.border_color * entity_color;
    var color = rect_params.background * entity_color;
    if (max(p.x, p.y) <= border_radius) {
        if (d > border_radius) {
            color.a = 0.;
        } else if (d > border_radius - rect_params.border_thickness) {
            color = border_color;
        }
    } else if (min(p.x, p.y) < rect_params.border_thickness) {
        color = border_color;
    }
    out.opacity = color.a;
    out.alpha_cutoff = 0.;
    out.base_color = from_srgb_to_linear(color.rgb);
    out.emissive_factor = vec3<f32>(0., 0., 0.);
    out.shading = 1.;
    out.normal = in.normal;
    return out;
}


// --------------------------------
// @module: StandardMaterial
// --------------------------------

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) texcoord: vec2<f32>,
    @location(1) world_position: vec4<f32>,
    @location(2) instance_index: u32,
    @location(3) world_tangent: vec3<f32>,
    @location(4) world_bitangent: vec3<f32>,
    @location(5) world_normal: vec3<f32>,
    @location(6) local_position: vec3<f32>,
};

@vertex
fn vs_main(@builtin(instance_index) instance_index: u32, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    let primitive = primitives.data[instance_index];
    let entity_loc = primitive.xy;
    var entity_primitives = get_entity_primitives(entity_loc);
    let mesh_index = entity_primitives[primitive.z].x;

    out.instance_index = instance_index;
    out.texcoord = get_mesh_texcoord0(mesh_index, vertex_index);

    let world = model_to_world(entity_loc, mesh_index, vertex_index);

    out.world_normal = world.normal;
    out.world_tangent = world.tangent;
    out.world_bitangent = cross(world.normal, world.tangent);
    out.world_position = world.pos;
    out.local_position = world.local.xyz;

    let clip = global_params.projection_view * world.pos;

    out.position = clip;
    return out;
}

fn get_material_in(in: VertexOutput, is_front: bool) -> MaterialInput {
    var material_in: MaterialInput;
    material_in.position = in.position;
    material_in.texcoord = in.texcoord;
    material_in.world_position = in.world_position.xyz / in.world_position.w;
    material_in.normal = in.world_normal;
    material_in.normal_matrix = mat3x3<f32>(
        in.world_tangent,
        in.world_bitangent,
        in.world_normal
    );
    material_in.instance_index = in.instance_index;
    material_in.entity_loc = primitives.data[in.instance_index].xy;
    material_in.local_position = in.local_position;
    return material_in;
}

@fragment
fn fs_shadow_main(in: VertexOutput, @builtin(front_facing) is_front: bool) {
    var material = get_material(get_material_in(in, is_front));

    if (material.opacity < material.alpha_cutoff) {
        discard;
    }
}

fn get_outline(instance_index: u32) -> vec4<f32> {
    let entity_loc = primitives.data[instance_index].xy;
    return get_entity_outline_or(entity_loc, vec4<f32>(0., 0., 0., 0.));
}

@fragment
fn fs_forward_lit_main(in: VertexOutput, @builtin(front_facing) is_front: bool) -> MainFsOut {
    let material_in = get_material_in(in, is_front);
    var material = get_material(material_in);

    if (material.opacity < material.alpha_cutoff) {
        discard;
    }

    if (!is_front) {
        material.normal = -material.normal;
    }

    material.normal = normalize(material.normal);

    return MainFsOut(
        shading(material, in.world_position),
        quat_from_mat3(material_in.normal_matrix)
    );
}

@fragment
fn fs_forward_unlit_main(in: VertexOutput, @builtin(front_facing) is_front: bool) -> MainFsOut {
    let material_in = get_material_in(in, is_front);
    var material = get_material(material_in);

    if (material.opacity < material.alpha_cutoff) {
        discard;
    }

    return MainFsOut(
        vec4<f32>(material.base_color, material.opacity),
        quat_from_mat3(material_in.normal_matrix)
    );
}

@fragment
fn fs_outlines_main(in: VertexOutput, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    var material = get_material(get_material_in(in, is_front));

    if (material.opacity < material.alpha_cutoff) {
        discard;
    }
    return get_outline(in.instance_index);
}
