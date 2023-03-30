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
// @module: Globals
// --------------------------------

@group(0)
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

@group(0)
@binding(1)
var<uniform> global_params: ForwardGlobalParams;

struct ShadowCameras {
    cameras: array<ShadowCamera>,
};

@group(0)
@binding(2)
var<storage> shadow_cameras: ShadowCameras;

@group(0)
@binding(3)
var shadow_sampler: sampler_comparison;
@group(0)
@binding(4)
var shadow_texture: texture_depth_2d_array;

@group(0)
@binding(5)
var solids_screen_color: texture_2d<f32>;

@group(0)
@binding(6)
var solids_screen_depth: texture_depth_2d;

@group(0)
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
// @module: Scatter
// --------------------------------
let atmo_radius = 6471e3;
let planet_radius = 6371e3;

struct Node {
    density: f32,
    origin: vec3<f32>,
    size: f32,
    children: array<u32, 8>,
};

struct CloudBuffer {
    clouds: array<Node>,
};

@group(1)
@binding(0)
var<storage> cloud_buffer: CloudBuffer;

struct CloudBuffer {
    clouds: array<Node>,
};

fn sphere_intersect(pos: vec3<f32>, dir: vec3<f32>, r: f32) -> vec2<f32> {
    let a = dot(dir, dir);
    let b = 2.0 * dot(dir, pos);
    let c = dot(pos, pos) - (r * r);
    let d = (b * b) - 4.0 * a * c;

    if (d < 0.) {
        return vec2<f32>(1e5, -1e5);
    }

    return vec2<f32>(
        max((-b - sqrt(d)) / (2.0 * a), 0.0),
        (-b + sqrt(d)) / (2.0 * a),
    );
}

let DISTANCE_THRESHOLD: f32 = 0.1;

fn cube_intersect(origin: f32, size: f32, ray_o: vec3<f32>, dir: vec3<f32>) ->
f32 {
    let origin = ray_o - origin;
    let inv_dir = vec3<f32>(1. / dir.x, 1. / dir.y, 1. / dir.z);
    let t1 = (-size - origin) * inv_dir;
    let t2 = (size - origin) * inv_dir;

    let tmin = min(t1, t2);
    let tmax = max(t1, t2);

    let tmin = max(tmin.x, max(tmin.y, tmin.z));
    let tmax = min(tmax.x, min(tmax.y, tmax.z));

    return f32(tmax > 0.) * (tmax - tmin);
}

// Returns the distance along the ray, or < 0
fn node_ray(node: Node, origin: vec3<f32>, dir: vec3<f32>) -> f32 {
    return 1.0;
}

fn scattering(depth: f32, pos: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    let orig = pos + vec3<f32>(0.0, 0.0, planet_radius);
    // Get atmosphere intersection
    let ray_l = sphere_intersect(orig, dir, atmo_radius);

    let dist =  ray_l.y - ray_l.x;

    let dir = normalize(dir);

    let steps = 16;
    let step_len = dist / f32(steps);

    let light_dir = global_params.sun_direction.xyz;
    let u = dot(dir, light_dir);
    let g = 0.76;
    let uu = u*u;
    let gg = g*g;

    let beta_ray = vec3<f32>(5.5e-6, 13.0e-6, 22.4e-6);
    // let beta_ray = vec3<f32>(3.8e-6, 5.5e-7, 16.1e-6);
    let beta_mie = vec3<f32>(21e-6);

    let allow_mie = depth >= ray_l.y || depth > global_params.camera_far * 0.9;
    // How likely is light from the sun to scatter to us
    let phase_ray = max(3.0 / (16.0*PI) * (1.0 + uu), 0.);
    // 3 / (16pi) * cos2()
    let phase_mie = max((3.0 / (8.0 * PI)) * ((1.0 - gg) * (1.0 + uu)) / ((2.0 +
    gg) * pow(1.0 + gg - 2.0*g*u, 1.5)), 0.);

    let phase_mie = phase_mie * f32(allow_mie);

    // Accumulation of scattered light
    var total_ray = vec3<f32>(0.);
    var total_mie = vec3<f32>(0.);

    // Optical depth
    var rayleigh = 0.0;
    var mie = 0.0;

    let Hr = 8e3;
    let Hm = 1.2e3;

    var pos_i = 0.0;

    // Primary ray
    for (var i = 0; i < steps; i = i + 1) {
        let p = orig + dir * pos_i;

        let height = length(p) - planet_radius;
        let hr = exp(-height / Hr) * step_len;
        let hm = exp(-height / Hm) * step_len;

        // Accumulate density along viewing ray
        rayleigh = rayleigh + max(hr, 0.);
        mie = mie + max(hm, 0.);

        // Distance from ray sample to the atmosphere towards the sun
        let ray = sphere_intersect(p, light_dir, atmo_radius);

        // if (dist < 0.0) { return vec3<f32>(1., 0., 0.); }

        // Cast ray into the sun
        let sun_steps = 8;
        let sun_len = ray.y / f32(sun_steps);
        // Density along light ray
        var l_r = 0.0;
        var l_m = 0.0;

        var pos_l = ray_l.x;

        for (var j = 0; j < sun_steps; j = j + 1) {
            // let l_pos = p + light_dir * f32(j) * sun_len;
            let p = p + light_dir * pos_l;

            let height_l = length(p) - planet_radius;
            let h_ray = exp(-height_l / Hr) * sun_len;
            let h_mie = exp(-height_l / Hm) * sun_len;

            l_r = l_r + max(h_ray, 0.);
            l_m = l_m + max(h_mie, 0.);

            pos_l = pos_l + sun_len;
        }

        // Add the results of light integration by using the accumulated density
        // and beta coeff
        let tau =
            beta_ray * (rayleigh + l_r)
            + beta_mie * (mie + l_m);

        let attn = exp(-tau);

        total_ray = total_ray + attn * hr;
        total_mie = total_mie + attn * hm;

        // Travel forward
        pos_i = pos_i + step_len;
    }

    let result =
    (
          total_ray * beta_ray * phase_ray
        + total_mie * beta_mie * phase_mie
    ) * 20.0;

    return result;
}

fn get_sky_color(
    depth: f32,
    origin: vec3<f32>,
    forward: vec3<f32>,
) -> vec3<f32> {
    let spot_rad = 1.0 - dot(forward, global_params.sun_direction.xyz);
    let d = 2000.0;
    let g = 3.0;
    let spot = exp(-pow(d * spot_rad, g));


    return scattering(depth, origin, forward)
    + global_params.sun_diffuse.rgb * spot;
}


// --------------------------------
// @module: Clouds
// --------------------------------
// [[group(1), binding(1)]]
// var<uniform> params: CloudParams;
// [[group(1), binding(2)]]
// var texture: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) uv: vec2<f32>,
};

struct Scene {
    color: vec3<f32>,
};

struct Conf {
    steps_i: i32,
    steps_l: i32,
};

struct Planet {
    pos: vec3<f32>,
    radius: f32,
    // Radius of atmosphere from planet core
    atmo_radius: f32,

};

fn camera_ray(res: vec3<f32>, coord: vec2<f32> ) -> vec3<f32> {
    let uv = coord.xy - vec2<f32>(0.5);
    return normalize(vec3<f32>(uv.x, uv.y, -1.0));
}

struct Volume {
    absorption: vec3<f32>, // How much light passes through

};

struct Sample {
    dist: f32,
    vol: Volume,
};

let air_volume   = Volume(vec3<f32>(1e-6, 1e-6, 1e-6));
let cloud_volume = Volume(vec3<f32>(0.1,  0.1,  0.1));

fn smooth_min(a: Sample, b: Sample, t: f32) -> Sample {
    let h = max(t - abs(a.dist - b.dist), 0.0);
    var vol = a.vol;
    if (b.dist < 0.0) {
        vol = b.vol;
    }

    return Sample(min(a.dist, b.dist) - h*h*h/(6.0*t*t), vol);
}

fn sdf(p: vec3<f32>) -> Sample {
    var min_d = Sample(10000.0, air_volume);
    // for (var i = 0; i < params.count; i = i + 1) {
    //     let dist = cloud_sdf(cloud_buffer.clouds[i], p);

    //     min_d = smooth_min(min_d, dist, 2.0);
    // }

    return min_d;
}

// fn calc_normal(pos: vec3<f32>) -> vec3<f32> {
//     // Center sample
//     let c = sdf(pos);
//     // Use offset samples to compute gradient / normal
//     let eps_zero = vec2<f32>(0.001, 0.0);
//     return normalize(vec3<f32>(sdf(pos + eps_zero.xyy), sdf(pos + eps_zero.yxy), sdf(pos + eps_zero.yyx)) - c);
// }

// fn raymarch(uv: vec2<f32>, origin: vec3<f32>, dir: vec3<f32>, scene: Scene, planet: Planet) -> vec4<f32> {
//     let max_steps = 100;
//     let eps = 0.01;
//     let close = 0.1;
//     let max_d = 1000.0;

//     var depth = 0.0;

//     var dens = 0.0;

//     for (var i = 0; i < max_steps; i = i + 1) {
//         let p = origin + depth * dir;
//         let dist = sdf(p);
//         let abs_dist = abs(dist);

//         // Boundary
//         if (dist < close) {
//             depth = depth + 0.1;
//         }

//         if (dist < 0.0) {
//             dens = dens + abs_dist;
//         }

//         if (depth > max_d) {
//             break;
//         }

//         depth = depth + abs_dist;
//     }

//     let opacity = exp(-dens);

//     // let sky = get_sky_color(uv, dir, 1e6, scene, planet);
//     let sky = vec4<f32>(0.,0.,1., 1.);

//     let cloud = vec4<f32>(1.,1.,1.,1.);
//     return mix(cloud, sky, opacity);
// }


@vertex
fn vs_main(@builtin(instance_index) instance_index: u32, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {

    var out: VertexOutput;
    let x = i32(vertex_index) / 2;
    let y = i32(vertex_index) & 1;
    let tc = vec2<f32>(
        f32(x) * 2.0,
        f32(y) * 2.0
    );
    out.position = vec4<f32>(
        tc.x * 2.0 - 1.0,
        1.0 - tc.y * 2.0,
        0.000001, 1.0
    );
    out.world_position = global_params.inv_projection_view * out.position;
    out.world_position = out.world_position / out.world_position.w;
    out.uv = tc;
    return out;
}

@fragment
fn fs_forward_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let scene = Scene(vec3<f32>(0., 0., 0.));
    let planet = Planet(vec3<f32>(0.0, 0.0, 0.0), 6371e3, 6471e3);
    let dir = normalize(in.world_position.xyz - global_params.camera_position.xyz);

    // let color = get_sky_color(in.uv, global_params.camera_position.xyz, dir, scene, planet);
    let depth = (1. - textureSampleLevel(solids_screen_depth, default_sampler, in.uv, 0.)) * global_params.camera_far;
    let color = get_sky_color(depth, global_params.camera_position.xyz, dir);

    let color = 1.0 - exp(-color);

    return vec4<f32>(apply_fog(color, global_params.camera_position.xyz, in.world_position.xyz), 1.0);
}
