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
    for (var i: i32=0; i < 1; i = i + 1) {
        let cam = shadow_cameras.cameras[i];
        let p = cam.viewproj * world_position;
        if (inside(p.xyz / p.w)) {
            return i;
        }
    }
    return 0;
}

fn fetch_shadow(light_angle: f32, world_position: vec4<f32>) -> f32 {
    for (var i: i32=0; i < 1; i = i + 1) {
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
// @module: Resources
// --------------------------------

@group(1)
@binding(0)
var<storage> mesh_metadatas: MeshMetadatas;

@group(1)
@binding(1)
var<storage> mesh_position: Vec3Buffer;
@group(1)
@binding(2)
var<storage> mesh_normal: Vec3Buffer;
@group(1)
@binding(3)
var<storage> mesh_tangent: Vec3Buffer;
@group(1)
@binding(4)
var<storage> mesh_texcoord0: Vec2Buffer;
@group(1)
@binding(5)
var<storage> mesh_joint: UVec4Buffer;
@group(1)
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

@group(1)
@binding(7)
var<storage> skins: Mat4x4Buffer;


// --------------------------------
// @module: Gizmo
// --------------------------------
struct Gizmo {
  model_matrix: mat4x4<f32>,
  color: vec3<f32>,
  corner: f32,
  scale: vec2<f32>,
  border_w: f32,
  corner_inner: f32,
};

struct GizmoBuffer {
  gizmos: array<Gizmo>,
};

@group(2)
@binding(0)
var<storage> gizmo_buffer: GizmoBuffer;

@group(2)
@binding(1)
var depth_buffer: texture_depth_2d;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) world_position: vec4<f32>,
  @location(1) uv: vec2<f32>,
  @location(2) @interpolate(flat) inst: u32,
  @location(3) ndc: vec3<f32>,
};

@vertex
fn vs_main(@builtin(instance_index) instance_index: u32,
@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
  let local_pos = get_raw_mesh_position(vertex_index);
  let uv = get_raw_mesh_uv(vertex_index);

  let gizmo = gizmo_buffer.gizmos[instance_index];

  let pos = gizmo.model_matrix * vec4<f32>(local_pos, 1.);
  let clip_pos = (global_params.projection_view * pos);

  let ndc = clip_pos.xyz / clip_pos.w;
  return VertexOutput(clip_pos, pos, uv, instance_index, ndc);
}

fn corner(radius: f32, uv: vec2<f32>, stretch: vec2<f32>, scale: vec2<f32>) -> bool {
  let pos_scale = max(scale, vec2<f32>(0., 0.));
  // Position rel center of quad
  let unstretched_mid = vec2<f32>(uv.x - 0.5, uv.y - 0.5) * 2.0 / pos_scale;
  let mid = unstretched_mid / stretch;

  // Pos to the nearest corner on quad
  let corner = vec2<f32>(sign(mid.x), sign(mid.y)) /  stretch;
  // The middle of the circle rounding the corner
  // let max_len = min(stretch.x, stretch.y);
  let aspect = (stretch.y / stretch.x);
  let short_side = max(stretch.x, stretch.y);
  let r = radius / short_side;
  let anchor = corner - vec2<f32>(sign(mid.x), sign(mid.y)) * r ;

  let rel = mid - anchor;
  let to_corner = corner - mid;

  let ang = dot(normalize(mid - anchor), normalize(corner - anchor));

  return length(mid - anchor) > r &&
    (ang > 0.707 || abs(unstretched_mid.x) > 1.0 || abs(unstretched_mid.y) > 1.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let gizmo = gizmo_buffer.gizmos[in.inst];

  let scale = gizmo.scale.xy;


  let uv = vec2<f32>(in.ndc.x * 0.5 + 0.5, 1.0 - (in.ndc.y * 0.5 + 0.5));

  let depth = in.ndc.z;

  let covered = textureSampleCompare(depth_buffer, shadow_sampler, uv, depth);
  // let depth = textureSample(depth_buffer, default_sampler, uv);

  let size = min(scale.x, scale.y);
  if corner(gizmo.corner, in.uv, scale, vec2<f32>(1.)) || !corner(gizmo.corner_inner, in.uv, scale,  (vec2<f32>(1.) - gizmo.border_w / scale.yx)) {
    discard;
  }

  return vec4<f32>(gizmo.color, 0.2 * covered + (1.0 - covered));
}
