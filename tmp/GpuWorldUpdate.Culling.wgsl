// --------------------------------
// @module: GpuWorld
// --------------------------------

struct EntityLayoutBuffer { data: array<i32>, };
@group(0)
@binding(0)
var<storage> entity_layout: EntityLayoutBuffer;


struct EntityMat4Buffer { data: array<mat4x4<f32>> };

@group(0)
@binding(1)
var<storage, read_write> entity_Mat4_data: EntityMat4Buffer;

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



fn set_entity_data_Mat4(component_index: u32, entity_loc: vec2<u32>, value: mat4x4<f32>) {
    entity_Mat4_data.data[u32(get_entity_component_offset_Mat4(component_index, entity_loc)) + entity_loc.y] = value;
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


fn set_entity_mesh_to_world(entity_loc: vec2<u32>, value: mat4x4<f32>) {
set_entity_data_Mat4(0u, entity_loc, value);
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


fn set_entity_renderer_cameras_visible(entity_loc: vec2<u32>, value: mat4x4<f32>) {
set_entity_data_Mat4(1u, entity_loc, value);
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


fn set_entity_lod_cutoffs(entity_loc: vec2<u32>, value: mat4x4<f32>) {
set_entity_data_Mat4(2u, entity_loc, value);
}



struct EntityVec4Buffer { data: array<vec4<f32>> };

@group(0)
@binding(2)
var<storage, read_write> entity_Vec4_data: EntityVec4Buffer;

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



fn set_entity_data_Vec4(component_index: u32, entity_loc: vec2<u32>, value: vec4<f32>) {
    entity_Vec4_data.data[u32(get_entity_component_offset_Vec4(component_index, entity_loc)) + entity_loc.y] = value;
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


fn set_entity_world_bounding_sphere(entity_loc: vec2<u32>, value: vec4<f32>) {
set_entity_data_Vec4(0u, entity_loc, value);
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


fn set_entity_visibility_from(entity_loc: vec2<u32>, value: vec4<f32>) {
set_entity_data_Vec4(1u, entity_loc, value);
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


fn set_entity_color(entity_loc: vec2<u32>, value: vec4<f32>) {
set_entity_data_Vec4(2u, entity_loc, value);
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


fn set_entity_outline(entity_loc: vec2<u32>, value: vec4<f32>) {
set_entity_data_Vec4(3u, entity_loc, value);
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


fn set_entity_gpu_lod(entity_loc: vec2<u32>, value: vec4<f32>) {
set_entity_data_Vec4(4u, entity_loc, value);
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


fn set_entity_skin(entity_loc: vec2<u32>, value: vec4<f32>) {
set_entity_data_Vec4(5u, entity_loc, value);
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


fn set_entity_ui_size(entity_loc: vec2<u32>, value: vec4<f32>) {
set_entity_data_Vec4(6u, entity_loc, value);
}



struct EntityUVec4Array20Buffer { data: array<array<vec4<u32>, 20>> };

@group(0)
@binding(3)
var<storage, read_write> entity_UVec4Array20_data: EntityUVec4Array20Buffer;

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



fn set_entity_data_UVec4Array20(component_index: u32, entity_loc: vec2<u32>, value: array<vec4<u32>, 20>) {
    entity_UVec4Array20_data.data[u32(get_entity_component_offset_UVec4Array20(component_index, entity_loc)) + entity_loc.y] = value;
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


fn set_entity_primitives(entity_loc: vec2<u32>, value: array<vec4<u32>, 20>) {
set_entity_data_UVec4Array20(0u, entity_loc, value);
}




// --------------------------------
// @module: CullingParams
// --------------------------------

struct Plane {
    normal: vec3<f32>,
    distance: f32,
};

fn plane_distance(plane: Plane, position: vec3<f32>) -> f32 {
    return dot(plane.normal, position) + plane.distance;
}

struct Camera {
    view: mat4x4<f32>,
    position: vec4<f32>,
    frustum_right: Plane,
    frustum_top: Plane,
    orthographic_size: vec2<f32>,
    frustum_near: f32,
    frustum_far: f32,
    cot_fov_2: f32,
};

struct Params {
    main_camera: Camera,
    shadow_cameras: array<Camera, 6>,
    lod_cutoff_scaling: f32,
};

@group(1)
@binding(0)
var<uniform> params: Params;

struct CameraCullResult {
    fully_contained: bool,
    inside: bool,
};

fn cull_camera(camera: Camera, bounding_sphere: vec4<f32>) -> CameraCullResult {
    var res: CameraCullResult;

	let center = (camera.view * vec4<f32>(bounding_sphere.xyz, 1.)).xyz;
	let radius = bounding_sphere.w;

    let sphere_mirrored = vec3<f32>(abs(center.xy), center.z);

    let top_dist = plane_distance(camera.frustum_top, sphere_mirrored);
    let right_dist = plane_distance(camera.frustum_right, sphere_mirrored);

    res.inside = !(top_dist > radius) &&
        !(right_dist > radius) &&

        center.z + radius > camera.frustum_near &&
        center.z - radius < camera.frustum_far;

    res.fully_contained = !(top_dist > -radius) &&
        !(right_dist > -radius) &&

        center.z - radius > camera.frustum_near &&
        center.z + radius < camera.frustum_far;

    return res;
}

fn get_lod(entity_loc: vec2<u32>) -> u32 {

    let bounding_sphere = get_entity_world_bounding_sphere(entity_loc);
	let radius = bounding_sphere.w;

    var lod_cutoffs = get_entity_lod_cutoffs(entity_loc) ;

    let dist = length(params.main_camera.position.xyz - bounding_sphere.xyz);
    let clip_space_radius = radius * params.main_camera.cot_fov_2 / dist;
    for (var i=0u; i < 4u; i = i + 1u) {
        for (var j=0u; j < 4u; j = j + 1u) {
            if (clip_space_radius >= lod_cutoffs[i][j] * params.lod_cutoff_scaling) {
                return i * 4u + j;
            }
        }
    }

    return 0u;
}

fn update(entity_loc: vec2<u32>) {
    if (has_entity_gpu_lod(entity_loc)) {
        set_entity_gpu_lod(entity_loc, vec4<f32>(f32(get_lod(entity_loc)), 0.0, 0.0, 0.0));
    }
    var cameras: mat4x4<f32>;
    let bounding_sphere = get_entity_world_bounding_sphere(entity_loc);
    cameras[0][0] = f32(cull_camera(params.main_camera, bounding_sphere).inside);

    for (var i=1u; i <= 5u; i = i + 1u) {
        let a = i >> 2u;
        let b = i & 3u;

        cameras[a][b] = 0.0;
    }
    for (var i=0u; i < 5u; i = i + 1u) {
        let radius = bounding_sphere.w;
        let pixel_size = vec2<f32>(radius) * 2. / params.shadow_cameras[i].orthographic_size;
        let min_pixel_size = min(pixel_size.x, pixel_size.y);

        if (min_pixel_size < 0.01) {
            break;
        }

        let res = cull_camera(params.shadow_cameras[i], bounding_sphere);
        if (res.inside) {
            let a = (i + 1u) >> 2u;
            let b = (i + 1u) & 3u;
            cameras[a][b] = 1.0;
        }
        if (res.fully_contained) {
            break;
        }
    }
    set_entity_renderer_cameras_visible(entity_loc, cameras);
}


// --------------------------------
// @module: GpuWorldUpdate
// --------------------------------

struct GpuUpdateChunksBuffer {
    data: array<vec4<u32>>,
};

@group(2)
@binding(0)
var<storage, read> gpu_update_chunks: GpuUpdateChunksBuffer;

@compute
@workgroup_size(32)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let chunk = gpu_update_chunks.data[global_id.y];
    let entity_loc = vec2<u32>(chunk.x, chunk.y + global_id.x);
    if (entity_loc.y < chunk.z) {
        update(entity_loc);
    }
}
