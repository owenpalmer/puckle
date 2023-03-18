use ambient_api::{
    global::{time, frametime, Quat},
    components::core::{
        model::{model_from_url},
        game_objects::player_camera,
        player::{player, user_id},
        primitives::{cube},
        transform::{lookat_center, translation, rotation, scale},
        rendering::{color, sky, sun},
        physics::{box_collider, visualizing}
    },
    physics::{raycast_first},
    player::{KeyCode, MouseButton},
    concepts::{make_transformable, make_perspective_infinite_reverse_camera},
    prelude::*,
};
use components::{timer, voxel_update, voxel_direction, player_camera_ref, player_camera_yaw, player_follower, has_gravity, player_mesh_ref, player_builder_ref};
use concepts::{make_voxel};
use std::cell::{RefCell};
use vox_format;

const VOX_DATA: &[u8] = include_bytes!("../assets/level.vox");

#[derive(Debug)]
struct VoxelMap {
    voxels: Vec<Vec<Vec<Option<EntityId>>>>,
    size: Vec<usize>,
}

impl VoxelMap {
    fn new(size: Vec<usize>) -> Self {
        Self {
            voxels: vec![vec![vec![None; size[2]]; size[1]]; size[0]],
            size: size,
        }
    }
    fn is_available(&self, vec3: Vec3) -> bool {
        let rounded = vec3.round();
        if rounded.abs() != rounded { return false; }
        let [x,y,z] = self.to_usize(rounded.to_array());
        if !self.in_bounds(x,y,z) { 
            if self.voxels[x][y][z].is_none() {
                return true;
            }
        }
        return false;
    }
    fn in_bounds(&self, x: usize, y: usize, z: usize) -> bool {
        x >= self.size[0] || y >= self.size[1] || z >= self.size[2] || x < 0 || y < 0 || z < 0
    }
    fn set(&mut self, vec3: Vec3, value: Option<EntityId>) {
        let [x,y,z] = self.to_usize(vec3.round().to_array());
        self.voxels[x][y][z] = value;
    }
    fn to_usize(&self, array: [f32; 3]) -> [usize; 3] {
        [
            array[0] as usize,
            array[1] as usize,
            array[2] as usize,
        ]
    }
}

fn load_level() {
    make_transformable().with_default(sun()).spawn();
    make_transformable().with_default(sky()).spawn();

    // Entity::new()
    //     .with_merge(make_transformable())
    //     .with(scale(), Vec3::ONE * 10.)
    //     .with(model_from_url(), asset_url("assets/level.glb").unwrap())
    //     .spawn();

    let chunks = vox_format::from_slice(VOX_DATA).expect("Could not get voxel data");
    for model in chunks.models {
        for voxel in model.voxels {
            let point = voxel.point;
            make_voxel()
                .with_merge(make_transformable())
                .with(box_collider(), Vec3::ONE)
                .with(translation(), vec3(point.x as f32, point.y as f32, point.z as f32))
                .with_default(cube())
                .with(color(), vec4(0.3, 1., 0., 1.))
                .spawn();
        }
    }
}

#[main]
pub async fn main() -> EventResult {
    let voxel_world = RefCell::new(VoxelMap::new(vec![40,40,40]));
    load_level();

    spawn_query((player(), user_id())).bind(move |players| {
        for (id, (_, user)) in players {
            let camera = Entity::new()
                .with_merge(make_perspective_infinite_reverse_camera())
                .with_default(player_camera())
                .with_default(player_camera_yaw())
                .with(user_id(), user)
                .with(translation(), vec3(14.5, 14.5, 10.))
                // .with_default(rotation())
                .with(lookat_center(), vec3(39., 39., 1.))
                .spawn();

            let buidler_box = Entity::new()
                .with_merge(make_transformable())
                .with_default(cube())
                .with(color(), vec4(0.5, 0.5, 1.0, 0.5))
                .spawn();

            let player_mesh = Entity::new()
                .with_merge(make_transformable())
                .with(model_from_url(), asset_url("assets/player.glb").unwrap())
                .with(translation(), vec3(39., 39., 1.))
                .with(scale(), Vec3::ONE * 0.5)
                .spawn();

            entity::add_components(
                id,
                make_voxel()
                    .with_merge(make_transformable())
                    .with(translation(), vec3(39., 39., 1.))
                    // .with_default(cube())
                    .with(player_camera_ref(), camera)
                    .with(player_follower(), vec3(50.,50.,50.))
                    .with(player_mesh_ref(), player_mesh)
                    .with(player_builder_ref(), buidler_box)
                    .with(has_gravity(), true)
                    .with(color(), vec4(0.1, 0.2, 1., 1.))
            );
        }
    });

    change_query(voxel_update()).requires((translation(), timer())).track_change(voxel_update()).bind(move |voxel_entities| {
        let mut number = 0;
        for (voxel_id, _) in voxel_entities {
            let direction = entity::get_component(voxel_id, voxel_update()).unwrap();
            let voxel_position = entity::get_component(voxel_id, translation()).unwrap();
            let mut destination_request = voxel_position + direction.normalize_or_zero();

            let mut world = voxel_world.borrow_mut();
            if world.is_available(destination_request) {
            } else if world.is_available(destination_request + Vec3::Z) {
                destination_request += Vec3::Z;
            } else {
                return;
            }
            entity::set_component(voxel_id, translation(), destination_request.round());
            entity::set_component(voxel_id, timer(), 0.0);
            world.set(voxel_position, None);
            world.set(destination_request, Some(voxel_id));

            let has_gravity = entity::get_component(voxel_id, has_gravity()).unwrap();
            if has_gravity {
                if world.is_available(destination_request - Vec3::Z) {
                    entity::set_component(voxel_id, voxel_update(), -Vec3::Z)
                }
            }
        }
    });

    query((player(), timer(), player_camera_ref(), player_mesh_ref(), player_builder_ref()))
    .build()
    .each_frame(move |players| {
        for (player_id, (_, timer_id, camera_id, mesh_id, builder)) in players {
            let Some((delta, pressed)) = player::get_raw_input_delta(player_id) else { continue; };

            let timer_duration = entity::get_component(player_id, timer()).unwrap();
            let delay = 0.075;

            // If the game has been running for less than 3 seconds, make the camera move slowly
            let mut follower_speed = 0.15;
            if time() < 3. {
                follower_speed = 0.015;
            }

            let player_pos = entity::get_component(player_id, translation()).unwrap();
            let camera_pos = entity::get_component(camera_id, translation()).unwrap();
            let follower_pos = entity::mutate_component(player_id, player_follower(), |f| *f = f.lerp(player_pos, follower_speed)).unwrap();
            entity::set_component(mesh_id, translation(), player_pos);

            // Get Camera Y Rotation
            let new_camera_yaw = entity::mutate_component(camera_id, player_camera_yaw(), |yaw| {
                let new = *yaw + (delta.mouse_position.y / 300.);
                if new > -0.7 && new < 0.7 {
                    *yaw = new;
                }
            }).unwrap();

            let player_x_rotation = Quat::from_rotation_x(new_camera_yaw);

            // Get Camera Z Rotation
            let player_direction = entity::mutate_component(player_id, voxel_direction(), |p| *p += delta.mouse_position.x / 150.).unwrap();
            let player_z_rotation = Quat::from_rotation_z(player_direction);

            // Apply rotations
            let camera_offset = vec3(0.,10.,5.);
            let new_camera_offset = player_z_rotation.mul_vec3(player_x_rotation.mul_vec3(camera_offset));
            entity::set_component(camera_id, translation(), follower_pos + new_camera_offset);

            entity::set_component(camera_id, lookat_center(), follower_pos);

            let ray_direction = follower_pos - camera_pos;
            let block = raycast_first(camera_pos + camera_pos.cross(Vec3::Y * 0.1), ray_direction.normalize_or_zero());
            match block {
                Some(value) => {
                    let block_placement = (value.position - (ray_direction * 0.1)).round();
                    entity::set_component(builder, translation(), block_placement);
                    if delta.mouse_buttons.contains(&MouseButton::Left) {
                        make_voxel()
                            .with(translation(), block_placement)
                            .with_default(cube())
                            .with(box_collider(), Vec3::ONE * 0.5)
                            .with(color(), vec4(1.0, 0., 0., 1.0))
                            .spawn();
                    }
                    if delta.mouse_buttons.contains(&MouseButton::Right) {
                        entity::despawn(value.entity);
                    }
                }
                None => return,
            }

            for (keycode, direction) in [
                (&KeyCode::W, -Vec3::Y),
                (&KeyCode::S, Vec3::Y),
                (&KeyCode::A, -Vec3::X),
                (&KeyCode::D, Vec3::X),
            ].iter() {
                let translate = || {
                    entity::set_component(player_id, voxel_update(), player_z_rotation.mul_vec3(*direction));
                    entity::set_component(mesh_id, rotation(), Quat::from_rotation_z((player_direction / std::f32::consts::FRAC_PI_2).round() * std::f32::consts::FRAC_PI_2));
                };
                if delta.keys.contains(keycode) {
                    translate();
                }
                if pressed.keys.contains(keycode) {
                    if timer_duration > delay {
                        translate();
                    }
                    entity::mutate_component(player_id, timer(), |x| *x += frametime());
                }
            }
        }
    });

    EventOk
}
