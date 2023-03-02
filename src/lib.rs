use ambient_api::{
    global::{frametime, State, Quat},
    components::core::{
        game_objects::player_camera,
        player::{player, user_id},
        primitives::{cube},
        transform::{lookat_center, translation, rotation, scale},
        rendering::color,
    },
    player::KeyCode,
    concepts::{make_transformable, make_perspective_infinite_reverse_camera},
    prelude::*,
    rand,
};
use components::{timer, voxel_update, voxel_direction, player_camera_ref, player_camera_yaw};
use concepts::{make_voxel};
use std::cell::{RefCell};

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

#[main]
pub async fn main() -> EventResult {
    let voxel_world = RefCell::new(VoxelMap::new(vec![40,40,40]));

    Entity::new()
        .with_merge(make_transformable())
        .with(translation(), vec3(19.5,19.5,-1.))
        .with_default(cube())
        .with(scale(), vec3(40.,40.,1.))
        .with(color(), vec4(0.1, 0.1, 0.1, 1.))
        .spawn();

    for x in 10..30 {
        for y in 10..30 {
            make_voxel()
                .with_merge(make_transformable())
                .with(translation(), vec3(x as f32, y as f32, 1.0))
                .with_default(cube())
                .with(color(), vec4(1., 0.9, 0., 1.))
                .spawn();
        }
    }
    for x in 12..25 {
        for y in 11..22 {
            make_voxel()
                .with_merge(make_transformable())
                .with(translation(), vec3(x as f32, y as f32, 1.0))
                .with_default(cube())
                .with(color(), vec4(0.5,1.,0.1, 1.0))
                .spawn();
        }
    }

    spawn_query((player(), user_id())).bind(move |players| {
        for (id, (_, user)) in players {
            let camera = Entity::new()
                .with_merge(make_perspective_infinite_reverse_camera())
                .with_default(player_camera())
                .with_default(player_camera_yaw())
                .with(user_id(), user)
                .with(translation(), vec3(14.5, 14.5, 10.))
                .with_default(rotation())
                .with(lookat_center(), vec3(0.,0.,1.))
                .spawn();

            entity::add_components(
                id,
                make_voxel()
                    .with_merge(make_transformable())
                    .with_default(translation())
                    .with_default(cube())
                    .with(player_camera_ref(), camera)
                    .with(color(), vec4(0., 0.9, 0., 1.))
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

            if world.is_available(destination_request- Vec3::Z) {
                entity::set_component(voxel_id, voxel_update(), -Vec3::Z)
            }
        }
    });

    query((player(), timer(), player_camera_ref()))
    .build()
    .each_frame(move |players| {
        for (player_id, (_, timer_id, camera_id)) in players {
            let Some((delta, pressed)) = player::get_raw_input_delta(player_id) else { continue; };

            let timer_duration = entity::get_component(player_id, timer()).unwrap();
            let delay = 0.075;
            let speed = 0.1;

            let camera_speed = 1.1;
            let player_pos = entity::get_component(player_id, translation()).unwrap();

            // Get Camera Y Rotation
            let camera_yaw = entity::mutate_component(camera_id, player_camera_yaw(), |p| *p += delta.mouse_position.y / 150.).unwrap();
            let player_y_rotation = Quat::from_rotation_y(camera_yaw);

            // Get Camera Z Rotation
            let player_direction = entity::mutate_component(player_id, voxel_direction(), |p| *p += delta.mouse_position.x / 150.).unwrap();
            let player_z_rotation = Quat::from_rotation_z(player_direction);

            // Apply rotations
            let camera_offset = vec3(0.,10.,5.);
            let new_camera_pos = player_z_rotation.mul_vec3(player_y_rotation.mul_vec3(camera_offset));
            entity::set_component(camera_id, translation(), player_pos + new_camera_pos);

            let current_lookat = entity::get_component(camera_id, lookat_center()).unwrap();
            entity::set_component(camera_id, lookat_center(), current_lookat.lerp(player_pos, camera_speed));
            entity::set_component(camera_id, lookat_center(), player_pos);

            for (keycode, direction) in [
                (&KeyCode::W, -Vec3::Y),
                (&KeyCode::S, Vec3::Y),
                (&KeyCode::A, -Vec3::X),
                (&KeyCode::D, Vec3::X),
            ].iter() {
                let translate = || {
                    entity::set_component(player_id, voxel_update(), player_z_rotation.mul_vec3(*direction));
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
