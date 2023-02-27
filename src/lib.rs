use ambient_api::{
    global::{frametime, State},
    components::core::{
        game_objects::player_camera,
        player::{player, user_id},
        primitives::{cube},
        transform::{lookat_center, translation, scale},
        rendering::color,
    },
    player::KeyCode,
    concepts::{make_transformable, make_perspective_infinite_reverse_camera},
    prelude::*,
};
use components::{timer, voxel_position };
use std::cell::RefCell;

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
    fn is_available(&self, x: usize, y: usize, z: usize) -> bool {
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
    fn set(&mut self, x: usize, y: usize, z: usize, value: Option<EntityId>) {
        self.voxels[x][y][z] = value;
    }
}

#[main]
pub async fn main() -> EventResult {

    let voxel_world = RefCell::new(VoxelMap::new(vec![10,10,10]));

    Entity::new()
        .with_merge(make_transformable())
        .with(translation(), vec3(4.5,4.5,0.))
        .with_default(cube())
        .with(scale(), vec3(10.,10.,1.))
        .with(color(), vec4(0.1, 0.1, 0.1, 1.))
        .spawn();

    Entity::new()
        .with_merge(make_transformable())
        .with(translation(), vec3(0.,6.,1.,))
        .with_default(cube())
        .with(timer(), 0.)
        .with(color(), vec4(1., 0.9, 0., 1.))
        .with(voxel_position(), vec3(0.,7.,1.))
        .spawn();

    let spawn_callback = move |players| {
        for (id, (_, user)) in players {
            let camera = Entity::new()
                .with_merge(make_perspective_infinite_reverse_camera())
                .with_default(player_camera())
                .with(user_id(), user)
                .with(translation(), vec3(14.5, 14.5, 10.))
                .with(lookat_center(), vec3(0., 0., 0.))
                .spawn();

            entity::add_components(
                id,
                Entity::new()
                    .with_merge(make_transformable())
                    .with_default(translation())
                    .with_default(cube())
                    .with(timer(), 0.)
                    .with(voxel_position(), vec3(0.,0.,1.))
                    .with(color(), vec4(0., 0.9, 0., 1.))
            );
        }
    };
    spawn_query((player(), user_id())).bind(spawn_callback);

    change_query(voxel_position()).requires(translation()).track_change(voxel_position()).bind(move |voxel_entities| {
        for (voxel_id, _) in voxel_entities {
            let new_pos = entity::get_component(voxel_id, voxel_position()).unwrap();
            let old_pos = entity::get_component(voxel_id, translation()).unwrap();

            // any values are negative
            if new_pos.abs() != new_pos {
                return;
            }

            let new_point = new_pos.to_array().map(|e| {e.ceil() as usize});
            let old_point = old_pos.to_array().map(|e| {e.ceil() as usize});

            let mut world = voxel_world.borrow_mut();

            if world.is_available(new_point[0], new_point[1], new_point[2]) {
                entity::set_component(voxel_id, translation(), new_pos);
                world.set(old_point[0], old_point[1], old_point[2], None);
                world.set(new_point[0], new_point[1], new_point[2], Some(voxel_id));
            }
        }
    });

    query((player(), timer()))
    .build()
    .each_frame(move |players| {
        for (player_id, (_, timer_id)) in players {
            let Some((delta, pressed)) = player::get_raw_input_delta(player_id) else { continue; };

            let timer_duration = entity::get_component(player_id, timer()).unwrap();
            let player_position = entity::get_component(player_id, translation()).unwrap();
            let delay = 0.1;

            for (keycode, direction) in [
                (&KeyCode::W, -Vec3::Y),
                (&KeyCode::S, Vec3::Y),
                (&KeyCode::A, -Vec3::X),
                (&KeyCode::D, Vec3::X),
            ].iter() {
                let translate = || {
                    entity::set_component(player_id, voxel_position(), (*direction * 1.0) + player_position);
                    entity::set_component(player_id, timer(), 0.0);
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
