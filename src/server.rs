use ambient_api::{
    global::{time, frametime, Quat},
    components::core::{
        app::main_scene,
        camera::aspect_ratio_from_window,
        model::{model_from_url},
        player::{player, user_id},
        primitives::{cube},
        transform::{lookat_center, translation, rotation, scale},
        rendering::{color, sky, sun},
        physics::{
            character_controller_height, character_controller_radius, physics_controlled,
            plane_collider, box_collider, visualizing, dynamic,
        },
    },
    player::KeyCode,
    concepts::{make_transformable, make_perspective_infinite_reverse_camera},
    prelude::*,
};
use components::{world_ref, voxel_world, voxel, player_camera_ref, player_camera_yaw, jumping, jump_timer};

#[main]
pub async fn main() -> EventResult {
    // For now, the world is only one chunk
    let world_id = Entity::new().with(voxel_world(), vec![]).spawn();

    let size: i32 = 16;
    let mut voxels_list = vec![];
    for i in 0..size.pow(3) {
        let [x,y,z] = [
            (i % size), //x
            ((i/size) % size), //y
            (i / (size*size)), //z
        ];
        // make floor
        if(z == 0){
            voxels_list.push(
                Entity::new()
                    .with_default(cube())
                    .with_default(voxel())
                    .with(world_ref(), world_id)
                    .with(scale(), Vec3::ONE)
                    .with(box_collider(), Vec3::ONE)
                    .with(translation(), vec3(x as f32, y as f32, z as f32))
                    .spawn()
            );
        } else if (z == 1 && x > 5 && x < 10) {
            voxels_list.push(
                Entity::new()
                    .with_default(cube())
                    .with_default(voxel())
                    .with(world_ref(), world_id)
                    .with(scale(), Vec3::ONE)
                    .with(box_collider(), Vec3::ONE)
                    .with(translation(), vec3(x as f32, y as f32, z as f32))
                    .spawn()
            );
        } else {
            voxels_list.push(
                Entity::new()
                    // .with_default(cube())
                    .with_default(voxel())
                    .with(world_ref(), world_id)
                    .with(scale(), Vec3::ONE)
                    // .with(box_collider(), Vec3::ONE)
                    .with(translation(), vec3(x as f32, y as f32, z as f32))
                    .spawn()
            );
        }
    }

    entity::set_component(world_id, voxel_world(), voxels_list);

    spawn_query((player(), user_id())).bind(move |players| {
        for (id, (_, user)) in players {
            let camera = Entity::new()
                .with_merge(make_perspective_infinite_reverse_camera())
                .with(aspect_ratio_from_window(), EntityId::resources())
                .with_default(player_camera_yaw())
                .with(translation(), vec3(30., 25., 10.))
                .with_default(main_scene())
                .with(lookat_center(), vec3(0., 0., 0.))
                .spawn();

            entity::add_components(
                id,
                Entity::new()
                    .with_merge(make_transformable())
                    .with_default(jumping())
                    .with_default(jump_timer())
                    .with(model_from_url(), asset_url("assets/player.glb").unwrap())
                    .with(scale(), Vec3::ONE * 0.5)
                    .with_default(cube())
                    .with(player_camera_ref(), camera)
                    .with(world_ref(), world_id)
                    .with(translation(), vec3(5.,5.,20.))
                    .with_default(physics_controlled())
                    .with(character_controller_height(), 0.9)
                    .with(character_controller_radius(), 0.3)
                    .with(color(), vec4(1.0, 0.0, 1., 1.))
            );
        }
    });

    query((player(), player_camera_ref()))
    .build()
    .each_frame(move |players| {
        for (player_id, (_, camera_id)) in players {
            let Some((delta, pressed)) = player::get_raw_input_delta(player_id) else { continue; };
            let speed = 0.1;

            let player_pos = entity::get_component(player_id, translation()).unwrap();
            let camera_pos = entity::get_component(camera_id, translation()).unwrap();
            let camera_zoom = 10.0;

            // Get Camera Y Rotation
            let new_camera_yaw = entity::mutate_component(camera_id, player_camera_yaw(), |yaw| {
                let new = *yaw + (delta.mouse_position.y / 300.);
                if new > -0.7 && new < 0.7 {
                    *yaw = new;
                }
            }).unwrap();

            let player_x_rotation = Quat::from_rotation_x(new_camera_yaw);

            // Get Camera Z Rotation
            let player_z_rotation = entity::mutate_component(player_id, rotation(), |p| *p *= Quat::from_rotation_z(delta.mouse_position.x / 150.)).unwrap();

            // Apply rotations
            let new_camera_offset = player_z_rotation.mul_vec3(player_x_rotation.mul_vec3(Vec3::Y * camera_zoom));
            entity::set_component(camera_id, translation(), player_pos + new_camera_offset);

            entity::set_component(camera_id, lookat_center(), player_pos);


            let is_jumping = entity::get_component(player_id, jumping()).unwrap();
            let jump_time = entity::mutate_component(player_id, jump_timer(), |x| *x += frametime()).unwrap();

            if jump_time > 0.2 {
                entity::set_component(player_id, jumping(), false);
            }
            if is_jumping {
                let jump_speed = (0.2 * f32::exp(-3. * jump_time));
                physics::move_character(player_id, Vec3::Z * jump_speed, 0.01, frametime());
            } else {
                physics::move_character(player_id, Vec3::Z * -0.3, 0.01, frametime());
            }
            if delta.keys.contains(&KeyCode::Space) && !is_jumping {
                entity::set_component(player_id, jumping(), true);
                entity::set_component(player_id, jump_timer(), 0.);
            }

            for (keycode, direction) in [
                (&KeyCode::W, -Vec3::Y),
                (&KeyCode::S, Vec3::Y),
                (&KeyCode::A, -Vec3::X),
                (&KeyCode::D, Vec3::X),
            ].iter() {
                if pressed.keys.contains(keycode) {
                    physics::move_character(player_id, player_z_rotation.mul_vec3(*direction).normalize_or_zero() * speed, 0.01, frametime());
                }
            }
        }
    });

    EventOk
}
