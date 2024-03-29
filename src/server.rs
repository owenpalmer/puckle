use rand::Rng;
use ambient_api::{
    global::{time, frametime, Quat},
    components::core::{
        app::{main_scene, window_physical_size},
        camera::*,
        prefab::prefab_from_url,
        model::model_from_url,
        player::{player, user_id},
        primitives::{cube},
        transform::{lookat_target, translation, rotation, scale},
        rendering::{color, sky, sun, water, fog_density, fog_height_falloff},
        physics::{
            character_controller_height, character_controller_radius, physics_controlled,
            plane_collider, box_collider, visualizing, dynamic,
        },
    },
    // player::KeyCode,
    concepts::{make_transformable, make_perspective_infinite_reverse_camera},
    entity::{AnimationAction, AnimationController},
    prelude::*,
    // message::server::{MessageExt, Source, Target},
    physics::{raycast_first},
};
use components::*;

use camera::CameraState;
mod camera;

#[main]
pub async fn main() -> ResultEmpty {
    // For now, the world is only one chunk
    let world_id = Entity::new().with(voxel_world(), vec![]).spawn();

    let mut rng = rand::thread_rng();
    let mut voxels_list = vec![];

    let chunks = vox_format::from_slice(include_bytes!("../assets/island_level.vox")).expect("Could not get voxel data");
    for model in chunks.models {
        for voxel in model.voxels {
            let point = voxel.point;
            let random = rng.gen_range(0.9..1.0);
            let voxel_color = match u8::from(voxel.color_index) {
                174 => vec4(0.3, 0.2, 0.1, 1.0),
                138 => vec4(0.12, 0.12, 0.1, 1.0),
                253 => vec4(0.17, 0.25, 0.0, 1.0),
                250 => vec4(0.2, 0.2, 0.2, 1.0),
                96 => vec4(0.4, 0.3, 0.2 * random, 1.0),
                9 => vec4(0.9, 0.5 * random, 0.1, 1.0),
                _ => vec4(1.0, 0.0, 0.0, 1.0),
            };
            Entity::new()
                .with_merge(make_transformable())
                .with(translation(), vec3(point.x as f32, point.y as f32, (point.z as f32) - 7.))
                .with_default(cube())
                .with(box_collider(), Vec3::ONE)
                .with(color(), voxel_color)
                .spawn();
        }
    }
    entity::set_component(world_id, voxel_world(), voxels_list);

    // Water
    Entity::new()
        .with_merge(make_transformable())
        .with_default(water())
        .with(scale(), Vec3::ONE * 1000.)
        .spawn();

    // Spawn sky and sun
    make_transformable().with_default(sky()).spawn();
    Entity::new()
        .with_merge(make_transformable())
        .with_default(sun())
        .with(rotation(), Quat::from_rotation_y(-0.2))
        .with_default(main_scene())
        .with(fog_density(), 0.05)
        .spawn();

    spawn_query((player(), user_id())).bind(move |players| {
        for (id, (_, user)) in players {
            let camera = Entity::new()
                .with_merge(make_perspective_infinite_reverse_camera())
                .with(aspect_ratio_from_window(), entity::resources())
                .with_default(camera_pitch())
                .with_default(camera_yaw())
                .with(camera_zoom(), -10.0)
                .with(fovy(), 1.)
                .with_default(fog())
                .with(camera_mode(), 0)
                .with(user_id(), user.clone())
                .with_default(main_scene())
                .spawn();
            
            println!("server player: {}", id);

            entity::add_components(
                id,
                Entity::new()
                    .with_merge(make_transformable())
                    .with_default(jumping())
                    .with_default(jump_timer())
                    .with_default(current_animation())
                    .with(
                        prefab_from_url(),
                        asset::url("assets/greg.fbx").unwrap(),
                    )
                    .with(scale(), Vec3::ONE * 0.5)
                    .with_default(cube())
                    .with(player_camera_ref(), camera)
                    .with(world_ref(), world_id)
                    .with(translation(), vec3(20.,20.,40.))
                    .with_default(physics_controlled())
                    .with(character_controller_height(), 0.9)
                    .with(character_controller_radius(), 0.3)
                    .with(color(), vec4(1.0, 0.0, 1., 1.))
            );
        }
    });

    // TODO: find better way of doing this :)
    fn player_jump_controller(player_id: EntityId, jump_key_delta: bool) {
        let is_jumping = entity::get_component(player_id, jumping()).unwrap();
        let jump_time = entity::mutate_component(player_id, jump_timer(), |x| *x += frametime()).unwrap();
        // If the player is past the jump time, start falling
        if jump_time > 0.2 {
            entity::set_component(player_id, jumping(), false);
        }
        if is_jumping {
            physics::move_character(player_id, Vec3::Z * 0.10, 0.01, frametime());
        } else {
            physics::move_character(player_id, Vec3::Z * -0.3, 0.01, frametime());
        }
        if jump_key_delta && !is_jumping && jump_time > 0.2 {
            entity::set_component(player_id, jumping(), true);
            entity::set_component(player_id, jump_timer(), 0.);
        }
    }

    fn add_voxel(pos: Vec3) {
        Entity::new()
            .with_merge(make_transformable())
            .with(translation(), pos)
            .with_default(cube())
            .with(box_collider(), Vec3::ONE)
            .with(color(), vec4(0.3, 0.3, 0.3, 1.))
            .spawn();
    }

    messages::CameraMode::subscribe(move|source, msg| {
        entity::set_component(msg.camera_id, camera_mode(), msg.mode);
        if msg.mode == 1{
            entity::set_component(msg.camera_id, fovy(), 0.10);
            entity::set_component(msg.camera_id, camera_zoom(), -200.);
        } else {
            entity::set_component(msg.camera_id, fovy(), 1.);
            entity::set_component(msg.camera_id, camera_zoom(), -10.);
        }
        if msg.mode == 2 {
            entity::set_component(msg.camera_id, fovy(), 1.7);
            entity::set_component(msg.camera_id, camera_zoom(), 1.);
        }
    });

    messages::Input::subscribe(move|source, msg| {
        let Some(player_id) = source.clone().client_entity_id() else { return; };
        let Some(user_id) = source.client_user_id() else { return; };

        if msg.left_click {
            if let Some(hit) = raycast_first(msg.ray_origin, msg.ray_dir) {
                entity::despawn(hit.entity);
            }
        }
        if msg.right_click {
            if let Some(hit) = raycast_first(msg.ray_origin, msg.ray_dir) {
                add_voxel((hit.position - (msg.ray_dir * 0.05)).round());
            }
        }

        let camera_id = entity::get_component(player_id, player_camera_ref()).unwrap();
        let player_pos = entity::get_component(player_id, translation()).unwrap();
        let camera_pos = entity::get_component(camera_id, translation()).unwrap();
        let camera_mode = entity::get_component(camera_id, camera_mode()).unwrap();
        
        let camera_state = CameraState(camera_id);

        if camera_mode == 1 {
            camera_state
                .zoom(msg.camera_zoom * 10.)
                .isometric_view(player_pos);
        } else {
            camera_state
                .rotate(msg.camera_rotation)
                .zoom(msg.camera_zoom)
                .translate_around_origin(player_pos + Vec3::Z * 2.);
        }

        player_jump_controller(player_id, msg.jump);

        let move_character = move |direction| {
            let speed = 0.10;
            let player_direction = Quat::from_rotation_z(camera_state.get_yaw());
            entity::set_component(player_id, rotation(), player_direction);
            physics::move_character(player_id, player_direction.mul_vec3(direction).normalize_or_zero() * speed, 0.01, frametime());
        };

        let set_animation = |name| {
            entity::set_component(player_id, current_animation(), String::from(name));
            entity::set_animation_controller(
                player_id,
                AnimationController {
                    actions: &[AnimationAction {
                        clip_url: &asset::url(format!("assets/greg.fbx/animations/{}.anim", name)).unwrap(),
                        looping: true,
                        weight: 1.,
                    }],
                    apply_base_pose: false,
                },
            );
        };

        // // If none of the WASD are pressed, the the idle animation isn't already playing, set the animation to idle
        if msg.no_wasd {
            if entity::get_component(player_id, current_animation()).unwrap() != String::from("Idle") {
                set_animation("Idle");
            }
        } 
        if msg.wasd_delta {
            set_animation("Running");
        }

        if msg.w {
            move_character(-Vec3::Y)
        }
        if msg.a {
            move_character(-Vec3::X)
        }
        if msg.s {
            move_character( Vec3::Y)
        }
        if msg.d {
            move_character( Vec3::X)
        }
    });

    OkEmpty
}
