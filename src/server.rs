use rand::Rng;
use ambient_api::{
    global::{time, frametime, Quat},
    components::core::{
        app::{main_scene, window_physical_size},
        camera::{aspect_ratio_from_window, fog, projection_view},
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
use components::{world_ref, voxel_world, voxel, player_camera_ref, player_camera_pitch, player_camera_yaw, player_camera_zoom, jumping, jump_timer, current_animation};

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
                .with_default(player_camera_pitch())
                .with_default(player_camera_yaw())
                .with(player_camera_zoom(), 10.0)
                .with_default(fog())
                .with(translation(), vec3(30., 25., 10.))
                .with_default(main_scene())
                .with(lookat_target(), vec3(0., 0., 0.))
                .spawn();

            println!("{}",id);

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

    messages::Input::subscribe(|source, msg| {
        let Some(player_id) = source.client_entity_id() else { return; };

        let camera_id = entity::get_component(player_id, player_camera_ref()).unwrap();
        let player_pos = entity::get_component(player_id, translation()).unwrap();
        let camera_pos = entity::get_component(camera_id, translation()).unwrap();

        // messages::CameraProjectionView::new(
        //     entity::get_component(camera_id, projection_view()).unwrap(),
        // ).send(Target::RemoteBroadcastReliable);

        let camera_zoom = entity::mutate_component(camera_id, player_camera_zoom(), |zoom| {
            let new = *zoom - msg.camera_zoom;
            if new > 1. && new < 50. {
                *zoom = new;
            }
        }).unwrap();

        let camera_pitch = entity::mutate_component(camera_id, player_camera_pitch(), |pitch| {
            // Calculate new pitch
            let new = *pitch + (msg.camera_rotation.y / 300.);
            // If pitch is within bounds, set it
            if new > -0.5 && new < 0.9 {
                *pitch = new;
            }
        }).unwrap();

        player_jump_controller(player_id, msg.jump);

        let camera_pitch_rotation = Quat::from_rotation_x(camera_pitch);
        // Get Camera Z Rotation
        let camera_yaw = entity::mutate_component(camera_id, player_camera_yaw(), |p| *p += msg.camera_rotation.x / 150.).unwrap();
        // let camera_yaw = 1.0;

        let camera_yaw_rotation = Quat::from_rotation_z(camera_yaw);
        // Apply rotations to find new camera position relative to player
        let camera_offset = camera_yaw_rotation.mul_vec3(camera_pitch_rotation.mul_vec3(Vec3::Y * camera_zoom));
        // Set new camera position. Make camera Z increase as you look down (better for building)
        entity::set_component(camera_id, translation(), player_pos + camera_offset + (Vec3::Z * 5.));
        entity::set_component(camera_id, lookat_target(), player_pos + Vec3::Z * 2.);

        let move_character = |direction| {
            let speed = 0.10;
            physics::move_character(player_id, camera_yaw_rotation.mul_vec3(direction).normalize_or_zero() * speed, 0.01, frametime());
            entity::set_component(player_id, rotation(), camera_yaw_rotation);
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

        // If none of the WASD are pressed, the the idle animation isn't already playing, set the animation to idle
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
