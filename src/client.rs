use ambient_api::{
    components::core::{
        app::{main_scene, window_physical_size},
    },
    prelude::*,
};

// use components::{player_camera_ref};
use components::{world_ref, voxel_world, voxel, player_camera_ref, player_camera_pitch, player_camera_yaw, player_camera_zoom, jumping, jump_timer, current_animation};

#[main]
fn main() {
    let mut cursor_lock = input::CursorLockGuard::new(true);
    ambient_api::messages::Frame::subscribe(move |_| {
        let (delta, input) = input::get_delta();
        let user_id = entity::get_component(entity::resources(), local_user_id()).unwrap();
        let player_id = player::get_by_user_id(&user_id).unwrap();

        if !cursor_lock.auto_unlock_on_escape(&input) {
            return;
        }

        let window_size = entity::get_component(entity::resources(), window_physical_size()).unwrap();

        let mut msg = messages::Input {
            left_click: delta.mouse_buttons.contains(&MouseButton::Left),
            w: input.keys.contains(&KeyCode::W),
            a: input.keys.contains(&KeyCode::A),
            s: input.keys.contains(&KeyCode::S),
            d: input.keys.contains(&KeyCode::D),
            jump: delta.keys.contains(&KeyCode::Space),
            no_wasd: [&KeyCode::W, &KeyCode::A, &KeyCode::S, &KeyCode::D].iter().all(|c| !input.keys.contains(c)),
            wasd_delta: [&KeyCode::W, &KeyCode::A, &KeyCode::S, &KeyCode::D].iter().any(|c| delta.keys.contains(c)),
            camera_rotation: delta.mouse_position,
            camera_zoom: delta.mouse_wheel,
            mouse_position: input.mouse_position,
            ray_origin: Vec3::ZERO,
            ray_dir: Vec3::ZERO,
        };

        if let Some(camera_id) = entity::get_component(player_id, player_camera_ref()) {
            let ray = camera::screen_to_world_direction(camera_id, input.mouse_position);
            msg.ray_origin = ray.origin;
            msg.ray_dir = ray.dir;
        }
        msg.send_server_unreliable();
    });
}