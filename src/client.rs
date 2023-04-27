use ambient_api::{
    components::core::{
        app::{main_scene, window_physical_size},
        camera::*
    },
    prelude::*,
};

use components::*;

#[main]
fn main() {
    let mut cursor_lock = input::CursorLockGuard::new(true);
    ambient_api::messages::Frame::subscribe(move |_| {
        let (delta, input) = input::get_delta();
        let user_id = entity::get_component(entity::resources(), local_user_id()).unwrap();
        let player_id = player::get_by_user_id(&user_id).unwrap();
        let Some(camera_id) = entity::get_component(player_id, player_camera_ref()) else { return; };

        let ray = camera::screen_to_world_direction(camera_id, input.mouse_position);

        // 0 = First Person
        // 1 = Build Mode
        // 2 = Third Person
        let mode = [&KeyCode::Key1, &KeyCode::Key2, &KeyCode::Key3]
            .iter()
            .position(|c| delta.keys.contains(c));

        if let Some(mode) = mode {
            let mode = mode as u32;
            cursor_lock.set_locked(true);
            if mode == 1 {
                cursor_lock.set_locked(false);
            }
            messages::CameraMode {
                mode,
                camera_id
            }.send_server_unreliable();
        }

        let msg = messages::Input {
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
            ray_origin: ray.origin,
            ray_dir: ray.dir,
        };

        msg.send_server_unreliable();
    });
}