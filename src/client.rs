use ambient_api::{
    components::core::{
        app::{main_scene, window_physical_size, window_logical_size},
        camera::*
    },
    prelude::*,
};

use components::*;

#[element_component]
fn App(_hooks: &mut Hooks) -> Element {
    let window_size = entity::get_component(entity::resources(), window_logical_size()).unwrap();
    let center_x = window_size.x as f32 / 2.;
    let center_y = window_size.y as f32 / 2.;
    Group::el([
        Line.el()
            .with(line_from(), vec3(center_x - 10., center_y, 0.))
            .with(line_to(), vec3(center_x + 10., center_y, 0.))
            .with(line_width(), 2.)
            .with(background_color(), vec4(1., 1., 1., 1.)),
        Line.el()
            .with(line_from(), vec3(center_x, center_y - 10., 0.))
            .with(line_to(), vec3(center_x, center_y + 10., 0.))
            .with(line_width(), 2.)
            .with(background_color(), vec4(1., 1., 1., 1.)),
    ])
}

#[main]
fn main() {
    App.el().spawn_interactive();
    let mut cursor_lock = input::CursorLockGuard::new(true);
    ambient_api::messages::Frame::subscribe(move |_| {
        let (delta, input) = input::get_delta();
        let user_id = entity::get_component(entity::resources(), local_user_id()).unwrap();
        let player_id = player::get_by_user_id(&user_id).unwrap();
        let Some(camera_id) = entity::get_component(player_id, player_camera_ref()) else { return; };
        let Some(camera_mode) = entity::get_component(camera_id, camera_mode()) else { return; };

        let mut screen_pos = camera::screen_to_clip_space(input.mouse_position);
        if camera_mode == 0  || camera_mode == 2 {
            screen_pos = Vec2::ZERO;
        }
        let ray = camera::clip_space_ray(camera_id, screen_pos);

        // 0 = Third Person
        // 1 = Build Mode
        // 2 = First Person
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
            right_click: delta.mouse_buttons.contains(&MouseButton::Right),
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