use ambient_api::{
    components::core::{
        app::{main_scene, window_physical_size},
    },
    prelude::*,
};

#[main]
fn main() {
    ambient_api::messages::Frame::subscribe(move |_| {
        let (delta, input) = input::get_delta();

        let window_size = entity::get_component(entity::resources(), window_physical_size()).unwrap();

        let msg = messages::Input {
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
            window_size: vec2(window_size.x as f32, window_size.y as f32),
        };
        msg.send_server_unreliable();
    });
}