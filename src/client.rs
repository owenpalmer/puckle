use ambient_api::{
    components::core::{
        app::{main_scene, window_physical_size},
    },
    message::client::{MessageExt, Target},
    player::MouseButton,
    player::KeyCode,
    prelude::*,
};

#[main]
fn main() {
    messages::CameraProjectionView::subscribe(move |source, msg| {
        let (delta, input) = player::get_raw_input_delta();

        let window_size = entity::get_component(entity::resources(), window_physical_size()).unwrap();

        let ndc_x = (2.0 * input.mouse_position.x / window_size.x as f32) - 1.0;
        let ndc_y = 1.0 - (2.0 * input.mouse_position.y / window_size.y as f32);
        // println!("B: x:{}, y:{}", ndc_x, ndc_y);

        let clip_space_coords = vec4(ndc_x, ndc_y, -0.1, 1.0);
        let camera_projection_view = msg.camera_projection_view;
        let world_space_coords = camera_projection_view.inverse() * clip_space_coords;
        let world_space_position = vec3(world_space_coords.x, world_space_coords.y, world_space_coords.z) / world_space_coords.w;

        messages::MouseToWorld {
            world_space_position: world_space_position,
            left_mouse: delta.mouse_buttons.contains(&MouseButton::Left),
        }.send(Target::RemoteReliable);
    });

    on(event::FRAME, move |_| {
        let (delta, input) = player::get_raw_input_delta();

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
        msg.send(Target::RemoteReliable);
    });
}