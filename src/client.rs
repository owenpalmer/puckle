use ambient_api::{
    message::client::{MessageExt, Target},
    player::MouseButton,
    player::KeyCode,
    prelude::*,
};

#[main]
fn main() {
    on(event::FRAME, |_| {
        let (delta, input) = player::get_raw_input_delta();

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
        };
        msg.send(Target::RemoteReliable);
    });
}