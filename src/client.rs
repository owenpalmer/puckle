use ambient_api::{
    components::core::{
        app::{main_scene, window_physical_size},
    },
    prelude::*,
};

#[main]
fn main() {
    messages::CamInit::subscribe(|source, data| {
        ambient_api::messages::Frame::subscribe(move |_| {
            let input = input::get();
            let ray = camera::screen_to_world_direction(data.camera_id, input.mouse_position);
    
            // Send screen ray to server
            messages::MouseRay {
                ray_origin: ray.origin,
                ray_dir: ray.dir,
            }
            .send_server_unreliable();
        });
    });

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
        };
        msg.send_server_unreliable();
    });
}