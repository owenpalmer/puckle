use ambient_api::prelude::*;

use crate::components::{
    camera_pitch, camera_yaw, camera_zoom,
};

pub struct CameraState(pub EntityId);
impl CameraState {
    pub fn translate_around_origin(&self, origin: Vec3) -> &Self {
        let rot = self.get_rotation();
        let offset = rot.mul_vec3(Vec3::Z * self.get_zoom());
        let position = origin + offset;
        entity::set_component(self.0, translation(), position);
        self
    }
    pub fn rotate(&self, rot: Vec2) -> &Self {
        let pitch = entity::mutate_component(self.0, camera_pitch(), |pitch| {
            *pitch = f32::clamp(*pitch + (rot.y / 300.0), 0., std::f32::consts::PI);
        }).unwrap();
        let yaw = entity::mutate_component(self.0, camera_yaw(), |p| {
            *p += rot.x / 150.;
        }).unwrap();

        let yaw = Quat::from_rotation_z(yaw);
        let pitch = Quat::from_rotation_x(pitch);
        entity::set_component(self.0, rotation(), yaw * pitch);

        self
    }
    pub fn get_rotation(&self) -> Quat {
        entity::get_component(self.0, rotation()).unwrap()
    }
    pub fn get_yaw(&self) -> f32 {
        entity::get_component(self.0, camera_yaw()).unwrap()
    }
    pub fn get_pitch(&self) -> f32 {
        entity::get_component(self.0, camera_pitch()).unwrap()
    }
    pub fn get_zoom(&self) -> f32 {
        entity::get_component(self.0, camera_zoom()).unwrap()
    }
    pub fn zoom(&self, delta: f32) -> &Self {
        entity::mutate_component(self.0, camera_zoom(), |radius| {
            // *radius = f32::clamp(*radius + delta, 1., 50.);
            println!("{}", *radius);
            *radius += delta;
        });
        self
    }
}
