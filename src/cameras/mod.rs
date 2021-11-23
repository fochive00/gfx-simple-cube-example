
mod camera_proj3;
pub use camera_proj3::CameraProj3;

extern crate nalgebra as na;
pub trait Camera {
    fn transform(&self) -> na::Matrix4<f32>;
    fn update_view(&mut self);
    fn event_handler<T>(&mut self, event: &winit::event::Event<T>);
}