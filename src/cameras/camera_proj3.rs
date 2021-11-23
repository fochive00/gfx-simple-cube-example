// use std::ops::Add;

extern crate nalgebra as na;

use winit::event::{Event, VirtualKeyCode};
use std::f32::consts::PI;

#[allow(dead_code)]
pub struct CameraProj3 {
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,

    position: na::Point3<f32>,
    look_direction: na::Vector3<f32>,
    right_direction: na::Vector3<f32>,

    movement_speed: f32,
    rotation_speed: f32,

    flip_y: bool,

    view: na::Matrix4<f32>,
    proj: na::Matrix4<f32>
}

impl CameraProj3 {
    pub fn new() -> Self {
        let aspect = 16.0 / 9.0;
        let fovy = 3.14 / 4.0;
        let znear = 0.1;
        let zfar = 1000.0;

        let position = na::point!(2.0, 2.0, 2.0);
        let target = na::point!(0.0, 0.0, 0.0);
        let look_direction = (target - position).normalize();
        let right_direction = look_direction.cross(&na::Vector3::z()).normalize();
        
        let movement_speed = 0.1;
        let rotation_speed = 0.1;

        let flip_y = true;

        let view = na::Isometry3::look_at_rh(
            &position, 
            &target,
            &right_direction.cross(&look_direction).normalize()
        ).to_homogeneous();

        let proj = na::Perspective3::new(
            aspect,
            fovy,
            znear,
            zfar
        ).to_homogeneous();

        println!("projective: {:?}", proj);

        Self {
            aspect,
            fovy,
            znear,
            zfar,
            position,
            look_direction,
            right_direction,
            movement_speed,
            rotation_speed,
            flip_y,
            view,
            proj
        }
    }


    fn up_direction(&self) -> na::Vector3<f32> {
        self.right_direction.cross(&self.look_direction)
    }
}

impl super::Camera for CameraProj3 {
    fn transform(&self) -> na::Matrix4<f32> {
        self.proj * self.view
    }

    fn update_view(&mut self) {
        self.view = {
            let view = na::Isometry3::look_at_rh(
                &self.position, 
                &(self.position + self.look_direction),
                &self.right_direction.cross(&self.look_direction).normalize()
            ).to_homogeneous();

            view
        }
    }

    fn event_handler<T>(&mut self, event: &Event<T>) {
        match event {
            winit::event::Event::WindowEvent { event: window_event, .. } => {
                match window_event {
                    winit::event::WindowEvent::KeyboardInput {
                        input:
                            winit::event::KeyboardInput {
                                virtual_keycode: Some(vkey),
                                ..
                            },
                        ..
                    } => {
                        match vkey {
                            VirtualKeyCode::W => self.position = self.position + self.movement_speed * self.look_direction,
                            VirtualKeyCode::S => self.position = self.position - self.movement_speed * self.look_direction,
        
                            VirtualKeyCode::A => self.position = self.position - self.movement_speed  * self.right_direction,
                            VirtualKeyCode::D => self.position = self.position + self.movement_speed  * self.right_direction,
        
                            VirtualKeyCode::C => {
                                let mut direction = self.up_direction();
                                if self.flip_y {
                                    direction = -direction;
                                }
                                
                                self.position = self.position + self.movement_speed  * direction;
                            }
                            VirtualKeyCode::V => {
                                let mut direction = self.up_direction();
                                if self.flip_y {
                                    direction = -direction;
                                }
                                self.position = self.position - self.movement_speed  * direction;
                            }
                            _ => ()
                        }
                        
                    },
        
                    _ => ()
                }
            }

            winit::event::Event::DeviceEvent { event: device_event, .. } => {
                match device_event {
                    winit::event::DeviceEvent::MouseMotion { delta } => {
                        let     x = (delta.0 as f32) * self.rotation_speed / 180.0 * PI;
                        let mut y = (delta.1 as f32) * self.rotation_speed / 180.0 * PI;

                        if self.flip_y {
                            y *= -1.0;
                        }
                        // println!("x: {}, y: {}", x, y);

                        let axis = na::Unit::new_normalize(self.look_direction.cross(&self.right_direction));
                        let rot_quat1 = na::UnitQuaternion::from_axis_angle(&axis, x);

                        let axis = na::Unit::new_normalize(-1.0 * self.right_direction);
                        let rot_quat2 = na::UnitQuaternion::from_axis_angle(&axis, y);

                        let rot_quat = rot_quat1.nlerp(&rot_quat2, 0.5);
                        self.look_direction = rot_quat * self.look_direction;
                        self.right_direction = rot_quat * self.right_direction;
                    }
                    _ => ()
                }
                
            }
            _ => ()
        }
        

        self.update_view();
    }
}