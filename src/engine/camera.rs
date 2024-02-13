use std::f32::consts::FRAC_PI_2;
use cgmath::{InnerSpace, Matrix4, Point3, Rad, Vector3};
use cgmath::prelude::*;
use winit::event::WindowEvent;
use winit::keyboard::KeyCode;
use winit::keyboard::PhysicalKey::Code;

pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.5,
    0.0, 0.0, 0.0, 1.0,
);

const SAFE_FRAC_PI_2: f32 = FRAC_PI_2 - 0.0001;

pub struct Camera {
    pub position: Point3<f32>,
    pub yaw: Rad<f32>,
    pub pitch: Rad<f32>,
}

impl Camera {
    pub fn new<
        V: Into<Point3<f32>>,
        Y: Into<Rad<f32>>,
        P: Into<Rad<f32>>,
    >(
        position: V,
        yaw: Y,
        pitch: P,
    ) -> Self {
        Self {
            position: position.into(),
            yaw: yaw.into(),
            pitch: pitch.into(),
        }
    }

    pub fn calc_matrix(&self) -> Matrix4<f32> {
        let (sin_pitch, cos_pitch) = self.pitch.0.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.0.sin_cos();

        Matrix4::look_to_rh(
            self.position,
            Vector3 {
                x: cos_yaw * cos_pitch,
                y: sin_pitch,
                z: sin_yaw * cos_pitch,
            }.normalize(),
            Vector3::unit_y(),
        )
    }
}

pub struct Projection {
    aspect: f32,
    fov_y: Rad<f32>,
    z_near: f32,
    z_far: f32,
}

impl Projection {
    pub fn new<F: Into<Rad<f32>>>(
        width: u32,
        height: u32,
        fov_y: F,
        z_near: f32,
        z_far: f32,
    ) -> Self {
        Self {
            aspect: width as f32 / height as f32,
            fov_y: fov_y.into(),
            z_near,
            z_far,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height as f32;
    }

    pub fn calc_matrix(&self) -> Matrix4<f32> {
        OPENGL_TO_WGPU_MATRIX * cgmath::perspective(self.fov_y, self.aspect, self.z_near, self.z_far)
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    view_position: [f32; 4],
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_position: [0.0; 4],
            view_proj: Matrix4::identity().into(),
        }
    }

    pub fn update_view_proj(&mut self, camera: &Camera, projection: &Projection) {
        self.view_position = camera.position.to_homogeneous().into();
        self.view_proj = (projection.calc_matrix() * camera.calc_matrix()).into();
    }
}

pub trait CameraController {
    fn process_input(&mut self, event: &winit::event::WindowEvent) -> bool;

    fn update_camera(&self, camera: &mut Camera, delta_time: f32);
}

pub struct FreeFlyController {
    move_speed: f32,
    rotate_speed: f32,

    strafe_left: bool,
    strafe_right: bool,
    forward: bool,
    backward: bool,

    look_up: f32,
    look_down: f32,
    look_left: f32,
    look_right: f32,
}

impl FreeFlyController {
    pub fn new(move_speed: f32, rotate_speed: f32) -> Self {
        Self {
            move_speed,
            rotate_speed,
            strafe_right: false,
            strafe_left: false,
            forward: false,
            backward: false,
            look_up: 0.0,
            look_right: 0.0,
            look_down: 0.0,
            look_left: 0.0,
        }
    }
}

impl CameraController for FreeFlyController {
    fn process_input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput { event, ..} => {
                let is_pressed = event.state == winit::event::ElementState::Pressed;

                match event.physical_key {
                    Code(KeyCode::KeyW) => self.forward = is_pressed,
                    Code(KeyCode::KeyS) => self.backward = is_pressed,
                    Code(KeyCode::KeyA) => self.strafe_left = is_pressed,
                    Code(KeyCode::KeyD) => self.strafe_right = is_pressed,

                    // temporary
                    Code(KeyCode::ArrowUp) => self.look_up = is_pressed as i32 as f32,
                    Code(KeyCode::ArrowDown) => self.look_down = is_pressed as i32 as f32,
                    Code(KeyCode::ArrowLeft) => self.look_left = is_pressed as i32 as f32,
                    Code(KeyCode::ArrowRight) => self.look_right = is_pressed as i32 as f32,
                    _ => return false,
                }
            }
            WindowEvent::CursorMoved {
                position,
                ..
            } => {
                // let center = winit::dpi::PhysicalPosition::new(position.x as f64, position.y as f64);
                // let delta = center - winit::dpi::PhysicalPosition::new(800.0, 450.0);
                // self.look_right = delta.x as f32;
                // self.look_up = delta.y as f32;
                return false
            }
            _ => return false
        };

        true
    }

    fn update_camera(&self, camera: &mut Camera, delta_time: f32) {
        // move camera relative to look direction
        let (sin_yaw, cos_yaw) = camera.yaw.0.sin_cos();
        let (sin_pitch, cos_pitch) = camera.pitch.0.sin_cos();

        let forward = Vector3 {
            x: cos_pitch * cos_yaw,
            y: sin_pitch,
            z: cos_pitch * sin_yaw,
        }.normalize();
        let right = forward.cross(Vector3::unit_y());

        if self.forward {
            camera.position += forward * self.move_speed * delta_time;
        }
        if self.backward {
            camera.position -= forward * self.move_speed * delta_time;
        }
        if self.strafe_left {
            camera.position -= right * self.move_speed * delta_time;
        }
        if self.strafe_right {
            camera.position += right * self.move_speed * delta_time;
        }

        // rotate camera when arrow keys are pressed
        camera.yaw += Rad(self.look_right * self.rotate_speed * delta_time);
        camera.yaw -= Rad(self.look_left * self.rotate_speed * delta_time);
        camera.pitch += Rad(self.look_up * self.rotate_speed * delta_time);
        camera.pitch -= Rad(self.look_down * self.rotate_speed * delta_time);
        println!("yaw: {}, pitch: {}", camera.yaw.0, camera.pitch.0);
    }
}

