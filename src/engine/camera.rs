use crate::world::player::{LookDirection, Position};
use nalgebra as na;
use nalgebra::{Isometry3, Matrix4, Perspective3, Point3, Vector3};
use ordered_float::Float;
use rapier3d_f64::prelude as rapier;
use winit::event::WindowEvent;
use winit::keyboard::KeyCode;
use winit::keyboard::PhysicalKey::Code;

pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0,
);

pub struct Camera {
    pub position: Point3<f32>,
    pub yaw: Rad<f32>,
    pub pitch: Rad<f32>,
}

impl Camera {
    pub fn new<V: Into<Point3<f32>>, Y: Into<Rad<f32>>, P: Into<Rad<f32>>>(
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
        let direction = self.direction();
        let up = Vector3::y();

        Isometry3::look_at_rh(&self.position, &(self.position + direction), &up).to_homogeneous()
    }

    pub fn direction(&self) -> Vector3<f32> {
        let (sin_pitch, cos_pitch) = self.pitch.0.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.0.sin_cos();

        Vector3::new(cos_yaw * cos_pitch, sin_pitch, sin_yaw * cos_pitch).normalize()
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
        OPENGL_TO_WGPU_MATRIX
            * Matrix4::new_perspective(self.aspect, self.fov_y.0, self.z_near, self.z_far)
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    //view_position: [f32; 4],
    proj_matrix: [[f32; 4]; 4],
    view_matrix: [[f32; 4]; 4],
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            //view_position: [0.0; 4],
            proj_matrix: Matrix4::identity().into(),
            view_matrix: Matrix4::identity().into(),
        }
    }

    pub fn update_view_proj(&mut self, camera: &Camera, projection: &Projection) {
        // self.view_position = camera.position.coords.push(1.0).into();
        self.view_matrix = camera.calc_matrix().into();
        self.proj_matrix = projection.calc_matrix().into();
    }

    pub fn update_view_proj_with_view(&mut self, projection: &Projection, view: &Matrix4<f32>) {
        // self.view_position = camera.position.coords.push(1.0).into();
        self.view_matrix = view.clone().into();
        self.proj_matrix = projection.calc_matrix().into()
    }
}

pub struct PlayerController {
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

impl PlayerController {
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

    pub fn update(
        &self,
        rigid_body: &mut rapier::RigidBody,
        direction: &mut LookDirection,
        delta_time: f32,
    ) {
        let (sin_pitch, cos_pitch) = direction.pitch.as_radians().0.sin_cos();
        let (sin_yaw, cos_yaw) = direction.yaw.as_radians().0.sin_cos();

        let forward = Vector3::new(cos_yaw * cos_pitch, sin_pitch, sin_yaw * cos_pitch).normalize();
        let right = forward.cross(&Vector3::y_axis());

        let multiplier = 0.1;

        if self.forward {
            rigid_body.apply_torque_impulse((forward * self.move_speed * delta_time * multiplier).map(|x| x as f64), true);
            //position.0 += forward * self.move_speed * delta_time;
        }
        if self.backward {
            rigid_body.apply_torque_impulse((-forward * self.move_speed * delta_time * multiplier).map(|x| x as f64), true);
            //position.0 -= forward * self.move_speed * delta_time;
        }
        if self.strafe_left {
            rigid_body.apply_torque_impulse((-right * self.move_speed * delta_time * multiplier).map(|x| x as f64), true);
            //position.0 -= right * self.move_speed * delta_time;
        }
        if self.strafe_right {
            rigid_body.apply_torque_impulse((right * self.move_speed * delta_time * multiplier).map(|x| x as f64), true);
            //position.0 += right * self.move_speed * delta_time;
        }

        let multiplier = 0.25;

        // rotate camera when arrow keys are pressed
        direction.yaw.0 += self.look_right * self.rotate_speed * delta_time * multiplier;
        direction.yaw.0 -= self.look_left * self.rotate_speed * delta_time * multiplier;
        direction.pitch.0 += self.look_up * self.rotate_speed * delta_time * multiplier;
        direction.pitch.0 -= self.look_down * self.rotate_speed * delta_time * multiplier;

        // clamp pitch up and down
        direction.pitch.0 = direction.pitch.0.clamp(-90.0, 90.0);
    }

    pub fn process_input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
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
            WindowEvent::CursorMoved { position, .. } => {
                // let center = winit::dpi::PhysicalPosition::new(position.x as f64, position.y as f64);
                // let delta = center - winit::dpi::PhysicalPosition::new(800.0, 450.0);
                // self.look_right = delta.x as f32;
                // self.look_up = delta.y as f32;
                return false;
            }
            _ => return false,
        };

        true
    }
}

pub trait CameraController {
    fn process_input(&mut self, event: &WindowEvent) -> bool;

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
            WindowEvent::KeyboardInput { event, .. } => {
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
            WindowEvent::CursorMoved { position, .. } => {
                // let center = winit::dpi::PhysicalPosition::new(position.x as f64, position.y as f64);
                // let delta = center - winit::dpi::PhysicalPosition::new(800.0, 450.0);
                // self.look_right = delta.x as f32;
                // self.look_up = delta.y as f32;
                return false;
            }
            _ => return false,
        };

        true
    }

    fn update_camera(&self, camera: &mut Camera, delta_time: f32) {
        // move camera relative to look direction
        let forward = camera.direction();
        let right = forward.cross(&Vector3::y_axis());

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
        camera.yaw.0 += self.look_right * self.rotate_speed * delta_time;
        camera.yaw.0 -= self.look_left * self.rotate_speed * delta_time;
        camera.pitch.0 += self.look_up * self.rotate_speed * delta_time;
        camera.pitch.0 -= self.look_down * self.rotate_speed * delta_time;

        // clamp pitch up and down
        //camera.pitch.0 = camera.pitch.as_degrees().0.clamp(-90.0, 90.0).to_radians();
    }
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct Deg<T: Float>(pub T);

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct Rad<T: Float>(pub T);

impl<T: Float> Deg<T> {
    pub fn as_radians(&self) -> Rad<T> {
        Rad(self.0.to_radians())
    }
}

impl<T: Float> Rad<T> {
    pub fn as_degrees(&self) -> Deg<T> {
        Deg(self.0.to_degrees())
    }
}

impl<T: Float> Into<Rad<T>> for Deg<T> {
    fn into(self) -> Rad<T> {
        Rad(self.0.to_radians())
    }
}

impl<T: Float> Into<Deg<T>> for Rad<T> {
    fn into(self) -> Deg<T> {
        Deg(self.0.to_degrees())
    }
}

impl From<Rad<f32>> for Rad<f64> {
    fn from(rad: Rad<f32>) -> Rad<f64> {
        Rad(rad.0 as f64)
    }
}

impl From<Deg<f32>> for Deg<f64> {
    fn from(deg: Deg<f32>) -> Deg<f64> {
        Deg(deg.0 as f64)
    }
}
