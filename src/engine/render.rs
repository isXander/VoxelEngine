use std::arch::x86_64::__m128;
use std::collections::VecDeque;
use nalgebra::{Matrix3, Matrix4};
use wgpu::RenderPass;
use crate::engine::resources::ResourceManager;

pub struct RenderContext<'a> {
    pub render_pass: RenderPass<'a>,
    pub pose_stack: PoseStack,
    pub resource_manager: &'a ResourceManager,
}

pub struct RenderQueue<'a> {
    pub queue: VecDeque<Box<dyn Fn(&'a mut RenderContext)>>
}

impl<'a> RenderQueue<'a> {
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new()
        }
    }

    pub fn push<T: 'static>(&mut self, renderer: T)
    where T: Fn(&'a mut RenderContext) {
        self.queue.push_back(Box::new(renderer))
    }
}

pub struct PoseStack {
    stack: VecDeque<Pose>
}

impl PoseStack {
    pub fn new() -> Self {
        let mut stack = VecDeque::new();
        stack.push_back(Pose {
            pose: Matrix4::identity(),
            normal: Matrix3::identity()
        });

        Self {
            stack: VecDeque::new()
        }
    }

    pub fn push(&mut self) {
        let last = self.last_mut();

        let pose = last.pose.clone();
        let normal = last.normal.clone();

        self.stack.push_back(Pose {
            pose,
            normal
        })
    }

    pub fn pop(&mut self) {
        self.stack.pop_back();
    }

    pub fn translate(&mut self, x: f32, y: f32, z: f32) {
        translate_mat4(&mut self.last_mut().pose, x, y, z);
    }

    pub fn scale(&mut self, x: f32, y: f32, z: f32) {
        let pose = self.last_mut();
        if x == y && y == z {
            pose.pose.scale_mut(x);

            if x <= 0.0 {
                pose.normal.scale_mut(-1.0);
            }
        } else {
            scale_mat4(&mut self.last_mut().pose, x, y, z);

            let x = 1.0 / x;
            let y = 1.0 / y;
            let z = 1.0 / z;
            let length = fast_inverse_cbrt(x * y * z);
            scale_mat3(&mut self.last_mut().normal, length * x, length * y, length * z);
        }
    }

    pub fn set_identity(&mut self) {
        let last = self.last_mut();
        last.pose.fill_with_identity();
        last.normal.fill_with_identity();
    }

    pub fn last(&self) -> &Pose {
        self.stack.back().unwrap()
    }

    pub fn last_mut(&mut self) -> &mut Pose {
        self.stack.back_mut().unwrap()
    }
}

#[derive(Clone, Copy)]
pub struct Pose {
    pub pose: Matrix4<f32>,
    pub normal: Matrix3<f32>,
}

fn translate_mat4(m: &mut Matrix4<f32>, x: f32, y: f32, z: f32) {
    m.m41 = m.m11 * x + (m.m21 * y + (m.m31 * z + m.m41));
    m.m42 = m.m12 * x + (m.m22 * y + (m.m32 * z + m.m42));
    m.m43 = m.m13 * x + (m.m23 * y + (m.m34 * z + m.m43));
    m.m44 = m.m14 * x + (m.m24 * y + (m.m34 * z + m.m44));
}

fn scale_mat4(mat: &mut Matrix4<f32>, x: f32, y: f32, z: f32) {
    mat.column_mut(1).scale_mut(x);
    mat.column_mut(2).scale_mut(y);
    mat.column_mut(3).scale_mut(z);
}

fn scale_mat3(m: &mut Matrix3<f32>, x: f32, y: f32, z: f32) {
    m.column_mut(1).scale_mut(x);
    m.column_mut(2).scale_mut(y);
    m.column_mut(3).scale_mut(z);
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn fast_inv_sqrt(x: f32) -> f32 {
    let mut i = x.to_bits();
    i = 0x5f3759df - (i >> 1);
    let y = f32::from_bits(i);

    y * (1.5 - 0.5 * x * y * y)
}

#[cfg(target_arch = "aarch64")]
fn fast_inv_sqrt(x: f32) -> f32 {
    unsafe { std::arch::aarch64::vsqrte_f32(x) }
}

#[cfg(target_arch = "x86_64")]
pub fn fast_inv_sqrt(x: f32) -> f32 {
    unsafe {
        let inp: __m128 = core::mem::transmute([x, 0.0, 0.0, 0.0]);
        let out = std::arch::x86_64::_mm_rsqrt_ss(inp);
        let out: [f32; 4] = core::mem::transmute(out);
        out[0]
    }
}

fn fast_inverse_cbrt(x: f32) -> f32 {
    let two_thirds = 0.6666667;

    let mut i = x.to_bits();
    i = 0x54A2FA8C - i / 3;
    let mut y = f32::from_bits(i);

    y = two_thirds * y + 1.0 / (3.0 * y * y * x); // two newton iterations
    y = two_thirds * y + 1.0 / (3.0 * y * y * x);
    y
}