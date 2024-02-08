mod texture;
mod model;
mod resources;

use std::mem;
use std::ops::AddAssign;
use std::path::Path;
use std::sync::Arc;
use cgmath::{Deg, InnerSpace, Matrix4, perspective, Point3, Quaternion, Vector3, Zero};
use cgmath::prelude::*;
use wgpu::PresentMode;
use wgpu::util::DeviceExt;
use winit::{event::*, event_loop::EventLoop, window::WindowBuilder};
use winit::event_loop::EventLoopWindowTarget;
use winit::keyboard::Key::Named;
use winit::keyboard::{NamedKey, KeyCode};
use winit::keyboard::PhysicalKey::Code;
use winit::window::Window;
use crate::model::{DrawModel, ModelVertex, Vertex};
use crate::resources::ResourceManager;
use crate::texture::{Texture};

const NUM_INSTANCES_PER_ROW: u32 = 10;
const INSTANCE_DISPLACEMENT: Vector3<f32> = Vector3::new(NUM_INSTANCES_PER_ROW as f32 * 0.5, 0.0, NUM_INSTANCES_PER_ROW as f32 * 0.5);

// The coordinate system in Wgpu is based on DirectX and Metal's coordinate systems.
// That means that in normalized device coordinates (opens new window),
// the x-axis and y-axis are in the range of -1.0 to +1.0, and the z-axis is 0.0 to +1.0.
// The cgmath crate (as well as most game math crates) is built for OpenGL's coordinate system.
// This matrix will scale and translate our scene from OpenGL's coordinate system to WGPU's.
#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.5,
    0.0, 0.0, 0.0, 1.0,
);

struct Camera {
    eye: Point3<f32>,
    target: Point3<f32>,
    up: Vector3<f32>,
    aspect: f32,
    fov_y: f32,
    z_near: f32,
    z_far: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

impl Camera {
    fn build_view_projection_matrix(&self) -> Matrix4<f32> {
        // moves the world to be at the position and rotation of the camera
        let view = Matrix4::look_at_rh(self.eye, self.target, self.up);
        // warps the scene to get the effect of depth
        let proj = perspective(Deg(self.fov_y), self.aspect, self.z_near, self.z_far);

        OPENGL_TO_WGPU_MATRIX * proj * view
    }

    fn offset_eye(&mut self, offset: &Vector3<f32>) {
        self.eye.add_assign(*offset);
    }
}

impl CameraUniform {
    fn new() -> Self {
        Self {
            view_proj: Matrix4::identity().into()
        }
    }

    fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix().into();
    }
}

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: Arc<Window>,
    depth_texture: Texture,
    render_pipeline: wgpu::RenderPipeline,
    resource_manager: ResourceManager,
    camera: Camera,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    instances: Vec<Instance>,
    instance_buffer: wgpu::Buffer,
}
struct Instance {
    position: Vector3<f32>,
    rotation: Quaternion<f32>,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
}

impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        InstanceRaw {
            model: (Matrix4::from_translation(self.position) * Matrix4::from(self.rotation)).into()
        }
    }
}

impl InstanceRaw {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

// as to not duplicate certain vertices, define a unique set above, and describe it's layout below
const INDICES: &[u16] = &[
    0, 1, 4,
    1, 2, 4,
    2, 3, 4,
];

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        // instance is the main interface with wgpu, use this to create all other stuff
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL,
            ..Default::default()
        });

        // surface is what the gpu draws to
        let surface = instance.create_surface(window.clone()).expect("cant create surface");

        // adapter is the interface to the GPU
        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance, // picks preference of GPU
                compatible_surface: Some(&surface), // must be compatible with our surface
                force_fallback_adapter: false,
            },
        ).await.expect("cant create adapter");

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                label: None,
            },
            None,
        ).await.expect("cant create device");

        let surface_caps = surface.get_capabilities(&adapter);

        // gets how the draw surface will be formatted (duh)
        let surface_format = surface_caps.formats.iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]); // if not found, go with the first format, could break shaders
        // configures the surface, we tell it what it's for, it's width etc. all the gpu needs to know to draw to it
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT, // textures will be used to write to the screen
            format: surface_format, // how the texture will be formatted on the gpu
            width: size.width, // width and height MUST NOT be 0, causes crash
            height: size.height,
            present_mode: surface_caps.present_modes.iter() // determines how to 'present' the surface to the display, e.g. vsync
                .copied()
                .find(|f| f == &PresentMode::AutoVsync)
                .unwrap_or(PresentMode::Fifo), // fifo supported on all
            desired_maximum_frame_latency: 2,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        // apply the configuration
        surface.configure(&device, &config);

        let depth_texture = Texture::create_depth_texture(&device, &config, "depth texture");

        // bind group describes a set of resources and how they can be accessed by shader
        let texture_bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        wgpu::BindGroupLayoutEntry { // sampled texture
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT, // only visible to frag shader
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry { // sampler
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            // This should match the filterable field of the
                            // corresponding Texture entry above.
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                    label: Some("texture_bind_group_layout")
                });

        let resource_manager = ResourceManager::new(
            Path::new("./res"),
            &device,
            &queue,
            &texture_bind_group_layout,
        );

        let camera = Camera {
            // position the camera 1 unit up and 2 units back
            // +z is out of the screen
            eye: (0.0, 2.0, 2.0).into(),
            // have it look at the origin
            target: (0.0, 1.0, 0.0).into(),
            // which way is up
            up: Vector3::unit_y(),
            aspect: config.width as f32 / config.height as f32,
            fov_y: 45.0,
            z_near: 0.1, // near plane
            z_far: 100.0, // far plane
        };
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        let camera_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Camera buffer"),
                contents: bytemuck::cast_slice(&[camera_uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );

        let camera_bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX, // only need cam info in vertex shader
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }
                ],
                label: Some("Camera bind group layout")
            }
        );
        let camera_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &camera_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: camera_buffer.as_entire_binding()
                    }
                ],
                label: Some("Camera bind group")
            }
        );

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    &texture_bind_group_layout,
                    &camera_bind_group_layout
                ],
                push_constant_ranges: &[],
            });

        // future me please add comments to this
        // https://sotrh.github.io/learn-wgpu/beginner/tutorial3-pipeline/#using-a-pipeline
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main", // must specify the entrypoint
                buffers: &[
                    ModelVertex::desc(), InstanceRaw::desc(),
                ], // tells wgpu what type of vertices we want to pass to the shader
            },
            fragment: Some(wgpu::FragmentState { // technically optional
                module: &shader,
                entry_point: "fs_main", // must specify the entrypoint
                targets: &[Some(wgpu::ColorTargetState { // tells wgpu what colour outputs to set up
                    format: config.format, // only need the one output for now, use surface's format so it's easy to copy
                    blend: Some(wgpu::BlendState::REPLACE), // no blend just replace
                    write_mask: wgpu::ColorWrites::ALL, // use all colour channels
                })],
            }),
            primitive: wgpu::PrimitiveState { // how to interpret vertices to convert them to triangles
                topology: wgpu::PrimitiveTopology::TriangleList, // every 3 vertices will correspond to one triangle
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw, // tells wgpu how to determine whether a given triangle is facing forward or not
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less, // less means front to back
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1, // how many samples the pipeline will use
                mask: !0, // specifies which samples we are using
                alpha_to_coverage_enabled: false, // anti aliasing
            },
            multiview: None,
        });

        let instances = (0..NUM_INSTANCES_PER_ROW).flat_map(|z| {
            (0..NUM_INSTANCES_PER_ROW).map(move |x| {
                let position = Vector3 { x: (x as f32) * 3.0, y: 0.0, z: (z as f32) * 3.0 } - INSTANCE_DISPLACEMENT;

                let rotation = if position.is_zero() {
                    Quaternion::from_axis_angle(Vector3::unit_z(), Deg(0.0))
                } else {
                    Quaternion::from_axis_angle(position.normalize(), Deg(45.0))
                };

                Instance {
                    position, rotation,
                }
            })
        }).collect::<Vec<_>>();

        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(&instance_data),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );

        Self {
            surface,
            device,
            queue,
            config,
            size,
            window,
            depth_texture,
            render_pipeline,
            resource_manager,
            camera,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            instances,
            instance_buffer,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture = Texture::create_depth_texture(&self.device, &self.config, "depth")
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        false
    }

    fn update(&mut self) {
        self.camera_uniform.update_view_proj(&self.camera);
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[self.camera_uniform]));
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // will wait for the surface to provide a new texture that we render to
        let output = self.surface.get_current_texture()?;

        // we need to do this to control how render code interacts with texture
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        // create a command executor to create commands to send to the gpu.
        // most modern frameworks expect commands to be stored in a buffer before being sent,
        // the encoder builds a command buffer to send
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder")
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    }
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            let rustiscool_tex = &self.resource_manager.get_texture(&String::from("rustiscool.png")).expect("cant get tex");
            let model = &self.resource_manager.get_model(&String::from("cube.obj")).expect("cant get model");

            render_pass.set_pipeline(&self.render_pipeline);

            render_pass.set_bind_group(0, &rustiscool_tex.bind_group, &[]);
            render_pass.set_bind_group(1, &self.camera_bind_group, &[]);

            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));

            render_pass.draw_model_instanced(&model, 0..self.instances.len() as u32)
        } // extra block tells borrower to drop any borrows, begin_render_pass borrows encoder, so to finish, we need to drop it

        // finish the command buffer and SEND
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

pub async fn run() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("cant create eventloop");
    let window = Arc::new(WindowBuilder::new().build(&event_loop).expect("cant create window"));

    let mut state = State::new(window).await;

    event_loop.run(move |event, event_loop_window_target| {
        match event {
            Event::WindowEvent { ref event, window_id }
                if window_id == state.window().id() => window_event(&mut state, event, event_loop_window_target),
            Event::NewEvents(_) => {
                state.window().request_redraw();
            }
            _ => {}
        }
    }).expect("TODO: panic message");
}

fn window_event(state: &mut State, ref event: &WindowEvent, event_loop_window_target: &EventLoopWindowTarget<()>) {
    if state.input(event) {
        return;
    }

    match event {
        WindowEvent::RedrawRequested => {
            state.update();
            match state.render() {
                Ok(_) => {}
                Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                Err(wgpu::SurfaceError::OutOfMemory) => event_loop_window_target.exit(),
                Err(e) => eprintln!("{:?}", e),
            }
        }

        WindowEvent::CloseRequested | WindowEvent::KeyboardInput {
            event: KeyEvent {
                state: ElementState::Pressed,
                logical_key: Named(NamedKey::Escape),
                ..
            },
            ..
        } => event_loop_window_target.exit(),

        WindowEvent::KeyboardInput {
            event: KeyEvent {
                state: ElementState::Pressed,
                physical_key: Code(KeyCode::KeyW),
                ..
            },
            ..
        } => {
            state.camera.offset_eye(&Vector3 { x: 0.5, y: 0.0, z: 0.0 })
        }
        WindowEvent::KeyboardInput {
            event: KeyEvent {
                state: ElementState::Pressed,
                physical_key: Code(KeyCode::KeyS),
                ..
            },
            ..
        } => {
            state.camera.offset_eye(&Vector3 { x: -0.5, y: 0.0, z: 0.0 })
        }

        WindowEvent::Resized(physical_size) => {
            state.resize(*physical_size);
        }
        WindowEvent::ScaleFactorChanged { inner_size_writer, .. } => {
            // TODO: how tf do you do this
        }
        _ => {}
    }
}
