pub mod texture;
pub mod model;
pub mod resources;
pub mod camera;

use std::mem;
use std::path::Path;
use std::sync::Arc;
use cgmath::{Deg, InnerSpace, Matrix3, Matrix4, Point3, Quaternion, Vector3, Zero};
use cgmath::prelude::*;
use wgpu::{include_wgsl, Color, PresentMode};
use wgpu::util::DeviceExt;
use winit::{event::*, event_loop::EventLoop, window::WindowBuilder};
use winit::event_loop::EventLoopWindowTarget;
use winit::keyboard::Key::Named;
use winit::keyboard::NamedKey;
use winit::window::{CursorGrabMode, Window};
use camera::{Camera, CameraUniform, CameraController, FreeFlyController, Projection};
use model::{DrawLight, DrawModel, ModelVertex, Vertex};
use resources::ResourceManager;
use texture::Texture;

use crate::voxel::chunk::Chunk;

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

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: Arc<Window>,
    depth_texture: Texture,
    render_pipeline: wgpu::RenderPipeline,
    light_uniform: LightUniform,
    light_buffer: wgpu::Buffer,
    light_bind_group: wgpu::BindGroup,
    light_render_pipeline: wgpu::RenderPipeline,
    resource_manager: ResourceManager,
    camera: Camera,
    projection: Projection,
    camera_controller: FreeFlyController,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    instances: Vec<Instance>,
    instance_buffer: wgpu::Buffer,
    chunk: Chunk,
    chunk_model: model::Model,
}
struct Instance {
    position: Vector3<f32>,
    rotation: Quaternion<f32>,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
    normal: [[f32; 3]; 3],
}

impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        let model = Matrix4::from_translation(self.position) * Matrix4::from(self.rotation);
        InstanceRaw {
            model: model.into(),
            normal: Matrix3::from(self.rotation).into(),
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
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 19]>() as wgpu::BufferAddress,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 22]>() as wgpu::BufferAddress,
                    shader_location: 11,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        // instance is the main interface with wgpu, use this to create all other stuff
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // surface is what the gpu draws to
        let surface = instance.create_surface(window.clone())
            .expect("cant create surface");

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
                        // diffuse texture
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
                        // normal map
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                    label: Some("texture_bind_group_layout")
                });

        let light_uniform = LightUniform {
            position: [2.0, 2.0, 2.0],
            _padding: 0,
            color: [1.0, 1.0, 1.0],
            _padding2: 0,
        };

        // we'll want to update our light's position, so COPY_DST
        let light_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Light View buffer"),
                contents: bytemuck::cast_slice(&[light_uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST
            }
        );

        let light_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("Light bind group layout")
            });
        let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &light_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: light_buffer.as_entire_binding()
            }],
            label: None,
        });

        let resource_manager = ResourceManager::new(
            Path::new("./res"),
            &device,
            &queue,
            &texture_bind_group_layout,
        );

        let camera = Camera::new(
            Point3::new(0.0, 2.0, -5.0),
            Deg(90.0),
            Deg(-15.0),
        );
        let projection = Projection::new(
            config.width,
            config.height,
            Deg(45.0),
            0.1,
            100.0,
        );
        let camera_controller = FreeFlyController::new(6.0, 2.0);
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera, &projection);

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
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT, 
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

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    &texture_bind_group_layout,
                    &camera_bind_group_layout,
                    &light_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let render_pipeline = {
            let shader = include_wgsl!("shaders/shader.wgsl");
            create_render_pipeline(
                &device,
                &render_pipeline_layout,
                config.format,
                Some(Texture::DEPTH_FORMAT),
                &[ModelVertex::desc(), InstanceRaw::desc()],
                shader,
                "Render Pipeline",
            )
        };

        let light_render_pipeline = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Light Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout, &light_bind_group_layout],
                push_constant_ranges: &[],
            });
            let shader = include_wgsl!("shaders/light.wgsl");

            create_render_pipeline(
                &device,
                &layout,
                config.format,
                Some(Texture::DEPTH_FORMAT),
                &[ModelVertex::desc()],
                shader,
                "Light Render Pipeline",
            )
        };

        // let instances = (0..NUM_INSTANCES_PER_ROW).flat_map(|z| {
        //     (0..NUM_INSTANCES_PER_ROW).map(move |x| {
        //         let position = Vector3 { x: (x as f32) * 3.0, y: 0.0, z: (z as f32) * 3.0 } - INSTANCE_DISPLACEMENT;

        //         let rotation = if position.is_zero() {
        //             Quaternion::from_axis_angle(Vector3::unit_z(), Deg(0.0))
        //         } else {
        //             Quaternion::from_axis_angle(position.normalize(), Deg(45.0))
        //         };
        //         // let rotation = Quaternion::from_axis_angle((0.0, 1.0, 0.0).into(), Deg(180.0));

        //         Instance {
        //             position, rotation,
        //         }
        //     })
        // }).collect::<Vec<_>>();
        let instances = vec![Instance {
            position: Vector3::new(0.0, 0.0, 0.0),
            rotation: Quaternion::from_axis_angle((0.0, 1.0, 0.0).into(), Deg(0.0)),
        }];

        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(&instance_data),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );

        let chunk = Chunk::new_plane(0);
        let chunk_mesh = chunk.create_mesh(&device, &queue, (0, 0));
        let chunk_material = model::Material::new(
            &device,
            &queue,
            "chunk_material",
            Some(Texture::from_color(&device, &queue, &Color::GREEN, "green").unwrap()),
            None,
            None,
            &texture_bind_group_layout,
        );
        let chunk_model = model::Model {
            meshes: vec![chunk_mesh],
            materials: vec![chunk_material],
        };

        Self {
            surface,
            device,
            queue,
            config,
            size,
            window,
            depth_texture,
            light_uniform,
            light_buffer,
            light_bind_group,
            light_render_pipeline,
            render_pipeline,
            resource_manager,
            camera,
            projection,
            camera_controller,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            instances,
            instance_buffer,
            chunk,
            chunk_model,
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
            self.depth_texture = Texture::create_depth_texture(&self.device, &self.config, "depth");
            self.projection.resize(new_size.width, new_size.height);
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        self.camera_controller.process_input(event)
    }

    fn update(&mut self, delta_time: f32) {
        // update the light
        {
            let old_position: Vector3<_> = self.light_uniform.position.into();
            self.light_uniform.position =
                (Quaternion::from_axis_angle((0.0, 1.0, 0.0).into(), Deg(1.0))
                    * old_position)
                    .into();
            self.queue.write_buffer(&self.light_buffer, 0, bytemuck::cast_slice(&[self.light_uniform]));
        }

        self.camera_controller.update_camera(&mut self.camera, delta_time);
        self.camera_uniform.update_view_proj(&self.camera, &self.projection);
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

            let model = &self.resource_manager.get_model(&String::from("cube.obj")).unwrap();
            let chunk_model = &self.chunk_model;

            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));

            render_pass.set_pipeline(&self.light_render_pipeline);
            render_pass.draw_light_model(
                model,
                &self.camera_bind_group,
                &self.light_bind_group,
            );

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.draw_model_instanced(chunk_model, 0..self.instances.len() as u32, &self.camera_bind_group, &self.light_bind_group);
        } // extra block tells borrower to drop any borrows, begin_render_pass borrows encoder, so to finish, we need to drop it

        // finish the command buffer and SEND
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

fn create_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    color_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    vertex_layouts: &[wgpu::VertexBufferLayout],
    shader: wgpu::ShaderModuleDescriptor,
    name: &str,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(shader);

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(name),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main", // must specify the entrypoint
            buffers: vertex_layouts, // tells wgpu what type of vertices we want to pass to the shader
        },
        fragment: Some(wgpu::FragmentState { // technically optional
            module: &shader,
            entry_point: "fs_main", // must specify the entrypoint
            targets: &[Some(wgpu::ColorTargetState { // tells wgpu what colour outputs to set up
                format: color_format, // only need the one output for now, use surface's format so it's easy to copy
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
        depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
            format,
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
    })
}

// Blinn-Phong Model

// represents a colored point in space.
// WGSL struct field alignments must be powers of 2.
// e.g. a vec3 would be 12 bytes, but it would get bumped up to 16
// so padding is used
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LightUniform {
    position: [f32; 3],
    _padding: u32,
    color: [f32; 3],
    _padding2: u32,
}

pub async fn run() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("cant create eventloop");
    let window = Arc::new(WindowBuilder::new().build(&event_loop).expect("cant create window"));

    let mut state = State::new(window).await;

    //state.window.set_cursor_grab(CursorGrabMode::Locked).expect("Could not grab cursor");

    let mut last_render_time = std::time::Instant::now();

    event_loop.run(move |event, event_loop_window_target| {
        match event {
            Event::WindowEvent { ref event, window_id }
                if window_id == state.window().id() => window_event(&mut state, event, event_loop_window_target, &mut last_render_time),
            _ => {}
        };

        state.window.request_redraw();
    }).expect("TODO: panic message");
}

fn window_event(state: &mut State, ref event: &WindowEvent, event_loop_window_target: &EventLoopWindowTarget<()>, last_render_time: &mut std::time::Instant) {
    if state.input(event) {
        return;
    }

    match event {
        WindowEvent::RedrawRequested => {
            let now = std::time::Instant::now();
            let dt = now.duration_since(*last_render_time).as_secs_f32();
            *last_render_time = now;

            state.update(dt);
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

        WindowEvent::Resized(physical_size) => {
            state.resize(*physical_size);
        }
        WindowEvent::ScaleFactorChanged { inner_size_writer, .. } => {
            // TODO: how tf do you do this
        }
        _ => {}
    }
}
