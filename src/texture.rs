use std::collections::HashMap;
use std::fs;
use std::path::Path;
use image::GenericImageView;
use anyhow::*;

pub struct Texture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

impl Texture {
    pub fn from_bytes(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bytes: &[u8],
        label: &str,
    ) -> Result<Self> {
        let img = image::load_from_memory(bytes)?;
        Self::from_image(device, queue, &img, Some(label))
    }

    pub fn from_image(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        img: &image::DynamicImage,
        label: Option<&str>,
    ) -> Result<Self> {
        let rgba = img.to_rgba8();
        let dimensions = img.dimensions();

        // all textures represented in 3d, z is depth, set 2d texture as depth 1
        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(
            &wgpu::TextureDescriptor {
                label,
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                // TEXTURE_BINDING tells wgpu that we want to use this texture in shaders
                // COPY_DST means that this texture is a copy destination
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            }
        );

        // writes to the texture with the image data
        queue.write_texture(
            // tells wgpu where to copy pixel data
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            // actual pixel data
            &rgba,
            // layout of texture
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * dimensions.0),
                rows_per_image: Some(dimensions.1),
            },
            size,
        );

        // texture view offers us a view into the texture
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        // sampler controls how uv -> color from texture
        let sampler = device.create_sampler(
            &wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::ClampToEdge, // these tell the sampler what to do when uv is out of bounds
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear, // what to do when the image is too small and needs to be enlarged
                min_filter: wgpu::FilterMode::Nearest, // what to do when the image is too large and needs to be shrank
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            }
        );

        Ok(Self { texture, view, sampler })
    }
}

pub struct TextureManager {
    textures: HashMap<String, RegisteredTexture>
}

pub struct RegisteredTexture {
    pub texture: Texture,
    pub bind_group: wgpu::BindGroup,
}

impl TextureManager {
    const VALID_FILE_PATHS: &'static [&'static str] = &["png", "webp"];

    pub fn new(
        directory: &Path,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bind_group_layout: &wgpu::BindGroupLayout
    ) -> Self {
        let mut textures: HashMap<String, RegisteredTexture> = HashMap::new();

        let paths = fs::read_dir(directory).unwrap();

        for path in paths {
            let file = path.unwrap().path();
            let extension = file.extension().unwrap();

            let is_file_texture = Self::VALID_FILE_PATHS.iter().any(|&ext| ext == extension);
            if !is_file_texture {
                continue;
            }

            let file_name = file.file_name().unwrap().to_str().unwrap();
            let file_path_relative = file.strip_prefix(directory).unwrap();

            let texture_data = fs::read(&file).unwrap();
            let texture_data_arr = texture_data.as_slice();
            let texture = Texture::from_bytes(device, queue, texture_data_arr, file_name).unwrap();

            let bind_group = device.create_bind_group(
                &wgpu::BindGroupDescriptor {
                    layout: bind_group_layout,
                    entries: &[ // filling in the data described by the layout
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&texture.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&texture.sampler),
                        }
                    ],
                    label: None
                }
            );

            let registered = RegisteredTexture {
                texture,
                bind_group,
            };

            textures.insert(file_path_relative.to_str().unwrap().to_string(), registered);
        }

        Self {
            textures
        }
    }

    pub fn get_texture(&self, texture_name: &String) -> Option<&RegisteredTexture> {
        self.textures.get(texture_name)
    }
}