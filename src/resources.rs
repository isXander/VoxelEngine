use std::collections::HashMap;
use std::fs;
use std::io::{BufReader, Cursor};
use std::path::Path;
use wgpu::util::DeviceExt;
use crate::model::{Material, Mesh, Model, ModelVertex};
use crate::texture::{RegisteredTexture, Texture};

pub struct ResourceManager {
    textures: HashMap<String, RegisteredTexture>,
    models: HashMap<String, Model>
}

impl ResourceManager {
    const TEXTURE_EXTENSIONS: &'static [&'static str] = &["png", "webp"];
    const MODEL_EXTENSIONS: &'static [&'static str] = &["obj"];

    pub fn new(
        directory: &Path,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bind_group_layout: &wgpu::BindGroupLayout
    ) -> Self {
        let mut textures: HashMap<String, RegisteredTexture> = HashMap::new();
        let mut models: HashMap<String, Model> = HashMap::new();

        let paths = fs::read_dir(directory).expect("can't read dir");

        for path in paths {
            let file = path.expect("cant get path").path();
            let extension = file.extension().expect("cant get extension");

            let file_name = file.file_name().expect("cant get file name").to_str().expect("cant get file name as string");
            let file_path_relative = file.strip_prefix(directory).expect("cant strip prefix");

            let file_data = fs::read(&file).expect("cant read data");
            let file_data_arr = file_data.as_slice();

            if Self::TEXTURE_EXTENSIONS.iter().any(|&ext| ext == extension) {
                let texture = Self::create_texture(file_data_arr, file_path_relative, device, queue, bind_group_layout);
                textures.insert(file_path_relative.to_str().unwrap().to_string(), texture);
            } else if Self::MODEL_EXTENSIONS.iter().any(|&ext| ext == extension) {
                let model = Self::create_model(file_data_arr, file_path_relative, directory, device, queue, bind_group_layout);
                models.insert(file_path_relative.to_str().unwrap().to_string(), model);
            }
        }

        Self {
            textures,
            models,
        }
    }

    fn create_texture(
        texture_data: &[u8],
        file_path: &Path,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bind_group_layout: &wgpu::BindGroupLayout
    ) -> RegisteredTexture {
        let texture_id = file_path.to_str().unwrap();
        let texture = Texture::from_bytes(device, queue, texture_data, texture_id).unwrap();

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

        RegisteredTexture {
            texture,
            bind_group,
        }
    }

    fn create_model(
        model_data: &[u8],
        path: &Path,
        root_directory: &Path,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Model {
        let obj_cursor = Cursor::new(model_data);
        let mut obj_reader = BufReader::new(obj_cursor);

        let (models, obj_materials) = tobj::load_obj_buf(
            &mut obj_reader,
            &tobj::LoadOptions {
                triangulate: true,
                single_index: true,
                ..Default::default()
            },
            |p| {
                let path = root_directory.join(p);
                println!("{:?}", path);
                let mat_text = fs::read_to_string(path).unwrap();
                tobj::load_mtl_buf(&mut BufReader::new(Cursor::new(mat_text)))
            },
        ).unwrap();

        let mut materials = Vec::new();
        for m in obj_materials.unwrap() {
            let path = root_directory.join(&m.diffuse_texture.unwrap());
            let bytes = fs::read(&path).unwrap();

            let diffuse_texture = Self::create_texture(bytes.as_slice(), path.as_path(), device, queue, bind_group_layout);
            materials.push(Material {
                name: m.name,
                texture: diffuse_texture
            })
        }

        let meshes = models
            .into_iter()
            .map(|m| {
                let vertices = (0..m.mesh.positions.len() / 3)
                    .map(|i| {
                        let position: [f32; 3] = [
                            m.mesh.positions[i * 3],
                            m.mesh.positions[i * 3 + 1],
                            m.mesh.positions[i * 3 + 2],
                        ];
                        let tex_coords: [f32; 2] = if !m.mesh.texcoords.is_empty() {
                            [
                                m.mesh.texcoords[i * 2],
                                1.0 - m.mesh.texcoords[i * 2 + 1],
                            ]
                        } else {
                            [0.0, 0.0]
                        };
                        let normal: [f32; 3] = if !m.mesh.normals.is_empty() {
                            [
                                m.mesh.normals[i * 3],
                                m.mesh.normals[i * 3 + 1],
                                m.mesh.normals[i * 3 + 2],
                            ]
                        } else {
                            [0.0, 0.0, 0.0]
                        };

                        ModelVertex {
                            position,
                            tex_coords,
                            normal,
                        }
                    }).collect::<Vec<_>>();

                let vertex_buffer = device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("{:?} Vertex Buffer", path.to_str().unwrap())),
                        contents: bytemuck::cast_slice(&vertices),
                        usage: wgpu::BufferUsages::VERTEX,
                    }
                );
                let index_buffer = device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("{:?} Index Buffer", path.to_str().unwrap())),
                        contents: bytemuck::cast_slice(&m.mesh.indices),
                        usage: wgpu::BufferUsages::INDEX,
                    }
                );

                Mesh {
                    name: path.to_str().unwrap().to_string(),
                    vertex_buffer,
                    index_buffer,
                    num_elements: m.mesh.indices.len() as u32,
                    material: m.mesh.material_id.unwrap_or(0),
                }
            }).collect::<Vec<_>>();

        Model {
            meshes,
            materials,
        }
    }

    pub fn get_texture(&self, texture_name: &String) -> Option<&RegisteredTexture> {
        self.textures.get(texture_name)
    }

    pub fn get_model(&self, model_name: &String) -> Option<&Model> {
        self.models.get(model_name)
    }
}
