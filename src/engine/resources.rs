use crate::engine::model::{Material, Mesh, Model, ModelVertex};
use crate::engine::texture::{RegisteredTexture, Texture};
use cgmath::{Vector2, Vector3};
use std::collections::HashMap;
use std::fs;
use std::io::{BufReader, Cursor};
use std::path::Path;
use wgpu::util::DeviceExt;

pub struct ResourceManager {
    textures: HashMap<String, RegisteredTexture>,
    models: HashMap<String, Model>,
}

impl ResourceManager {
    const TEXTURE_EXTENSIONS: &'static [&'static str] = &["png", "webp"];
    const MODEL_EXTENSIONS: &'static [&'static str] = &["obj"];

    pub fn new(
        directory: &Path,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let mut textures: HashMap<String, RegisteredTexture> = HashMap::new();
        let mut models: HashMap<String, Model> = HashMap::new();

        let paths = fs::read_dir(directory).expect("can't read dir");

        for path in paths {
            let file = path.expect("cant get path").path();
            let extension = file.extension().expect("cant get extension");

            let file_name = file
                .file_name()
                .expect("cant get file name")
                .to_str()
                .expect("cant get file name as string");
            let file_path_relative = file.strip_prefix(directory).expect("cant strip prefix");

            let file_data = fs::read(&file).expect("cant read data");
            let file_data_arr = file_data.as_slice();

            if Self::TEXTURE_EXTENSIONS.iter().any(|&ext| ext == extension) {
                //let texture = Self::create_bind_texture(file_data_arr, file_path_relative, device, queue, bind_group_layout);
                //textures.insert(file_path_relative.to_str().unwrap().to_string(), texture);
            } else if Self::MODEL_EXTENSIONS.iter().any(|&ext| ext == extension) {
                let model = Self::create_model(
                    file_data_arr,
                    file_path_relative,
                    directory,
                    device,
                    queue,
                    bind_group_layout,
                );
                models.insert(file_path_relative.to_str().unwrap().to_string(), model);
            }
        }

        Self { textures, models }
    }

    fn create_bind_texture(
        texture_data: &[u8],
        file_path: &Path,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> RegisteredTexture {
        let texture_id = file_path.to_str().unwrap();
        let texture = Texture::from_bytes(device, queue, texture_data, texture_id, false).unwrap();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: bind_group_layout,
            entries: &[
                // filling in the data described by the layout
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture.sampler),
                },
            ],
            label: None,
        });

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
        )
        .unwrap();

        let mut materials = Vec::new();
        for m in obj_materials.unwrap() {
            let diffuse_texture = m.diffuse_texture.map(|path| {
                let path = root_directory.join(path);
                let bytes = fs::read(&path).expect("Couldn't load diffuse texture");
                Texture::from_bytes(device, queue, bytes.as_slice(), path.to_str().unwrap(), false)
                    .unwrap()
            });

            let normal_texture = m.normal_texture.map(|path| {
                let path = root_directory.join(path);
                let bytes = fs::read(&path).expect("Couldn't load normal texture");
                Texture::from_bytes(device, queue, bytes.as_slice(), path.to_str().unwrap(), true)
                    .unwrap()
            });

            let ambient_texture = m.ambient_texture.map(|path| {
                let path = root_directory.join(path);
                let bytes = fs::read(&path).expect("Couldn't load ambient texture");
                Texture::from_bytes(device, queue, bytes.as_slice(), path.to_str().unwrap(), false)
                    .unwrap()
            });

            let specular_texture = m.specular_texture.map(|path| {
                let path = root_directory.join(path);
                let bytes = fs::read(&path).expect("Couldn't load specular texture");
                Texture::from_bytes(device, queue, bytes.as_slice(), path.to_str().unwrap(), false)
                    .unwrap()
            });

            materials.push(Material::new(
                device,
                queue,
                &m.name.as_str(),
                diffuse_texture,
                normal_texture,
                specular_texture,
                bind_group_layout,
            ))
        }

        let meshes = models
            .into_iter()
            .map(|m| {
                let mut vertices = (0..m.mesh.positions.len() / 3)
                    .map(|i| {
                        let position: [f32; 3] = [
                            m.mesh.positions[i * 3],
                            m.mesh.positions[i * 3 + 1],
                            m.mesh.positions[i * 3 + 2],
                        ];
                        let tex_coords: [f32; 2] = if !m.mesh.texcoords.is_empty() {
                            [m.mesh.texcoords[i * 2], 1.0 - m.mesh.texcoords[i * 2 + 1]]
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
                            tangent: [0.0; 3], // TODO
                            bitangent: [0.0; 3],
                        }
                    })
                    .collect::<Vec<_>>();

                let indicies = &m.mesh.indices;
                let mut triangles_included = vec![0; vertices.len()];

                // calculate tangents and bitangents
                for c in indicies.chunks(3) {
                    let v0 = vertices[c[0] as usize];
                    let v1 = vertices[c[1] as usize];
                    let v2 = vertices[c[2] as usize];

                    let pos0: Vector3<_> = v0.position.into();
                    let pos1: Vector3<_> = v1.position.into();
                    let pos2: Vector3<_> = v2.position.into();

                    let uv0: Vector2<_> = v0.tex_coords.into();
                    let uv1: Vector2<_> = v1.tex_coords.into();
                    let uv2: Vector2<_> = v2.tex_coords.into();

                    // calculate edges of the triangle
                    let delta_pos1 = pos1 - pos0;
                    let delta_pos2 = pos2 - pos0;

                    // this will give us the direction of the UVs
                    let delta_uv1 = uv1 - uv0;
                    let delta_uv2 = uv2 - uv0;

                    // Solving the following system of equations will
                    // give us the tangent and bitangent.
                    //     delta_pos1 = delta_uv1.x * T + delta_u.y * B
                    //     delta_pos2 = delta_uv2.x * T + delta_uv2.y * B
                    // Luckily, the place I found this equation provided
                    // the solution!
                    let r = 1.0 / (delta_uv1.x * delta_uv2.y - delta_uv1.y * delta_uv2.x);
                    let tangent = (delta_pos1 * delta_uv2.y - delta_pos2 * delta_uv1.y) * r;
                    // We flip the bitangent to enable right-handed normal
                    // maps with wgpu texture coordinate system
                    let bitangent = (delta_pos2 * delta_uv1.x - delta_pos1 * delta_uv2.x) * -r;

                    // We'll use the same tangent/bitangent for each vertex in the triangle
                    vertices[c[0] as usize].tangent = (tangent + Vector3::from(vertices[c[0] as usize].tangent)).into();
                    vertices[c[1] as usize].tangent = (tangent + Vector3::from(vertices[c[1] as usize].tangent)).into();
                    vertices[c[2] as usize].tangent = (tangent + Vector3::from(vertices[c[2] as usize].tangent)).into();
                    vertices[c[0] as usize].bitangent = (bitangent + Vector3::from(vertices[c[0] as usize].bitangent)).into();
                    vertices[c[1] as usize].bitangent = (bitangent + Vector3::from(vertices[c[1] as usize].bitangent)).into();
                    vertices[c[2] as usize].bitangent = (bitangent + Vector3::from(vertices[c[2] as usize].bitangent)).into();

                    // Used to average the tangents/bitangents
                    triangles_included[c[0] as usize] += 1;
                    triangles_included[c[1] as usize] += 1;
                    triangles_included[c[2] as usize] += 1;
                }

                // Average the tangents/bitangents
                for (i, n) in triangles_included.into_iter().enumerate() {
                    let denom = 1.0 / n as f32;
                    let v = &mut vertices[i];
                    v.tangent = (Vector3::from(v.tangent) * denom).into();
                    v.bitangent = (Vector3::from(v.bitangent) * denom).into();
                }

                let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("{:?} Vertex Buffer", path.to_str().unwrap())),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("{:?} Index Buffer", path.to_str().unwrap())),
                    contents: bytemuck::cast_slice(&m.mesh.indices),
                    usage: wgpu::BufferUsages::INDEX,
                });

                Mesh {
                    name: path.to_str().unwrap().to_string(),
                    vertex_buffer,
                    index_buffer,
                    num_elements: m.mesh.indices.len() as u32,
                    material: m.mesh.material_id.unwrap_or(0),
                }
            })
            .collect::<Vec<_>>();

        Model { meshes, materials }
    }

    pub fn get_texture(&self, texture_name: &String) -> Option<&RegisteredTexture> {
        self.textures.get(texture_name)
    }

    pub fn get_model(&self, model_name: &String) -> Option<&Model> {
        self.models.get(model_name)
    }
}
