use crate::engine::model::{Material, Mesh, Model, ModelVertex};
use crate::engine::texture::{RegisteredTexture, Texture};
use cgmath::{Vector2, Vector3};
use image::{GenericImage, GenericImageView};
use std::collections::HashMap;
use std::fs;
use std::io::{BufReader, Cursor};
use std::path::Path;
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub struct ResourceManager {
    textures: HashMap<String, RegisteredTexture>,
    models: HashMap<String, Model>,
    texture_atlas: Arc<TextureAtlas>,
}

impl ResourceManager {
    const TEXTURE_EXTENSIONS: &'static [&'static str] = &[".tex.png", ".tex.webp"];
    const MODEL_EXTENSIONS: &'static [&'static str] = &[".obj"];
    const ATLAS_EXTENSIONS: &'static [&'static str] = &[".atlas.png"];

    pub fn new(
        directory: &Path,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let mut textures: HashMap<String, RegisteredTexture> = HashMap::new();
        let mut models: HashMap<String, Model> = HashMap::new();

        let paths = fs::read_dir(directory).expect("can't read dir");

        let mut atlas_images = Vec::new();

        for path in paths {
            let file = path.expect("cant get path").path();

            let file_name = file
                .file_name()
                .expect("cant get file name")
                .to_str()
                .expect("cant get file name as string");
            let file_path_relative = file.strip_prefix(directory).expect("cant strip prefix");

            let file_data = fs::read(&file).expect("cant read data");
            let file_data_arr = file_data.as_slice();

            if Self::TEXTURE_EXTENSIONS.iter().any(|&ext| file_name.ends_with(ext)) {
                let texture = Self::create_bind_texture(file_data_arr, file_path_relative, device, queue, bind_group_layout);
                textures.insert(file_path_relative.to_str().unwrap().to_string(), texture);
            } else if Self::MODEL_EXTENSIONS.iter().any(|&ext| file_name.ends_with(ext)) {
                let model = Self::create_model(
                    file_data_arr,
                    file_path_relative,
                    directory,
                    device,
                    queue,
                    bind_group_layout,
                );
                models.insert(file_path_relative.to_str().unwrap().to_string(), model);
            } else if Self::ATLAS_EXTENSIONS.iter().any(|&ext| file_name.ends_with(ext)) {
                atlas_images.push((file_path_relative.to_str().unwrap().to_string(), file_data));
            }
        }

        let texture_atlas = Arc::new(Self::stitch_textures(device, queue, bind_group_layout, atlas_images));

        Self { textures, models, texture_atlas }
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
                m.name.as_str(),
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

    pub fn stitch_textures(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture_bind_group_layout: &wgpu::BindGroupLayout,
        textures: Vec<(String, Vec<u8>)>
    ) -> TextureAtlas {
        let textures = textures.iter().map(|(name, data)| {
            let img = image::load_from_memory(data).unwrap();
            let rgba = img.to_rgba8();
            let dimensions = img.dimensions();
            if !(dimensions.0 * dimensions.1).is_power_of_two() {
                panic!("Texture dimensions must be a power of two");
            }
            if dimensions.0 != dimensions.1 {
                panic!("Texture dimensions must be square");
            }
            (name, rgba, dimensions)
        }).collect::<Vec<_>>();

        let (largest_w, largest_h) = textures.iter().fold((0, 0), |(lw, lh), (_, _, (w, h))| {
            (lw.max(*w), lh.max(*h))
        });

        let ratio = largest_w as f32 / largest_h as f32;
        let cols_f32 = ((textures.len() as f32).sqrt() / ratio.sqrt()).ceil();
        let rows_f32 = (textures.len() as f32 / cols_f32).ceil();
        let cols = cols_f32 as u32;
        let rows = rows_f32 as u32;
        
        let mut atlas = image::RgbaImage::new(
            largest_w * cols, 
            largest_h * rows,
        );
        let mut cells = HashMap::new();

        for (i, tex) in textures.iter().enumerate() {
            let (name, img, _) = tex;
            let (col, row) = ((i as f32 % cols_f32) as u32, (i as f32 / cols_f32).floor() as u32);

            // scale the texture to the largest texture size
            let img = image::imageops::resize(img, largest_w, largest_h, image::imageops::FilterType::Nearest);

            let (x, y) = (col * largest_w, row * largest_h);
            atlas.copy_from(&img, x, y).unwrap();

            let cell = CellPosition { x: col, y: row };
            cells.insert(name.to_string(), cell);
        };

        let processed_atlas = image::DynamicImage::ImageRgba8(atlas);
        let texture = Texture::from_image(device, queue, &processed_atlas, Some("atlas"), false).unwrap();

        let material = Material::new(
            device,
            queue,
            "atlas",
            Some(texture),
            None,
            None,
            texture_bind_group_layout,
        );

        TextureAtlas {
            material,
            width: largest_w * cols,
            height: largest_h * rows,
            cell_width: largest_w,
            cell_height: largest_h,
            cells,
        }
    }

    pub fn get_texture(&self, texture_name: &String) -> Option<&RegisteredTexture> {
        self.textures.get(texture_name)
    }

    pub fn get_model(&self, model_name: &String) -> Option<&Model> {
        self.models.get(model_name)
    }

    pub fn get_atlas(&self) -> &Arc<TextureAtlas> {
        &self.texture_atlas
    }

}

pub struct TextureAtlas {
    pub material: Material,
    pub width: u32,
    pub height: u32,
    pub cell_width: u32,
    pub cell_height: u32,
    pub cells: HashMap<String, CellPosition>,
}

impl TextureAtlas {
    pub fn absolute_uv_cell(&self, cell: &CellPosition, u: f32, v: f32) -> (f32, f32) {
        let x = cell.x as f32 * self.cell_width as f32 / self.width as f32;
        let y = cell.y as f32 * self.cell_height as f32 / self.height as f32;
        let w = self.cell_width as f32 / self.width as f32;
        let h = self.cell_height as f32 / self.height as f32;
        
        (x + u * w, y + v * h)
    }

    pub fn absolute_uv_tex(&self, texture_name: &String, u: f32, v: f32) -> Option<(f32, f32)> {
        let cell = self.cells.get(texture_name)?;
        Some(self.absolute_uv_cell(cell, u, v))
    }
}

pub struct CellPosition {
    x: u32,
    y: u32,
}
