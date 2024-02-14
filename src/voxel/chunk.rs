use std::{collections::{HashMap, VecDeque}, hash::Hash, usize, vec};
use bytemuck::cast_box;
use enum_map::{enum_map, EnumMap};
use lazy_static::lazy_static;
use ndarray::{Array3, ArrayBase};
use noise::{NoiseFn, Perlin, Seedable};

use crate::engine;
use std::iter::Iterator;

use super::voxel::{Voxel, VoxelType, Face};

#[derive(Clone)]
pub struct Chunk {
    voxels: Box<[[[Voxel; 16]; 128]; 16]>,
}

impl Chunk {
    pub fn new_empty() -> Self {
        // let mut voxels_1d = Vec::with_capacity(16 * 128 * 16);
        // voxels_1d.resize_with(16 * 128 * 16, || Voxel::create_default_type(VoxelType::Air));

        // let voxels_1d: Box<[Voxel; 16 * 128 * 16]> = voxels_1d.into_boxed_slice().try_into().unwrap();
        // let voxels_3d: Box<[[[Voxel; 16]; 128]; 16]> = cast_box(voxels_1d).unwrap();

        // Self {
        //     voxels: cast_box(voxels_1d.into_boxed_slice()).unwrap(),
        // }

        // TODO: this will break when Voxel becomes to large due to stack size, careful!
        let voxels = core::array::from_fn(|_| {
            core::array::from_fn(|_| {
                core::array::from_fn(|_| {
                    Voxel::create_default_type(VoxelType::Air)
                })
            })
        });

        Self { voxels: Box::new(voxels) }
    }

    pub fn new_plane(y: usize) -> Self {
        let mut chunk = Self::new_empty();

        for x in 0..16 {
            for z in 0..16 {
                chunk.set_voxel(x, y, z, Voxel::create_default_type(VoxelType::Grass));
            }
        }

        chunk
    }

    pub fn new_hill() -> Self {
        let mut chunk = Self::new_empty();

        for x in 0..16 {
            for z in 0..16 {
                let y = x;
                for y in 0..y as usize {
                    chunk.set_voxel(x, y, z, Voxel::create_default_type(VoxelType::Grass));
                }
            }
        }

        chunk
    }

    pub fn new_heightmap<F>(sampler: F) -> Self 
    where F: Fn(usize, usize) -> usize {
        let mut chunk = Self::new_empty();

        for x in 0..16 {
            for z in 0..16 {
                let y = sampler(x, z);
                for y in 0..y {
                    chunk.set_voxel(x, y, z, Voxel::create_default_type(VoxelType::Grass));
                }
            }
        }

        chunk
    }

    pub fn set_voxel(&mut self, x: usize, y: usize, z: usize, voxel: Voxel) {
        self.voxels[x][y][z] = voxel;
    }

    pub fn mod_voxel<F>(&mut self, x: usize, y: usize, z: usize, f: F) where F: Fn(&mut Voxel) {
        f(&mut self.voxels[x][y][z]);
    }

    pub fn get_voxel(&self, x: usize, y: usize, z: usize) -> &Voxel {
        &self.voxels[x][y][z]
    }

    pub fn create_mesh(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        chunk_pos: (i32, i32),
        texture_atlas: &engine::resources::TextureAtlas,
    ) -> engine::model::Mesh {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let (mut chunk_x, mut chunk_z) = chunk_pos;
        chunk_x *= 16;
        chunk_z *= 16;

        for x in 0..16 {
            for y in 0..128 {
                for z in 0..16 {
                    let voxel = self.get_voxel(x, y, z);

                    // skip air voxels (aka empty)
                    if voxel.voxel_type == VoxelType::Air {
                        continue;
                    }

                    // check neighbors
                    let neighbors = [
                        if y < 127 { Some(self.get_voxel(x, y + 1, z)) } else { None }, // above
                        if y > 0 { Some(self.get_voxel(x, y - 1, z)) } else { None }, // below
                        if z > 0 { Some(self.get_voxel(x, y, z - 1)) } else { None }, // north
                        if z < 15 { Some(self.get_voxel(x, y, z + 1)) } else { None }, // south
                        if x < 15 { Some(self.get_voxel(x + 1, y, z)) } else { None }, // east
                        if x > 0 { Some(self.get_voxel(x - 1, y, z)) } else { None }, // west
                    ];

                    for (i, neighbor) in neighbors.iter().enumerate() {
                        let is_face_visible = neighbor.is_none() || neighbor.unwrap().voxel_type == VoxelType::Air;

                        if !is_face_visible {
                            continue;
                        }

                        let face = Face::from_ordinal(i);
                        let texture = FACE_TEXTURES[voxel.voxel_type].get_texture(&face);

                        let (face_verts, face_indicies) = create_face(chunk_x as f32 + x as f32, y as f32, chunk_z as f32 + z as f32, &face, texture, texture_atlas);
                        for index in face_indicies {
                            indices.push(index + vertices.len() as u32);
                        }
                        vertices.extend(face_verts);
                    }
                }
            }
        };

        // println!("starting indices");
        // let mut unique_verts = vertices.clone();
        // unique_verts.dedup();
        // for vert in vertices.iter() {
        //     indices.push(unique_verts.iter()
        //         .position(|v| v == vert).unwrap() as u32);
        // }
        // println!("ending indicies");

        engine::model::Mesh::new(device, queue, format!("Chunk Mesh {},{}", chunk_pos.0, chunk_pos.1).as_str(), &vertices, &indices, 0)
    }
}

pub fn create_face(
    x: f32, y: f32, z: f32,
    face: &Face,
    texture_name: &'static str,
    atlas: &engine::resources::TextureAtlas,
) -> (Vec<engine::model::ModelVertex>, Vec<u32>) {
    let cell = atlas.cells.get(texture_name).unwrap();

    let mut vertices = Vec::new();

    let (norm_x, norm_y, norm_z) = match face {
        Face::Top => (0.0, 1.0, 0.0),
        Face::Bottom => (0.0, -1.0, 0.0),
        Face::North => (0.0, 0.0, -1.0),
        Face::South => (0.0, 0.0, 1.0),
        Face::East => (1.0, 0.0, 0.0),
        Face::West => (-1.0, 0.0, 0.0),
    };

    struct VertexData {
        x: f32,
        y: f32,
        z: f32,
        u: f32,
        v: f32,
    }
    let (top_left, bottom_left, bottom_right, top_right) = match face {
        Face::Top => (
            VertexData { x: x, y: y + 1.0, z: z, u: 0.0, v: 1.0 },
            VertexData { x: x, y: y + 1.0, z: z + 1.0, u: 0.0, v: 0.0 },
            VertexData { x: x + 1.0, y: y + 1.0, z: z + 1.0, u: 1.0, v: 0.0 },
            VertexData { x: x + 1.0, y: y + 1.0, z: z, u: 1.0, v: 1.0 },
        ),
        Face::Bottom => (
            VertexData { x: x, y: y, z: z, u: 0.0, v: 1.0 },
            VertexData { x: x + 1.0, y: y, z: z, u: 0.0, v: 0.0 },
            VertexData { x: x + 1.0, y: y, z: z + 1.0, u: 1.0, v: 0.0 },
            VertexData { x: x, y: y, z: z + 1.0, u: 1.0, v: 1.0 },
        ),
        Face::North => (
            VertexData { x: x, y: y, z: z, u: 0.0, v: 1.0 },
            VertexData { x: x, y: y + 1.0, z: z, u: 0.0, v: 0.0 },
            VertexData { x: x + 1.0, y: y + 1.0, z: z, u: 1.0, v: 0.0 },
            VertexData { x: x + 1.0, y: y, z: z, u: 1.0, v: 1.0 },
        ),
        Face::South => (
            VertexData { x: x + 1.0, y: y, z: z + 1.0, u: 0.0, v: 1.0 },
            VertexData { x: x + 1.0, y: y + 1.0, z: z + 1.0, u: 0.0, v: 0.0 },
            VertexData { x: x, y: y + 1.0, z: z + 1.0, u: 1.0, v: 0.0 },
            VertexData { x: x, y: y, z: z + 1.0, u: 1.0, v: 1.0 },
        ),
        Face::East => (
            VertexData { x: x + 1.0, y: y, z: z, u: 0.0, v: 1.0 },
            VertexData { x: x + 1.0, y: y + 1.0, z: z, u: 0.0, v: 0.0 },
            VertexData { x: x + 1.0, y: y + 1.0, z: z + 1.0, u: 1.0, v: 0.0 },
            VertexData { x: x + 1.0, y: y, z: z + 1.0, u: 1.0, v: 1.0 },
        ),
        Face::West => (
            VertexData { x: x, y: y, z: z + 1.0, u: 0.0, v: 1.0 },
            VertexData { x: x, y: y + 1.0, z: z + 1.0, u: 0.0, v: 0.0 },
            VertexData { x: x, y: y + 1.0, z: z, u: 1.0, v: 0.0 },
            VertexData { x: x, y: y, z: z, u: 1.0, v: 1.0 },
        ),
    };
    
    let vertices_data = [
        &top_left, &bottom_left, &bottom_right, &top_right,
    ];
    let indicies = vec![
        0, 1, 2,
        0, 2, 3,
    ];

    for vert in vertices_data.iter() {
        let (u, v) = atlas.absolute_uv_cell(cell, vert.u, vert.v);

        vertices.push(engine::model::ModelVertex {
            position: [vert.x, vert.y, vert.z],
            normal: [norm_x, norm_y, norm_z],
            tex_coords: [u, v],
            bitangent: [0.0, 0.0, 0.0],
            tangent: [0.0, 0.0, 0.0],
        });
    }

    (vertices, indicies)
}

lazy_static! {
    static ref FACE_TEXTURES: EnumMap<VoxelType, VoxelFaceTextures> = enum_map! {
        VoxelType::Air => VoxelFaceTextures::new_uniform("air"),
        VoxelType::Grass => VoxelFaceTextures {
            top: "grass_top.atlas.png",
            bottom: "dirt.atlas.png",
            north: "grass_side.atlas.png",
            south: "grass_side.atlas.png",
            east: "grass_side.atlas.png",
            west: "grass_side.atlas.png",
        },
    };
}

#[derive(Clone, Debug)]
struct VoxelFaceTextures {
    top: &'static str,
    bottom: &'static str,
    north: &'static str,
    south: &'static str,
    east: &'static str,
    west: &'static str,
}

impl VoxelFaceTextures {
    fn new_uniform(texture: &'static str) -> Self {
        Self {
            top: texture,
            bottom: texture,
            north: texture,
            south: texture,
            east: texture,
            west: texture,
        }
    }

    fn get_texture(&self, face: &Face) -> &'static str {
        match face {
            Face::Top => &self.top,
            Face::Bottom => &self.bottom,
            Face::North => &self.north,
            Face::South => &self.south,
            Face::East => &self.east,
            Face::West => &self.west,
        }
    }
}

pub struct ChunkManager {
    chunks: HashMap<i64, ChunkState>,
    center_chunk: (i32, i32),
    render_distance: usize,
}

impl ChunkManager {
    pub fn new(render_distance: usize) -> Self {
        let mut chunks = HashMap::new();
        Self::fill_map_renderdistance(&mut chunks, (0, 0), render_distance);

        Self {
            chunks,
            center_chunk: (0, 0),
            render_distance,
        }
    }

    pub fn get_chunk(&self, x: i32, z: i32) -> &ChunkState {
        &self.chunks.get(&Self::pack_coordinates(x, z)).unwrap()
    }

    pub fn get_chunk_mut(&mut self, x: i32, z: i32) -> &mut ChunkState {
        self.chunks.get_mut(&Self::pack_coordinates(x, z)).unwrap()
    }

    pub fn get_center_chunk(&self) -> &ChunkState {
        &self.get_chunk(self.center_chunk.0, self.center_chunk.1)
    }

    pub fn get_center_chunk_mut(&mut self) -> &mut ChunkState {
        self.get_chunk_mut(self.center_chunk.0, self.center_chunk.1)
    }

    pub fn set_center_chunk(&mut self, x: i32, z: i32) {
        if self.center_chunk == (x, z) {
            return;
        }

        self.center_chunk = (x, z);

        // completely remove chunks outside of render distance from the map
        let mut to_remove = Vec::new();
        for coords in self.chunks.keys().into_iter() {
            let (cx, cz) = Self::unpack_coordinates(*coords);
            if (cx - x).abs() > self.render_distance as i32 || (cz - z).abs() > self.render_distance as i32 {
                to_remove.push(*coords);
            }
        }
        for coords in to_remove {
            self.chunks.remove(&coords);
        }

        // load all chunks within render distance
        Self::fill_map_renderdistance(&mut self.chunks, self.center_chunk, self.render_distance)
    }

    pub fn generate_chunk_meshes(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, texture_atlas: &engine::resources::TextureAtlas) {
        let total_chunks = self.chunks.len();

        for (i, (coords, chunk_state)) in self.chunks.iter_mut().enumerate() {
            let chunk_pos = Self::unpack_coordinates(*coords);
            if let ChunkState::Loaded(LoadedChunk::NoMesh { chunk }) = chunk_state {
                let mesh = chunk.create_mesh(device, queue, chunk_pos, texture_atlas);
                *chunk_state = ChunkState::Loaded(LoadedChunk::MeshGenerated { chunk: chunk.clone(), mesh });

                println!("Loaded chunk {}/{}", i, total_chunks);
            }
        }
    }

    pub fn get_all_chunks(&self) -> Vec<&ChunkState> {
        self.chunks.values().collect()
    }

    pub fn pos_to_chunk_coords(x: f32, z: f32) -> (i32, i32) {
        (x.floor() as i32 / 16, z.floor() as i32 / 16)
    }

    fn fill_map_renderdistance(chunks: &mut HashMap<i64, ChunkState>, center_chunk: (i32, i32), render_distance: usize) {
        let perlin = Perlin::new(1);

        for cx in (center_chunk.0 - render_distance as i32)..(center_chunk.0 + render_distance as i32) {
            for cz in (center_chunk.1 - render_distance as i32)..(center_chunk.1 + render_distance as i32) {
                if !chunks.contains_key(&Self::pack_coordinates(cx, cz)) {
                    println!("{}", perlin.get([42.4, 28.9]));
                    // let state = ChunkState::LoadScheduled;
                    // for now lets block the main thread
                    let state = ChunkState::Loaded(
                        LoadedChunk::NoMesh { 
                            chunk: Chunk::new_heightmap(|x, z| (perlin.get([cx as f64 * x as f64 * 100.0, cz as f64 * z as f64 * 100.0]) + 1.0 * 12.0) as usize) 
                        }
                    );

                    chunks.insert(Self::pack_coordinates(cx, cz), state);
                }
            }
        }
    }

    fn pack_coordinates(x: i32, z: i32) -> i64 {
        (x as i64) << 32 | (z as i64)
    }

    fn unpack_coordinates(packed: i64) -> (i32, i32) {
        let x = (packed >> 32) as i32;
        let z = packed as i32;
        (x, z)
    }
}

pub enum ChunkState {
    LoadScheduled,
    Loading,
    Loaded(LoadedChunk),
    Failed,
}

pub enum LoadedChunk {
    MeshGenerated { chunk: Chunk, mesh: engine::model::Mesh },
    NoMesh { chunk: Chunk },
}