use std::{collections::{HashMap, VecDeque}, hash::Hash, sync::{mpsc::{Receiver, Sender}, Arc, Mutex}, usize, vec};
use bytemuck::cast_box;
use enum_map::{enum_map, EnumMap};
use lazy_static::lazy_static;
use ndarray::{Array3, ArrayBase};
use noise::{NoiseFn, Perlin, Seedable};
use threadpool::ThreadPool;

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
    where F: Fn(f32, f32) -> f32 {
        let mut chunk = Self::new_empty();

        for x in 0..16 {
            for z in 0..16 {
                let y = sampler(x as f32, z as f32);
                for y in 0..y.floor() as usize {
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
        chunk_pos: (i32, i32),
        texture_atlas: &engine::resources::TextureAtlas,
    ) -> engine::model::MeshData {
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

        engine::model::MeshData {
            vertices,
            indices,
        }
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
    thread_pool: ThreadPool,
    chunk_thread_sender: Sender<(i64, ChunkState)>,
    chunk_thread_receiver: Receiver<(i64, ChunkState)>,
}

impl ChunkManager {
    pub fn new(render_distance: usize, workers: usize) -> Self {
        let chunks = HashMap::new();
        let thread_pool = ThreadPool::new(workers);
        let (chunk_thread_sender, chunk_thread_receiver) = std::sync::mpsc::channel();

        Self {
            chunks,
            center_chunk: (0, 0),
            render_distance,
            thread_pool,
            chunk_thread_sender,
            chunk_thread_receiver,
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

    pub fn set_center_chunk(&mut self, x: i32, z: i32, texture_atlas: &engine::resources::TextureAtlas) {
        if self.center_chunk == (x, z) {
            return;
        }

        self.center_chunk = (x, z);

        // completely remove chunks outside of render distance from the map
        let mut to_remove = Vec::new();
        for coords in self.chunks.keys().into_iter() {
            let (cx, cz) = Self::unpack_coordinates(*coords);
            let distance = ((cx - x).pow(2) + (cz - z).pow(2)) as f32;
            if distance.sqrt() > self.render_distance as f32 {
                to_remove.push(*coords);
            }
        }
        for coords in to_remove {
            self.chunks.remove(&coords);
        }

        // load all chunks within render distance
        self.fill_map_renderdistance(texture_atlas);
    }

    pub fn upload_chunk_meshes(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, texture_atlas: &engine::resources::TextureAtlas) {
        let total_chunks = self.chunks.len();

        for (i, (coords, chunk_state)) in self.chunks.iter_mut().enumerate() {
            let chunk_pos = Self::unpack_coordinates(*coords);
            if let ChunkState::Loaded(LoadedChunk::MeshCreated { chunk, mesh_data }) = chunk_state {
                let mesh = engine::model::Mesh::from_data(device, queue, format!("Chunk Mesh {},{}", chunk_pos.0, chunk_pos.1).as_str(), mesh_data, 0);
                *chunk_state = ChunkState::Loaded(LoadedChunk::MeshUploaded { chunk: chunk.clone(), mesh });
            }
        }
    }

    pub fn get_all_chunks(&self) -> Vec<&ChunkState> {
        self.chunks.values().collect()
    }

    pub fn pos_to_chunk_coords(x: f32, z: f32) -> (i32, i32) {
        (x.floor() as i32 / 16, z.floor() as i32 / 16)
    }

    fn fill_map_renderdistance(&mut self, texture_atlas: &engine::resources::TextureAtlas) {
        let perlin = Perlin::new(1);
        let render_distance = self.render_distance as i32;

        // fill the map with chunks within render distance, in a circle around the center chunk
        // make sure to use pi for circularity
        // use euclidean tile-space distance to determine if a chunk is within render distance
        for x in -render_distance as i32..=render_distance as i32 {
            for z in -render_distance as i32..=render_distance as i32 {
                let chunk_x = self.center_chunk.0 + x;
                let chunk_z = self.center_chunk.1 + z;
                let distance = (x.pow(2) + z.pow(2)) as f32;
                if distance.sqrt() > render_distance as f32 {
                    continue;
                }

                match self.chunks.get(&Self::pack_coordinates(chunk_x, chunk_z)) {
                    Some(ChunkState::Loaded(_)) => continue,
                    Some(ChunkState::LoadScheduled) => continue,
                    Some(ChunkState::Loading) => continue,
                    Some(ChunkState::Failed) => continue,
                    None => (),
                }

                self.chunks.insert(Self::pack_coordinates(chunk_x, chunk_z), ChunkState::LoadScheduled);
                let sender = self.chunk_thread_sender.clone();

                self.thread_pool.execute(move || {
                    sender.send((Self::pack_coordinates(chunk_x, chunk_z), ChunkState::Loading)).unwrap();

                    let chunk = Chunk::new_heightmap(|vx, vz| {
                        let scale = 100.0;
                        let noise = (perlin.get([((chunk_x * 16) as f64 + vx as f64) / scale + 0.5, ((chunk_z * 16) as f64 + vz as f64) / scale + 0.5]) + 1.0) * 40.0;
                        noise as f32
                    });
                    let mesh_data = chunk.create_mesh((chunk_x, chunk_z), texture_atlas);

                    let state = ChunkState::Loaded(
                        LoadedChunk::MeshCreated {
                            chunk,
                            mesh_data,
                        },
                    );

                    sender.send((Self::pack_coordinates(chunk_x, chunk_z), state)).unwrap();
                });
            }
        }
    }

    pub fn receive_generated_chunks(&mut self) {
        while let Ok((coords, state)) = self.chunk_thread_receiver.try_recv() {
            self.chunks.insert(coords, state);
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
    MeshUploaded { chunk: Chunk, mesh: engine::model::Mesh },
    MeshCreated { chunk: Chunk, mesh_data: engine::model::MeshData },
}