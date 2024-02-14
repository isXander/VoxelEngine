use std::{collections::HashMap, sync::{mpsc::{Receiver, Sender}, Arc}, usize, vec};
use enum_map::{enum_map, EnumMap};
use lazy_static::lazy_static;
use noise::{NoiseFn, Perlin};
use threadpool::ThreadPool;

use crate::engine;
use std::iter::Iterator;
use crate::engine::model::{Mesh, MeshData};
use crate::engine::resources::TextureAtlas;

use super::voxel::{Voxel, Type, Face};

#[derive(Clone, Debug)]
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
                    Voxel::create_default_type(Type::Air)
                })
            })
        });

        Self { voxels: Box::new(voxels) }
    }

    pub fn new_plane(y: usize) -> Self {
        let mut chunk = Self::new_empty();

        for x in 0..16 {
            for z in 0..16 {
                chunk.set_voxel(x, y, z, Voxel::create_default_type(Type::Dirt));
            }
        }

        chunk
    }

    pub fn new_hill() -> Self {
        let mut chunk = Self::new_empty();

        for x in 0..16 {
            for z in 0..16 {
                let y = x;
                for y in 0..y {
                    chunk.set_voxel(x, y, z, Voxel::create_default_type(Type::Dirt));
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
                    let voxel_type = match y {
                        0..=15 => Some(Type::Sand),
                        16..=50 => Some(Type::Dirt),
                        51..=128 => Some(Type::Snow),
                        _ => None,
                    };

                    if let Some(voxel_type) = voxel_type {
                        chunk.set_voxel(x, y, z, Voxel::create_default_type(voxel_type));
                    }
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
        neighbour_left: &Chunk,
        neighbour_right: &Chunk,
        neighbour_up: &Chunk,
        neighbour_down: &Chunk,
        texture_atlas: &Arc<engine::resources::TextureAtlas>,
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
                    if voxel.voxel_type == Type::Air {
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
                        let face = Face::from_ordinal(i);

                        let neighbor = neighbor.or_else(|| {
                            // must be edge of chunk on x or z, get neighboring voxel from neighboring chunk
                            match face {
                                Face::North => Some(neighbour_up.get_voxel(x, y, 15)),
                                Face::South => Some(neighbour_down.get_voxel(x, y, 0)),
                                Face::East => Some(neighbour_right.get_voxel(0, y, z)),
                                Face::West => Some(neighbour_left.get_voxel(15, y, z)),
                                _ => None,
                            }
                        });

                        let is_face_visible = neighbor.is_none() || neighbor.unwrap().voxel_type == Type::Air;

                        if !is_face_visible {
                            continue;
                        }

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

        MeshData {
            vertices,
            indices,
        }
    }
}

fn create_face(
    x: f32, y: f32, z: f32,
    face: &Face,
    texture_name: &'static str,
    atlas: &Arc<engine::resources::TextureAtlas>,
) -> (Vec<engine::model::ModelVertex>, Vec<u32>) {
    let cell = atlas.cells.get(texture_name).unwrap();

    let mut vertices = Vec::new();

    let (normal_vector, tangent_vector, bitangent_vector) = match face {
        Face::Top => ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
        Face::Bottom => ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        Face::North => ([0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        Face::South => ([0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        Face::East => ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        Face::West => ([-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
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
    let indices = vec![
        0, 1, 2,
        0, 2, 3,
    ];

    for vert in vertices_data.iter() {
        let (u, v) = atlas.absolute_uv_cell(cell, vert.u, vert.v);

        vertices.push(engine::model::ModelVertex {
            position: [vert.x, vert.y, vert.z],
            normal: normal_vector,
            tex_coords: [u, v],
            bitangent: tangent_vector,
            tangent: bitangent_vector,
        });
    }

    (vertices, indices)
}

lazy_static! {
    static ref FACE_TEXTURES: EnumMap<Type, VoxelFaceTextures> = enum_map! {
        Type::Air => VoxelFaceTextures::new_uniform("air"),
        Type::Dirt => VoxelFaceTextures::new_uniform("dirt.atlas.png"),
        Type::Sand => VoxelFaceTextures::new_uniform("sand.atlas.png"),
        Type::Snow => VoxelFaceTextures::new_uniform("snow.atlas.png"),
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
            Face::Top => self.top,
            Face::Bottom => self.bottom,
            Face::North => self.north,
            Face::South => self.south,
            Face::East => self.east,
            Face::West => self.west,
        }
    }
}

pub struct ChunkManager {
    chunks: HashMap<u64, ChunkState>,
    center_chunk: (i32, i32),
    render_distance: usize,

    thread_pool: ThreadPool,

    chunkgen_thread_sender: Sender<(u64, ChunkState)>,
    chunkgen_thread_receiver: Receiver<(u64, ChunkState)>,

    chunkmesh_out_sender: Sender<(u64, MeshData)>,
    chunkmesh_out_receiver: Receiver<(u64, MeshData)>,
}

impl ChunkManager {
    pub fn new(render_distance: usize, workers: usize) -> Self {
        let chunks = HashMap::new();
        let thread_pool = ThreadPool::new(workers);

        let (chunkgen_thread_sender, chunkgen_thread_receiver) = std::sync::mpsc::channel();
        let (chunkmesh_out_sender, chunkmesh_out_receiver) = std::sync::mpsc::channel();

        Self {
            chunks,
            center_chunk: (0, 0),
            render_distance,
            thread_pool,
            chunkgen_thread_sender,
            chunkgen_thread_receiver,


            chunkmesh_out_sender,
            chunkmesh_out_receiver,
        }
    }

    pub fn get_chunk(&self, x: i32, z: i32) -> Option<&ChunkState> {
        self.chunks.get(&Self::pack_coordinates(x, z))
    }

    pub fn get_chunk_mut(&mut self, x: i32, z: i32) -> Option<&mut ChunkState> {
        self.chunks.get_mut(&Self::pack_coordinates(x, z))
    }

    pub fn get_center_chunk(&self) -> Option<&ChunkState> {
        self.get_chunk(self.center_chunk.0, self.center_chunk.1)
    }

    pub fn get_center_chunk_mut(&mut self) -> Option<&mut ChunkState> {
        self.get_chunk_mut(self.center_chunk.0, self.center_chunk.1)
    }

    pub fn set_center_chunk(&mut self, x: i32, z: i32) {
        if self.center_chunk == (x, z) {
            return;
        }

        self.center_chunk = (x, z);

        // completely remove chunks outside of render distance from the map
        let mut to_remove = Vec::new();
        for coords in self.chunks.keys() {
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
        self.fill_map_renderdistance();
    }

    pub fn get_all_chunks(&self) -> Vec<&ChunkState> {
        self.chunks.values().collect()
    }

    pub fn pos_to_chunk_coords(x: f32, z: f32) -> (i32, i32) {
        (x.floor() as i32 / 16, z.floor() as i32 / 16)
    }

    pub fn fill_map_renderdistance(&mut self) {
        let perlin = Perlin::new(1);
        let render_distance = self.render_distance as i32;

        // fill the map with chunks within render distance, in a circle around the center chunk
        // make sure to use pi for circularity
        // use euclidean tile-space distance to determine if a chunk is within render distance
        for x in -render_distance..=render_distance {
            for z in -render_distance..=render_distance {
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

                let sender = self.chunkgen_thread_sender.clone();

                self.thread_pool.execute(move || {
                    sender.send((Self::pack_coordinates(chunk_x, chunk_z), ChunkState::Loading)).unwrap();

                    let chunk = Chunk::new_heightmap(|vx, vz| {
                        let scale = 100.0;
                        let noise = (perlin.get([((chunk_x * 16) as f64 + vx as f64) / scale + 0.5, ((chunk_z * 16) as f64 + vz as f64) / scale + 0.5]) + 1.0) * 40.0;
                        noise as f32 + 1.0
                    });

                    let state = ChunkState::Loaded(
                        LoadedChunk::Stored {
                            chunk: Arc::new(chunk),
                        },
                    );

                    sender.send((Self::pack_coordinates(chunk_x, chunk_z), state)).unwrap();
                });
            }
        }
    }

    pub fn receive_generated_chunks(&mut self, texture_atlas: &Arc<TextureAtlas>) {
        let mut new_chunks = Vec::new();

        while let Ok(received) = self.chunkgen_thread_receiver.try_recv() {
            new_chunks.push(received);
        }

        // sort new chunks from closest to center to farthest, also place new chunks adjacent to eachother
        new_chunks.sort_by(|(coords_a, _), (coords_b, _)| {
            let (ax, az) = Self::unpack_coordinates(*coords_a);
            let (bx, bz) = Self::unpack_coordinates(*coords_b);
            let distance_a = ((ax - self.center_chunk.0).pow(2) + (az - self.center_chunk.1).pow(2)) as f32;
            let distance_b = ((bx - self.center_chunk.0).pow(2) + (bz - self.center_chunk.1).pow(2)) as f32;
            distance_a.partial_cmp(&distance_b).unwrap()
        });

        for (coords, state) in new_chunks {
            let (x, z) = Self::unpack_coordinates(coords);

            self.chunks.insert(coords, state);
            let state = self.chunks.get(&coords).unwrap();

            if let ChunkState::Loaded(LoadedChunk::Stored {..}) = state {
                // once chunk has generated, check its neighbours if they are now able to mesh, with all 4 of their neighbours

                // loop thru newly gened's neighbours
                for (nx, nz) in [(0, 1), (0, -1), (1, 0), (-1, 0)] { // diagonal neighbours aren't needed
                    let (nx, nz) = (x + nx, z + nz);
                    let neighbour = self.get_chunk(nx, nz);

                    if let Some(ChunkState::Loaded(LoadedChunk::Stored { chunk: neighbour_chunk })) = neighbour {
                        // we found a stored neighbour, now check if this chunk now has all 4 neighbours

                        let surrounding_neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)].iter()
                            .filter_map(|(nnx, nnz)| {
                                let (nnx, nnz) = (nx + nnx, nz + nnz);
                                let nn_state = self.get_chunk(nnx, nnz);

                                match nn_state {
                                    Some(ChunkState::Loaded(LoadedChunk::Stored { chunk })) => Some(chunk),
                                    Some(ChunkState::Loaded(LoadedChunk::Meshed { chunk, ..})) => Some(chunk),
                                    _ => None,
                                }
                            }).collect::<Vec<_>>();

                        let neighbour_is_surrounded = surrounding_neighbors.len() == 4;

                        // this neighbour can now be meshed
                        if neighbour_is_surrounded {
                            let nn_down = surrounding_neighbors.get(0).unwrap();
                            let nn_up = surrounding_neighbors.get(1).unwrap();
                            let nn_right = surrounding_neighbors.get(2).unwrap();
                            let nn_left = surrounding_neighbors.get(3).unwrap();

                            self.dispatch_chunk_mesh(
                                nx, nz,
                                neighbour_chunk,
                                nn_left, nn_right, nn_up, nn_down,
                                texture_atlas,
                            )
                        }
                    }
                };
            }
        }
    }

    fn dispatch_chunk_mesh(
        &self,
        chunk_x: i32, chunk_z: i32,
        chunk: &Arc<Chunk>,
        neighbour_left: &Arc<Chunk>,
        neighbour_right: &Arc<Chunk>,
        neighbour_up: &Arc<Chunk>,
        neighbour_down: &Arc<Chunk>,
        texture_atlas: &Arc<TextureAtlas>
    ) {
        let sender = self.chunkmesh_out_sender.clone();
        let chunk = chunk.clone();
        let neighbour_left = neighbour_left.clone();
        let neighbour_right = neighbour_right.clone();
        let neighbour_up = neighbour_up.clone();
        let neighbour_down = neighbour_down.clone();
        let texture_atlas = texture_atlas.clone();

        self.thread_pool.execute(move || {
            let mesh_data = chunk.create_mesh(
                (chunk_x, chunk_z),
                &neighbour_left,
                &neighbour_right,
                &neighbour_up,
                &neighbour_down,
                &texture_atlas
            );
            let packed = Self::pack_coordinates(chunk_x, chunk_z);

            sender.send((packed, mesh_data)).unwrap();
        });
    }

    pub fn upload_chunk_meshes(&mut self, device: &wgpu::Device) {
        while let Ok((coords, mesh_data)) = self.chunkmesh_out_receiver.try_recv() {
            match self.chunks.get_mut(&coords) {
                Some(chunk_state) => {
                    if let ChunkState::Loaded(LoadedChunk::Stored { chunk }) = chunk_state {
                        let mesh = Mesh::from_data(device, "Chunk Mesh", &mesh_data, 0);
                        *chunk_state = ChunkState::Loaded(LoadedChunk::Meshed { chunk: chunk.clone(), mesh });
                    }
                },
                None => eprintln!("Chunk mesh upload failed, chunk not found in map"),
            }

        }
    }

    fn pack_coordinates(x: i32, z: i32) -> u64 {
        (x as u32 as u64) << 32 | (z as u32 as u64)
    }

    fn unpack_coordinates(packed: u64) -> (i32, i32) {
        let x = (packed >> 32) as i32;
        let z = (packed & 0xFFFFFFFF) as i32;
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
    Meshed { chunk: Arc<Chunk>, mesh: engine::model::Mesh },
    Stored { chunk: Arc<Chunk> },
}