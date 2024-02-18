use enum_map::{enum_map, EnumMap};
use lazy_static::lazy_static;
use noise::{NoiseFn, Perlin};
use std::collections::{HashSet, VecDeque};
use std::{
    collections::HashMap,
    sync::{
        mpsc::{Receiver, Sender},
        Arc,
    },
    usize, vec,
};
use threadpool::ThreadPool;

use crate::engine;
use crate::engine::model::{Mesh, MeshData};
use crate::engine::resources::TextureAtlas;
use crate::voxel::math::floor_div;
use crate::world::physics::components::Collider;
use rapier3d_f64::{
    geometry::{SharedShape, TriMeshFlags},
    prelude::Point,
};
use std::iter::Iterator;

use super::voxel::{Face, Type, Voxel};

pub const CHUNK_WIDTH: usize = 16;
pub const CHUNK_HEIGHT: usize = 128;

#[derive(Clone, Debug)]
pub struct Chunk {
    voxels: Box<[[[Voxel; CHUNK_WIDTH]; CHUNK_HEIGHT]; CHUNK_WIDTH]>,
}

impl Chunk {
    pub fn new_empty() -> Self {
        // TODO: this will break when Voxel becomes to large due to stack size, careful!
        let voxels = core::array::from_fn(|_| {
            core::array::from_fn(|_| {
                core::array::from_fn(|_| Voxel::create_default_type(Type::Air))
            })
        });

        Self {
            voxels: Box::new(voxels),
        }
    }

    pub fn new_plane(y: usize) -> Self {
        let mut chunk = Self::new_empty();

        for x in 0..CHUNK_WIDTH {
            for z in 0..CHUNK_WIDTH {
                chunk.set_voxel(x, y, z, Voxel::create_default_type(Type::Dirt));
            }
        }

        chunk
    }

    pub fn new_hill() -> Self {
        let mut chunk = Self::new_empty();

        for x in 0..CHUNK_WIDTH {
            for z in 0..CHUNK_WIDTH {
                let y = x;
                for y in 0..y {
                    chunk.set_voxel(x, y, z, Voxel::create_default_type(Type::Dirt));
                }
            }
        }

        chunk
    }

    pub fn new_heightmap<F>(sampler: F) -> Self
    where
        F: Fn(f32, f32) -> f32,
    {
        let mut chunk = Self::new_empty();

        for x in 0..CHUNK_WIDTH {
            for z in 0..CHUNK_WIDTH {
                let y = sampler(x as f32, z as f32);
                for y in 0..y.floor() as usize {
                    let voxel_type = match y {
                        0..=15 => Some(Type::Sand),
                        16..=50 => Some(Type::Dirt),
                        51..=128 => Some(Type::Snow),
                        _ => None,
                    };

                    if let Some(voxel_type) = voxel_type {
                        let voxel = chunk.get_voxel_mut(x, y, z);
                        *voxel = Voxel::create_default_type(voxel_type)
                    }
                }
            }
        }

        chunk
    }

    pub fn mod_voxel<F>(&mut self, x: usize, y: usize, z: usize, f: F)
    where
        F: Fn(&mut Voxel),
    {
        f(&mut self.voxels[x][y][z]);
    }

    pub fn get_voxel(&self, x: usize, y: usize, z: usize) -> &Voxel {
        &self.voxels[x][y][z]
    }

    pub fn get_voxel_mut(&mut self, x: usize, y: usize, z: usize) -> &mut Voxel {
        &mut self.voxels[x][y][z]
    }

    pub fn set_voxel(&mut self, x: usize, y: usize, z: usize, voxel: Voxel) {
        self.voxels[x][y][z] = voxel;
    }

    pub fn create_mesh(
        &self,
        chunk_pos: (i32, i32),
        neighbour_west: &Chunk,
        neighbour_east: &Chunk,
        neighbour_north: &Chunk,
        neighbour_south: &Chunk,
        texture_atlas: &Arc<TextureAtlas>,
        absolute_mesh: bool,
    ) -> MeshData {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let (mut cx, mut cz) = chunk_pos;
        cx *= CHUNK_WIDTH as i32;
        cz *= CHUNK_WIDTH as i32;

        for vx in 0..CHUNK_WIDTH {
            for vy in 0..CHUNK_HEIGHT {
                for vz in 0..CHUNK_WIDTH {
                    let voxel = self.get_voxel(vx, vy, vz);

                    // skip air voxels (aka empty)
                    if voxel.voxel_type == Type::Air {
                        continue;
                    }

                    // check neighbors
                    let neighbors = [
                        if vy < CHUNK_HEIGHT - 1 {
                            Some(self.get_voxel(vx, vy + 1, vz))
                        } else {
                            None
                        }, // above
                        if vy > 0 {
                            Some(self.get_voxel(vx, vy - 1, vz))
                        } else {
                            None
                        }, // below
                        if vz > 0 {
                            Some(self.get_voxel(vx, vy, vz - 1))
                        } else {
                            None
                        }, // north
                        if vz < CHUNK_WIDTH - 1 {
                            Some(self.get_voxel(vx, vy, vz + 1))
                        } else {
                            None
                        }, // south
                        if vx < CHUNK_WIDTH - 1 {
                            Some(self.get_voxel(vx + 1, vy, vz))
                        } else {
                            None
                        }, // east
                        if vx > 0 {
                            Some(self.get_voxel(vx - 1, vy, vz))
                        } else {
                            None
                        }, // west
                    ];

                    for (i, neighbor) in neighbors.iter().enumerate() {
                        let face = Face::from_ordinal(i);

                        let neighbor = neighbor.or_else(|| {
                            // must be edge of chunk on x or z, get neighboring voxel from neighboring chunk
                            match face {
                                Face::North => {
                                    Some(neighbour_north.get_voxel(vx, vy, CHUNK_WIDTH - 1))
                                }
                                Face::South => Some(neighbour_south.get_voxel(vx, vy, 0)),
                                Face::East => Some(neighbour_east.get_voxel(0, vy, vz)),
                                Face::West => Some(neighbour_west.get_voxel(CHUNK_WIDTH - 1, vy, vz)),
                                _ => None,
                            }
                        });

                        let is_face_visible =
                            neighbor.is_none() || neighbor.unwrap().voxel_type == Type::Air;

                        if !is_face_visible {
                            continue;
                        }

                        let texture = FACE_TEXTURES[voxel.voxel_type].get_texture(&face);

                        let face_x = vx as f32 + if absolute_mesh { cx as f32 } else { 0.0 };
                        let face_y = vz as f32 + if absolute_mesh { cz as f32 } else { 0.0 };
                        let (face_verts, face_indicies) = create_face(
                            face_x,
                            vy as f32,
                            face_y,
                            &face,
                            texture,
                            texture_atlas,
                        );
                        for index in face_indicies {
                            indices.push(index + vertices.len() as u32);
                        }
                        vertices.extend(face_verts);
                    }
                }
            }
        }

        MeshData { vertices, indices }
    }
}

fn create_face(
    x: f32,
    y: f32,
    z: f32,
    face: &Face,
    texture_name: &'static str,
    atlas: &Arc<TextureAtlas>,
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
            VertexData {
                x: x,
                y: y + 1.0,
                z: z,
                u: 0.0,
                v: 1.0,
            },
            VertexData {
                x: x,
                y: y + 1.0,
                z: z + 1.0,
                u: 0.0,
                v: 0.0,
            },
            VertexData {
                x: x + 1.0,
                y: y + 1.0,
                z: z + 1.0,
                u: 1.0,
                v: 0.0,
            },
            VertexData {
                x: x + 1.0,
                y: y + 1.0,
                z: z,
                u: 1.0,
                v: 1.0,
            },
        ),
        Face::Bottom => (
            VertexData {
                x: x,
                y: y,
                z: z,
                u: 0.0,
                v: 1.0,
            },
            VertexData {
                x: x + 1.0,
                y: y,
                z: z,
                u: 0.0,
                v: 0.0,
            },
            VertexData {
                x: x + 1.0,
                y: y,
                z: z + 1.0,
                u: 1.0,
                v: 0.0,
            },
            VertexData {
                x: x,
                y: y,
                z: z + 1.0,
                u: 1.0,
                v: 1.0,
            },
        ),
        Face::North => (
            VertexData {
                x: x,
                y: y,
                z: z,
                u: 0.0,
                v: 1.0,
            },
            VertexData {
                x: x,
                y: y + 1.0,
                z: z,
                u: 0.0,
                v: 0.0,
            },
            VertexData {
                x: x + 1.0,
                y: y + 1.0,
                z: z,
                u: 1.0,
                v: 0.0,
            },
            VertexData {
                x: x + 1.0,
                y: y,
                z: z,
                u: 1.0,
                v: 1.0,
            },
        ),
        Face::South => (
            VertexData {
                x: x + 1.0,
                y: y,
                z: z + 1.0,
                u: 0.0,
                v: 1.0,
            },
            VertexData {
                x: x + 1.0,
                y: y + 1.0,
                z: z + 1.0,
                u: 0.0,
                v: 0.0,
            },
            VertexData {
                x: x,
                y: y + 1.0,
                z: z + 1.0,
                u: 1.0,
                v: 0.0,
            },
            VertexData {
                x: x,
                y: y,
                z: z + 1.0,
                u: 1.0,
                v: 1.0,
            },
        ),
        Face::East => (
            VertexData {
                x: x + 1.0,
                y: y,
                z: z,
                u: 0.0,
                v: 1.0,
            },
            VertexData {
                x: x + 1.0,
                y: y + 1.0,
                z: z,
                u: 0.0,
                v: 0.0,
            },
            VertexData {
                x: x + 1.0,
                y: y + 1.0,
                z: z + 1.0,
                u: 1.0,
                v: 0.0,
            },
            VertexData {
                x: x + 1.0,
                y: y,
                z: z + 1.0,
                u: 1.0,
                v: 1.0,
            },
        ),
        Face::West => (
            VertexData {
                x: x,
                y: y,
                z: z + 1.0,
                u: 0.0,
                v: 1.0,
            },
            VertexData {
                x: x,
                y: y + 1.0,
                z: z + 1.0,
                u: 0.0,
                v: 0.0,
            },
            VertexData {
                x: x,
                y: y + 1.0,
                z: z,
                u: 1.0,
                v: 0.0,
            },
            VertexData {
                x: x,
                y: y,
                z: z,
                u: 1.0,
                v: 1.0,
            },
        ),
    };

    let vertices_data = [&top_left, &bottom_left, &bottom_right, &top_right];
    let indices = vec![0, 1, 2, 0, 2, 3];

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
    pub view: ChunkView,
    center_chunk: (i32, i32),
    render_distance: usize,

    thread_pool: ThreadPool,

    chunkgen_thread_sender: Sender<(u64, ChunkState)>,
    chunkgen_thread_receiver: Receiver<(u64, ChunkState)>,

    chunkmesh_out_sender: Sender<(u64, MeshData, Collider)>,
    chunkmesh_out_receiver: Receiver<(u64, MeshData, Collider)>,
}

impl ChunkManager {
    pub fn new(render_distance: usize, workers: usize) -> Self {
        let chunks = ChunkView {
            chunks: HashMap::new(),
            tickets: VecDeque::new(),
            updated_center_chunk: None,
        };
        let thread_pool = ThreadPool::new(workers);

        let (chunkgen_thread_sender, chunkgen_thread_receiver) = std::sync::mpsc::channel();
        let (chunkmesh_out_sender, chunkmesh_out_receiver) = std::sync::mpsc::channel();

        Self {
            view: chunks,
            center_chunk: (0, 0),
            render_distance,
            thread_pool,
            chunkgen_thread_sender,
            chunkgen_thread_receiver,

            chunkmesh_out_sender,
            chunkmesh_out_receiver,
        }
    }

    pub fn get_center_chunk(&self) -> Option<&ChunkState> {
        self.view
            .get_chunk(self.center_chunk.0, self.center_chunk.1)
    }

    pub fn get_center_chunk_mut(&mut self) -> Option<&mut ChunkState> {
        self.view
            .get_chunk_mut(self.center_chunk.0, self.center_chunk.1)
    }

    pub fn set_center_chunk(&mut self, x: i32, z: i32) {
        if self.center_chunk == (x, z) && !self.view.chunks.is_empty() {
            return;
        }

        self.center_chunk = (x, z);

        // completely remove chunks outside of render distance from the map
        let mut to_remove = Vec::new();
        for coords in self.view.chunks.keys() {
            let (cx, cz) = unpack_coordinates(*coords);
            let distance = ((cx - x).pow(2) + (cz - z).pow(2)) as f32;
            if distance.sqrt() > self.render_distance as f32 {
                to_remove.push(*coords);
            }
        }
        for coords in to_remove {
            self.view.chunks.remove(&coords);
        }

        // load all chunks within render distance
        self.fill_map_renderdistance();
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

                match self.view.get_chunk(chunk_x, chunk_z) {
                    Some(ChunkState::Loaded(_)) => continue,
                    Some(ChunkState::LoadScheduled) => continue,
                    Some(ChunkState::Loading) => continue,
                    Some(ChunkState::Failed) => continue,
                    None => (),
                }

                self.view.chunks.insert(
                    pack_coordinates(chunk_x, chunk_z),
                    ChunkState::LoadScheduled,
                );

                let sender = self.chunkgen_thread_sender.clone();

                self.thread_pool.execute(move || {
                    sender
                        .send((pack_coordinates(chunk_x, chunk_z), ChunkState::Loading))
                        .unwrap();

                    let chunk = Chunk::new_heightmap(|vx, vz| {
                        let scale = 100.0;
                        let noise = (perlin.get([
                            ((chunk_x * CHUNK_WIDTH as i32) as f64 + vx as f64) / scale + 0.5,
                            ((chunk_z * CHUNK_WIDTH as i32) as f64 + vz as f64) / scale + 0.5,
                        ]) + 1.0)
                            * 40.0;
                        noise as f32 + 1.0
                    });

                    let state = ChunkState::Loaded(LoadedChunk::Stored {
                        chunk: Arc::new(chunk),
                    });

                    sender
                        .send((pack_coordinates(chunk_x, chunk_z), state))
                        .unwrap();
                });
            }
        }
    }

    pub fn update(&mut self, texture_atlas: &Arc<TextureAtlas>) {
        match self.view.updated_center_chunk.take() {
            Some((chunk_x, chunk_z)) => self.set_center_chunk(chunk_x, chunk_z),
            None => {}
        }

        let mut chunks_to_remesh = HashSet::new();

        while !self.view.tickets.is_empty() {
            if let Some(ticket) = self.view.tickets.pop_front() {
                match ticket {
                    ChunkTicket::ChunkUpdate { chunk_x, chunk_z }
                    | ChunkTicket::VoxelUpdate {
                        chunk_x, chunk_z, ..
                    } => {
                        chunks_to_remesh.insert((chunk_x, chunk_z));
                        chunks_to_remesh.insert((chunk_x - 1, chunk_z));
                        chunks_to_remesh.insert((chunk_x + 1, chunk_z));
                        chunks_to_remesh.insert((chunk_x, chunk_z - 1));
                        chunks_to_remesh.insert((chunk_x, chunk_z + 1));
                    }
                }
            }
        }

        for (chunk_x, chunk_z) in chunks_to_remesh {
            self.remesh_chunk(chunk_x, chunk_z, texture_atlas);
        }
    }

    pub fn receive_generated_chunks(&mut self, texture_atlas: &Arc<TextureAtlas>) {
        let mut new_chunks = Vec::new();

        while let Ok(received) = self.chunkgen_thread_receiver.try_recv() {
            new_chunks.push(received);
        }

        // sort new chunks from closest to center to farthest, also place new chunks adjacent to eachother
        new_chunks.sort_by(|(coords_a, _), (coords_b, _)| {
            let (ax, az) = unpack_coordinates(*coords_a);
            let (bx, bz) = unpack_coordinates(*coords_b);
            let distance_a =
                ((ax - self.center_chunk.0).pow(2) + (az - self.center_chunk.1).pow(2)) as f32;
            let distance_b =
                ((bx - self.center_chunk.0).pow(2) + (bz - self.center_chunk.1).pow(2)) as f32;
            distance_a.partial_cmp(&distance_b).unwrap()
        });

        for (coords, state) in new_chunks {
            let (x, z) = unpack_coordinates(coords);

            self.view.chunks.insert(coords, state);
            let state = self.view.chunks.get(&coords).unwrap();

            if let ChunkState::Loaded(LoadedChunk::Stored { .. }) = state {
                // once chunk has generated, check its neighbours if they are now able to mesh, with all 4 of their neighbours

                // loop thru newly gened's neighbours
                for (nx, nz) in [(0, 1), (0, -1), (1, 0), (-1, 0)] {
                    // diagonal neighbours aren't needed
                    let (nx, nz) = (x + nx, z + nz);
                    let neighbour = self.view.get_chunk(nx, nz);

                    if let Some(ChunkState::Loaded(LoadedChunk::Stored {
                        chunk: neighbour_chunk,
                    })) = neighbour
                    {
                        // we found a stored neighbour, now check if this chunk now has all 4 neighbours

                        let surrounding_neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                            .iter()
                            .filter_map(|(nnx, nnz)| {
                                let (nnx, nnz) = (nx + nnx, nz + nnz);
                                let nn_state = self.view.get_chunk(nnx, nnz);

                                match nn_state {
                                    Some(ChunkState::Loaded(LoadedChunk::Stored { chunk })) => {
                                        Some(chunk)
                                    }
                                    Some(ChunkState::Loaded(LoadedChunk::Meshed {
                                        chunk, ..
                                    })) => Some(chunk),
                                    _ => None,
                                }
                            })
                            .collect::<Vec<_>>();

                        let neighbour_is_surrounded = surrounding_neighbors.len() == 4;

                        // this neighbour can now be meshed
                        if neighbour_is_surrounded {
                            let nn_down = surrounding_neighbors.get(0).unwrap();
                            let nn_up = surrounding_neighbors.get(1).unwrap();
                            let nn_right = surrounding_neighbors.get(2).unwrap();
                            let nn_left = surrounding_neighbors.get(3).unwrap();

                            self.dispatch_chunk_mesh(
                                nx,
                                nz,
                                neighbour_chunk,
                                nn_left,
                                nn_right,
                                nn_up,
                                nn_down,
                                texture_atlas,
                            )
                        }
                    }
                }
            }
        }
    }

    fn dispatch_chunk_mesh(
        &self,
        chunk_x: i32,
        chunk_z: i32,
        chunk: &Arc<Chunk>,
        neighbour_west: &Arc<Chunk>,
        neighbour_east: &Arc<Chunk>,
        neighbour_north: &Arc<Chunk>,
        neighbour_south: &Arc<Chunk>,
        texture_atlas: &Arc<TextureAtlas>,
    ) {
        let sender = self.chunkmesh_out_sender.clone();
        let chunk = chunk.clone();
        let neighbour_west = neighbour_west.clone();
        let neighbour_east = neighbour_east.clone();
        let neighbour_north = neighbour_north.clone();
        let neighbour_south = neighbour_south.clone();
        let texture_atlas = texture_atlas.clone();

        self.thread_pool.execute(move || {
            let mesh_data = chunk.create_mesh(
                (chunk_x, chunk_z),
                &neighbour_west,
                &neighbour_east,
                &neighbour_north,
                &neighbour_south,
                &texture_atlas,
                false,
            );
            let packed = pack_coordinates(chunk_x, chunk_z);

            // create collider
            let vertex_positions = mesh_data
                .vertices
                .iter()
                .map(|v| Point::new(v.position[0] as f64, v.position[1] as f64, v.position[2] as f64))
                .collect::<Vec<_>>();
            let grouped_indices = mesh_data
                .indices
                .chunks(3)
                .map(|c| [c[0], c[1], c[2]])
                .collect::<Vec<_>>();
            let collider = Collider {
                shape: SharedShape::trimesh(vertex_positions, grouped_indices),
            };

            sender.send((packed, mesh_data, collider)).unwrap();
        });
    }

    pub fn upload_chunk_meshes(&mut self, device: &wgpu::Device) {
        while let Ok((coords, mesh_data, collider)) = self.chunkmesh_out_receiver.try_recv() {
            match self.view.chunks.get_mut(&coords) {
                Some(chunk_state) => {
                    match chunk_state {
                        ChunkState::Loaded(LoadedChunk::Stored { chunk })
                        | ChunkState::Loaded(LoadedChunk::Meshed { chunk, .. }) => {
                            // actually upload the mesh to the buffer on the main thread
                            let mesh = Mesh::from_data(device, "Chunk Mesh", &mesh_data, 0);

                            *chunk_state = ChunkState::Loaded(LoadedChunk::Meshed {
                                chunk: chunk.clone(),
                                mesh,
                                collider,
                            });
                        }
                        _ => eprintln!("Chunk mesh upload failed, chunk not ready"),
                    }
                }
                None => eprintln!("Chunk mesh upload failed, chunk not found in map"),
            }
        }
    }

    pub fn remesh_chunk(&mut self, chunk_x: i32, chunk_z: i32, texture_atlas: &Arc<TextureAtlas>) {
        let chunk = match self.view.get_chunk(chunk_x, chunk_z) {
            Some(ChunkState::Loaded(LoadedChunk::Stored { chunk })) => chunk,
            Some(ChunkState::Loaded(LoadedChunk::Meshed { chunk, .. })) => chunk,
            _ => return,
        };

        let neighbours = self
            .view
            .get_chunk_neighbours_exists(chunk_x, chunk_z)
            .expect("Chunk neighbours not loaded yet");

        let [neighbour_west, neighbour_east, neighbour_north, neighbour_south] = neighbours;

        self.dispatch_chunk_mesh(
            chunk_x,
            chunk_z,
            chunk,
            neighbour_west,
            neighbour_east,
            neighbour_north,
            neighbour_south,
            texture_atlas,
        );
    }

    pub fn remesh_chunk_and_neighbours(
        &mut self,
        chunk_x: i32,
        chunk_z: i32,
        texture_atlas: &Arc<TextureAtlas>,
    ) {
        self.remesh_chunk(chunk_x, chunk_z, texture_atlas);

        let neighbours = self
            .view
            .get_chunk_neighbours_exists(chunk_x, chunk_z)
            .expect("Chunk neighbours not loaded yet");

        for (i, neighbour) in neighbours.iter().enumerate() {
            let (nx, nz) = match i {
                0 => (chunk_x - 1, chunk_z),
                1 => (chunk_x + 1, chunk_z),
                2 => (chunk_x, chunk_z - 1),
                3 => (chunk_x, chunk_z + 1),
                _ => unreachable!(),
            };

            let surrounding_neighbours = self.view.get_chunk_neighbours_exists(nx, nz).unwrap();
            let [nn_west, nn_east, nn_north, nn_south] = surrounding_neighbours;

            self.dispatch_chunk_mesh(
                nx,
                nz,
                neighbour,
                nn_west,
                nn_east,
                nn_north,
                nn_south,
                texture_atlas,
            )
        }
    }
}

pub fn pack_coordinates(x: i32, z: i32) -> u64 {
    (x as u32 as u64) << 32 | (z as u32 as u64)
}

pub fn unpack_coordinates(packed: u64) -> (i32, i32) {
    let x = (packed >> 32) as i32;
    let z = (packed & 0xFFFFFFFF) as i32;
    (x, z)
}

pub fn pos_to_chunk_coords(x: i32, z: i32) -> (i32, i32) {
    (
        floor_div(x, CHUNK_WIDTH as i32),
        floor_div(z, CHUNK_WIDTH as i32),
    )
}

pub fn pos_to_voxel_coords(x: i32, z: i32) -> (usize, usize) {
    let cw = CHUNK_WIDTH as i32;
    (
        (((x % cw) + cw) % cw) as usize,
        (((z % cw) + cw) % cw) as usize,
    )
}

pub struct ChunkView {
    chunks: HashMap<u64, ChunkState>,
    tickets: VecDeque<ChunkTicket>,

    pub updated_center_chunk: Option<(i32, i32)>,
}

impl ChunkView {
    pub fn mod_voxel<F>(&mut self, x: i32, y: i32, z: i32, f: F) -> Result<(), ()>
    where
        F: Fn(&mut Voxel),
    {
        let (chunk_x, chunk_z) = pos_to_chunk_coords(x, z);

        let chunk = match self.get_chunk_mut(chunk_x, chunk_z) {
            Some(ChunkState::Loaded(LoadedChunk::Stored { chunk })) => chunk,
            Some(ChunkState::Loaded(LoadedChunk::Meshed { chunk, .. })) => chunk,
            _ => return Err(()),
        };

        let (voxel_x, voxel_z) = pos_to_voxel_coords(x, z);

        if let Some(chunk) = Arc::get_mut(chunk) {
            chunk.mod_voxel(voxel_x, y as usize, voxel_z, f);
        }
        Ok(())
    }

    pub fn get_voxel(&self, x: i32, y: i32, z: i32) -> Result<&Voxel, ()> {
        let (chunk_x, chunk_z) = pos_to_chunk_coords(x, z);

        let chunk = match self.get_chunk(chunk_x, chunk_z) {
            Some(ChunkState::Loaded(LoadedChunk::Stored { chunk })) => chunk,
            Some(ChunkState::Loaded(LoadedChunk::Meshed { chunk, .. })) => chunk,
            _ => return Err(()),
        };

        let (voxel_x, voxel_z) = pos_to_voxel_coords(x, z);

        Ok(chunk.get_voxel(voxel_x, y as usize, voxel_z))
    }

    pub fn get_voxel_mut(&mut self, x: i32, y: i32, z: i32) -> Result<&mut Voxel, ()> {
        let (chunk_x, chunk_z) = pos_to_chunk_coords(x, z);

        let chunk = match self.get_chunk_mut(chunk_x, chunk_z) {
            Some(ChunkState::Loaded(LoadedChunk::Stored { chunk })) => chunk,
            Some(ChunkState::Loaded(LoadedChunk::Meshed { chunk, .. })) => chunk,
            _ => return Err(()),
        };

        let (voxel_x, voxel_z) = pos_to_voxel_coords(x, z);

        match Arc::get_mut(chunk) {
            Some(chunk) => Ok(chunk.get_voxel_mut(voxel_x, y as usize, voxel_z)),
            None => Err(()),
        }
    }

    pub fn set_voxel(&mut self, x: i32, y: i32, z: i32, voxel: Voxel) -> Result<(), ()> {
        let (chunk_x, chunk_z) = pos_to_chunk_coords(x, z);

        let chunk = match self.get_chunk_mut(chunk_x, chunk_z) {
            Some(ChunkState::Loaded(LoadedChunk::Stored { chunk })) => chunk,
            Some(ChunkState::Loaded(LoadedChunk::Meshed { chunk, .. })) => chunk,
            _ => return Err(()),
        };

        let (voxel_x, voxel_z) = pos_to_voxel_coords(x, z);

        if let Some(chunk) = Arc::get_mut(chunk) {
            chunk.set_voxel(voxel_x, y as usize, voxel_z, voxel);
            self.tickets.push_back(ChunkTicket::VoxelUpdate {
                chunk_x,
                chunk_z,
                voxel_x,
                voxel_y: y as usize,
                voxel_z,
            })
        }
        Ok(())
    }

    pub fn get_chunk(&self, x: i32, z: i32) -> Option<&ChunkState> {
        self.chunks.get(&pack_coordinates(x, z))
    }

    pub fn get_chunk_mut(&mut self, x: i32, z: i32) -> Option<&mut ChunkState> {
        self.chunks.get_mut(&pack_coordinates(x, z))
    }

    pub fn get_chunk_neighbours(&self, chunk_x: i32, chunk_z: i32) -> [Option<&ChunkState>; 4] {
        [
            self.get_chunk(chunk_x - 1, chunk_z), // west
            self.get_chunk(chunk_x + 1, chunk_z), // east
            self.get_chunk(chunk_x, chunk_z - 1), // north
            self.get_chunk(chunk_x, chunk_z + 1), // south
        ]
    }

    pub fn get_chunk_neighbours_exists(
        &self,
        chunk_x: i32,
        chunk_z: i32,
    ) -> Option<[&Arc<Chunk>; 4]> {
        let binding = self
            .get_chunk_neighbours(chunk_x, chunk_z)
            .iter()
            .filter_map(|n| match n {
                Some(ChunkState::Loaded(LoadedChunk::Stored { chunk })) => Some(chunk),
                Some(ChunkState::Loaded(LoadedChunk::Meshed { chunk, .. })) => Some(chunk),
                _ => None,
            })
            .collect::<Vec<_>>();
        let neighbours = binding.as_slice();
        if neighbours.len() != 4 {
            None
        } else {
            Some(neighbours.try_into().unwrap())
        }
    }

    pub fn get_all_chunks(&self) -> Vec<&ChunkState> {
        self.chunks.values().collect()
    }
}

pub enum ChunkState {
    LoadScheduled,
    Loading,
    Loaded(LoadedChunk),
    Failed,
}

pub enum LoadedChunk {
    Meshed {
        chunk: Arc<Chunk>,
        mesh: Mesh,
        collider: Collider,
    },
    Stored {
        chunk: Arc<Chunk>,
    },
}

pub enum ChunkTicket {
    ChunkUpdate {
        chunk_x: i32,
        chunk_z: i32,
    },
    VoxelUpdate {
        chunk_x: i32,
        chunk_z: i32,
        voxel_x: usize,
        voxel_y: usize,
        voxel_z: usize,
    },
}
