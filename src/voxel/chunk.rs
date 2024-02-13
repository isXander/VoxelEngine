use std::usize;

use wgpu::core::device;

use crate::engine;

pub struct Chunk {
    voxels: [[[Voxel; 16]; 128]; 16],
}

impl Chunk {
    pub fn new() -> Self {
        Self {
            voxels: [[[Voxel { voxel_type: VoxelType::Air }; 16]; 128]; 16],
        }
    }

    pub fn new_plane(y: usize) -> Self {
        let mut chunk = Self::new();

        for x in 0..16 {
            for z in 0..16 {
                chunk.set_voxel(x, y, z, Voxel { voxel_type: VoxelType::Grass });
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

    pub fn get_voxel(&self, x: usize, y: usize, z: usize) -> Voxel {
        self.voxels[x][y][z]
    }

    pub fn create_mesh(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        chunk_pos: (i32, i32),
    ) -> engine::model::Mesh {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

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

                        vertices.extend(create_face(x as f32, y as f32, z as f32, Face::from_ordinal(i)));
                        //indices.extend(face_indices.iter().map(|index| index + vertices.len() as u32));
                    }
                }
            }
        };

        // TODO: implement greedy meshing
        // TODO: implement indicies
        for i in 0..vertices.len() {
            indices.push(i as u32);
        }

        engine::model::Mesh::new(device, queue, format!("Chunk Mesh {},{}", chunk_pos.0, chunk_pos.1).as_str(), &vertices, &indices, 0)
    }
}

fn create_face(
    x: f32, y: f32, z: f32,
    face: Face,
) -> Vec<engine::model::ModelVertex> {
    let mut vertices = Vec::new();

    println!("Creating face: {:?}", face);

    let (x_axis, y_axis, z_axis, x_off, y_off, z_off, norm_x, norm_y, norm_z) = match face {
        Face::Top => (1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0),
        Face::Bottom => (1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0),
        Face::North => (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0),
        Face::South => (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        Face::East => (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        Face::West => (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0),
    };

    let top_left = (x + x_off, y + y_off, z + z_off, 0.0, 0.0);
    let top_right = (x + x_off + x_axis, y + y_off, z + z_off, 0.0, 1.0);
    let bottom_left = (x + x_off, y + y_off + y_axis, z + z_off + z_axis, 1.0, 0.0);
    let bottom_right = (x + x_off + x_axis, y + y_off + y_axis, z + z_off + z_axis, 1.0, 1.0);
    
    let vertices_data = [
        top_left, bottom_left, bottom_right,
        top_left, bottom_right, top_right,
    ];

    for (x, y, z, u, v) in vertices_data.iter() {
        vertices.push(engine::model::ModelVertex {
            position: [*x, *y, *z],
            normal: [norm_x, norm_y, norm_z],
            tex_coords: [*u, *v],
            bitangent: [0.0, 0.0, 0.0],
            tangent: [0.0, 0.0, 0.0],
        });
    }

    vertices

}

#[derive(Clone, Copy, Debug)]
pub struct Voxel {
    voxel_type: VoxelType,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum VoxelType {
    Air,
    Grass,
}

#[derive(Debug)]
pub enum Face {
    Top,
    Bottom,
    North,
    South,
    East,
    West,
}

impl Face {
    fn to_ordinal(&self) -> usize {
        match self {
            Face::Top => 0,
            Face::Bottom => 1,
            Face::North => 2,
            Face::South => 3,
            Face::East => 4,
            Face::West => 5,
        }
    }

    fn from_ordinal(ordinal: usize) -> Self {
        match ordinal {
            0 => Face::Top,
            1 => Face::Bottom,
            2 => Face::North,
            3 => Face::South,
            4 => Face::East,
            5 => Face::West,
            _ => unreachable!("Invalid face index"),
        }
    }
}