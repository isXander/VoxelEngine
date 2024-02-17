use std::collections::VecDeque;
use hecs::World;
use hecs_schedule::{CommandBuffer, Read, SubWorld, Write};
use nalgebra::Point3;
use crate::voxel::chunk::{ChunkManager, ChunkState, ChunkView, LoadedChunk, pos_to_chunk_coords, pos_to_voxel_coords};
use crate::world::physics::components::{Collider, RigidBody};
use crate::world::player::*;

pub(crate) fn system_chunks_focus_player(world: SubWorld<(&Position, &PlayerMarker)>, mut chunk_view: Write<ChunkView>) {
    for (id, (Position(position), _)) in &mut world.query::<(&Position, &PlayerMarker)>() {
        let (chunk_x, chunk_z) = pos_to_chunk_coords(position.x.floor() as i32, position.z.floor() as i32);
        chunk_view.updated_center_chunk = Some((chunk_x, chunk_z))
    }
}

pub struct ChunkPos {
    x: i32,
    z: i32,
}

type ChunkComponents<'a> = (&'a ChunkPos, &'a RigidBody, &'a Collider);

pub fn system_update_chunk_physics(w: SubWorld<(&Position, &PlayerMarker, ChunkComponents)>, mut cmd: Write<CommandBuffer>, chunk_view: Read<ChunkView>) {
    for (id, (Position(pos), _)) in &mut w.query::<(&Position, &PlayerMarker)>() {
        let (chunk_x, chunk_z) = pos_to_chunk_coords(pos.x.floor() as i32, pos.z.floor() as i32);
        let (voxel_x, voxel_z) = pos_to_voxel_coords(pos.x.floor() as i32, pos.z.floor() as i32);

        let mut to_add = VecDeque::new();
        let mut to_remove = VecDeque::new();

        if voxel_x < 3 {
            to_add.push_back((chunk_x - 1, chunk_z));
            to_remove.push_back((chunk_x + 1, chunk_z));
        } else if voxel_x > 12 {
            to_add.push_back((chunk_x + 1, chunk_z));
            to_remove.push_back((chunk_x - 1, chunk_z));
        } else {
            to_remove.push_back((chunk_x + 1, chunk_z));
            to_remove.push_back((chunk_x - 1, chunk_z));
        }

        if voxel_z < 3 {
            to_add.push_back((chunk_x, chunk_z - 1));
            to_remove.push_back((chunk_x, chunk_z + 1));
        } else if voxel_z > 12 {
            to_add.push_back((chunk_x, chunk_z + 1));
            to_remove.push_back((chunk_x, chunk_z - 1));
        } else {
            to_remove.push_back((chunk_x, chunk_z + 1));
            to_remove.push_back((chunk_x, chunk_z - 1));
        }

        for (chunk_x, chunk_z) in to_add {
            if let Some(ChunkState::Loaded(LoadedChunk::Meshed { collider, .. })) = chunk_view.get_chunk(chunk_x, chunk_z) {
                if !w.query::<&ChunkPos>().iter().any(|(_, ChunkPos { x, z })| *x == chunk_x && *z == chunk_z) {
                    let collider = collider.clone();

                    println!("Added chunk: {:?}", (chunk_x, chunk_z));
                    cmd.spawn((
                        ChunkPos { x: chunk_x, z: chunk_z },
                        RigidBody::Static,
                        collider,
                        Position(Point3::new(chunk_x as f32 * 16.0, 0.0, chunk_z as f32 * 16.0)),
                    ));
                }
            }
        }

        for (chunk_x, chunk_z) in to_remove {
            for (entity, ChunkPos { x, z }) in &mut w.query::<&ChunkPos>() {
                if *x == chunk_x && *z == chunk_z {
                    println!("Removed chunk: {:?}", (chunk_x, chunk_z));
                    cmd.despawn(entity);
                }
            }
        }
    };
}