use crate::voxel::chunk::{
    pos_to_chunk_coords, pos_to_voxel_coords, ChunkManager, ChunkState, ChunkView, LoadedChunk,
};
use crate::world::physics::components::{Collider, RigidBody};
use crate::world::player::*;
use hecs::World;
use hecs_schedule::{CommandBuffer, Read, SubWorld, Write};
use nalgebra::Point3;
use std::collections::{HashSet, VecDeque};

pub(crate) fn system_chunks_focus_player(
    world: SubWorld<(&Position, &PlayerMarker)>,
    mut chunk_view: Write<ChunkView>,
) {
    for (id, (Position(position), _)) in &mut world.query::<(&Position, &PlayerMarker)>() {
        let (chunk_x, chunk_z) =
            pos_to_chunk_coords(position.x.floor() as i32, position.z.floor() as i32);
        chunk_view.updated_center_chunk = Some((chunk_x, chunk_z))
    }
}

pub struct ChunkPos {
    x: i32,
    z: i32,
}

type ChunkComponents<'a> = (&'a ChunkPos, &'a RigidBody, &'a Collider);

pub fn system_update_chunk_physics(
    w: SubWorld<(&Position, &PlayerMarker, ChunkComponents)>,
    mut cmd: Write<CommandBuffer>,
    chunk_view: Read<ChunkView>,
) {
    let mut active_chunks = HashSet::new();

    for (id, (Position(pos), _)) in &mut w.query::<(&Position, &PlayerMarker)>() {
        let (chunk_x, chunk_z) = pos_to_chunk_coords(pos.x.floor() as i32, pos.z.floor() as i32);
        let (voxel_x, voxel_z) = pos_to_voxel_coords(pos.x.floor() as i32, pos.z.floor() as i32);

        active_chunks.insert((chunk_x, chunk_z));

        if voxel_x < 4 {
            active_chunks.insert((chunk_x - 1, chunk_z));
        } else if voxel_x > 11 {
            active_chunks.insert((chunk_x + 1, chunk_z));
        }

        if voxel_z < 4 {
            active_chunks.insert((chunk_x, chunk_z - 1));
        } else if voxel_z > 11 {
            active_chunks.insert((chunk_x, chunk_z + 1));
        }

        //debug
        for x in -5..=5 {
            for z in -5..=5 {
                active_chunks.insert((chunk_x + x, chunk_z + z));
            }
        }
    }

    // remove all chunks that are not in this this
    for (entity, ChunkPos { x, z }) in &mut w.query::<&ChunkPos>() {
        if !active_chunks.contains(&(*x, *z)) {
            println!("Despawning");
            cmd.despawn(entity);
        } else {
            active_chunks.remove(&(*x, *z));
        }
    }

    for (chunk_x, chunk_z) in active_chunks {
        if let Some(ChunkState::Loaded(LoadedChunk::Meshed { collider, .. })) =
            chunk_view.get_chunk(chunk_x, chunk_z)
        {
            let collider = collider.clone();

            println!("Added chunk: {:?}", (chunk_x, chunk_z));
            cmd.spawn((
                ChunkPos {
                    x: chunk_x,
                    z: chunk_z,
                },
                RigidBody::Static,
                collider,
                Position(Point3::new(
                    chunk_x as f32 * 16.0,
                    0.0,
                    chunk_z as f32 * 16.0,
                )),
            ));
        }
    }
}
