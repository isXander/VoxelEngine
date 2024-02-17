use hecs::World;
use hecs_schedule::{CommandBuffer, Read, SubWorld, Write};
use crate::voxel::chunk::{ChunkManager, ChunkView, pos_to_chunk_coords};
use crate::world::player::*;

pub(crate) fn system_chunks_focus_player(world: SubWorld<(&Position, &PlayerMarker)>, mut chunk_view: Write<ChunkView>) {
    for (id, (Position(position), _)) in &mut world.query::<(&Position, &PlayerMarker)>() {
        let (chunk_x, chunk_z) = pos_to_chunk_coords(position.x.floor() as i32, position.z.floor() as i32);
        chunk_view.updated_center_chunk = Some((chunk_x, chunk_z))
    }
}

pub fn system_update_chunk_physics(w: SubWorld<(&Position, &PlayerMarker)>, chunk_view: Read<ChunkView>) {
    for (id, (Position(pos), _)) in &mut w.query::<(&Position, &PlayerMarker)>() {
        let (chunk_x, chunk_z) = pos_to_chunk_coords(pos.x.floor() as i32, pos.z.floor() as i32);


    };
}