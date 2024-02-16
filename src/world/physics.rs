use hecs::World;
use crate::voxel::chunk::ChunkManager;
use crate::world::player::*;

pub(crate) fn system_update_chunk_physics(world: &mut World) {
    for (id, (Position(pos), _)) in &mut world.query::<(&Position, &PlayerMarker)>() {
        let (chunk_x, chunk_z) = ChunkManager::pos_to_chunk_coords(pos.x.floor() as i32, pos.z.floor() as i32);


    }
}