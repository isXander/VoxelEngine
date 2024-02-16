use hecs::World;
use crate::voxel::chunk::ChunkManager;
use crate::world::player::*;

pub(crate) fn system_chunks_focus_player(world: &mut World, ctx: &mut Context, stage: &mut UpdateStage) {
    for (id, (Position(position), _)) in &mut world.query::<(&Position, &PlayerMarker)>() {
        let (chunk_x, chunk_z) = ChunkManager::pos_to_chunk_coords(position.x.floor() as i32, position.z.floor() as i32);
        stage.chunk_manager.set_center_chunk(chunk_x, chunk_z);
        stage.chunk_manager.fill_map_renderdistance();
    }
}

pub(crate) fn system_update_chunk_physics(world: &mut World, ctx: &mut Context) {
    for (id, (Position(pos), _)) in &mut world.query::<(&Position, &PlayerMarker)>() {
        let (chunk_x, chunk_z) = ChunkManager::pos_to_chunk_coords(pos.x.floor() as i32, pos.z.floor() as i32);


    }
}