use nalgebra::Point3;

use self::scheduler::WorldScheduler;

pub mod chunk;
pub mod player;
pub mod scheduler;
pub mod physics;

pub fn setup(scheduler: &mut WorldScheduler) {
    scheduler.add_start_systems(vec![
        |world, ctx| player::system_player_spawn(world, ctx, player::Position(Point3::new(0.0, 0.0, 0.0))),
        chunk::system_update_chunk_physics,
    ]);

    scheduler.add_update_systems(vec![
        player::system_player_update_controller,
        player::system_player_update_camera,
        chunk::system_chunks_focus_player,
    ]);

    scheduler.add_input_systems(vec![
        player::system_player_input_controller,
    ]);
}