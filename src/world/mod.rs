use nalgebra::Point3;

pub mod physics;
pub mod player;
pub mod scheduler;

pub fn setup() -> scheduler::WorldScheduler {
    let mut scheduler = scheduler::WorldScheduler::new();

    scheduler.add_start_systems(vec![
        |world| physics::system_update_chunk_physics(world),
    ]);

    scheduler.add_update_systems(vec![
        player::system_player_update_camera,
    ]);

    scheduler.add_input_systems(vec![
        player::system_player_update_controller,
    ]);

    scheduler
}