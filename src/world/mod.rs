use hecs_schedule::*;
use nalgebra::Point3;
use crate::voxel::chunk::ChunkView;
use crate::world::app::App;

pub mod chunk;
pub mod player;
pub mod scheduler;
pub mod physics;
pub mod app;

pub fn build_app(context: app::Context) -> App {
    let mut app = App::builder();

    app.start_schedule
        .add_system(|cmd: Write<CommandBuffer>, chunk_view: Write<ChunkView>| player::system_player_spawn(cmd, chunk_view, player::Position(Point3::new(0.0, 80.0, 0.0))))
    ;

    app.update_schedule
        .add_system(player::system_player_update_controller)
        .add_system(player::system_player_update_camera)
        .add_system(chunk::system_chunks_focus_player)
        .add_system(chunk::system_update_chunk_physics)
        .add_system(player::system_player_update_position)
    ;

    app.fixed_update_schedule
        .add_system(physics::systems::system_physics_step)
        .add_system(physics::systems::systems_physics_create_ordered)
    ;

    app.input_schedule
        .add_system(player::system_player_input_controller)
        .add_system(player::system_player_input_break)
    ;

    app.build(context)
}