use hecs_schedule::*;
use nalgebra::Point3;
use crate::world::app::App;

pub mod chunk;
pub mod player;
pub mod scheduler;
pub mod physics;
pub mod app;

pub fn build_app(context: app::Context) -> App {
    let mut app = App::builder();

    app.start_schedule
        .add_system(|cmd: Write<CommandBuffer>| player::system_player_spawn(cmd, player::Position(Point3::new(0.0, 60.0, 0.0))))
    ;

    app.update_schedule
        .add_system(player::system_player_update_controller)
        .add_system(player::system_player_update_camera)
        .add_system(chunk::system_chunks_focus_player)
    ;

    app.input_schedule
        .add_system(player::system_player_input_controller)
        .add_system(player::system_player_input_break)
    ;

    app.build(context)
}