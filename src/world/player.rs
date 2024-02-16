use hecs::World;
use nalgebra::{Point3, Vector3};
use crate::engine::camera::{Camera, Deg, PlayerController};
use crate::world::scheduler::{InputStage, UpdateStage};

pub struct Position(pub Point3<f32>);
pub struct LookDirection {
    pub yaw: Deg<f32>,
    pub pitch: Deg<f32>,
}
pub struct Velocity(pub Vector3<f32>);

pub struct PlayerMarker;

pub(crate) fn system_player_spawn(world: &mut World, position: Position) {
    world.spawn((
        Camera::new(position.0, Deg(90.0), Deg(0.0)),
        position,
        Velocity(Vector3::identity()),
        PlayerMarker,
        PlayerController::new(200.0, 2.0),
    ));
}

pub(crate) fn system_player_update_camera(world: &mut World, stage: &UpdateStage) {
    for (_, (Position(pos), direction, camera)) in &mut world.query::<(&Position, &LookDirection, &mut Camera)>() {
        camera.position = pos.clone();
        camera.yaw = direction.yaw.into();
        camera.pitch = direction.pitch.into();
    }
}

pub(crate) fn system_player_update_controller(world: &mut World, stage: &InputStage) {
    for (_, (controller)) in &mut world.query::<(&mut PlayerController)>() {
        controller.process_input(stage.event);
    }
}

