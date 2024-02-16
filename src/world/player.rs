use hecs::World;
use nalgebra::{Point3, Vector3};
use crate::engine::camera::{Camera, Deg, PlayerController};
use crate::world::scheduler::{InputStage, UpdateStage};

use super::scheduler::Context;

pub struct Position(pub Point3<f32>);
pub struct LookDirection {
    pub yaw: Deg<f32>,
    pub pitch: Deg<f32>,
}
pub struct Velocity(pub Vector3<f32>);

pub struct PlayerMarker;
pub struct BoundCameraMarker;

pub(crate) fn system_player_spawn(world: &mut World, ctx: &mut Context, position: Position) {
    let yaw = Deg(90.0);
    let pitch = Deg(0.0);

    world.spawn((
        Camera::new(position.0, yaw, pitch),
        position,
        LookDirection { yaw, pitch },
        Velocity(Vector3::identity()),
        PlayerMarker,
        PlayerController::new(200.0, 8.0),
        BoundCameraMarker,
    ));
}

pub(crate) fn system_player_update_camera(world: &mut World, ctx: &mut Context, stage: &mut UpdateStage) {
    for (_, (Position(pos), direction, camera)) in &mut world.query::<(&Position, &LookDirection, &mut Camera)>() {
        camera.position = pos.clone();
        camera.yaw = direction.yaw.into();
        camera.pitch = direction.pitch.into();
    }
}

pub(crate) fn system_player_update_controller(world: &mut World, ctx: &mut Context, stage: &mut UpdateStage) {
    for (_, (controller, position, look_direction)) in &mut world.query::<(&PlayerController, &mut Position, &mut LookDirection)>() {
        controller.update(position, look_direction, stage.delta_time);
    }
}

pub(crate) fn system_player_input_controller(world: &mut World, ctx: &mut Context, stage: &InputStage) {
    for (_, (controller)) in &mut world.query::<(&mut PlayerController)>() {
        controller.process_input(stage.event);
        
    }
}

