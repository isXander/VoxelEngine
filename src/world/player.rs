use crate::engine::camera::{Camera, Deg, PlayerController};
use crate::voxel::chunk::{pos_to_chunk_coords, ChunkView};
use crate::voxel::math::raycast;
use crate::voxel::voxel;
use crate::world::physics::components::{Collider, RapierRigidBodyHandle, RigidBody};
use crate::world::physics::context::PhysicsContext;
use hecs_schedule::{CommandBuffer, Read, SubWorld, Write};
use nalgebra::{vector, Point3, Vector3};
use rapier3d::prelude::SharedShape;
use winit::event::WindowEvent;
use winit::keyboard::KeyCode;
use winit::keyboard::PhysicalKey::Code;

pub struct Position(pub Point3<f32>);
pub struct LookDirection {
    pub yaw: Deg<f32>,
    pub pitch: Deg<f32>,
}
pub struct Velocity(pub Vector3<f32>);

pub struct PlayerMarker;
pub struct BoundCameraMarker;

pub(crate) fn system_player_spawn(mut cmd: Write<CommandBuffer>, position: Position) {
    let yaw = Deg(90.0);
    let pitch = Deg(0.0);

    cmd.spawn((
        Camera::new(position.0, yaw, pitch),
        position,
        LookDirection { yaw, pitch },
        Velocity(Vector3::identity()),
        PlayerMarker,
        PlayerController::new(2.0, 8.0),
        BoundCameraMarker,
        Collider::new(SharedShape::ball(1.0)),
        RigidBody::Dynamic,
    ));
}

pub(crate) fn system_player_update_camera(
    world: SubWorld<(&Position, &LookDirection, &mut Camera)>,
) {
    for (_, (Position(pos), direction, camera)) in
        &mut world.query::<(&Position, &LookDirection, &mut Camera)>()
    {
        camera.position = pos.clone() + vector![0.0, 1.8, 0.0]; // eye height
        camera.yaw = direction.yaw.into();
        camera.pitch = direction.pitch.into();

        let (chunk_x, chunk_z) = pos_to_chunk_coords(
            camera.position.x.floor() as i32,
            camera.position.z.floor() as i32,
        );
        //println!("Ppos: {},{}", chunk_x, chunk_z);
    }
}

pub(crate) fn system_player_update_position(
    world: SubWorld<(&RapierRigidBodyHandle, &mut Position)>,
    physics: Read<PhysicsContext>,
    delta_time: Read<f32>,
) {
    for (_, (RapierRigidBodyHandle(handle), Position(pos))) in
        &mut world.query::<(&RapierRigidBodyHandle, &mut Position)>()
    {
        let body = &physics.bodies[*handle];
        let pos_now: Vector3<f32> = body.position().translation.vector.into();
        let pos_next = body.next_position().translation.vector.into();

        *pos = pos_now.lerp(&pos_next, *delta_time).into();
    }
}

pub(crate) fn system_player_update_controller(
    world: SubWorld<(
        &PlayerController,
        &mut Position,
        &mut RapierRigidBodyHandle,
        &mut LookDirection,
    )>,
    mut physics: Write<PhysicsContext>,
    delta_time: Read<f32>,
) {
    for (_, (controller, body_handle, look_direction)) in &mut world.query::<(
        &PlayerController,
        &mut RapierRigidBodyHandle,
        &mut LookDirection,
    )>() {
        let mut body = &mut physics.bodies[body_handle.0];
        controller.update(&mut body, look_direction, *delta_time);
    }
}

pub(crate) fn system_player_input_controller(
    world: SubWorld<(&mut PlayerController)>,
    event: Read<WindowEvent>,
) {
    for (_, controller) in &mut world.query::<(&mut PlayerController)>() {
        controller.process_input(&event);
    }
}

pub(crate) fn system_player_input_break(
    world: SubWorld<(&PlayerMarker, &Camera)>,
    mut chunk_view: Write<ChunkView>,
    event: Read<WindowEvent>,
) {
    let event: &WindowEvent = &event;
    match event {
        WindowEvent::KeyboardInput { event, .. }
            if event.state == winit::event::ElementState::Pressed =>
        {
            match event.physical_key {
                Code(KeyCode::Space) => {
                    for (_, (camera, _)) in &mut world.query::<(&Camera, &PlayerMarker)>() {
                        let raycast_result =
                            raycast(&camera.position, &camera.direction().into(), 100.0, |pos| {
                                chunk_view
                                    .get_voxel(pos.x, pos.y, pos.z)
                                    .unwrap()
                                    .voxel_type
                                    != voxel::Type::Air
                            });

                        if let Some(raycast_result) = raycast_result {
                            println!("Raycast hit: {:?}", raycast_result);

                            chunk_view
                                .set_voxel(
                                    raycast_result.position.x,
                                    raycast_result.position.y,
                                    raycast_result.position.z,
                                    voxel::Voxel::create_default_type(voxel::Type::Air),
                                )
                                .expect("could not set voxel");
                        }
                    }
                }
                _ => {}
            }
        }
        _ => {}
    }
}
