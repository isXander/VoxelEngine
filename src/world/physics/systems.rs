use std::collections::VecDeque;
use hecs::{Entity, World};
use hecs_schedule::{CommandBuffer, Read, SubWorld, Write};
use rapier3d::prelude as rapier;
use crate::world::physics::components::{Collider, ColliderComponents, RapierColliderHandle, RapierRigidBodyHandle, RigidBody, RigidbodyComponents};
use crate::world::physics::context::{PhysicsConfig, PhysicsContext};
use crate::world::player::Position;

pub fn system_physics_step(mut physics: Write<PhysicsContext>, config: Read<PhysicsConfig>) {
    physics.step(&config)
}

pub fn systems_physics_create_ordered(world: SubWorld<(ColliderComponents, RigidbodyComponents)>, mut cmd: Write<CommandBuffer>, mut physics: Write<PhysicsContext>) {
    let mut cmd = &mut *cmd;
    let mut physics = &mut *physics;

    for (id, (body, collider)) in &mut world.query::<(Option<RigidbodyComponents>, Option<ColliderComponents>)>().without::<&RapierRigidBodyHandle>() {
        let body_handle = if let Some(body) = body {
            Some(create_body((id, body), physics))
        } else {
            None
        };

        let collider_handle = if let Some(collider) = collider {
            Some(create_collider((id, collider), physics, &body_handle.as_ref()))
        } else {
            None
        };


        if let Some(handle) = body_handle {
            cmd.insert_one(id, handle);
        }
        if let Some(handle) = collider_handle {
            cmd.insert_one(id, handle);
        }
    }
}

fn create_body(entity: (Entity, RigidbodyComponents), physics: &mut PhysicsContext) -> RapierRigidBodyHandle {
    let (entity, (body, position)) = entity;

    // found all unregistered bodies
    let mut body = match body {
        RigidBody::Dynamic => rapier::RigidBodyBuilder::dynamic(),
        RigidBody::Static => rapier::RigidBodyBuilder::fixed(),
    };

    if let Some(position) = position {
        body = body.position(position.0.clone().into());
    }

    body = body.additional_mass(1.0);

    // TODO: all the diff collider option things here

    let handle = physics.bodies.insert(body);
    physics.body2entity.insert(handle, entity);

    RapierRigidBodyHandle(handle)
}

fn create_collider(entity: (Entity, ColliderComponents), physics: &mut PhysicsContext, new_body: &Option<&RapierRigidBodyHandle>) -> RapierColliderHandle {
    let (id, (collider, existing_body)) = entity;
    let body_handle = new_body.or(existing_body);

    // found all unregistered colliders
    let mut collider = rapier::ColliderBuilder::new(collider.shape.clone());

    // TODO: all the diff collider option things here
    collider = collider.mass(1.0);

    let handle = if let Some(body_handle) = body_handle {
        physics.colliders.insert_with_parent(collider, body_handle.0, &mut physics.bodies)
    } else {
        physics.colliders.insert(collider)
    };
    physics.collider2entity.insert(handle, id);

    RapierColliderHandle(handle)
}

pub fn systems_physics_remove_bodies(world: Read<World>, mut cmd: Write<CommandBuffer>, mut physics: Write<PhysicsContext>) {
    let physics = &mut *physics;

    let mut to_remove = VecDeque::new();

    for (entity, body_handle) in &mut world.query::<&RapierRigidBodyHandle>().without::<&RigidBody>() {
        cmd.remove_one::<&RapierRigidBodyHandle>(entity);
        to_remove.push_back(body_handle.0);
    }

    for (handle, _) in physics.bodies.iter() {
        if let Some(entity) = physics.body2entity.get(&handle) {
            if !world.contains(*entity) {
                // entity has been removed
                to_remove.push_back(handle);
            }
        }
    }

    for handle in to_remove {
        physics.remove_body(handle);
    }
}
