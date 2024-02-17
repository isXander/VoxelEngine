use hecs_schedule::{CommandBuffer, Read, SubWorld, Write};
use rapier3d::prelude::*;
use crate::world::physics::components::ColliderComponents;
use crate::world::physics::context::{PhysicsConfig, PhysicsContext};

pub fn system_physics_step(mut physics: Write<PhysicsContext>, config: Read<PhysicsConfig>) {
    physics.step(&config)
}

pub fn systems_physics_sync_collider(world: SubWorld<ColliderComponents>, mut cmd: Write<CommandBuffer>, mut physics: Write<PhysicsContext>) {
    for (id, (collider)) in &mut world.query::<ColliderComponents>().without::<&ColliderHandle>() {
        // found all unregistered colliders

        let collider = ColliderBuilder::new(collider.shape.clone());

        // TODO: all the diff collider option things here

        let handle = physics.colliders.insert(collider.build());

        cmd.insert_one(id, handle)
    }
}
