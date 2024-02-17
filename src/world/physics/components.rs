use rapier3d::prelude as rapier;
use crate::world::player::Position;

#[derive(Clone)]
pub struct Collider {
    pub shape: rapier::SharedShape
}

impl Collider {
    pub fn new(shape: rapier::SharedShape) -> Self {
        Self {
            shape
        }
    }
}

pub type ColliderComponents<'a> = (
    &'a Collider,

    Option<&'a RapierRigidBodyHandle>
);

pub struct RapierColliderHandle(pub rapier::ColliderHandle);

pub enum RigidBody {
    Dynamic,
    Static,
}

pub type RigidbodyComponents<'a> = (
    &'a RigidBody,
    Option<&'a Position>
);

pub struct RapierRigidBodyHandle(pub rapier::RigidBodyHandle);