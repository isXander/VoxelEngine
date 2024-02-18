use rapier3d_f64::prelude as rapier;
use rapier3d_f64::prelude::LockedAxes;
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

pub struct ColliderMass(pub f64);

pub type ColliderComponents<'a> = (
    &'a Collider,

    Option<&'a ColliderMass>,
    Option<&'a RapierRigidBodyHandle>
);

pub struct RapierColliderHandle(pub rapier::ColliderHandle);

pub enum RigidBody {
    Dynamic,
    Static,
}

pub struct LockPosition {
    pub x: bool,
    pub y: bool,
    pub z: bool,
}

impl LockPosition {
    pub fn new(all: bool) -> Self {
        Self {
            x: all,
            y: all,
            z: all,
        }
    }

    pub fn bitflags(&self) -> u8 {
        let mut flag = LockedAxes::empty().bits();
        if self.x {
            flag |= LockedAxes::TRANSLATION_LOCKED_X.bits()
        }
        if self.y {
            flag |= LockedAxes::TRANSLATION_LOCKED_Y.bits()
        }
        if self.z {
            flag |= LockedAxes::TRANSLATION_LOCKED_Z.bits()
        }

        flag
    }
}

pub struct LockRotation {
    pub x: bool,
    pub y: bool,
    pub z: bool,
}

impl LockRotation {
    pub fn new(all: bool) -> Self {
        Self {
            x: all,
            y: all,
            z: all,
        }
    }

    pub fn bitflags(&self) -> u8 {
        let mut flag = LockedAxes::empty().bits();
        if self.x {
            flag |= LockedAxes::ROTATION_LOCKED_X.bits()
        }
        if self.y {
            flag |= LockedAxes::ROTATION_LOCKED_Y.bits()
        }
        if self.z {
            flag |= LockedAxes::ROTATION_LOCKED_Z.bits()
        }

        flag
    }
}

pub struct BodyDamping {
    pub linear: f64,
    pub angular: f64,
}

pub struct CCD;

pub type RigidbodyComponents<'a> = (
    &'a RigidBody,
    Option<&'a Position>,
    Option<&'a LockPosition>,
    Option<&'a LockRotation>,
    Option<&'a BodyDamping>,
    Option<&'a CCD>
);

pub struct RapierRigidBodyHandle(pub rapier::RigidBodyHandle);