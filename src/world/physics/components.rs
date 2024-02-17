use rapier3d::prelude::SharedShape;

pub struct Collider {
    pub shape: SharedShape
}

impl Collider {
    pub fn new(shape: SharedShape) -> Self {
        Self {
            shape
        }
    }
}

pub type ColliderComponents<'a> = (
    &'a Collider
);