use std::slice::Iter;
use nalgebra::Vector3;

#[derive(Clone, Copy, Debug)]
pub struct Voxel {
    pub voxel_type: Type,
}

impl Voxel {
    #[must_use]
    pub fn create_default_type(voxel_type: Type) -> Self {
        Self {
            voxel_type,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug, enum_map::Enum)]
pub enum Type {
    Air,
    Dirt,
    Sand,
    Snow,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, enum_map::Enum)]
pub enum Face {
    Top,
    Bottom,
    North,
    South,
    East,
    West,
}

impl Face {
    #[must_use]
    pub fn to_ordinal(&self) -> usize {
        match self {
            Face::Top => 0,
            Face::Bottom => 1,
            Face::North => 2,
            Face::South => 3,
            Face::East => 4,
            Face::West => 5,
        }
    }

    #[must_use]
    pub fn from_ordinal(ordinal: usize) -> Self {
        match ordinal {
            0 => Face::Top,
            1 => Face::Bottom,
            2 => Face::North,
            3 => Face::South,
            4 => Face::East,
            5 => Face::West,
            _ => unreachable!("Invalid face index"),
        }
    }

    pub fn offset_vector(&self) -> Vector3<i32> {
        match self {
            Face::Top => Vector3::new(0, 1, 0),
            Face::Bottom => Vector3::new(0, -1, 0),
            Face::North => Vector3::new(0, 0, -1),
            Face::South => Vector3::new(0, 0, 1),
            Face::East => Vector3::new(1, 0, 0),
            Face::West => Vector3::new(-1, 0, 0),
        }
    }

    pub fn iterator() -> Iter<'static, Face> {
        static FACES: [Face; 6] = [Face::Top, Face::Bottom, Face::North, Face::South, Face::East, Face::West];
        FACES.iter()
    }
}