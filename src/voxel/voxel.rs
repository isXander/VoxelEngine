#[derive(Clone, Copy, Debug)]
pub struct Voxel {
    pub voxel_type: VoxelType,
}

impl Voxel {
    pub fn create_default_type(voxel_type: VoxelType) -> Self {
        Self {
            voxel_type,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug, enum_map::Enum)]
pub enum VoxelType {
    Air,
    Grass,
}

#[derive(Debug)]
pub enum Face {
    Top,
    Bottom,
    North,
    South,
    East,
    West,
}

impl Face {
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
}