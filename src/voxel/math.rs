use nalgebra::{Point3, Vector3};
use crate::voxel::voxel::Face;
use ordered_float::OrderedFloat;
use crate::voxel::chunk::CHUNK_HEIGHT;

pub trait BlockPos {
    fn offset_face(&mut self, face: &Face, n: i32);


}

impl BlockPos for Vector3<i32> {
    fn offset_face(&mut self, face: &Face, n: i32) {
        match face {
            Face::Top => self.y += n,
            Face::Bottom => self.y -= n,
            Face::North => self.z -= n,
            Face::South => self.z += n,
            Face::East => self.x += n,
            Face::West => self.x -= n,
        }
    }
}

// TODO: determine face
#[derive(Debug)]
pub struct CastResult {
    pub position: Point3<i32>,
    pub face: Face,
}

pub fn raycast<Query>(
    origin: &Point3<f32>,
    direction: &Vector3<f32>,
    length: f32,
    query: Query
) -> Option<CastResult>
where
    Query: Fn(Point3<i32>) -> bool ,
{
    let direction = direction.normalize();
    dbg!(direction);

    // iterate along the ray with defined step size
    let step_size = 0.1; // Smaller step size -> higher accuracy, more computation
    for distance in (0..(length / step_size) as usize).map(|x| x as f32 * step_size) {
        let point = origin + direction * distance;
        let block_pos = point.map(|x| x.floor() as i32);

        if block_pos.y < 0 || block_pos.y >= CHUNK_HEIGHT as i32 {
            continue;
        }

        let face = get_nearest_face(&point.coords);

        if query(block_pos) {
            return Some(CastResult {
                position: block_pos,
                face,
            });
        }
    }

    None
}

fn get_nearest_face(
    vec: &Vector3<f32>,
) -> Face {
    *Face::iterator()
        .min_by_key(|face| {
            let offset = face.offset_vector().map(|x| x as f32);
            let dot = vec.dot(&offset);
            let distance = (vec - offset * dot).magnitude();
            OrderedFloat(distance)
        })
        .unwrap()
}

pub fn floor_div(x: i32, y: i32) -> i32 {
    let div = x / y;
    let rem = x % y;
    if rem != 0 && (x < 0) != (y < 0) {
        div - 1
    } else {
        div
    }
}
