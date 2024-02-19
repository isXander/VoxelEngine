pub mod engine;
pub mod voxel;
pub mod world;

pub async fn run() {
    engine::run().await;
}