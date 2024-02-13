pub mod engine;
pub mod voxel;

pub async fn run() {
    engine::run().await
}