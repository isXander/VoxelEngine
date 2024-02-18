pub mod engine;
pub mod voxel;
pub mod world;

pub async fn run() {
    //engine::render::fast_inv_sqrt(2552.2435);
    engine::run().await;
}