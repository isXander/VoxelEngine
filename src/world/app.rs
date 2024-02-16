use hecs::*;
use hecs_scheduler::*;

pub struct App {
    pub chunk_manager: ChunkManager,
    pub physics: PhysicsContext,
}

pub struct StageManager {
    pub start_schedule: Schedule,
    
}

pub struct StartStage;
pub struct UpdateStage<'a> {
    pub delta_time: f32,
}
pub struct FixedUpdateStage;
pub struct InputStage<'a> {
    pub event: &'a WindowEvent,
}