use crate::voxel::chunk::ChunkView;
use crate::world::physics::context::{PhysicsConfig, PhysicsContext};
use hecs::World;
use hecs_schedule::{Schedule, ScheduleBuilder};
use winit::event::WindowEvent;
use crate::engine::render::RenderContext;
use crate::engine::resources::ResourceManager;

pub struct Context {
    pub physics_context: PhysicsContext,
    pub physics_config: PhysicsConfig,
}

pub struct App {
    pub world: World,

    pub context: Context,

    pub start_schedule: Schedule,
    pub update_schedule: Schedule,
    pub fixed_update_schedule: Schedule,
    pub input_schedule: Schedule,
}

struct Unit;

impl App {
    pub fn builder() -> AppBuilder {
        AppBuilder {
            start_schedule: Schedule::builder(),
            update_schedule: Schedule::builder(),
            fixed_update_schedule: Schedule::builder(),
            input_schedule: Schedule::builder(),
        }
    }

    pub fn run_start_stage(&mut self, chunk_view: &mut ChunkView) {
        self.start_schedule
            .execute((&mut self.world, chunk_view))
            .expect("Failed to run start schedule");
    }

    pub fn run_update_stage(&mut self, chunk_view: &mut ChunkView, render_context: &mut RenderContext) {
        self.update_schedule
            .execute((
                &mut self.world,
                chunk_view,
                render_context,
                &mut self.context.physics_context,
                &mut self.context.physics_config,
            ))
            .expect("Failed to run update schedule");
    }

    pub fn run_fixed_update_stage(&mut self) {
        self.fixed_update_schedule
            .execute((
                &mut self.world,
                &mut self.context.physics_context,
                &mut self.context.physics_config,
            ))
            .expect("Failed to run fixed update schedule");
    }

    pub fn run_input_stage(&mut self, chunk_view: &mut ChunkView, event: &mut WindowEvent) {
        self.input_schedule
            .execute((&mut self.world, chunk_view, event))
            .expect("Failed to run input schedule");
    }
}

pub struct AppBuilder {
    pub start_schedule: ScheduleBuilder,
    pub update_schedule: ScheduleBuilder,
    pub fixed_update_schedule: ScheduleBuilder,
    pub input_schedule: ScheduleBuilder,
}

impl AppBuilder {
    pub fn inject_plugin(&mut self, plugin: impl Plugin) {
        plugin.inject_systems(self);
    }

    pub fn build(&mut self, context: Context) -> App {
        App {
            world: World::new(),
            context,
            start_schedule: self.start_schedule.build(),
            update_schedule: self.update_schedule.build(),
            fixed_update_schedule: self.fixed_update_schedule.build(),
            input_schedule: self.input_schedule.build(),
        }
    }
}

pub trait Plugin {
    fn inject_systems(&self, app: &mut AppBuilder);

    // TODO: figure out a way to inject data to the execute block / add more context to the app
}
