use hecs::World;
use winit::event::WindowEvent;

pub struct WorldScheduler {
    pub world: World,

    start_systems: Vec<fn(&mut World)>,
    update_systems: Vec<fn(&mut World, &UpdateStage)>,
    fixed_update_systems: Vec<fn(&mut World)>,
    input_systems: Vec<fn(&mut World, &InputStage)>,
}

impl WorldScheduler {
    pub fn new() -> Self {
        Self {
            world: World::new(),
            start_systems: Vec::new(),
            update_systems: Vec::new(),
            fixed_update_systems: Vec::new(),
            input_systems: Vec::new(),
        }
    }

    pub fn add_start_systems(&mut self, systems: Vec<fn(&mut World)>) {
        self.start_systems.extend(systems);
    }

    pub fn add_update_systems(&mut self, systems: Vec<fn(&mut World, &UpdateStage)>) {
        self.update_systems.extend(systems);
    }

    pub fn add_fixed_update_systems(&mut self, systems: Vec<fn(&mut World)>) {
        self.fixed_update_systems.extend(systems);
    }

    pub fn add_input_systems(&mut self, systems: Vec<fn(&mut World, &InputStage)>) {
        self.input_systems.extend(systems);
    }

    // TODO: potentially just use bevy_ecs with a proper scheduler
    pub fn run_start_stage(&mut self) {
        for system in &self.start_systems {
            system(&mut self.world);
        }
    }

    pub fn run_update_stage(&mut self, delta_time: f32) {
        let stage = UpdateStage { delta_time };
        for system in &self.update_systems {
            system(&mut self.world, &stage);
        }
    }

    pub fn run_fixed_update_stage(&mut self) {
        for system in &self.fixed_update_systems {
            system(&mut self.world);
        }
    }

    pub fn run_input_stage(&mut self, event: &WindowEvent) {
        let stage = InputStage { event };
        for system in &self.input_systems {
            system(&mut self.world, &stage);
        }
    }
}

pub struct StartStage;
pub struct UpdateStage {
    pub delta_time: f32,
}
pub struct FixedUpdateStage;
pub struct InputStage<'a> {
    pub event: &'a WindowEvent,
}