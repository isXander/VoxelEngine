// use hecs::World;
// use winit::event::WindowEvent;

// use crate::voxel::chunk::ChunkManager;

// use super::physics::PhysicsContext;

// pub struct WorldScheduler {
//     pub world: World,

//     start_systems: Vec<fn(&mut World, &mut Context)>,
//     update_systems: Vec<fn(&mut World, &mut Context, &mut UpdateStage)>,
//     fixed_update_systems: Vec<fn(&mut World, &mut Context)>,
//     input_systems: Vec<fn(&mut World, &mut Context, &InputStage)>,
// }

// impl WorldScheduler {
//     pub fn new() -> Self {
//         Self {
//             world: World::new(),
//             context,
//             start_systems: Vec::new(),
//             update_systems: Vec::new(),
//             fixed_update_systems: Vec::new(),
//             input_systems: Vec::new(),
//         }
//     }

//     pub fn add_start_systems(&mut self, systems: Vec<fn(&mut World, &mut Context)>) {
//         self.start_systems.extend(systems);
//     }

//     pub fn add_update_systems(&mut self, systems: Vec<fn(&mut World, &mut Context, &mut UpdateStage)>) {
//         self.update_systems.extend(systems);
//     }

//     pub fn add_fixed_update_systems(&mut self, systems: Vec<fn(&mut World, &mut Context)>) {
//         self.fixed_update_systems.extend(systems);
//     }

//     pub fn add_input_systems(&mut self, systems: Vec<fn(&mut World, &mut Context, &InputStage)>) {
//         self.input_systems.extend(systems);
//     }

//     // TODO: potentially just use bevy_ecs with a proper scheduler
//     pub fn run_start_stage(&mut self) {
//         for system in &self.start_systems {
//             system(&mut self.world, &mut self.context);
//         }
//     }

//     pub fn run_update_stage(&mut self, chunk_manager: &mut ChunkManager, delta_time: f32) {
//         let mut stage = UpdateStage { chunk_manager, delta_time };
//         for system in &self.update_systems {
//             system(&mut self.world, &mut self.context, &mut stage);
//         }
//     }

//     pub fn run_fixed_update_stage(&mut self) {
//         for system in &self.fixed_update_systems {
//             system(&mut self.world, &mut self.context);
//         }
//     }

//     pub fn run_input_stage(&mut self, event: &WindowEvent) {
//         let stage = InputStage { event };
//         for system in &self.input_systems {
//             system(&mut self.world, &mut self.context, &stage);
//         }
//     }
// }

