use nalgebra::Vector3;
use rapier3d::prelude::*;

pub struct PhysicsContext {
    // Detects sleeping bodies to reduce computation
    pub islands: IslandManager,
    // Detects potential contact pairs
    pub broad_phase: BroadPhase,
    // Computes contact points, tests interactions, 
    pub narrow_phase: NarrowPhase,
    // Set of rigid bodies part of the sim
    pub bodies: RigidBodySet,
    // Set of colliders part of the sim
    pub colliders: ColliderSet,
    // Set of impulse joints part of the sim
    pub impulse_joints: ImpulseJointSet,
    // Set of multibody joints part of the sim
    pub multibody_joints: MultibodyJointSet,
    // Handles continuous collision detection
    pub ccd_solver: CCDSolver,
    // Handles steppping the sim
    pub physics_pipeline: PhysicsPipeline,
    // Controls low-level coefficient of the sim
    pub integration_parameters: IntegrationParameters,
    pub physics_scale: f32,
    pub event_handler: Option<Box<dyn EventHandler>>,
}

impl PhysicsContext {
    pub fn step(&mut self, config: &PhysicsConfig) {
        self.physics_pipeline.step(
            &config.gravity,
            &self.integration_parameters,
            &mut self.islands,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd_solver,
            None,
            &(),
            &(),
        )
    }
}

impl Default for PhysicsContext {
    fn default() -> Self {
        Self {
            islands: IslandManager::new(),
            broad_phase: BroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            physics_pipeline: PhysicsPipeline::new(),
            integration_parameters: IntegrationParameters::default(),
            physics_scale: 1.0,
            event_handler: None,
        }
    }
}

pub struct PhysicsConfig {
    pub gravity: Vector3<f32>,
}