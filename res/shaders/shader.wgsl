#include "fog.wgsl"

// Vertex Shader

struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,

    // rotation
    @location(9) normal_matrix_0: vec3<f32>,
    @location(10) normal_matrix_1: vec3<f32>,
    @location(11) normal_matrix_2: vec3<f32>,
};

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) bitangent: vec3<f32>,
};

// declare struct to store output of our shader
struct VertexOutput {
    // builtin tells wgpu that this value is what we want to use for clip coords, aka gl_Position
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) tangent_position: vec3<f32>,
    @location(2) tangent_light_position: vec3<f32>,
//    @location(3) tangent_view_position: vec3<f32>,
    @location(3) fog_distance: f32,
};

struct CameraUniform {
//    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
    view_matrix: mat4x4<f32>,
}
@group(1) @binding(0) // need to specify the diff bind group
var<uniform> camera: CameraUniform;

@vertex // entrypoint
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );
    let normal_matrix = mat3x3<f32>(
        instance.normal_matrix_0,
        instance.normal_matrix_1,
        instance.normal_matrix_2,
    );

    let model_view_matrix = model_matrix * camera.view_matrix;

//    let world_normal = normalize(normal_matrix * model.normal);
//    let world_tangent = normalize(normal_matrix * model.tangent);
//    let world_bitangent = normalize(normal_matrix * model.bitangent);
//    let tangent_matrix = transpose(mat3x3<f32>(
//        world_tangent,
//        world_bitangent,
//        world_normal,
//    ));

    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
//    var world_position: vec4<f32> = model_matrix * vec4<f32>(model.position, 1.0);
//    out.clip_position = model_view_matrix * vec4<f32>(model.position, 1.0);
//    out.tangent_position = tangent_matrix * world_position.xyz;
////    out.tangent_view_position = tangent_matrix * camera.view_pos.xyz;
//    out.tangent_light_position = tangent_matrix * light.position;
//    out.fog_distance = fog_distance_vert(model_matrix * camera.view_matrix, model.position, true);

    out.clip_position = camera.view_proj * model_view_matrix * vec4<f32>(model.position, 1.0);
    out.fog_distance = fog_distance_vert(model_view_matrix, model.position, true);
    return out;
}

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;
@group(0)@binding(2)
var t_normal: texture_2d<f32>;
@group(0) @binding(3)
var s_normal: sampler;
@group(0) @binding(4)
var t_specular: texture_2d<f32>;
@group(0) @binding(5)
var s_specular: sampler;

struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
    intensity: f32,
    range: f32,
}
@group(2) @binding(0)
var<uniform> light: Light;

const coordinate_system = mat3x3<f32>(
    vec3(1, 0, 0), // x-axis (right)
    vec3(0, 1, 0), // y-axis (up)
    vec3(0, 0, 1)  // z-axis (forward)
);

@fragment // entrypoint
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let object_color = textureSample(t_diffuse, s_diffuse, in.tex_coords); // sets colour of frag to brown
    let object_normal = textureSample(t_normal, s_normal, in.tex_coords);
    let object_specular = textureSample(t_specular, s_specular, in.tex_coords);

    // without ambient lighting, shadows would be pitch black
    let ambient_strength = 0.2;
    let ambient_color = light.color * ambient_strength;

    // Create the lighting vectors
//    let tangent_normal = object_normal.xyz * 2.0 - 1.0;
//    let light_dir = normalize(in.tangent_light_position - in.tangent_position);
//    let view_dir = normalize(in.tangent_view_position - in.tangent_position);
//    let half_dir = normalize(view_dir + light_dir);

    // light.intensity is the brightness of the light (0.0 - 1.0)
    // light.range is the distance the light can travel

//    let diffuse_strength = max(dot(tangent_normal, light_dir), 0.0) * light.intensity * (1.0 - clamp(length(in.tangent_light_position - in.tangent_position) / light.range, 0.0, 1.0));
//    let diffuse_color = light.color * diffuse_strength * (1 - ambient_strength);

//    let specular_strength = pow(max(dot(tangent_normal, half_dir), 0.0), 32.0) * object_specular.x;
//    let specular_color = specular_strength * light.color;

    let lit_result = (ambient_color/* + diffuse_color + specular_color*/) * object_color.xyz;
    var output = vec4<f32>(lit_result, object_color.a);
    output = fog_linear_frag(output, in.fog_distance, 50.0, 500.0, vec4<f32>(0.1, 0.2, 0.3, 1.0));

    return output;
}
