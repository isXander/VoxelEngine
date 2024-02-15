fn fog_distance_vert(model_view_matrix: mat4x4<f32>, pos: vec3<f32>, shape: bool) -> f32 {
    if shape {
        return length((model_view_matrix * vec4<f32>(pos, 1.0)).xyz);
    } else {
        let dist_xz = length((model_view_matrix * vec4<f32>(pos.x, 0.0, pos.z, 1.0)).xyz);
        let dist_y = length((model_view_matrix * vec4<f32>(0.0, pos.y, 0.0, 1.0)).xyz);
        return max(dist_xz, dist_y);
    }
}

fn fog_linear_frag(in_color: vec4<f32>, vertex_distance: f32, fog_start: f32, fog_end: f32, fog_color: vec4<f32>) -> vec4<f32> {
    if vertex_distance <= fog_start {
        return in_color;
    }

    let fog_value = select(1.0, smoothstep(fog_start, fog_end, vertex_distance), vertex_distance < fog_end);
    return vec4<f32>(mix(in_color.rgb, fog_color.rgb, fog_value * fog_color.a), in_color.a);
}