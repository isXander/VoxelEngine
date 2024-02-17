use lazy_static::lazy_static;
use regex::Regex;
use std::collections::HashMap;
use std::path::Path;

pub struct ShaderPreprocessor {
    root_directory: &'static Path,

    compiled_shaders: HashMap<String, wgpu::ShaderModule>,
    processed_shaders: HashMap<String, String>,
}

lazy_static! {
    static ref INCLUDE_DIRECTIVE_REGEX: Regex = Regex::new(r#"^\s*#include "(.+)"\s*$"#).unwrap();
}

impl ShaderPreprocessor {
    pub fn new(root_directory: &'static Path) -> Self {
        Self {
            root_directory,
            compiled_shaders: HashMap::new(),
            processed_shaders: HashMap::new(),
        }
    }

    pub fn precompile_shaders(&mut self, device: &wgpu::Device, shaders: Vec<&str>) {
        for shader in shaders {
            let shader_path = self.root_directory.join(shader);
            let shader_content = std::fs::read_to_string(shader_path).unwrap();

            let processed = self.preprocess_shader(shader_content, shader, &self.root_directory);
            let compiled = Self::compile_shader(&processed, shader, device);

            self.processed_shaders.insert(shader.to_string(), processed);
            self.compiled_shaders.insert(shader.to_string(), compiled);
        }
    }

    pub fn get_compiled_shader(&self, name: &str) -> Option<&wgpu::ShaderModule> {
        self.compiled_shaders.get(name)
    }

    pub fn get_or_compile_shader(
        &mut self,
        name: &str,
        device: &wgpu::Device,
    ) -> &wgpu::ShaderModule {
        let path = &self.root_directory.join(name);

        if self.compiled_shaders.contains_key(name) {
            return self.compiled_shaders.get(name).unwrap();
        }

        let preprocessed = self.get_or_preprocess_shader(name, path);

        let compiled = Self::compile_shader(preprocessed, name, device);
        self.compiled_shaders.insert(name.to_string(), compiled);
        self.compiled_shaders.get(name).unwrap()
    }

    fn get_or_preprocess_shader(&mut self, name: &str, path: &Path) -> &String {
        let path_str = path.to_str().unwrap();
        if self.processed_shaders.contains_key(path_str) {
            return self.processed_shaders.get(path_str).unwrap();
        }

        println!("Preprocessing shader: {}", path_str);
        let unprocessed = std::fs::read_to_string(path).unwrap();

        let parent_directory = path.parent().unwrap();
        let processed = self.preprocess_shader(unprocessed, name, parent_directory);

        self.processed_shaders.insert(name.to_string(), processed);
        self.processed_shaders.get(name).unwrap()
    }

    fn preprocess_shader(
        &mut self,
        unprocessed: String,
        name: &str,
        parent_directory: &Path,
    ) -> String {
        let mut processed = unprocessed.clone();

        let mut count = 0;
        let lines = unprocessed.lines();
        for (i, line) in lines.enumerate() {
            if let Some(captures) = INCLUDE_DIRECTIVE_REGEX.captures(line) {
                let include_path = captures.get(1).unwrap().as_str();

                let include_full_path = if include_path.starts_with('~') {
                    let include_path = include_path.strip_prefix('~').unwrap();
                    self.root_directory.join(include_path)
                } else {
                    parent_directory.join(include_path)
                };

                let preprocessed_include =
                    self.get_or_preprocess_shader(include_path, &include_full_path);

                processed = Self::include_shader(processed.clone(), preprocessed_include, i);
                count += 1;
            }
        }

        println!(
            "Processed {} include directives for shader {}.",
            count, name
        );

        processed
    }

    fn include_shader(base: String, include: &String, insert_line: usize) -> String {
        let mut lines = base.lines().collect::<Vec<&str>>();
        lines.remove(insert_line);
        lines.insert(insert_line, include.as_str());
        lines.join("\n")
    }

    fn compile_shader(processed: &String, name: &str, device: &wgpu::Device) -> wgpu::ShaderModule {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(processed.into()),
        })
    }
}
