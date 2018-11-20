extern crate glium;
extern crate livesplit_rendering;

use glium::{
    glutin,
    index::PrimitiveType,
    program::ProgramCreationInput,
    texture::{ClientFormat, RawImage2d},
    uniform,
    vertex::AttributeType,
    Blend, BlendingFunction, DrawParameters, IndexBuffer, LinearBlendingFactor, Surface, Texture2d,
    VertexBuffer,
};
use glutin::{DeviceEvent, ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use livesplit_core::{
    layout::{editor::Editor as LayoutEditor, Layout, LayoutSettings},
    run::parser::composite,
    Run, Segment, Timer,
};
use livesplit_rendering::{
    core as livesplit_core, Backend, IndexPair, Mesh, Pos, Renderer, Rgba, Transform, Vertex,
};
use std::{
    borrow::Cow,
    fs::File,
    io::{BufReader, Seek, SeekFrom},
    mem,
};

const DEFAULT_VERTEX_SHADER: &str = r#"#version 150
uniform mat3x3 transform;
uniform vec4 color_tl;
uniform vec4 color_tr;
uniform vec4 color_bl;
uniform vec4 color_br;

in vec2 position;
in vec2 texcoord;

out vec4 color;
out vec2 texcoord_inter;

void main() {
    vec4 left = mix(color_tl, color_bl, texcoord.y);
    vec4 right = mix(color_tr, color_br, texcoord.y);
    color = mix(left, right, texcoord.x);

    vec3 pos = vec3(position, 1) * transform;
    gl_Position = vec4(vec2(2, -2) * pos.xy + vec2(-1, 1), 0, 1);
    texcoord_inter = texcoord;
}"#;

const DEFAULT_FRAGMENT_SHADER: &str = r#"#version 150
uniform sampler2D tex;
uniform float use_texture;

in vec4 color;
in vec2 texcoord_inter;

out vec4 outColor;

void main() {
    vec4 tex_color = (use_texture != 0) ? texture(tex, texcoord_inter) : vec4(1, 1, 1, 1);
    outColor = color * tex_color;
}"#;

struct LongLivingData<'display> {
    display: &'display glium::Display,
    program: glium::Program,
    meshes: Vec<(VertexBuffer<Vertex>, IndexBuffer<u16>)>,
    textures: Vec<Texture2d>,
    draw_params: DrawParameters<'static>,
}

struct GliumBackend<'frame, 'display: 'frame> {
    data: &'frame mut LongLivingData<'display>,
    frame: &'frame mut glium::Frame,
}

impl<'frame, 'display: 'frame> Backend for GliumBackend<'frame, 'display> {
    fn create_mesh(&mut self, Mesh { vertices, indices }: &Mesh) -> IndexPair {
        let vertices = unsafe {
            VertexBuffer::new_raw(
                self.data.display,
                vertices,
                Cow::Borrowed(&[
                    (Cow::Borrowed("position"), 0, AttributeType::F32F32, false),
                    (Cow::Borrowed("texcoord"), 8, AttributeType::F32F32, false),
                ]),
                mem::size_of::<livesplit_rendering::Vertex>(),
            ).unwrap()
        };
        let indices =
            IndexBuffer::new(self.data.display, PrimitiveType::TrianglesList, indices).unwrap();
        let idx = self.data.meshes.len();
        self.data.meshes.push((vertices, indices));
        [idx, 0, 0]
    }

    fn render_mesh(
        &mut self,
        [idx, _, _]: IndexPair,
        transform: Transform,
        [tl, tr, br, bl]: [Rgba; 4],
        texture: Option<IndexPair>,
    ) {
        let (vertices, indices) = &self.data.meshes[idx];
        let [x1, y1, z1, x2, y2, z2] = transform.to_column_major_array();
        if let Some([tex_idx, _, _]) = texture {
            self.frame
                .draw(
                    vertices,
                    indices,
                    &self.data.program,
                    &uniform! {
                        transform: [[x1, y1, z1], [x2, y2, z2], [0.0, 0.0, 0.0]],
                        color_tl: tl,
                        color_tr: tr,
                        color_br: br,
                        color_bl: bl,
                        tex: &self.data.textures[tex_idx],
                        use_texture: 1.0f32,
                    },
                    &self.data.draw_params,
                ).unwrap();
        } else {
            self.frame
                .draw(
                    vertices,
                    indices,
                    &self.data.program,
                    &uniform! {
                        transform: [[x1, y1, z1], [x2, y2, z2], [0.0, 0.0, 0.0]],
                        color_tl: tl,
                        color_tr: tr,
                        color_br: br,
                        color_bl: bl,
                        use_texture: 0.0f32,
                    },
                    &self.data.draw_params,
                ).unwrap();
        }
    }

    fn free_mesh(&mut self, [vbo, ebo, _]: IndexPair) {}

    fn create_texture(&mut self, width: u32, height: u32, data: &[u8]) -> IndexPair {
        let texture = Texture2d::new(
            self.data.display,
            RawImage2d {
                data: Cow::Borrowed(data),
                width,
                height,
                format: ClientFormat::U8U8U8U8,
            },
        ).unwrap();
        let idx = self.data.textures.len();
        self.data.textures.push(texture);
        [idx, 0, 0]
    }
    fn free_texture(&mut self, [texture, _, _]: IndexPair) {}

    fn resize(&mut self, height: f32) {}
}

fn main() {
    let mut events_loop = glium::glutin::EventsLoop::new();

    let window = glium::glutin::WindowBuilder::new()
        .with_dimensions((250, 500).into())
        .with_title("LiveSplit One")
        .with_resizable(true);

    let context = glium::glutin::ContextBuilder::new()
        .with_vsync(true)
        .with_hardware_acceleration(None)
        .with_gl(glutin::GlRequest::GlThenGles {
            opengl_version: (3, 2),
            opengles_version: (2, 0),
        }).with_srgb(true);

    let display = glium::Display::new(window, context, &events_loop).unwrap();

    let program = glium::Program::new(
        &display,
        ProgramCreationInput::SourceCode {
            vertex_shader: DEFAULT_VERTEX_SHADER,
            fragment_shader: DEFAULT_FRAGMENT_SHADER,
            geometry_shader: None,
            tessellation_control_shader: None,
            tessellation_evaluation_shader: None,
            transform_feedback_varyings: None,
            outputs_srgb: true,
            uses_point_size: false,
        },
    ).unwrap();

    let mut data = LongLivingData {
        program,
        display: &display,
        meshes: vec![],
        textures: vec![],
        draw_params: DrawParameters {
            blend: Blend {
                color: BlendingFunction::Addition {
                    source: LinearBlendingFactor::SourceAlpha,
                    destination: LinearBlendingFactor::OneMinusSourceAlpha,
                },
                alpha: BlendingFunction::Addition {
                    source: LinearBlendingFactor::One,
                    destination: LinearBlendingFactor::OneMinusSourceAlpha,
                },
                constant_value: (0.0, 0.0, 0.0, 0.0),
            },
            ..Default::default()
        },
    };

    // let path = r"../4cg.lss";
    // let file = BufReader::new(File::open(path).unwrap());
    // let mut run = composite::parse(file, Some(path.into()), true).unwrap().run;

    let mut run = Run::new();
    run.set_game_name("Game");
    run.set_category_name("Category");
    run.push_segment(Segment::new("Time"));

    run.fix_splits();
    let mut timer = Timer::new(run).unwrap();

    let mut layout = Layout::default_layout();
    // layout.general_settings_mut().background = livesplit_core::settings::Gradient::Transparent;

    let mut renderer = Renderer::new();

    let mut closed = false;
    while !closed {
        let layout_state = layout.state(&timer);

        let mut target = display.draw();
        let (width, height) = target.get_dimensions();
        target.clear_color(0.0, 0.0, 0.0, 1.0);
        if height > 0 {
            renderer.render(
                &mut GliumBackend {
                    data: &mut data,
                    frame: &mut target,
                },
                (width as _, height as _),
                &layout_state,
            );
        }
        target.finish().unwrap();

        events_loop.poll_events(|ev| match ev {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => closed = true,
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(key),
                            ..
                        },
                    ..
                } => match key {
                    VirtualKeyCode::Numpad1 => timer.split_or_start(),
                    VirtualKeyCode::Numpad2 => timer.skip_split(),
                    VirtualKeyCode::Numpad3 => timer.reset(true),
                    VirtualKeyCode::Numpad4 => timer.switch_to_previous_comparison(),
                    VirtualKeyCode::Numpad5 => timer.toggle_pause(),
                    VirtualKeyCode::Numpad6 => timer.switch_to_next_comparison(),
                    VirtualKeyCode::Numpad8 => timer.undo_split(),
                    _ => {}
                },
                WindowEvent::DroppedFile(path) => {
                    let mut file = BufReader::new(File::open(&path).unwrap());
                    if composite::parse(&mut file, Some(path), true)
                        .map_err(drop)
                        .and_then(|run| timer.set_run(run.run).map_err(drop))
                        .is_err()
                    {
                        let _ = file.seek(SeekFrom::Start(0));
                        if let Ok(settings) = LayoutSettings::from_json(file) {
                            layout = Layout::from_settings(settings);
                        }
                    }
                }
                _ => {}
            },
            Event::DeviceEvent { event, .. } => match event {
                DeviceEvent::Key(KeyboardInput {
                    state: ElementState::Pressed,
                    virtual_keycode: Some(key),
                    ..
                }) => match key {
                    VirtualKeyCode::Numpad1 => timer.split_or_start(),
                    VirtualKeyCode::Numpad2 => timer.skip_split(),
                    VirtualKeyCode::Numpad3 => timer.reset(true),
                    VirtualKeyCode::Numpad4 => timer.switch_to_previous_comparison(),
                    VirtualKeyCode::Numpad5 => timer.toggle_pause(),
                    VirtualKeyCode::Numpad6 => timer.switch_to_next_comparison(),
                    VirtualKeyCode::Numpad8 => timer.undo_split(),
                    _ => {}
                },
                _ => {}
            },
            _ => {}
        });

        // std::thread::sleep(std::time::Duration::from_millis(33));
    }
}
