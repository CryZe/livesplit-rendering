#![windows_subsystem = "windows"]

extern crate gl;
extern crate glutin;
// extern crate iui;
extern crate azul;
extern crate livesplit_rendering;

use gl::types::{GLchar, GLint, GLuint};
use glutin::{
    DeviceEvent, ElementState, Event, GlContext, KeyboardInput, MouseScrollDelta, VirtualKeyCode,
    WindowEvent,
};
// use iui::controls::{Button, Combobox, VerticalBox};
// use iui::prelude::*;

use std::fs::File;
use std::io::{BufReader, Seek, SeekFrom};
use std::sync::{Arc, Mutex};

use livesplit_core::{
    component::{
        blank_space, current_comparison, current_pace, delta, detailed_timer, graph,
        possible_time_save, previous_segment, separator, splits, sum_of_best, text, timer, title,
        total_playtime,
    },
    layout::{editor::Editor as LayoutEditor, Layout, LayoutSettings},
    run::parser::composite,
    settings::{Gradient, ListGradient},
    Run, Segment, Timer,
};
use livesplit_rendering::core as livesplit_core;
use livesplit_rendering::{Backend, IndexPair, Mesh, Pos, Renderer, Rgba, Transform, Vertex};

mod layout_editor;

const DEFAULT_VERTEX_SHADER: &str = r#"#version 150
uniform mat3x2 transform;
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

    vec2 pos = transform * vec3(position, 1);
    gl_Position = vec4(vec2(2, -2) * pos + vec2(-1, 1), 0, 1);
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

struct GlBackend {
    pos_attrib: GLuint,
    texcoord_attrib: GLuint,
    transform: GLint,
    color_tl: GLint,
    color_tr: GLint,
    color_bl: GLint,
    color_br: GLint,
    tex: GLint,
    use_texture: GLint,
    new_height: Option<f32>,
}

impl Backend for GlBackend {
    fn create_mesh(&mut self, Mesh { vertices, indices }: &Mesh) -> IndexPair {
        unsafe {
            let mut buffers = [0, 0];
            gl::GenBuffers(2, buffers.as_mut_ptr());
            let [vbo, ebo] = buffers;
            gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
            gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ebo);

            gl::BufferData(
                gl::ARRAY_BUFFER,
                (vertices.len() * std::mem::size_of::<Vertex>()) as _,
                vertices.as_ptr() as *const _,
                gl::STATIC_DRAW,
            );

            gl::BufferData(
                gl::ELEMENT_ARRAY_BUFFER,
                (indices.len() * std::mem::size_of::<u16>()) as _,
                indices.as_ptr() as *const _,
                gl::STATIC_DRAW,
            );

            [vbo as _, ebo as _, indices.len()]
        }
    }

    fn render_mesh(
        &mut self,
        [vbo, ebo, len]: IndexPair,
        transform: Transform,
        [tl, tr, br, bl]: [Rgba; 4],
        texture: Option<IndexPair>,
    ) {
        unsafe {
            gl::BindBuffer(gl::ARRAY_BUFFER, vbo as _);
            gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ebo as _);

            // gl::EnableVertexAttribArray(self.pos_attrib);
            gl::VertexAttribPointer(
                self.pos_attrib,
                2,
                gl::FLOAT,
                gl::FALSE,
                std::mem::size_of::<Vertex>() as _,
                std::ptr::null_mut(),
            );

            // gl::EnableVertexAttribArray(self.texcoord_attrib);
            gl::VertexAttribPointer(
                self.texcoord_attrib,
                2,
                gl::FLOAT,
                gl::FALSE,
                std::mem::size_of::<Vertex>() as _,
                (2 * std::mem::size_of::<f32>()) as *const _,
            );

            gl::UniformMatrix3x2fv(
                self.transform,
                1,
                0,
                transform.to_row_major_array().as_ptr() as *const _,
            );
            gl::Uniform4f(self.color_tl, tl[0], tl[1], tl[2], tl[3]);
            gl::Uniform4f(self.color_tr, tr[0], tr[1], tr[2], tr[3]);
            gl::Uniform4f(self.color_bl, bl[0], bl[1], bl[2], bl[3]);
            gl::Uniform4f(self.color_br, br[0], br[1], br[2], br[3]);

            if let Some([texture, _, _]) = texture {
                gl::ActiveTexture(gl::TEXTURE0);
                gl::BindTexture(gl::TEXTURE_2D, texture as _);
                gl::Uniform1i(self.tex, 0);
                gl::Uniform1f(self.use_texture, 1.0);
            } else {
                gl::Uniform1f(self.use_texture, 0.0);
            }

            gl::DrawElements(
                gl::TRIANGLES,
                len as _,
                gl::UNSIGNED_SHORT,
                std::ptr::null_mut(),
            );
        }
    }

    fn free_mesh(&mut self, [vbo, ebo, _]: IndexPair) {
        unsafe {
            gl::DeleteBuffers(2, [vbo as GLuint, ebo as GLuint].as_ptr());
        }
    }

    fn create_texture(&mut self, width: u32, height: u32, data: &[u8]) -> IndexPair {
        unsafe {
            let mut texture = 0;
            gl::GenTextures(1, &mut texture);

            gl::BindTexture(gl::TEXTURE_2D, texture);
            gl::TexImage2D(
                gl::TEXTURE_2D,
                0,
                gl::RGBA8 as _,
                width as _,
                height as _,
                0,
                gl::RGBA,
                gl::UNSIGNED_BYTE,
                data.as_ptr() as *const _,
            );
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as _);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as _);
            gl::GenerateMipmap(gl::TEXTURE_2D);

            [texture as usize, 0, 0]
        }
    }
    fn free_texture(&mut self, [texture, _, _]: IndexPair) {
        unsafe {
            gl::DeleteTextures(1, &(texture as GLuint));
        }
    }

    fn resize(&mut self, height: f32) {
        self.new_height = Some(height);
    }
}

fn main() {
    // let path = r"C:\Projekte\one-offs\livesplit-one-quicksilver\static\8dj.lss";
    // let path = r"C:\Projekte\one-offs\livesplit-one-quicksilver\static\Portal - Inbounds.lss";
    // let file = BufReader::new(File::open(path).unwrap());
    // let mut run = composite::parse(file, Some(path.into()), true).unwrap().run;
    let mut run = Run::new();
    run.push_segment(Segment::new("Foo"));

    run.fix_splits();
    let mut timer = Timer::new(run).unwrap();

    // let mut layout = Layout::from_settings(
    //     LayoutSettings::from_json(
    //         File::open(r"C:\Projekte\one-offs\livesplit-one-quicksilver\layout (5).ls1l").unwrap(),
    //     ).unwrap(),
    // );
    let mut layout = Layout::new();

    layout.push(title::Component::with_settings(title::Settings {
        text_alignment: livesplit_core::settings::Alignment::Center,
        ..Default::default()
    }));
    layout.push(splits::Component::with_settings(splits::Settings {
        // background: ListGradient::Same(Gradient::Transparent),
        show_thin_separators: true,
        display_two_rows: true,
        visual_split_count: 10,
        show_column_labels: true,
        // columns: vec![
        //     splits::ColumnSettings {
        //         start_with: splits::ColumnStartWith::ComparisonSegmentTime,
        //         update_with: splits::ColumnUpdateWith::SegmentTime,
        //         ..Default::default()
        //     },
        //     splits::ColumnSettings {
        //         start_with: splits::ColumnStartWith::Empty,
        //         update_with: splits::ColumnUpdateWith::SegmentDelta,
        //         ..Default::default()
        //     },
        // ],
        ..Default::default()
    }));

    // layout.push(separator::Component::new());
    // layout.push(timer::Component::with_settings(timer::Settings {
    //     height: 30,
    //     ..Default::default()
    // }));
    // layout.push(separator::Component::new());
    layout.push(Box::new(detailed_timer::Component::with_settings(
        detailed_timer::Settings {
            show_segment_name: true,
            display_icon: true,
            ..Default::default()
        },
    )));
    // layout.push(separator::Component::new());
    // layout.push(pb_chance::Component::new());
    layout.push(previous_segment::Component::with_settings(
        previous_segment::Settings {
            display_two_rows: true,
            ..Default::default()
        },
    ));
    layout.push(delta::Component::with_settings(delta::Settings {
        display_two_rows: true,
        ..Default::default()
    }));
    // layout.push(title::Component::with_settings(title::Settings {
    //     text_alignment: livesplit_core::settings::Alignment::Center,
    //     ..Default::default()
    // }));
    // layout.push(graph::Component::with_settings(graph::Settings {
    //     show_best_segments: true,
    //     ..Default::default()
    // }));

    let layout_editor = Arc::new(Mutex::new(LayoutEditor::new(layout).unwrap()));
    // layout_editor::open_window(layout_editor.clone());

    let mut events_loop = glutin::EventsLoop::new();
    let window = glutin::WindowBuilder::new()
        .with_title("LiveSplit One")
        .with_dimensions((250.0, 500.0).into());
    // .with_decorations(false)
    // .with_transparency(true)
    // .with_resizable(true);

    let context = glutin::ContextBuilder::new()
        .with_srgb(true)
        .with_multisampling(16);
    let gl_window = glutin::GlWindow::new(window, context, &events_loop).unwrap();

    unsafe { gl_window.make_current().unwrap() };

    let gl_context = gl_window.context();
    gl::load_with(|ptr| gl_context.get_proc_address(ptr) as *const _);
    gl::Viewport::load_with(|ptr| gl_context.get_proc_address(ptr) as *const _);

    if cfg!(debug_assertions) {
        let version = unsafe {
            let data = std::ffi::CStr::from_ptr(gl::GetString(gl::VERSION) as *const _)
                .to_bytes()
                .to_vec();
            String::from_utf8(data).unwrap()
        };

        println!("OpenGL version {}", version);
    }

    let (mut width, mut height);
    let mut gl_backend;
    let mut renderer = Renderer::new();

    unsafe {
        gl::BlendFuncSeparate(
            gl::SRC_ALPHA,
            gl::ONE_MINUS_SRC_ALPHA,
            gl::ONE,
            gl::ONE_MINUS_SRC_ALPHA,
        );
        gl::Enable(gl::BLEND);

        let vertex = gl::CreateShader(gl::VERTEX_SHADER);
        gl::ShaderSource(
            vertex,
            1,
            &(DEFAULT_VERTEX_SHADER.as_ptr() as *const GLchar),
            &(DEFAULT_VERTEX_SHADER.len() as GLint),
        );
        gl::CompileShader(vertex);

        if cfg!(debug_assertions) {
            let mut is_compiled = 0;
            gl::GetShaderiv(vertex, gl::COMPILE_STATUS, &mut is_compiled);
            if is_compiled == 0 {
                let mut max_len = 0;
                gl::GetShaderiv(vertex, gl::INFO_LOG_LENGTH, &mut max_len);
                let mut buf = vec![0u8; max_len as usize];
                gl::GetShaderInfoLog(vertex, max_len, &mut max_len, buf.as_mut_ptr() as *mut _);

                let msg = std::ffi::CStr::from_bytes_with_nul(&buf)
                    .unwrap()
                    .to_string_lossy();
                panic!("Failed to compile shader:\n{}", msg);
            }
        }

        let fragment = gl::CreateShader(gl::FRAGMENT_SHADER);
        gl::ShaderSource(
            fragment,
            1,
            &(DEFAULT_FRAGMENT_SHADER.as_ptr() as *const GLchar),
            &(DEFAULT_FRAGMENT_SHADER.len() as GLint),
        );
        gl::CompileShader(fragment);

        if cfg!(debug_assertions) {
            let mut is_compiled = 0;
            gl::GetShaderiv(fragment, gl::COMPILE_STATUS, &mut is_compiled);
            if is_compiled == 0 {
                let mut max_len = 0;
                gl::GetShaderiv(fragment, gl::INFO_LOG_LENGTH, &mut max_len);
                let mut buf = vec![0u8; max_len as usize];
                gl::GetShaderInfoLog(fragment, max_len, &mut max_len, buf.as_mut_ptr() as *mut _);

                let msg = std::ffi::CStr::from_bytes_with_nul(&buf)
                    .unwrap()
                    .to_string_lossy();
                panic!("Failed to compile shader:\n{}", msg);
            }
        }

        let shader = gl::CreateProgram();
        gl::AttachShader(shader, vertex);
        gl::AttachShader(shader, fragment);

        gl::LinkProgram(shader);
        gl::UseProgram(shader);

        let mut vao = 0;
        gl::GenVertexArrays(1, &mut vao);

        let pos_attrib = gl::GetAttribLocation(shader, "position\0".as_ptr() as *const GLchar) as _;
        gl::EnableVertexAttribArray(pos_attrib);
        // gl::VertexAttribPointer(
        //     pos_attrib,
        //     2,
        //     gl::FLOAT,
        //     gl::FALSE,
        //     std::mem::size_of::<Vertex>() as _,
        //     std::ptr::null_mut(),
        // );

        let texcoord_attrib =
            gl::GetAttribLocation(shader, "texcoord\0".as_ptr() as *const GLchar) as _;
        gl::EnableVertexAttribArray(texcoord_attrib);
        // gl::VertexAttribPointer(
        //     texcoord_attrib,
        //     2,
        //     gl::FLOAT,
        //     gl::FALSE,
        //     std::mem::size_of::<Vertex>() as _,
        //     // (2 * std::mem::size_of::<f32>()) as *const _,
        //     std::ptr::null_mut(),
        // );

        let size = gl_window.get_inner_size().unwrap();
        width = size.width as f32;
        height = size.height as f32;
        gl::Viewport(0, 0, size.width as _, size.height as _);

        gl_backend = GlBackend {
            pos_attrib,
            texcoord_attrib,
            transform: gl::GetUniformLocation(shader, "transform\0".as_ptr() as *const GLchar) as _,
            color_tl: gl::GetUniformLocation(shader, "color_tl\0".as_ptr() as *const GLchar) as _,
            color_tr: gl::GetUniformLocation(shader, "color_tr\0".as_ptr() as *const GLchar) as _,
            color_bl: gl::GetUniformLocation(shader, "color_bl\0".as_ptr() as *const GLchar) as _,
            color_br: gl::GetUniformLocation(shader, "color_br\0".as_ptr() as *const GLchar) as _,
            tex: gl::GetUniformLocation(shader, "tex\0".as_ptr() as *const GLchar) as _,
            use_texture: gl::GetUniformLocation(shader, "use_texture\0".as_ptr() as *const GLchar)
                as _,
            new_height: None,
        };
    }

    let mut end_loop = false;
    while !end_loop {
        // end_loop = !event_loop.next_tick(&ui);
        events_loop.poll_events(|event| match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => end_loop = true,
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => unsafe {
                width = size.width as f32;
                height = size.height as f32;
                gl::Viewport(0, 0, size.width as _, size.height as _);
            },
            Event::WindowEvent {
                event: WindowEvent::DroppedFile(path),
                ..
            } => {
                let mut file = BufReader::new(File::open(&path).unwrap());
                if composite::parse(&mut file, Some(path), true)
                    .map_err(drop)
                    .and_then(|run| timer.set_run(run.run).map_err(drop))
                    .is_err()
                {
                    let _ = file.seek(SeekFrom::Start(0));
                    if let Ok(settings) = LayoutSettings::from_json(file) {
                        if let Ok(editor) = LayoutEditor::new(Layout::from_settings(settings)) {
                            *layout_editor.lock().unwrap() = editor;
                        }
                    }
                }
            }
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
                DeviceEvent::MouseWheel { delta } => {
                    let mut scroll = match delta {
                        MouseScrollDelta::LineDelta(_, y) => -y as i32,
                        MouseScrollDelta::PixelDelta(delta) => (delta.y / 15.0) as i32,
                    };
                    // while scroll < 0 {
                    //     layout.scroll_up();
                    //     scroll += 1;
                    // }
                    // while scroll > 0 {
                    //     layout.scroll_down();
                    //     scroll -= 1;
                    // }
                }
                _ => {}
            },
            _ => {}
        });

        let layout_state = layout_editor.lock().unwrap().layout_state(&timer);

        unsafe {
            gl::ClearColor(0.0, 0.0, 0.0, 0.0);
            gl::Clear(gl::COLOR_BUFFER_BIT);
        }

        if height > 0.0 {
            renderer.render(&mut gl_backend, (width, height), &layout_state);
        }

        if let Some(new_height) = gl_backend.new_height.take() {
            height = new_height;
            gl_window.set_inner_size((width as f64, height as f64).into());
        }

        gl_window.swap_buffers().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(33));
    }
}
