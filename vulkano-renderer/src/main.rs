#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use self::livesplit_core::{
    layout::{self, Layout, LayoutSettings},
    run::parser::composite,
    Run, Segment, Timer,
};
use livesplit_rendering::{
    core as livesplit_core, Backend, IndexPair, Mesh, Renderer, Rgba, Transform,
};
use std::sync::Arc;
use std::{
    fs::File,
    io::{BufReader, Seek, SeekFrom},
};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSetImg;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSetSampler;
use vulkano::instance::debug::{DebugCallback, MessageTypes};
use vulkano::instance::InstanceExtensions;
use vulkano::pipeline::vertex::VertexSource;
use vulkano::pipeline::GraphicsPipelineAbstract;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{
        pool::standard::StandardCommandPoolBuilder, AutoCommandBufferBuilder, CommandBuffer,
        DynamicState,
    },
    descriptor::descriptor_set::PersistentDescriptorSet,
    device::{Device, DeviceExtensions, Features, Queue},
    format::{ClearValue, Format},
    framebuffer::{Framebuffer, Subpass},
    image::{AttachmentImage, Dimensions, ImageLayout, ImageUsage, ImmutableImage, MipmapsCount},
    instance::{self, Instance, PhysicalDevice},
    pipeline::{blend, viewport::Viewport, GraphicsPipeline},
    sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode},
    swapchain::{
        self, AcquireError, PresentMode, SurfaceTransform, Swapchain, SwapchainCreationError,
    },
    sync::{self, FlushError, GpuFuture},
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    dpi, ElementState, Event, EventsLoop, Icon, KeyboardInput, MouseScrollDelta, VirtualKeyCode,
    Window, WindowBuilder, WindowEvent,
};

struct LongLivingData<'display, GraphicsPipeline> {
    window: &'display Window,
    device: &'display Arc<Device>,
    queue: &'display Arc<Queue>,
    meshes: Vec<(
        Arc<CpuAccessibleBuffer<[MyVertex]>>,
        Arc<CpuAccessibleBuffer<[u16]>>,
    )>,
    textures: Vec<
        Arc<
            PersistentDescriptorSet<
                GraphicsPipeline,
                (
                    ((), PersistentDescriptorSetImg<Arc<ImmutableImage<Format>>>),
                    PersistentDescriptorSetSampler,
                ),
            >,
        >,
    >,
    pipeline_texture: &'display GraphicsPipeline,
    pipeline_no_texture: &'display GraphicsPipeline,
    dynamic_state: DynamicState,
}

struct VulkanoBackend<'frame, 'display: 'frame, GraphicsPipeline> {
    data: &'frame mut LongLivingData<'display, GraphicsPipeline>,
    cmd_builder: Option<AutoCommandBufferBuilder<StandardCommandPoolBuilder>>,
}

impl<'frame, 'display: 'frame, Gp> Backend for VulkanoBackend<'frame, 'display, Gp>
where
    Gp: GraphicsPipelineAbstract
        + VertexSource<Arc<CpuAccessibleBuffer<[MyVertex]>>>
        + Send
        + Sync
        + 'static
        + Clone,
{
    fn create_mesh(&mut self, Mesh { vertices, indices }: &Mesh) -> IndexPair {
        let vertices = CpuAccessibleBuffer::from_iter(
            self.data.device.clone(),
            BufferUsage::all(),
            vertices.iter().map(|v| MyVertex {
                position: [v.x, v.y],
                texcoord: [v.u, v.v],
            }),
        )
        .unwrap();

        let indices = CpuAccessibleBuffer::from_iter(
            self.data.device.clone(),
            BufferUsage::all(),
            indices.iter().cloned(),
        )
        .unwrap();

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

        #[repr(C)]
        struct Constants {
            transform: [[f32; 4]; 2],
            color_tl: [f32; 4],
            color_tr: [f32; 4],
            color_bl: [f32; 4],
            color_br: [f32; 4],
        }

        if let Some([tex_idx, _, _]) = texture {
            self.cmd_builder = Some(
                self.cmd_builder
                    .take()
                    .unwrap()
                    .draw_indexed(
                        self.data.pipeline_texture.clone(),
                        &self.data.dynamic_state,
                        vertices.clone(),
                        indices.clone(),
                        self.data.textures[tex_idx].clone(),
                        Constants {
                            transform: [[x1, y1, z1, 0.0], [x2, y2, z2, 0.0]],
                            color_tl: tl,
                            color_tr: tr,
                            color_bl: bl,
                            color_br: br,
                        },
                    )
                    .unwrap(),
            );
        } else {
            self.cmd_builder = Some(
                self.cmd_builder
                    .take()
                    .unwrap()
                    .draw_indexed(
                        self.data.pipeline_no_texture.clone(),
                        &self.data.dynamic_state,
                        vertices.clone(),
                        indices.clone(),
                        (),
                        Constants {
                            transform: [[x1, y1, z1, 0.0], [x2, y2, z2, 0.0]],
                            color_tl: tl,
                            color_tr: tr,
                            color_bl: bl,
                            color_br: br,
                        },
                    )
                    .unwrap(),
            );
        }
    }

    fn free_mesh(&mut self, [vbo, ebo, _]: IndexPair) {}

    fn create_texture(&mut self, width: u32, height: u32, data: &[u8]) -> IndexPair {
        let source = CpuAccessibleBuffer::from_iter(
            self.data.queue.device().clone(),
            BufferUsage::transfer_source(),
            data.iter().cloned(),
        )
        .unwrap();

        let (image, init) = ImmutableImage::uninitialized(
            self.data.device.clone(),
            Dimensions::Dim2d { width, height },
            Format::R8G8B8A8Unorm,
            MipmapsCount::Log2,
            ImageUsage {
                transfer_source: true, // for blits
                transfer_destination: true,
                sampled: true,
                ..ImageUsage::none()
            },
            ImageLayout::ShaderReadOnlyOptimal,
            self.data.device.active_queue_families(),
        )
        .unwrap();

        let init = Arc::new(init);

        let mip_levels = image.mipmap_levels();

        let dimensions = Dimensions::Dim2d { width, height };

        let mut cb =
            AutoCommandBufferBuilder::new(self.data.device.clone(), self.data.queue.family())
                .unwrap()
                .copy_buffer_to_image_dimensions(
                    source,
                    init.clone(),
                    [0, 0, 0],
                    dimensions.width_height_depth(),
                    0,
                    dimensions.array_layers_with_cube(),
                    0,
                )
                .unwrap();

        let (mut mip_width, mut mip_height) = (width as i32, height as i32);

        for mip_level in 1..mip_levels {
            cb = cb
                .blit_image(
                    init.clone(),
                    [0, 0, 0],
                    [mip_width, mip_height, 1],
                    0,
                    mip_level - 1,
                    init.clone(),
                    [0, 0, 0],
                    [
                        if mip_width > 1 { mip_width / 2 } else { 1 },
                        if mip_height > 1 { mip_height / 2 } else { 1 },
                        1,
                    ],
                    0,
                    mip_level,
                    1,
                    Filter::Linear,
                )
                .unwrap();

            if mip_width > 1 {
                mip_width /= 2;
            }
            if mip_height > 1 {
                mip_height /= 2;
            }
        }

        let image_future = cb
            .build()
            .unwrap()
            .execute(self.data.queue.clone())
            .unwrap();

        // TODO: Idk if this is necessary
        // image_future
        //     .then_signal_fence_and_flush()
        //     .unwrap()
        //     .wait(None)
        //     .unwrap();

        let sampler = Sampler::new(
            self.data.device.clone(),
            Filter::Linear,
            Filter::Linear,
            MipmapMode::Linear,
            SamplerAddressMode::ClampToEdge,
            SamplerAddressMode::ClampToEdge,
            SamplerAddressMode::ClampToEdge,
            0.0,
            1.0,
            0.0,
            mip_levels as _,
        )
        .unwrap();

        let set = Arc::new(
            PersistentDescriptorSet::start(self.data.pipeline_texture.clone(), 0)
                .add_sampled_image(image, sampler)
                .unwrap()
                .build()
                .unwrap(),
        );

        let idx = self.data.textures.len();
        self.data.textures.push(set);
        [idx, 0, 0]
    }

    fn free_texture(&mut self, [texture, _, _]: IndexPair) {}

    fn resize(&mut self, height: f32) {
        // FIXME: Resizing doesn't just affect the height when the DPI is not
        // 100% on at least Windows.
        let window = self.data.window;
        let dpi = window.get_hidpi_factor();
        let old_logical_size = window.get_inner_size().unwrap();
        let new_physical_size = dpi::PhysicalSize::new(0.0, height as f64).to_logical(dpi);
        let new_logical_size =
            dpi::LogicalSize::new(old_logical_size.width as f64, new_physical_size.height);
        window.set_inner_size(new_logical_size);
    }
}

#[derive(Copy, Clone)]
struct MyVertex {
    position: [f32; 2],
    texcoord: [f32; 2],
}

vulkano::impl_vertex!(MyVertex, position, texcoord);

fn main() {
    #[cfg(not(debug_assertions))]
    let (layers, extensions) = (None, vulkano_win::required_extensions());
    #[cfg(debug_assertions)]
    let (layers, extensions) = (
        instance::layers_list()
            .ok()
            .and_then(|mut layers| {
                layers.find(|l| l.name() == "VK_LAYER_LUNARG_standard_validation")
            })
            .map(|_| "VK_LAYER_LUNARG_standard_validation"),
        InstanceExtensions {
            ext_debug_report: true,
            ..vulkano_win::required_extensions()
        },
    );
    let instance = Instance::new(None, &extensions, layers).unwrap();

    #[cfg(debug_assertions)]
    let all = MessageTypes {
        error: true,
        warning: true,
        performance_warning: true,
        information: true,
        debug: true,
    };

    #[cfg(debug_assertions)]
    let _debug_callback = DebugCallback::new(&instance, all, |msg| {
        let ty = if msg.ty.error {
            "error"
        } else if msg.ty.warning {
            "warning"
        } else if msg.ty.performance_warning {
            "performance_warning"
        } else if msg.ty.information {
            "information"
        } else if msg.ty.debug {
            "debug"
        } else {
            panic!("no-impl");
        };
        println!("{} {}: {}", msg.layer_prefix, ty, msg.description);
    })
    .ok();

    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
    let queue_family = physical
        .queue_families()
        .find(|&q| q.supports_graphics())
        .unwrap();

    let (device, mut queues) = Device::new(
        physical,
        &Features::none(),
        &DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::none()
        },
        [(queue_family, 0.5)].iter().cloned(),
    )
    .unwrap();
    let queue = queues.next().unwrap();

    let mut events_loop = EventsLoop::new();
    let surface = WindowBuilder::new()
        .with_dimensions((300, 500).into())
        .with_title("LiveSplit One")
        .with_window_icon(Some(
            Icon::from_bytes(include_bytes!("../../icon.png")).unwrap(),
        ))
        .with_resizable(true)
        .with_transparency(true)
        .build_vk_surface(&events_loop, instance.clone())
        .unwrap();
    let window = surface.window();

    let caps = surface.capabilities(physical).unwrap();
    let dimensions = caps.current_extent.unwrap_or([1280, 1024]);
    let alpha = caps.supported_composite_alpha.iter().next().unwrap();
    let format = caps.supported_formats[0].0;

    let (mut swapchain, images) = Swapchain::new(
        device.clone(),
        surface.clone(),
        caps.min_image_count,
        format,
        dimensions,
        1,
        caps.supported_usage_flags,
        &queue,
        SurfaceTransform::Identity,
        alpha,
        PresentMode::Fifo,
        true,
        None,
    )
    .unwrap();

    let vs = vs::Shader::load(device.clone()).unwrap();
    let fs_no_texture = fs_no_texture::Shader::load(device.clone()).unwrap();
    let fs_texture = fs_texture::Shader::load(device.clone()).unwrap();

    let [mut width, mut height] = images[0].dimensions();

    let samples = 8;

    let mut intermediary = AttachmentImage::transient_multisampled(
        device.clone(),
        [width, height],
        samples,
        swapchain.format(),
    )
    .unwrap();

    let render_pass = Arc::new(
        vulkano::single_pass_renderpass!(device.clone(),
            attachments: {
                intermediary: {
                    load: Clear,
                    store: DontCare,
                    format: swapchain.format(),
                    samples: samples,
                },
                color: {
                    load: DontCare,
                    store: Store,
                    format: swapchain.format(),
                    samples: 1,
                }
            },
            pass: {
                color: [intermediary],
                depth_stencil: {},
                resolve: [color],
            }
        )
        .unwrap(),
    );

    let mut framebuffers = images
        .iter()
        .map(|image| {
            Arc::new(
                Framebuffer::start(render_pass.clone())
                    .add(intermediary.clone())
                    .unwrap()
                    .add(image.clone())
                    .unwrap()
                    .build()
                    .unwrap(),
            )
        })
        .collect::<Vec<_>>();

    let pipeline_no_texture = Arc::new(
        GraphicsPipeline::start()
            .vertex_input_single_buffer::<MyVertex>()
            .vertex_shader(vs.main_entry_point(), ())
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs_no_texture.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .blend_collective(blend::AttachmentBlend {
                alpha_source: blend::BlendFactor::One,
                ..blend::AttachmentBlend::alpha_blending()
            })
            .build(device.clone())
            .unwrap(),
    );
    let pipeline_texture = Arc::new(
        GraphicsPipeline::start()
            .vertex_input_single_buffer::<MyVertex>()
            .vertex_shader(vs.main_entry_point(), ())
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs_texture.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .blend_collective(blend::AttachmentBlend {
                alpha_source: blend::BlendFactor::One,
                ..blend::AttachmentBlend::alpha_blending()
            })
            .build(device.clone())
            .unwrap(),
    );

    let dynamic_state = DynamicState {
        viewports: Some(vec![Viewport {
            origin: [0.0, 0.0],
            dimensions: [width as f32, height as f32],
            depth_range: 0.0..1.0,
        }]),
        ..DynamicState::none()
    };

    // let path = r"C:\Users\Christopher Serr\Documents\Splits\Portal - Inbounds.lss";
    // let file = BufReader::new(File::open(path).unwrap());
    // let mut run = composite::parse(file, Some(path.into()), true).unwrap().run;

    let mut run = Run::new();
    run.set_game_name("Game");
    run.set_category_name("Category");
    run.push_segment(Segment::new("Time"));

    run.fix_splits();
    let mut timer = Timer::new(run).unwrap();

    let mut layout = Layout::default_layout();
    layout.general_settings_mut().background = livesplit_core::settings::Gradient::Plain(
        livesplit_core::settings::Color::hsla(0.0, 0.0, 0.06, 0.75),
    );

    let mut renderer = Renderer::new();

    let mut long_living_data = LongLivingData {
        window,
        queue: &queue,
        device: &device,
        dynamic_state,
        meshes: vec![],
        textures: vec![],
        pipeline_texture: &pipeline_texture,
        pipeline_no_texture: &pipeline_no_texture,
    };

    let mut previous_frame_end = Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>;

    let mut closed = false;
    let mut recreate_swapchain = false;
    while !closed {
        previous_frame_end.cleanup_finished();

        if recreate_swapchain {
            if let Some(dimensions) = window.get_inner_size() {
                let dimensions: (u32, u32) =
                    dimensions.to_physical(window.get_hidpi_factor()).into();
                width = dimensions.0;
                height = dimensions.1;
            } else {
                return;
            }

            let (new_swapchain, new_images) =
                match swapchain.recreate_with_dimension([width, height]) {
                    Ok(r) => r,
                    Err(SwapchainCreationError::UnsupportedDimensions) => continue,
                    Err(err) => panic!("{:?}", err),
                };

            swapchain = new_swapchain;

            long_living_data.dynamic_state = DynamicState {
                viewports: Some(vec![Viewport {
                    origin: [0.0, 0.0],
                    dimensions: [width as f32, height as f32],
                    depth_range: 0.0..1.0,
                }]),
                ..DynamicState::none()
            };

            intermediary = AttachmentImage::transient_multisampled(
                device.clone(),
                [width, height],
                samples,
                swapchain.format(),
            )
            .unwrap();

            framebuffers = new_images
                .iter()
                .map(|image| {
                    Arc::new(
                        Framebuffer::start(render_pass.clone())
                            .add(intermediary.clone())
                            .unwrap()
                            .add(image.clone())
                            .unwrap()
                            .build()
                            .unwrap(),
                    )
                })
                .collect::<Vec<_>>();

            recreate_swapchain = false;
        }

        let layout_state = layout.state(&timer);

        let (image_num, acquire_future) =
            match swapchain::acquire_next_image(swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    recreate_swapchain = true;
                    continue;
                }
                Err(err) => panic!("{:?}", err),
            };

        if height > 0 {
            let mut backend = VulkanoBackend {
                data: &mut long_living_data,
                cmd_builder: AutoCommandBufferBuilder::primary_one_time_submit(
                    device.clone(),
                    queue.family(),
                )
                .unwrap()
                .begin_render_pass(
                    framebuffers[image_num].clone(),
                    false,
                    vec![[0.0, 0.0, 0.0, 0.0].into(), ClearValue::None],
                )
                .unwrap()
                .into(),
            };

            renderer.render(&mut backend, (width as _, height as _), &layout_state);

            let command_buffer = backend
                .cmd_builder
                .take()
                .unwrap()
                .end_render_pass()
                .unwrap()
                .build()
                .unwrap();

            let future = previous_frame_end
                .join(acquire_future)
                .then_execute(queue.clone(), command_buffer)
                .unwrap()
                .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                .then_signal_fence_and_flush();

            match future {
                Ok(future) => {
                    previous_frame_end = Box::new(future) as Box<_>;
                }
                Err(FlushError::OutOfDate) => {
                    recreate_swapchain = true;
                    previous_frame_end = Box::new(sync::now(device.clone())) as Box<_>;
                }
                Err(e) => {
                    println!("{:?}", e);
                    previous_frame_end = Box::new(sync::now(device.clone())) as Box<_>;
                }
            }
        }

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
                WindowEvent::MouseWheel { delta, .. } => {
                    let mut scroll = match delta {
                        MouseScrollDelta::LineDelta(_, y) => -y as i32,
                        MouseScrollDelta::PixelDelta(delta) => (delta.y / 15.0) as i32,
                    };
                    while scroll < 0 {
                        layout.scroll_up();
                        scroll += 1;
                    }
                    while scroll > 0 {
                        layout.scroll_down();
                        scroll -= 1;
                    }
                }
                // WindowEvent::MouseInput {
                //     button: MouseButton::Left,
                //     state: ElementState::Pressed,
                //     ..
                // } => {
                //     dragging = Some((
                //         cached_mouse_pos,
                //         display.gl_window().get_position().unwrap(),
                //     ));
                // }
                // WindowEvent::MouseInput {
                //     button: MouseButton::Left,
                //     state: ElementState::Released,
                //     ..
                // } => {
                //     dragging = None;
                // }
                WindowEvent::DroppedFile(path) => {
                    let mut file = BufReader::new(File::open(&path).unwrap());
                    if composite::parse(&mut file, Some(path), true)
                        .map_err(drop)
                        .and_then(|run| timer.set_run(run.run).map_err(drop))
                        .is_err()
                    {
                        let _ = file.seek(SeekFrom::Start(0));
                        if let Ok(settings) = LayoutSettings::from_json(&mut file) {
                            layout = Layout::from_settings(settings);
                        } else {
                            let _ = file.seek(SeekFrom::Start(0));
                            if let Ok(parsed_layout) = layout::parser::parse(&mut file) {
                                layout = parsed_layout;
                            }
                        }
                    }
                }
                _ => {}
            },
            _ => {}
        });
    }
}

mod vs {
    vulkano_shaders::shader! {
    ty: "vertex",
        src: "
#version 450

layout(push_constant) uniform Data {
    mat2x4 transform;
    vec4 color_tl;
    vec4 color_tr;
    vec4 color_bl;
    vec4 color_br;
} data;

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texcoord;

layout(location = 0) out vec4 color;
layout(location = 1) out vec2 outTexcoord;

void main() {
    vec4 left = mix(data.color_tl, data.color_bl, texcoord.y);
    vec4 right = mix(data.color_tr, data.color_br, texcoord.y);
    color = mix(left, right, texcoord.x);

    vec2 pos = vec4(position, 1, 0) * data.transform;
    gl_Position = vec4(vec2(2, 2) * pos.xy + vec2(-1, -1), 0, 1);
    outTexcoord = texcoord;
}"
    }
}

mod fs_texture {
    vulkano_shaders::shader! {
    ty: "fragment",
        src: "
#version 450

layout(push_constant) uniform Data {
    mat2x4 transform;
    vec4 color_tl;
    vec4 color_tr;
    vec4 color_bl;
    vec4 color_br;
} data;

layout(location = 0) in vec4 color;
layout(location = 1) in vec2 texcoord;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D tex;

void main() {
    outColor = color * texture(tex, texcoord);
}"
    }
}

mod fs_no_texture {
    vulkano_shaders::shader! {
    ty: "fragment",
        src: "
#version 450

layout(push_constant) uniform Data {
    mat2x4 transform;
    vec4 color_tl;
    vec4 color_tr;
    vec4 color_bl;
    vec4 color_br;
} data;

layout(location = 0) in vec4 color;
layout(location = 1) in vec2 texcoord;
layout(location = 0) out vec4 outColor;

void main() {
    outColor = color;
}"
    }
}
