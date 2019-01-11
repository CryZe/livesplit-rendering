use azul;
use azul::prelude::*;
use livesplit_core::{
    component::{
        blank_space, current_comparison, current_pace, delta, detailed_timer, graph,
        possible_time_save, previous_segment, separator, splits, sum_of_best, text, timer, title,
        total_playtime,
    },
    layout,
    settings::Value,
};
use std::{
    sync::{Arc, Barrier, Mutex},
    thread,
};

pub struct LayoutEditorWindow {
    editor: Arc<Mutex<layout::Editor>>,
    state: layout::editor::State,
    add_menu_open: bool,
}

impl azul::traits::Layout for LayoutEditorWindow {
    fn layout(&self, _window_id: WindowInfo<Self>) -> Dom<Self>
    where
        Self: Sized,
    {
        let mut component_list = Dom::new(NodeType::Div).with_id("component-list");
        for (i, name) in self.state.components.iter().enumerate() {
            let mut child = Dom::new(NodeType::Div)
                .with_child(Dom::new(NodeType::Label(name.clone())).with_class("label"))
                .with_callback(On::MouseUp, Callback(select_component));

            if i == self.state.selected_component as usize {
                child.add_class("selected-component");
            } else if i % 2 == 1 {
                child.add_class("odd-component");
            } else {
                child.add_class("even-component");
            }

            component_list.add_child(child);
        }

        let mut settings_list = Dom::new(NodeType::Div).with_id("layout-settings");
        for (i, field) in self.state.component_settings.fields.iter().enumerate() {
            let mut child = Dom::new(NodeType::Div)
                .with_child(Dom::new(NodeType::Label(field.text.clone())).with_class("label"))
                .with_class("setting");

            match &field.value {
                Value::Bool(active) => child.add_child(
                    Dom::new(NodeType::Div)
                        .with_class("setting-value")
                        .with_child(Dom::new(NodeType::Label(active.to_string())))
                        .with_callback(
                            On::MouseUp,
                            Callback(|app_state: &mut AppState<LayoutEditorWindow>, event| {
                                let (_, parent) =
                                    event.get_index_in_parent(event.hit_dom_node).unwrap();
                                let (index, _) = event.get_index_in_parent(parent).unwrap();
                                let mut state = app_state.data.lock().unwrap();
                                let old_value =
                                    match &state.state.component_settings.fields[index].value {
                                        Value::Bool(val) => *val,
                                        _ => unreachable!(),
                                    };
                                state.state = {
                                    let mut editor = state.editor.lock().unwrap();
                                    editor.set_component_settings_value(
                                        index,
                                        Value::Bool(!old_value),
                                    );
                                    editor.state()
                                };
                                UpdateScreen::Redraw
                            }),
                        ),
                ),
                _ => {}
            }

            settings_list.add_child(child);
        }

        let mut buttons = Dom::new(NodeType::Div).with_id("buttons");

        let mut add_component_button = Dom::new(NodeType::Div)
            .with_child(Dom::new(NodeType::Label(String::from("\u{f067}"))))
            .with_callback(On::MouseUp, Callback(add_component))
            .with_class("clickable-button");

        if self.add_menu_open {
            let mut context_menu = Dom::new(NodeType::Div).with_class("context-menu");

            for &component in &[
                "Current Comparison",
                "Current Pace",
                "Delta",
                "Detailed Timer",
                "Graph",
                "Possible Time Save",
                "Previous Segment",
                "Splits",
                "Sum of Best Segments",
                "Text",
                "Timer",
                "Title",
                "Total Playtime",
                "Blank Space",
                "Separator",
            ] {
                context_menu.add_child(
                    Dom::new(NodeType::Div)
                        .with_class("context-menu-entry")
                        .with_child(Dom::new(NodeType::Label(component.into())).with_class("label"))
                        .with_callback(On::MouseUp, Callback(add_chosen_component)),
                );
            }

            add_component_button.add_child(context_menu);
        }

        buttons.add_child(add_component_button);

        for &(name, callback, is_clickable) in &[
            (
                "\u{f068}",
                Callback(remove_component),
                self.state.buttons.can_remove,
            ),
            ("\u{f0c5}", Callback(duplicate_component), true),
            (
                "\u{f062}",
                Callback(move_component_up),
                self.state.buttons.can_move_up,
            ),
            (
                "\u{f063}",
                Callback(move_component_down),
                self.state.buttons.can_move_down,
            ),
        ] {
            let mut child = Dom::new(NodeType::Div)
                .with_child(Dom::new(NodeType::Label(String::from(name))))
                .with_callback(On::MouseUp, callback);

            if is_clickable {
                child.add_class("clickable-button");
                child.add_callback(On::MouseUp, callback);
            } else {
                child.add_class("unclickable-button");
            }

            buttons.add_child(child);
        }

        Dom::new(NodeType::Div).with_id("body").with_child(
            Dom::new(NodeType::Div).with_id("layout-editor").with_child(
                Dom::new(NodeType::Div)
                    .with_id("component-list-editor")
                    .with_child(buttons)
                    .with_child(Dom::new(NodeType::Div).with_class("spacing-column"))
                    .with_child(component_list)
                    .with_child(Dom::new(NodeType::Div).with_class("spacing-column"))
                    .with_child(settings_list),
            ),
        )
    }
}

fn select_component(
    app_state: &mut AppState<LayoutEditorWindow>,
    event: WindowEvent<LayoutEditorWindow>,
) -> UpdateScreen {
    if let Some((index, _)) = event.get_index_in_parent(event.hit_dom_node) {
        let mut state = app_state.data.lock().unwrap();
        state.state = {
            let mut editor = state.editor.lock().unwrap();
            editor.select(index);
            editor.state()
        };
        UpdateScreen::Redraw
    } else {
        UpdateScreen::DontRedraw
    }
}

fn add_chosen_component(
    app_state: &mut AppState<LayoutEditorWindow>,
    event: WindowEvent<LayoutEditorWindow>,
) -> UpdateScreen {
    if let Some((index, _)) = event.get_index_in_parent(event.hit_dom_node) {
        let mut state = app_state.data.lock().unwrap();
        let component: layout::Component = match index {
            0 => current_comparison::Component::new().into(),
            1 => current_pace::Component::new().into(),
            2 => delta::Component::new().into(),
            3 => Box::new(detailed_timer::Component::new()).into(),
            4 => graph::Component::new().into(),
            5 => possible_time_save::Component::new().into(),
            6 => previous_segment::Component::new().into(),
            7 => splits::Component::new().into(),
            8 => sum_of_best::Component::new().into(),
            9 => text::Component::new().into(),
            10 => timer::Component::new().into(),
            11 => title::Component::new().into(),
            12 => total_playtime::Component::new().into(),
            13 => blank_space::Component::new().into(),
            14 => separator::Component::new().into(),
            _ => unreachable!(),
        };
        state.state = {
            let mut editor = state.editor.lock().unwrap();
            editor.add_component(component);
            editor.state()
        };
        state.add_menu_open = false;
        UpdateScreen::Redraw
    } else {
        UpdateScreen::DontRedraw
    }
}

fn add_component(
    app_state: &mut AppState<LayoutEditorWindow>,
    _event: WindowEvent<LayoutEditorWindow>,
) -> UpdateScreen {
    let mut state = app_state.data.lock().unwrap();
    state.add_menu_open ^= true;
    UpdateScreen::Redraw
}

fn remove_component(
    app_state: &mut AppState<LayoutEditorWindow>,
    _event: WindowEvent<LayoutEditorWindow>,
) -> UpdateScreen {
    let mut state = app_state.data.lock().unwrap();
    state.state = {
        let mut editor = state.editor.lock().unwrap();
        editor.remove_component();
        editor.state()
    };
    UpdateScreen::Redraw
}

fn duplicate_component(
    app_state: &mut AppState<LayoutEditorWindow>,
    _event: WindowEvent<LayoutEditorWindow>,
) -> UpdateScreen {
    let mut state = app_state.data.lock().unwrap();
    state.state = {
        let mut editor = state.editor.lock().unwrap();
        editor.duplicate_component();
        editor.state()
    };
    UpdateScreen::Redraw
}

fn move_component_up(
    app_state: &mut AppState<LayoutEditorWindow>,
    _event: WindowEvent<LayoutEditorWindow>,
) -> UpdateScreen {
    let mut state = app_state.data.lock().unwrap();
    state.state = {
        let mut editor = state.editor.lock().unwrap();
        editor.move_component_up();
        editor.state()
    };
    UpdateScreen::Redraw
}

fn move_component_down(
    app_state: &mut AppState<LayoutEditorWindow>,
    _event: WindowEvent<LayoutEditorWindow>,
) -> UpdateScreen {
    let mut state = app_state.data.lock().unwrap();
    state.state = {
        let mut editor = state.editor.lock().unwrap();
        editor.move_component_down();
        editor.state()
    };
    UpdateScreen::Redraw
}

pub fn open_window(editor: Arc<Mutex<layout::Editor>>) {
    // We use a barrier to wait on the thread's window creation. Otherwise winit
    // hijacks the keyboard in some weird way.
    let barrier = Arc::new(Barrier::new(2));
    thread::spawn({
        let barrier = barrier.clone();
        move || {
            let state = editor.lock().unwrap().state();
            let mut app = App::new(
                LayoutEditorWindow {
                    editor,
                    state,
                    add_menu_open: false,
                },
                AppConfig::default(),
            );

            app.add_font(
                FontId::ExternalFont("font-awesome".into()),
                &mut &include_bytes!("font-awesome.ttf")[..],
            ).unwrap();

            app.add_font(
                FontId::ExternalFont("FiraSans".into()),
                &mut &*livesplit_rendering::TEXT_FONT,
            ).unwrap();

            macro_rules! CSS_PATH {
                () => {
                    concat!(env!("CARGO_MANIFEST_DIR"), "/src/style.css")
                };
            }

            #[cfg(debug_assertions)]
            let css = Css::hot_reload_override_native(CSS_PATH!()).unwrap();
            #[cfg(not(debug_assertions))]
            let css = Css::override_native(include_str!(CSS_PATH!())).unwrap();

            let mut window_options = WindowCreateOptions::default();
            window_options.state.title = String::from("Layout Editor");
            window_options.state.size.dimensions.width = 650.0;
            window_options.state.size.dimensions.height = 450.0;

            let window = Window::new(window_options, css).unwrap();
            barrier.wait();
            app.run(window).unwrap();
        }
    });
    barrier.wait();
}
