#[cfg(feature = "dx11")]
extern crate gfx_backend_dx11 as back;
#[cfg(feature = "dx12")]
extern crate gfx_backend_dx12 as back;
#[cfg(not(any(
    feature = "vulkan",
    feature = "dx11",
    feature = "dx12",
    feature = "metal",
    feature = "gl",
)))]
extern crate gfx_backend_empty as back;
#[cfg(feature = "gl")]
extern crate gfx_backend_gl as back;
#[cfg(feature = "metal")]
extern crate gfx_backend_metal as back;
#[cfg(feature = "vulkan")]
extern crate gfx_backend_vulkan as back;


mod renderer;
mod entities;
mod pipelines;
mod cameras;
mod fps_calculator;

use fps_calculator::FPScalculator;
use renderer::Renderer;
use crate::cameras::Camera;

use std::thread;
use std::sync::{Arc, Mutex};
use std::time;

use gfx_hal as hal;
use hal::{
    prelude::*,
    window,
};


#[cfg_attr(rustfmt, rustfmt_skip)]
pub const DIMS: window::Extent2D = window::Extent2D { width: 960, height: 540 };
pub const TITLE: &str = "Cube";

fn main() {
    env_logger::init();

    #[cfg(not(any(
        feature = "vulkan",
        feature = "dx11",
        feature = "dx12",
        feature = "metal",
        feature = "gl",
    )))]
    eprintln!(
        "You are running the example with the empty backend, no graphical output is to be expected"
    );

    let event_loop = winit::event_loop::EventLoop::new();

    let window_builder = winit::window::WindowBuilder::new()
        .with_min_inner_size(winit::dpi::Size::Logical(winit::dpi::LogicalSize::new(
            64.0, 64.0,
        )))
        .with_inner_size(winit::dpi::Size::Physical(winit::dpi::PhysicalSize::new(
            DIMS.width,
            DIMS.height,
        )))
        .with_title(TITLE.to_string());

    // instantiate backend
    let window = window_builder.build(&event_loop).unwrap();

    let instance = back::Instance::create(TITLE, 1)
        .expect("Failed to create an instance!");

    let surface = unsafe {
        instance
            .create_surface(&window)
            .expect("Failed to create a surface!")
    };

    let adapter = {
        let mut adapters = instance.enumerate_adapters();
        for adapter in &adapters {
            println!("{:?}", adapter.info);
        }
        adapters.remove(0)
    };

    let mut renderer = Renderer::new(instance, surface, adapter);

    renderer.render();

    // Grab and hide the cursor
    window.set_cursor_visible(false);
    window.set_cursor_grab(true)
        .expect("Failed to grab the cursor.");

    let fps_calculator = Arc::new(Mutex::new(FPScalculator::new()));
    let fps_calculator_clone = Arc::clone(&fps_calculator);
    thread::spawn(move || {
        loop {
            let fps = fps_calculator_clone.lock().unwrap().fps();
            println!("fps: {}", fps);
    
            let one_second = time::Duration::from_secs(2);
            thread::sleep(one_second);
        }
    });

    // It is important that the closure move captures the Renderer,
    // otherwise it will not be dropped when the event loop exits.
    event_loop.run(move |event, _, control_flow| {
        *control_flow = winit::event_loop::ControlFlow::Poll;

        renderer.camera().event_handler(&event);
        match event {
            winit::event::Event::WindowEvent { event, .. } => {
                match event {

                    winit::event::WindowEvent::CloseRequested => {
                        *control_flow = winit::event_loop::ControlFlow::Exit
                    }

                    winit::event::WindowEvent::KeyboardInput {
                        input:
                            winit::event::KeyboardInput {
                                virtual_keycode: Some(winit::event::VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => *control_flow = winit::event_loop::ControlFlow::Exit,

                    winit::event::WindowEvent::Resized(dims) => {
                        println!("resized to {:?}", dims);
                        renderer.dimensions_set(window::Extent2D {
                            width: dims.width,
                            height: dims.height,
                        });
                        renderer.recreate_swapchain();
                    }

                    _ => {}
                }
            },
            winit::event::Event::RedrawEventsCleared => {
                renderer.render();
                fps_calculator.lock().unwrap().count_one_frame();
            }
            _ => {}
        }
    });
}
