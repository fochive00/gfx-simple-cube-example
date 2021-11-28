
use crate::entities::{self, Vertex};
use crate::cameras::{self, Camera};
use crate::pipelines::Pipeline;

// use std::cell::RefCell;
use std::rc::Rc;
use gfx_hal as hal;

use hal::{
    buffer, command, format,
    format::ChannelType,
    image, memory,
    pool,
    prelude::*,
    pso,
    pso::ShaderStageFlags,
    queue::QueueGroup,
    window,
};

use std::{
    borrow::Borrow,
    iter,
    mem::{self, ManuallyDrop},
    ptr,
};

use super::DIMS;

pub struct Renderer<B: hal::Backend> {
    desc_pool: ManuallyDrop<B::DescriptorPool>,
    surface: ManuallyDrop<B::Surface>,
    format: hal::format::Format,
    dimensions: window::Extent2D,
    viewport: pso::Viewport,
    framebuffer: ManuallyDrop<B::Framebuffer>,
    pipeline: Pipeline<B>,
    desc_set: Option<B::DescriptorSet>,
    set_layout: ManuallyDrop<B::DescriptorSetLayout>,
    submission_complete_semaphores: Vec<B::Semaphore>,
    submission_complete_fences: Vec<B::Fence>,
    cmd_pools: Vec<B::CommandPool>,
    cmd_buffers: Vec<B::CommandBuffer>,
    cube: entities::Cube,
    camera: cameras::CameraProj3,
    vertex_buffer: ManuallyDrop<B::Buffer>,
    index_buffer: ManuallyDrop<B::Buffer>,
    vertex_buffer_memory: ManuallyDrop<B::Memory>,
    index_buffer_memory: ManuallyDrop<B::Memory>,
    frames_in_flight: usize,
    frame: u64,
    // These members are dropped in the declaration order.
    device: Rc<B::Device>,
    adapter: hal::adapter::Adapter<B>,
    queue_group: QueueGroup<B>,
    instance: B::Instance,
}

impl<B> Renderer<B>
where
    B: hal::Backend,
{
    pub fn new(
        instance: B::Instance,
        mut surface: B::Surface,
        adapter: hal::adapter::Adapter<B>,
    ) -> Renderer<B> {
        let memory_types = adapter.physical_device.memory_properties().memory_types;
        // let limits = adapter.physical_device.properties().limits;

        // Build a new device and associated command queues
        let family = adapter
            .queue_families
            .iter().find(|family| {
                surface.supports_queue_family(family) && family.queue_type().supports_graphics()
            })
            .expect("No queue family supports presentation");

        let physical_device = &adapter.physical_device;

        let mut gpu = unsafe {
            physical_device
                .open(
                    &[(family, &[1.0])],
                    hal::Features::empty()
                )
                .unwrap()
        };

        let queue_group = gpu.queue_groups.pop().unwrap();
        
        // Logical device
        let device = Rc::new(gpu.device);

        let command_pool = unsafe {
            (*device).create_command_pool(queue_group.family, pool::CommandPoolCreateFlags::empty())
        }
        .expect("Can't create command pool");

        // Setup renderpass and pipeline
        let set_layout = ManuallyDrop::new(
            unsafe {
                (*device).create_descriptor_set_layout(
                    vec![]
                    .into_iter(),
                    iter::empty(),
                )
            }
            .expect("Can't create descriptor set layout"),
        );

        // Descriptors
        let mut desc_pool = ManuallyDrop::new(
            unsafe {
                (*device).create_descriptor_pool(
                    1, // sets
                    vec![
                        pso::DescriptorRangeDesc {
                            ty: pso::DescriptorType::Image {
                                ty: pso::ImageDescriptorType::Sampled {
                                    with_sampler: false,
                                },
                            },
                            count: 1,
                        },
                        pso::DescriptorRangeDesc {
                            ty: pso::DescriptorType::Sampler,
                            count: 1,
                        },
                    ]
                    .into_iter(),
                    pso::DescriptorPoolCreateFlags::empty(),
                )
            }
            .expect("Can't create descriptor pool"),
        );
        let desc_set = unsafe {
            desc_pool.allocate_one(&set_layout)
        }.unwrap();

        // Buffer allocations
        println!("Memory types: {:?}", memory_types);

        let cube = entities::Cube::new();
        let camera = cameras::CameraProj3::new();

        let (vertex_buffer, vertex_buffer_memory) = 
            create_buffer::<B, Vertex>(device.clone(), memory_types.clone(), buffer::Usage::VERTEX, cube.vertices());

        let (index_buffer, index_buffer_memory) = 
            create_buffer::<B, u16>(device.clone(), memory_types.clone(), buffer::Usage::INDEX, cube.indices());

        let caps = surface.capabilities(&adapter.physical_device);
        let formats = surface.supported_formats(&adapter.physical_device);
        println!("formats: {:?}", formats);
        let format = formats.map_or(format::Format::Rgba8Srgb, |formats| {
            formats
                .iter()
                .find(|format| format.base_format().1 == ChannelType::Srgb)
                .map(|format| *format)
                .unwrap_or(formats[0])
        });

        let swap_config = window::SwapchainConfig::from_caps(&caps, format, DIMS);
        let fat = swap_config.framebuffer_attachment();
        println!("{:?}", swap_config);
        let extent = swap_config.extent;
        unsafe {
            surface
                .configure_swapchain(&*device, swap_config)
                .expect("Can't configure swapchain");
        };

        let pipeline = Pipeline::new(device.clone(), format, &*set_layout);

        let framebuffer = ManuallyDrop::new(unsafe {
            (*device)
                .create_framebuffer(
                    pipeline.render_pass(),
                    iter::once(fat),
                    image::Extent {
                        width: DIMS.width,
                        height: DIMS.height,
                        depth: 1,
                    },
                )
                .unwrap()
        });

        // Define maximum number of frames we want to be able to be "in flight" (being computed
        // simultaneously) at once
        let frames_in_flight = 3;

        // The number of the rest of the resources is based on the frames in flight.
        let mut submission_complete_semaphores = Vec::with_capacity(frames_in_flight);
        let mut submission_complete_fences = Vec::with_capacity(frames_in_flight);
        // Note: We don't really need a different command pool per frame in such a simple demo like this,
        // but in a more 'real' application, it's generally seen as optimal to have one command pool per
        // thread per frame. There is a flag that lets a command pool reset individual command buffers
        // which are created from it, but by default the whole pool (and therefore all buffers in it)
        // must be reset at once. Furthermore, it is often the case that resetting a whole pool is actually
        // faster and more efficient for the hardware than resetting individual command buffers, so it's
        // usually best to just make a command pool for each set of buffers which need to be reset at the
        // same time (each frame). In our case, each pool will only have one command buffer created from it,
        // though.
        let mut cmd_pools = Vec::with_capacity(frames_in_flight);
        let mut cmd_buffers = Vec::with_capacity(frames_in_flight);

        cmd_pools.push(command_pool);
        for _ in 1..frames_in_flight {
            unsafe {
                cmd_pools.push(
                    (*device)
                        .create_command_pool(
                            queue_group.family,
                            pool::CommandPoolCreateFlags::empty(),
                        )
                        .expect("Can't create command pool"),
                );
            }
        }

        for i in 0..frames_in_flight {
            submission_complete_semaphores.push(
                (*device)
                    .create_semaphore()
                    .expect("Could not create semaphore"),
            );
            submission_complete_fences
                .push((*device).create_fence(true).expect("Could not create fence"));
            cmd_buffers.push(unsafe { cmd_pools[i].allocate_one(command::Level::Primary) });
        }

        // Rendering setup
        let viewport = pso::Viewport {
            rect: pso::Rect {
                x: 0,
                y: 0,
                w: extent.width as _,
                h: extent.height as _,
            },
            depth: 0.0..1.0,
        };

        Renderer {
            instance,
            device,
            queue_group,
            desc_pool,
            surface: ManuallyDrop::new(surface),
            adapter,
            format,
            dimensions: DIMS,
            viewport,
            framebuffer,
            pipeline,
            desc_set: Some(desc_set),
            set_layout,
            submission_complete_semaphores,
            submission_complete_fences,
            cmd_pools,
            cmd_buffers,
            cube,
            camera,
            vertex_buffer,
            index_buffer,
            vertex_buffer_memory,
            index_buffer_memory,
            frames_in_flight,
            frame: 0,
        }
    }

    pub fn recreate_swapchain(&mut self) {
        let caps = self.surface.capabilities(&self.adapter.physical_device);
        let swap_config = window::SwapchainConfig::from_caps(&caps, self.format, self.dimensions);
        println!("{:?}", swap_config);

        let extent = swap_config.extent.to_extent();
        self.viewport.rect.w = extent.width as _;
        self.viewport.rect.h = extent.height as _;

        unsafe {
            (*self.device).wait_idle().unwrap();
            (*self.device)
                .destroy_framebuffer(ManuallyDrop::into_inner(ptr::read(&self.framebuffer)));
            self.framebuffer = ManuallyDrop::new(
                (*self.device)
                    .create_framebuffer(
                        self.pipeline.render_pass(),
                        iter::once(swap_config.framebuffer_attachment()),
                        extent,
                    )
                    .unwrap(),
            )
        };

        unsafe {
            self.surface
                .configure_swapchain(&*self.device, swap_config)
                .expect("Can't create swapchain");
        }
    }

    pub fn render(&mut self) {
        let surface_image = unsafe {
            match self.surface.acquire_image(!0) {
                Ok((image, _)) => image,
                Err(_) => {
                    self.recreate_swapchain();
                    return;
                }
            }
        };

        // Compute index into our resource ring buffers based on the frame number
        // and number of frames in flight. Pay close attention to where this index is needed
        // versus when the swapchain image index we got from acquire_image is needed.
        let frame_idx = self.frame as usize % self.frames_in_flight;

        // Wait for the fence of the previous submission of this frame and reset it; ensures we are
        // submitting only up to maximum number of frames_in_flight if we are submitting faster than
        // the gpu can keep up with. This would also guarantee that any resources which need to be
        // updated with a CPU->GPU data copy are not in use by the GPU, so we can perform those updates.
        // In this case there are none to be done, however.
        unsafe {
            let fence = &mut self.submission_complete_fences[frame_idx];
            (*self.device)
                .wait_for_fence(fence, !0)
                .expect("Failed to wait for fence");
            (*self.device)
                .reset_fence(fence)
                .expect("Failed to reset fence");
            self.cmd_pools[frame_idx].reset(false);
        }

        self.cube.update();
        self.camera().update_view();

        // Rendering
        let cmd_buffer = &mut self.cmd_buffers[frame_idx];
        unsafe {
            cmd_buffer.begin_primary(command::CommandBufferFlags::ONE_TIME_SUBMIT);

            cmd_buffer.set_viewports(0, iter::once(self.viewport.clone()));
            cmd_buffer.set_scissors(0, iter::once(self.viewport.rect));
            cmd_buffer.bind_graphics_pipeline(self.pipeline.pipeline());

            cmd_buffer.bind_vertex_buffers(
                0,
                iter::once((&*self.vertex_buffer, buffer::SubRange::WHOLE)),
            );
            
            cmd_buffer.bind_index_buffer(
                &*self.index_buffer,
                buffer::SubRange::WHOLE,
                hal::IndexType::U16,
            );

            cmd_buffer.bind_graphics_descriptor_sets(
                self.pipeline.pipeline_layout(),
                0,
                self.desc_set.as_ref().into_iter(),
                iter::empty(),
            );

            cmd_buffer.begin_render_pass(
                &self.pipeline.render_pass(),
                &self.framebuffer,
                self.viewport.rect,
                iter::once(command::RenderAttachmentInfo {
                    image_view: surface_image.borrow(),
                    clear_value: command::ClearValue {
                        color: command::ClearColor {
                            float32: [0.0, 0.0, 0.0, 1.0],
                        },
                    },
                }),
                command::SubpassContents::Inline,
            );

            cmd_buffer.push_graphics_constants(
                self.pipeline.pipeline_layout(),
                ShaderStageFlags::VERTEX,
                0,
                push_constant_bytes(&(self.camera.transform() * self.cube.transform())),
            );

            // let vertex_count = self.cube.vertices().len() as u32;
            cmd_buffer.draw_indexed(0..(self.cube.indices().len() as u32), 0, 0..(self.cube.indices().len() / 3 ) as u32);
            // cmd_buffer.draw(0..vertex_count, 0..1);
            cmd_buffer.end_render_pass();
            cmd_buffer.finish();

            self.queue_group.queues[0].submit(
                iter::once(&*cmd_buffer),
                iter::empty(),
                iter::once(&self.submission_complete_semaphores[frame_idx]),
                Some(&mut self.submission_complete_fences[frame_idx]),
            );

            // present frame
            let result = self.queue_group.queues[0].present(
                &mut self.surface,
                surface_image,
                Some(&mut self.submission_complete_semaphores[frame_idx]),
            );

            if result.is_err() {
                self.recreate_swapchain();
            }
        }

        // Increment our frame
        self.frame += 1;
    }

    pub fn dimensions_set(&mut self, dimensions: window::Extent2D) {
        self.dimensions = dimensions;
    }

    pub fn camera(&mut self) -> &mut cameras::CameraProj3 {
        &mut self.camera
    }
}

impl<B> Drop for Renderer<B>
where
    B: hal::Backend,
{
    fn drop(&mut self) {
        (*self.device).wait_idle().unwrap();
        unsafe {
            // TODO: When ManuallyDrop::take (soon to be renamed to ManuallyDrop::read) is stabilized we should use that instead.
            let _ = self.desc_set.take();
            (*self.device).destroy_descriptor_pool(ManuallyDrop::into_inner(ptr::read(&self.desc_pool)));
            (*self.device).destroy_descriptor_set_layout(ManuallyDrop::into_inner(ptr::read(&self.set_layout)));

            (*self.device).destroy_buffer(ManuallyDrop::into_inner(ptr::read(&self.vertex_buffer)));
            (*self.device).destroy_buffer(ManuallyDrop::into_inner(ptr::read(&self.index_buffer)));
            
            for p in self.cmd_pools.drain(..) {
                (*self.device).destroy_command_pool(p);
            }

            for s in self.submission_complete_semaphores.drain(..) {
                (*self.device).destroy_semaphore(s);
            }

            for f in self.submission_complete_fences.drain(..) {
                (*self.device).destroy_fence(f);
            }

            (*self.device).destroy_framebuffer(ManuallyDrop::into_inner(ptr::read(&self.framebuffer)));

            self.surface.unconfigure_swapchain(&*self.device);

            (*self.device).free_memory(ManuallyDrop::into_inner(ptr::read(&self.vertex_buffer_memory)));
            (*self.device).free_memory(ManuallyDrop::into_inner(ptr::read(&self.index_buffer_memory)));

            self.instance.destroy_surface(ManuallyDrop::into_inner(ptr::read(&self.surface)));
        }
        println!("DROPPED!");
    }
}

unsafe fn push_constant_bytes<T>(push_constants: &T) -> &[u32] {
    let size_in_bytes = std::mem::size_of::<T>();
    let size_in_u32s = size_in_bytes / std::mem::size_of::<u32>();
    let start_ptr = push_constants as *const T as *const u32;
    std::slice::from_raw_parts(start_ptr, size_in_u32s)
}

fn create_buffer<B: hal::Backend, T> (
    device: Rc<B::Device>,
    memory_types: Vec<hal::adapter::MemoryType>,
    usage: buffer::Usage,
    src: Vec<T>
) -> (ManuallyDrop<B::Buffer>, ManuallyDrop<B::Memory>) {

    let buffer_stride = mem::size_of::<T>() as u64;
    let buffer_len = src.len() as u64 * buffer_stride;
    assert_ne!(buffer_len, 0);
    // let non_coherent_alignment = limits.non_coherent_atom_size as u64;
    // let padded_buffer_len = ((buffer_len + non_coherent_alignment - 1)
    //     / non_coherent_alignment)
    //     * non_coherent_alignment;
    
    let mut buffer = ManuallyDrop::new(
        unsafe {
            (*device).create_buffer(
                buffer_len,
                usage,
                memory::SparseFlags::empty(),
            )
        }
        .unwrap(),
    );

    let buffer_req = unsafe {
        (*device).get_buffer_requirements(&buffer)
    };

    let upload_type = memory_types
        .iter()
        .enumerate()
        .position(|(id, mem_type)| {
            // type_mask is a bit field where each bit represents a memory type. If the bit is set
            // to 1 it means we can use that type for our buffer. So this code finds the first
            // memory type that has a `1` (or, is allowed), and is visible to the CPU.
            buffer_req.type_mask & (1 << id) != 0
                && mem_type.properties.contains(memory::Properties::CPU_VISIBLE)
        })
        .unwrap()
        .into();

    let buffer_memory = unsafe {
        let mut buffer_memory = (*device)
            .allocate_memory(upload_type, buffer_req.size)
            .unwrap();
        (*device)
            .bind_buffer_memory(&buffer_memory, 0, &mut buffer)
            .unwrap();
    
        let mapping = (*device).map_memory(&mut buffer_memory, memory::Segment::ALL).unwrap();

        ptr::copy_nonoverlapping(src[..].as_ptr() as *const u8, mapping, buffer_len as usize);
        (*device)
            .flush_mapped_memory_ranges(iter::once((&buffer_memory, memory::Segment::ALL)))
            .unwrap();
        (*device).unmap_memory(&mut buffer_memory);
        ManuallyDrop::new(buffer_memory)
    };

    (buffer, buffer_memory)
}
