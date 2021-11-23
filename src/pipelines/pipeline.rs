
use crate::entities::Vertex;

// use std::cell::RefCell;
use std::rc::Rc;

use gfx_hal as hal;
use gfx_auxil as auxil;

use hal::{
    format,
    image, pass,
    pass::Subpass,
    prelude::*,
    pso,
    pso::{
        ShaderStageFlags, VertexInputRate, InputAssemblerDesc,
        Primitive, PrimitiveAssemblerDesc
    },
};

use std::{
    io::Cursor,
    iter,
    mem::ManuallyDrop,
    ptr,
};

const ENTRY_NAME: &str = "main";

pub struct Pipeline<B: hal::Backend> {
    device: Rc<B::Device>,
    render_pass: ManuallyDrop<B::RenderPass>,
    pipeline_layout: ManuallyDrop<B::PipelineLayout>,
    pipeline: ManuallyDrop<B::GraphicsPipeline>,
    pipeline_cache: ManuallyDrop<B::PipelineCache>,
}


impl<B: hal::Backend> Pipeline<B> {
    pub fn new(
        device: Rc<B::Device>,
        format: hal::format::Format,
        set_layout: &B::DescriptorSetLayout,
    ) -> Self {
        let render_pass = create_render_pass::<B>(device.clone(), format);
        let pipeline_layout = create_pipeline_layout::<B>(device.clone(), set_layout);
        let pipeline_cache = load_pipeline_cache::<B>(device.clone());
        let pipeline = create_pipeline::<B>(device.clone(), &*render_pass, &*pipeline_layout, &pipeline_cache);
        save_pipeline_cache::<B>(device.clone(), &pipeline_cache);

        Self {
            device: device,
            render_pass,
            pipeline_layout,
            pipeline,
            pipeline_cache
        }
    }

    pub fn render_pass(&self) -> &B::RenderPass {
        &*self.render_pass
    }

    pub fn pipeline_layout(&self) -> &B::PipelineLayout {
        &*self.pipeline_layout
    }
    
    pub fn pipeline(&self) -> &B::GraphicsPipeline {
        &*self.pipeline
    }
}

impl<B> Drop for Pipeline<B> where B: hal::Backend {
    fn drop(&mut self) {
        (*self.device).wait_idle().unwrap();

        unsafe {
            (*self.device)
                .destroy_pipeline_layout(ManuallyDrop::into_inner(ptr::read(
                    &self.pipeline_layout,
                )));

            (*self.device)
                .destroy_render_pass(ManuallyDrop::into_inner(ptr::read(
                    &self.render_pass,
                )));

            (*self.device)
                .destroy_graphics_pipeline(ManuallyDrop::into_inner(ptr::read(
                    &self.pipeline,
                )));
            
            (*self.device)
                .destroy_pipeline_cache(ManuallyDrop::into_inner(ptr::read(
                    &self.pipeline_cache
                )));
        }

    }
}

fn create_render_pass<B: hal::Backend>(device: Rc<B::Device>, format: hal::format::Format) -> ManuallyDrop<B::RenderPass> {
        let attachment = pass::Attachment {
            format: Some(format),
            samples: 1,
            ops: pass::AttachmentOps::new(
                pass::AttachmentLoadOp::Clear,
                pass::AttachmentStoreOp::Store,
            ),
            stencil_ops: pass::AttachmentOps::DONT_CARE,
            layouts: image::Layout::Undefined..image::Layout::Present,
        };

        let subpass = pass::SubpassDesc {
            colors: &[(0, image::Layout::ColorAttachmentOptimal)],
            depth_stencil: None,
            inputs: &[],
            resolves: &[],
            preserves: &[],
        };

        ManuallyDrop::new(
            unsafe {
                (*device).create_render_pass(
                    iter::once(attachment),
                    iter::once(subpass),
                    iter::empty(),
                )
            }
            .expect("Can't create render pass"),
        )
}

fn create_pipeline_layout<B: hal::Backend>(device: Rc<B::Device>, set_layout: &B::DescriptorSetLayout) -> ManuallyDrop<B::PipelineLayout> {
    let push_constant_bytes = std::mem::size_of::<[[f32; 4]; 4]>() as u32;

    ManuallyDrop::new(
        unsafe {
            (*device).create_pipeline_layout(
                iter::once(set_layout),
                [(ShaderStageFlags::VERTEX, 0..push_constant_bytes)].into_iter()
            )
        }
        .expect("Can't create pipeline layout"),
    )
}

fn create_pipeline<B: hal::Backend>(
        device: Rc<B::Device>,
        render_pass: &B::RenderPass,
        pipeline_layout: &B::PipelineLayout,
        pipeline_cache: &B::PipelineCache
    ) -> ManuallyDrop<B::GraphicsPipeline> {
    
    let vs_module = {
        let spirv =
            auxil::read_spirv(Cursor::new(&include_bytes!("../shaders/vert.spv")[..]))
                .unwrap();
        
        unsafe {
            (*device).create_shader_module(&spirv)
        }.unwrap()
    };

    let fs_module = {
        let spirv =
            auxil::read_spirv(Cursor::new(&include_bytes!("../shaders/frag.spv")[..]))
                .unwrap();
        unsafe {
            (*device).create_shader_module(&spirv)
        }.unwrap()
    };
    
    let (vs_entry, fs_entry) = (
        pso::EntryPoint {
            entry: ENTRY_NAME,
            module: &vs_module,
            specialization: pso::Specialization::default(),
        },
        pso::EntryPoint {
            entry: ENTRY_NAME,
            module: &fs_module,
            specialization: pso::Specialization::default(),
        },
    );

    let primitive_assembler = {
        PrimitiveAssemblerDesc::Vertex {
            // We need to add a new section to our primitive assembler so
            // that it understands how to interpret the vertex data it is
            // given.
            //
            // We start by giving it a `binding` number, which is more or
            // less an ID or a slot for the vertex buffer. You'll see it
            // used later.
            //
            // The `stride` is the size of one item in the buffer.
            // The `rate` defines how to progress through the buffer.
            // Passing `Vertex` to this tells it to advance after every
            // vertex. This is usually what you want to do if you're not
            // making use of instanced rendering.
            buffers: &[pso::VertexBufferDesc {
                binding: 0,
                stride: std::mem::size_of::<Vertex>() as u32,
                rate: VertexInputRate::Vertex,
            }],

            // Then we need to define the attributes _within_ the vertices.
            // For us this is the `position` and the `normal`.
            //
            // The vertex buffer we just defined has a `binding` number of
            // `0`. The `location` refers to the location in the `layout`
            // definition in the vertex shader.
            //
            // Finally the `element` describes the size and position of the
            // attribute. Both of our elements are 3-component 32-bit float
            // vectors, and so the `format` is `Rgb32Sfloat`. (I don't know
            // why it's `Rgb` and not `Xyz` or `Vec3` but here we are.)
            //
            // Note that the second attribute has an offset of `12` bytes,
            // because it has 3 4-byte floats before it (e.g. the previous
            // attribute).
            attributes: &[
                pso::AttributeDesc {
                    location: 0,
                    binding: 0,
                    element: pso::Element {
                        format: format::Format::Rgb32Sfloat,
                        offset: 0,
                    },
                },
                pso::AttributeDesc {
                    location: 1,
                    binding: 0,
                    element: pso::Element {
                        format: format::Format::Rgb32Sfloat,
                        offset: 12,
                    },
                },
            ],
            input_assembler: InputAssemblerDesc::new(Primitive::TriangleList),
            vertex: vs_entry,
            tessellation: None,
            geometry: None,
        }
    };

    let subpass = Subpass {
        index: 0,
        main_pass: render_pass,
    };

    let mut pipeline_desc = pso::GraphicsPipelineDesc::new(
        primitive_assembler,
        pso::Rasterizer {
            cull_face: pso::Face::BACK,
            ..pso::Rasterizer::FILL
        },
        Some(fs_entry),
        pipeline_layout,
        subpass,
    );

    pipeline_desc.blender.targets.push(pso::ColorBlendDesc {
        mask: pso::ColorMask::ALL,
        blend: Some(pso::BlendState::ALPHA),
    });

    let pipeline = unsafe {
        (*device).create_graphics_pipeline(&pipeline_desc, Some(&pipeline_cache))
    };

    unsafe {
        (*device).destroy_shader_module(vs_module);
    }
    unsafe {
        (*device).destroy_shader_module(fs_module);
    }

    ManuallyDrop::new(pipeline.unwrap())
}

fn load_pipeline_cache<B: hal::Backend>(device: Rc<B::Device>) -> ManuallyDrop<B::PipelineCache> {
    let pipeline_cache_path = "cube_pipeline_cache";

    let previous_pipeline_cache_data = std::fs::read(pipeline_cache_path);

    if let Err(error) = previous_pipeline_cache_data.as_ref() {
        println!("Error loading the previous pipeline cache data: {}", error);
    }

    ManuallyDrop::new(unsafe {
        (*device)
            .create_pipeline_cache(
                previous_pipeline_cache_data
                    .as_ref()
                    .ok()
                    .map(|vec| &vec[..]),
            )
            .expect("Can't create pipeline cache")
    })
}

fn save_pipeline_cache<B: hal::Backend>(device: Rc<B::Device>, pipeline_cache: &B::PipelineCache) {
    let pipeline_cache_path = "cube_pipeline_cache";

    let pipeline_cache_data = unsafe {
        (*device).get_pipeline_cache_data(pipeline_cache).unwrap()
    };

    std::fs::write(pipeline_cache_path, &pipeline_cache_data).unwrap();
    log::info!(
        "Wrote the pipeline cache to {} ({} bytes)",
        pipeline_cache_path,
        pipeline_cache_data.len()
    );
}

