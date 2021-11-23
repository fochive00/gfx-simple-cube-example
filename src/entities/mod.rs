mod cube;
pub use cube::Cube;

mod triangle;
pub use triangle::Triangle;


#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct Vertex {
    pos: [f32; 3],
    color: [f32; 3],
}