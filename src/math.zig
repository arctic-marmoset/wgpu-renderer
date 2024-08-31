const c = @import("c.zig");

// glTF coordinate system.
pub const world_up: c.vec3 = .{ 0.0, 1.0, 0.0 };
pub const world_forward: c.vec3 = .{ 0.0, 0.0, 1.0 };
pub const world_right: c.vec3 = .{ -1.0, 0.0, 0.0 };

pub fn perspectiveInverseDepth(
    vfov: f32,
    aspect: f32,
    near: f32,
    destination: *align(32) c.mat4,
) void {
    const focal_length = 1.0 / @tan(vfov / 2.0);

    const x = focal_length / aspect;
    const y = focal_length;
    const a = 0.0;
    const b = near;

    c.glmc_mat4_zero(destination);
    destination[0][0] = x;
    destination[1][1] = y;
    destination[2][2] = a;
    destination[2][3] = 1.0;
    destination[3][2] = b;
}
