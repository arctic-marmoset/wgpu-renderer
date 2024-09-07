const std = @import("std");

const c = @import("c.zig");

// Vulkan clip space coordinate system.
pub const world_up: Vec3 = .{ 0.0, -1.0, 0.0 };
pub const world_forward: Vec3 = .{ 0.0, 0.0, 1.0 };
pub const world_right: Vec3 = .{ 1.0, 0.0, 0.0 };

pub const Vec2 = @Vector(2, f32);
pub const Vec3 = @Vector(3, f32);
pub const Vec4 = @Vector(4, f32);

pub const Mat4 = [4]Vec4;

pub fn vec3Dot(a: Vec3, b: Vec3) f32 {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

pub fn vec3Norm2(vec: Vec3) f32 {
    return vec3Dot(vec, vec);
}

pub fn vec3Norm(vec: Vec3) f32 {
    return @sqrt(vec3Norm2(vec));
}

pub fn vec3Scale(vec: Vec3, factor: f32) Vec3 {
    const factor_vec3: Vec3 = @splat(factor);
    const scaled = vec * factor_vec3;
    return scaled;
}

pub fn vec3Normalize(vec: Vec3) Vec3 {
    const norm = vec3Norm(vec);

    if (norm < std.math.floatEps(f32)) {
        @branchHint(.unlikely);
        return std.mem.zeroes(Vec3);
    }

    return vec3Scale(vec, 1.0 / norm);
}

pub fn vec3Cross(a: Vec3, b: Vec3) Vec3 {
    return .{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

pub fn vec3CrossNormalize(a: Vec3, b: Vec3) Vec3 {
    return vec3Normalize(vec3Cross(a, b));
}

pub fn forwardVectorFromEuler(pitch: f32, yaw: f32) Vec3 {
    const forward = vec3Normalize(.{
        @cos(pitch) * @sin(yaw),
        @sin(pitch),
        @cos(pitch) * @cos(yaw),
    });

    return forward;
}

pub fn vec4Scale(vec: Vec4, factor: f32) Vec4 {
    const factor_vec4: Vec4 = @splat(factor);
    const scaled = vec * factor_vec4;
    return scaled;
}

pub fn mat4Identity() Mat4 {
    return .{
        .{ 1.0, 0.0, 0.0, 0.0 },
        .{ 0.0, 1.0, 0.0, 0.0 },
        .{ 0.0, 0.0, 1.0, 0.0 },
        .{ 0.0, 0.0, 0.0, 1.0 },
    };
}

pub fn scale(mat: Mat4, factor: Vec3) Mat4 {
    const scaled: Mat4 = .{
        vec4Scale(mat[0], factor[0]),
        vec4Scale(mat[1], factor[1]),
        vec4Scale(mat[2], factor[2]),
        mat[3],
    };

    return scaled;
}

pub fn lookAt(
    position: Vec3,
    target: Vec3,
    up: Vec3,
) Mat4 {
    const forward = vec3Normalize(target - position);
    const right = vec3CrossNormalize(forward, up);
    const local_up = vec3Cross(right, forward);

    const translation_x = vec3Dot(position, right);
    const translation_y = vec3Dot(position, local_up);
    const translation_z = vec3Dot(position, forward);

    const view: Mat4 = .{
        .{ right[0], local_up[0], forward[0], 0.0 },
        .{ right[1], local_up[1], forward[1], 0.0 },
        .{ right[2], local_up[2], forward[2], 0.0 },
        .{ -translation_x, -translation_y, -translation_z, 1.0 },
    };

    return view;
}

pub fn perspectiveInverseDepth(
    vfov: f32,
    aspect: f32,
    near: f32,
) Mat4 {
    const focal_length = 1.0 / @tan(vfov / 2.0);

    const x = focal_length / aspect;
    const y = focal_length;
    const a = 0.0;
    const b = near;

    const projection: Mat4 = .{
        .{ x, 0.0, 0.0, 0.0 },
        .{ 0.0, y, 0.0, 0.0 },
        .{ 0.0, 0.0, a, 1.0 },
        .{ 0.0, 0.0, b, 0.0 },
    };

    return projection;
}
