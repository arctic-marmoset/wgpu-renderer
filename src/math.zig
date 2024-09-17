const std = @import("std");

const c = @import("c.zig");

pub const Extent2D = struct {
    width: u32,
    height: u32,

    pub fn aspectRatio(self: Extent2D) f32 {
        return @as(f32, @floatFromInt(self.width)) / @as(f32, @floatFromInt(self.height));
    }
};

pub const CoordinateSystem = struct {
    right: Axis,
    up: Axis,
    forward: Axis,

    const Axis = struct {
        name: Name,
        sign: Sign,

        pub fn vector(self: Axis) Vec3 {
            var vec = vec3Zero();
            vec[@intFromEnum(self.name)] = @intFromEnum(self.sign);
            return vec;
        }

        const Name = enum(comptime_int) { x, y, z };
        const Sign = enum(comptime_int) { positive = 1, negative = -1 };
    };

    pub const blender: CoordinateSystem = .{
        // zig fmt: off
        .right   = .{ .name = .x, .sign = .positive },
        .up      = .{ .name = .z, .sign = .positive },
        .forward = .{ .name = .y, .sign = .positive },
        // zig fmt: on
    };

    pub const vulkan: CoordinateSystem = .{
        // zig fmt: off
        .right   = .{ .name = .x, .sign = .positive },
        .up      = .{ .name = .y, .sign = .negative },
        .forward = .{ .name = .z, .sign = .positive },
        // zig fmt: on
    };

    pub const glTF: CoordinateSystem = .{
        // zig fmt: off
        .right   = .{ .name = .x, .sign = .negative },
        .up      = .{ .name = .y, .sign = .positive },
        .forward = .{ .name = .z, .sign = .positive },
        // zig fmt: on
    };

    pub fn transform(source: CoordinateSystem, target: CoordinateSystem) Mat4 {
        var mat = mat4Zero();
        mat[@intFromEnum(target.forward.name)][@intFromEnum(source.forward.name)] =
            @intFromEnum(source.forward.sign) * @intFromEnum(target.forward.sign);
        mat[@intFromEnum(target.up.name)][@intFromEnum(source.up.name)] =
            @intFromEnum(source.up.sign) * @intFromEnum(target.up.sign);
        mat[@intFromEnum(target.right.name)][@intFromEnum(source.right.name)] =
            @intFromEnum(source.right.sign) * @intFromEnum(target.right.sign);
        mat[3][3] = 1.0;
        return mat;
    }
};

pub const Vec2 = @Vector(2, f32);
pub const Vec3 = @Vector(3, f32);
pub const Vec4 = @Vector(4, f32);

pub const Vec2i = @Vector(2, i32);
pub const Vec2u = @Vector(2, u32);

pub const Mat4 = [4]Vec4;
pub const Mat4x3 = [3]Vec4;

pub fn vec2Zero() Vec2 {
    return .{ 0.0, 0.0 };
}

pub fn vec3Zero() Vec3 {
    return .{ 0.0, 0.0, 0.0 };
}

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
        return vec3Zero();
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

// TODO: Maybe compute this generically for any coordinate system?
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

pub fn vec2iZero() Vec2i {
    return .{ 0, 0 };
}

pub fn vec2uZero() Vec2u {
    return .{ 0, 0 };
}

pub fn mat4Zero() Mat4 {
    return .{
        .{ 0.0, 0.0, 0.0, 0.0 },
        .{ 0.0, 0.0, 0.0, 0.0 },
        .{ 0.0, 0.0, 0.0, 0.0 },
        .{ 0.0, 0.0, 0.0, 0.0 },
    };
}

pub fn mat4Identity() Mat4 {
    return .{
        .{ 1.0, 0.0, 0.0, 0.0 },
        .{ 0.0, 1.0, 0.0, 0.0 },
        .{ 0.0, 0.0, 1.0, 0.0 },
        .{ 0.0, 0.0, 0.0, 1.0 },
    };
}

pub fn mat4x3Identity() Mat4x3 {
    return .{
        .{ 1.0, 0.0, 0.0, 0.0 },
        .{ 0.0, 1.0, 0.0, 0.0 },
        .{ 0.0, 0.0, 1.0, 0.0 },
    };
}

pub fn mat4Mul(a: Mat4, b: Mat4) Mat4 {
    var product: Mat4 = undefined;

    inline for (0..4) |row| {
        var vx: Vec4 = @splat(a[row][0]);
        var vy: Vec4 = @splat(a[row][1]);
        var vz: Vec4 = @splat(a[row][2]);
        var vw: Vec4 = @splat(a[row][3]);

        vx = vx * b[0];
        vy = vy * b[1];
        vz = vz * b[2];
        vw = vw * b[3];
        vx = vx + vz;
        vy = vy + vw;
        vx = vx + vy;
        product[row] = vx;
    }

    return product;
}

pub fn mat4Inverse(mat: Mat4) Mat4 {
    var mut_mat = mat;
    var mat_inverse: Mat4 = undefined;
    c.glmc_mat4_inv(&mut_mat, &mat_inverse);
    return mat_inverse;
}

pub fn mat4Transpose(mat: Mat4) Mat4 {
    var mut_mat = mat;
    c.glmc_mat4_transpose(&mut_mat);
    return mut_mat;
}

pub fn translate(mat: Mat4, vec: Vec3) Mat4 {
    var mut_translation: c.vec3 = vec;
    var translated = mat;
    c.glmc_translate(&translated, &mut_translation);
    return translated;
}

pub fn rotateQuat(mat: Mat4, quat: Vec4) Mat4 {
    var mut_mat = mat;
    var mut_quat: c.versor align(32) = quat;
    c.glmc_quat_rotate(&mut_mat, &mut_quat, &mut_mat);
    return mut_mat;
}

pub fn rotateAxis(mat: Mat4, angle: f32, axis: Vec3) Mat4 {
    var mut_mat = mat;
    var mut_axis: c.vec3 = axis;
    c.glmc_rotate(&mut_mat, angle, &mut_axis);
    return mut_mat;
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

pub fn scaleUniform(mat: Mat4, factor: f32) Mat4 {
    const factor_vec3: Vec3 = @splat(factor);
    const scaled = scale(mat, factor_vec3);
    return scaled;
}

pub fn mat4x3Truncating(mat: Mat4) Mat4x3 {
    return mat[0..3].*;
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
