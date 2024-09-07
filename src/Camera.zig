const std = @import("std");

const c = @import("c.zig");
const math = @import("math.zig");

const Camera = @This();

position: math.Vec3,
pitch: f32,
yaw: f32,

const Matrices = struct {
    view: math.Mat4,
};

pub const MoveDirection = struct {
    forward: bool = false,
    backward: bool = false,
    left: bool = false,
    right: bool = false,
    up: bool = false,
    down: bool = false,

    pub fn normalize(self: *MoveDirection) void {
        if (self.forward and self.backward) {
            self.forward = false;
            self.backward = false;
        }
        if (self.left and self.right) {
            self.left = false;
            self.right = false;
        }
        if (self.up and self.down) {
            self.up = false;
            self.down = false;
        }
    }
};

pub fn init(position: math.Vec3, target: math.Vec3) Camera {
    const direction: math.Vec3 = math.vec3Normalize(target - position);

    const pitch = std.math.asin(direction[1]);
    const yaw = std.math.atan2(direction[0], direction[2]);

    const camera: Camera = .{
        .position = position,
        .yaw = yaw,
        .pitch = pitch,
    };

    return camera;
}

pub fn translate(self: *Camera, delta_time: f32, move_direction: MoveDirection) void {
    const forward = math.forwardVectorFromEuler(self.pitch, self.yaw);

    var changed = false;

    const move_speed = 2.0 * delta_time;
    const move_speed_vec3: math.Vec3 = @splat(move_speed);
    if (move_direction.forward) {
        changed = true;
        self.position += forward * move_speed_vec3;
    } else if (move_direction.backward) {
        changed = true;
        self.position += forward * -move_speed_vec3;
    }

    const right = math.vec3CrossNormalize(forward, math.world_up);
    if (move_direction.left) {
        changed = true;
        self.position += right * -move_speed_vec3;
    } else if (move_direction.right) {
        changed = true;
        self.position += right * move_speed_vec3;
    }

    if (move_direction.up) {
        changed = true;
        self.position += math.world_up * move_speed_vec3;
    } else if (move_direction.down) {
        changed = true;
        self.position += math.world_up * -move_speed_vec3;
    }
}

pub fn updateOrientation(self: *Camera, offset: math.Vec2) void {
    const sensitivity = 0.005;
    const pitch_limit = 0.5 * std.math.pi - 0.01;

    const delta_yaw = sensitivity * offset[0];
    const delta_pitch = sensitivity * offset[1];

    self.yaw = @mod(self.yaw + delta_yaw, 2.0 * std.math.pi);
    self.pitch = std.math.clamp(self.pitch + delta_pitch, -pitch_limit, pitch_limit);
}

pub fn computeMatrices(self: Camera) Matrices {
    const forward = math.forwardVectorFromEuler(self.pitch, self.yaw);
    const target = self.position + forward;

    const matrices: Matrices = .{
        .view = math.lookAt(self.position, target, math.world_up),
    };

    return matrices;
}
