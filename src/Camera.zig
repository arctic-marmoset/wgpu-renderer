const std = @import("std");

const c = @import("c.zig");
const math = @import("math.zig");

const Engine = @import("Engine.zig");

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

    pub fn normalized(direction: MoveDirection) MoveDirection {
        var mut_direction = direction;
        mut_direction.normalize();
        return mut_direction;
    }

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

pub const InitOptions = struct {
    position: math.Vec3,
    target: math.Vec3,
};

pub fn init(options: InitOptions) Camera {
    const direction: math.Vec3 = math.vec3Normalize(options.target - options.position);

    const pitch = std.math.asin(direction[1]);
    const yaw = std.math.atan2(direction[0], direction[2]);

    const camera: Camera = .{
        .position = options.position,
        .yaw = yaw,
        .pitch = pitch,
    };

    return camera;
}

pub fn translate(self: *Camera, delta_time: f32, move_direction: MoveDirection) void {
    const forward = math.forwardVectorFromEuler(self.pitch, self.yaw);

    var changed = false;

    const sensitivity = 2.0;
    const move_amount = delta_time * sensitivity;
    const move_speed_vec3: math.Vec3 = @splat(move_amount);
    if (move_direction.forward) {
        changed = true;
        self.position += forward * move_speed_vec3;
    } else if (move_direction.backward) {
        changed = true;
        self.position += forward * -move_speed_vec3;
    }

    const right = math.vec3CrossNormalize(forward, Engine.world_space.up.vector());
    if (move_direction.left) {
        changed = true;
        self.position += right * -move_speed_vec3;
    } else if (move_direction.right) {
        changed = true;
        self.position += right * move_speed_vec3;
    }

    if (move_direction.up) {
        changed = true;
        self.position += Engine.world_space.up.vector() * move_speed_vec3;
    } else if (move_direction.down) {
        changed = true;
        self.position += Engine.world_space.up.vector() * -move_speed_vec3;
    }
}

pub fn updateOrientation(self: *Camera, delta: math.Vec2) void {
    const sensitivity = 0.002;
    const move_amount = sensitivity;
    const pitch_limit = 0.5 * std.math.pi - 0.01;

    const delta_yaw = move_amount * delta[0];
    const delta_pitch = move_amount * delta[1];

    self.yaw = @mod(self.yaw + delta_yaw, 2.0 * std.math.pi);
    self.pitch = std.math.clamp(self.pitch + delta_pitch, -pitch_limit, pitch_limit);
}

pub fn computeMatrices(self: Camera) Matrices {
    const forward = math.forwardVectorFromEuler(self.pitch, self.yaw);
    const target = self.position + forward;

    const matrices: Matrices = .{
        .view = math.lookAt(self.position, target, Engine.world_space.up.vector()),
    };

    return matrices;
}
