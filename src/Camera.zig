const c = @import("c.zig");

const math = @import("math.zig");

const Camera = @This();

// TODO: Maybe don't use quaternions?
// Quaternions are useful if we're getting absolute data:
// https://www.reddit.com/r/opengl/comments/im0tex/comment/g3wc05r
// Need to track pitch separately anyway to be able to clamp it.
position: c.vec3,
orientation: c.versor align(32),
view: c.mat4 align(32),

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

pub fn init(position: c.vec3, target: c.vec3) Camera {
    var mut_position = position;
    var mut_target = target;

    var camera: Camera = undefined;
    c.glm_vec3_copy(&mut_position, &camera.position);

    var mut_world_up = math.world_up;
    c.glm_quat_forp(&mut_position, &mut_target, &mut_world_up, &camera.orientation);

    constructViewTransform(camera.position, camera.orientation, &camera.view);
    return camera;
}

pub fn translate(self: *Camera, move_direction: MoveDirection) void {
    var mut_world_forward = math.world_forward;
    // TODO: I don't know why I need to negate this, but not doing so causes
    // forward/back, left/right to be flipped.
    c.glm_vec3_negate(&mut_world_forward);
    var forward: c.vec3 = undefined;
    c.glmc_quat_rotatev(&self.orientation, &mut_world_forward, &forward);

    var changed = false;

    const move_speed = 0.01;
    if (move_direction.forward) {
        changed = true;
        c.glm_vec3_muladds(&forward, move_speed, &self.position);
    } else if (move_direction.backward) {
        changed = true;
        c.glm_vec3_muladds(&forward, -move_speed, &self.position);
    }

    var mut_world_up = math.world_up;
    var right: c.vec3 = undefined;
    c.glm_vec3_cross(&mut_world_up, &forward, &right);
    c.glm_vec3_normalize(&right);
    if (move_direction.left) {
        changed = true;
        c.glm_vec3_muladds(&right, -move_speed, &self.position);
    } else if (move_direction.right) {
        changed = true;
        c.glm_vec3_muladds(&right, move_speed, &self.position);
    }

    if (move_direction.up) {
        changed = true;
        c.glm_vec3_muladds(&mut_world_up, move_speed, &self.position);
    } else if (move_direction.down) {
        changed = true;
        c.glm_vec3_muladds(&mut_world_up, -move_speed, &self.position);
    }

    if (changed) {
        constructViewTransform(self.position, self.orientation, &self.view);
    }
}

pub fn updateOrientation(self: *Camera, offset: c.vec2) void {
    var mut_offset = offset;
    var mut_x_axis: c.vec3 = .{ 1.0, 0.0, 0.0 };
    var mut_world_up = math.world_up;

    var scale: c.vec2 = undefined;
    c.glm_vec2_fill(&scale, 0.001);
    var delta: c.vec2 = undefined;
    c.glm_vec2_mul(&scale, &mut_offset, &delta);

    var pitch: c.versor align(32) = undefined;
    c.glm_quatv(&pitch, -delta[1], &mut_x_axis);
    var yaw: c.versor align(32) = undefined;
    c.glm_quatv(&yaw, delta[0], &mut_world_up);

    c.glmc_quat_mul(&yaw, &self.orientation, &self.orientation);
    c.glmc_quat_mul(&self.orientation, &pitch, &self.orientation);
    c.glmc_quat_normalize(&self.orientation);

    constructViewTransform(self.position, self.orientation, &self.view);
}

fn constructViewTransform(
    position: c.vec3,
    orientation: c.versor,
    destination: *align(32) c.mat4,
) void {
    var mut_world_up = math.world_up;
    var mut_world_forward = math.world_forward;
    var mut_position = position;
    var mut_orientation: c.versor align(32) = orientation;

    var forward: c.vec3 = undefined;
    c.glmc_quat_rotatev(&mut_orientation, &mut_world_forward, &forward);

    var target: c.vec3 = undefined;
    c.glm_vec3_add(&mut_position, &forward, &target);
    c.glmc_lookat(&mut_position, &target, &mut_world_up, destination);
}
