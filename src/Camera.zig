const c = @import("c.zig");

const Camera = @This();

position: c.vec3,
orientation: c.versor align(32),
view: c.mat4 align(32),

const world_up: c.vec3 = .{ 0.0, 1.0, 0.0 };
const world_forward: c.vec3 = .{ 0.0, 0.0, 1.0 };

pub fn init(position: c.vec3, target: c.vec3) Camera {
    var mut_position = position;
    var mut_target = target;

    var camera: Camera = undefined;
    c.glm_vec3_copy(&mut_position, &camera.position);

    var forward: c.vec3 = undefined;
    c.glm_vec3_sub(&mut_target, &mut_position, &forward);
    c.glm_vec3_normalize(&forward);

    var mut_world_up = world_up;
    c.glmc_quat_forp(&mut_position, &mut_target, &mut_world_up, &camera.orientation);

    constructViewTransform(camera.position, camera.orientation, &camera.view);
    return camera;
}

pub fn updateOrientation(self: *Camera, offset: c.vec2) void {
    var mut_offset = offset;
    var mut_x_axis: c.vec3 = .{ 1.0, 0.0, 0.0 };
    var mut_world_up = world_up;

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
    var mut_world_up = world_up;
    var mut_world_forward = world_forward;
    var mut_position = position;
    var mut_orientation: c.versor align(32) = orientation;

    var forward: c.vec3 = undefined;
    c.glmc_quat_rotatev(&mut_orientation, &mut_world_forward, &forward);

    var target: c.vec3 = undefined;
    c.glm_vec3_add(&mut_position, &forward, &target);
    c.glmc_lookat(&mut_position, &target, &mut_world_up, destination);
}
