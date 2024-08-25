const c = @import("c.zig");

pub fn getFramebufferSize(window: *c.GLFWwindow) struct {
    width: u32,
    height: u32,
} {
    var width: c_int = 0;
    var height: c_int = 0;
    c.glfwGetFramebufferSize(window, &width, &height);
    return .{ .width = @intCast(width), .height = @intCast(height) };
}
