const c = @import("c.zig");
const math = @import("math.zig");

pub fn getFramebufferSize(window: *c.GLFWwindow) math.Extent2D {
    var width: c_int = 0;
    var height: c_int = 0;
    c.glfwGetFramebufferSize(window, &width, &height);
    return .{ .width = @intCast(width), .height = @intCast(height) };
}
