const c = @import("c.zig");

pub const Extent2D = struct {
    width: u32,
    height: u32,

    pub fn aspectRatio(self: Extent2D) f32 {
        return @as(f32, @floatFromInt(self.width)) / @as(f32, @floatFromInt(self.height));
    }
};

pub fn getFramebufferSize(window: *c.GLFWwindow) Extent2D {
    var width: c_int = 0;
    var height: c_int = 0;
    c.glfwGetFramebufferSize(window, &width, &height);
    return .{ .width = @intCast(width), .height = @intCast(height) };
}
