const std = @import("std");

const Engine = @import("Engine.zig");

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer std.debug.assert(gpa.deinit() == .ok);
    // TODO: Maybe use c_allocator in release builds.
    const allocator = gpa.allocator();

    const engine = try Engine.init(allocator);
    defer engine.deinit();

    try engine.run();
}
