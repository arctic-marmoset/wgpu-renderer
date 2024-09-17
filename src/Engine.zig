const std = @import("std");
const builtin = @import("builtin");

const c = @import("c.zig");
const glfw = @import("glfw.zig");
const math = @import("math.zig");

const Camera = @import("Camera.zig");
const Renderer = @import("Renderer.zig");

const Engine = @This();

allocator: std.mem.Allocator,

data_dir: std.fs.Dir,

window: *c.GLFWwindow,

renderer: Renderer,
imgui_io: *c.ImGuiIO,
last_instant: std.time.Instant,

camera: Camera,
models: std.ArrayListUnmanaged(Renderer.Model),

mouse_captured: bool,
last_mouse_position: math.Vec2,
mouse_moved: bool,
last_mouse_delta: math.Vec2,

pub const model_space = math.CoordinateSystem.glTF;
pub const world_space = math.CoordinateSystem.vulkan;

// Heap allocating lets us freely pass the pointer to callbacks.
pub fn init(allocator: std.mem.Allocator) !*Engine {
    const engine = try allocator.create(Engine);
    errdefer allocator.destroy(engine);

    var data_dir = try openDataDir(allocator);
    errdefer data_dir.close();

    if (builtin.target.os.tag == .linux) {
        if (c.glfwPlatformSupported(c.GLFW_PLATFORM_WAYLAND) == c.GLFW_TRUE) {
            c.glfwInitHint(c.GLFW_PLATFORM, c.GLFW_PLATFORM_WAYLAND);
        }
    }
    if (c.glfwInit() != c.GLFW_TRUE) return error.GlfwInitFailed;
    errdefer c.glfwTerminate();

    c.glfwWindowHint(c.GLFW_CLIENT_API, c.GLFW_NO_API);
    c.glfwWindowHint(c.GLFW_COCOA_RETINA_FRAMEBUFFER, c.GLFW_TRUE);
    const window = c.glfwCreateWindow(
        1280,
        720,
        "3D Renderer (WGPU)",
        null,
        null,
    ) orelse return error.CreateMainWindowFailed;
    errdefer c.glfwDestroyWindow(window);
    c.glfwSetWindowUserPointer(window, engine);
    _ = c.glfwSetKeyCallback(window, keyActionCallback);
    _ = c.glfwSetMouseButtonCallback(window, mouseButtonActionCallback);
    _ = c.glfwSetCursorPosCallback(window, mousePositionChangedCallback);
    _ = c.glfwSetFramebufferSizeCallback(window, framebufferSizeChangedCallback);
    c.glfwSetWindowSizeLimits(window, 640, 360, c.GLFW_DONT_CARE, c.GLFW_DONT_CARE);
    c.glfwSetInputMode(window, c.GLFW_CURSOR, c.GLFW_CURSOR_DISABLED);
    if (c.glfwRawMouseMotionSupported() == c.GLFW_TRUE) {
        c.glfwSetInputMode(window, c.GLFW_RAW_MOUSE_MOTION, c.GLFW_TRUE);
    }
    const mouse_position: math.Vec2 = blk: {
        var x: f64 = undefined;
        var y: f64 = undefined;
        c.glfwGetCursorPos(window, &x, &y);
        break :blk .{ @floatCast(x), @floatCast(y) };
    };

    const vertex_shader_bytecode = try data_dir.readFileAllocOptions(
        allocator,
        "shaders/basic.vert.spv",
        1024 * 1024,
        null,
        @alignOf(u32),
        null,
    );
    defer allocator.free(vertex_shader_bytecode);
    const fragment_shader_bytecode = try data_dir.readFileAllocOptions(
        allocator,
        "shaders/basic.frag.spv",
        1024 * 1024,
        null,
        @alignOf(u32),
        null,
    );
    defer allocator.free(fragment_shader_bytecode);

    var renderer = try Renderer.init(allocator, .{
        .window = window,
        .vertex_shader_bytecode = vertex_shader_bytecode,
        .fragment_shader_bytecode = fragment_shader_bytecode,
    });
    errdefer renderer.deinit();

    var models = std.ArrayListUnmanaged(Renderer.Model){};
    errdefer models.deinit(allocator);
    const arena = try renderer.loadModel(allocator, data_dir, "meshes/arena.glb", math.mat4Identity());
    try models.append(allocator, arena);
    const dragon = try renderer.loadModel(
        allocator,
        data_dir,
        "meshes/stanford_dragon.glb",
        math.translate(math.mat4Identity(), math.vec3Scale(world_space.up.vector(), -1.0)),
    );
    try models.append(allocator, dragon);
    const crate = try renderer.loadModel(
        allocator,
        data_dir,
        "meshes/crate.glb",
        math.scaleUniform(math.translate(math.mat4Identity(), math.vec3Scale(world_space.up.vector(), -1.4)), 0.4),
    );
    try models.append(allocator, crate);
    const porche = try renderer.loadModel(
        allocator,
        data_dir,
        "meshes/porche.glb",
        math.rotateAxis(
            math.translate(
                math.mat4Identity(),
                math.vec3Scale(world_space.forward.vector(), 2.0) + math.vec3Scale(world_space.up.vector(), -1.95),
            ),
            std.math.degreesToRadians(90.0),
            world_space.up.vector(),
        ),
    );
    try models.append(allocator, porche);

    engine.* = .{
        .allocator = allocator,

        .data_dir = data_dir,

        .window = window,

        .renderer = renderer,
        .imgui_io = c.ImGui_GetIO(),
        .last_instant = std.time.Instant.now() catch unreachable,

        .camera = .init(.{
            .position = math.vec3Scale(world_space.forward.vector(), -2.5),
            .target = world_space.forward.vector(),
        }),
        .models = models,

        .mouse_captured = true,
        .last_mouse_position = mouse_position,
        .mouse_moved = false,
        .last_mouse_delta = math.vec2Zero(),
    };

    return engine;
}

pub fn deinit(self: *Engine) void {
    self.models.deinit(self.allocator);
    self.renderer.deinit();

    c.glfwDestroyWindow(self.window);
    self.data_dir.close();

    self.allocator.destroy(self);
}

pub fn run(self: *Engine) !void {
    while (self.isRunning()) {
        c.glfwPollEvents();
        self.tick();
    }
}

fn tick(self: *Engine) void {
    const now = std.time.Instant.now() catch unreachable;
    const delta_time_ns = now.since(self.last_instant);
    self.last_instant = now;
    const delta_time_ns_f64: f64 = @floatFromInt(delta_time_ns);
    const delta_time_s_f64 = delta_time_ns_f64 / std.time.ns_per_s;
    const delta_time: f32 = @floatCast(delta_time_s_f64);

    self.update(delta_time);
    self.renderer.renderFrame(delta_time, self.camera, self.models.items);
}

fn isRunning(self: *Engine) bool {
    return c.glfwWindowShouldClose(self.window) != c.GLFW_TRUE;
}

fn update(self: *Engine, delta_time: f32) void {
    if (!self.imgui_io.WantCaptureKeyboard) {
        self.camera.translate(delta_time, .normalized(.{
            .forward = c.glfwGetKey(self.window, c.GLFW_KEY_W) == c.GLFW_PRESS,
            .backward = c.glfwGetKey(self.window, c.GLFW_KEY_S) == c.GLFW_PRESS,
            .left = c.glfwGetKey(self.window, c.GLFW_KEY_A) == c.GLFW_PRESS,
            .right = c.glfwGetKey(self.window, c.GLFW_KEY_D) == c.GLFW_PRESS,
            .up = c.glfwGetKey(self.window, c.GLFW_KEY_SPACE) == c.GLFW_PRESS,
            .down = c.glfwGetKey(self.window, c.GLFW_KEY_LEFT_SHIFT) == c.GLFW_PRESS,
        }));
    }

    if (self.mouse_captured and self.mouse_moved) {
        self.mouse_moved = false;
        self.camera.updateOrientation(delta_time, self.last_mouse_delta);
    }
}

fn onFramebufferSizeChanged(self: *Engine, extent: math.Extent2D) void {
    self.renderer.recreateWindowSizedResources(extent);
}

fn onKeyAction(self: *Engine, key: c_int, scancode: c_int, action: c_int, modifiers: c_int) void {
    _ = scancode; // autofix
    _ = action; // autofix
    _ = modifiers; // autofix
    // TODO: This should probably be in `Engine.update`.
    if (key == c.GLFW_KEY_ESCAPE) {
        if (self.mouse_captured) {
            self.mouse_captured = false;
            c.glfwSetInputMode(self.window, c.GLFW_CURSOR, c.GLFW_CURSOR_NORMAL);
        }
    }
}

fn onMouseButtonAction(self: *Engine, button: c_int, action: c_int, modifiers: c_int) void {
    _ = button; // autofix
    _ = action; // autofix
    _ = modifiers; // autofix
    // TODO: This should probably be in `Engine.update`.
    if (!self.imgui_io.WantCaptureMouse) {
        if (!self.mouse_captured) {
            self.mouse_captured = true;
            c.glfwSetInputMode(self.window, c.GLFW_CURSOR, c.GLFW_CURSOR_DISABLED);
        }
    }
}

fn onMousePositionChanged(self: *Engine, x: f64, y: f64) void {
    if (!self.mouse_captured) return;
    self.mouse_moved = true;
    const position: math.Vec2 = .{ @floatCast(x), @floatCast(y) };
    const delta = position - self.last_mouse_position;
    self.last_mouse_position = position;
    self.last_mouse_delta = delta;
}

fn framebufferSizeChangedCallback(
    window: ?*c.GLFWwindow,
    width: c_int,
    height: c_int,
) callconv(.C) void {
    const engine: *Engine = @ptrCast(@alignCast(c.glfwGetWindowUserPointer(window)));
    engine.onFramebufferSizeChanged(.{
        .width = @intCast(width),
        .height = @intCast(height),
    });
}

fn keyActionCallback(
    window: ?*c.GLFWwindow,
    key: c_int,
    scancode: c_int,
    action: c_int,
    modifiers: c_int,
) callconv(.C) void {
    const engine: *Engine = @ptrCast(@alignCast(c.glfwGetWindowUserPointer(window)));
    engine.onKeyAction(key, scancode, action, modifiers);
}

fn mouseButtonActionCallback(
    window: ?*c.GLFWwindow,
    button: c_int,
    action: c_int,
    modifiers: c_int,
) callconv(.C) void {
    const engine: *Engine = @ptrCast(@alignCast(c.glfwGetWindowUserPointer(window)));
    engine.onMouseButtonAction(button, action, modifiers);
}

fn mousePositionChangedCallback(
    window: ?*c.GLFWwindow,
    x: f64,
    y: f64,
) callconv(.C) void {
    const engine: *Engine = @ptrCast(@alignCast(c.glfwGetWindowUserPointer(window)));
    engine.onMousePositionChanged(x, y);
}

fn openDataDir(allocator: std.mem.Allocator) !std.fs.Dir {
    const exe_dir_path = try std.fs.selfExeDirPathAlloc(allocator);
    defer allocator.free(exe_dir_path);
    const data_dir_path = try std.fs.path.join(allocator, &.{ exe_dir_path, "..", "data" });
    defer allocator.free(data_dir_path);
    return std.fs.openDirAbsolute(data_dir_path, .{});
}
