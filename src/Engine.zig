const std = @import("std");
const builtin = @import("builtin");

const c = @import("c.zig");
const fmt = @import("fmt.zig");
const glfw = @import("glfw.zig");
const math = @import("math.zig");
const mem = @import("mem.zig");
const wgpu = @import("wgpu.zig");

const Camera = @import("Camera.zig");
const Gltf = @import("zgltf").Gltf;

const Engine = @This();

allocator: std.mem.Allocator,

data_dir: std.fs.Dir,

window: *c.GLFWwindow,
surface: c.WGPUSurface,
surface_format: c.WGPUTextureFormat,
present_mode: c.WGPUPresentMode,

instance: c.WGPUInstance,
device: c.WGPUDevice,
queue: c.WGPUQueue,

model_bind_group_layout: c.WGPUBindGroupLayout,
texture_bind_group_layout: c.WGPUBindGroupLayout,
pipeline: c.WGPURenderPipeline,

depth_format: c.WGPUTextureFormat,
depth_texture: c.WGPUTexture,
depth_texture_view: c.WGPUTextureView,

imgui_context: *c.ImGuiContext,
imgui_io: *c.ImGuiIO,
last_instant: std.time.Instant,

default_sampler_bind_group: c.WGPUBindGroup,
fallback_base_texture: c.WGPUTextureView,
fallback_specular_map: c.WGPUTextureView,
frame: FrameRenderData,
dragon: ModelRenderData,
arena: ModelRenderData,
cube: ModelRenderData,
// TODO: This shouldn't have to be an array of meshes. We need to rework the
// "Model" concept to match glTF's "scene". I.e. models should be composed of
// multiple nodes of meshes.
porche: struct {
    meshes: []ModelRenderData = &.{},
},

camera: Camera,

mouse_captured: bool,
last_mouse_position: math.Vec2,

pub const model_space = math.CoordinateSystem.glTF;
pub const world_space = math.CoordinateSystem.vulkan;
// This transform converts from the model space coordinate system to the
// world space. It should probably be applied to all objects.
pub const model_transform = math.CoordinateSystem.transform(model_space, world_space);

// TODO: Move these to build.zig and pass them to buildShaders.
const SetIndex = enum(comptime_int) {
    frame,
    sampler,
    model,
    texture,
};

const Vertex = struct {
    position: math.Vec3,
    normal: math.Vec3,
    uv: math.Vec2,
};

const FrameUniform = extern struct {
    view: math.Mat4,
    proj: math.Mat4,
    camera_position: math.Vec3 align(16),
};

const ModelUniform = extern struct {
    model: math.Mat4,
    normal: math.Mat4x3,
};

const FrameRenderData = struct {
    uniform_data: FrameUniform,
    ubo: c.WGPUBuffer,
    bind_group: c.WGPUBindGroup,

    pub fn deinit(self: FrameRenderData) void {
        c.wgpuBindGroupRelease(self.bind_group);
        c.wgpuBufferRelease(self.ubo);
    }
};

const ModelRenderData = struct {
    uniform_data: ModelUniform,
    ubo: c.WGPUBuffer,
    vbo: c.WGPUBuffer,
    vbo_content_size: usize,
    ibo: c.WGPUBuffer,
    ibo_content_size: usize,
    index_count: usize,
    uniform_bind_group: c.WGPUBindGroup,
    texture_bind_group: c.WGPUBindGroup,

    pub fn deinit(self: ModelRenderData) void {
        c.wgpuBindGroupRelease(self.texture_bind_group);
        c.wgpuBindGroupRelease(self.uniform_bind_group);
        c.wgpuBufferDestroy(self.ibo);
        c.wgpuBufferRelease(self.ibo);
        c.wgpuBufferDestroy(self.vbo);
        c.wgpuBufferRelease(self.vbo);
        c.wgpuBufferDestroy(self.ubo);
        c.wgpuBufferRelease(self.ubo);
    }
};

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

    const instance = try wgpu.createInstance(.{});
    errdefer c.wgpuInstanceRelease(instance);

    const surface = c.glfwCreateWGPUSurface(instance, window) orelse return error.CreateSurfaceFailed;
    errdefer c.wgpuSurfaceRelease(surface);

    const adapter = try wgpu.instanceRequestAdapter(instance, .{ .compatibleSurface = surface });
    defer c.wgpuAdapterRelease(adapter);

    var adapter_properties: c.WGPUAdapterProperties = .{};
    c.wgpuAdapterGetProperties(adapter, &adapter_properties);
    std.log.debug("adapter properties:\n" ++
        "\tvendor id:\t{}\n" ++
        "\tdevice id:\t{}\n" ++
        "\tvendor name:\t{?s}\n" ++
        "\tdevice name:\t{?s}\n" ++
        "\tarchitecture:\t{?s}\n" ++
        "\tdescription:\t{?s}\n" ++
        "\tadapter type:\t{s}\n" ++
        "\tbackend type:\t{s}", .{
        adapter_properties.vendorID,
        adapter_properties.deviceID,
        adapter_properties.vendorName,
        adapter_properties.name,
        adapter_properties.architecture,
        adapter_properties.driverDescription,
        wgpu.adapterTypeToString(adapter_properties.adapterType),
        wgpu.backendTypeToString(adapter_properties.backendType),
    });

    var adapter_limits: c.WGPUSupportedLimits = .{};
    if (c.wgpuAdapterGetLimits(adapter, &adapter_limits) == 0) {
        std.log.warn("failed to get adapter limits", .{});
    } else {
        std.log.debug("adapter limits:", .{});
        inline for (comptime std.meta.fieldNames(c.WGPULimits)) |field_name| {
            std.log.debug("\t" ++ field_name ++ ": {}", .{@field(adapter_limits.limits, field_name)});
        }
    }

    const adapter_features = try wgpu.adapterEnumerateFeatures(adapter, allocator);
    defer allocator.free(adapter_features);
    std.log.debug("adapter features: [{s}]", .{
        fmt.fmtSliceElementFormatter(
            c.WGPUFeatureName,
            adapter_features,
            wgpu.formatFeature,
        ),
    });

    std.mem.sortUnstable(c.WGPUFeatureName, adapter_features, {}, std.sort.asc(c.WGPUFeatureName));
    const required_features = [_]c.WGPUFeatureName{
        c.WGPUFeatureName_TextureCompressionBC,
    };
    if (!mem.containsAll(c.WGPUFeatureName, adapter_features, &required_features)) {
        return error.RequiredFeaturesMissing;
    }

    const device = try wgpu.adapterRequestDevice(adapter, .{
        .requiredFeatureCount = required_features.len,
        .requiredFeatures = &required_features,
        .deviceLostCallback = onDeviceLost,
    });
    errdefer c.wgpuDeviceRelease(device);
    c.wgpuDeviceSetUncapturedErrorCallback(device, onUncapturedError, null);

    var device_limits: c.WGPUSupportedLimits = .{};
    if (c.wgpuDeviceGetLimits(device, &device_limits) == 0) {
        std.log.warn("failed to get device limits", .{});
    } else {
        std.log.debug("device limits:", .{});
        inline for (comptime std.meta.fieldNames(c.WGPULimits)) |field_name| {
            std.log.debug("\t" ++ field_name ++ ": {}", .{@field(device_limits.limits, field_name)});
        }
    }

    const device_features = try wgpu.deviceEnumerateFeatures(device, allocator);
    defer allocator.free(device_features);
    std.log.debug("device features: [{s}]", .{
        fmt.fmtSliceElementFormatter(
            c.WGPUFeatureName,
            device_features,
            wgpu.formatFeature,
        ),
    });

    const queue = c.wgpuDeviceGetQueue(device);
    errdefer c.wgpuQueueRelease(queue);

    const frame_bind_group_layout = c.wgpuDeviceCreateBindGroupLayout(device, &.{
        .entryCount = 1,
        .entries = &.{
            .binding = 0,
            .visibility = c.WGPUShaderStage_Vertex | c.WGPUShaderStage_Fragment,
            .buffer = .{
                .type = c.WGPUBufferBindingType_Uniform,
                .hasDynamicOffset = @intFromBool(false),
                .minBindingSize = @sizeOf(FrameUniform),
            },
        },
    });
    defer c.wgpuBindGroupLayoutRelease(frame_bind_group_layout);
    const sampler_bind_group_layout = c.wgpuDeviceCreateBindGroupLayout(device, &.{
        .entryCount = 1,
        .entries = &.{
            .binding = 0,
            .visibility = c.WGPUShaderStage_Fragment,
            .sampler = .{
                .type = c.WGPUSamplerBindingType_Filtering,
            },
        },
    });
    defer c.wgpuBindGroupLayoutRelease(sampler_bind_group_layout);
    const model_bind_group_layout = c.wgpuDeviceCreateBindGroupLayout(device, &.{
        .entryCount = 1,
        .entries = &.{
            .binding = 0,
            .visibility = c.WGPUShaderStage_Vertex,
            .buffer = .{
                .type = c.WGPUBufferBindingType_Uniform,
                .hasDynamicOffset = @intFromBool(false),
                .minBindingSize = @sizeOf(ModelUniform),
            },
        },
    });
    errdefer c.wgpuBindGroupLayoutRelease(model_bind_group_layout);
    const texture_bind_group_layout_entries: []const c.WGPUBindGroupLayoutEntry = &.{
        .{
            .binding = 0,
            .visibility = c.WGPUShaderStage_Fragment,
            .texture = .{
                .sampleType = c.WGPUTextureSampleType_Float,
                .viewDimension = c.WGPUTextureViewDimension_2D,
                .multisampled = @intFromBool(false),
            },
        },
        .{
            .binding = 1,
            .visibility = c.WGPUShaderStage_Fragment,
            .texture = .{
                .sampleType = c.WGPUTextureSampleType_Float,
                .viewDimension = c.WGPUTextureViewDimension_2D,
                .multisampled = @intFromBool(false),
            },
        },
    };
    const texture_bind_group_layout = c.wgpuDeviceCreateBindGroupLayout(device, &.{
        .entryCount = @intCast(texture_bind_group_layout_entries.len),
        .entries = texture_bind_group_layout_entries.ptr,
    });
    errdefer c.wgpuBindGroupLayoutRelease(texture_bind_group_layout);

    const bind_group_layouts: []const c.WGPUBindGroupLayout = &.{
        frame_bind_group_layout,
        sampler_bind_group_layout,
        model_bind_group_layout,
        texture_bind_group_layout,
    };
    const pipeline_layout = c.wgpuDeviceCreatePipelineLayout(device, &.{
        .bindGroupLayoutCount = @intCast(bind_group_layouts.len),
        .bindGroupLayouts = bind_group_layouts.ptr,
    });
    defer c.wgpuPipelineLayoutRelease(pipeline_layout);

    const vs_data = try data_dir.readFileAllocOptions(
        allocator,
        "shaders/basic.vert.spv",
        4 * 1024 * 1024,
        null,
        @alignOf(u32),
        null,
    );
    defer allocator.free(vs_data);
    const vs_module = try wgpu.deviceCreateShaderModuleSPIRV(
        device,
        std.mem.bytesAsSlice(u32, vs_data),
    );
    defer c.wgpuShaderModuleRelease(vs_module);

    const fs_data = try data_dir.readFileAllocOptions(
        allocator,
        "shaders/basic.frag.spv",
        4 * 1024 * 1024,
        null,
        @alignOf(u32),
        null,
    );
    defer allocator.free(fs_data);
    const fs_module = try wgpu.deviceCreateShaderModuleSPIRV(
        device,
        std.mem.bytesAsSlice(u32, fs_data),
    );
    defer c.wgpuShaderModuleRelease(fs_module);

    var surface_capabilities: c.WGPUSurfaceCapabilities = .{};
    c.wgpuSurfaceGetCapabilities(surface, adapter, &surface_capabilities);
    defer c.wgpuSurfaceCapabilitiesFreeMembers(surface_capabilities);

    const surface_formats = mem.sliceFromParts(surface_capabilities.formats, surface_capabilities.formatCount);
    if (surface_formats.len == 0) return error.NoSurfaceFormatsAvailable;
    std.mem.sortUnstable(c.WGPUTextureFormat, surface_formats, {}, std.sort.asc(c.WGPUTextureFormat));
    std.log.debug("surface formats: [{s}]", .{
        fmt.fmtSliceElementFormatter(
            c.WGPUTextureFormat,
            surface_formats,
            wgpu.formatTextureFormat,
        ),
    });

    const present_modes = mem.sliceFromParts(surface_capabilities.presentModes, surface_capabilities.presentModeCount);
    if (present_modes.len == 0) return error.NoPresentModesAvailable;
    std.mem.sortUnstable(c.WGPUPresentMode, present_modes, {}, std.sort.asc(c.WGPUPresentMode));
    std.log.debug("present modes: [{s}]", .{
        fmt.fmtSliceElementFormatter(
            c.WGPUPresentMode,
            present_modes,
            wgpu.formatPresentMode,
        ),
    });

    const desired_surface_formats: []const c.WGPUTextureFormat = &.{
        c.WGPUTextureFormat_RGBA16Float,
        c.WGPUTextureFormat_BGRA8UnormSrgb,
        c.WGPUTextureFormat_RGBA8UnormSrgb,
        c.WGPUTextureFormat_BGRA8Unorm,
        c.WGPUTextureFormat_RGBA8Unorm,
    };
    const surface_format = mem.findFirstOf(
        c.WGPUTextureFormat,
        surface_formats,
        desired_surface_formats,
    ) orelse return error.DesiredSurfaceFormatNotAvailable;
    std.log.debug("selected surface format: {s}", .{wgpu.textureFormatToString(surface_format)});

    // Don't include FIFO because we'll fall back to it anyway if none of the
    // desired modes are available; FIFO is guaranteed by the WebGPU spec to be
    // available.
    const desired_present_modes: []const c.WGPUPresentMode = &.{
        c.WGPUPresentMode_Mailbox,
        c.WGPUPresentMode_Immediate,
        c.WGPUPresentMode_FifoRelaxed,
    };
    const present_mode = mem.findFirstOf(
        c.WGPUPresentMode,
        present_modes,
        desired_present_modes,
    ) orelse c.WGPUPresentMode_Fifo;
    std.log.debug("selected present mode: {s}", .{wgpu.presentModeToString(present_mode)});

    const vertex_attributes = wgpu.vertexAttributesFromType(Vertex, .{});

    // Must declare type otherwise compilation fails on macOS with:
    //      error: expected type '[*c]const c_uint', found '*const c_int'
    //          .viewFormats = &depth_format,
    const depth_format: c.WGPUTextureFormat = c.WGPUTextureFormat_Depth32Float;

    const pipeline = c.wgpuDeviceCreateRenderPipeline(device, &.{
        .layout = pipeline_layout,
        .vertex = .{
            .module = vs_module,
            .entryPoint = "main",
            .bufferCount = 1,
            .buffers = &.{
                .arrayStride = @sizeOf(Vertex),
                .stepMode = c.WGPUVertexStepMode_Vertex,
                .attributeCount = vertex_attributes.len,
                .attributes = &vertex_attributes,
            },
        },
        .primitive = .{
            .topology = c.WGPUPrimitiveTopology_TriangleList,
            .frontFace = c.WGPUFrontFace_CCW,
            .cullMode = c.WGPUCullMode_Back,
        },
        .depthStencil = &.{
            .format = depth_format,
            .depthWriteEnabled = @intFromBool(true),
            .depthCompare = c.WGPUCompareFunction_GreaterEqual,
            .stencilFront = .{
                .compare = c.WGPUCompareFunction_Always,
                .failOp = c.WGPUStencilOperation_Keep,
                .depthFailOp = c.WGPUStencilOperation_Keep,
                .passOp = c.WGPUStencilOperation_Keep,
            },
            .stencilBack = .{
                .compare = c.WGPUCompareFunction_Always,
                .failOp = c.WGPUStencilOperation_Keep,
                .depthFailOp = c.WGPUStencilOperation_Keep,
                .passOp = c.WGPUStencilOperation_Keep,
            },
            .stencilReadMask = ~@as(u32, 0),
            .stencilWriteMask = ~@as(u32, 0),
            .depthBias = 0,
            .depthBiasSlopeScale = 0,
            .depthBiasClamp = 0,
        },
        .multisample = .{
            .count = 1,
            .mask = ~@as(u32, 0),
            .alphaToCoverageEnabled = @intFromBool(false),
        },
        .fragment = &.{
            .module = fs_module,
            .entryPoint = "main",
            .targetCount = 1,
            .targets = &.{
                .format = surface_format,
                .blend = &.{
                    .color = .{
                        .operation = c.WGPUBlendOperation_Add,
                        .srcFactor = c.WGPUBlendFactor_SrcAlpha,
                        .dstFactor = c.WGPUBlendFactor_OneMinusSrcAlpha,
                    },
                    .alpha = .{
                        .operation = c.WGPUBlendOperation_Add,
                        .srcFactor = c.WGPUBlendFactor_Zero,
                        .dstFactor = c.WGPUBlendFactor_One,
                    },
                },
                .writeMask = c.WGPUColorWriteMask_All,
            },
        },
    });
    errdefer c.wgpuRenderPipelineRelease(pipeline);

    const framebuffer_extent = glfw.getFramebufferSize(window);

    const camera = Camera.init(.{
        .position = math.vec3Scale(world_space.forward.vector(), -2.5),
        .target = world_space.forward.vector(),
    });

    const frame: FrameRenderData = blk: {
        const aspect_ratio = framebuffer_extent.aspectRatio();

        const camera_matrices = camera.computeMatrices();
        const uniform_data: FrameUniform = .{
            .view = camera_matrices.view,
            .proj = math.perspectiveInverseDepth(std.math.degreesToRadians(80.0), aspect_ratio, 0.01),
            .camera_position = camera.position,
        };
        const ubo = c.wgpuDeviceCreateBuffer(device, &.{
            .usage = c.WGPUBufferUsage_Uniform | c.WGPUBufferUsage_CopyDst,
            .size = @sizeOf(@TypeOf(uniform_data)),
        });
        errdefer c.wgpuBufferRelease(ubo);
        errdefer c.wgpuBufferDestroy(ubo);
        c.wgpuQueueWriteBuffer(queue, ubo, 0, &uniform_data, @sizeOf(@TypeOf(uniform_data)));

        const bind_group = c.wgpuDeviceCreateBindGroup(device, &.{
            .layout = frame_bind_group_layout,
            .entryCount = 1,
            .entries = &.{
                .binding = 0,
                .buffer = ubo,
                .offset = 0,
                .size = @sizeOf(@TypeOf(uniform_data)),
            },
        });
        errdefer c.wgpuBindGroupRelease(bind_group);

        break :blk .{
            .uniform_data = uniform_data,
            .ubo = ubo,
            .bind_group = bind_group,
        };
    };
    errdefer frame.deinit();

    const linear_sampler = c.wgpuDeviceCreateSampler(device, &.{
        .addressModeU = c.WGPUAddressMode_Repeat,
        .addressModeV = c.WGPUAddressMode_Repeat,
        .addressModeW = c.WGPUAddressMode_Repeat,
        .magFilter = c.WGPUFilterMode_Linear,
        .minFilter = c.WGPUFilterMode_Linear,
        .mipmapFilter = c.WGPUMipmapFilterMode_Linear,
        .lodMinClamp = 0,
        .lodMaxClamp = std.math.floatMax(f32),
        .maxAnisotropy = 16.0,
    });
    defer c.wgpuSamplerRelease(linear_sampler);

    const default_sampler_bind_group = c.wgpuDeviceCreateBindGroup(device, &.{
        .layout = sampler_bind_group_layout,
        .entryCount = 1,
        .entries = &.{
            .binding = 0,
            .sampler = linear_sampler,
        },
    });
    errdefer c.wgpuBindGroupRelease(default_sampler_bind_group);

    const imgui_context = c.ImGui_CreateContext(null) orelse return error.ImGuiCreateContextFailed;
    errdefer c.ImGui_DestroyContext(imgui_context);
    if (!c.ImGuiBackendInitialize(&.{
        .window = window,
        .device = device,
        .renderTargetFormat = surface_format,
        .depthStencilFormat = depth_format,
    })) {
        return error.ImGuiBackendInitializeFailed;
    }
    errdefer c.ImGuiBackendTerminate();

    engine.* = .{
        .allocator = allocator,

        .data_dir = data_dir,

        .window = window,
        .surface = surface,
        .surface_format = surface_format,
        .present_mode = present_mode,

        .instance = instance,
        .device = device,
        .queue = queue,

        .model_bind_group_layout = model_bind_group_layout,
        .texture_bind_group_layout = texture_bind_group_layout,
        .pipeline = pipeline,

        .depth_format = depth_format,
        // Deferred to Engine.createSwapChain.
        .depth_texture = null,
        .depth_texture_view = null,

        .imgui_context = imgui_context,
        .imgui_io = c.ImGui_GetIO(),
        .last_instant = std.time.Instant.now() catch unreachable,

        .default_sampler_bind_group = default_sampler_bind_group,
        .frame = frame,
        // Deferred to Engine.loadTexture.
        .fallback_base_texture = undefined,
        .fallback_specular_map = undefined,
        // Deferred to Engine.loadModel.
        .dragon = undefined,
        .arena = undefined,
        .cube = undefined,
        .porche = .{},

        .camera = camera,

        .mouse_captured = true,
        .last_mouse_position = mouse_position,
    };
    // TODO: Can we make it clear what resources need to be valid for these post-init functions?
    // Could maybe resolve by factoring out base graphics API resources to GraphicsContext.
    engine.createSwapChain(framebuffer_extent.width, framebuffer_extent.height);

    engine.fallback_base_texture = try engine.loadTexture("textures/missing_bc7.ktx2");
    errdefer c.wgpuTextureViewRelease(engine.fallback_base_texture);
    engine.fallback_specular_map = try engine.loadTexture("textures/missing_specular_bc4u.ktx2");
    errdefer c.wgpuTextureViewRelease(engine.fallback_specular_map);

    var dragon_translate = math.mat4Identity();
    var dragon_down: c.vec3 = math.vec3Scale(world_space.up.vector(), -1.0);
    c.glmc_translate(&dragon_translate, &dragon_down);
    engine.dragon = try engine.loadModel(.{
        .base_texture_path = "textures/stanford_dragon/stanford_dragon_base_bc7.ktx2",
        .kind = .{ .scene = "meshes/stanford_dragon.glb" },
        .transform = math.scaleUniform(dragon_translate, 2.0),
    });
    engine.arena = try engine.loadModel(.{
        .kind = .{ .scene = "meshes/arena.glb" },
    });
    var down: c.vec3 = math.vec3Scale(world_space.up.vector(), -1.4);
    var cube_translate_scale = math.mat4Identity();
    c.glmc_translate(&cube_translate_scale, &down);
    cube_translate_scale = math.scaleUniform(cube_translate_scale, 0.4);
    engine.cube = try engine.loadModel(.{
        .base_texture_path = "textures/crate/crate_base_bc7.ktx2",
        .specular_map_path = "textures/crate/crate_specular_bc4u.ktx2",
        .kind = .{ .scene = "meshes/cube.glb" },
        .transform = cube_translate_scale,
    });

    const porche_scene_data = try data_dir.readFileAllocOptions(
        allocator,
        "meshes/porche.glb",
        8 * 1024 * 1024,
        null,
        @alignOf(u32),
        null,
    );
    defer allocator.free(porche_scene_data);
    var porche_gltf = Gltf.init(allocator);
    defer porche_gltf.deinit();
    try porche_gltf.parse(porche_scene_data);
    engine.porche.meshes = try allocator.alloc(ModelRenderData, porche_gltf.data.nodes.items.len);
    errdefer allocator.free(engine.porche.meshes);
    var translate_rotate = math.mat4Identity();
    var forward_down: c.vec3 = math.vec3Scale(world_space.forward.vector(), 2.5) + math.vec3Scale(world_space.up.vector(), -1.95);
    c.glmc_translate(&translate_rotate, &forward_down);
    var vertical_axis: c.vec3 = world_space.up.vector();
    c.glmc_rotate(&translate_rotate, std.math.degreesToRadians(90.0), &vertical_axis);
    for (engine.porche.meshes, porche_gltf.data.nodes.items) |*renderable, node| {
        // TODO: Handle loading different textures for each renderable.
        // TODO: Load nodes in hierarchical order so we can apply inherited
        // transforms.
        // TODO: Read material data (e.g. colour).
        renderable.* = try engine.loadModel(.{
            .kind = .{
                .node = .{
                    .gltf = porche_gltf,
                    .data = node,
                },
            },
            .transform = translate_rotate,
        });
    }

    return engine;
}

pub fn deinit(self: *Engine) void {
    c.ImGuiBackendTerminate();
    c.ImGui_DestroyContext(self.imgui_context);

    for (self.porche.meshes) |mesh| mesh.deinit();
    self.allocator.free(self.porche.meshes);
    self.cube.deinit();
    self.arena.deinit();
    self.dragon.deinit();
    c.wgpuTextureViewRelease(self.fallback_specular_map);
    c.wgpuTextureViewRelease(self.fallback_base_texture);

    self.frame.deinit();

    c.wgpuBindGroupRelease(self.default_sampler_bind_group);

    c.wgpuTextureViewRelease(self.depth_texture_view);
    c.wgpuTextureDestroy(self.depth_texture);
    c.wgpuTextureRelease(self.depth_texture);

    c.wgpuRenderPipelineRelease(self.pipeline);
    c.wgpuBindGroupLayoutRelease(self.texture_bind_group_layout);
    c.wgpuBindGroupLayoutRelease(self.model_bind_group_layout);

    c.wgpuQueueRelease(self.queue);
    c.wgpuDeviceRelease(self.device);
    c.wgpuSurfaceUnconfigure(self.surface);
    c.wgpuSurfaceRelease(self.surface);
    c.wgpuInstanceRelease(self.instance);
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

const LoadModelOptions = struct {
    base_texture_path: ?[]const u8 = null,
    specular_map_path: ?[]const u8 = null,
    kind: union(enum) {
        scene: []const u8,
        node: struct {
            gltf: Gltf,
            data: Gltf.Node,
        },
    },
    transform: math.Mat4 = math.mat4Identity(),
};

fn loadModel(self: *Engine, options: LoadModelOptions) !ModelRenderData {
    // TODO: Do tonemapping so HDR textures look ok with SDR display.
    var base: c.WGPUTextureView = null;
    if (options.base_texture_path) |path| {
        base = try self.loadTexture(path);
    }
    defer if (base) |view| c.wgpuTextureViewRelease(view);

    var specular: c.WGPUTextureView = null;
    if (options.specular_map_path) |path| {
        specular = try self.loadTexture(path);
    }
    defer if (specular) |view| c.wgpuTextureViewRelease(view);

    // TODO: Maybe factor this out into a separate function.
    const scene_data: []align(@alignOf(u32)) u8, const gltf_owned, var gltf: Gltf, const node: Gltf.Node =
        switch (options.kind) {
        .node => |info| .{ &.{}, false, info.gltf, info.data },
        .scene => |path| blk: {
            const scene_data = try self.data_dir.readFileAllocOptions(
                self.allocator,
                path,
                4 * 1024 * 1024,
                null,
                @alignOf(u32),
                null,
            );
            errdefer self.allocator.free(scene_data);
            var gltf = Gltf.init(self.allocator);
            errdefer gltf.deinit();
            try gltf.parse(scene_data);

            const scene_index = gltf.data.scene.?;
            const scene = gltf.data.scenes.items[scene_index];
            const node_index = scene.nodes.?.items[0];
            const node = gltf.data.nodes.items[node_index];

            break :blk .{ scene_data, true, gltf, node };
        },
    };
    defer self.allocator.free(scene_data);
    defer if (gltf_owned) gltf.deinit();

    const mesh_index = node.mesh.?;
    const mesh = gltf.data.meshes.items[mesh_index];

    // https://microsoft.github.io/mixed-reality-extension-sdk/gltf-gen/interfaces/gltf.node.html:
    // A node can have _either_ a matrix or any combination of
    // translation/rotation/scale (TRS) properties.
    var transform: math.Mat4 = math.mat4Identity();
    // 1. Apply node transforms (which are in glTF space).
    if (node.matrix) |matrix| {
        transform = @bitCast(matrix);
    } else {
        var mut_translation: c.vec3 = node.translation;
        c.glmc_translate(&transform, &mut_translation);
        var mut_rotation: c.versor align(32) = node.rotation;
        c.glmc_quat_rotate(&transform, &mut_rotation, &transform);
        transform = math.scale(transform, node.scale);
    }
    // 2. Convert to world space.
    transform = math.mat4Mul(transform, model_transform);
    // 3. Apply custom transforms (which are in world space).
    transform = math.mat4Mul(transform, options.transform);

    var model_matrix: c.mat4 align(32) = transform;
    var model_inv: math.Mat4 = undefined;
    c.glmc_mat4_inv(&model_matrix, &model_inv);
    var normal_matrix: math.Mat4 = undefined;
    c.glmc_mat4_transpose_to(&model_inv, &normal_matrix);

    const uniform_data: ModelUniform = .{
        .model = model_matrix,
        .normal = normal_matrix[0..3].*,
    };
    const ubo = c.wgpuDeviceCreateBuffer(self.device, &.{
        .usage = c.WGPUBufferUsage_Uniform | c.WGPUBufferUsage_CopyDst,
        .size = @sizeOf(@TypeOf(uniform_data)),
    });
    errdefer c.wgpuBufferRelease(ubo);
    errdefer c.wgpuBufferDestroy(ubo);
    c.wgpuQueueWriteBuffer(self.queue, ubo, 0, &uniform_data, @sizeOf(@TypeOf(uniform_data)));

    var vertex_count: usize = 0;
    var index_count: usize = 0;
    for (mesh.primitives.items) |primitive| {
        index_count += @intCast(gltf.data.accessors.items[primitive.indices.?].count);
        for (primitive.attributes.items) |attribute| {
            switch (attribute) {
                .position => |index| {
                    vertex_count += @intCast(gltf.data.accessors.items[index].count);
                    break;
                },
                else => {},
            }
        }
    }

    const vertices = try self.allocator.alloc(Vertex, vertex_count);
    defer self.allocator.free(vertices);
    const indices = try self.allocator.alloc(u32, index_count);
    defer self.allocator.free(indices);
    var vertex_cursor: usize = 0;
    var index_cursor: usize = 0;
    for (mesh.primitives.items) |primitive| {
        var position_accessor: Gltf.Accessor = undefined;
        var normal_accessor: Gltf.Accessor = undefined;
        var uv_accessor: Gltf.Accessor = undefined;
        for (primitive.attributes.items) |attribute| {
            switch (attribute) {
                .position => |index| {
                    position_accessor = gltf.data.accessors.items[index];
                },
                .normal => |index| {
                    normal_accessor = gltf.data.accessors.items[index];
                },
                .texcoord => |index| {
                    uv_accessor = gltf.data.accessors.items[index];
                },
                else => {},
            }
        }

        std.debug.assert(position_accessor.component_type == .float);
        std.debug.assert(normal_accessor.component_type == .float);
        std.debug.assert(uv_accessor.component_type == .float);

        var position_iter = position_accessor.iterator(f32, gltf, gltf.glb_binary.?);
        var normal_iter = normal_accessor.iterator(f32, gltf, gltf.glb_binary.?);
        var uv_iter = uv_accessor.iterator(f32, gltf, gltf.glb_binary.?);
        while (position_iter.next()) |position| : (vertex_cursor += 1) {
            const normal = normal_iter.next().?;
            const uv = uv_iter.next().?;
            vertices[vertex_cursor] = .{
                .position = position[0..][0..3].*,
                .normal = normal[0..][0..3].*,
                .uv = uv[0..][0..2].*,
            };
        }

        const index_accessor = gltf.data.accessors.items[primitive.indices.?];
        std.debug.assert(index_accessor.component_type == .unsigned_short);
        var index_iter = index_accessor.iterator(u16, gltf, gltf.glb_binary.?);
        while (index_iter.next()) |index| : (index_cursor += 1) {
            indices[index_cursor] = index[0];
        }
    }

    const vbo_content_size = mem.sizeOfElements(vertices);
    const vbo = c.wgpuDeviceCreateBuffer(self.device, &.{
        .usage = c.WGPUBufferUsage_Vertex | c.WGPUBufferUsage_CopyDst,
        .size = vbo_content_size,
    });
    errdefer c.wgpuBufferRelease(vbo);
    errdefer c.wgpuBufferDestroy(vbo);
    c.wgpuQueueWriteBuffer(self.queue, vbo, 0, vertices.ptr, vbo_content_size);

    const ibo_content_size = mem.sizeOfElements(indices);
    const ibo = c.wgpuDeviceCreateBuffer(self.device, &.{
        .usage = c.WGPUBufferUsage_Index | c.WGPUBufferUsage_CopyDst,
        .size = ibo_content_size,
    });
    errdefer c.wgpuBufferRelease(ibo);
    errdefer c.wgpuBufferDestroy(ibo);
    c.wgpuQueueWriteBuffer(self.queue, ibo, 0, indices.ptr, ibo_content_size);

    const uniform_bind_group = c.wgpuDeviceCreateBindGroup(self.device, &.{
        .layout = self.model_bind_group_layout,
        .entryCount = 1,
        .entries = &.{
            .buffer = ubo,
            .size = @sizeOf(@TypeOf(uniform_data)),
        },
    });
    errdefer c.wgpuBindGroupRelease(uniform_bind_group);

    const texture_bind_group_entries: []const c.WGPUBindGroupEntry = &.{
        .{ .binding = 0, .textureView = base orelse self.fallback_base_texture },
        .{ .binding = 1, .textureView = specular orelse self.fallback_specular_map },
    };
    const texture_bind_group = c.wgpuDeviceCreateBindGroup(self.device, &.{
        .layout = self.texture_bind_group_layout,
        .entryCount = @intCast(texture_bind_group_entries.len),
        .entries = texture_bind_group_entries.ptr,
    });
    errdefer c.wgpuBindGroupRelease(texture_bind_group);

    return .{
        .uniform_data = uniform_data,
        .ubo = ubo,
        .vbo = vbo,
        .vbo_content_size = vbo_content_size,
        .ibo = ibo,
        .ibo_content_size = ibo_content_size,
        .index_count = index_count,
        .uniform_bind_group = uniform_bind_group,
        .texture_bind_group = texture_bind_group,
    };
}

fn loadTexture(self: *Engine, path: []const u8) !c.WGPUTextureView {
    const texture_data = try self.data_dir.readFileAlloc(
        self.allocator,
        path,
        512 * 1024 * 1024,
    );
    defer self.allocator.free(texture_data);

    var ktx_texture: ?*c.ktxTexture2 = null;
    if (c.ktxTexture2_CreateFromMemory(
        texture_data.ptr,
        @intCast(texture_data.len),
        c.KTX_TEXTURE_CREATE_NO_FLAGS,
        &ktx_texture,
    ) != c.KTX_SUCCESS) {
        return error.CreateKtxTextureFailed;
    }
    defer c.ktxTexture2_Destroy(ktx_texture);

    const texture, const texture_format = try wgpu.deviceLoadTexture(
        self.device,
        self.queue,
        ktx_texture.?,
    );
    defer c.wgpuTextureRelease(texture);
    errdefer c.wgpuTextureDestroy(texture);

    const texture_view = c.wgpuTextureCreateView(texture, &.{
        .format = texture_format,
        .dimension = c.WGPUTextureViewDimension_2D,
        .baseMipLevel = 0,
        .mipLevelCount = ktx_texture.?.numLevels,
        .baseArrayLayer = 0,
        .arrayLayerCount = ktx_texture.?.numLayers,
        .aspect = c.WGPUTextureAspect_All,
    });
    errdefer c.wgpuTextureViewRelease(texture_view);

    return texture_view;
}

fn tick(self: *Engine) void {
    const now = std.time.Instant.now() catch unreachable;
    const delta_time_ns = now.since(self.last_instant);
    self.last_instant = now;
    const delta_time_ns_f64: f64 = @floatFromInt(delta_time_ns);
    const delta_time_s_f64 = delta_time_ns_f64 / std.time.ns_per_s;
    const delta_time: f32 = @floatCast(delta_time_s_f64);

    self.update(delta_time);
    self.renderFrame(delta_time);
}

fn createSwapChain(self: *Engine, width: u32, height: u32) void {
    c.wgpuSurfaceConfigure(self.surface, &.{
        .device = self.device,
        .format = self.surface_format,
        .usage = c.WGPUTextureUsage_RenderAttachment,
        .alphaMode = c.WGPUCompositeAlphaMode_Auto,
        .width = width,
        .height = height,
        .presentMode = self.present_mode,
    });

    const depth_texture = c.wgpuDeviceCreateTexture(self.device, &.{
        .usage = c.WGPUTextureUsage_RenderAttachment,
        .dimension = c.WGPUTextureDimension_2D,
        .size = .{
            .width = width,
            .height = height,
            .depthOrArrayLayers = 1,
        },
        .format = self.depth_format,
        .mipLevelCount = 1,
        .sampleCount = 1,
        .viewFormatCount = 1,
        .viewFormats = &self.depth_format,
    });
    errdefer c.wgpuTextureRelease(depth_texture);
    errdefer c.wgpuTextureDestroy(depth_texture);
    const depth_texture_view = c.wgpuTextureCreateView(depth_texture, &.{
        .format = self.depth_format,
        .dimension = c.WGPUTextureViewDimension_2D,
        .baseMipLevel = 0,
        .mipLevelCount = 1,
        .baseArrayLayer = 0,
        .arrayLayerCount = 1,
        .aspect = c.WGPUTextureAspect_DepthOnly,
    });
    errdefer c.wgpuTextureViewRelease(depth_texture_view);

    self.depth_texture = depth_texture;
    self.depth_texture_view = depth_texture_view;
}

fn isRunning(self: *Engine) bool {
    return c.glfwWindowShouldClose(self.window) != c.GLFW_TRUE;
}

fn renderFrame(self: *Engine, delta_time: f32) void {
    const view = wgpu.surfaceGetNextTextureView(
        self.surface,
        self.surface_format,
    ) catch |err| switch (err) {
        error.SurfaceTextureOutOfDate,
        error.SurfaceTextureSuboptimal,
        => {
            const extent = glfw.getFramebufferSize(self.window);
            self.recreateSwapChain(extent.width, extent.height);
            return;
        },
        error.OutOfMemory,
        error.DeviceLost,
        => {
            std.debug.panic("fatal error attempting to acquire next texture view: {s}", .{
                @errorName(err),
            });
        },
    };
    defer c.wgpuTextureViewRelease(view);

    c.ImGuiBackendBeginFrame();
    c.ImGui_NewFrame();

    renderStatistics(delta_time);

    c.ImGui_Render();

    const encoder = c.wgpuDeviceCreateCommandEncoder(self.device, &.{});
    defer c.wgpuCommandEncoderRelease(encoder);

    const render_pass = c.wgpuCommandEncoderBeginRenderPass(encoder, &.{
        .colorAttachmentCount = 1,
        .colorAttachments = &.{
            .view = view,
            .loadOp = c.WGPULoadOp_Clear,
            .storeOp = c.WGPUStoreOp_Store,
            .clearValue = wgpu.color(1.0, 0.0, 1.0, 1.0),
        },
        .depthStencilAttachment = &.{
            .view = self.depth_texture_view,
            .depthLoadOp = c.WGPULoadOp_Clear,
            .depthStoreOp = c.WGPUStoreOp_Store,
            .depthClearValue = 0.0,
            .depthReadOnly = @intFromBool(false),
        },
    });
    defer c.wgpuRenderPassEncoderRelease(render_pass);
    c.wgpuRenderPassEncoderSetPipeline(render_pass, self.pipeline);
    c.wgpuRenderPassEncoderSetBindGroup(
        render_pass,
        @intFromEnum(SetIndex.frame),
        self.frame.bind_group,
        0,
        null,
    );
    c.wgpuRenderPassEncoderSetBindGroup(
        render_pass,
        @intFromEnum(SetIndex.sampler),
        self.default_sampler_bind_group,
        0,
        null,
    );
    inline for (.{ self.dragon, self.arena, self.cube }) |model| {
        c.wgpuRenderPassEncoderSetBindGroup(
            render_pass,
            @intFromEnum(SetIndex.model),
            model.uniform_bind_group,
            0,
            null,
        );
        c.wgpuRenderPassEncoderSetBindGroup(
            render_pass,
            @intFromEnum(SetIndex.texture),
            model.texture_bind_group,
            0,
            null,
        );
        c.wgpuRenderPassEncoderSetVertexBuffer(
            render_pass,
            0,
            model.vbo,
            0,
            model.vbo_content_size,
        );
        c.wgpuRenderPassEncoderSetIndexBuffer(
            render_pass,
            model.ibo,
            c.WGPUIndexFormat_Uint32,
            0,
            model.ibo_content_size,
        );
        c.wgpuRenderPassEncoderDrawIndexed(render_pass, @intCast(model.index_count), 1, 0, 0, 0);
    }
    // FIXME: This is duplicated from the single-mesh model draw loop.
    for (self.porche.meshes) |model| {
        c.wgpuRenderPassEncoderSetBindGroup(
            render_pass,
            @intFromEnum(SetIndex.model),
            model.uniform_bind_group,
            0,
            null,
        );
        c.wgpuRenderPassEncoderSetBindGroup(
            render_pass,
            @intFromEnum(SetIndex.texture),
            model.texture_bind_group,
            0,
            null,
        );
        c.wgpuRenderPassEncoderSetVertexBuffer(
            render_pass,
            0,
            model.vbo,
            0,
            model.vbo_content_size,
        );
        c.wgpuRenderPassEncoderSetIndexBuffer(
            render_pass,
            model.ibo,
            c.WGPUIndexFormat_Uint32,
            0,
            model.ibo_content_size,
        );
        c.wgpuRenderPassEncoderDrawIndexed(render_pass, @intCast(model.index_count), 1, 0, 0, 0);
    }
    c.ImGuiBackendEndFrame(render_pass);
    c.wgpuRenderPassEncoderEnd(render_pass);

    const command_buffer = c.wgpuCommandEncoderFinish(encoder, &.{});
    defer c.wgpuCommandBufferRelease(command_buffer);
    c.wgpuQueueSubmit(self.queue, 1, &command_buffer);

    c.wgpuSurfacePresent(self.surface);
    _ = c.wgpuDevicePoll(self.device, @intFromBool(false), null);
}

fn update(self: *Engine, delta_time: f32) void {
    if (!self.imgui_io.WantCaptureKeyboard) {
        var move_direction: Camera.MoveDirection = .{};
        if (c.glfwGetKey(self.window, c.GLFW_KEY_W) == c.GLFW_PRESS) {
            move_direction.forward = true;
        }
        if (c.glfwGetKey(self.window, c.GLFW_KEY_A) == c.GLFW_PRESS) {
            move_direction.left = true;
        }
        if (c.glfwGetKey(self.window, c.GLFW_KEY_S) == c.GLFW_PRESS) {
            move_direction.backward = true;
        }
        if (c.glfwGetKey(self.window, c.GLFW_KEY_D) == c.GLFW_PRESS) {
            move_direction.right = true;
        }
        if (c.glfwGetKey(self.window, c.GLFW_KEY_SPACE) == c.GLFW_PRESS) {
            move_direction.up = true;
        }
        if (c.glfwGetKey(self.window, c.GLFW_KEY_LEFT_SHIFT) == c.GLFW_PRESS) {
            move_direction.down = true;
        }
        move_direction.normalize();
        self.camera.translate(delta_time, move_direction);
        const camera_matrices = self.camera.computeMatrices();
        self.frame.uniform_data.view = camera_matrices.view;
        self.frame.uniform_data.camera_position = self.camera.position;
        c.wgpuQueueWriteBuffer(self.queue, self.frame.ubo, 0, &self.frame.uniform_data, @sizeOf(FrameUniform));
    }
}

fn onFramebufferSizeChanged(self: *Engine, width: u32, height: u32) void {
    self.recreateSwapChain(width, height);
    const aspect_ratio = @as(f32, @floatFromInt(width)) / @as(f32, @floatFromInt(height));
    self.frame.uniform_data.proj = math.perspectiveInverseDepth(std.math.degreesToRadians(80.0), aspect_ratio, 0.01);
}

fn recreateSwapChain(self: *Engine, width: u32, height: u32) void {
    c.wgpuTextureViewRelease(self.depth_texture_view);
    c.wgpuTextureDestroy(self.depth_texture);
    c.wgpuTextureRelease(self.depth_texture);

    self.createSwapChain(width, height);
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
    const position: math.Vec2 = .{ @floatCast(x), @floatCast(y) };
    const delta = position - self.last_mouse_position;
    self.last_mouse_position = position;
    // TODO: This should be in `Engine.update`; we don't have `delta_time` here.
    self.camera.updateOrientation(delta);
}

fn framebufferSizeChangedCallback(
    window: ?*c.GLFWwindow,
    width: c_int,
    height: c_int,
) callconv(.C) void {
    const engine: *Engine = @ptrCast(@alignCast(c.glfwGetWindowUserPointer(window)));
    engine.onFramebufferSizeChanged(@intCast(width), @intCast(height));
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

fn onDeviceLost(
    reason: c.WGPUDeviceLostReason,
    maybe_message: ?[*:0]const u8,
    user_data: ?*anyopaque,
) callconv(.C) void {
    _ = user_data;

    const format = "device lost: reason: {s}";
    const args = .{wgpu.deviceLostReasonToString(reason)};

    std.log.err(format, args);
    if (maybe_message) |message| {
        std.log.err("{s}", .{message});
    }

    if (builtin.mode == .Debug) {
        @breakpoint();
    }
}

fn onUncapturedError(
    @"type": c.WGPUErrorType,
    maybe_message: ?[*:0]const u8,
    user_data: ?*anyopaque,
) callconv(.C) void {
    _ = user_data;

    const format = "uncaptured device error: type: {s}";
    const args = .{wgpu.errorTypeToString(@"type")};

    std.log.err(format, args);
    if (maybe_message) |message| {
        std.log.err("{s}", .{message});
    }

    if (builtin.mode == .Debug) {
        @breakpoint();
    }
}

fn renderStatistics(delta_time: f32) void {
    const padding = 8.0;
    const viewport = c.ImGui_GetMainViewport();
    const position: c.ImVec2 = .{
        .x = viewport.*.WorkPos.x + padding,
        .y = viewport.*.WorkPos.y + padding,
    };

    c.ImGui_PushStyleVar(c.ImGuiStyleVar_WindowBorderSize, 0.0);
    c.ImGui_SetNextWindowBgAlpha(0.5);
    c.ImGui_SetNextWindowPos(position, c.ImGuiCond_Always);
    var stats_open = true;
    _ = c.ImGui_Begin(
        "Frame Statistics",
        &stats_open,
        c.ImGuiWindowFlags_NoDecoration |
            c.ImGuiWindowFlags_AlwaysAutoResize |
            c.ImGuiWindowFlags_NoSavedSettings |
            c.ImGuiWindowFlags_NoFocusOnAppearing |
            c.ImGuiWindowFlags_NoNav |
            c.ImGuiWindowFlags_NoMove,
    );
    c.ImGui_Text("Frametime: %8.5f ms", delta_time * std.time.ms_per_s);
    c.ImGui_End();
    c.ImGui_PopStyleVar();
}

fn openDataDir(allocator: std.mem.Allocator) !std.fs.Dir {
    const exe_dir_path = try std.fs.selfExeDirPathAlloc(allocator);
    defer allocator.free(exe_dir_path);
    const data_dir_path = try std.fs.path.join(allocator, &.{ exe_dir_path, "..", "data" });
    defer allocator.free(data_dir_path);
    return std.fs.openDirAbsolute(data_dir_path, .{});
}
