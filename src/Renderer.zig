const std = @import("std");
const builtin = @import("builtin");
const resources = @import("resources");

const c = @import("c.zig");
const definitions = @import("renderer/definitions.zig");
const fmt = @import("fmt.zig");
const glfw = @import("glfw.zig");
const log = std.log.scoped(.renderer);
const math = @import("math.zig");
const mem = @import("mem.zig");
const wgpu = @import("wgpu.zig");

const Camera = @import("Camera.zig");
const Engine = @import("Engine.zig");
const Gltf = @import("zgltf").Gltf;

const Renderer = @This();

allocator: std.mem.Allocator,

device: c.WGPUDevice,
queue: c.WGPUQueue,

window: *c.GLFWwindow,
surface: c.WGPUSurface,
surface_format: c.WGPUTextureFormat,
present_mode: c.WGPUPresentMode,
swapchain: Swapchain,
should_recreate_swapchain: bool = false,
vsync: bool = false,

depth_format: c.WGPUTextureFormat,

bind_group_layouts: std.EnumArray(BindGroup, c.WGPUBindGroupLayout),
pipeline: c.WGPURenderPipeline,

imgui_context: *c.ImGuiContext,
settings_open: bool = true,

frame_data: FrameData,

linear_sampler_bind_group: c.WGPUBindGroup,
fallback_texture_bind_group: c.WGPUBindGroup,
textures: std.StringArrayHashMapUnmanaged(c.WGPUTextureView),
models: std.ArrayListUnmanaged(Model) = .{},

pub const BindGroup = definitions.BindGroup;

pub const Model = struct {
    meshes: []Mesh,
    materials: []Material,

    fn deinit(self: Model, allocator: std.mem.Allocator) void {
        for (self.meshes) |mesh| mesh.deinit(allocator);
        allocator.free(self.meshes);
        for (self.materials) |material| material.deinit();
        allocator.free(self.materials);
    }
};

pub const Mesh = struct {
    primitives: []Primitive,

    fn deinit(self: Mesh, allocator: std.mem.Allocator) void {
        for (self.primitives) |primitive| primitive.deinit();
        allocator.free(self.primitives);
    }
};

pub const Primitive = struct {
    material: ?usize,
    uniform_data: UniformData,
    uniform_buffer: c.WGPUBuffer,
    uniform_bind_group: c.WGPUBindGroup,
    vertex_buffer: c.WGPUBuffer,
    vertex_buffer_size: usize,
    index_buffer: c.WGPUBuffer,
    index_buffer_size: usize,
    index_count: usize,

    pub const UniformData = extern struct {
        model: math.Mat4,
        normal: math.Mat4x3,
    };

    fn deinit(self: Primitive) void {
        c.wgpuBufferDestroy(self.index_buffer);
        c.wgpuBufferRelease(self.index_buffer);
        c.wgpuBufferDestroy(self.vertex_buffer);
        c.wgpuBufferRelease(self.vertex_buffer);
        c.wgpuBindGroupRelease(self.uniform_bind_group);
        c.wgpuBufferDestroy(self.uniform_buffer);
        c.wgpuBufferRelease(self.uniform_buffer);
    }
};

pub const Material = struct {
    texture_bind_group: c.WGPUBindGroup,

    fn deinit(self: Material) void {
        c.wgpuBindGroupRelease(self.texture_bind_group);
    }
};

// This transform converts from the model space coordinate system to the
// world space. It should probably be applied to all objects.
const model_to_world_transform = math.CoordinateSystem.transform(Engine.model_space, Engine.world_space);

const Swapchain = struct {
    extent: math.Extent2D,
    depth_texture: c.WGPUTexture,
    depth_texture_view: c.WGPUTextureView,

    fn deinit(self: Swapchain) void {
        c.wgpuTextureViewRelease(self.depth_texture_view);
        c.wgpuTextureDestroy(self.depth_texture);
        c.wgpuTextureRelease(self.depth_texture);
    }
};

const FrameData = struct {
    uniform_data: UniformData,
    uniform_buffer: c.WGPUBuffer,
    uniform_bind_group: c.WGPUBindGroup,

    pub const UniformData = extern struct {
        view: math.Mat4,
        proj: math.Mat4,
        camera_position: math.Vec3,
    };

    fn deinit(self: FrameData) void {
        c.wgpuBindGroupRelease(self.uniform_bind_group);
        c.wgpuBufferDestroy(self.uniform_buffer);
        c.wgpuBufferRelease(self.uniform_buffer);
    }

    fn uploadToGpu(self: FrameData, queue: c.WGPUQueue) void {
        c.wgpuQueueWriteBuffer(
            queue,
            self.uniform_buffer,
            0,
            &self.uniform_data,
            @sizeOf(@TypeOf(self.uniform_data)),
        );
    }
};

const Vertex = struct {
    position: math.Vec3,
    normal: math.Vec3,
    uv: math.Vec2,
};

const InitOptions = struct {
    window: *c.GLFWwindow,
    vertex_shader_bytecode: []align(@alignOf(u32)) const u8,
    fragment_shader_bytecode: []align(@alignOf(u32)) const u8,
};

pub fn init(allocator: std.mem.Allocator, options: InitOptions) !Renderer {
    // NOTE: We could pass a WGPUInstanceExtras with
    // WGPUInstanceFlag_DiscardHalLabels set to gain some extra performance.
    const instance = try wgpu.createInstance(.{});
    defer c.wgpuInstanceRelease(instance);

    const surface = c.glfwCreateWGPUSurface(instance, options.window) orelse return error.CreateSurfaceFailed;
    errdefer c.wgpuSurfaceRelease(surface);

    const adapter = try wgpu.instanceRequestAdapter(instance, .{ .compatibleSurface = surface });
    defer c.wgpuAdapterRelease(adapter);

    var adapter_properties: c.WGPUAdapterProperties = .{};
    c.wgpuAdapterGetProperties(adapter, &adapter_properties);
    log.debug("adapter properties:\n" ++
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
        wgpu.adapterTypeName(adapter_properties.adapterType),
        wgpu.backendTypeName(adapter_properties.backendType),
    });

    var adapter_limits: c.WGPUSupportedLimits = .{};
    if (c.wgpuAdapterGetLimits(adapter, &adapter_limits) == 0) {
        log.warn("failed to get adapter limits", .{});
    } else {
        log.debug("adapter limits:", .{});
        inline for (comptime std.meta.fieldNames(c.WGPULimits)) |field_name| {
            log.debug("\t" ++ field_name ++ ": {}", .{@field(adapter_limits.limits, field_name)});
        }
    }

    const adapter_features = try wgpu.adapterEnumerateFeatures(adapter, allocator);
    defer allocator.free(adapter_features);
    log.debug("adapter features: [{s}]", .{
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
        log.warn("failed to get device limits", .{});
    } else {
        log.debug("device limits:", .{});
        inline for (comptime std.meta.fieldNames(c.WGPULimits)) |field_name| {
            log.debug("\t" ++ field_name ++ ": {}", .{@field(device_limits.limits, field_name)});
        }
    }

    const device_features = try wgpu.deviceEnumerateFeatures(device, allocator);
    defer allocator.free(device_features);
    log.debug("device features: [{s}]", .{
        fmt.fmtSliceElementFormatter(
            c.WGPUFeatureName,
            device_features,
            wgpu.formatFeature,
        ),
    });

    const queue = c.wgpuDeviceGetQueue(device);
    errdefer c.wgpuQueueRelease(queue);

    const per_frame_bind_group_layout = c.wgpuDeviceCreateBindGroupLayout(device, &.{
        .entryCount = 1,
        .entries = &.{
            .binding = 0,
            .visibility = c.WGPUShaderStage_Vertex | c.WGPUShaderStage_Fragment,
            .buffer = .{
                .type = c.WGPUBufferBindingType_Uniform,
                .hasDynamicOffset = @intFromBool(false),
                .minBindingSize = @sizeOf(FrameData.UniformData),
            },
        },
    });
    errdefer c.wgpuBindGroupLayoutRelease(per_frame_bind_group_layout);
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
    errdefer c.wgpuBindGroupLayoutRelease(sampler_bind_group_layout);
    const per_model_bind_group_layout = c.wgpuDeviceCreateBindGroupLayout(device, &.{
        .entryCount = 1,
        .entries = &.{
            .binding = 0,
            .visibility = c.WGPUShaderStage_Vertex,
            .buffer = .{
                .type = c.WGPUBufferBindingType_Uniform,
                .hasDynamicOffset = @intFromBool(false),
                .minBindingSize = @sizeOf(Primitive.UniformData),
            },
        },
    });
    errdefer c.wgpuBindGroupLayoutRelease(per_model_bind_group_layout);
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
    };
    const texture_bind_group_layout = c.wgpuDeviceCreateBindGroupLayout(device, &.{
        .entryCount = @intCast(texture_bind_group_layout_entries.len),
        .entries = texture_bind_group_layout_entries.ptr,
    });
    errdefer c.wgpuBindGroupLayoutRelease(texture_bind_group_layout);

    const bind_group_layouts = std.EnumArray(BindGroup, c.WGPUBindGroupLayout).init(.{
        .frame = per_frame_bind_group_layout,
        .sampler = sampler_bind_group_layout,
        .texture = texture_bind_group_layout,
        .model = per_model_bind_group_layout,
    });
    const pipeline_layout = c.wgpuDeviceCreatePipelineLayout(device, &.{
        .bindGroupLayoutCount = bind_group_layouts.values.len,
        .bindGroupLayouts = &bind_group_layouts.values,
    });
    defer c.wgpuPipelineLayoutRelease(pipeline_layout);

    const vertex_shader_module = try wgpu.deviceCreateShaderModuleSPIRV(
        device,
        std.mem.bytesAsSlice(u32, options.vertex_shader_bytecode),
    );
    defer c.wgpuShaderModuleRelease(vertex_shader_module);

    const fragment_shader_module = try wgpu.deviceCreateShaderModuleSPIRV(
        device,
        std.mem.bytesAsSlice(u32, options.fragment_shader_bytecode),
    );
    defer c.wgpuShaderModuleRelease(fragment_shader_module);

    var surface_capabilities: c.WGPUSurfaceCapabilities = .{};
    c.wgpuSurfaceGetCapabilities(surface, adapter, &surface_capabilities);
    defer c.wgpuSurfaceCapabilitiesFreeMembers(surface_capabilities);

    const surface_formats = mem.sliceFromParts(surface_capabilities.formats, surface_capabilities.formatCount);
    if (surface_formats.len == 0) return error.NoSurfaceFormatsAvailable;
    std.mem.sortUnstable(c.WGPUTextureFormat, surface_formats, {}, std.sort.asc(c.WGPUTextureFormat));
    log.debug("surface formats: [{s}]", .{
        fmt.fmtSliceElementFormatter(
            c.WGPUTextureFormat,
            surface_formats,
            wgpu.formatTextureFormat,
        ),
    });

    const present_modes = mem.sliceFromParts(surface_capabilities.presentModes, surface_capabilities.presentModeCount);
    if (present_modes.len == 0) return error.NoPresentModesAvailable;
    std.mem.sortUnstable(c.WGPUPresentMode, present_modes, {}, std.sort.asc(c.WGPUPresentMode));
    log.debug("present modes: [{s}]", .{
        fmt.fmtSliceElementFormatter(
            c.WGPUPresentMode,
            present_modes,
            wgpu.formatPresentMode,
        ),
    });

    // TODO: Differentiate between HDR and SDR formats.
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
    log.debug("selected surface format: {s}", .{wgpu.textureFormatName(surface_format)});

    // TODO: Differentiate between VSync and non-VSync modes.
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
    log.debug("selected present mode: {s}", .{wgpu.presentModeName(present_mode)});

    const vertex_attributes = wgpu.vertexAttributesFromType(Vertex, .{});

    // Must declare type otherwise compilation fails on macOS with:
    //      error: expected type '[*c]const c_uint', found '*const c_int'
    //          .viewFormats = &depth_format,
    const depth_format: c.WGPUTextureFormat = c.WGPUTextureFormat_Depth32Float;

    const pipeline = c.wgpuDeviceCreateRenderPipeline(device, &.{
        .layout = pipeline_layout,
        .vertex = .{
            .module = vertex_shader_module,
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
            .module = fragment_shader_module,
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

    const framebuffer_extent = glfw.getFramebufferSize(options.window);

    const frame_data: FrameData = frame_data: {
        const znear = 0.01;
        const uniform_data: FrameData.UniformData = .{
            .view = math.mat4Identity(),
            .proj = math.perspectiveInverseDepth(
                std.math.degreesToRadians(80.0),
                framebuffer_extent.aspectRatio(),
                znear,
            ),
            .camera_position = math.vec3Zero(),
        };
        const uniform_buffer = c.wgpuDeviceCreateBuffer(device, &.{
            .usage = c.WGPUBufferUsage_Uniform | c.WGPUBufferUsage_CopyDst,
            .size = @sizeOf(@TypeOf(uniform_data)),
        });
        errdefer c.wgpuBufferRelease(uniform_buffer);
        errdefer c.wgpuBufferDestroy(uniform_buffer);
        c.wgpuQueueWriteBuffer(queue, uniform_buffer, 0, &uniform_data, @sizeOf(@TypeOf(uniform_data)));

        const uniform_bind_group = c.wgpuDeviceCreateBindGroup(device, &.{
            .layout = per_frame_bind_group_layout,
            .entryCount = 1,
            .entries = &.{
                .binding = 0,
                .buffer = uniform_buffer,
                .offset = 0,
                .size = @sizeOf(@TypeOf(uniform_data)),
            },
        });
        errdefer c.wgpuBindGroupRelease(uniform_bind_group);

        break :frame_data .{
            .uniform_data = uniform_data,
            .uniform_buffer = uniform_buffer,
            .uniform_bind_group = uniform_bind_group,
        };
    };
    errdefer frame_data.deinit();

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

    const linear_sampler_bind_group = c.wgpuDeviceCreateBindGroup(device, &.{
        .layout = sampler_bind_group_layout,
        .entryCount = 1,
        .entries = &.{
            .binding = 0,
            .sampler = linear_sampler,
        },
    });
    errdefer c.wgpuBindGroupRelease(linear_sampler_bind_group);

    const swapchain = createSwapchain(
        device,
        framebuffer_extent,
        surface,
        present_mode,
        surface_format,
        depth_format,
    );
    errdefer swapchain.deinit();

    const imgui_context = c.ImGui_CreateContext(null) orelse return error.ImGuiCreateContextFailed;
    errdefer c.ImGui_DestroyContext(imgui_context);
    if (!c.ImGuiBackendInitialize(&.{
        .window = options.window,
        .device = device,
        .renderTargetFormat = surface_format,
        .depthStencilFormat = depth_format,
    })) {
        return error.ImGuiBackendInitializeFailed;
    }
    errdefer c.ImGuiBackendTerminate();

    const fallback_texture = try loadTextureFromMemory(device, queue, resources.textures.fallback);
    errdefer c.wgpuTextureViewRelease(fallback_texture);
    var textures = std.StringArrayHashMapUnmanaged(c.WGPUTextureView){};
    errdefer textures.deinit(allocator);
    const fallback_texture_path = try allocator.dupe(u8, "builtin://fallback-texture");
    errdefer allocator.free(fallback_texture_path);
    try textures.put(allocator, fallback_texture_path, fallback_texture);

    const fallback_texture_bind_group = c.wgpuDeviceCreateBindGroup(device, &.{
        .layout = bind_group_layouts.get(.texture),
        .entryCount = 1,
        .entries = &.{
            .binding = 0,
            .textureView = fallback_texture,
        },
    });

    return .{
        .allocator = allocator,

        .device = device,
        .queue = queue,

        .window = options.window,
        .surface = surface,
        .surface_format = surface_format,
        .present_mode = present_mode,
        .swapchain = swapchain,

        .depth_format = depth_format,

        .bind_group_layouts = bind_group_layouts,
        .pipeline = pipeline,

        .imgui_context = imgui_context,

        .frame_data = frame_data,

        .linear_sampler_bind_group = linear_sampler_bind_group,
        .fallback_texture_bind_group = fallback_texture_bind_group,
        .textures = textures,
    };
}

pub fn deinit(self: Renderer) void {
    {
        for (self.models.items) |model| model.deinit(self.allocator);
        var models = self.models;
        models.deinit(self.allocator);
    }

    {
        var iterator = self.textures.iterator();
        while (iterator.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            c.wgpuTextureViewRelease(entry.value_ptr.*);
        }
        var textures = self.textures;
        textures.deinit(self.allocator);
    }
    c.wgpuBindGroupRelease(self.linear_sampler_bind_group);

    c.ImGuiBackendTerminate();
    c.ImGui_DestroyContext(self.imgui_context);

    self.frame_data.deinit();

    c.wgpuRenderPipelineRelease(self.pipeline);
    for (self.bind_group_layouts.values) |layout| {
        c.wgpuBindGroupLayoutRelease(layout);
    }

    self.swapchain.deinit();
    c.wgpuSurfaceUnconfigure(self.surface);
    c.wgpuSurfaceRelease(self.surface);

    c.wgpuQueueRelease(self.queue);
    c.wgpuDeviceRelease(self.device);
}

pub fn recreateSwapchain(self: *Renderer, extent: math.Extent2D) void {
    // Need to defer recreation until extent is non-zero.
    if (extent.width == 0.0 and extent.height == 0.0) {
        self.should_recreate_swapchain = true;
        return;
    }

    self.should_recreate_swapchain = false;
    self.swapchain.deinit();
    // TODO: This should use the same logic as in init.
    self.present_mode = c.WGPUPresentMode_Immediate;
    if (self.vsync) {
        self.present_mode = c.WGPUPresentMode_Fifo;
    }
    const swapchain = createSwapchain(
        self.device,
        extent,
        self.surface,
        self.present_mode,
        self.surface_format,
        self.depth_format,
    );
    errdefer swapchain.deinit();
    self.swapchain = swapchain;

    self.frame_data.uniform_data.proj = math.perspectiveInverseDepth(
        std.math.degreesToRadians(80.0),
        extent.aspectRatio(),
        0.01,
    );
}

pub fn loadModel(
    self: *Renderer,
    allocator: std.mem.Allocator,
    dir: std.fs.Dir,
    path: []const u8,
    post_transform: math.Mat4,
) !Model {
    const blob = try dir.readFileAllocOptions(
        allocator,
        path,
        512 * 1024 * 1024,
        null,
        @alignOf(u32),
        null,
    );
    defer allocator.free(blob);

    var gltf = Gltf.init(allocator);
    defer gltf.deinit();
    try gltf.parse(blob);
    log.debug("{s}: glTF version {s}, generated by {?s}", .{
        path,
        gltf.data.asset.version,
        gltf.data.asset.generator,
    });

    try self.textures.ensureUnusedCapacity(self.allocator, gltf.data.images.items.len);
    for (gltf.data.images.items) |image| {
        if (image.uri) |uri| {
            log.debug("image: {s}", .{uri});
            const texture = try loadTexture(allocator, self.device, self.queue, dir, uri);
            errdefer c.wgpuTextureViewRelease(texture);
            const uri_owned = try self.allocator.dupe(u8, uri);
            errdefer self.allocator.free(uri_owned);
            self.textures.putAssumeCapacity(uri_owned, texture);
        }
    }

    var materials: std.ArrayListUnmanaged(Material) = .{};
    try materials.ensureTotalCapacityPrecise(self.allocator, gltf.data.materials.items.len);
    errdefer materials.deinit(self.allocator);
    errdefer for (materials.items) |material| material.deinit();
    for (gltf.data.materials.items) |material| {
        log.debug("{s}:\n" ++
            "\tdouble sided: {}\n" ++
            "\tbase color: {d}\n" ++
            "\tmetallic factor: {d}\n" ++
            "\troughness factor: {d}\n" ++
            "\ttexture source: {?s}", .{
            material.name,
            material.is_double_sided,
            material.metallic_roughness.base_color_factor,
            material.metallic_roughness.metallic_factor,
            material.metallic_roughness.roughness_factor,
            source: {
                const texture = material.metallic_roughness.base_color_texture orelse break :source null;
                const image_index = gltf.data.textures.items[texture.index].source orelse break :source null;
                break :source gltf.data.images.items[image_index].uri;
            },
        });

        const texture_bind_group = blk: {
            if (material.metallic_roughness.base_color_texture) |texture| {
                const image_index = gltf.data.textures.items[texture.index].source.?;
                const uri = gltf.data.images.items[image_index].uri.?;
                if (self.textures.get(uri)) |texture_view| {
                    break :blk c.wgpuDeviceCreateBindGroup(self.device, &.{
                        .layout = self.bind_group_layouts.get(.texture),
                        .entryCount = 1,
                        .entries = &.{
                            .binding = 0,
                            .textureView = texture_view,
                        },
                    });
                } else {
                    log.err("failed to find texture: {s} for material: {s}", .{
                        uri,
                        material.name,
                    });
                }
            }

            break :blk self.fallback_texture_bind_group;
        };

        materials.appendAssumeCapacity(.{
            .texture_bind_group = texture_bind_group,
        });
    }

    const scene_index = gltf.data.scene orelse return error.DefaultSceneMissing;
    const scene = gltf.data.scenes.items[scene_index];
    log.debug("default scene: {s}", .{scene.name});
    const nodes = scene.nodes orelse return error.TopLevelNodesMissing;

    var meshes = std.ArrayListUnmanaged(Mesh){};
    try meshes.ensureTotalCapacityPrecise(self.allocator, gltf.data.meshes.items.len);
    errdefer meshes.deinit(self.allocator);
    try self.loadNodes(gltf, nodes.items, post_transform, &meshes);

    const model: Model = .{
        .meshes = try meshes.toOwnedSlice(self.allocator),
        .materials = try materials.toOwnedSlice(self.allocator),
    };

    try self.models.append(self.allocator, model);

    return model;
}

fn loadNodes(
    self: *Renderer,
    gltf: Gltf,
    nodes: []const usize,
    post_transform: math.Mat4,
    meshes: *std.ArrayListUnmanaged(Mesh),
) !void {
    for (nodes) |node_index| {
        const node = gltf.data.nodes.items[node_index];
        log.debug("node: {s}", .{node.name});

        // https://microsoft.github.io/mixed-reality-extension-sdk/gltf-gen/interfaces/gltf.node.html:
        // A node can have _either_ a matrix or any combination of
        // translation/rotation/scale (TRS) properties.
        var transform: math.Mat4 = math.mat4Identity();
        // 1. Apply node transforms (which are in glTF space).
        if (node.matrix) |matrix| {
            transform = @bitCast(matrix);
        } else {
            transform = math.translate(transform, node.translation);
            transform = math.rotateQuat(transform, node.rotation);
            transform = math.scale(transform, node.scale);
        }
        // 2. Convert to world space.
        transform = math.mat4Mul(transform, model_to_world_transform);
        // 3. Apply custom transform.
        transform = math.mat4Mul(transform, post_transform);

        const model_matrix = transform;
        const normal_matrix = math.mat4x3Truncating(math.mat4Transpose(math.mat4Inverse(model_matrix)));

        const uniform_data: Primitive.UniformData = .{
            .model = model_matrix,
            .normal = normal_matrix,
        };

        const mesh_index = node.mesh orelse return error.MeshMissing;
        const gltf_mesh = gltf.data.meshes.items[mesh_index];
        log.debug("mesh: {s}", .{gltf_mesh.name});

        var primitives: std.ArrayListUnmanaged(Primitive) = .{};
        try primitives.ensureTotalCapacityPrecise(self.allocator, gltf_mesh.primitives.items.len);
        errdefer primitives.deinit(self.allocator);
        errdefer for (primitives.items) |primitive| primitive.deinit();
        for (gltf_mesh.primitives.items, 0..) |primitive, i| {
            const uniform_buffer = c.wgpuDeviceCreateBuffer(self.device, &.{
                .usage = c.WGPUBufferUsage_Uniform | c.WGPUBufferUsage_CopyDst,
                .size = @sizeOf(@TypeOf(uniform_data)),
            });
            errdefer c.wgpuBufferRelease(uniform_buffer);
            c.wgpuQueueWriteBuffer(
                self.queue,
                uniform_buffer,
                0,
                &uniform_data,
                @sizeOf(@TypeOf(uniform_data)),
            );

            const uniform_bind_group = c.wgpuDeviceCreateBindGroup(self.device, &.{
                .layout = self.bind_group_layouts.get(.model),
                .entryCount = 1,
                .entries = &.{
                    .binding = 0,
                    .buffer = uniform_buffer,
                    .offset = 0,
                    .size = @sizeOf(@TypeOf(uniform_data)),
                },
            });
            errdefer c.wgpuBindGroupRelease(uniform_bind_group);

            const material_index = primitive.material;
            if (material_index) |index| {
                log.debug("primitive {}: {s}", .{ i, gltf.data.materials.items[index].name });
            } else {
                log.warn("primitive {} has no material", .{i});
            }

            var maybe_position_accessor: ?Gltf.Accessor = null;
            var maybe_normal_accessor: ?Gltf.Accessor = null;
            var maybe_texcoord_accessor: ?Gltf.Accessor = null;
            for (primitive.attributes.items) |attribute| {
                switch (attribute) {
                    .position => |index| {
                        maybe_position_accessor = gltf.data.accessors.items[index];
                    },
                    .normal => |index| {
                        maybe_normal_accessor = gltf.data.accessors.items[index];
                    },
                    .texcoord => |index| {
                        maybe_texcoord_accessor = gltf.data.accessors.items[index];
                    },
                    else => {},
                }
            }

            const position_accessor = maybe_position_accessor orelse {
                log.err("primitive {} is missing vertex attribute: POSITION", .{i});
                continue;
            };
            const normal_accessor = maybe_normal_accessor orelse {
                log.err("primitive {} is missing vertex attribute: NORMAL", .{i});
                continue;
            };
            const texcoord_accessor = maybe_texcoord_accessor orelse {
                log.err("primitive {} is missing vertex attribute: TEXCOORD0", .{i});
                continue;
            };

            std.debug.assert(position_accessor.component_type == .float);
            std.debug.assert(normal_accessor.component_type == .float);
            std.debug.assert(texcoord_accessor.component_type == .float);

            var vertices = std.ArrayListUnmanaged(Vertex){};
            defer vertices.deinit(self.allocator);
            {
                var position_iterator = position_accessor.iterator(f32, gltf, gltf.glb_binary.?);
                var normal_iterator = normal_accessor.iterator(f32, gltf, gltf.glb_binary.?);
                var texcoord_iterator = texcoord_accessor.iterator(f32, gltf, gltf.glb_binary.?);
                while (position_iterator.next()) |position| {
                    const normal = normal_iterator.next().?;
                    const texcoord = texcoord_iterator.next().?;
                    try vertices.append(self.allocator, .{
                        .position = position[0..3].*,
                        .normal = normal[0..3].*,
                        .uv = texcoord[0..2].*,
                    });
                }
            }

            var indices = std.ArrayListUnmanaged(u32){};
            defer indices.deinit(self.allocator);
            {
                const index_accessor_index = primitive.indices orelse continue;
                const index_accessor = gltf.data.accessors.items[index_accessor_index];
                std.debug.assert(index_accessor.component_type == .unsigned_short);
                var index_iterator = index_accessor.iterator(u16, gltf, gltf.glb_binary.?);
                while (index_iterator.next()) |index| {
                    try indices.append(self.allocator, index[0]);
                }
            }

            const vertex_buffer_size = mem.sizeOfElements(vertices.items);
            const vertex_buffer = c.wgpuDeviceCreateBuffer(self.device, &.{
                .usage = c.WGPUBufferUsage_Vertex | c.WGPUBufferUsage_CopyDst,
                .size = vertex_buffer_size,
            });
            c.wgpuQueueWriteBuffer(self.queue, vertex_buffer, 0, vertices.items.ptr, vertex_buffer_size);

            const index_buffer_size = mem.sizeOfElements(indices.items);
            const index_count = indices.items.len;
            const index_buffer = c.wgpuDeviceCreateBuffer(self.device, &.{
                .usage = c.WGPUBufferUsage_Index | c.WGPUBufferUsage_CopyDst,
                .size = index_buffer_size,
            });
            c.wgpuQueueWriteBuffer(self.queue, index_buffer, 0, indices.items.ptr, index_buffer_size);

            primitives.appendAssumeCapacity(.{
                .material = material_index,
                .uniform_data = uniform_data,
                .uniform_buffer = uniform_buffer,
                .uniform_bind_group = uniform_bind_group,
                .vertex_buffer = vertex_buffer,
                .vertex_buffer_size = vertex_buffer_size,
                .index_buffer = index_buffer,
                .index_buffer_size = index_buffer_size,
                .index_count = index_count,
            });
        }

        meshes.appendAssumeCapacity(.{
            .primitives = try primitives.toOwnedSlice(self.allocator),
        });

        try self.loadNodes(gltf, node.children.items, post_transform, meshes);
    }
}

pub fn renderFrame(self: *Renderer, delta_time: f32, camera: Camera, models: []const Model) void {
    if (self.should_recreate_swapchain) {
        const extent = glfw.getFramebufferSize(self.window);
        self.recreateSwapchain(extent);
        // If recreation failed, then extent was still 0, and this flag will still be set.
        if (self.should_recreate_swapchain) {
            return;
        }
    }

    const camera_matrices = camera.computeMatrices();
    self.frame_data.uniform_data.view = camera_matrices.view;
    self.frame_data.uniform_data.camera_position = camera.position;
    self.frame_data.uploadToGpu(self.queue);

    c.ImGuiBackendBeginFrame();
    c.ImGui_NewFrame();

    renderStatistics(delta_time);
    if (c.ImGui_Begin("Settings", &self.settings_open, c.ImGuiWindowFlags_None)) {
        if (c.ImGui_Checkbox("VSync", &self.vsync)) {
            self.recreateSwapchain(self.swapchain.extent);
        }
    }
    c.ImGui_End();

    c.ImGui_Render();

    const view = wgpu.surfaceGetNextTextureView(
        self.surface,
        self.surface_format,
    ) catch |err| switch (err) {
        error.SurfaceTextureOutOfDate,
        error.SurfaceTextureSuboptimal,
        => {
            const extent = glfw.getFramebufferSize(self.window);
            self.recreateSwapchain(extent);
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
            .view = self.swapchain.depth_texture_view,
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
        @intFromEnum(BindGroup.frame),
        self.frame_data.uniform_bind_group,
        0,
        null,
    );
    c.wgpuRenderPassEncoderSetBindGroup(
        render_pass,
        @intFromEnum(BindGroup.sampler),
        self.linear_sampler_bind_group,
        0,
        null,
    );
    for (models) |model| {
        for (model.meshes) |mesh| {
            for (mesh.primitives) |primitive| {
                var texture_bind_group = self.fallback_texture_bind_group;
                if (primitive.material) |material_index| {
                    texture_bind_group = model.materials[material_index].texture_bind_group;
                }
                c.wgpuRenderPassEncoderSetBindGroup(
                    render_pass,
                    @intFromEnum(BindGroup.texture),
                    texture_bind_group,
                    0,
                    null,
                );
                c.wgpuRenderPassEncoderSetBindGroup(
                    render_pass,
                    @intFromEnum(BindGroup.model),
                    primitive.uniform_bind_group,
                    0,
                    null,
                );
                c.wgpuRenderPassEncoderSetVertexBuffer(
                    render_pass,
                    0,
                    primitive.vertex_buffer,
                    0,
                    primitive.vertex_buffer_size,
                );
                c.wgpuRenderPassEncoderSetIndexBuffer(
                    render_pass,
                    primitive.index_buffer,
                    c.WGPUIndexFormat_Uint32,
                    0,
                    primitive.index_buffer_size,
                );
                c.wgpuRenderPassEncoderDrawIndexed(render_pass, @intCast(primitive.index_count), 1, 0, 0, 0);
            }
        }
    }
    c.ImGuiBackendDraw(render_pass);
    c.wgpuRenderPassEncoderEnd(render_pass);

    const command_buffer = c.wgpuCommandEncoderFinish(encoder, &.{});
    defer c.wgpuCommandBufferRelease(command_buffer);
    c.wgpuQueueSubmit(self.queue, 1, &command_buffer);

    c.wgpuSurfacePresent(self.surface);
    _ = c.wgpuDevicePoll(self.device, @intFromBool(false), null);
}

fn createSwapchain(
    device: c.WGPUDevice,
    extent: math.Extent2D,
    surface: c.WGPUSurface,
    present_mode: c.WGPUPresentMode,
    surface_format: c.WGPUTextureFormat,
    depth_format: c.WGPUTextureFormat,
) Swapchain {
    c.wgpuSurfaceConfigure(surface, &.{
        .device = device,
        .format = surface_format,
        .usage = c.WGPUTextureUsage_RenderAttachment,
        .alphaMode = c.WGPUCompositeAlphaMode_Auto,
        .width = extent.width,
        .height = extent.height,
        .presentMode = present_mode,
    });

    const depth_texture = c.wgpuDeviceCreateTexture(device, &.{
        .usage = c.WGPUTextureUsage_RenderAttachment,
        .dimension = c.WGPUTextureDimension_2D,
        .size = .{
            .width = extent.width,
            .height = extent.height,
            .depthOrArrayLayers = 1,
        },
        .format = depth_format,
        .mipLevelCount = 1,
        .sampleCount = 1,
        .viewFormatCount = 1,
        .viewFormats = &depth_format,
    });
    errdefer c.wgpuTextureRelease(depth_texture);
    errdefer c.wgpuTextureDestroy(depth_texture);
    const depth_texture_view = c.wgpuTextureCreateView(depth_texture, &.{
        .format = depth_format,
        .dimension = c.WGPUTextureViewDimension_2D,
        .baseMipLevel = 0,
        .mipLevelCount = 1,
        .baseArrayLayer = 0,
        .arrayLayerCount = 1,
        .aspect = c.WGPUTextureAspect_DepthOnly,
    });
    errdefer c.wgpuTextureViewRelease(depth_texture_view);

    return .{
        .extent = extent,
        .depth_texture = depth_texture,
        .depth_texture_view = depth_texture_view,
    };
}

fn loadTexture(
    allocator: std.mem.Allocator,
    device: c.WGPUDevice,
    queue: c.WGPUQueue,
    dir: std.fs.Dir,
    path: []const u8,
) !c.WGPUTextureView {
    const blob = try dir.readFileAlloc(allocator, path, 4 * 1024 * 1024 * 1024);
    defer allocator.free(blob);
    return loadTextureFromMemory(device, queue, blob);
}

fn loadTextureFromMemory(device: c.WGPUDevice, queue: c.WGPUQueue, data: []const u8) !c.WGPUTextureView {
    var ktx_texture: ?*c.ktxTexture2 = null;
    if (c.ktxTexture2_CreateFromMemory(
        data.ptr,
        @intCast(data.len),
        c.KTX_TEXTURE_CREATE_NO_FLAGS,
        &ktx_texture,
    ) != c.KTX_SUCCESS) {
        return error.CreateKtxTextureFailed;
    }
    defer c.ktxTexture2_Destroy(ktx_texture);

    const texture, const texture_format = try wgpu.deviceLoadTexture(
        device,
        queue,
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

fn onDeviceLost(
    reason: c.WGPUDeviceLostReason,
    maybe_message: ?[*:0]const u8,
    user_data: ?*anyopaque,
) callconv(.C) void {
    _ = user_data;

    const format = "device lost: reason: {s}";
    const args = .{wgpu.deviceLostReasonName(reason)};

    log.err(format, args);
    if (maybe_message) |message| {
        log.err("{s}", .{message});
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
    const args = .{wgpu.errorTypeName(@"type")};

    log.err(format, args);
    if (maybe_message) |message| {
        log.err("{s}", .{message});
    }

    if (builtin.mode == .Debug) {
        @breakpoint();
    }
}
