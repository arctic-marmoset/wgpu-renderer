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

pipeline: c.WGPURenderPipeline,
mvp_bind_group: c.WGPUBindGroup,
texture_bind_group: c.WGPUBindGroup,

depth_format: c.WGPUTextureFormat,
depth_texture: c.WGPUTexture,
depth_texture_view: c.WGPUTextureView,

imgui_context: *c.ImGuiContext,
imgui_io: *c.ImGuiIO,
last_instant: std.time.Instant,

uniform: Uniform,
uniform_buffer: c.WGPUBuffer,
vbo: c.WGPUBuffer,
vbo_content_size: usize,
ibo: c.WGPUBuffer,
ibo_content_size: usize,
index_count: usize,

camera: Camera,

mouse_captured: bool,
last_mouse_position: c.vec2,

const Uniform = extern struct {
    model: c.mat4 align(32),
    view: c.mat4 align(32),
    proj: c.mat4 align(32),
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
    const mouse_position: c.vec2 = blk: {
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

    const mvp_bind_group_layout = c.wgpuDeviceCreateBindGroupLayout(device, &.{
        .entryCount = 1,
        .entries = &.{
            .binding = 0,
            .visibility = c.WGPUShaderStage_Vertex,
            .buffer = .{
                .type = c.WGPUBufferBindingType_Uniform,
                .hasDynamicOffset = @intFromBool(false),
                .minBindingSize = @sizeOf(Uniform),
            },
        },
    });
    defer c.wgpuBindGroupLayoutRelease(mvp_bind_group_layout);
    const texture_bind_group_layout_entries: []const c.WGPUBindGroupLayoutEntry = &.{
        .{
            .binding = 0,
            .visibility = c.WGPUShaderStage_Fragment,
            .sampler = .{
                .type = c.WGPUSamplerBindingType_Filtering,
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
    defer c.wgpuBindGroupLayoutRelease(texture_bind_group_layout);

    const bind_group_layouts: []const c.WGPUBindGroupLayout = &.{
        mvp_bind_group_layout,
        texture_bind_group_layout,
    };
    const pipeline_layout = c.wgpuDeviceCreatePipelineLayout(device, &.{
        .bindGroupLayoutCount = @intCast(bind_group_layouts.len),
        .bindGroupLayouts = bind_group_layouts.ptr,
    });
    defer c.wgpuPipelineLayoutRelease(pipeline_layout);

    const vs_module = try wgpu.deviceCreateShaderModuleSPIRV(
        device,
        allocator,
        data_dir,
        "shaders/basic.vert.spv",
    );
    defer c.wgpuShaderModuleRelease(vs_module);

    const fs_module = try wgpu.deviceCreateShaderModuleSPIRV(
        device,
        allocator,
        data_dir,
        "shaders/basic.frag.spv",
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

    const Vertex = struct { position: c.vec3, uv: c.vec2 };
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
            // I want my models to be glTF format, which uses RH (-X right,
            // +Y up) but WebGPU expects LH. Converting from RH to LH causes the
            // axes to flip, which also flips the winding order from glTF's CCW
            // to CW.
            .frontFace = c.WGPUFrontFace_CW,
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

    // TODO: Do tonemapping so HDR textures look ok with SDR display.
    const model_texture_data = try data_dir.readFileAlloc(
        allocator,
        "textures/stanford_dragon_base_bc7.ktx2",
        512 * 1024 * 1024,
    );
    defer allocator.free(model_texture_data);
    var model_ktx_texture: ?*c.ktxTexture2 = null;
    if (c.ktxTexture2_CreateFromMemory(
        model_texture_data.ptr,
        @intCast(model_texture_data.len),
        c.KTX_TEXTURE_CREATE_NO_FLAGS,
        &model_ktx_texture,
    ) != c.KTX_SUCCESS) {
        return error.CreateKtxTextureFailed;
    }
    const model_texture, const model_texture_format = try wgpu.deviceLoadTexture(
        device,
        queue,
        model_ktx_texture.?,
    );
    const model_texture_view = c.wgpuTextureCreateView(model_texture, &.{
        .format = model_texture_format,
        .dimension = c.WGPUTextureViewDimension_2D,
        .baseMipLevel = 0,
        .mipLevelCount = model_ktx_texture.?.numLevels,
        .baseArrayLayer = 0,
        .arrayLayerCount = model_ktx_texture.?.numLayers,
        .aspect = c.WGPUTextureAspect_All,
    });
    defer c.wgpuTextureViewRelease(model_texture_view);

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

    const framebuffer_size = glfw.getFramebufferSize(window);
    var camera = Camera.init(.{ 0.0, 0.0, -5.0 }, math.world_forward);
    var uniform: Uniform = undefined;
    // This model matrix converts from the glTF coordinate system to our world
    // space convention. It *must* be applied to all objects. The contents of
    // the matrix are defined as follows:
    // [ -1,  0, 0, 0 ] The glTF +X direction corresponds to our -X direction.
    // [  0, -1, 0, 0 ] The glTF +Y direction corresponds to our -Y direction.
    // [  0,  0, 1, 0 ] The glTF +Z direction is the same as our +Z direction.
    // [  0,  0, 0, 1 ]
    // TODO: Figure out where to put this. Should we just init all model
    // matrices like this? Or do we apply this in a post-processing step?
    // TODO: Any way to compute these (at comptime), based on the glTF axes and
    // our worldspace axes?
    // zig fmt: off
    uniform.model = .{
        .{ -1.0,  0.0, 0.0, 0.0 },
        .{  0.0, -1.0, 0.0, 0.0 },
        .{  0.0,  0.0, 1.0, 0.0 },
        .{  0.0,  0.0, 0.0, 1.0 },
    };
    // zig fmt: on
    // The dragon is kind of small. Scale it up.
    const scale = 10.0;
    var isotropic_scale: c.vec3 = undefined;
    c.glm_vec3_fill(&isotropic_scale, scale);
    c.glmc_scale(&uniform.model, &isotropic_scale);
    const aspect_ratio =
        @as(f32, @floatFromInt(framebuffer_size.width)) / @as(f32, @floatFromInt(framebuffer_size.height));
    c.glmc_mat4_copy(&camera.view, &uniform.view);
    math.perspectiveInverseDepth(std.math.degreesToRadians(80.0), aspect_ratio, 0.01, &uniform.proj);
    const uniform_buffer = c.wgpuDeviceCreateBuffer(device, &.{
        .usage = c.WGPUBufferUsage_Uniform | c.WGPUBufferUsage_CopyDst,
        .size = @sizeOf(Uniform),
    });
    errdefer c.wgpuBufferRelease(uniform_buffer);
    errdefer c.wgpuBufferDestroy(uniform_buffer);
    c.wgpuQueueWriteBuffer(queue, uniform_buffer, 0, &uniform, @sizeOf(Uniform));
    const mvp_bind_group = c.wgpuDeviceCreateBindGroup(device, &.{
        .layout = mvp_bind_group_layout,
        .entryCount = 1,
        .entries = &.{
            .binding = 0,
            .buffer = uniform_buffer,
            .offset = 0,
            .size = @sizeOf(Uniform),
        },
    });
    errdefer c.wgpuBindGroupRelease(mvp_bind_group);
    const texture_bind_group_entries: []const c.WGPUBindGroupEntry = &.{
        .{
            .binding = 0,
            .sampler = linear_sampler,
        },
        .{
            .binding = 1,
            .textureView = model_texture_view,
        },
    };
    const texture_bind_group = c.wgpuDeviceCreateBindGroup(device, &.{
        .layout = texture_bind_group_layout,
        .entryCount = @intCast(texture_bind_group_entries.len),
        .entries = texture_bind_group_entries.ptr,
    });
    errdefer c.wgpuBindGroupRelease(texture_bind_group);

    // TODO: Maybe factor this out into a separate function.
    const model_data = try data_dir.readFileAllocOptions(
        allocator,
        "meshes/stanford_dragon.glb",
        4 * 1024 * 1024,
        null,
        @alignOf(u32),
        null,
    );
    defer allocator.free(model_data);
    var gltf_model = Gltf.init(allocator);
    defer gltf_model.deinit();
    try gltf_model.parse(model_data);
    const gltf_scene_index = gltf_model.data.scene.?;
    const gltf_scene = gltf_model.data.scenes.items[gltf_scene_index];
    const gltf_node_index = gltf_scene.nodes.?.items[0];
    const gltf_node = gltf_model.data.nodes.items[gltf_node_index];
    const gltf_mesh_index = gltf_node.mesh.?;
    const gltf_mesh = gltf_model.data.meshes.items[gltf_mesh_index];
    var vertex_count: usize = 0;
    var index_count: usize = 0;
    for (gltf_mesh.primitives.items) |primitive| {
        index_count += @intCast(gltf_model.data.accessors.items[primitive.indices.?].count);
        for (primitive.attributes.items) |attribute| {
            switch (attribute) {
                .position => |index| {
                    vertex_count += @intCast(gltf_model.data.accessors.items[index].count);
                    break;
                },
                else => {},
            }
        }
    }
    const vertices = try allocator.alloc(Vertex, vertex_count);
    defer allocator.free(vertices);
    const indices = try allocator.alloc(u32, index_count);
    defer allocator.free(indices);
    var vertex_cursor: usize = 0;
    var index_cursor: usize = 0;
    for (gltf_mesh.primitives.items) |primitive| {
        var position_accessor: Gltf.Accessor = undefined;
        var uv_accessor: Gltf.Accessor = undefined;
        for (primitive.attributes.items) |attribute| {
            switch (attribute) {
                .position => |index| {
                    position_accessor = gltf_model.data.accessors.items[index];
                },
                .texcoord => |index| {
                    uv_accessor = gltf_model.data.accessors.items[index];
                },
                else => {},
            }
        }

        std.debug.assert(position_accessor.component_type == .float);
        std.debug.assert(uv_accessor.component_type == .float);

        var position_iter = position_accessor.iterator(f32, gltf_model, gltf_model.glb_binary.?);
        var uv_iter = uv_accessor.iterator(f32, gltf_model, gltf_model.glb_binary.?);
        while (position_iter.next()) |position| : (vertex_cursor += 1) {
            const uv = uv_iter.next().?;
            vertices[vertex_cursor] = .{
                .position = position[0..][0..3].*,
                .uv = uv[0..][0..2].*,
            };
        }

        const index_accessor = gltf_model.data.accessors.items[primitive.indices.?];
        std.debug.assert(index_accessor.component_type == .unsigned_short);
        var index_iter = index_accessor.iterator(u16, gltf_model, gltf_model.glb_binary.?);
        while (index_iter.next()) |index| : (index_cursor += 1) {
            indices[index_cursor] = index[0];
        }
    }
    const vbo_content_size = mem.sizeOfElements(vertices);
    const vbo = c.wgpuDeviceCreateBuffer(device, &.{
        .usage = c.WGPUBufferUsage_Vertex | c.WGPUBufferUsage_CopyDst,
        .size = vbo_content_size,
    });
    errdefer c.wgpuBufferRelease(vbo);
    errdefer c.wgpuBufferDestroy(vbo);
    c.wgpuQueueWriteBuffer(queue, vbo, 0, vertices.ptr, vbo_content_size);
    const ibo_content_size = mem.sizeOfElements(indices);
    const ibo = c.wgpuDeviceCreateBuffer(device, &.{
        .usage = c.WGPUBufferUsage_Index | c.WGPUBufferUsage_CopyDst,
        .size = ibo_content_size,
    });
    errdefer c.wgpuBufferRelease(ibo);
    errdefer c.wgpuBufferDestroy(ibo);
    c.wgpuQueueWriteBuffer(queue, ibo, 0, indices.ptr, ibo_content_size);

    const imgui_context = c.ImGui_CreateContext(null) orelse return error.ImGuiCreateContextFailed;
    errdefer c.ImGui_DestroyContext(imgui_context);
    if (!c.ImGuiBackendInitialize(&.{
        .window = window,
        .device = device,
        .renderTargetFormat = surface_format,
        .depthStencilFormat = depth_format,
    })) {
        return error.WGRInitializeFailed;
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

        .pipeline = pipeline,
        .mvp_bind_group = mvp_bind_group,
        .texture_bind_group = texture_bind_group,

        .depth_format = depth_format,
        // Deferred to Engine.createSwapChain.
        .depth_texture = null,
        .depth_texture_view = null,

        .imgui_context = imgui_context,
        .imgui_io = c.ImGui_GetIO(),
        .last_instant = std.time.Instant.now() catch unreachable,

        .uniform = uniform,
        .uniform_buffer = uniform_buffer,
        .vbo = vbo,
        .vbo_content_size = vbo_content_size,
        .ibo = ibo,
        .ibo_content_size = ibo_content_size,
        .index_count = index_count,

        .camera = camera,

        .mouse_captured = true,
        .last_mouse_position = mouse_position,
    };
    // TODO: Can we make it clear what resources need to be valid for createSwapChain?
    engine.createSwapChain(framebuffer_size.width, framebuffer_size.height);

    return engine;
}

pub fn deinit(self: *Engine) void {
    c.ImGuiBackendTerminate();
    c.ImGui_DestroyContext(self.imgui_context);

    c.wgpuBufferDestroy(self.uniform_buffer);
    c.wgpuBufferRelease(self.uniform_buffer);
    c.wgpuBufferDestroy(self.vbo);
    c.wgpuBufferRelease(self.vbo);
    c.wgpuBufferDestroy(self.ibo);
    c.wgpuBufferRelease(self.ibo);

    c.wgpuTextureViewRelease(self.depth_texture_view);
    c.wgpuTextureDestroy(self.depth_texture);
    c.wgpuTextureRelease(self.depth_texture);

    c.wgpuBindGroupRelease(self.mvp_bind_group);
    c.wgpuBindGroupRelease(self.texture_bind_group);

    c.wgpuRenderPipelineRelease(self.pipeline);
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

        const now = std.time.Instant.now() catch unreachable;
        const delta_time_ns = now.since(self.last_instant);
        self.last_instant = now;
        const delta_time_ns_f64: f64 = @floatFromInt(delta_time_ns);
        const delta_time_s_f64 = delta_time_ns_f64 / std.time.ns_per_s;
        const delta_time: f32 = @floatCast(delta_time_s_f64);

        self.update(delta_time);
        try self.renderFrame(delta_time);
    }
}

fn isRunning(self: *Engine) bool {
    return c.glfwWindowShouldClose(self.window) != c.GLFW_TRUE;
}

fn renderStatistics(self: *Engine, delta_time: f32) void {
    _ = self;

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

fn renderFrame(self: *Engine, delta_time: f32) !void {
    const view = wgpu.surfaceGetNextTextureView(
        self.surface,
        self.surface_format,
    ) orelse return error.GetNextTextureViewFailed;
    defer c.wgpuTextureViewRelease(view);

    c.ImGuiBackendBeginFrame();
    c.ImGui_NewFrame();

    self.renderStatistics(delta_time);

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
    c.wgpuRenderPassEncoderSetBindGroup(render_pass, 0, self.mvp_bind_group, 0, null);
    c.wgpuRenderPassEncoderSetBindGroup(render_pass, 1, self.texture_bind_group, 0, null);
    c.wgpuRenderPassEncoderSetVertexBuffer(render_pass, 0, self.vbo, 0, self.vbo_content_size);
    c.wgpuRenderPassEncoderSetIndexBuffer(render_pass, self.ibo, c.WGPUIndexFormat_Uint32, 0, self.ibo_content_size);
    c.wgpuRenderPassEncoderDrawIndexed(render_pass, @intCast(self.index_count), 1, 0, 0, 0);
    c.ImGuiBackendEndFrame(render_pass);
    c.wgpuRenderPassEncoderEnd(render_pass);

    const command_buffer = c.wgpuCommandEncoderFinish(encoder, &.{});
    defer c.wgpuCommandBufferRelease(command_buffer);
    c.wgpuQueueSubmit(self.queue, 1, &command_buffer);

    c.wgpuSurfacePresent(self.surface);
    _ = c.wgpuDevicePoll(self.device, @intFromBool(false), null);
}

fn update(self: *Engine, delta_time: f32) void {
    var mut_mat4_identity: c.mat4 align(32) = undefined;
    c.glmc_mat4_identity(&mut_mat4_identity);

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
        c.glmc_mat4_copy(&self.camera.view, &self.uniform.view);
        c.wgpuQueueWriteBuffer(self.queue, self.uniform_buffer, 0, &self.uniform, @sizeOf(Uniform));
    }
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

fn recreateSwapChain(self: *Engine, width: u32, height: u32) void {
    c.wgpuTextureViewRelease(self.depth_texture_view);
    c.wgpuTextureDestroy(self.depth_texture);
    c.wgpuTextureRelease(self.depth_texture);

    self.createSwapChain(width, height);
}

// TODO: Not sure I want to do the resizing in here.
fn onFramebufferSizeChanged(self: *Engine, width: u32, height: u32) void {
    self.recreateSwapChain(width, height);
    const aspect_ratio = @as(f32, @floatFromInt(width)) / @as(f32, @floatFromInt(height));
    math.perspectiveInverseDepth(std.math.degreesToRadians(80.0), aspect_ratio, 0.01, &self.uniform.proj);
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
    var position: c.vec2 = .{ @floatCast(x), @floatCast(y) };
    var delta: c.vec2 = undefined;
    c.glm_vec2_sub(&position, &self.last_mouse_position, &delta);
    self.last_mouse_position = position;
    // TODO: This should be in `Engine.update`.
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

fn openDataDir(allocator: std.mem.Allocator) !std.fs.Dir {
    const exe_dir_path = try std.fs.selfExeDirPathAlloc(allocator);
    defer allocator.free(exe_dir_path);
    const data_dir_path = try std.fs.path.join(allocator, &.{ exe_dir_path, "..", "data" });
    defer allocator.free(data_dir_path);
    return std.fs.openDirAbsolute(data_dir_path, .{});
}
