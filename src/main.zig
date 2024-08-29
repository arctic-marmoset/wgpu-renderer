const std = @import("std");
const builtin = @import("builtin");

const c = @import("c.zig");
const fmt = @import("fmt.zig");
const glfw = @import("glfw.zig");
const mem = @import("mem.zig");
const wgpu = @import("wgpu.zig");

const Gltf = @import("Gltf");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer std.debug.assert(gpa.deinit() == .ok);
    var data_dir = try openDataDir(gpa.allocator());
    defer data_dir.close();

    const instance = try wgpu.createInstance(.{});
    defer c.wgpuInstanceRelease(instance);

    const adapter = try wgpu.instanceRequestAdapter(instance, .{});
    defer c.wgpuAdapterRelease(adapter);

    var adapter_properties: c.WGPUAdapterProperties = .{};
    c.wgpuAdapterGetProperties(adapter, &adapter_properties);
    std.log.debug("adapter properties:", .{});
    std.log.debug("\tvendor id:\t{}", .{adapter_properties.vendorID});
    std.log.debug("\tdevice id:\t{}", .{adapter_properties.deviceID});
    std.log.debug("\tvendor name:\t{?s}", .{adapter_properties.vendorName});
    std.log.debug("\tdevice name:\t{?s}", .{adapter_properties.name});
    std.log.debug("\tarchitecture:\t{?s}", .{adapter_properties.architecture});
    std.log.debug("\tdescription:\t{?s}", .{adapter_properties.driverDescription});
    std.log.debug("\tadapter type:\t{s}", .{wgpu.adapterTypeToString(adapter_properties.adapterType)});
    std.log.debug("\tbackend type:\t{s}", .{wgpu.backendTypeToString(adapter_properties.backendType)});

    var adapter_limits: c.WGPUSupportedLimits = .{};
    if (c.wgpuAdapterGetLimits(adapter, &adapter_limits) == 0) {
        std.log.warn("failed to get adapter limits", .{});
    } else {
        std.log.debug("adapter limits:", .{});
        inline for (comptime std.meta.fieldNames(c.WGPULimits)) |field_name| {
            std.log.debug("\t" ++ field_name ++ ": {}", .{@field(adapter_limits.limits, field_name)});
        }
    }

    const adapter_features = try wgpu.adapterEnumerateFeatures(adapter, gpa.allocator());
    defer gpa.allocator().free(adapter_features);
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
    defer c.wgpuDeviceRelease(device);
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

    const device_features = try wgpu.deviceEnumerateFeatures(device, gpa.allocator());
    defer gpa.allocator().free(device_features);
    std.log.debug("device features: [{s}]", .{
        fmt.fmtSliceElementFormatter(
            c.WGPUFeatureName,
            device_features,
            wgpu.formatFeature,
        ),
    });

    const queue = c.wgpuDeviceGetQueue(device);
    defer c.wgpuQueueRelease(queue);

    if (builtin.target.os.tag == .linux) {
        if (c.glfwPlatformSupported(c.GLFW_PLATFORM_WAYLAND) == c.GLFW_TRUE) {
            c.glfwInitHint(c.GLFW_PLATFORM, c.GLFW_PLATFORM_WAYLAND);
        }
    }
    if (c.glfwInit() != c.GLFW_TRUE) return error.GlfwInitFailed;
    defer c.glfwTerminate();

    c.glfwWindowHint(c.GLFW_CLIENT_API, c.GLFW_NO_API);
    c.glfwWindowHint(c.GLFW_COCOA_RETINA_FRAMEBUFFER, c.GLFW_TRUE);
    c.glfwWindowHint(c.GLFW_RESIZABLE, c.GLFW_FALSE);
    const window = c.glfwCreateWindow(
        1280,
        720,
        "3D Renderer (WGPU)",
        null,
        null,
    ) orelse return error.CreateMainWindowFailed;
    defer c.glfwDestroyWindow(window);

    const surface = c.glfwCreateWGPUSurface(instance, window) orelse return error.CreateSurfaceFailed;
    defer c.wgpuSurfaceRelease(surface);

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

    // Don't include FIFO because we'll fall back to it anyway if none
    // of the desired modes are available; FIFO is guaranteed by the WebGPU spec
    // to be available.
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

    const framebuffer_size = glfw.getFramebufferSize(window);
    c.wgpuSurfaceConfigure(surface, &.{
        .device = device,
        .format = surface_format,
        .usage = c.WGPUTextureUsage_RenderAttachment,
        .alphaMode = c.WGPUCompositeAlphaMode_Auto,
        .width = framebuffer_size.width,
        .height = framebuffer_size.height,
        .presentMode = present_mode,
    });
    defer c.wgpuSurfaceUnconfigure(surface);

    const vs_module = try wgpu.deviceCreateShaderModuleSPIRV(
        device,
        gpa.allocator(),
        data_dir,
        "shaders/triangle.vs.spv",
    );
    defer c.wgpuShaderModuleRelease(vs_module);

    const fs_module = try wgpu.deviceCreateShaderModuleSPIRV(
        device,
        gpa.allocator(),
        data_dir,
        "shaders/triangle.fs.spv",
    );
    defer c.wgpuShaderModuleRelease(fs_module);

    const Vertex = struct { position: c.vec3, color: c.vec3 };
    const vertex_attributes = wgpu.vertexAttributesFromType(Vertex, .{});

    const pipeline = c.wgpuDeviceCreateRenderPipeline(device, &.{
        .layout = null,
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
        .depthStencil = null,
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
    defer c.wgpuRenderPipelineRelease(pipeline);

    // TODO: Maybe factor this out into a separate function.
    const triangle_data = try data_dir.readFileAllocOptions(
        gpa.allocator(),
        "meshes/triangle.glb",
        4 * 1024,
        null,
        @alignOf(u32),
        null,
    );
    defer gpa.allocator().free(triangle_data);
    var triangle_gltf = Gltf.init(gpa.allocator());
    defer triangle_gltf.deinit();
    try triangle_gltf.parse(triangle_data);
    const gltf_scene_index = triangle_gltf.data.scene.?;
    const gltf_scene = triangle_gltf.data.scenes.items[gltf_scene_index];
    const gltf_node_index = gltf_scene.nodes.?.items[0];
    const gltf_node = triangle_gltf.data.nodes.items[gltf_node_index];
    const gltf_mesh_index = gltf_node.mesh.?;
    const gltf_mesh = triangle_gltf.data.meshes.items[gltf_mesh_index];
    var vertex_count: usize = 0;
    var index_count: usize = 0;
    for (gltf_mesh.primitives.items) |primitive| {
        index_count += @intCast(triangle_gltf.data.accessors.items[primitive.indices.?].count);
        for (primitive.attributes.items) |attribute| {
            switch (attribute) {
                .position => |index| {
                    vertex_count += @intCast(triangle_gltf.data.accessors.items[index].count);
                    break;
                },
                else => {},
            }
        }
    }
    const vertices = try gpa.allocator().alloc(Vertex, vertex_count);
    defer gpa.allocator().free(vertices);
    const indices = try gpa.allocator().alloc(u32, index_count);
    defer gpa.allocator().free(indices);
    var vertex_cursor: usize = 0;
    var index_cursor: usize = 0;
    for (gltf_mesh.primitives.items) |primitive| {
        var position_accessor: Gltf.Accessor = undefined;
        var color_accessor: Gltf.Accessor = undefined;
        for (primitive.attributes.items) |attribute| {
            switch (attribute) {
                .position => |index| {
                    position_accessor = triangle_gltf.data.accessors.items[index];
                },
                .color => |index| {
                    color_accessor = triangle_gltf.data.accessors.items[index];
                },
                else => {},
            }
        }

        var position_iter = position_accessor.iterator(f32, &triangle_gltf, triangle_gltf.glb_binary.?);
        var color_iter = color_accessor.iterator(u16, &triangle_gltf, triangle_gltf.glb_binary.?);
        while (position_iter.next()) |position| : (vertex_cursor += 1) {
            const color = color_iter.next().?[0..][0..3];
            const r: f32 = @floatFromInt(color[0]);
            const g: f32 = @floatFromInt(color[1]);
            const b: f32 = @floatFromInt(color[2]);
            const range: f32 = @floatFromInt(std.math.maxInt(u16));
            vertices[vertex_cursor] = .{
                .position = position[0..][0..3].*,
                .color = .{
                    r / range,
                    g / range,
                    b / range,
                },
            };
        }

        const index_accessor = triangle_gltf.data.accessors.items[primitive.indices.?];
        std.debug.assert(index_accessor.component_type == .unsigned_short);
        var index_iter = index_accessor.iterator(u16, &triangle_gltf, triangle_gltf.glb_binary.?);
        while (index_iter.next()) |index| : (index_cursor += 1) {
            indices[index_cursor] = index[0];
        }
    }
    const vbo = c.wgpuDeviceCreateBuffer(device, &.{
        .label = "Vertex Buffer: Triangle",
        .usage = c.WGPUBufferUsage_Vertex | c.WGPUBufferUsage_CopyDst,
        .size = mem.sizeOfElements(vertices),
    });
    defer c.wgpuBufferRelease(vbo);
    defer c.wgpuBufferDestroy(vbo);
    c.wgpuQueueWriteBuffer(queue, vbo, 0, vertices.ptr, mem.sizeOfElements(vertices));

    while (c.glfwWindowShouldClose(window) != c.GLFW_TRUE) {
        c.glfwPollEvents();

        const view = wgpu.surfaceGetNextTextureView(
            surface,
            surface_format,
        ) orelse return error.GetNextTextureViewFailed;
        defer c.wgpuTextureViewRelease(view);

        const encoder = c.wgpuDeviceCreateCommandEncoder(device, &.{});
        defer c.wgpuCommandEncoderRelease(encoder);

        const render_pass = c.wgpuCommandEncoderBeginRenderPass(encoder, &.{
            .colorAttachmentCount = 1,
            .colorAttachments = &.{
                .view = view,
                .loadOp = c.WGPULoadOp_Clear,
                .storeOp = c.WGPUStoreOp_Store,
                .clearValue = wgpu.color(1.0, 0.0, 1.0, 1.0),
            },
        });
        defer c.wgpuRenderPassEncoderRelease(render_pass);
        c.wgpuRenderPassEncoderSetPipeline(render_pass, pipeline);
        c.wgpuRenderPassEncoderSetVertexBuffer(render_pass, 0, vbo, 0, mem.sizeOfElements(vertices));
        c.wgpuRenderPassEncoderDraw(render_pass, @intCast(vertices.len), 1, 0, 0);
        c.wgpuRenderPassEncoderEnd(render_pass);

        const command_buffer = c.wgpuCommandEncoderFinish(encoder, &.{});
        defer c.wgpuCommandBufferRelease(command_buffer);
        c.wgpuQueueSubmit(queue, 1, &command_buffer);

        c.wgpuSurfacePresent(surface);
        _ = c.wgpuDevicePoll(device, @intFromBool(false), null);
    }
}

fn openDataDir(allocator: std.mem.Allocator) !std.fs.Dir {
    const exe_dir_path = try std.fs.selfExeDirPathAlloc(allocator);
    defer allocator.free(exe_dir_path);
    const data_dir_path = try std.fs.path.join(allocator, &.{ exe_dir_path, "..", "data" });
    defer allocator.free(data_dir_path);
    return std.fs.openDirAbsolute(data_dir_path, .{});
}

fn onDeviceLost(
    reason: c.WGPUDeviceLostReason,
    maybe_message: ?[*:0]const u8,
    user_data: ?*anyopaque,
) callconv(.C) void {
    _ = user_data;
    std.log.err("device lost: reason: {s}", .{wgpu.deviceLostReasonToString(reason)});
    if (maybe_message) |message| {
        std.log.err("{s}", .{message});
    }
}

fn onUncapturedError(
    @"type": c.WGPUErrorType,
    maybe_message: ?[*:0]const u8,
    user_data: ?*anyopaque,
) callconv(.C) void {
    _ = user_data;
    std.log.err("uncaptured device error: type: {s}", .{wgpu.errorTypeToString(@"type")});
    if (maybe_message) |message| {
        std.log.err("{s}", .{message});
    }
}
