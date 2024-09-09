const std = @import("std");

const c = @import("c.zig");
const math = @import("math.zig");
const ktx = @import("ktx.zig");

const Nonnull = std.meta.Child;

const Instance = Nonnull(c.WGPUInstance);
const Adapter = Nonnull(c.WGPUAdapter);
const Device = Nonnull(c.WGPUDevice);

pub fn createInstance(descriptor: c.WGPUInstanceDescriptor) !Instance {
    const instance = c.wgpuCreateInstance(&descriptor) orelse
        return error.WgpuCreateInstanceFailed;

    errdefer c.wgpuInstanceRelease(instance);

    return instance;
}

pub fn instanceRequestAdapter(
    instance: Instance,
    options: c.WGPURequestAdapterOptions,
) !Adapter {
    var adapter_or_error: AdapterOrError = undefined;

    c.wgpuInstanceRequestAdapter(
        instance,
        &options,
        onRequestAdapterEnded,
        &adapter_or_error,
    );

    const adapter = try adapter_or_error;
    errdefer c.wgpuAdapterRelease(adapter);

    return adapter;
}

const AdapterOrError = error{RequestAdapterFailed}!Adapter;

fn onRequestAdapterEnded(
    status: c.WGPURequestAdapterStatus,
    adapter: c.WGPUAdapter,
    message: ?[*:0]const u8,
    user_data: ?*anyopaque,
) callconv(.C) void {
    const result: ?*AdapterOrError = @ptrCast(@alignCast(user_data));
    result.?.* = error.RequestAdapterFailed;

    if (status == c.WGPURequestAdapterStatus_Success) {
        result.?.* = adapter.?;
    } else {
        std.log.err("failed to get adapter: {s}", .{message.?});
    }
}

pub fn adapterEnumerateFeatures(
    adapter: c.WGPUAdapter,
    allocator: std.mem.Allocator,
) ![]c.WGPUFeatureName {
    const count = c.wgpuAdapterEnumerateFeatures(adapter, null);
    const features = try allocator.alloc(c.WGPUFeatureName, count);
    errdefer allocator.free(features);
    std.debug.assert(
        c.wgpuAdapterEnumerateFeatures(
            adapter,
            features.ptr,
        ) == count,
    );

    return features;
}

pub fn adapterRequestDevice(
    adapter: Adapter,
    descriptor: c.WGPUDeviceDescriptor,
) !Device {
    var device_or_error: DeviceOrError = undefined;

    c.wgpuAdapterRequestDevice(
        adapter,
        &descriptor,
        onRequestDeviceEnded,
        &device_or_error,
    );

    const device = try device_or_error;
    errdefer c.wgpuDeviceRelease(device);

    return device;
}

const DeviceOrError = error{RequestDeviceFailed}!Device;

fn onRequestDeviceEnded(
    status: c.WGPURequestDeviceStatus,
    device: c.WGPUDevice,
    message: ?[*:0]const u8,
    user_data: ?*anyopaque,
) callconv(.C) void {
    const result: ?*DeviceOrError = @ptrCast(@alignCast(user_data));
    result.?.* = error.RequestDeviceFailed;

    if (status == c.WGPURequestDeviceStatus_Success) {
        result.?.* = device.?;
    } else {
        std.log.err("failed to get device: {s}", .{message.?});
    }
}

pub fn deviceEnumerateFeatures(
    device: c.WGPUDevice,
    allocator: std.mem.Allocator,
) ![]c.WGPUFeatureName {
    const count = c.wgpuDeviceEnumerateFeatures(device, null);
    const features = try allocator.alloc(c.WGPUFeatureName, count);
    errdefer allocator.free(features);
    std.debug.assert(
        c.wgpuDeviceEnumerateFeatures(
            device,
            features.ptr,
        ) == count,
    );

    return features;
}

pub fn deviceLoadTexture(
    device: c.WGPUDevice,
    queue: c.WGPUQueue,
    ktx_texture: *c.ktxTexture2,
) !struct { c.WGPUTexture, c.WGPUTextureFormat } {
    // NOTE: Must declare type otherwise compilation fails on macOS.
    const format: c.WGPUTextureFormat = switch (ktx_texture.vkFormat) {
        c.VK_FORMAT_BC4_UNORM_BLOCK => c.WGPUTextureFormat_BC4RUnorm,
        c.VK_FORMAT_BC6H_SFLOAT_BLOCK => c.WGPUTextureFormat_BC6HRGBFloat,
        c.VK_FORMAT_BC6H_UFLOAT_BLOCK => c.WGPUTextureFormat_BC6HRGBUfloat,
        c.VK_FORMAT_BC7_SRGB_BLOCK => c.WGPUTextureFormat_BC7RGBAUnormSrgb,
        else => return error.UnsupportedFormat,
    };
    const block_size: u32 = switch (format) {
        c.WGPUTextureFormat_BC4RUnorm,
        => 8,
        c.WGPUTextureFormat_BC6HRGBFloat,
        c.WGPUTextureFormat_BC6HRGBUfloat,
        c.WGPUTextureFormat_BC7RGBAUnormSrgb,
        => 16,
        else => unreachable,
    };
    const block_width: u32 = switch (format) {
        c.WGPUTextureFormat_BC4RUnorm,
        c.WGPUTextureFormat_BC6HRGBFloat,
        c.WGPUTextureFormat_BC6HRGBUfloat,
        c.WGPUTextureFormat_BC7RGBAUnormSrgb,
        => 4,
        else => unreachable,
    };

    const view_formats = [_]c.WGPUTextureFormat{format};
    const texture = c.wgpuDeviceCreateTexture(device, &.{
        .usage = c.WGPUTextureUsage_CopyDst | c.WGPUTextureUsage_TextureBinding,
        .dimension = c.WGPUTextureDimension_2D,
        .size = .{
            .width = ktx_texture.baseWidth,
            .height = ktx_texture.baseHeight,
            .depthOrArrayLayers = 1,
        },
        .format = format,
        .mipLevelCount = ktx_texture.numLevels,
        .sampleCount = 1,
        .viewFormatCount = view_formats.len,
        .viewFormats = &view_formats,
    });
    errdefer c.wgpuTextureRelease(texture);
    errdefer c.wgpuTextureDestroy(texture);

    var callback_data = LoadTextureCallbackData{
        .queue = queue,
        .texture = texture,
        .block_size = block_size,
        .block_width = block_width,
    };
    if (ktx.textureIterateLoadLevelFaces(
        ktx_texture,
        loadTextureCompressed,
        &callback_data,
    ) != c.KTX_SUCCESS) {
        return error.LoadTextureFailed;
    }

    return .{ texture, format };
}

pub fn deviceCreateShaderModuleSPIRV(
    device: c.WGPUDevice,
    bytecode: []const u32,
) !c.WGPUShaderModule {
    const spirv_descriptor: c.WGPUShaderModuleSPIRVDescriptor = .{
        .chain = .{
            .next = null,
            .sType = c.WGPUSType_ShaderModuleSPIRVDescriptor,
        },
        .code = bytecode.ptr,
        .codeSize = @intCast(bytecode.len),
    };
    return c.wgpuDeviceCreateShaderModule(device, &.{
        .nextInChain = &spirv_descriptor.chain,
    });
}

fn vertexFormatFromType(comptime T: type, comptime normalized: bool) c.WGPUVertexFormat {
    return switch (T) {
        [2]u8 => if (normalized) c.WGPUVertexFormat_Unorm8x2 else c.WGPUVertexFormat_Uint8x2,
        [4]u8 => if (normalized) c.WGPUVertexFormat_Unorm8x4 else c.WGPUVertexFormat_Uint8x4,
        [2]i8 => if (normalized) c.WGPUVertexFormat_Snorm8x2 else c.WGPUVertexFormat_Sint8x2,
        [4]i8 => if (normalized) c.WGPUVertexFormat_Snorm8x4 else c.WGPUVertexFormat_Sint8x4,
        [2]u16 => if (normalized) c.WGPUVertexFormat_Unorm16x2 else c.WGPUVertexFormat_Uint16x2,
        [4]u16 => if (normalized) c.WGPUVertexFormat_Unorm16x4 else c.WGPUVertexFormat_Uint16x4,
        [2]i16 => if (normalized) c.WGPUVertexFormat_Snorm16x2 else c.WGPUVertexFormat_Sint16x2,
        [4]i16 => if (normalized) c.WGPUVertexFormat_Snorm16x4 else c.WGPUVertexFormat_Sint16x4,
        [2]f16 => c.WGPUVertexFormat_Float16x2,
        [4]f16 => c.WGPUVertexFormat_Float16x4,
        f32 => c.WGPUVertexFormat_Float32,
        [2]f32, math.Vec2 => c.WGPUVertexFormat_Float32x2,
        [3]f32, math.Vec3 => c.WGPUVertexFormat_Float32x3,
        [4]f32, math.Vec4 => c.WGPUVertexFormat_Float32x4,
        u32 => if (normalized) c.WGPUVertexFormat_Unorm32 else c.WGPUVertexFormat_Uint32,
        [2]u32 => if (normalized) c.WGPUVertexFormat_Unorm32x2 else c.WGPUVertexFormat_Uint32x2,
        [3]u32 => if (normalized) c.WGPUVertexFormat_Unorm32x3 else c.WGPUVertexFormat_Uint32x3,
        [4]u32 => if (normalized) c.WGPUVertexFormat_Unorm32x4 else c.WGPUVertexFormat_Uint32x4,
        i32 => if (normalized) c.WGPUVertexFormat_Snorm32 else c.WGPUVertexFormat_Sint32,
        [2]i32 => if (normalized) c.WGPUVertexFormat_Snorm32x2 else c.WGPUVertexFormat_Sint32x2,
        [3]i32 => if (normalized) c.WGPUVertexFormat_Snorm32x3 else c.WGPUVertexFormat_Sint32x3,
        [4]i32 => if (normalized) c.WGPUVertexFormat_Snorm32x4 else c.WGPUVertexFormat_Sint32x4,
        else => @compileError("unsupported vertex format type: " ++ @typeName(T)),
    };
}

/// Produces a type with the same field names as T, allowing one to specify
/// AttributeOptions for each of the fields.
pub fn TypeAttributeOptions(comptime T: type) type {
    const field_names = std.meta.fieldNames(T);

    var fields: [field_names.len]std.builtin.Type.StructField = undefined;
    for (field_names, &fields) |field_name, *field| {
        field.* = .{
            .name = field_name,
            .type = AttributeOptions,
            .default_value = &AttributeOptions{},
            .is_comptime = true,
            .alignment = @alignOf(AttributeOptions),
        };
    }

    return @Type(.{
        .@"struct" = .{
            .layout = .auto,
            .fields = &fields,
            .decls = &.{},
            .is_tuple = false,
        },
    });
}

pub const AttributeOptions = struct {
    normalized: bool = false,
};

/// Constructs an array of WGPUVertexAttribute from a type.
pub fn vertexAttributesFromType(
    comptime T: type,
    comptime type_attribute_options: TypeAttributeOptions(T),
) [std.meta.fields(T).len]c.WGPUVertexAttribute {
    const fields = std.meta.fields(T);

    var result: [fields.len]c.WGPUVertexAttribute = undefined;
    inline for (fields, 0..) |field, i| {
        const attribute_options = @field(type_attribute_options, field.name);
        result[i] = .{
            .format = vertexFormatFromType(field.type, attribute_options.normalized),
            .offset = @offsetOf(T, field.name),
            .shaderLocation = @intCast(i),
        };
    }

    return result;
}

pub fn surfaceGetNextTextureView(
    surface: c.WGPUSurface,
    format: c.WGPUTextureFormat,
) !Nonnull(c.WGPUTextureView) {
    var surface_texture: c.WGPUSurfaceTexture = undefined;
    c.wgpuSurfaceGetCurrentTexture(surface, &surface_texture);
    switch (surface_texture.status) {
        c.WGPUSurfaceGetCurrentTextureStatus_Success => {
            if (surface_texture.suboptimal == @intFromBool(true)) {
                return error.SurfaceTextureSuboptimal;
            }
        },
        c.WGPUSurfaceGetCurrentTextureStatus_Timeout,
        c.WGPUSurfaceGetCurrentTextureStatus_Outdated,
        c.WGPUSurfaceGetCurrentTextureStatus_Lost,
        => {
            if (surface_texture.texture) |texture| {
                c.wgpuTextureRelease(texture);
            }
            return error.SurfaceTextureOutOfDate;
        },
        c.WGPUSurfaceGetCurrentTextureStatus_OutOfMemory => return error.OutOfMemory,
        c.WGPUSurfaceGetCurrentTextureStatus_DeviceLost => return error.DeviceLost,
        else => unreachable,
    }

    const view = c.wgpuTextureCreateView(surface_texture.texture, &.{
        .format = format,
        .dimension = c.WGPUTextureViewDimension_2D,
        .baseMipLevel = 0,
        .mipLevelCount = 1,
        .baseArrayLayer = 0,
        .arrayLayerCount = 1,
        .aspect = c.WGPUTextureAspect_All,
    });

    return view.?;
}

const HexColor = packed struct(u32) {
    alpha: u8,
    blue: u8,
    green: u8,
    red: u8,
};

pub fn colorFromHexLiteral(comptime value: u32) c.WGPUColor {
    const hex_color: HexColor = @bitCast(value);
    const alpha: f32 = @floatFromInt(hex_color.alpha);
    const blue: f32 = @floatFromInt(hex_color.blue);
    const green: f32 = @floatFromInt(hex_color.green);
    const red: f32 = @floatFromInt(hex_color.red);
    return .{
        .r = red / 255.0,
        .g = green / 255.0,
        .b = blue / 255.0,
        .a = alpha / 255.0,
    };
}

pub fn color(red: f32, green: f32, blue: f32, alpha: f32) c.WGPUColor {
    return .{
        .r = red,
        .g = green,
        .b = blue,
        .a = alpha,
    };
}

const LoadTextureCallbackData = struct {
    queue: c.WGPUQueue,
    texture: c.WGPUTexture,
    block_size: u32,
    block_width: u32,
};

fn loadTextureCompressed(
    mip_level: c_int,
    face_index: c_int,
    c_width: c_int,
    c_height: c_int,
    c_depth: c_int,
    face_lod_size: c.ktx_uint64_t,
    pixels: ?*anyopaque,
    user_data: ?*anyopaque,
) callconv(.C) c.KTX_error_code {
    _ = face_index;
    const data: *LoadTextureCallbackData = @ptrCast(@alignCast(user_data.?));

    const width: u32 = @intCast(c_width);
    const height: u32 = @intCast(c_height);
    const depth: u32 = @intCast(c_depth);

    const blocks_per_row = std.math.divCeil(u32, width, data.block_width) catch unreachable;
    const bytes_per_row = blocks_per_row * data.block_size;

    const clamped_width = @max(width, data.block_width);
    const clamped_height = @max(height, data.block_width);

    c.wgpuQueueWriteTexture(
        data.queue,
        &.{
            .texture = data.texture,
            .mipLevel = @intCast(mip_level),
            .origin = .{ .x = 0, .y = 0, .z = 0 },
            .aspect = c.WGPUTextureAspect_All,
        },
        pixels,
        face_lod_size,
        &.{
            .offset = 0,
            .bytesPerRow = bytes_per_row,
            .rowsPerImage = blocks_per_row,
        },
        &.{
            .width = clamped_width,
            .height = clamped_height,
            .depthOrArrayLayers = depth,
        },
    );

    return c.KTX_SUCCESS;
}

pub fn textureFormatName(format: c.WGPUTextureFormat) []const u8 {
    return switch (format) {
        c.WGPUTextureFormat_Undefined => "Undefined",
        c.WGPUTextureFormat_R8Unorm => "R8Unorm",
        c.WGPUTextureFormat_R8Snorm => "R8Snorm",
        c.WGPUTextureFormat_R8Uint => "R8Uint",
        c.WGPUTextureFormat_R8Sint => "R8Sint",
        c.WGPUTextureFormat_R16Uint => "R16Uint",
        c.WGPUTextureFormat_R16Sint => "R16Sint",
        c.WGPUTextureFormat_R16Float => "R16Float",
        c.WGPUTextureFormat_RG8Unorm => "RG8Unorm",
        c.WGPUTextureFormat_RG8Snorm => "RG8Snorm",
        c.WGPUTextureFormat_RG8Uint => "RG8Uint",
        c.WGPUTextureFormat_RG8Sint => "RG8Sint",
        c.WGPUTextureFormat_R32Float => "R32Float",
        c.WGPUTextureFormat_R32Uint => "R32Uint",
        c.WGPUTextureFormat_R32Sint => "R32Sint",
        c.WGPUTextureFormat_RG16Uint => "RG16Uint",
        c.WGPUTextureFormat_RG16Sint => "RG16Sint",
        c.WGPUTextureFormat_RG16Float => "RG16Float",
        c.WGPUTextureFormat_RGBA8Unorm => "RGBA8Unorm",
        c.WGPUTextureFormat_RGBA8UnormSrgb => "RGBA8UnormSrgb",
        c.WGPUTextureFormat_RGBA8Snorm => "RGBA8Snorm",
        c.WGPUTextureFormat_RGBA8Uint => "RGBA8Uint",
        c.WGPUTextureFormat_RGBA8Sint => "RGBA8Sint",
        c.WGPUTextureFormat_BGRA8Unorm => "BGRA8Unorm",
        c.WGPUTextureFormat_BGRA8UnormSrgb => "BGRA8UnormSrgb",
        c.WGPUTextureFormat_RGB10A2Uint => "RGB10A2Uint",
        c.WGPUTextureFormat_RGB10A2Unorm => "RGB10A2Unorm",
        c.WGPUTextureFormat_RG11B10Ufloat => "RG11B10Ufloat",
        c.WGPUTextureFormat_RGB9E5Ufloat => "RGB9E5Ufloat",
        c.WGPUTextureFormat_RG32Float => "RG32Float",
        c.WGPUTextureFormat_RG32Uint => "RG32Uint",
        c.WGPUTextureFormat_RG32Sint => "RG32Sint",
        c.WGPUTextureFormat_RGBA16Uint => "RGBA16Uint",
        c.WGPUTextureFormat_RGBA16Sint => "RGBA16Sint",
        c.WGPUTextureFormat_RGBA16Float => "RGBA16Float",
        c.WGPUTextureFormat_RGBA32Float => "RGBA32Float",
        c.WGPUTextureFormat_RGBA32Uint => "RGBA32Uint",
        c.WGPUTextureFormat_RGBA32Sint => "RGBA32Sint",
        c.WGPUTextureFormat_Stencil8 => "Stencil8",
        c.WGPUTextureFormat_Depth16Unorm => "Depth16Unorm",
        c.WGPUTextureFormat_Depth24Plus => "Depth24Plus",
        c.WGPUTextureFormat_Depth24PlusStencil8 => "Depth24PlusStencil8",
        c.WGPUTextureFormat_Depth32Float => "Depth32Float",
        c.WGPUTextureFormat_Depth32FloatStencil8 => "Depth32FloatStencil8",
        c.WGPUTextureFormat_BC1RGBAUnorm => "BC1RGBAUnorm",
        c.WGPUTextureFormat_BC1RGBAUnormSrgb => "BC1RGBAUnormSrgb",
        c.WGPUTextureFormat_BC2RGBAUnorm => "BC2RGBAUnorm",
        c.WGPUTextureFormat_BC2RGBAUnormSrgb => "BC2RGBAUnormSrgb",
        c.WGPUTextureFormat_BC3RGBAUnorm => "BC3RGBAUnorm",
        c.WGPUTextureFormat_BC3RGBAUnormSrgb => "BC3RGBAUnormSrgb",
        c.WGPUTextureFormat_BC4RUnorm => "BC4RUnorm",
        c.WGPUTextureFormat_BC4RSnorm => "BC4RSnorm",
        c.WGPUTextureFormat_BC5RGUnorm => "BC5RGUnorm",
        c.WGPUTextureFormat_BC5RGSnorm => "BC5RGSnorm",
        c.WGPUTextureFormat_BC6HRGBUfloat => "BC6HRGBUfloat",
        c.WGPUTextureFormat_BC6HRGBFloat => "BC6HRGBFloat",
        c.WGPUTextureFormat_BC7RGBAUnorm => "BC7RGBAUnorm",
        c.WGPUTextureFormat_BC7RGBAUnormSrgb => "BC7RGBAUnormSrgb",
        c.WGPUTextureFormat_ETC2RGB8Unorm => "ETC2RGB8Unorm",
        c.WGPUTextureFormat_ETC2RGB8UnormSrgb => "ETC2RGB8UnormSrgb",
        c.WGPUTextureFormat_ETC2RGB8A1Unorm => "ETC2RGB8A1Unorm",
        c.WGPUTextureFormat_ETC2RGB8A1UnormSrgb => "ETC2RGB8A1UnormSrgb",
        c.WGPUTextureFormat_ETC2RGBA8Unorm => "ETC2RGBA8Unorm",
        c.WGPUTextureFormat_ETC2RGBA8UnormSrgb => "ETC2RGBA8UnormSrgb",
        c.WGPUTextureFormat_EACR11Unorm => "EACR11Unorm",
        c.WGPUTextureFormat_EACR11Snorm => "EACR11Snorm",
        c.WGPUTextureFormat_EACRG11Unorm => "EACRG11Unorm",
        c.WGPUTextureFormat_EACRG11Snorm => "EACRG11Snorm",
        c.WGPUTextureFormat_ASTC4x4Unorm => "ASTC4x4Unorm",
        c.WGPUTextureFormat_ASTC4x4UnormSrgb => "ASTC4x4UnormSrgb",
        c.WGPUTextureFormat_ASTC5x4Unorm => "ASTC5x4Unorm",
        c.WGPUTextureFormat_ASTC5x4UnormSrgb => "ASTC5x4UnormSrgb",
        c.WGPUTextureFormat_ASTC5x5Unorm => "ASTC5x5Unorm",
        c.WGPUTextureFormat_ASTC5x5UnormSrgb => "ASTC5x5UnormSrgb",
        c.WGPUTextureFormat_ASTC6x5Unorm => "ASTC6x5Unorm",
        c.WGPUTextureFormat_ASTC6x5UnormSrgb => "ASTC6x5UnormSrgb",
        c.WGPUTextureFormat_ASTC6x6Unorm => "ASTC6x6Unorm",
        c.WGPUTextureFormat_ASTC6x6UnormSrgb => "ASTC6x6UnormSrgb",
        c.WGPUTextureFormat_ASTC8x5Unorm => "ASTC8x5Unorm",
        c.WGPUTextureFormat_ASTC8x5UnormSrgb => "ASTC8x5UnormSrgb",
        c.WGPUTextureFormat_ASTC8x6Unorm => "ASTC8x6Unorm",
        c.WGPUTextureFormat_ASTC8x6UnormSrgb => "ASTC8x6UnormSrgb",
        c.WGPUTextureFormat_ASTC8x8Unorm => "ASTC8x8Unorm",
        c.WGPUTextureFormat_ASTC8x8UnormSrgb => "ASTC8x8UnormSrgb",
        c.WGPUTextureFormat_ASTC10x5Unorm => "ASTC10x5Unorm",
        c.WGPUTextureFormat_ASTC10x5UnormSrgb => "ASTC10x5UnormSrgb",
        c.WGPUTextureFormat_ASTC10x6Unorm => "ASTC10x6Unorm",
        c.WGPUTextureFormat_ASTC10x6UnormSrgb => "ASTC10x6UnormSrgb",
        c.WGPUTextureFormat_ASTC10x8Unorm => "ASTC10x8Unorm",
        c.WGPUTextureFormat_ASTC10x8UnormSrgb => "ASTC10x8UnormSrgb",
        c.WGPUTextureFormat_ASTC10x10Unorm => "ASTC10x10Unorm",
        c.WGPUTextureFormat_ASTC10x10UnormSrgb => "ASTC10x10UnormSrgb",
        c.WGPUTextureFormat_ASTC12x10Unorm => "ASTC12x10Unorm",
        c.WGPUTextureFormat_ASTC12x10UnormSrgb => "ASTC12x10UnormSrgb",
        c.WGPUTextureFormat_ASTC12x12Unorm => "ASTC12x12Unorm",
        c.WGPUTextureFormat_ASTC12x12UnormSrgb => "ASTC12x12UnormSrgb",
        else => unreachable,
    };
}

pub fn presentModeName(mode: c.WGPUPresentMode) []const u8 {
    return switch (mode) {
        c.WGPUPresentMode_Fifo => "Fifo",
        c.WGPUPresentMode_FifoRelaxed => "FifoRelaxed",
        c.WGPUPresentMode_Immediate => "Immediate",
        c.WGPUPresentMode_Mailbox => "Mailbox",
        else => unreachable,
    };
}

pub fn featureName(feature: c.WGPUFeatureName) []const u8 {
    return switch (feature) {
        c.WGPUFeatureName_Undefined => "Undefined",
        c.WGPUFeatureName_DepthClipControl => "DepthClipControl",
        c.WGPUFeatureName_Depth32FloatStencil8 => "Depth32FloatStencil8",
        c.WGPUFeatureName_TimestampQuery => "TimestampQuery",
        c.WGPUFeatureName_TextureCompressionBC => "TextureCompressionBC",
        c.WGPUFeatureName_TextureCompressionETC2 => "TextureCompressionETC2",
        c.WGPUFeatureName_TextureCompressionASTC => "TextureCompressionASTC",
        c.WGPUFeatureName_IndirectFirstInstance => "IndirectFirstInstance",
        c.WGPUFeatureName_ShaderF16 => "ShaderF16",
        c.WGPUFeatureName_RG11B10UfloatRenderable => "RG11B10UfloatRenderable",
        c.WGPUFeatureName_BGRA8UnormStorage => "BGRA8UnormStorage",
        c.WGPUFeatureName_Float32Filterable => "Float32Filterable",
        c.WGPUNativeFeature_PushConstants => "PushConstants",
        c.WGPUNativeFeature_TextureAdapterSpecificFormatFeatures => "TextureAdapterSpecificFormatFeatures",
        c.WGPUNativeFeature_MultiDrawIndirect => "MultiDrawIndirect",
        c.WGPUNativeFeature_MultiDrawIndirectCount => "MultiDrawIndirectCount",
        c.WGPUNativeFeature_VertexWritableStorage => "VertexWritableStorage",
        c.WGPUNativeFeature_TextureBindingArray => "TextureBindingArray",
        c.WGPUNativeFeature_SampledTextureAndStorageBufferArrayNonUniformIndexing => "SampledTextureAndStorageBufferArrayNonUniformIndexing",
        c.WGPUNativeFeature_PipelineStatisticsQuery => "PipelineStatisticsQuery",
        c.WGPUNativeFeature_StorageResourceBindingArray => "StorageResourceBindingArray",
        c.WGPUNativeFeature_PartiallyBoundBindingArray => "PartiallyBoundBindingArray",
        else => unreachable,
    };
}

pub fn adapterTypeName(kind: c.WGPUAdapterType) []const u8 {
    return switch (kind) {
        c.WGPUAdapterType_DiscreteGPU => "DiscreteGPU",
        c.WGPUAdapterType_IntegratedGPU => "IntegratedGPU",
        c.WGPUAdapterType_CPU => "CPU",
        c.WGPUAdapterType_Unknown => "Unknown",
        else => unreachable,
    };
}

pub fn backendTypeName(backend: c.WGPUBackendType) []const u8 {
    return switch (backend) {
        c.WGPUBackendType_Undefined => "Undefined",
        c.WGPUBackendType_Null => "Null",
        c.WGPUBackendType_WebGPU => "WebGPU",
        c.WGPUBackendType_D3D11 => "D3D11",
        c.WGPUBackendType_D3D12 => "D3D12",
        c.WGPUBackendType_Metal => "Metal",
        c.WGPUBackendType_Vulkan => "Vulkan",
        c.WGPUBackendType_OpenGL => "OpenGL",
        c.WGPUBackendType_OpenGLES => "OpenGLES",
        else => unreachable,
    };
}

pub fn deviceLostReasonName(reason: c.WGPUDeviceLostReason) []const u8 {
    return switch (reason) {
        c.WGPUDeviceLostReason_Undefined => "Undefined",
        c.WGPUDeviceLostReason_Destroyed => "Destroyed",
        else => unreachable,
    };
}

pub fn errorTypeName(kind: c.WGPUErrorType) []const u8 {
    return switch (kind) {
        c.WGPUErrorType_NoError => "NoError",
        c.WGPUErrorType_Validation => "Validation",
        c.WGPUErrorType_OutOfMemory => "OutOfMemory",
        c.WGPUErrorType_Internal => "Internal",
        c.WGPUErrorType_Unknown => "Unknown",
        c.WGPUErrorType_DeviceLost => "DeviceLost",
        else => unreachable,
    };
}

pub fn formatTextureFormat(
    format: c.WGPUTextureFormat,
    comptime fmt: []const u8,
    options: std.fmt.FormatOptions,
    writer: anytype,
) !void {
    return formatEnumValue(format, fmt, options, writer, textureFormatName);
}

pub fn formatPresentMode(
    mode: c.WGPUPresentMode,
    comptime fmt: []const u8,
    options: std.fmt.FormatOptions,
    writer: anytype,
) !void {
    return formatEnumValue(mode, fmt, options, writer, presentModeName);
}

pub fn formatFeature(
    feature: c.WGPUFeatureName,
    comptime fmt: []const u8,
    options: std.fmt.FormatOptions,
    writer: anytype,
) !void {
    return formatEnumValue(feature, fmt, options, writer, featureName);
}

pub fn formatAdapterType(
    kind: c.WGPUAdapterType,
    comptime fmt: []const u8,
    options: std.fmt.FormatOptions,
    writer: anytype,
) !void {
    return formatEnumValue(kind, fmt, options, writer, adapterTypeName);
}

pub fn formatBackendType(
    backend: c.WGPUBackendType,
    comptime fmt: []const u8,
    options: std.fmt.FormatOptions,
    writer: anytype,
) !void {
    return formatEnumValue(backend, fmt, options, writer, backendTypeName);
}

pub fn formatDeviceLostReason(
    reason: c.WGPUDeviceLostReason,
    comptime fmt: []const u8,
    options: std.fmt.FormatOptions,
    writer: anytype,
) !void {
    return formatEnumValue(reason, fmt, options, writer, deviceLostReasonName);
}

pub fn formatErrorType(
    kind: c.WGPUErrorType,
    comptime fmt: []const u8,
    options: std.fmt.FormatOptions,
    writer: anytype,
) !void {
    return formatEnumValue(kind, fmt, options, writer, errorTypeName);
}

fn formatEnumValue(
    raw_value: anytype,
    comptime fmt: []const u8,
    options: std.fmt.FormatOptions,
    writer: anytype,
    stringify: fn (
        value: @TypeOf(raw_value),
    ) []const u8,
) !void {
    const Format = enum {
        forwarded,
        string,
    };

    comptime var format: Format = .forwarded;
    inline for (fmt) |char| {
        switch (char) {
            's' => format = .string,
            else => format = .forwarded,
        }
    }

    switch (format) {
        .string => try writer.writeAll(stringify(raw_value)),
        else => try std.fmt.formatIntValue(raw_value, fmt, options, writer),
    }
}
