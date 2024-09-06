const std = @import("std");

const cargo = @import("build-src/cargo.zig");

const fs = std.fs;

pub fn build(b: *std.Build) !void {
    const requested_target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Zig defaults to windows-gnu as the "native" target on Windows. But:
    // a) Most Windows devs probably have MSVC.
    // b) Most Windows devs probably install Rust with the default MSVC toolchain.
    // So, in order to avoid the hassle of installing the necessary toolchain
    // components for the GNU ABI, we override it to MSVC.
    const target = blk: {
        var overridden = requested_target;
        if (requested_target.result.os.tag == .windows and requested_target.result.abi == .gnu) {
            std.log.warn("target `windows-gnu` is unsupported - overriding to `windows-msvc`", .{});
            overridden.query.abi = .msvc;
            overridden.result.abi = .msvc;
        }
        break :blk overridden;
    };

    const lib = cargo.addSharedLibrary(b, .{
        .manifest_path = b.path("wgpu-native/Cargo.toml"),
        .name = "wgpu_native",
        .target = target,
        .optimize = optimize,
    });
    lib.installHeadersDirectory(b.path("wgpu-native/ffi/webgpu-headers"), "webgpu", .{});
    lib.installHeader(b.path("wgpu-native/ffi/wgpu.h"), "webgpu/wgpu.h");

    b.installArtifact(lib);
}
