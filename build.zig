const std = @import("std");
const builtin = @import("builtin");

const buildShaders = @import("build-src/shader.zig").buildShaders;

const fs = std.fs;

pub fn build(b: *std.Build) !void {
    const requested_target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const install_docs = b.option(
        bool,
        "install-docs",
        "Copy docs to the install prefix during the install step",
    ) orelse false;

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

    const cglm = b.dependency("cglm", .{
        .target = target,
        .optimize = optimize,
    });

    const glfw = b.dependency("glfw", .{
        .target = target,
        .optimize = optimize,
        .shared = true,
    });

    const ktx = b.dependency("ktx", .{
        .target = target,
        .optimize = optimize,
        .static = false,
    });

    const wgpu = b.dependency("wgpu-native", .{
        .target = target,
        .optimize = optimize,
    });

    const zgltf = b.dependency("zgltf", .{
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "wgpu-renderer",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    if (target.result.os.tag == .windows) {
        // This needs to be explicitly defined if we want to use MSVC.
        exe.defineCMacro("MIDL_INTERFACE", "struct");
    }
    exe.addIncludePath(b.path("src"));
    exe.root_module.addImport("Gltf", zgltf.module("zgltf"));
    // FIXME: It would be nice if we could avoid linking the actual CGLM library
    //        since we're actually using the header-only interface. Linking here
    //        just increases code size for no reason.
    exe.linkLibrary(cglm.artifact("cglm"));
    exe.linkLibrary(glfw.artifact("glfw"));
    exe.linkLibrary(ktx.artifact("ktx"));
    exe.linkLibrary(wgpu.artifact("wgpu_native"));
    if (target.result.os.tag == .macos) {
        exe.addCSourceFiles(.{ .files = &.{"src/GLFW-WGPU-Bridge.m"} });
    } else {
        exe.addCSourceFiles(.{ .files = &.{"src/GLFW-WGPU-Bridge.c"} });
    }

    switch (target.result.os.tag) {
        .macos => {
            exe.linkFramework("CoreFoundation");
            exe.linkFramework("MetalKit");
            exe.linkFramework("QuartzCore");
        },
        .windows => {
            exe.linkSystemLibrary("d3dcompiler");
            exe.linkSystemLibrary("Gdi32");
            exe.linkSystemLibrary("Opengl32");
            exe.linkSystemLibrary("User32");
            exe.linkSystemLibrary("Userenv");
            exe.linkSystemLibrary("Ws2_32");
        },
        else => {
            std.debug.panic("unsupported platform: {}", .{target.result});
        },
    }

    _ = buildShaders(b, .{
        .step_name = "shaders",
        .description = "Compile shaders to SPIR-V bytecode",
        .optimize = optimize,
        .out_dir_path = "data/shaders",
        .sources = &.{
            "shaders/src/triangle.vs.hlsl",
            "shaders/src/triangle.fs.hlsl",
        },
    });

    const unit_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);
    b.getInstallStep().dependOn(test_step);

    b.installArtifact(exe);
    if (target.result.os.tag == .windows) {
        b.installArtifact(glfw.artifact("glfw"));
        b.installArtifact(ktx.artifact("ktx"));
        b.installArtifact(wgpu.artifact("wgpu_native"));
    }

    if (install_docs) {
        b.installDirectory(.{
            .source_dir = exe.getEmittedDocs(),
            .install_dir = .prefix,
            .install_subdir = "docs",
        });
    }
}
