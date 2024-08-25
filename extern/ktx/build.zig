const std = @import("std");
const builtin = @import("builtin");

const fs = std.fs;

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    // TODO: Handle static option.
    const static = b.option(bool, "static", "Build as static library") orelse false;
    // TODO: Implement features as options.
    const feature_gl_upload = true;
    const feature_etc_unpack = false;

    const cxx_sources = .{
        "KTX-Software/external/basisu/transcoder/basisu_transcoder.cpp",
        "KTX-Software/lib/basis_transcode.cpp",
        "KTX-Software/lib/miniz_wrapper.cpp",
        "KTX-Software/lib/etcunpack.cxx",
    };

    var c_sources = std.ArrayList([]const u8).init(b.allocator);
    c_sources.appendSlice(&.{
        "KTX-Software/external/basisu/zstd/zstd.c",
        "KTX-Software/external/dfdutils/createdfd.c",
        "KTX-Software/external/dfdutils/colourspaces.c",
        "KTX-Software/external/dfdutils/interpretdfd.c",
        "KTX-Software/external/dfdutils/printdfd.c",
        "KTX-Software/external/dfdutils/queries.c",
        "KTX-Software/external/dfdutils/vk2dfd.c",
        "KTX-Software/lib/checkheader.c",
        "KTX-Software/lib/filestream.c",
        "KTX-Software/lib/hashlist.c",
        "KTX-Software/lib/info.c",
        "KTX-Software/lib/memstream.c",
        "KTX-Software/lib/strings.c",
        "KTX-Software/lib/swap.c",
        "KTX-Software/lib/texture.c",
        "KTX-Software/lib/texture1.c",
        "KTX-Software/lib/texture2.c",
        "KTX-Software/lib/vkformat_check.c",
        "KTX-Software/lib/vkformat_str.c",
        "KTX-Software/lib/vkformat_typesize.c",
    }) catch @panic("OOM");

    if (feature_gl_upload) {
        c_sources.appendSlice(&.{
            "KTX-Software/lib/gl_funcs.c",
            "KTX-Software/lib/glloader.c",
        }) catch @panic("OOM");
    }

    const linkage: std.builtin.LinkMode = if (static) .static else .dynamic;
    const lib = std.Build.Step.Compile.create(b, .{
        .name = "ktx",
        .root_module = .{
            .target = target,
            .optimize = optimize,
        },
        .kind = .lib,
        .linkage = linkage,
    });
    lib.addIncludePath(b.path("KTX-Software/include"));
    lib.addIncludePath(b.path("KTX-Software/external"));
    lib.addIncludePath(b.path("KTX-Software/external/basisu/transcoder"));
    lib.addIncludePath(b.path("KTX-Software/external/basisu/zstd"));
    lib.addIncludePath(b.path("KTX-Software/utils"));
    lib.addIncludePath(b.path("KTX-Software/other_include"));
    lib.installHeadersDirectory(b.path("KTX-Software/include"), "", .{});
    lib.installHeader(b.path("KTX-Software/lib/vkformat_enum.h"), "vkformat_enum.h");
    lib.defineCMacro("BASISD_SUPPORT_FXT1", "0");
    lib.defineCMacro("KTX_FEATURE_KTX2", null);
    if (target.result.os.tag == .windows) {
        lib.defineCMacro("KTX_API", "__declspec(dllexport)");
    }
    lib.addCSourceFiles(.{
        .files = c_sources.items,
    });
    lib.addCSourceFiles(.{
        .files = &cxx_sources,
    });
    if (optimize == .Debug) {
        lib.defineCMacro("_DEBUG", null);
        lib.defineCMacro("DEBUG", null);
        lib.defineCMacro("LIBKTX", null);
        lib.defineCMacro(
            "SUPPORT_SOFTWARE_ETC_UNPACK",
            b.fmt("{}", .{@intFromBool(feature_etc_unpack)}),
        );
    }
    lib.linkLibC();
    if (!(target.result.os.tag == .windows and target.result.abi == .msvc)) {
        lib.linkLibCpp();
    }

    b.installArtifact(lib);
}
