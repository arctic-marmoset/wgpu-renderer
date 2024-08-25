const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    // TODO: Implement static option.
    const static = true;

    const lib = b.addStaticLibrary(.{
        .name = "cglm",
        .target = target,
        .optimize = optimize,
    });
    lib.installHeadersDirectory(b.path("cglm/include"), "", .{});
    if (static) {
        lib.defineCMacro("CGLM_STATIC", null);
    }
    lib.addCSourceFiles(.{
        .files = &sources,
        .flags = &.{
            "-std=c99",
        },
    });
    lib.linkLibC();

    b.installArtifact(lib);
}

const sources = .{
    "cglm/src/aabb2d.c",
    "cglm/src/affine.c",
    "cglm/src/affine2d.c",
    "cglm/src/bezier.c",
    "cglm/src/box.c",
    "cglm/src/cam.c",
    "cglm/src/clipspace/ortho_lh_no.c",
    "cglm/src/clipspace/ortho_lh_zo.c",
    "cglm/src/clipspace/ortho_rh_no.c",
    "cglm/src/clipspace/ortho_rh_zo.c",
    "cglm/src/clipspace/persp_lh_no.c",
    "cglm/src/clipspace/persp_lh_zo.c",
    "cglm/src/clipspace/persp_rh_no.c",
    "cglm/src/clipspace/persp_rh_zo.c",
    "cglm/src/clipspace/project_no.c",
    "cglm/src/clipspace/project_zo.c",
    "cglm/src/clipspace/view_lh_no.c",
    "cglm/src/clipspace/view_lh_zo.c",
    "cglm/src/clipspace/view_rh_no.c",
    "cglm/src/clipspace/view_rh_zo.c",
    "cglm/src/config.h",
    "cglm/src/curve.c",
    "cglm/src/ease.c",
    "cglm/src/euler.c",
    "cglm/src/frustum.c",
    "cglm/src/io.c",
    "cglm/src/ivec2.c",
    "cglm/src/ivec3.c",
    "cglm/src/ivec4.c",
    "cglm/src/mat2.c",
    "cglm/src/mat2x3.c",
    "cglm/src/mat2x4.c",
    "cglm/src/mat3.c",
    "cglm/src/mat3x2.c",
    "cglm/src/mat3x4.c",
    "cglm/src/mat4.c",
    "cglm/src/mat4x2.c",
    "cglm/src/mat4x3.c",
    "cglm/src/plane.c",
    "cglm/src/project.c",
    "cglm/src/quat.c",
    "cglm/src/ray.c",
    "cglm/src/sphere.c",
    "cglm/src/swift/empty.c",
    "cglm/src/vec2.c",
    "cglm/src/vec3.c",
    "cglm/src/vec4.c",
};
