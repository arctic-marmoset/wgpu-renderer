//! This module provides utility functions to wrap `cargo build` output in Zig
//! artifacts (i.e., std.Build.Step.Compile).

const std = @import("std");
const builtin = @import("builtin");

const rust = @import("rust.zig");

pub const Options = struct {
    manifest_path: std.Build.LazyPath,
    name: []const u8,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
};

pub fn addSharedLibrary(b: *std.Build, options: Options) *std.Build.Step.Compile {
    const cargo_build_step = BuildStep.create(b, .{
        .manifest_path = options.manifest_path,
        .name = options.name,
        .kind = .lib,
        .linkage = .dynamic,
        .target = options.target,
        .optimize = options.optimize,
    });

    return cargo_build_step;
}

const BuildStepOptions = struct {
    manifest_path: std.Build.LazyPath,
    name: []const u8,
    kind: std.Build.Step.Compile.Kind,
    linkage: std.builtin.LinkMode,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
};

/// A Step that replaces the standard Compile step's makeFn with one runs
/// `cargo build` and assigns the output to the Compile step's generated path.
const BuildStep = struct {
    compile: std.Build.Step.Compile,
    path: std.Build.LazyPath,
    sub_path: []const u8,

    pub fn create(owner: *std.Build, options: BuildStepOptions) *std.Build.Step.Compile {
        const compile = std.Build.Step.Compile.create(owner, .{
            .name = options.name,
            .kind = options.kind,
            .linkage = options.linkage,
            .root_module = .{
                .target = options.target,
                .optimize = options.optimize,
            },
        });
        compile.step.makeFn = make;

        const rust_target = rust.Target.fromZig(options.target.result) catch
            @panic("cargo build: unsupported target");
        const rust_target_string = owner.fmt("{}", .{rust_target});
        const optimize_name = if (options.optimize == .Debug) "debug" else "release";

        const run_cargo = owner.addSystemCommand(&.{ "cargo", "build" });
        run_cargo.addArg("--manifest-path");
        run_cargo.addFileArg(options.manifest_path);
        if (options.optimize != .Debug) run_cargo.addArg("--release");
        run_cargo.addArgs(&.{ "--target", rust_target_string });
        run_cargo.addArg("--target-dir");
        const output_path = run_cargo.addOutputDirectoryArg(options.name);
        if (options.target.query.cpu_arch != null and
            options.target.query.os_tag != null and
            options.target.query.abi != null)
        {
            const linker_path = owner.pathJoin(&.{
                owner.build_root.path.?,
                "build-src",
                "linker.py",
            });

            comptime var linker_wrapper_header: []const u8 = "#!/usr/bin/env sh";
            comptime var linker_wrapper_name: []const u8 = "linker";
            comptime var shell_args = "$@";
            if (builtin.os.tag == .windows) {
                linker_wrapper_header = "@echo off";
                linker_wrapper_name = linker_wrapper_name ++ ".bat";
                shell_args = "%*";
            }

            const write_linker_wrapper = owner.addWriteFiles();
            const linker_wrapper = write_linker_wrapper.add(linker_wrapper_name, owner.fmt(
                \\{s}
                \\python3 {s} {s}-{s}-{s} {s}
            , .{
                linker_wrapper_header,
                linker_path,
                @tagName(options.target.result.cpu.arch),
                @tagName(options.target.result.os.tag),
                @tagName(options.target.result.abi),
                shell_args,
            }));
            const copy_linker_wrapper = owner.addWriteFiles();
            const executable_linker_wrapper = copy_linker_wrapper.addCopyFile(linker_wrapper, linker_wrapper_name);
            if (builtin.os.tag != .windows) {
                const chmod = owner.addSystemCommand(&.{ "chmod", "u+x" });
                chmod.addFileArg(executable_linker_wrapper);
                run_cargo.step.dependOn(&chmod.step);
            }
            run_cargo.setCwd(executable_linker_wrapper.dirname());
            run_cargo.addArgs(&.{
                "--config",
                owner.fmt("target.{s}.linker=\"./linker\"", .{rust_target_string}),
            });
        }
        // Compile step cannot run until after `cargo build`.
        output_path.addStepDependencies(&compile.step);

        const cargo_build_step = owner.allocator.create(BuildStep) catch @panic("OOM");
        cargo_build_step.* = .{
            .compile = compile.*,
            .path = output_path,
            .sub_path = owner.pathJoin(&.{ rust_target_string, optimize_name }),
        };

        return &cargo_build_step.compile;
    }

    fn make(step: *std.Build.Step, options: std.Build.Step.MakeOptions) !void {
        _ = options;
        const b = step.owner;

        const compile: *std.Build.Step.Compile = @fieldParentPtr("step", step);
        const cargo_build_step: *BuildStep = @fieldParentPtr("compile", compile);
        const root_path = cargo_build_step.path.getPath2(b, step);
        const sub_path = cargo_build_step.sub_path;
        const full_path = b.pathJoin(&.{ root_path, sub_path });
        if (compile.emit_directory) |directory| {
            directory.path = full_path;
        }
        if (compile.generated_bin) |bin| {
            bin.path = b.pathJoin(&.{ full_path, compile.out_filename });
        }
        if (compile.generated_implib) |implib| {
            implib.path = b.pathJoin(&.{ full_path, b.fmt("{s}.lib", .{compile.name}) });
        }
        if (compile.generated_pdb) |pdb| {
            pdb.path = b.pathJoin(&.{ full_path, b.fmt("{s}.pdb", .{compile.name}) });
        }

        var all_cached = true;
        for (step.dependencies.items) |dependency| {
            all_cached = all_cached and dependency.result_cached;
        }
        step.result_cached = all_cached;
    }
};
