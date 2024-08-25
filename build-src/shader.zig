const std = @import("std");

const fs = std.fs;

pub const ShadersOptions = struct {
    step_name: []const u8,
    description: []const u8,
    optimize: std.builtin.OptimizeMode,
    debug_symbols: bool = false,
    out_dir_path: []const u8 = "shaders",
    sources: []const []const u8 = &.{},
    file_extensions: struct {
        bytecode: []const u8,
        depfile: []const u8,
    } = .{
        .bytecode = ".spv",
        .depfile = ".d",
    },
};

pub fn buildShaders(
    b: *std.Build,
    options: ShadersOptions,
) *std.Build.Step {
    const optimize_flags: []const []const u8 = switch (options.optimize) {
        .Debug => &.{"-O0"},
        .ReleaseSmall => &.{"-Os"},
        else => &.{"-O"},
    };

    const step = b.step(options.step_name, options.description);

    for (options.sources) |source| {
        const source_file_name = fs.path.basename(source);
        const source_extension = fs.path.extension(source_file_name);
        const is_hlsl = std.mem.eql(u8, source_extension, ".hlsl");
        const shader_name = if (is_hlsl)
            fs.path.stem(source_file_name)
        else
            source_file_name;

        const output_file = b.fmt("{s}{s}", .{ shader_name, options.file_extensions.bytecode });
        const depfile = b.fmt("{s}{s}", .{ source_file_name, options.file_extensions.depfile });

        const command = b.addSystemCommand(&.{"glslc"});
        command.addArg("-MD");
        command.addArg("-MF");
        _ = command.addDepFileOutputArg(depfile);
        command.addArg("-o");
        const shader_binary = command.addOutputFileArg(output_file);
        if (options.debug_symbols) command.addArg("-g");
        command.addArgs(optimize_flags);
        command.addFileArg(b.path(source));
        step.dependOn(&command.step);

        const install_path = b.pathJoin(&.{ options.out_dir_path, output_file });
        b.getInstallStep().dependOn(
            &b.addInstallFile(shader_binary, install_path).step,
        );
    }

    return step;
}
