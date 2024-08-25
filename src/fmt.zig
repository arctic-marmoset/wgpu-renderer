const std = @import("std");

pub fn fmtSliceElementFormatter(
    comptime T: anytype,
    slice: []const T,
    comptime formatter: fn (
        element: T,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) anyerror!void,
) SliceFormatter(T, formatter) {
    return .{ .data = slice };
}

pub fn SliceFormatter(
    comptime T: anytype,
    comptime formatter: fn (
        element: T,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) anyerror!void,
) type {
    return struct {
        data: []const T,

        pub fn format(
            self: @This(),
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            var delimiter: []const u8 = "";
            for (self.data) |element| {
                try writer.writeAll(delimiter);
                try formatter(element, fmt, options, writer);
                delimiter = ", ";
            }
        }
    };
}
