const std = @import("std");

pub fn sliceFromParts(ptr: anytype, len: usize) []std.meta.Elem(@TypeOf(ptr)) {
    const unwrapped_ptr = ptr orelse return &.{};
    return unwrapped_ptr[0..len];
}

/// Requires `haystack` and `needles` to be sorted in the same order.
pub fn containsAll(comptime T: type, haystack: []const T, needles: []const T) bool {
    var needles_found: usize = 0;

    for (haystack) |element| {
        // Found all needles.
        if (needles_found == needles.len) {
            return true;
        }

        // Found this needle. Go to the next one.
        if (element == needles[needles_found]) {
            needles_found += 1;
        }
    }

    // Exhausted the haystack without finding all needles.
    return false;
}

pub fn findFirstOf(comptime T: type, haystack: []const T, needles: []const T) ?T {
    for (needles) |needle| {
        if (std.mem.indexOfScalar(T, haystack, needle) != null) {
            return needle;
        }
    }

    return null;
}

pub fn iota(slice: anytype, start: std.meta.Elem(@TypeOf(slice))) void {
    for (slice, 0..) |*element, i| {
        element.* = @as(@TypeOf(start), @intCast(i)) + start;
    }
}

pub fn sizeOfElements(object: anytype) usize {
    const Type = @TypeOf(object);
    switch (@typeInfo(Type)) {
        .Pointer => |pointer| {
            switch (pointer.size) {
                .One, .Many => return @sizeOf(pointer.child),
                .Slice => return @sizeOf(pointer.child) * object.len,
                .C => @compileError("cannot get content size of C-style array as length is unknown"),
            }
        },
        else => @compileError("not a slice-like type: " ++ @typeName(Type)),
    }
}
