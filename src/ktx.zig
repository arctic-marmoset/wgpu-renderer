const std = @import("std");

const c = @import("c.zig");

pub inline fn textureIterateLoadLevelFaces(
    texture: anytype,
    callback: c.PFNKTXITERCB,
    user_data: ?*anyopaque,
) c.KTX_error_code {
    switch (@TypeOf(texture)) {
        *c.ktxTexture1,
        *c.ktxTexture2,
        *c.ktxTexture,
        => {},
        else => @compileError("`texture` must be a ktxTexture"),
    }

    return texture.*.vtbl.?.*.IterateLoadLevelFaces.?(@ptrCast(texture), callback, user_data);
}
