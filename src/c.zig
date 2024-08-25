pub usingnamespace @cImport({
    @cDefine("CGLM_FORCE_DEPTH_ZERO_TO_ONE", {});
    @cDefine("CGLM_FORCE_LEFT_HANDED", {});
    @cInclude("cglm/cglm.h");

    @cInclude("GLFW/glfw3.h");
    @cInclude("GLFW-WGPU-Bridge.h");

    @cInclude("wgpu.h");

    @cInclude("ktx.h");
    @cInclude("vkformat_enum.h");
});
