pub usingnamespace @cImport({
    @cInclude("ImGuiBackend.h");

    @cDefine("CGLM_FORCE_DEPTH_ZERO_TO_ONE", {});
    // CGLM has to be LH because the transform matrices need to be aware of the
    // clip space coordinate system (which is LH in WebGPU).
    @cDefine("CGLM_FORCE_LEFT_HANDED", {});
    @cInclude("cglm/call.h");

    @cInclude("GLFW/glfw3.h");
    @cInclude("GLFW-WGPU-Bridge.h");

    @cInclude("webgpu/wgpu.h");

    @cInclude("ktx.h");
    @cInclude("vkformat_enum.h");

    @cInclude("cimgui.h");
});
