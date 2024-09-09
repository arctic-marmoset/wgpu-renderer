#include "ImGuiBackend.h"

#include <imgui_impl_glfw.h>
#include <imgui_impl_wgpu.h>

bool ImGuiBackendInitialize(const ImGuiBackendInitializeInfo *info) {
    ImGui_ImplWGPU_InitInfo wgpuInitInfo;
    wgpuInitInfo.Device = info->device;
    wgpuInitInfo.RenderTargetFormat = info->renderTargetFormat;
    wgpuInitInfo.DepthStencilFormat = info->depthStencilFormat;

    bool result = true;
    result = result && ImGui_ImplGlfw_InitForOther(info->window, true);
    result = result && ImGui_ImplWGPU_Init(&wgpuInitInfo);
    return result;
}

void ImGuiBackendTerminate() {
    ImGui_ImplWGPU_Shutdown();
    ImGui_ImplGlfw_Shutdown();
}

void ImGuiBackendBeginFrame() {
    ImGui_ImplWGPU_NewFrame();
    ImGui_ImplGlfw_NewFrame();
}

void ImGuiBackendDraw(WGPURenderPassEncoder renderPassEncoder) {
    ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), renderPassEncoder);
}
