#include "WGR-ImGui-Bridge.h"

#include <imgui_impl_glfw.h>
#include <imgui_impl_wgpu.h>

bool WGRImGuiInitialize(
    GLFWwindow *window,
    const WGRImGuiInitializeInfo *info,
    ImGuiContext **outContext
) {
    assert("outContext must be a valid (non-null) pointer" && outContext);
    auto *imGuiContext = ImGui::CreateContext();
    *outContext = imGuiContext;

    ImGui_ImplWGPU_InitInfo wgpuInitInfo;
    wgpuInitInfo.Device = info->device;
    wgpuInitInfo.RenderTargetFormat = info->renderTargetFormat;
    wgpuInitInfo.DepthStencilFormat = info->depthStencilFormat;

    bool result = true;
    result = result && ImGui_ImplGlfw_InitForOther(window, true);
    result = result && ImGui_ImplWGPU_Init(&wgpuInitInfo);
    return result;
}

void WGRImGuiTerminate(ImGuiContext *context) {
    ImGui_ImplWGPU_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext(context);
}

void WGRImGuiBeginFrame(WGRUIState *state) {
    ImGui_ImplWGPU_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::ShowDemoWindow(&state->demoWindowOpen);
    ImGui::Render();
}

void WGRImGuiEndFrame(WGPURenderPassEncoder renderPassEncoder) {
    ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), renderPassEncoder);
}

bool WGRImGuiWantsCaptureMouse() {
    return ImGui::GetIO().WantCaptureMouse;
}

bool WGRImGuiWantsCaptureKeyboard() {
    return ImGui::GetIO().WantCaptureKeyboard;
}
