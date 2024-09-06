#pragma once

#include <GLFW/glfw3.h>
#include <webgpu/webgpu.h>

#include <stdbool.h>

struct ImGuiContext;

struct WGRUIState {
    bool demoWindowOpen;
};

struct WGRImGuiInitializeInfo {
    WGPUDevice device;
    WGPUTextureFormat renderTargetFormat;
    WGPUTextureFormat depthStencilFormat;
};

#if defined(__cplusplus)
extern "C" {
#endif

bool WGRImGuiInitialize(
    GLFWwindow *window,
    const struct WGRImGuiInitializeInfo *info,
    struct ImGuiContext **outContext
);

void WGRImGuiTerminate(struct ImGuiContext *context);

void WGRImGuiBeginFrame(struct WGRUIState *state);

void WGRImGuiEndFrame(WGPURenderPassEncoder renderPassEncoder);

bool WGRImGuiWantsCaptureMouse(void);

bool WGRImGuiWantsCaptureKeyboard(void);

#if defined(__cplusplus)
}
#endif
