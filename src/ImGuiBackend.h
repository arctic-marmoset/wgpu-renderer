#pragma once

#include <GLFW/glfw3.h>
#include <webgpu/webgpu.h>

#include <stdbool.h>

typedef struct {
    GLFWwindow *window;
    WGPUDevice device;
    WGPUTextureFormat renderTargetFormat;
    WGPUTextureFormat depthStencilFormat;
} ImGuiBackendInitializeInfo;

#if defined(__cplusplus)
extern "C" {
#endif

bool ImGuiBackendInitialize(const ImGuiBackendInitializeInfo *info);

void ImGuiBackendTerminate(void);

void ImGuiBackendBeginFrame(void);

void ImGuiBackendDraw(WGPURenderPassEncoder renderPassEncoder);

#if defined(__cplusplus)
}
#endif
