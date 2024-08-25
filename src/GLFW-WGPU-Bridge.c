#include "GLFW-WGPU-Bridge.h"

#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include <assert.h>
#include <stddef.h>

WGPUSurface glfwCreateWGPUSurface(WGPUInstance instance, GLFWwindow *const window) {
    if (!window) {
        return NULL;
    }
    assert("GLFWwindow should not be NULL" && window);

    // TODO: Handle platforms other than Windows.
    const HWND hWnd = glfwGetWin32Window(window);
    const HINSTANCE hInstance = GetModuleHandleW(NULL);
    const WGPUSurfaceDescriptorFromWindowsHWND fromWindows = {
        .chain = { .sType = WGPUSType_SurfaceDescriptorFromWindowsHWND },
        .hinstance = hInstance,
        .hwnd = hWnd,
    };

    return wgpuInstanceCreateSurface(
        instance,
        &(WGPUSurfaceDescriptor){
            .nextInChain = &fromWindows.chain,
        }
    );
}
