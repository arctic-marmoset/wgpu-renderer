#include "GLFW-WGPU-Bridge.h"

#if defined(_WIN64) || defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#elif defined(__linux__)
#define GLFW_EXPOSE_NATIVE_X11
#define GLFW_EXPOSE_NATIVE_WAYLAND
#endif
#include <GLFW/glfw3native.h>

#include <assert.h>
#include <stddef.h>

WGPUSurface glfwCreateWGPUSurface(WGPUInstance instance, GLFWwindow *const window) {
    if (!window) {
        return NULL;
    }
    assert("GLFWwindow should not be NULL" && window);

#if defined(GLFW_EXPOSE_NATIVE_WIN32)
    const HWND hWnd = glfwGetWin32Window(window);
    const HINSTANCE hInstance = GetModuleHandleW(NULL);
    const WGPUSurfaceDescriptorFromWindowsHWND fromWindowsHWND = {
        .chain = { .sType = WGPUSType_SurfaceDescriptorFromWindowsHWND },
        .hinstance = hInstance,
        .hwnd = hWnd,
    };
    const WGPUChainedStruct *chain = &fromWindowsHWND.chain;
#elif defined(GLFW_EXPOSE_NATIVE_X11) && defined(GLFW_EXPOSE_NATIVE_WAYLAND)
    WGPUSurfaceDescriptorFromXlibWindow fromXlibWindow;
    WGPUSurfaceDescriptorFromWaylandSurface fromWaylandSurface;
    WGPUChainedStruct *chain = NULL;

    switch (glfwGetPlatform()) {
    case GLFW_PLATFORM_X11: {
        Display *const display = glfwGetX11Display();
        const Window x11Window = glfwGetX11Window(window);
        fromXlibWindow = (WGPUSurfaceDescriptorFromXlibWindow){
            .chain = { .sType = WGPUSType_SurfaceDescriptorFromXlibWindow },
            .display = display,
            .window = x11Window,
        };
        chain = &fromXlibWindow.chain;
    } break;
    case GLFW_PLATFORM_WAYLAND: {
        struct wl_display *const display = glfwGetWaylandDisplay();
        struct wl_surface *const surface = glfwGetWaylandWindow(window);
        fromWaylandSurface = (WGPUSurfaceDescriptorFromWaylandSurface){
            .chain = { .sType = WGPUSType_SurfaceDescriptorFromWaylandSurface },
            .display = display,
            .surface = surface,
        };
        chain = &fromWaylandSurface.chain;
    } break;
    default:
        return NULL;
    }
#endif

    return wgpuInstanceCreateSurface(
        instance,
        &(WGPUSurfaceDescriptor){
            .nextInChain = chain,
        }
    );
}
