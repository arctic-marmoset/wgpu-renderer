#include "GLFW-WGPU-Bridge.h"

#import <QuartzCore/QuartzCore.h>

#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3native.h>

#include <assert.h>
#include <stddef.h>

WGPUSurface glfwCreateWGPUSurface(WGPUInstance instance, GLFWwindow *const window) {
    if (!window) {
        return NULL;
    }
    assert("GLFWwindow should not be NULL" && window);

    NSWindow *const nsWindow = glfwGetCocoaWindow(window);
    NSView *const contentView = nsWindow.contentView;
    if (!contentView) {
        return NULL;
    }
    assert("NSWindow.contentView should not be nil" && nsWindow.contentView);

    CAMetalLayer *const layer = [CAMetalLayer layer];
    contentView.wantsLayer = YES;
    contentView.layer = layer;

    const WGPUSurfaceDescriptorFromMetalLayer fromMetalLayer = {
        .chain = { .sType = WGPUSType_SurfaceDescriptorFromMetalLayer },
        .layer = layer,
    };

    return wgpuInstanceCreateSurface(
        instance,
        &(WGPUSurfaceDescriptor){
            .nextInChain = &fromMetalLayer.chain,
        }
    );
}
