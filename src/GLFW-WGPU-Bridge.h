#pragma once

#include <GLFW/glfw3.h>

#include <wgpu.h>

#ifdef __cplusplus
extern "C" {
#endif

WGPUSurface glfwCreateWGPUSurface(WGPUInstance instance, GLFWwindow *window);

#ifdef __cplusplus
}
#endif
