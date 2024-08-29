#version 460 core

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 outColor;

layout(std140, set = 0, binding = 0) uniform Ubo
{
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

// NOTE: I switched to GLSL because glslc actually inserts a transpose operation
// for matrix multiplications in HLSL and stuff wasn't rendering on the screen.
// I might switch back to HLSL when I figure out how to prevent that.
void main() {
    const vec4 modelPosition = vec4(inPosition, 1.0);
    const vec4 worldPosition = ubo.model * modelPosition;
    const vec4 viewPosition = ubo.view * worldPosition;
    const vec4 clipPosition = ubo.proj * viewPosition;
    gl_Position = clipPosition;
    outColor = inColor;
}
