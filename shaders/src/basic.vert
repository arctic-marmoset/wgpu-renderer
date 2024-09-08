#version 460 core

#include "basic.glsl"

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 outWorldPosition;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec2 outUV;

layout(std140, set = FRAME_SET_INDEX, binding = 0) uniform Frame
{
    mat4 view;
    mat4 proj;
} uFrame;

layout(std140, set = MODEL_SET_INDEX, binding = 0) uniform Model
{
    mat4 model;
    mat3 normal;
} uModel;

// NOTE: I switched to GLSL because glslc actually inserts a transpose operation
// for matrix multiplications in HLSL and stuff wasn't rendering on the screen.
// I might switch back to HLSL when I figure out how to prevent that.
void main() {
    const vec4 modelPosition = vec4(inPosition, 1.0);
    const vec4 worldPosition = uModel.model * modelPosition;
    const vec4 viewPosition = uFrame.view * worldPosition;
    const vec4 clipPosition = uFrame.proj * viewPosition;
    gl_Position = clipPosition;
    outWorldPosition = worldPosition.xyz;
    outNormal = uModel.normal * inNormal;
    outUV = inUV;
}
