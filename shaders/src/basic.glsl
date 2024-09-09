#ifndef SHADER_BASIC_GLSL
#define SHADER_BASIC_GLSL

layout(std140, set = FRAME_SET_INDEX, binding = 0) uniform Frame
{
    mat4 view;
    mat4 proj;
    vec3 cameraPosition;
} uFrame;

#endif
