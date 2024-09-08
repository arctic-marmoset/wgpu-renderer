#version 460 core

#include "basic.glsl"

layout(location = 0) in vec2 inUV;

layout(location = 0) out vec4 outColor;

layout(set = SAMPLER_SET_INDEX, binding = 0) uniform sampler uLinearSampler;

layout(set = TEXTURE_SET_INDEX, binding = 0) uniform texture2D uAlbedo;

void main() {
    outColor = texture(sampler2D(uAlbedo, uLinearSampler), inUV);
}
