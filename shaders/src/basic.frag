#version 460 core

layout(location = 0) in vec2 inUV;

layout(location = 0) out vec4 outColor;

layout(set = 1, binding = 0) uniform sampler uLinearSampler;
layout(set = 1, binding = 1) uniform texture2D uAlbedo;

void main() {
    outColor = texture(sampler2D(uAlbedo, uLinearSampler), inUV);
}
