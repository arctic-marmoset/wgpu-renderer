#version 460 core

#include "basic.glsl"

layout(location = 0) in vec3 inWorldPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec4 outColor;

layout(set = SAMPLER_SET_INDEX, binding = 0) uniform sampler uLinearSampler;

layout(set = TEXTURE_SET_INDEX, binding = 0) uniform texture2D uAlbedo;

const vec3 kLightColor = vec3(0.86, 0.65, 0.35);
const vec3 kLightDirection = normalize(vec3(1.0, -1.0, 1.0));
const float kAmbientAmount = 0.1;

void main() {
    const vec4 textureSample = texture(sampler2D(uAlbedo, uLinearSampler), inUV);
    const vec3 diffuseColor = textureSample.rgb;
    const float specularAmount = textureSample.a;
    const vec3 worldNormal = normalize(inNormal);
    const vec3 cameraDirection = normalize(uFrame.cameraPosition - inWorldPosition);

    const vec3 ambientLight = kAmbientAmount * kLightColor;

    const float diffuseAmount = max(0.0, dot(worldNormal, kLightDirection));
    const vec3 diffuseLight = diffuseAmount * kLightColor;

    const vec3 reflectDirection = reflect(-kLightDirection, worldNormal);
    const float specularLight = specularAmount * pow(max(0.0, dot(cameraDirection, reflectDirection)), 32.0);

    const vec3 color =
        (ambientLight + diffuseLight) * diffuseColor +
        specularLight * kLightColor;

    outColor = vec4(color, 1.0);
}
