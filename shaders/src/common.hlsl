#ifndef SHADER_COMMON_HLSL
#define SHADER_COMMON_HLSL

struct VSOutput
{
    float4 ClipPosition : SV_POSITION;
    [[vk::location(0)]] float3 Color : COLOR;
};

#endif
