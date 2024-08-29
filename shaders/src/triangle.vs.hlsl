#pragma shader_stage(vertex)

#include "common.hlsl"

struct VSInput
{
    [[vk::location(0)]] float3 Position : POSITION;
    [[vk::location(1)]] float3 Color : COLOR;
};

VSOutput main(VSInput input)
{
    VSOutput output;
    output.ClipPosition = float4(input.Position, 1.0);
    output.Color = input.Color;
    return output;
}
