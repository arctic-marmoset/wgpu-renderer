#pragma shader_stage(fragment)

#include "common.hlsl"

float4 main(VSOutput input) : SV_TARGET
{
    return float4(input.Color, 1.0);
}
