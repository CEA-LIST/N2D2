[{{SECTION_NAME}}_1x1] conv_def
Input={{INPUT}}
KernelDims=1 1
NbOutputs=$([{{INPUT}}]NbOutputs * {{EXPANSION}})
Stride=1
Padding=0
{% if RESIDUAL exists %}
WeightsFiller.Scaling=$(${L}**(-1.0/(2*3-2)) if ${L} > 0 else 1.0)
{% endif %}

[{{SECTION_NAME}}_3x3] conv_def
Input={{SECTION_NAME}}_1x1
KernelDims=3 3
NbOutputs=$([{{INPUT}}]NbOutputs * {{EXPANSION}})
Stride={{STRIDES}}
Padding=1
Mapping.ChannelsPerGroup=1
{% if RESIDUAL exists %}
WeightsFiller.Scaling=$(${L}**(-1.0/(2*3-2)) if ${L} > 0 else 1.0)
{% endif %}

[{{SECTION_NAME}}_1x1_linear]
Input={{SECTION_NAME}}_3x3
Type=Conv
KernelDims=1 1
NbOutputs={{NB_FILTERS}}
Stride=1
Padding=0
WeightsFiller=HeFiller
{% if RESIDUAL exists %}
WeightsFiller.Scaling=$(0.0 if ${L} > 0 else 1.0)
{% endif %}
ActivationFunction=Linear
ConfigSection=common.config

{% if RESIDUAL exists %}
[{{SECTION_NAME}}_sum]
Input={{SECTION_NAME}}_1x1_linear,{{INPUT}}
Type=ElemWise
NbOutputs={{NB_FILTERS}}
Operation=Sum
{% endif %}
