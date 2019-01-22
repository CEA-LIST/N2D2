[{{SECTION_NAME}}_1x1] conv_def
Input={{INPUT}}
KernelDims=1 1
NbOutputs={{NB_FILTERS}}
Stride=1
Padding=0
WeightsFiller.Scaling=$(${L}**(-1.0/(2*3-2)) if ${L} > 0 else 1.0)

[{{SECTION_NAME}}_3x3] conv_def
Input={{SECTION_NAME}}_1x1
KernelDims=3 3
NbOutputs={{NB_FILTERS}}
Stride={{STRIDES}}
Padding=1
WeightsFiller.Scaling=$(${L}**(-1.0/(2*3-2)) if ${L} > 0 else 1.0)

[{{SECTION_NAME}}_1x1_x4] conv_def
Input={{SECTION_NAME}}_3x3
KernelDims=1 1
NbOutputs=$(4 * {{NB_FILTERS}})
Stride=1
Padding=0
WeightsFiller.Scaling=$(0.0 if ${L} > 0 else 1.0)
ActivationFunction=Linear

{% if PROJECTION_SHORTCUT exists %}
[{{SECTION_NAME}}_1x1_proj] conv_def
Input={{INPUT}}
KernelDims=1 1
NbOutputs=$(4 * {{NB_FILTERS}})
Stride={{STRIDES}}
Padding=0
ActivationFunction=Linear

{% set INPUT %}{{SECTION_NAME}}_1x1_proj{% endset %}
{% endif %}

[{{SECTION_NAME}}_sum]
Input={{SECTION_NAME}}_1x1_x4,{{INPUT}}
Type=ElemWise
NbOutputs=$(4 * {{NB_FILTERS}})
Operation=Sum
{% if NO_RELU not_exists %}
ActivationFunction=Rectifier
{% endif %}
