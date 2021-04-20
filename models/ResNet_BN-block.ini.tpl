[{{SECTION_NAME}}_3x3_1] conv_def
Input={{INPUT}}
KernelDims=3 3
NbOutputs={{NB_FILTERS}}
Stride={{STRIDES}}
Padding=1
WeightsFiller.Scaling=$(${L}**(-1.0/(2*2-2)) if ${L} > 0 else 1.0)

[{{SECTION_NAME}}_bn_1] bn_def
Input={{SECTION_NAME}}_3x3_1
NbOutputs=[{{SECTION_NAME}}_3x3_1]NbOutputs

[{{SECTION_NAME}}_3x3_2] conv_def
Input={{SECTION_NAME}}_bn_1
KernelDims=1 1
NbOutputs=$({{NB_FILTERS}})
Stride=1
Padding=0
WeightsFiller.Scaling=$(0.0 if ${L} > 0 else 1.0)

[{{SECTION_NAME}}_bn_2] bn_def
Input={{SECTION_NAME}}_3x3_2
NbOutputs=[{{SECTION_NAME}}_3x3_2]NbOutputs
ActivationFunction=Linear

{% if PROJECTION_SHORTCUT exists %}
[{{SECTION_NAME}}_1x1_proj] conv_def
Input={{INPUT}}
KernelDims=1 1
NbOutputs=$({{NB_FILTERS}})
Stride={{STRIDES}}
Padding=0

[{{SECTION_NAME}}_bn_proj] bn_def
Input={{SECTION_NAME}}_1x1_proj
NbOutputs=[{{SECTION_NAME}}_1x1_proj]NbOutputs
ActivationFunction=Linear

{% set INPUT %}{{SECTION_NAME}}_bn_proj{% endset %}
{% endif %}

[{{SECTION_NAME}}_sum]
Input={{SECTION_NAME}}_bn_2,{{INPUT}}
Type=ElemWise
NbOutputs={{NB_FILTERS}}
Operation=Sum
ActivationFunction=Rectifier
