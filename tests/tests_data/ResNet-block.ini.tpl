[{{SECTION_NAME}}_3x3_1] conv_def
Input={{INPUT}}
KernelDims=3 3
NbOutputs={{NB_FILTERS}}
Stride={{STRIDES}}
Padding=1

[{{SECTION_NAME}}_3x3_2] conv_def
Input={{SECTION_NAME}}_3x3_1
KernelDims=1 1
NbOutputs=$({{NB_FILTERS}})
Stride=1
Padding=0
ActivationFunction=Linear

{% if PROJECTION_SHORTCUT exists %}
[{{SECTION_NAME}}_1x1_proj] conv_def
Input={{INPUT}}
KernelDims=1 1
NbOutputs=$({{NB_FILTERS}})
Stride={{STRIDES}}
Padding=0
ActivationFunction=Linear

{% set INPUT %}{{SECTION_NAME}}_1x1_proj{% endset %}
{% endif %}

[{{SECTION_NAME}}_sum]
Input={{SECTION_NAME}}_3x3_2,{{INPUT}}
Type=ElemWise
NbOutputs={{NB_FILTERS}}
Operation=Sum
ActivationFunction=Rectifier
