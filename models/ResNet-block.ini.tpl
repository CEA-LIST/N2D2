[{{SECTION_NAME}}_3x3_1] conv_def
Input={{INPUT}}
KernelDims=3 3
NbOutputs={{NB_FILTERS}}
Stride={{STRIDES}}
Padding=1
WeightsFiller.Scaling=$(${L}**(-1.0/(2*2-2)) if ${L} > 0 else 1.0)
 
;NOTE: This has been corrected but not tested!!
[{{SECTION_NAME}}_3x3_2] conv_def
Input={{SECTION_NAME}}_3x3_1
KernelDims=3 3
NbOutputs=$({{NB_FILTERS}})
Stride=1
Padding=1
WeightsFiller.Scaling=$(0.0 if ${L} > 0 else 1.0)
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
