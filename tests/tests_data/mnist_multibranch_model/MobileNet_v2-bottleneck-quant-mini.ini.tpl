[{{SECTION_NAME}}_1x1] conv_def
Input={{INPUT}}
KernelDims=1 1
NbOutputs=$([{{INPUT}}]NbOutputs * {{EXPANSION}})
Stride=1
Padding=0

[{{SECTION_NAME}}_bn_1x1] bn_def
Input={{SECTION_NAME}}_1x1
NbOutputs=[{{SECTION_NAME}}_1x1]NbOutputs
ConfigSection=bn.config

[{{SECTION_NAME}}_3x3] conv_def
Input={{SECTION_NAME}}_bn_1x1
KernelDims=3 3
NbOutputs=$([{{INPUT}}]NbOutputs * {{EXPANSION}})
Stride={{STRIDES}}
Padding=1
Mapping.ChannelsPerGroup=1

[{{SECTION_NAME}}_bn_3x3] bn_def
Input={{SECTION_NAME}}_3x3
NbOutputs=[{{SECTION_NAME}}_3x3]NbOutputs
ConfigSection=bn.config

[{{SECTION_NAME}}_1x1_linear] conv_def
Input={{SECTION_NAME}}_bn_3x3
Type=Conv
KernelDims=1 1
NbOutputs={{NB_FILTERS}}
Stride=1
Padding=0
ConfigSection=common.config

[{{SECTION_NAME}}_bn_1x1_linear] bn_def
Input={{SECTION_NAME}}_1x1_linear
NbOutputs=[{{SECTION_NAME}}_1x1_linear]NbOutputs
ConfigSection=bn.config

{% if RESIDUAL exists %}
[{{SECTION_NAME}}_bn_sum]
Input={{SECTION_NAME}}_bn_1x1_linear,{{INPUT}}
Type=ElemWise
NbOutputs={{NB_FILTERS}}
Operation=Sum
ActivationFunction=Linear
QAct=SAT
QAct.Range=${A_RANGE}
QAct.Alpha=${CONV_QUANT_ALPHA}
QActSolver=SGD
QActSolver.LearningRate=${LR}
QActSolver.LearningRatePolicy=${Policy}
QActSolver.Momentum=${MOMENTUM}
QActSolver.Decay=${WD}
{% endif %}
