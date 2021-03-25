Quantization aware training
===========================

**N2D2-IP only: available upon request.**

Currently, two quantization aware training (QAT) methods are implemented:

- SAT :cite:`jin2019efficient`;
- LSQ :cite:`bhalgat2020lsq`.

These two methods are currently at the top of the state-of-the-art, summarized
in the figure below. Each dot represents one DNN (from the MobileNet or ResNet 
family), quantized with the number of bits indicated beside.

.. figure:: _static/qat_sota.png
   :alt: QAT state-of-the-art.

