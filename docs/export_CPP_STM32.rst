Export: C++/STM32
=================

**N2D2-IP only: available upon request.**

Export type: ``CPP_STM32``
 C++ export for STM32.

::

    n2d2 MobileNet_ONNX.ini -seed 1 -w /dev/null -export CPP_STM32

Principle
---------

This export inherit the properties and optimizations from the C++ export, but
includes optimized kernels for the Cortex-M4 and the Cortex-M7. Please refer
to the :ref:`export_cpp-label` for the available export parameters.

SIMD
  The ``SMLAD`` intrinsic is used to do two 16-bit signed integers multiplications with
  accumulation. To extend the 8-bit data to the necessary 16-bit, the ``XTB16`` intrinsic is used.

Loop unrolling
  The unrolling of the loops can be done with ``#pragma GCC unroll NB_ITERATIONS``
  but it does not always perform as well as expected. Some loops are manually unrolled instead using C++
  templates. This increases the size of the compiled binary further but it provides a faster inference.

Usage of intrinsics
  Intrinsics provided by ARM are preferred to normal library methods calls
  when possible. For example the ``SSAT`` and ``USAT`` intrinsics are used to clamp the output value resulting
  in better results than a naive call to the std::clamp method.


Usage
-----

::

    n2d2 MobileNet_ONNX.ini -seed 1 -w /dev/null -export CPP_STM32 -fuse -nbbits 8 -calib -1 -db-export 100 -test

This command generates a C++ project in the sub-directory ``export_CPP_STM32_int8``.
This project is ready to be cross-compiled with a ``Makefile``, using the
*GNU Arm Embedded Toolchain* (which provides the ``arm-none-eabi-gcc`` compiler).

``make``
  To cross-compile the project using the *GNU Arm Embedded Toolchain*. An ELF
  binary file is generated in ``bin/n2d2_stm32.elf``.

``make flash``
  To flash the board using OpenOCD with the previously generated ``bin/n2d2_stm32.elf`` binary. In the
  provided Makefile, the default OpenOCD location is ``/usr/local/bin/openocd``
  and the default script is ``stm32h7x3i_eval.cfg``, for the STM32H7x3I evaluation
  board family. These can be changed in the first lines of the Makefile.
