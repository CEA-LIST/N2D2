# N2D2 STM32 CPP Export

## Requirements

To compile the project:

* GCC (emulation)
* [arm-none-eabi-gcc](https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-rm/downloads), the GNU Arm Embedded Toolchain (STM32)

To flash on board :

* STM32H7 board (all tests have been made on STM32H743ZIT6)
* STM32L4 board (all tests have been made on STM32L4R5ZIT6)
* [STM32CubeProgrammer](https://www.st.com/en/development-tools/stm32cubeprog.html)


## Compile the export

### For emulation

To compile the export for emulation purpose, run
```
make emulator
```

You should have a binary named `n2d2_stm32_emulator` in the `/bin` folder.

If you want to save the outputs of each layer for studying, add the *SAVE_OUTPUTS* option
```
make emulator SAVE_OUTPUTS=1
```

### For STM32H7

To compile the export for STM32H7, run
```
make export_h7
```

You should have a binary named `n2d2_stm32_h7.elf` in the `/bin` folder.

### For STM32L4

To compile the export for STM32H7, run
```
make export_l4
```

You should have a binary named `n2d2_stm32_l4.elf` in the `/bin` folder.


## Test the export

### In emulation

To test the emulator, you can run the binary

```
./bin/n2d2_stm32_emulator
```

### On board

After compiling the export, open the **STM32CubeProgrammer** software tool, connect the board and open **`n2d2_stm32_xx.elf`** with the `Open file` option. Then download the binary on board.

You can display the results of the project on MobaXterm 
(don't forget to push the `Reset` button to start the project on board)

You can also use [openocd](http://openocd.org/doc/html/About.html) library to flash the binary. 


## Details about the files

The information about your model are stored in `dnn/`. In `dnn/include`, you will find the information about the layers in your model while in `dnn/src`, you will find functions to use the model.

All information about the export architecture are stored in the `targets/`.