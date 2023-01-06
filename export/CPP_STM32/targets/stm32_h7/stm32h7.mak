
TARGET := ${BINDIR}/n2d2_stm32_h7.elf
EXPORTDIR := targets/stm32_h7
FLASH_LD := ${EXPORTDIR}/STM32H743ZITx_FLASH.ld
HAL_HEADER_FILE := stm32h7xx_hal.h

INCLUDE_DIRS := -I. -I./include -I./dnn/include

COMMON_FLAGS := -DHAL_HEADER="<${HAL_HEADER_FILE}>" -DNDEBUG -Ofast -Wall -Wno-unused-variable -specs=nano.specs -c -mcpu=cortex-m7 -mfloat-abi=hard -mfpu=fpv5-d16 -DUSE_HAL_DRIVER -DSTM32H743xx -DARM_MATH_CM7 -MMD 
INCLUDE_DIRS := ${INCLUDE_DIRS} -I${EXPORTDIR}/Inc -I${EXPORTDIR}/Drivers/STM32H7xx_HAL_Driver/Inc -I${EXPORTDIR}/Drivers/STM32H7xx_HAL_Driver/Inc/Legacy -I${EXPORTDIR}/Drivers/CMSIS/Device/ST/STM32H7xx/Include -I${EXPORTDIR}/Drivers/CMSIS/Include
LINK_FLAGS := ${LINK_FLAGS} -mcpu=cortex-m7 -mfloat-abi=hard -mfpu=fpv5-d16 -T${FLASH_LD} -lc -lm -lnosys -specs=nano.specs -flto -Wl,--print-memory-usage


ifdef PRINTFLOAT
	LINK_FLAGS := ${LINK_FLAGS} -u _printf_float
endif

PREFIX := arm-none-eabi

CC :=${PREFIX}-gcc
CC_FLAGS := ${COMMON_FLAGS} -std=gnu11 
CC_SRCS := $(shell find ${EXPORTDIR} -name '*.c')
CC_OBJS := $(patsubst %.c, ${OBJDIR}/%.c.o, ${CC_SRCS})
DEPENDENCIES := $(patsubst %.c.o, %.c.d, ${CC_OBJS})

CXX :=${PREFIX}-g++
CXX_FLAGS := ${COMMON_FLAGS} -std=c++14 -fno-exceptions -fno-rtti
CXX_SRCS := $(shell find src dnn/src ${EXPORTDIR} -name '*.cpp')
CXX_OBJS := $(patsubst %.cpp, ${OBJDIR}/%.cpp.o, ${CXX_SRCS})
DEPENDENCIES := $(DEPENDENCIES) $(patsubst %.cpp.o, %.cpp.d, ${CXX_OBJS})

ASM :=${PREFIX}-gcc
ASM_FLAGS := ${COMMON_FLAGS} -x assembler-with-cpp
ASM_SRCS := $(shell find ${EXPORTDIR} -name '*.s')
ASM_OBJS := $(patsubst %.s, ${OBJDIR}/%.s.o, ${ASM_SRCS})


all: build

build: ${CC_OBJS} ${CXX_OBJS} ${ASM_OBJS}
	@mkdir -p $(dir ${TARGET})
	${CC} ${CC_OBJS} ${CXX_OBJS} ${ASM_OBJS} ${LINK_FLAGS} -o ${TARGET} 

${OBJDIR}/%.c.o: %.c
	@mkdir -p $(dir $@)
	${CC} ${CC_FLAGS} ${INCLUDE_DIRS} -o $@ $<

${OBJDIR}/%.cpp.o: %.cpp
	@mkdir -p $(dir $@)
	${CXX} ${CXX_FLAGS} ${INCLUDE_DIRS} -o $@ $<

${OBJDIR}/%.s.o: %.s 
	@mkdir -p $(dir $@)
	${ASM} ${ASM_FLAGS} -o $@ $<


OPENOCD := /usr/local/bin/openocd
OPENOCD_SCRIPT := /usr/local/share/openocd/scripts/board/stm32h7x3i_eval.cfg

flash:
	${OPENOCD} -f ${OPENOCD_SCRIPT}  -c "init ; program ${TARGET} ; reset ; shutdown"
