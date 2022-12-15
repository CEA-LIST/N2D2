OPENOCD := /usr/local/bin/openocd
OPENOCD_SCRIPT := /usr/local/share/openocd/scripts/board/stm32l4discovery.cfg


TARGET := bin/n2d2_stm32.elf
FLASH_LD := STM32L4R5ZITx_FLASH.ld
HAL_HEADER_FILE := stm32l4xx_hal.h
OBJDIR := build


COMMON_FLAGS := -DHAL_HEADER="<${HAL_HEADER_FILE}>" -DNDEBUG -Ofast -Wall -Wno-unused-variable -specs=nano.specs -c -mcpu=cortex-m4 -mfloat-abi=hard -mfpu=fpv4-sp-d16 -DUSE_HAL_DRIVER -DSTM32L4R5xx -MMD -mthumb
INCLUDE_DIRS := -IInc -I./Drivers/STM32L4xx_HAL_Driver/Inc -I./Drivers/STM32L4xx_HAL_Driver/Inc/Legacy -I./Drivers/CMSIS/Device/ST/STM32L4xx/Include -I./Drivers/CMSIS/Include -I../include -I../dnn/include -I.
LINK_FLAGS := -mcpu=cortex-m4 -mfloat-abi=hard -mfpu=fpv4-sp-d16 -T${FLASH_LD} -lc -lm -lnosys -specs=nano.specs -flto -Wl,--print-memory-usage


PREFIX := arm-none-eabi
CC := $(PREFIX)-gcc
CC_FLAGS := ${COMMON_FLAGS} -std=gnu11 
CC_SRCS := $(shell find -name '*.c')
CC_OBJS := $(patsubst %.c,$(OBJDIR)/%.c.o,$(CC_SRCS))
DEPENDENCIES := $(patsubst %.c.o,%.c.d,$(CC_OBJS))

CXX := $(PREFIX)-g++
CXX_FLAGS := ${COMMON_FLAGS} -std=c++14 -fno-exceptions -fno-rtti
CXX_SRCS := $(shell find . ../src ../dnn/src -name '*.cpp')
CXX_OBJS := $(patsubst %.cpp,$(OBJDIR)/%.cpp.o,$(CXX_SRCS))
DEPENDENCIES := $(DEPENDENCIES) $(patsubst %.cpp.o,%.cpp.d,$(CXX_OBJS))

ASM := $(PREFIX)-gcc
ASM_FLAGS := ${COMMON_FLAGS} -x assembler-with-cpp
ASM_SRCS := $(shell find -name '*.s')
ASM_OBJS := $(patsubst %.s,$(OBJDIR)/%.s.o,$(ASM_SRCS))




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

flash:
	${OPENOCD} -f ${OPENOCD_SCRIPT}  -c "init ; program ${TARGET} ; reset ; shutdown"

clean:
	rm -f $(DEPENDENCIES)
	rm -f ${CC_OBJS}
	rm -f ${CXX_OBJS}
	rm -f ${ASM_OBJS}
	rm -f ${TARGET}

-include $(DEPENDENCIES)
