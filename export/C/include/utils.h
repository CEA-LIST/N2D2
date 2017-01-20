/*
    (C) Copyright 2014 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

    This software is governed by the CeCILL-C license under French law and
    abiding by the rules of distribution of free software.  You can  use,
    modify and/ or redistribute the software under the terms of the CeCILL-C
    license as circulated by CEA, CNRS and INRIA at the following URL
    "http://www.cecill.info".

    As a counterpart to the access to the source code and  rights to copy,
    modify and redistribute granted by the license, users are provided only
    with a limited warranty  and the software's author,  the holder of the
    economic rights,  and the successive licensors  have only  limited
    liability.

    The fact that you are presently reading this means that you have had
    knowledge of the CeCILL-C license and that you accept its terms.
*/

#ifndef N2D2_EXPORTC_UTILS_H
#define N2D2_EXPORTC_UTILS_H

#include <assert.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h> // (u)intx_t typedef

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#ifndef WIN32
// Text attributes
#define ESC_ALL_OFF "\033[0m"
#define ESC_BOLD "\033[1m"
#define ESC_UNDERSCORE "\033[4m"
#define ESC_BLINK "\033[5m"

// Foreground colors
#define ESC_FG_BLACK "\033[30m"
#define ESC_FG_RED "\033[31m"
#define ESC_FG_GREEN "\033[32m"
#define ESC_FG_YELLOW "\033[33m"
#define ESC_FG_BLUE "\033[34m"
#define ESC_FG_MAGENTA "\033[35m"
#define ESC_FG_CYAN "\033[36m"
#define ESC_FG_WHITE "\033[37m"
#define ESC_FG_GRAY "\033[90m"
#define ESC_FG_LIGHT_RED "\033[91m"
#define ESC_FG_LIGHT_GREEN "\033[92m"
#define ESC_FG_LIGHT_YELLOW "\033[93m"
#define ESC_FG_LIGHT_BLUE "\033[94m"
#define ESC_FG_LIGHT_MAGENTA "\033[95m"
#define ESC_FG_LIGHT_CYAN "\033[96m"

// Background colors
#define ESC_BG_BLACK "\033[40m"
#define ESC_BG_RED "\033[41m"
#define ESC_BG_GREEN "\033[42m"
#define ESC_BG_YELLOW "\033[43m"
#define ESC_BG_BLUE "\033[44m"
#define ESC_BG_MAGENTA "\033[45m"
#define ESC_BG_CYAN "\033[46m"
#define ESC_BG_WHITE "\033[47m"
#define ESC_BG_GRAY "\033[100m"
#define ESC_BG_LIGHT_RED "\033[101m"
#define ESC_BG_LIGHT_GREEN "\033[102m"
#define ESC_BG_LIGHT_YELLOW "\033[103m"
#define ESC_BG_LIGHT_BLUE "\033[104m"
#define ESC_BG_LIGHT_MAGENTA "\033[105m"
#define ESC_BG_LIGHT_CYAN "\033[106m"
#else
// Text attributes
#define ESC_ALL_OFF ""
#define ESC_BOLD ""
#define ESC_UNDERSCORE ""
#define ESC_BLINK ""

// Foreground colors
#define ESC_FG_BLACK ""
#define ESC_FG_RED ""
#define ESC_FG_GREEN ""
#define ESC_FG_YELLOW ""
#define ESC_FG_BLUE ""
#define ESC_FG_MAGENTA ""
#define ESC_FG_CYAN ""
#define ESC_FG_WHITE ""

// Background colors
#define ESC_BG_BLACK ""
#define ESC_BG_RED ""
#define ESC_BG_GREEN ""
#define ESC_BG_YELLOW ""
#define ESC_BG_BLUE ""
#define ESC_BG_MAGENTA ""
#define ESC_BG_CYAN ""
#define ESC_BG_WHITE ""
#endif

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

// min() and max()
static inline char char_max(char a, char b)
{
    return (a > b) ? a : b;
}
static inline char char_min(char a, char b)
{
    return (a < b) ? a : b;
}
static inline unsigned char uchar_max(unsigned char a, unsigned char b)
{
    return (unsigned char)((a > b) ? a : b);
}
static inline unsigned char uchar_min(unsigned char a, unsigned char b)
{
    return (unsigned char)((a < b) ? a : b);
}
static inline short short_max(short a, short b)
{
    return (a > b) ? a : b;
}
static inline short short_min(short a, short b)
{
    return (a < b) ? a : b;
}
static inline unsigned short ushort_max(unsigned short a, unsigned short b)
{
    return (unsigned short)((a > b) ? a : b);
}
static inline unsigned short ushort_min(unsigned short a, unsigned short b)
{
    return (unsigned short)((a < b) ? a : b);
}
static inline int int_max(int a, int b)
{
    return (a > b) ? a : b;
}
static inline int int_min(int a, int b)
{
    return (a < b) ? a : b;
}
static inline unsigned int uint_max(unsigned int a, unsigned int b)
{
    return (a > b) ? a : b;
}
static inline unsigned int uint_min(unsigned int a, unsigned int b)
{
    return (a < b) ? a : b;
}

static inline int8_t int8_max(int8_t a, int8_t b)
{
    return (a > b) ? a : b;
}
static inline int8_t int8_min(int8_t a, int8_t b)
{
    return (a < b) ? a : b;
}
static inline uint8_t uint8_max(uint8_t a, uint8_t b)
{
    return (a > b) ? a : b;
}
static inline uint8_t uint8_min(uint8_t a, uint8_t b)
{
    return (a < b) ? a : b;
}
static inline int16_t int16_max(int16_t a, int16_t b)
{
    return (a > b) ? a : b;
}
static inline int16_t int16_min(int16_t a, int16_t b)
{
    return (a < b) ? a : b;
}
static inline uint16_t uint16_max(uint16_t a, uint16_t b)
{
    return (a > b) ? a : b;
}
static inline uint16_t uint16_min(uint16_t a, uint16_t b)
{
    return (a < b) ? a : b;
}
static inline int32_t int32_max(int32_t a, int32_t b)
{
    return (a > b) ? a : b;
}
static inline int32_t int32_min(int32_t a, int32_t b)
{
    return (a < b) ? a : b;
}
static inline uint32_t uint32_max(uint32_t a, uint32_t b)
{
    return (a > b) ? a : b;
}
static inline uint32_t uint32_min(uint32_t a, uint32_t b)
{
    return (a < b) ? a : b;
}

// swap()
static inline void char_swap(char* a, char* b)
{
    char tmp = *a;
    *a = *b;
    *b = tmp;
}
static inline void uchar_swap(unsigned char* a, unsigned char* b)
{
    unsigned char tmp = *a;
    *a = *b;
    *b = tmp;
}
static inline void short_swap(short* a, short* b)
{
    short tmp = *a;
    *a = *b;
    *b = tmp;
}
static inline void ushort_swap(unsigned short* a, unsigned short* b)
{
    unsigned short tmp = *a;
    *a = *b;
    *b = tmp;
}
static inline void int_swap(int* a, int* b)
{
    int tmp = *a;
    *a = *b;
    *b = tmp;
}
static inline void uint_swap(unsigned int* a, unsigned int* b)
{
    unsigned int tmp = *a;
    *a = *b;
    *b = tmp;
}

static inline void int8_swap(int8_t* a, int8_t* b)
{
    int8_t tmp = *a;
    *a = *b;
    *b = tmp;
}
static inline void uint8_swap(uint8_t* a, uint8_t* b)
{
    uint8_t tmp = *a;
    *a = *b;
    *b = tmp;
}
static inline void int16_swap(int16_t* a, int16_t* b)
{
    int16_t tmp = *a;
    *a = *b;
    *b = tmp;
}
static inline void uint16_swap(uint16_t* a, uint16_t* b)
{
    uint16_t tmp = *a;
    *a = *b;
    *b = tmp;
}
static inline void int32_swap(int32_t* a, int32_t* b)
{
    int32_t tmp = *a;
    *a = *b;
    *b = tmp;
}
static inline void uint32_swap(uint32_t* a, uint32_t* b)
{
    uint32_t tmp = *a;
    *a = *b;
    *b = tmp;
}

static inline unsigned int isqrt(unsigned int x)
{
    unsigned int op, res, one;
    op = x;
    res = 0;

/* "one" starts at the highest power of four <= than the argument. */
#if UINT_MAX == 0xFFFFU
    one = 1 << 14; /* second-to-top bit set */
#elif UINT_MAX == 0xFFFFFFFFUL
    one = 1 << 30; /* second-to-top bit set */
#else
#error "isqrt(): unsupported unsigned int type size"
#endif

    while (one > op)
        one >>= 2;

    while (one != 0) {
        if (op >= res + one) {
            op -= res + one;
            res += one << 1; // <-- faster than 2 * one
        }

        res >>= 1;
        one >>= 2;
    }

    return res;
}

#endif // N2D2_EXPORTC_UTILS_H
