/*
    (C) Copyright 2015 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    Inna KUCHER (inna.kucher@cea.fr)
                    Vincent TEMPLIER (vincent.templier@cea.fr)

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

#ifndef N2D2_EXPORTCPP_TYPEDEFS_HPP
#define N2D2_EXPORTCPP_TYPEDEFS_HPP

#include <type_traits>
#include <limits>

// Generic custom bit-width types
template <int BITWIDTH>
struct data {};

template <int BITWIDTH>
struct udata{};

namespace std {
    // Specialization of STL, allows to use std::is_unsigned<> for example.
    template <int BITWIDTH>
    struct is_integral<data<BITWIDTH>>
        : std::is_integral<decltype(data<BITWIDTH>::value)>::type {};
    template <int BITWIDTH>
    struct is_integral<udata<BITWIDTH>>
        : std::is_integral<decltype(udata<BITWIDTH>::value)>::type {};
    template <int BITWIDTH>
    struct is_floating_point<data<BITWIDTH>>
        : std::is_floating_point<decltype(data<BITWIDTH>::value)>::type {};
    template <int BITWIDTH>
    struct is_unsigned<data<BITWIDTH>>
        : std::is_unsigned<decltype(data<BITWIDTH>::value)>::type {};

    template <int BITWIDTH>
    class numeric_limits<data<BITWIDTH>> {
    public:
        static constexpr int is_integer = (BITWIDTH > 0);
        static constexpr int is_signed = true;
        static constexpr int digits = std::abs(BITWIDTH);
        static constexpr decltype(data<BITWIDTH>::value) min() noexcept
            { return (BITWIDTH > 0) ? -(1 << (BITWIDTH - 1)) :
                std::numeric_limits<decltype(data<BITWIDTH>::value)>::min(); };
        static constexpr decltype(data<BITWIDTH>::value) lowest() noexcept
            { return (BITWIDTH > 0) ? -(1 << (BITWIDTH - 1)) :
                std::numeric_limits<decltype(data<BITWIDTH>::value)>::lowest(); };
        static constexpr decltype(data<BITWIDTH>::value) max() noexcept
            { return (BITWIDTH > 0) ? ((1 << (BITWIDTH - 1)) - 1) :
                std::numeric_limits<decltype(data<BITWIDTH>::value)>::max(); };
    };
    template <int BITWIDTH>
    class numeric_limits<udata<BITWIDTH>> {
    public:
        static constexpr int is_integer = true;
        static constexpr int is_signed = false;
        static constexpr int digits = BITWIDTH;
        static constexpr decltype(data<BITWIDTH>::value) min() noexcept
            { return 0; };
        static constexpr decltype(data<BITWIDTH>::value) lowest() noexcept
            { return 0; };
        static constexpr decltype(data<BITWIDTH>::value) max() noexcept
            { return ((1 << BITWIDTH) - 1); };
    };
}

// Custom bit-width types specializations
template <>
struct data<-64>
{
    data<-64>() = default;
    constexpr data<-64>(double v): value(v) {};
    constexpr operator double() { return value; }
    union {
        double value;
    };
};

template <>
struct data<-32>
{
    data<-32>() = default;
    constexpr data<-32>(float v): value(v) {};
    constexpr operator float() { return value; }
    union {
        float value;
    };
};

template <>
struct data<-16>
{
    data<-16>() = default;
    constexpr data<-16>(float v): value(v) {};
    constexpr operator float() { return value; }
    union {
        float value;
    };
};

template <>
struct data<32>
{
    data<32>() = default;
    constexpr data<32>(int32_t v): value(v) {};
    constexpr operator int32_t() { return value; }
    union {
        int32_t value;
    };
};

template <>
struct udata<32>
{
    udata<32>() = default;
    constexpr udata<32>(uint32_t v): value(v) {};
    constexpr operator uint32_t() { return value; }
    union {
        uint32_t value;
    };
};

template <>
struct data<16>
{
    data<16>() = default;
    constexpr data<16>(int16_t v): value(v) {};
    constexpr operator int16_t() { return value; }
    union {
        int16_t value;
    };
};

template <>
struct udata<16>
{
    udata<16>() = default;
    constexpr udata<16>(uint16_t v): value(v) {};
    constexpr operator uint16_t() { return value; }
    union {
        uint16_t value;
    };
};

template <>
struct data<8>
{
    data<8>() = default;
    constexpr data<8>(int8_t v): value(v) {};
    constexpr operator int8_t() { return value; }
    union {
        int8_t value;
    };
};

template <>
struct udata<8>
{
    udata<8>() = default;
    constexpr udata<8>(uint8_t v): value(v) {};
    constexpr operator uint8_t() { return value; }
    union {
        uint8_t value;
    };
};

template <>
struct data<4>
{
    data<4>() = default;
    constexpr data<4>(uint8_t v): value(v) {};
    constexpr operator uint8_t() { return value; }
    union {
        int8_t value;
        uint8_t uvalue;
        struct
        {
            int8_t op0 : 4;
            int8_t op1 : 4;
        } fields;
    };
};

template <>
struct udata<4>
{
    udata<4>() = default;
    constexpr udata<4>(uint8_t v): value(v) {};
    constexpr operator uint8_t() { return value; }
    union {
        uint8_t value;
        uint8_t uvalue;
        struct
        {
            uint8_t op0 : 4;
            uint8_t op1 : 4;
        } fields;
    };
};

template <>
struct data<2>
{
    data<2>() = default;
    constexpr data<2>(uint8_t v): value(v) {};
    constexpr operator uint8_t() { return value; }
    union {
        int8_t value;
        uint8_t uvalue;
        struct
        {
            int8_t op0 : 2;
            int8_t op1 : 2;
            int8_t op2 : 2;
            int8_t op3 : 2;
        } fields;
    };
};

template <>
struct udata<2>
{
    udata<2>() = default;
    constexpr udata<2>(uint8_t v): value(v) {};
    constexpr operator uint8_t() { return value; }
    union {
        uint8_t value;
        uint8_t uvalue;
        struct
        {
            uint8_t op0 : 2;
            uint8_t op1 : 2;
            uint8_t op2 : 2;
            uint8_t op3 : 2;
        } fields;
    };
};

template <>
struct data<1>
{
    data<1>() = default;
    constexpr data<1>(uint8_t v): value(v) {};
    constexpr operator uint8_t() { return value; }
    union {
        int8_t value;
        uint8_t uvalue;
        struct
        {
            int8_t op0 : 1;
            int8_t op1 : 1;
            int8_t op2 : 1;
            int8_t op3 : 1;
            int8_t op4 : 1;
            int8_t op5 : 1;
            int8_t op6 : 1;
            int8_t op7 : 1;
        } fields;
    };
};

template <>
struct udata<1>
{
    udata<1>() = default;
    constexpr udata<1>(uint8_t v): value(v) {};
    constexpr operator uint8_t() { return value; }
    union {
        uint8_t value;
        uint8_t uvalue;
        struct
        {
            uint8_t op0 : 1;
            uint8_t op1 : 1;
            uint8_t op2 : 1;
            uint8_t op3 : 1;
            uint8_t op4 : 1;
            uint8_t op5 : 1;
            uint8_t op6 : 1;
            uint8_t op7 : 1;
        } fields;
    };
};

typedef struct PackSupport {
    uint8_t         accumulator;
    unsigned int    cptAccumulator;
} PackSupport;


#endif // N2D2_EXPORTCPP_TYPEDEFS_HPP
