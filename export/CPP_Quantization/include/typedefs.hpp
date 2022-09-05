/**
 ******************************************************************************
 * @file     typedefs.hpp
 * @brief    File to contain the structs and enums for the network functions
 *           It provides two datatypes to build quantized layers with any 
 *           precision (number of bits)
 * 
 ******************************************************************************
 * @attention
 * 
 * (C) Copyright 2021 CEA LIST. All Rights Reserved.
 *  Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
 *                  Inna KUCHER (inna.kucher@cea.fr)
 *                  Vincent TEMPLIER (vincent.templier@cea.fr)
 * 
 * This software is governed by the CeCILL-C license under French law and
 * abiding by the rules of distribution of free software. You can use,
 * modify and/ or redistribute the software under the terms of the CeCILL-C
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 * 
 * As a counterpart to the access to the source code and rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty and the software's author, the holder of the
 * economic rights, and the successive licensors have only limited
 * liability.
 * 
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL-C license and that you accept its terms.
 * 
 ******************************************************************************
 */

#ifndef __N2D2_EXPORTCPP_TYPEDEFS_HPP__
#define __N2D2_EXPORTCPP_TYPEDEFS_HPP__

#include <cmath>
#include <type_traits>
#include <limits>
#include <cstdint>


// ----------------------------------------------------------------------------
// -------------------- Generic custom bit-width types ------------------------
// ----------------------------------------------------------------------------

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
    struct is_unsigned<udata<BITWIDTH>>
        : std::is_unsigned<decltype(udata<BITWIDTH>::value)>::type {};

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


// ----------------------------------------------------------------------------
// -------------- Custom bit-width types operator overloading -----------------
// ----------------------------------------------------------------------------

// data
// template<int BITWIDTH, typename T>
// constexpr data<BITWIDTH>& operator+=(data<BITWIDTH>& d, T rhs)
//     {return d.value += decltype(data<BITWIDTH>::value)(rhs);}

// template<int BITWIDTH, typename T>
// constexpr data<BITWIDTH> operator+(data<BITWIDTH> d, T rhs)
//     {return d += rhs;}

// template<int BITWIDTH, typename T>
// constexpr data<BITWIDTH>& operator-=(data<BITWIDTH>& d, T rhs)
//     {return d.value -= decltype(data<BITWIDTH>::value)(rhs);}

// template<int BITWIDTH, typename T>
// constexpr data<BITWIDTH> operator-(data<BITWIDTH> d, T rhs)
//     {return d -= rhs;}

// template<int BITWIDTH, typename T>
// constexpr data<BITWIDTH>& operator*=(data<BITWIDTH>& d, T rhs)
//     {return d.value *= decltype(data<BITWIDTH>::value)(rhs);}

// template<int BITWIDTH, typename T>
// constexpr data<BITWIDTH> operator*(data<BITWIDTH> d, T rhs)
//     {return d *= rhs;}

// template<int BITWIDTH, typename T>
// constexpr data<BITWIDTH>& operator/=(data<BITWIDTH>& d, T rhs)
//     {return d.value /= decltype(data<BITWIDTH>::value)(rhs);}

// template<int BITWIDTH, typename T>
// constexpr data<BITWIDTH> operator/(data<BITWIDTH> d, T rhs)
//     {return d /= rhs;}


// udata
// template<int BITWIDTH, typename T>
// constexpr udata<BITWIDTH>& operator+=(udata<BITWIDTH>& d, T rhs)
//     {return d.value += decltype(udata<BITWIDTH>::value)(rhs);}

// template<int BITWIDTH, typename T>
// constexpr udata<BITWIDTH> operator+(udata<BITWIDTH> d, T rhs)
//     {return d += rhs;}

// template<int BITWIDTH, typename T>
// constexpr udata<BITWIDTH>& operator-=(udata<BITWIDTH>& d, T rhs)
//     {return d.value -= decltype(udata<BITWIDTH>::value)(rhs);}

// template<int BITWIDTH, typename T>
// constexpr udata<BITWIDTH> operator-(udata<BITWIDTH> d, T rhs)
//     {return d -= rhs;}

// template<int BITWIDTH, typename T>
// constexpr udata<BITWIDTH>& operator*=(udata<BITWIDTH>& d, T rhs)
//     {return d.value *= decltype(udata<BITWIDTH>::value)(rhs);}

// template<int BITWIDTH, typename T>
// constexpr udata<BITWIDTH> operator*(udata<BITWIDTH> d, T rhs)
//     {return d *= rhs;}

// template<int BITWIDTH, typename T>
// constexpr udata<BITWIDTH>& operator/=(udata<BITWIDTH>& d, T rhs)
//     {return d.value /= decltype(udata<BITWIDTH>::value)(rhs);}

// template<int BITWIDTH, typename T>
// constexpr udata<BITWIDTH> operator/(udata<BITWIDTH> d, T rhs)
//     {return d /= rhs;}


// ----------------------------------------------------------------------------
// ---------------- Custom bit-width types specializations --------------------
// ----------------------------------------------------------------------------

// Data structure for double
template <>
struct data<-64>
{
    data<-64>() = default;
    constexpr data<-64>(double v): value(v) {};
    constexpr operator double() const { return value; }
    union {
        double value;
    };
};

// Data structure for float
template <>
struct data<-32>
{
    data<-32>() = default;
    constexpr data<-32>(float v): value(v) {};
    constexpr operator float() const { return value; }
    union {
        float value;
    };
};

// Data structure for half float
template <>
struct data<-16>
{
    data<-16>() = default;
    constexpr data<-16>(float v): value(v) {};
    constexpr operator float() const { return value; }
    union {
        float value;
    };
};

// Data structure for int32
template <>
struct data<32>
{
    data<32>() = default;
    constexpr data<32>(int32_t v): value(v) {};
    constexpr operator int32_t() const { return value; }
    union {
        int32_t value;
    };
};

// Data structure for uint32
template <>
struct udata<32>
{
    udata<32>() = default;
    constexpr udata<32>(uint32_t v): value(v) {};
    constexpr operator uint32_t() const { return value; }
    union {
        uint32_t value;
    };
};

// Data structure for int16
template <>
struct data<16>
{
    data<16>() = default;
    constexpr data<16>(int16_t v): value(v) {};
    constexpr operator int16_t() const { return value; }
    union {
        int16_t value;
    };
};

// Data structure for uint16
template <>
struct udata<16>
{
    udata<16>() = default;
    constexpr udata<16>(uint16_t v): value(v) {};
    constexpr operator uint16_t() const { return value; }
    union {
        uint16_t value;
    };
};

// Data structure for int8
template <>
struct data<8>
{
    data<8>() = default;
    constexpr data<8>(int8_t v): value(v) {};
    constexpr operator int8_t() const { return value; }
    union {
        int8_t value;
    };
};

// Data structure for uint8
template <>
struct udata<8>
{
    udata<8>() = default;
    constexpr udata<8>(uint8_t v): value(v) {};
    constexpr operator uint8_t() const { return value; }
    union {
        uint8_t value;
    };
};

// Data structure for int7
template <>
struct data<7>
{
    data<7>() = default;
    constexpr data<7>(int8_t v): value(v) {};
    constexpr operator int8_t() const { return value; }
    union {
        int8_t value;
    };
};

// Data structure for uint7
template <>
struct udata<7>
{
    udata<7>() = default;
    constexpr udata<7>(uint8_t v): value(v) {};
    constexpr operator uint8_t() const { return value; }
    union {
        uint8_t value;
    };
};

// Data structure for int6
template <>
struct data<6>
{
    data<6>() = default;
    constexpr data<6>(int8_t v): value(v) {};
    constexpr operator int8_t() const { return value; }
    union {
        int8_t value;
    };
};

// Data structure for uint6
template <>
struct udata<6>
{
    udata<6>() = default;
    constexpr udata<6>(uint8_t v): value(v) {};
    constexpr operator uint8_t() const { return value; }
    union {
        uint8_t value;
    };
};

// Data structure for int5
template <>
struct data<5>
{
    data<5>() = default;
    constexpr data<5>(int8_t v): value(v) {};
    constexpr operator int8_t() const { return value; }
    union {
        int8_t value;
    };
};

// Data structure for uint5
template <>
struct udata<5>
{
    udata<5>() = default;
    constexpr udata<5>(uint8_t v): value(v) {};
    constexpr operator uint8_t() const { return value; }
    union {
        uint8_t value;
    };
};

// Data structure for 2 * int4
template <>
struct data<4>
{
    data<4>() = default;
    constexpr data<4>(uint8_t v): value(v) {};
    constexpr operator uint8_t() const { return value; }
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

// Data structure for 2 * uint4
template <>
struct udata<4>
{
    udata<4>() = default;
    constexpr udata<4>(uint8_t v): value(v) {};
    constexpr operator uint8_t() const { return value; }
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

// Data structure for 2 * int3
template <>
struct data<3>
{
    data<3>() = default;
    constexpr data<3>(uint8_t v): value(v) {};
    constexpr operator uint8_t() const { return value; }
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

// Data structure for 2 * uint3
template <>
struct udata<3>
{
    udata<3>() = default;
    constexpr udata<3>(uint8_t v): value(v) {};
    constexpr operator uint8_t() const { return value; }
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

// Data structure for 4 * int2
template <>
struct data<2>
{
    data<2>() = default;
    constexpr data<2>(uint8_t v): value(v) {};
    constexpr operator uint8_t() const { return value; }
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

// Data structure for 4 * uint2
template <>
struct udata<2>
{
    udata<2>() = default;
    constexpr udata<2>(uint8_t v): value(v) {};
    constexpr operator uint8_t() const { return value; }
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

// Data structure for 8 * int1
template <>
struct data<1>
{
    data<1>() = default;
    constexpr data<1>(uint8_t v): value(v) {};
    constexpr operator uint8_t() const { return value; }
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

// Data structure for 8 * uint1
template <>
struct udata<1>
{
    udata<1>() = default;
    constexpr udata<1>(uint8_t v): value(v) {};
    constexpr operator uint8_t() const { return value; }
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


// ----------------------------------------------------------------------------
// ------------------------- Structures and Unions ----------------------------
// ----------------------------------------------------------------------------

/* Object for compressing the outputs after mac operations */
typedef struct PackSupport {
    uint8_t         accumulator;
    unsigned int    cptAccumulator;
} PackSupport;

/* Union to access the data<32>/data<8>/data<4>/data<1> types */
union n2d2_dataword
{
    data<32> word;
    data<8> bytes[4];
    data<4> half_bytes[4];
    data<1> bitfields[4];
};

/* Union to access the udata<32>/udata<8>/udata<4>/udata<1> types */
union n2d2_udataword
{
    udata<32> word;
    udata<8> bytes[4];
    udata<4> half_bytes[4];
    udata<1> bitfields[4];
};


#endif  // __N2D2_EXPORTCPP_TYPEDEFS_HPP__
