/**
 ******************************************************************************
 * @file    support_functions.hpp
 * @brief   Provide multiple general functions to use in other files.
 *          Also provide several functions to use with 
 *          the custom bit-width types defined in typedefs.hpp
 * 
 ******************************************************************************
 * @attention
 * 
 * (C) Copyright 2022 CEA LIST. All Rights Reserved.
 *  Contributor(s): Vincent TEMPLIER (vincent.templier@cea.fr)
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

#ifndef __N2D2_SUPPORT_FUNCTIONS_HPP__
#define __N2D2_SUPPORT_FUNCTIONS_HPP__

#include "typedefs.hpp"



// ----------------------------------------------------------------------------
// -------------- Custom bit-width types fprintf adaptation -------------------
// ----------------------------------------------------------------------------


/**
 * @brief   Display into a stream the values stored in a data<4> or udata<4>
 * 
 * @tparam  Data_T  data type (should be data<4> or udata<4>)
 * 
 * @param[in]   file        Pointer to the stream
 * @param[in]   dataValue   Data value to display
 * @returns                 None
 * 
 */
template<typename Data_T,
         typename std::enable_if<(std::numeric_limits<Data_T>::digits == 4
         || std::numeric_limits<Data_T>::digits == 3)>::type* = nullptr>
__attribute__((always_inline)) static inline
void fprintf_dataBitwidth_4 (FILE* file, 
                             const Data_T dataValue)
{
    if (std::is_unsigned<Data_T>::value) {
        fprintf(file, "%d, ", (uint32_t)dataValue.fields.op1);
        fprintf(file, "%d", (uint32_t)dataValue.fields.op0);
    } 
    else {
        fprintf(file, "%d, ", (int8_t)dataValue.fields.op1);
        fprintf(file, "%d", (int8_t)dataValue.fields.op0);
    }
}

template<typename Data_T,
         typename std::enable_if<!(std::numeric_limits<Data_T>::digits == 4
         || std::numeric_limits<Data_T>::digits == 3)>::type* = nullptr>
__attribute__((always_inline)) static inline
void fprintf_dataBitwidth_4 (FILE* file, 
                             const Data_T /*dataValue*/)
{
    fprintf(file, "NaN");
}


/**
 * @brief   Display into a stream the values stored in a data<2> or udata<2>
 * 
 * @tparam  Data_T  data type (should be data<2> or udata<2>)
 * 
 * @param[in]   file        Pointer to the stream
 * @param[in]   dataValue   Data value to display
 * @returns                 None
 * 
 */
template<typename Data_T,
         typename std::enable_if<(
            std::numeric_limits<Data_T>::digits == 2)>::type* = nullptr>
__attribute__((always_inline)) static inline
void fprintf_dataBitwidth_2 (FILE* file, 
                             const Data_T dataValue)
{
    if (std::is_unsigned<Data_T>::value) {
        fprintf(file, "%d, ", (uint32_t)dataValue.fields.op3);
        fprintf(file, "%d, ", (uint32_t)dataValue.fields.op2);
        fprintf(file, "%d, ", (uint32_t)dataValue.fields.op1);
        fprintf(file, "%d", (uint32_t)dataValue.fields.op0);
    } 
    else {
        fprintf(file, "%d, ", (int32_t)dataValue.fields.op3);
        fprintf(file, "%d, ", (int32_t)dataValue.fields.op2);
        fprintf(file, "%d, ", (int32_t)dataValue.fields.op1);
        fprintf(file, "%d", (int32_t)dataValue.fields.op0);
    }
}

template<typename Data_T,
         typename std::enable_if<(
            std::numeric_limits<Data_T>::digits != 2)>::type* = nullptr>
__attribute__((always_inline)) static inline
void fprintf_dataBitwidth_2 (FILE* file, 
                             const Data_T /*dataValue*/)
{
    fprintf(file, "NaN");
}


/**
 * @brief   Display into a stream the values stored in a data<1> or udata<1>
 * 
 * @tparam  Data_T  data type (should be data<1> or udata<1>)
 * 
 * @param[in]   file        Pointer to the stream
 * @param[in]   dataValue   Data value to display
 * @returns                 None
 * 
 */
template<typename Data_T,
         typename std::enable_if<(
            std::numeric_limits<Data_T>::digits == 1)>::type* = nullptr>
__attribute__((always_inline)) static inline
void fprintf_dataBitwidth_1 (FILE* file, 
                             const Data_T dataValue)
{
    fprintf(file, "%d, ", (uint32_t)dataValue.fields.op7);
    fprintf(file, "%d, ", (uint32_t)dataValue.fields.op6);
    fprintf(file, "%d, ", (uint32_t)dataValue.fields.op5);
    fprintf(file, "%d, ", (uint32_t)dataValue.fields.op4);
    fprintf(file, "%d, ", (uint32_t)dataValue.fields.op3);
    fprintf(file, "%d, ", (uint32_t)dataValue.fields.op2);
    fprintf(file, "%d, ", (uint32_t)dataValue.fields.op1);
    fprintf(file, "%d", (uint32_t)dataValue.fields.op0);
}

template<typename Data_T,
         typename std::enable_if<(
            std::numeric_limits<Data_T>::digits != 1)>::type* = nullptr>
__attribute__((always_inline)) static inline
void fprintf_dataBitwidth_1 (FILE* file, 
                             const Data_T /*dataValue*/)
{
    fprintf(file, "NaN");
}


/**
 * @brief   Display into a stream the values stored in 
 *          a data<BITWIDTH> or udata<BITWIDTH>
 * @details Different displays depending the format 
 *          and bitwidth of the data structure. 
 *          Possibility to print in a pacted or unpacted way
 *          if the bitwith is equal or less than 4.
 * 
 * @tparam  Data_T  data type (should be data<BITWIDTH> or udata<BITWIDTH>)
 * 
 * @param[in]   file        Pointer to the stream
 * @param[in]   dataValue   Data value to display
 * @param[in]   isPacted    Boolean used to display the data value 
 *                          in a pacted or unpacted format
 * @returns                 None
 * 
 */
template<typename Data_T>
__attribute__((always_inline)) static inline
void fprintf_dataBitwidth (FILE* file, 
                           const Data_T dataValue, 
                           bool isPacted = true)
{
    if (std::is_floating_point<Data_T>::value) {
        switch(std::numeric_limits<Data_T>::digits) {
            case 64: 
                fprintf(file, "%lf", (double)dataValue);
                break;
            case 32: 
            case 16: 
                fprintf(file, "%f", (float)dataValue);
                break;
            default: 
                fprintf(file, "NaN");
                break;
        }
    }
    else {
        if (std::numeric_limits<Data_T>::digits > 4) {
            if (std::is_unsigned<Data_T>::value) {
                switch(std::numeric_limits<Data_T>::digits) {
                    case 32: 
                        fprintf(file, "%d", (uint32_t)dataValue);
                        break;
                    case 16: 
                        fprintf(file, "%hd", (uint16_t)dataValue);
                        break;
                    case 8:
                    case 7:
                    case 6:
                    case 5:
                        fprintf(file, "%d", (uint8_t)dataValue);
                        break;
                    default: 
                        fprintf(file, "NaN");
                        break;
                }
            } else {
                switch(std::numeric_limits<Data_T>::digits) {
                    case 32: 
                        fprintf(file, "%d", (int32_t)dataValue);
                        break;
                    case 16: 
                        fprintf(file, "%hd", (int16_t)dataValue);
                        break;
                    case 8:
                    case 7:
                    case 6:
                    case 5:
                        fprintf(file, "%d", (int8_t)dataValue);
                        break;
                    default: 
                        fprintf(file, "NaN");
                        break;
                }
            }
        }
        else {
            if (isPacted) {
                fprintf(file, "0x%x", (uint32_t)dataValue);
            } 
            else {
                switch(std::numeric_limits<Data_T>::digits) {
                    case 4: 
                    case 3:
                        fprintf_dataBitwidth_4(file, dataValue);
                        break;
                    case 2:
                        fprintf_dataBitwidth_2(file, dataValue);
                        break;
                    case 1:
                        fprintf_dataBitwidth_1(file, dataValue);
                        break;
                    default: 
                        fprintf(file, "NaN");
                        break;
                }
            }
        }  
    }
}



#endif  // __N2D2_SUPPORT_FUNCTIONS_HPP__
