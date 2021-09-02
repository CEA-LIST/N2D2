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

#ifndef N2D2_EXPORTCPP_TYPEDEF_UNION_H
#define N2D2_EXPORTCPP_TYPEDEF_UNION_H

typedef union
{
    __uint8_t uvector;
    __int8_t svector;
    struct
    {
        __int8_t op0 : 4;
        __int8_t op1 : 4;
    } sfields;
    struct
    {
        __uint8_t op0 : 4;
        __uint8_t op1 : 4;
    } ufields;
} T4_8_Vector;

typedef union
{
    __uint8_t uvector;
    __int8_t svector;
    struct
    {
        __int8_t op0 : 2;
        __int8_t op1 : 2;
        __int8_t op2 : 2;
        __int8_t op3 : 2;
    } sfields;
    struct
    {
        __uint8_t op0 : 2;
        __uint8_t op1 : 2;
        __uint8_t op2 : 2;
        __uint8_t op3 : 2;
    } ufields;
} T2_8_Vector;

typedef union
{
    __uint8_t uvector;
    __int8_t svector;
    struct
    {
        __int8_t op0 : 1;
        __int8_t op1 : 1;
        __int8_t op2 : 1;
        __int8_t op3 : 1;
        __int8_t op4 : 1;
        __int8_t op5 : 1;
        __int8_t op6 : 1;
        __int8_t op7 : 1;
    } sfields;
    struct
    {
        __uint8_t op0 : 1;
        __uint8_t op1 : 1;
        __uint8_t op2 : 1;
        __uint8_t op3 : 1;
        __uint8_t op4 : 1;
        __uint8_t op5 : 1;
        __uint8_t op6 : 1;
        __uint8_t op7 : 1;
    } ufields;
} T1_8_Vector;

typedef struct PackSupport {
    uint8_t         accumulator;
    unsigned int    cptAccumulator;
    int             nb_bit;
} PackSupport;


#endif // N2D2_EXPORTCPP_TYPEDEF_UNION_H