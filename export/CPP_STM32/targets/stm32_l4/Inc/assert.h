/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

    This file is not part of the open source version of N2D2 and is NOT under
    the CeCILL-C license. This code is the property of the CEA. It can not be
    copied or disseminated without its authorization.
*/
#ifndef ASSERT_H
#define ASSERT_H

#ifdef __cplusplus
extern "C" {
#endif 

inline void assert_failure(const char* msg, const char* file, int line) {
//    printf("Assert failure: %s in %s:%d.\r\n", msg, file, line);
    while(1) {}
}

#ifdef NDEBUG
#define assert(test) ((void)0)
#define assertm(test, msg) ((void)0)
#else
#define assert(test) do { if(!(test)) { assert_failure("error", __FILE__, __LINE__); } } while(0)
#define assertm(test, msg) do { if(!(test)) { assert_failure(msg, __FILE__, __LINE__); } } while(0)
#endif 

#ifdef __cplusplus
}
#endif 

#endif
