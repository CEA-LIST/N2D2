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

#ifndef N2D2_KEY_H
#define N2D2_KEY_H

#ifdef WIN32
#define KEY_TAB 9
#define KEY_ESC 27
#define KEY_ENTER 13
#define KEY_HOME (36 << 16)
#define KEY_LEFT (37 << 16)
#define KEY_UP (38 << 16)
#define KEY_RIGHT (39 << 16)
#define KEY_DOWN (40 << 16)
#define KEY_PAGEUP (33 << 16)
#define KEY_PAGEDOWN (34 << 16)
#define KEY_END (35 << 16)
#define KEY_INSERT (45 << 16)
#define KEY_NUMPADENTER 13
#define KEY_NUMPADDOT 42
#define KEY_NUMPADADD 43
#define KEY_NUMPADSUB 45
#define KEY_NUMPADDIV 47
#define KEY_NUMPAD0 48
#define KEY_NUMPAD1 49
#define KEY_NUMPAD2 50
#define KEY_NUMPAD3 51
#define KEY_NUMPAD4 52
#define KEY_NUMPAD5 53
#define KEY_NUMPAD6 54
#define KEY_NUMPAD7 55
#define KEY_NUMPAD8 56
#define KEY_NUMPAD9 57
#define KEY_F1 (112 << 16)
#define KEY_F2 (113 << 16)
#define KEY_F3 (114 << 16)
#define KEY_F4 (115 << 16)
#define KEY_F5 (116 << 16)
#define KEY_F6 (117 << 16)
#define KEY_F7 (118 << 16)
#define KEY_F8 (119 << 16)
#define KEY_F9 (120 << 16)
#define KEY_F10 (121 << 16)
#define KEY_F11 (122 << 16)
#define KEY_F12 (123 << 16)
#define KEY_BACKSPACE 8
#define KEY_TILDE                                                              \
    233 // key labeled ("2 é ~"), not the same as key above TAB, which is not
// recognized by OpenCV
#else
#define KEY_TAB (0x100000 | 9)
#define KEY_ESC (0x100000 | 27)
#define KEY_ENTER (0x100000 | 10)
#define KEY_HOME (0x10FF00 | 80)
#define KEY_LEFT (0x10FF00 | 81)
#define KEY_UP (0x10FF00 | 82)
#define KEY_RIGHT (0x10FF00 | 83)
#define KEY_DOWN (0x10FF00 | 84)
#define KEY_PAGEUP (0x10FF00 | 85)
#define KEY_PAGEDOWN (0x10FF00 | 86)
#define KEY_END (0x10FF00 | 87)
#define KEY_INSERT (0x10FF00 | 99)
#define KEY_NUMPADENTER (0x10FF00 | 141)
#define KEY_NUMPADDOT (0x10FF00 | 170)
#define KEY_NUMPADADD (0x10FF00 | 171)
#define KEY_NUMPADSUB (0x10FF00 | 173)
#define KEY_NUMPADDIV (0x10FF00 | 175)
#define KEY_NUMPAD0 (0x10FF00 | 176)
#define KEY_NUMPAD1 (0x10FF00 | 177)
#define KEY_NUMPAD2 (0x10FF00 | 178)
#define KEY_NUMPAD3 (0x10FF00 | 179)
#define KEY_NUMPAD4 (0x10FF00 | 180)
#define KEY_NUMPAD5 (0x10FF00 | 181)
#define KEY_NUMPAD6 (0x10FF00 | 182)
#define KEY_NUMPAD7 (0x10FF00 | 183)
#define KEY_NUMPAD8 (0x10FF00 | 184)
#define KEY_NUMPAD9 (0x10FF00 | 185)
#define KEY_F1 (0x10FF00 | 190)
#define KEY_F2 (0x10FF00 | 191)
#define KEY_F3 (0x10FF00 | 192)
#define KEY_F4 (0x10FF00 | 193)
#define KEY_F5 (0x10FF00 | 194)
#define KEY_F6 (0x10FF00 | 195)
#define KEY_F7 (0x10FF00 | 196)
#define KEY_F8 (0x10FF00 | 197)
#define KEY_F9 (0x10FF00 | 198)
#define KEY_F10 (0x10FF00 | 199)
#define KEY_F11 (0x10FF00 | 200)
#define KEY_F12 (0x10FF00 | 201)
#define KEY_BACKSPACE (0x10FF00 | 8)
#define KEY_TILDE (0x100000 | 5053)
#endif

#endif // N2D2_KEY_H
