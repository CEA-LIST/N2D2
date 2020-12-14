/* Source: https://stackoverflow.com/questions/27369580/codeblocks-and-c-undefined-reference-to-getline */
#ifndef GETLINE_EXPORTC_DEEPNET_H
#define GETLINE_EXPORTC_DEEPNET_H

#include <cstdio>
#include <cstdlib>
#include <cerrno>

int getDelim(char **linep, size_t *n, int delim, FILE *fp);
int getLine(char **linep, size_t *n, FILE *fp);

#endif // GETLINE_EXPORTC_DEEPNET_H

