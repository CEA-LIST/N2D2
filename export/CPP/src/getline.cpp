#include "getline.hpp"

int getDelim(char **linep, size_t *n, int delim, FILE *fp)
{
    int ch;
    size_t i = 0;
    if(!linep || !n || !fp){
        errno = EINVAL;
        return -1;
    }
    if(*linep == NULL){
        if(NULL==(*linep = (char*)malloc(*n=128))){
            *n = 0;
            errno = ENOMEM;
            return -1;
        }
    }
    while((ch = fgetc(fp)) != EOF){
        if(i + 1 >= *n){
            char *temp = (char*) realloc(*linep, *n + 128);
            if(!temp){
                errno = ENOMEM;
                return -1;
            }
            *n += 128;
            *linep = temp;
        }
        (*linep)[i++] = (char)ch;
        if(ch == delim)
            break;
    }
    (*linep)[i] = '\0';
    return (!i && ch == EOF) ? -1 : (int)i;
}

int getLine(char **linep, size_t *n, FILE *fp)
{
    return getDelim(linep, n, '\n', fp);
}

