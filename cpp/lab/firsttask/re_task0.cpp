```cpp
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#define SUCCESS 0
#define FAILURE 1

//Default length of one line
#define LINELENGTH 30
//To throw off useless char,the number of char will be read at one time
#define THROWLENGTH 100

//Read line to buffer
int readLine(FILE *file, int length, char *buffer)
{
    char *result;
    //It's need one more byte to store '\0'
    result = fgets(buffer, length + 1, file);
    if (result == NULL)
    {
        return FAILURE;
    }
    //To throw off useless word.
    char *throwBuffer = (char *)malloc(THROWLENGTH * sizeof(char));
    if (strlen(buffer) == length)
    {
        do
        {
            memset(throwBuffer, '\0', THROWLENGTH);
            fgets(throwBuffer, THROWLENGTH, file);
        } while (strlen(throwBuffer) == THROWLENGTH - 1);
    }
    free(throwBuffer);
    return SUCCESS;
}
int printLine(char *buffer)
{
    printf("%s\n", buffer);
    return 0;
}
int task0(const char *fileName, int length)
{
    FILE *myfile = fopen(fileName, "r");
    if (myfile == NULL)
    {
        return FAILURE;
    }
    //It's need one more byte to store '\0'
    int memSize = (length + 1) * sizeof(char);
    char *buffer = (char *)malloc(memSize);
    
    while (readLine(myfile, length, buffer) == SUCCESS)
    {
        printLine(buffer);
        memset(buffer, '\0', memSize);
    }
    free(buffer);
    return SUCCESS;
}
int main(int argc, char const *argv[])
{
    if (argc == 2)
    {
        int defaultLength = LINELENGTH;
        if (task0(argv[1], defaultLength) != SUCCESS)
        {
            printf("File doesn't exist!\n");
        }
    }
    return 0;
}
```
