
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
int readLineAndCutoff(FILE *file, int length, char *buffer)
{
    char *result;
    //It's need one more byte to store '\n'
    result = fgets(buffer, length + 1, file);
    //File is read over.
    if (result == 0)
    {
        return FAILURE;
    }
    //To judge this line is longer than set 'length'.
    if (strlen(buffer)==length&&buffer[length-1]!='\n')
    {
        //Set final byte to '\n' manually
        buffer[strlen(buffer)]='\n';
        char *throwBuffer = new char[THROWLENGTH * sizeof(char)];
        do
        {
            fgets(throwBuffer, THROWLENGTH, file);
        } while (strlen(throwBuffer) == THROWLENGTH - 1);
        delete [] throwBuffer;
    }
    return SUCCESS;
}

//Check file whether available, call cut off function
int task0(const char *fileName, int length)
{
    FILE *myfile = fopen(fileName, "r");
    if (myfile == 0)
    {
        return FAILURE;
    }
    //It's need one more byte to store '\n',and one more byte to store '\0'
    int memSize = (length + 2) * sizeof(char);
    char *buffer = new char[memSize];
    //To prevent string overflow.
    buffer[memSize-1]='\0';
    while (readLineAndCutoff(myfile, length, buffer) == SUCCESS)
    {
        printf("%s", buffer);
    }
    delete [] buffer;
    return SUCCESS;
}
//The main function, accept file path and call function.
int main(int argc, char const *argv[])
{
    if (argc == 2)
    {
        int defaultLength = LINELENGTH;
        if (task0(argv[1], defaultLength) != SUCCESS)
        {
            printf("File doesn't exist!\n");
            return 1;
        }

    }
    return 0;
}

