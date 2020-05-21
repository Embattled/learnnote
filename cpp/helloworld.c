#include <stdio.h>
union cupack {
    char c[4];
    unsigned int u;
} data;

int main(void)
{
    //To set u=1, then u will be 0x0001.
    data.u=1;
    // If pc is big-endian, the 0x0001 in memorize is 0 0 0 1.
    if (data.c[3]==1)
    {
        printf("System is big-endian.\n");
    }
    // If pc is little-endian, the 0x0001 in memorize is 1 0 0 0 
    if (data.c[0]==1)
    {
        printf("System is little-endian.\n");
    }
    return 0;
}