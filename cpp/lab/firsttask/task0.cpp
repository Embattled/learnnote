
/*
テキストファイル(英数字だけでよい)を一行ずつ読み取って、その文字列を
画面に表示する。文字列用のバッファ(配列)は、例えば30バイト(文字)のように
固定長で定義し、それより長い行については、バッファに収まるだけ読み込んで
表示し、はみ出た右側の部分は改行コードまで読み捨てる。
(上記のような一行読み取りを行う関数を作る)
*/

#define BUFFER 30
#define SUCCESS 0
#define FAILURE 1
#include <stdio.h>

//Read one line and print
int readLine(FILE *file)
{
    char buffer[BUFFER];
    int nextChar, result;
    int index = 0;
    while (true)
    {
        nextChar = fgetc(file);
        if (nextChar == '\n')
        {
            result = SUCCESS;
            break;
        }
        if (nextChar == EOF)
        {
            result = FAILURE;
            break;   
        }
        if (index < BUFFER)
            buffer[index++] = nextChar;
    }
    printf("%s\n", buffer);
    return result;
}

int main(int argc, char const *argv[])
{
    FILE *myfile = fopen(argv[1], "r");
    // while (readLine(myfile) == SUCCESS){}
    fclose(myfile);
    return 0;
}
