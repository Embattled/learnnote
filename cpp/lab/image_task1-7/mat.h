#ifndef MAT_H
#define MAT_H

#define OPENERROR 0b00000001
#define HEADERROR 0b00000010
#define NOTMATCH 0b00000100
#define OUTPATHERROR 0b00001000
#define WRITEERROR 0b00010000

namespace ryu
{
    class mat
    {
    private:
        const char *cfilePath;

        // 0 for format, 1 for width, 2 for high, 3 for color depth
        int property[4];

        //Property for bin and asc
        //pixelNum = width * high
        long long pixelNum;

        // p1,p2 = 1 , p3 = 3
        int colorNum;

        //srcValueNumber = pixelNum * colorNum;
        long long srcValueNumber;

        // Open file and read content
        int readFile();

        // write file

        // Check error and print information for easily debug.
        char status = 0;

        // Default parameter is to check image::status
        int statusCheck();

    public:
        int *ascContent = nullptr;

        void printInfor();
        int getFormat() const { return property[0]; }
        int getWidth() const { return property[1]; }
        int getHigh() const { return property[2]; }
        int getColordepth() const { return property[3]; }
        long long getPixelNum() const { return pixelNum; };
        long long getValueNum() const { return srcValueNumber; }
        int writeAscFile(const char *outputPath = nullptr);
        int initialize(int format, int width, int high, int colordepth = 0);
        mat(const char *path);
        mat(const mat &copymat);
        void operator=(const mat &copy);
        mat(void);
        ~mat();
    };
} // namespace ryu
#endif