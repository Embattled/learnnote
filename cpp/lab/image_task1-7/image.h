#ifndef IMAGE_H
#define IMAGE_H
#endif

#define SUCCESS 0
#define FAILURE 1
#define LOGICALERROR 4
#define OPENERROR 5
#define HEADERROR 6
#define WRITEERROR 7
#define READERROR 8
#define OUTPATHERROR 9
#define NOTMATCH 10

namespace std
{
    class ifstream;
    class ofstream;
    class sfstream;
    class streampos;
} // namespace std

class image
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
    int *ascContent = nullptr;

    //property for bin only
    // p4=1/8     p5 and p6 = colorDepth/255
    int valueSize;
    //valueSize * colorNum
    int pixelSize;
    // pixelSize * width
    long long rowSize;
    char *binBuffer = nullptr;

    std::ifstream *fileStream;
    // position for pixel'content.
    std::streampos *pixelPos;

    // Open file and read head information
    int fileCheck();
    // Read pixel value to buffer
    int readContent();
    // Create output file
    int outFileMake(std::ofstream &outFile, const char *outputPath);
    // Write head information to output file, default parameter is same as source file.
    int writeHead(std::ofstream &outPut, int format, int width, int high, int colorDepth);
    // write pixel content to output file.
    int writeContentAsc(std::ofstream &outfile, int *buffer, long long valueNumber);
    // Redirect stream pointer to first value of pixel content.
    int contentPos();

    // Check error and print information for easily debug.
    int status = SUCCESS;
    // Default parameter is to check image::status
    int statusCheck(int result = -1);

    //These are private function, which is process pixel value.
    int pgm2ppm(std::ofstream &outfile);
    int pgm2pbm(std::ofstream &outfile, int threshold = 0);

    int flipVertical(std::ofstream &outfile);
    int flipHorizontal(std::ofstream &outfile);

    int nearestNeighbor(int *valueBuffer, int newWidth, int newHigh);
    int bilinearInterpolation(int *valueBuffer, int newWidth, int newHigh);

    int smoothPixel(int *valueBuffer, int weightOfOwn);

    int rotationPixel(int *valueBuffer, float Angle);

    int sobelPixel(int *valueBuffer);
    int laplacianPixel(int *valueBuffer);

public:
    image(const char *path);
    ~image();

    // print source image information.
    int printInfor();

    // Copy was implementation for all six format, if source was binary file, the output file will be convert to ascii file.
    int copy(const char *outputPath);
    // Convert was implementation for p2 and p5 pgm format, the output file will be convert to ascii file..
    int convert(const char *outputPath);

    int flip(char mode, const char *outputPath);
    int shrink(char mode, const char *times, const char *outputPath);
    int smooth(const char *cstrParameter, const char *outputPath);
    int rotation(const char *cstrParameter, const char *outputPath);
    int filter(char mode, const char *outputPath);
};
