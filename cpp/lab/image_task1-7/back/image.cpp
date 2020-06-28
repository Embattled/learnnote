#include "image.h"
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstring>
#include <ctype.h>

#define _USE_MATH_DEFINES
#include <math.h>

#define SUCCESS 0
#define FAILURE 1
#define LOGICALERROR 4
#define OPENERROR 5
#define HEADERROR 6
#define WRITEERROR 7
#define READERROR 8
#define OUTPATHERROR 9
#define NOTMATCH 10
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

    std::ifstream fileStream;
    // position for pixel'content.
    std::streampos pixelPos;

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

image::image(const char *path)
{
    cfilePath = path;
    if (statusCheck(fileCheck()) != SUCCESS)
    {
        exit(1);
    }
}

image::~image()
{
    delete ascContent;
    delete binBuffer;
    if (fileStream.is_open())
    {
        fileStream.close();
    }
}
int image::fileCheck()
{
    fileStream.open(cfilePath, std::ios::in | std::ios::binary);
    if (!fileStream.is_open())
    {
        return OPENERROR;
    }

    std::string format;
    std::string line;
    fileStream >> format;
    if (format.length() != 2 || format.find('P') != 0)
    {
        return HEADERROR;
    }

    int propertyIndex = 0;
    int type = format.c_str()[1] - 48;
    if (type > 6 || type < 1)
    {
        return HEADERROR;
    }
    property[propertyIndex++] = type;
    property[3] = 0;
    int colorDepthExist = (type == 1 || type == 4) ? 3 : 4;

    //To delete annotation
    while (propertyIndex < colorDepthExist)
    {
        std::getline(fileStream, line);
        int annotationPos = line.find('#');
        if (annotationPos != std::string::npos)
        {
            line = line.substr(0, annotationPos);
        }

        //If the all line is annotation
        //To judge the string whether has property value or just space char.
        bool isBlankString = true;
        for (int i = 0; i < line.length(); i++)
        {
            if (isdigit(line.c_str()[i]))
            {
                isBlankString = false;
                break;
            }
        }
        if (isBlankString)
        {
            continue;
        }
        std::stringstream sstr(line);
        while (!sstr.eof())
        {
            sstr >> property[propertyIndex++];
        }
        if (sstr.fail() || propertyIndex > colorDepthExist)
        {
            return HEADERROR;
        }
    }
    pixelPos = fileStream.tellg();
    pixelNum = property[1] * property[2];
    colorNum = property[0] == 3 || property[0] == 6 ? 3 : 1;
    srcValueNumber = pixelNum * colorNum;
    return statusCheck(readContent());
}
int image::readContent()
{
    ascContent = new int[srcValueNumber];
    // For binary file.
    if (property[0] == 4)
    {
        std::cout << "This format was not implementation." << std::endl;
        return FAILURE;
    }

    if (property[0] > 4)
    {
        valueSize = ceil(property[3] / 255);
        pixelSize = valueSize * colorNum;

        rowSize = property[1] * pixelSize;

        int allSize = pixelSize * pixelNum;

        binBuffer = new char[allSize];
        fileStream.read(binBuffer, allSize);

        std::streampos pos = fileStream.tellg();
        fileStream.seekg(0, std::ios::end);

        std::streampos endpos = fileStream.tellg();
        if (!fileStream.good())
        {
            return NOTMATCH;
        }

        int *buffer = new int;
        for (long long i = 0; i < srcValueNumber; i++)
        {
            memset(buffer, 0, sizeof(int));
            memcpy(buffer, &binBuffer[i * valueSize], valueSize);
            ascContent[i] = *buffer;
        }
        return SUCCESS;
    }
    else // For asc file.
    {
        ascContent = new int[srcValueNumber];
        long long index = 0;
        while (!fileStream.eof() && index < srcValueNumber)
        {
            fileStream >> ascContent[index++];
        }
        if (index != srcValueNumber)
        {
            return NOTMATCH;
        }
        else
        {
            return SUCCESS;
        }
    }
}

int image::outFileMake(std::ofstream &outFile, const char *outputPath)
{
    std::string output;
    if (outputPath == nullptr)
    {
        output = "output.ppm";
    }
    else
    {
        output = outputPath;
    }
    outFile.open(output, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!outFile.is_open())
    {
        status = OUTPATHERROR;
    }
    return statusCheck();
}

int image::writeHead(std::ofstream &outPut, int format = 0, int width = 0, int high = 0, int colorDepth = 0)
{
    format = format == 0 ? property[0] : format;
    width = width == 0 ? property[1] : width;
    high = high == 0 ? property[2] : high;
    if (format == 1)
        colorDepth = 0;
    else
        colorDepth = colorDepth == 0 ? property[3] : colorDepth;
    outPut << "P" << format << "\n"
           << width << " " << high << "\n";
    if (colorDepth != 0)
        outPut << colorDepth << "\n";
    if (outPut.fail())
    {
        status = WRITEERROR;
    }
    return statusCheck();
}
int image::writeContentAsc(std::ofstream &outfile, int *buffer = nullptr, long long valueNumber = -1)
{
    buffer = buffer == nullptr ? ascContent : buffer;
    valueNumber = valueNumber == -1 ? srcValueNumber : valueNumber;
    int newRow = 0;
    for (long long index = 0; index < valueNumber; index++)
    {
        outfile << buffer[index];
        if (++newRow == 16)
        {
            outfile << '\n';
        }
        else
        {
            outfile << ' ';
        }
    }
    if (!outfile.good())
    {
        status = WRITEERROR;
    }
    return statusCheck();
}
int image::contentPos()
{
    fileStream.clear();
    fileStream.seekg(pixelPos);
    return SUCCESS;
}
int image::statusCheck(int result)
{
    result = result == -1 ? status : result;
    if (result)
    {
        switch (result)
        {
        case FAILURE:
            std::cout << "Error occur!" << std::endl;
            break;
        case LOGICALERROR:
            std::cout << "Program has logical error." << std::endl;
            break;
        case OPENERROR:
            std::cout << "Source file don't exist." << std::endl;
            break;
        case HEADERROR:
            std::cout << "Source file has illegal head." << std::endl;
            break;
        case WRITEERROR:
            std::cout << "Write process error." << std::endl;
            break;
        case READERROR:
            std::cout << "Read process error." << std::endl;
            break;
        case OUTPATHERROR:
            std::cout << "Output destination path error." << std::endl;
            break;
        case NOTMATCH:
            std::cout << "The number of pixel value has not match with image resolution." << std::endl;
            break;

        default:
            break;
        }
        return FAILURE;
    }
    else
    {
        return SUCCESS;
    }
}
int image::printInfor()
{
    std::cout << "Input image is P" << property[0] << " Image" << std::endl;
    std::cout << property[1] << "x" << property[2] << std::endl;
    if (property[3] != 0)
        std::cout << "Color depth: " << property[3] << std::endl;
    return SUCCESS;
}
int image::copy(const char *outputPath)
{
    std::ofstream outputStream;
    outFileMake(outputStream, outputPath);
    int newFormat = property[0] > 3 ? property[0] - 3 : property[0];
    writeHead(outputStream, newFormat);
    writeContentAsc(outputStream);
    if (outputStream.good())
    {
        outputStream.close();
        return SUCCESS;
    }
    return FAILURE;
}

int image::convert(const char *outputPath)
{
    std::string outStr = outputPath;
    std::string outType;
    if (outStr.find('.' != std::string::npos))
    {
        outType = outStr.substr(outStr.find('.'));
    }
    else
    {
        status = OUTPATHERROR;
    }

    std::ofstream outputStream;

    if (property[0] == 2 || property[0] == 5)
    {
        outFileMake(outputStream, outputPath);
        if (outType == ".pbm")
        {
            statusCheck(writeHead(outputStream, 1));
            pgm2pbm(outputStream);
        }
        else if (outType == ".ppm")
        {
            statusCheck(writeHead(outputStream, 3));
            pgm2ppm(outputStream);
        }
    }
    else
    {
        return FAILURE;
    }

    outputStream.close();
    return statusCheck();
}

int image::pgm2ppm(std::ofstream &outfile)
{
    int newRow = 0;
    for (long long index = 0; index < srcValueNumber; index++)
    {
        int value = ascContent[index];
        outfile << value << " " << value << " " << value;
        if (++newRow == 5)
        {
            newRow = 0;
            outfile << '\n';
        }
        else
        {
            outfile << ' ';
        }
    }
    return SUCCESS;
}

int image::pgm2pbm(std::ofstream &outfile, int threshold)
{
    if (threshold == 0)
        threshold = property[3] / 2;
    int newRow = 0;

    long long value;
    for (long long index = 0; index < srcValueNumber; index++)
    {
        value = ascContent[index];
        value > threshold ? outfile << "0" : outfile << "1";
        if (++newRow == 16)
        {
            newRow = 0;
            outfile << "\n";
        }
        else
        {
            outfile << ' ';
        }
    }
    return SUCCESS;
}

int image::flip(char mode, const char *outputPath)
{
    std::ofstream outputStream;
    outFileMake(outputStream, outputPath);
    int newFormat = property[0] > 3 ? property[0] - 3 : property[0];
    writeHead(outputStream, newFormat);

    if (mode == 'v')
    {
        status = flipVertical(outputStream);
    }
    else if (mode == 'h')
    {
        status = flipHorizontal(outputStream);
    }

    outputStream.close();
    return statusCheck();
}
int image::flipVertical(std::ofstream &outfile)
{
    int newRow = 0;
    for (int y = property[2] - 1; y >= 0; y--)
    {
        long long rowIndex = y * property[1];

        for (int x = 0; x < property[1]; x++)
        {
            long long index = (rowIndex + x) * colorNum;
            for (int color = 0; color < colorNum; color++)
            {
                outfile << ascContent[index++];
                if (++newRow == 16)
                {
                    outfile << '\n';
                }
                else
                {
                    outfile << ' ';
                }
            }
        }
    }
    return SUCCESS;
}
int image::flipHorizontal(std::ofstream &outfile)
{
    int newRow = 0;
    for (int y = 0; y < property[2]; y++)
    {
        long long rowIndex = y * property[1];
        for (int x = property[1] - 1; x >= 0; x--)
        {
            long long index = (rowIndex + x) * colorNum;
            for (int color = 0; color < colorNum; color++)
            {
                outfile << ascContent[index++];
                if (++newRow == 16)
                {
                    outfile << '\n';
                }
                else
                {
                    outfile << ' ';
                }
            }
        }
    }

    return SUCCESS;
}
int image::shrink(char mode, const char *cstrTimes, const char *outputPath)
{
    if (!(property[0] == 2 || property[0] == 5))
    {
        std::cout << "Not support this format" << std::endl;
        return FAILURE;
    }

    float parameter = std::stof(cstrTimes);

    if (parameter == 0.0)
    {
        return FAILURE;
    }
    int newWidth, newHigh;
    newWidth = (int)round(property[1] * parameter);
    newHigh = (int)round(property[2] * parameter);
    long long valueNumber = newWidth * newHigh;
    int *valueBuffer = new int[valueNumber];

    if (mode == 'n')
        nearestNeighbor(valueBuffer, newWidth, newHigh);
    else if (mode == 'i')
        bilinearInterpolation(valueBuffer, newWidth, newHigh);

    std::ofstream outputStream;
    outFileMake(outputStream, outputPath);
    writeHead(outputStream, 2, newWidth, newHigh, property[3]);
    status = writeContentAsc(outputStream, valueBuffer, valueNumber);
    statusCheck();

    return SUCCESS;
}
int image::nearestNeighbor(int *valueBuffer, int newWidth, int newHigh)
{
    long long valueNumber = newWidth * newHigh;
    float HShrinkTimes = (float)newWidth / (float)property[1];
    float VShrinkTimes = (float)newHigh / (float)property[2];

    int *oldRow = new int[newHigh];
    for (int i = 1; i <= newHigh; i++)
    {
        oldRow[i - 1] = (int)round(i / VShrinkTimes) - 1;
    }

    int *oldColumn = new int[newWidth];
    for (int i = 1; i <= newWidth; i++)
    {
        oldColumn[i - 1] = (int)round(i / HShrinkTimes) - 1;
    }
    for (int row = 0; row < newHigh; row++)
    {
        for (int column = 0; column < newWidth; column++)
        {
            long long index = row * newWidth + column;
            long long oldIndex = oldRow[row] * property[1] + oldColumn[column];
            valueBuffer[index] = ascContent[oldIndex];
        }
    }
    delete oldColumn;
    delete oldRow;
    return SUCCESS;
}
int image::bilinearInterpolation(int *valueBuffer, int newWidth, int newHigh)
{
    long long valueNumber = newWidth * newHigh;

    float xShrinkTimes = (float)(property[1] - 1) / (float)(newWidth + 1);

    int *oldXCeil = new int[newWidth];
    int *oldXFloor = new int[newWidth];
    float *xCeilWeight = new float[newWidth];
    float *xFloorWeight = new float[newWidth];
    for (int i = 1; i <= newWidth; i++)
    {
        float floatCoordinate = i * xShrinkTimes;
        oldXCeil[i - 1] = (int)ceil(floatCoordinate);
        oldXFloor[i - 1] = (int)floor(floatCoordinate);
        xCeilWeight[i - 1] = floatCoordinate - oldXFloor[i - 1];
        xFloorWeight[i - 1] = 1 - xCeilWeight[i - 1];
    }

    float *xInterpolation = new float[newWidth * property[2]]{0};
    for (int row = 0; row < property[2]; row++)
    {
        long long rowIndex = row * newWidth;
        long long rowOldIndex = row * property[1];
        for (int x = 0; x < newWidth; x++)
        {
            xInterpolation[rowIndex + x] += ascContent[rowOldIndex + oldXCeil[x]] * xCeilWeight[x];
            xInterpolation[rowIndex + x] += ascContent[rowOldIndex + oldXFloor[x]] * xFloorWeight[x];
        }
    }

    float yShrinkTimes = (float)(property[2] - 1) / (float)(newHigh + 1);
    int *oldYCeil = new int[newHigh];
    int *oldYFloor = new int[newHigh];
    float *yCeilWeight = new float[newHigh];
    float *yFloorWeight = new float[newHigh];
    for (int i = 1; i <= newHigh; i++)
    {
        float floatCoordinate = i * yShrinkTimes;
        oldYCeil[i - 1] = (int)ceil(floatCoordinate);
        oldYFloor[i - 1] = (int)floor(floatCoordinate);
        yCeilWeight[i - 1] = floatCoordinate - oldYFloor[i - 1];
        yFloorWeight[i - 1] = 1 - yCeilWeight[i - 1];
    }

    float *addBuffer = new float[valueNumber]{0};
    for (int row = 0; row < newHigh; row++)
    {
        long long rowIndex = row * newWidth;
        long long ceilRowIndex = oldYCeil[row] * newWidth;
        long long floorRowIndex = oldYFloor[row] * newWidth;

        float yCeilW = yCeilWeight[row];
        float yFloorW = yFloorWeight[row];
        for (int x = 0; x < newWidth; x++)
        {
            addBuffer[rowIndex + x] += xInterpolation[ceilRowIndex + x] * yCeilW;
            addBuffer[rowIndex + x] += xInterpolation[floorRowIndex + x] * yFloorW;
        }
    }

    for (int i = 0; i < valueNumber; i++)
    {
        valueBuffer[i] = (int)round(addBuffer[i]);
    }

    delete oldXFloor;
    delete oldXCeil;
    delete xCeilWeight;
    delete xFloorWeight;
    delete xInterpolation;

    delete oldYFloor;
    delete oldYCeil;
    delete yCeilWeight;
    delete yFloorWeight;
    delete addBuffer;

    return SUCCESS;
}

int image::smooth(const char *cstrParameter, const char *outputPath)
{
    if (!(property[0] == 2 || property[0] == 5))
    {
        std::cout << "Not support this format, Only PGM image on P2 is supporting." << std::endl;
        return FAILURE;
    }

    std::string strParameter(cstrParameter);
    int parameter = std::stoi(strParameter);
    if (parameter == 0)
    {
        std::cout << "Parameter error." << std::endl;
        return FAILURE;
    }

    std::ofstream outputStream;
    outFileMake(outputStream, outputPath);
    writeHead(outputStream, 2);
    int *valueBuffer = new int[srcValueNumber];
    smoothPixel(valueBuffer, parameter);
    status = writeContentAsc(outputStream, valueBuffer, srcValueNumber);
    delete valueBuffer;
    if (statusCheck())
    {
        return FAILURE;
    }
    return SUCCESS;
}
int image::smoothPixel(int *valueBuffer, int parameter)
{
    for (int y = 0; y < property[2]; y++)
    {
        long long rowOffset = property[1] * y;
        for (int x = 0; x < property[1]; x++)
        {

            int rowArroundIndex[8] = {0};
            rowArroundIndex[0] = rowOffset + x - property[1] - 1;
            rowArroundIndex[1] = rowOffset + x - property[1];
            rowArroundIndex[2] = rowOffset + x - property[1] + 1;
            rowArroundIndex[3] = rowOffset + x - 1;
            rowArroundIndex[4] = rowOffset + x + 1;
            rowArroundIndex[5] = rowOffset + x + property[1] - 1;
            rowArroundIndex[6] = rowOffset + x + property[1];
            rowArroundIndex[7] = rowOffset + x + property[1] + 1;

            if (y == 0)
            {
                rowArroundIndex[0] = -1;
                rowArroundIndex[1] = -1;
                rowArroundIndex[2] = -1;
            }
            if (y == property[2] - 1)
            {
                rowArroundIndex[5] = -1;
                rowArroundIndex[6] = -1;
                rowArroundIndex[7] = -1;
            }
            if (x == 0)
            {
                rowArroundIndex[0] = -1;
                rowArroundIndex[3] = -1;
                rowArroundIndex[5] = -1;
            }
            if (x == property[1] - 1)
            {
                rowArroundIndex[2] = -1;
                rowArroundIndex[4] = -1;
                rowArroundIndex[7] = -1;
            }

            int scale = 0;
            for (int i = 0; i < 8; i++)
            {
                scale += (rowArroundIndex[i] == -1) ? 0 : 1;
                valueBuffer[rowOffset + x] += (rowArroundIndex[i] == -1) ? 0 : ascContent[rowArroundIndex[i]];
            }
            scale += parameter;
            valueBuffer[rowOffset + x] += parameter * ascContent[rowOffset + x];
            valueBuffer[rowOffset + x] = (int)round(valueBuffer[rowOffset + x] / (float)scale);
        }
    }

    return SUCCESS;
}
int image::rotation(const char *cstrParameter, const char *outputPath)
{

    if (!(property[0] == 2 || property[0] == 5))
    {
        std::cout << "Not support this format, Only PGM image on P2 is supporting." << std::endl;
        return FAILURE;
    }

    float angleNumber = std::stof(cstrParameter);
    float angle = angleNumber * M_PI / 180;
    float cosAngle = cosf(angle);
    float sinAngle = sinf(angle);

    int newWidth = (int)round(property[1] * fabsf(cosAngle) + property[2] * fabsf(sinAngle));
    int newHigh = (int)round(property[1] * fabsf(sinAngle) + property[2] * fabsf(cosAngle));
    // int newWidth = (int)round(property[1] * fabsf(cosAngle) + property[2] * fabsf(sinAngle));
    // int newHigh = (int)round(property[1] * fabsf(sinAngle) + property[2] * fabsf(cosAngle));
    long long valueNumber = newWidth * newHigh;
    int *valueBuffer = new int[valueNumber]{0};
    rotationPixel(valueBuffer, angle);

    std::ofstream outputStream;
    outFileMake(outputStream, outputPath);
    writeHead(outputStream, 2, newWidth, newHigh);
    writeContentAsc(outputStream, valueBuffer, valueNumber);
    delete valueBuffer;

    return statusCheck();
}
int image::rotationPixel(int *valueBuffer, float angle)
{

    float cosAngle = cosf(angle);
    float sinAngle = sinf(angle);

    int xOldOffset = property[1] / 2;
    int yOldOffset = property[2] / 2;

    int newWidth = (int)round(property[1] * fabsf(cosAngle) + property[2] * fabsf(sinAngle));
    int newHigh = (int)round(property[1] * fabsf(sinAngle) + property[2] * fabsf(cosAngle));
    long long valueNumber = newWidth * newHigh;

    int xNewOffset = newWidth / 2;
    int yNewOffset = newHigh / 2;

    for (int y = 0; y < property[2]; y++)
    {
        long long rowOffset = y * property[1];
        for (int x = 0; x < property[1]; x++)
        {
            int xOld = x - xOldOffset;
            int yOld = y - yOldOffset;

            int xNew = (int)round(xOld * cosAngle - yOld * sinAngle + xNewOffset);
            int yNew = (int)round(xOld * sinAngle + yOld * cosAngle + yNewOffset);

            if (xNew < 0)
                xNew = 0;
            if (xNew >= newWidth)
                xNew = newWidth - 1;
            if (yNew < 0)
                yNew = 0;
            if (yNew >= newHigh)
                yNew = newHigh - 1;

            int newOffset = xNew + newWidth * yNew;

            valueBuffer[newOffset] = ascContent[rowOffset + x];
        }
    }
    return SUCCESS;
}

int image::filter(char mode, const char *outputPath)
{
    if (!(property[0] == 2 || property[0] == 5))
    {
        std::cout << "Not support this format, Only PGM image on P2 is supporting." << std::endl;
        return FAILURE;
    }

    int *valueBuffer = new int[srcValueNumber]{0};

    if (mode == 's')
    {
        sobelPixel(valueBuffer);
    }
    else if (mode == 'l')
    {
        laplacianPixel(valueBuffer);
    }
    std::ofstream outputStream;
    outFileMake(outputStream, outputPath);
    writeHead(outputStream, 2);
    writeContentAsc(outputStream, valueBuffer);

    delete valueBuffer;

    return statusCheck();
}

int image::sobelPixel(int *valueBuffer)
{
    int scale = 4;
    for (int y = 0; y < property[2]; y++)
    {
        long long upRowOffset = (y - 1) * property[1];
        long long nowRowOffset = y * property[1];
        long long downRowOffset = (y + 1) * property[1];
        for (int x = 0; x < property[1]; x++)
        {
            int up = (y == 0) ? 0 : ascContent[upRowOffset + x];
            int down = ((y + 1) == property[2]) ? 0 : ascContent[downRowOffset + x];
            int left = (x == 0) ? 0 : ascContent[nowRowOffset + x - 1];
            int right = ((x + 1) == property[1]) ? 0 : ascContent[nowRowOffset + x + 1];

            int upL = (y == 0 || x == 0) ? 0 : ascContent[upRowOffset + x - 1];
            int upR = (y == 0 || (x + 1) == property[1]) ? 0 : ascContent[upRowOffset + x + 1];
            int downL = ((y + 1) == property[2] || x == 0) ? 0 : ascContent[downRowOffset + x - 1];
            int downR = ((y + 1) == property[2] || (x + 1) == property[1]) ? 0 : ascContent[downRowOffset + x + 1];

            int gx = 0, gy = 0;
            gx = (1) * upL + (2) * left + (1) * downL + (-1) * upR + (-2) * right + (-1) * downR;
            gy = (1) * upL + (2) * up + (1) * upR + (-1) * downL + (-2) * down + (-1) * downR;

            // valueBuffer[nowRowOffset + x] = (int)round(abs(gx) + abs(gy)  / (float)scale * 2);
            valueBuffer[nowRowOffset + x] = (int)round(sqrt(gx * gx + gy * gy) / ((float)scale * sqrt(2)));
        }
    }
    return SUCCESS;
}
int image::laplacianPixel(int *valueBuffer)
{
    float scale = (float)1 / 9;
    for (int y = 0; y < property[2]; y++)
    {
        long long upRowOffset = (y - 1) * property[1];
        long long nowRowOffset = y * property[1];
        long long downRowOffset = (y + 1) * property[1];
        for (int x = 0; x < property[1]; x++)
        {
            int up = (y == 0) ? 0 : ascContent[upRowOffset + x];
            int down = ((y + 1) == property[2]) ? 0 : ascContent[downRowOffset + x];
            int left = (x == 0) ? 0 : ascContent[nowRowOffset + x - 1];
            int right = ((x + 1) == property[1]) ? 0 : ascContent[nowRowOffset + x + 1];

            int upL = (y == 0 || x == 0) ? 0 : ascContent[upRowOffset + x - 1];
            int upR = (y == 0 || (x + 1) == property[1]) ? 0 : ascContent[upRowOffset + x + 1];
            int downL = ((y + 1) == property[2] || x == 0) ? 0 : ascContent[downRowOffset + x - 1];
            int downR = ((y + 1) == property[2] || (x + 1) == property[1]) ? 0 : ascContent[downRowOffset + x + 1];

            int add = up + upL + upR + left + right + down + downL + downR - 8 * ascContent[nowRowOffset + x];

            valueBuffer[nowRowOffset + x] = (int)round(scale * abs(add));
        }
    }
    return SUCCESS;
}