
#include "mat.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <math.h>
#include <cstring>

using namespace ryu;

mat::mat(const char *path)
{
    cfilePath = path;
    readFile();
    if (statusCheck() != 0)
    {
        exit(0);
    }
}
mat::mat(const ryu::mat &copymat)
{
    // std::cout << "copy run" << std::endl;
    for (size_t i = 0; i < 4; i++)
    {
        property[i] = copymat.property[i];
    }
    pixelNum = copymat.pixelNum;
    colorNum = copymat.colorNum;
    srcValueNumber = copymat.srcValueNumber;

    delete[] ascContent;
    ascContent = new int[srcValueNumber];
    //deep copy
    memcpy(ascContent, copymat.ascContent, srcValueNumber * sizeof(int));
}
void mat::operator=(const mat &copymat)
{
    for (size_t i = 0; i < 4; i++)
    {
        property[i] = copymat.property[i];
    }
    pixelNum = copymat.pixelNum;
    colorNum = copymat.colorNum;
    srcValueNumber = copymat.srcValueNumber;

    delete[] ascContent;
    ascContent = new int[srcValueNumber];
    //deep copy
    memcpy(ascContent, copymat.ascContent, srcValueNumber * sizeof(int));
}
mat::mat(void)
{
}

mat::~mat()
{
    delete[] ascContent;
}
int mat::initialize(int format, int width, int high, int colordepth)
{
    if (format < 1 || format > 3 || width < 1 || high < 1)
    {
        std::cout << "Property error!" << std::endl;
        return 1;
    }
    property[0] = format;
    property[1] = width;
    property[2] = high;
    if (format != 1 && format != 4)
    {
        if (colordepth < 1)
        {
            std::cout << "initialize property colordepth error!" << std::endl;
            return 1;
        }
        property[3] = colordepth;
    }

    pixelNum = property[1] * property[2];
    colorNum = property[0] == 3 || property[0] == 6 ? 3 : 1;
    srcValueNumber = pixelNum * colorNum;

    delete[] ascContent;
    ascContent = new int[srcValueNumber]{0};

    return 0;
}
void mat::printInfor()
{
    std::cout << "This image is P" << property[0] << " Image" << std::endl;
    std::cout << property[1] << "x" << property[2] << std::endl;
    if (property[3] != 0)
        std::cout << "Color depth: " << property[3] << std::endl;
}

int mat::readFile()
{
    std::ifstream fileStream;

    fileStream.open(cfilePath, std::ios::in | std::ios::binary);
    if (!fileStream.is_open())
    {
        status = status | OPENERROR;
        return 0;
    }

    std::string format;
    std::string line;
    fileStream >> format;
    if (format.length() != 2 || format.find('P') != 0)
    {
        status = status | HEADERROR;
        return 0;
    }

    int propertyIndex = 0;
    int type = format.c_str()[1] - 48;
    if (type > 6 || type < 1)
    {
        status = status | HEADERROR;
        return 0;
    }
    property[propertyIndex++] = type;
    property[3] = 0;
    int colorDepthExist = (type == 1 || type == 4) ? 3 : 4;

    //To delete annotation
    while (propertyIndex < colorDepthExist)
    {
        std::getline(fileStream, line);
        std::size_t annotationPos = line.find('#');
        if (annotationPos != std::string::npos)
        {
            line = line.substr(0, annotationPos);
        }

        //If the all characters in this line is annotation
        //To judge the string whether has property value or just space chars line.
        bool isBlankString = true;
        for (std::string::size_type i = 0; i < line.length(); i++)
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
            status = status | HEADERROR;
            return 0;
        }
    }

    pixelNum = property[1] * property[2];
    colorNum = property[0] == 3 || property[0] == 6 ? 3 : 1;
    srcValueNumber = pixelNum * colorNum;

    ascContent = new int[srcValueNumber];
    // For binary file.
    if (property[0] > 3)
    {
        char *binBuffer = nullptr;
        if (property[0] == 4)
        {
            int srcValueSize = pixelNum * colorNum / 8;
            binBuffer = new char[srcValueSize];
            fileStream.read(binBuffer, srcValueSize);

            std::streampos pos = fileStream.tellg();

            fileStream.seekg(0, std::ios::end);
            std::streampos endpos = fileStream.tellg();
            if (!fileStream.good())
            {
                status = status | NOTMATCH;
                return 0;
            }

            char buffer;
            for (long long i = 0; i < srcValueNumber;)
            {
                memset(&buffer, 0, sizeof(char));
                memcpy(&buffer, &binBuffer[i], 1);
                for (int j = 0; j < 8; j++)
                {

                    if ((buffer & 0b00000001) == 1)
                        ascContent[i++] = 1;
                    else
                        ascContent[i++] = 0;
                    buffer = buffer >> 1;
                }
            }
        }

        if (property[0] > 4)
        {
            int valueSize = ceil(property[3] / 255);

            int srcValueSize = valueSize * colorNum * pixelNum;

            binBuffer = new char[srcValueSize];
            fileStream.read(binBuffer, srcValueSize);

            std::streampos pos = fileStream.tellg();

            fileStream.seekg(0, std::ios::end);
            std::streampos endpos = fileStream.tellg();
            if (!fileStream.good())
            {
                status = status | NOTMATCH;
                return 0;
            }

            int *buffer = new int;
            for (long long i = 0; i < srcValueNumber; i = i + valueSize)
            {
                memset(buffer, 0, sizeof(int));
                memcpy(buffer, &binBuffer[i], valueSize);
                ascContent[i] = *buffer;
            }
        }
        delete[] binBuffer;
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
            status = status | NOTMATCH;
            return 0;
        }
    }
    property[0] = property[0] < 4 ? property[0] : property[0] - 3;
    return 0;
}

int mat::writeAscFile(const char *inoutputPath)
{
    //write head
    std::string outputpath;
    if (inoutputPath == nullptr)
    {
        outputpath = "output.pgm";
    }
    else
    {
        outputpath = inoutputPath;
    }
    std::ofstream outFile;
    outFile.open(outputpath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!outFile.is_open())
    {
        status = status | OUTPATHERROR;
        return 0;
    }

    int format = property[0];
    outFile << "P" << format << "\n"
            << property[1] << " " << property[2] << "\n";
    if (property[0] != 1 && property[0] != 4)
        outFile << property[3] << "\n";
    if (outFile.fail())
    {
        status = status | WRITEERROR;
        return 0;
    }

    //write content
    int newRow = 0;
    for (long long index = 0; index < srcValueNumber; index++)
    {
        outFile << ascContent[index];
        if (++newRow == 16)
        {
            outFile << '\n';
            newRow = 0;
        }
        else
        {
            outFile << ' ';
        }
    }
    if (!outFile.good())
    {
        status = status | WRITEERROR;
        return 0;
    }
    if (statusCheck() != 0)
    {
        exit(0);
    }
    return 0;
}

int mat::statusCheck()
{
    if (status == 0)
    {
        // std::cout << "Execute success!" << std::endl;
        return 0;
    }

    if ((status & OPENERROR) != 0)
        std::cout << "Source file don't exist." << std::endl;
    if ((status & HEADERROR) != 0)
        std::cout << "Source file has illegal head." << std::endl;
    if ((status & NOTMATCH) != 0)
        std::cout << "The number of pixel value has not match with image resolution." << std::endl;
    if ((status & OUTPATHERROR) != 0)
        std::cout << "Output destination path error." << std::endl;
    if ((status & WRITEERROR) != 0)
        std::cout << "Write process error." << std::endl;
    return 1;
}
