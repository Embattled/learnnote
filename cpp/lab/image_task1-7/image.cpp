#include "image.h"
#include "mat.h"

#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstring>
#include <ctype.h>

#define _USE_MATH_DEFINES
#include <math.h>

namespace ryu
{
    int pgm2ppm(const mat *source, mat *destination)
    {
        if (source->getFormat() != 2)
        {
            std::cout << "Source image doesn't a pgm image.";
            return 1;
        }
        int format = 3;
        int width = source->getWidth();
        int high = source->getHigh();
        int colordepth = source->getColordepth();
        destination->initialize(format, width, high, colordepth);
        long long srcValueNumber = width * high;
        for (long long sourceIndex = 0, destinIndex = 0; sourceIndex < srcValueNumber;)
        {
            for (size_t i = 0; i < 3; i++)
            {
                destination->ascContent[destinIndex++] = source->ascContent[sourceIndex];
            }
            sourceIndex += 1;
        }
        return 0;
    }

    int pgm2pbm(const mat *source, mat *destination, int threshold)
    {
        if (source->getFormat() != 2)
        {
            std::cout << "Source image doesn't pgm image.";
            return 1;
        }

        if (threshold == 0)
            threshold = source->getColordepth() / 2;

        int format = 1;
        int width = source->getWidth();
        int high = source->getHigh();
        destination->initialize(format, width, high);
        long long srcValueNumber = width * high;

        for (long long index = 0; index < srcValueNumber; index++)
        {
            destination->ascContent[index] = source->ascContent[index] > threshold ? 0 : 1;
        }
        return 0;
    }

    int flipVertical(mat *source)
    {
        int format = source->getFormat();
        int width = source->getWidth();
        int high = source->getHigh();
        int colorNum = format == 3 || format == 6 ? 3 : 1;
        long long srcValueNumber = width * high * colorNum;

        int *oldData = source->ascContent;
        int *newData = new int[srcValueNumber];

        long long destinIndex = 0;
        //last row start
        for (int y = high - 1; y >= 0; y--)
        {
            long long sourceIndex = y * width * colorNum;
            for (int x = 0; x < width * colorNum; x++)
            {
                newData[destinIndex++] = oldData[sourceIndex++];
            }
        }
        source->ascContent = newData;
        delete[] oldData;

        return 0;
    }

    int flipHorizontal(mat *source)
    {
        int format = source->getFormat();
        int width = source->getWidth();
        int high = source->getHigh();
        int colorNum = format == 3 || format == 6 ? 3 : 1;
        long long srcValueNumber = width * high * colorNum;

        int *oldData = source->ascContent;
        int *newData = new int[srcValueNumber];

        long long destinIndex = 0;
        //first row start
        for (int y = 0; y < high; y++)
        {
            long long rowBias = y * width;
            for (int x = width - 1; x >= 0; x--)
            {
                long long sourceDataIndex = (rowBias + x) * colorNum;
                for (int color = 0; color < colorNum; color++)
                {
                    newData[destinIndex++] = oldData[sourceDataIndex++];
                }
            }
        }

        source->ascContent = newData;
        delete[] oldData;
        return 0;
    }

    int shrinkpgm(char mode, const mat *source, mat *destination, double times)
    {
        int format = source->getFormat();
        if (format != 2)
        {
            std::cout << "Only support gray format." << std::endl;
            return 1;
        }
        int newWidth = (int)round(source->getWidth() * times);
        int newHigh = (int)round(source->getHigh() * times);

        destination->initialize(2, newWidth, newHigh, source->getColordepth());

        if (mode == 'n')
        {
            return nearestNeighbor(source, destination, times);
        }
        else if (mode == 'b')
        {
            return bilinearInterpolation(source, destination, times);
        }
        else
        {
            std::cout << "Illegal mode." << std::endl;
            return 1;
        }

        return 0;
    }
    int nearestNeighbor(const mat *source, mat *destination, double times)
    {
        int oldWidth = source->getWidth();
        int oldHigh = source->getHigh();
        int newWidth = (int)round(oldWidth * times);
        int newHigh = (int)round(oldHigh * times);

        float HShrinkTimes = (float)newWidth / oldWidth;
        float VShrinkTimes = (float)newHigh / oldHigh;

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
                long long newIndex = row * newWidth + column;
                long long oldIndex = oldRow[row] * oldWidth + oldColumn[column];
                destination->ascContent[newIndex] = source->ascContent[oldIndex];
            }
        }
        return 0;
    }
    int bilinearInterpolation(const mat *source, mat *destination, double times)
    {
        int oldWidth = source->getWidth();
        int oldHigh = source->getHigh();
        int newWidth = (int)round(oldWidth * times);
        int newHigh = (int)round(oldHigh * times);

        long long valueNumber = newWidth * newHigh;
        // float HShrinkTimes = (float)newWidth / oldWidth;
        // float VShrinkTimes = (float)newHigh / oldHigh;

        float xShrinkTimes = (float)(oldWidth - 1) / (float)(newWidth + 1);

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

        float *xInterpolation = new float[newWidth * oldHigh]{0};
        for (int row = 0; row < oldHigh; row++)
        {
            long long rowIndex = row * newWidth;
            long long rowOldIndex = row * oldWidth;
            for (int x = 0; x < newWidth; x++)
            {
                xInterpolation[rowIndex + x] += source->ascContent[rowOldIndex + oldXCeil[x]] * xCeilWeight[x];
                xInterpolation[rowIndex + x] += source->ascContent[rowOldIndex + oldXFloor[x]] * xFloorWeight[x];
            }
        }

        float yShrinkTimes = (float)(oldHigh - 1) / (float)(newHigh + 1);
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
            destination->ascContent[i] = (int)round(addBuffer[i]);
        }

        delete[] oldXFloor;
        delete[] oldXCeil;
        delete[] xCeilWeight;
        delete[] xFloorWeight;
        delete[] xInterpolation;

        delete[] oldYFloor;
        delete[] oldYCeil;
        delete[] yCeilWeight;
        delete[] yFloorWeight;
        delete[] addBuffer;
        return 0;
    }

    int rotation(const mat *source, mat *destination, double angle)
    {
        int format = source->getFormat();
        if (format != 2)
        {
            std::cout << "Only support gray format." << std::endl;
            return 1;
        }
        float rad = angle * M_PI / 180;
        float cosAngle = cosf(rad);
        float sinAngle = sinf(rad);

        int width = source->getWidth();
        int high = source->getHigh();

        int newWidth = (int)round(width * fabsf(cosAngle) + high * fabsf(sinAngle));
        int newHigh = (int)round(width * fabsf(sinAngle) + high * fabsf(cosAngle));

        destination->initialize(format, newWidth, newHigh, source->getColordepth());

        int xOldOffset = width / 2;
        int yOldOffset = high / 2;

        int xNewOffset = newWidth / 2;
        int yNewOffset = newHigh / 2;

        for (int y = 0; y < high; y++)
        {
            long long rowOffset = y * width;
            for (int x = 0; x < width; x++)
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

                destination->ascContent[newOffset] = source->ascContent[rowOffset + x];
            }
        }
        return 0;
    }

    int convolution(const mat *source, mat *destination, int core[9])
    {
        int format = source->getFormat();
        if (format != 2)
        {
            std::cout << "Only support gray format." << std::endl;
            return 1;
        }
        int width = source->getWidth();
        int high = source->getHigh();
        destination->initialize(format, width, high, source->getColordepth());

        int scale = 0;
        for (int i = 0; i < 9; i++)
        {
            scale += core[i];
        }
        bool needNormalizing = scale == 0 ? false : true;

        for (int y = 0; y < high; y++)
        {
            long long rowOffset = width * y;
            for (int x = 0; x < width; x++)
            {
                long long rowArroundIndex[9] = {0};
                rowArroundIndex[0] = rowOffset + x - width - 1;
                rowArroundIndex[1] = rowOffset + x - width;
                rowArroundIndex[2] = rowOffset + x - width + 1;
                rowArroundIndex[3] = rowOffset + x - 1;
                rowArroundIndex[4] = rowOffset + x;
                rowArroundIndex[5] = rowOffset + x + 1;
                rowArroundIndex[6] = rowOffset + x + width - 1;
                rowArroundIndex[7] = rowOffset + x + width;
                rowArroundIndex[8] = rowOffset + x + width + 1;

                int whetherEdge[9] = {0};
                if (y == 0)
                {
                    whetherEdge[0] = 1;
                    whetherEdge[1] = 1;
                    whetherEdge[2] = 1;
                }
                if (y == high - 1)
                {
                    whetherEdge[6] = 1;
                    whetherEdge[7] = 1;
                    whetherEdge[8] = 1;
                }
                if (x == 0)
                {
                    whetherEdge[0] = 1;
                    whetherEdge[3] = 1;
                    whetherEdge[6] = 1;
                }
                if (x == width - 1)
                {
                    whetherEdge[2] = 1;
                    whetherEdge[5] = 1;
                    whetherEdge[8] = 1;
                }

                int scaleNum = 0;

                for (int i = 0; i < 9; i++)
                {
                    scaleNum += (1 - whetherEdge[i]) * core[i];
                    if (whetherEdge[i] == 0)
                    {
                        destination->ascContent[rowOffset + x] += core[i] * source->ascContent[rowArroundIndex[i]];
                    }
                }
                if (needNormalizing)
                {
                    destination->ascContent[rowOffset + x] = (int)round(destination->ascContent[rowOffset + x] / (float)abs(scaleNum));
                }
            }
        }
        return 0;
    }

    int smooth(const mat *source, mat *destination, int weight)
    {
        int format = source->getFormat();
        if (format != 2)
        {
            std::cout << "Only support gray format." << std::endl;
            return 1;
        }
        int core[9] = {1, 1, 1, 1, weight, 1, 1, 1, 1};
        convolution(source, destination, core);
        return 0;
    }

    int sobel(const mat *source, mat *destination)
    {
        int format = source->getFormat();
        if (format != 2)
        {
            std::cout << "Only support gray format." << std::endl;
            return 1;
        }
        int gxCore[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
        int gyCore[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

        mat *gx = new mat;
        mat *gy = new mat;
        convolution(source, gx, gxCore);
        convolution(source, gy, gyCore);

        destination->initialize(source->getFormat(), source->getWidth(), source->getHigh(), source->getColordepth());

        int scale = 4;
        for (long long i = 0; i < destination->getValueNum(); i++)
        {
            destination->ascContent[i] = (int)round(sqrt(gx->ascContent[i] * gx->ascContent[i] + gy->ascContent[i] * gy->ascContent[i]) / ((float)scale * sqrt(2)));
        }
        return 0;
    }

    int laplacian(const mat *source, mat *destination)
    {
        int format = source->getFormat();
        if (format != 2)
        {
            std::cout << "Only support gray format." << std::endl;
            return 1;
        }
        int core[9] = {1, 1, 1, 1, -8, 1, 1, 1, 1};
        convolution(source, destination, core);

        float scale = 8;
        for (long long i = 0; i < destination->getValueNum(); i++)
        {
            destination->ascContent[i] = (int)round(abs(destination->ascContent[i]) / scale);
        }
        return 0;
    }
} // namespace ryu
