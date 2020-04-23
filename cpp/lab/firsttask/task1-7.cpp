
#include "image.cpp"

int main(int argc, char const *argv[])
{
    if (argc == 2)
    {
        image myImage(argv[1]);
        myImage.printInfor();
    }
    else if (argc > 2)
    {
        char *outputPath;
        if (argc > 3)
        {
            outputPath = new char[200];
            strcpy(outputPath, argv[3]);
        }
        else
        {
            // Use default outputPath.
            outputPath = nullptr;
        }

        image myImage(argv[1]);
        if (strcmp(argv[2], "copy") == 0)
        {
            myImage.copy(outputPath);
        }
        else if (strcmp(argv[2], "convert") == 0)
        {
            myImage.convert(outputPath);
        }
        else if (strcmp(argv[2], "flipv") == 0)
        {
            myImage.flip('v', outputPath);
        }
        else if (strcmp(argv[2], "fliph") == 0)
        {
            myImage.flip('h', outputPath);
        }
        else if (strcmp(argv[2], "shrinkn") == 0)
        {
            if (argc == 5)
            {
                myImage.shrink('n',argv[4], outputPath);
            }
            else
            {
                std::cout << "Please input shrink times." << std::endl;
            }
        }
        else if (strcmp(argv[2], "shrinki") == 0)
        {
            if (argc == 5)
            {
                myImage.shrink('i',argv[4], outputPath);
            }
            else
            {
                std::cout << "Please input shrink times." << std::endl;
            }
        }
        else if (strcmp(argv[2], "rotation") == 0)
        {
            if (argc == 5)
            {
                myImage.rotation(argv[4], outputPath);
            }
            else
            {
                std::cout << "Please input rotate angle." << std::endl;
            }
        }
        else if (strcmp(argv[2], "smooth") == 0)
        {
            if (argc == 5)
            {
                myImage.smooth(argv[4], outputPath);
            }
            else
            {
                myImage.smooth("1", outputPath);
            }
        }
        else if (strcmp(argv[2], "sobel") == 0)
        {
            myImage.filter('s', outputPath);
        }
        else if (strcmp(argv[2], "laplacian") == 0)
        {
            myImage.filter('l', outputPath);
        }
        else
        {
            std::cout << "Command wrong. " << std::endl;
        }
    }

    return 0;
}
