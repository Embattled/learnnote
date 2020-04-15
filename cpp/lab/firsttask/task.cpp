
#include <iostream>
#include <fstream>
#include <string>
#define SUCCESS 0
#define OPENERROR 1
#define ERRORFILE 2
#define WRERROR 3
#define NOTMATCH 5
#define LOGICALERROR

int sourceType, sourceWidth, sourceHigh, sourceColorDepth;

// To check whether this file has right head and print information.
int fileCheck(char const *fileName)
{
  std::ifstream myfile(fileName, std::ios::in);
  if (!myfile.is_open())
  {
    std::cout << "File doesn't exist!" << std::endl;
    return OPENERROR;
  }

  std::string format;
  int type;
  myfile >> format;
  if (format.length() != 2 || format.find('P') != 0)
  {
    myfile.close();
    return ERRORFILE;
  }
  type = format.c_str()[1] - 48;
  if (type > 6 || type < 1)
  {
    myfile.close();
    return ERRORFILE;
  }
  sourceType = type;

  int high, width, colorDepth = 0;
  myfile >> width >> high;
  if (type != 1 && type != 4)
  {
    myfile >> colorDepth;
  }
  if (myfile.good())
  {
    sourceWidth = width;
    sourceHigh = high;
    sourceColorDepth = colorDepth;
    myfile.close();

    std::cout << "This image is P" << type << " Image" << std::endl;
    std::cout << width << "x" << high << "\tColor depth: " << colorDepth << std::endl;
    return SUCCESS;
  }
  else
  {
    myfile.close();
    return ERRORFILE;
  }
}

//This function is for task1
int imageCopy(char const *inName, char const *outName)
{
  std::ifstream inputFile(inName, std::ios::binary | std::ios::in | std::ios::ate);
  std::streampos size = inputFile.tellg();
  inputFile.seekg(0, std::ios_base::beg);

  char *buffer = new char[size];
  inputFile.read(buffer, size);

  std::ofstream outputFile(outName, std::ios::binary | std::ios::out | std::ios::trunc);
  outputFile.write(buffer, size);

  delete buffer;
  inputFile.close();
  outputFile.close();
  if (inputFile.good() && outputFile.good())
  {
    return SUCCESS;
  }
  else
  {
    return WRERROR;
  }
}

//This two function is for task2
int pgm2ppm(char const *inName, char const *outName)
{
  if(sourceType==2)
  {
    std::ifstream infile(inName,std::ios::in|std::ios::ate);

  }
  else if (sourceType==5)
  {
    std::ifstream infile(inName,std::ios::in|std::ios::binary);

  }
  return LOGICALERROR;
}
int pgm2pbm()
{

}

int imageConvert(char const *inName, char const *outName)
{

}



int main(int argc, char const *argv[])
{
  int result = SUCCESS;

  //Command recognition.
  if (argc == 1)
  {

    std::cout << "Please input your command" << std::endl;
    std::cout << "For task1, print size of image and copy it. Like:\ntask input.pgm output.pgm" << std::endl;

    return SUCCESS;
  }
  else if (argc == 2)
  {
    result = fileCheck(argv[1]);
  }
  else if (argc == 3)
  {
    //For task1, copy image.
    result = imageCopy(argv[1], argv[2]);
  }
  else if (argc==4)
  {
    
    if (!strcmp(argv[2],">"))
    {
      result=imageConvert(argv[1],argv[3]);
    }
    
  }
  

  return 0;
}
