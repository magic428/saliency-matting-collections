#pragma once

#include <iostream>
#include <string>

#define BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/filesystem.hpp>
#undef BOOST_NO_CXX11_SCOPED_ENUMS
#include "boost/algorithm/string.hpp"

using namespace boost::filesystem;
namespace bfs = boost::filesystem;
using namespace std;

//TODO
/*
CmFile::mkdir(outDir);
CmFile::exists(salDir + names[i] + "_RCC.png"))

CmFile::get_filepaths(inImgW, names, inDir);
CmFile::WriteNullFile(nameNE + format("%d.nul", i));
CmFile::GetNameNE(names[i]);
CmFile::get_sub_folders(inDir, subFold);
CmFile::Copy(inDir + namesNE[i] + ".jpg", outDir + namesNE[i] + ".jpg");
*/


struct FileOps
{

    static bool mkdir(std::string&  path);

    /**
     * \brief: Check whether a file exist
     * 
     * \param: path - a directory path
    */
    static bool exists(std::string& path);
    
    static inline string GetFolder(std::string& path);

    /**
     * \brief: Get subfolders of a root directory  
     * 
     * \param: path - a root directory path
     * \param: sub_folders - subfolders in the root directory
    */
    static std::size_t get_sub_folders(std::string& folder, std::vector<std::string>& sub_folders);


    /**
     * \brief: Check if the filename matches the extensions
     *         eg: Image files has extensions: "*.jpg, *.png, *.jpeg"
     * 
     * \param: file         filename to check
     * \param: extensions   file extensions
     * 
     * \return: true - file matches extensions; 
     *          false - file not match
    */
    static bool check_ext(const std::string &file, const std::vector<std::string> &extensions);

    /**
     * \brief: Get file names from a directory with wildcard, NO SUBFOLDERS
     *         eg: "*.jpg, *.png, *.jpeg"
     * 
     * \param: path         a directory path
     * \param: names        Image name in the directory path
     * \param: extensions   file extensions
     * 
     * \return: filenames number; 
    */
    static std::size_t get_filepaths_in_a_folder( std::string &path, 
        std::vector<std::string> &names,
        std::vector<std::string> &extensions);

    /**
     * \brief: Get file names from a directory with extensions
     *         eg: imagfiles with "*.jpg, *.png,*.jpeg"
     * 
     * \param: path         a directory path
     * \param: names        file name in the directory path,
     * \param: extensions   file extensions
     * \param: has_subdir   also get file names in the subfolers  
     * 
     * \return: filenames number; 
     *
     * Note: MayBe WITH SUBFOLDERS
    */
    static std::size_t get_filepaths(std::string& root_path, 
        std::vector<std::string> &names, 
        std::vector<std::string> &extensions, 
        bool has_subdir = false);

    /**
     * \brief: Get image paths from a directory, *.jpg, *.png,*.jpeg
     * 
     * \param: path         a directory path
     * \param: paths        Image name in the directory path
     * \param: has_subdir   also get image names in the subfolers  
     * 
     * \return: filenames number; 
    */
    static std::size_t get_image_paths(std::string& root_path, 
        std::vector<std::string> &paths, bool has_subdir = false);

    static std::size_t get_image_names(std::string& root_path, 
    std::vector<std::string> &names, bool has_subdir = false);
    
    /**
     * \brief: Get file path names without extension 
     * 
     * \param: path - a directory path
     * \param: names - Image name in the directory path
     *
    */
    static std::size_t get_image_paths_without_ext(std::string& root_path, 
        std::vector<std::string> &names, 
        bool has_subdir = false);

    /**
     * \brief: Get file names without extension 
     * 
     * \param: path - a directory path
     * \param: names - Image name in the directory path
     * \param: has_subdir   also get image names in the subfolers  
     *
    */
    static std::size_t get_image_names_without_ext(std::string& root_path, 
        std::vector<std::string> &names, 
        bool has_subdir = false);

    /**
     * \brief: Get file names(not file paths) without extension 
     * 
     * \param: path - a directory path
     *
    */
    static inline std::string GetNameNE(std::string& path);
    
    /**
     * \brief: Copy file once 
     * 
     * \param: src  source file path
     * \param: dst  destination file path
     *
    */
    static inline bool copy_file(std::string &src, std::string &dst);

#if 0
    static inline bool files_exists(std::string& fileW);
    
    static string BrowseFile(const char* strFilter = "Images (*.jpg;*.png)\0*.jpg;*.png\0All (*.*)\0*.*\0\0", bool isOpen = true);
    static string BrowseFolder(); 

    static inline string GetSubFolder(std::string& path);
    static inline string GetName(std::string& path);
    static inline string GetPathNE(std::string& path);
    static inline string GetNameNoSuffix(std::string& path, std::string &suffix);


    static int GetNamesNoSuffix(std::string& root_path, std::vector<std::string> &namesNS, std::string suffix, string &dir); //e.g. suffix = "_C.jpg"
    static int GetNamesNoSuffix(std::string& root_path, std::vector<std::string> &namesNS, std::string suffix) {string dir; return GetNamesNoSuffix(root_path, namesNS, suffix, dir);}

    static inline string GetExtention(std::string name);

    static inline bool FolderExist(std::string& strPath);

    static inline string GetWkDir();

    // Eg: RenameImages("D:/DogImages/*.jpg", "F:/Images", "dog", ".jpg");
    static int Rename(std::string& srcNames, std::string& dstDir, const char* nameCommon, const char* nameExt);
    static void RenameSuffix(std::string dir, std::string orgSuf, std::string dstSuf);

    static int ChangeImgFormat(std::string &imgW, std::string dstW); // "./*.jpg", "./Out/%s_O.png"

    static inline void RmFile(std::string& fileW);
    static void RmFolder(std::string& dir);
    static void CleanFolder(std::string& dir, bool subFolder = false);

    static string GetFatherFolder(std::string &folder) {return GetFolder(folder.substr(0, folder.size() - 1));}


    inline static bool Move(std::string &src, std::string &dst, DWORD dwFlags = MOVEFILE_REPLACE_EXISTING | MOVEFILE_COPY_ALLOWED | MOVEFILE_WRITE_THROUGH);
    static bool Move2Dir(std::string &srcW, std::string dstDir);
    static bool Copy2Dir(std::string &srcW, std::string dstDir);

    //Load mask image and threshold thus noisy by compression can be removed
    static cv::Mat LoadMask(std::string& fileName);

    static void WriteNullFile(std::string& fileName) {FILE *f = fopen(_S(fileName), "w"); fclose(f);}
    static void AppendStr(std::string fileName, std::string str);

    static void ChkImgs(std::string &imgW);

    static void RunProgram(std::string &fileName, std::string &parameters = "", bool waiteF = false, bool showW = true);
    static string GetCompName(); // Get the name of computer

    static void SegOmpThrdNum(double ratio = 0.8);

    // Copy files and add suffix. e.g. copyAddSuffix("./*.jpg", "./Imgs/", "_Img.jpg")
    static void copyAddSuffix(std::string &srcW, std::string &dstDir, std::string &dstSuffix);

    static std::vector<std::string> loadStrList(std::string &fName);

    // Write matrix to binary file
    static bool matWrite(std::string& filename, CMat& M);
    static bool matWrite(FILE *f, CMat& M); // default FILE mode: "wb"
    // Read matrix from binary file
    static bool matRead( const string& filename, cv::Mat& M);
    static bool matRead(FILE *f, cv::Mat& M); // default FILE mode: "rb"

    // Needs 7-Zip to be installed and 7z.exe to be put under available path. compressLevel = 9 gives maximal compression
    static void ZipFiles(std::string &filesW, std::string &zipFileName, int compressLevel = 0);
    static void FileOps::UnZipFiles(std::string &zipFileName, std::string &tgtDir, bool overwriteWarning = true);
#endif
};

#if 0
/************************************************************************/
/* Implementation of inline functions                                   */
/************************************************************************/

string FileOps::GetSubFolder(std::string& path)
{
    string folder = path.substr(0, path.find_last_of("\\/"));
    return folder.substr(folder.find_last_of("\\/")+1);
}

string FileOps::GetName(std::string& path)
{
    int start = path.find_last_of("\\/")+1;
    int end = path.find_last_not_of(' ')+1;
    return path.substr(start, end - start);
}

string FileOps::GetNameNE(std::string& path)
{
    int start = path.find_last_of("\\/")+1;
    int end = path.find_last_of('.');
    if (end >= 0)
        return path.substr(start, end - start);
    else
        return path.substr(start,  path.find_last_not_of(' ')+1 - start);
}

string FileOps::GetNameNoSuffix(std::string& path, std::string &suffix)
{
    int start = path.find_last_of("\\/")+1;
    int end = path.size() - suffix.size();
    CV_Assert(path.substr(end) == suffix);
    if (end >= 0)
        return path.substr(start, end - start);
    else
        return path.substr(start,  path.find_last_not_of(' ')+1 - start);	
}

string FileOps::GetPathNE(std::string& path)
{
    int end = path.find_last_of('.');
    if (end >= 0)
        return path.substr(0, end);
    else
        return path.substr(0,  path.find_last_not_of(' ') + 1);
}

string FileOps::GetExtention(std::string name)
{
    return name.substr(name.find_last_of('.'));
}



bool FileOps::Move(std::string &src, std::string &dst, DWORD dwFlags)
{
    return MoveFileExA(src.c_str(), dst.c_str(), dwFlags);
}

void FileOps::RmFile(std::string& fileW)
{ 
    std::vector<std::string> names;
    string dir;
    int fNum = FileOps::get_filepaths(fileW, names, dir);
    for (int i = 0; i < fNum; i++)
        ::DeleteFileA(_S(dir + names[i]));
}




string FileOps::GetWkDir()
{	
    string wd;
    wd.resize(1024);
    DWORD len = GetCurrentDirectoryA(1024, &wd[0]);
    wd.resize(len);
    return wd;
}

bool FileOps::FolderExist(std::string& strPath)
{
    int i = (int)strPath.size() - 1;
    for (; i >= 0 && (strPath[i] == '\\' || strPath[i] == '/'); i--)
        ;
    string str = strPath.substr(0, i+1);

    WIN32_FIND_DATAA  wfd;
    HANDLE hFind = FindFirstFileA(_S(str), &wfd);
    bool rValue = (hFind != INVALID_HANDLE_VALUE) && (wfd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY);   
    FindClose(hFind);
    return rValue;
}
#endif
