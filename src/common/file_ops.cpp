#include "common/file_ops.h"

/**
 * \brief: Create Directory
 * 
 * \param: path - a directory path
*/
bool FileOps::mkdir(std::string & path)
{
    if(path.size() == 0)
        return false;
    
    boost::system::error_code ec;
    bfs::path bst_path(path);
    bfs::create_directories(bst_path, ec);
    
    if(ec)
        return false;
    else 
        return true;
}

/**
 * \brief: Check whether a file exist
 * 
 * \param: path - a directory path
*/
bool FileOps::exists(std::string& path)
{
    if (path.size() == 0)
        return false;

    bfs::path bst_path(path);

    return bfs::exists(bst_path);
}

string FileOps::GetFolder(std::string& path)
{
    return path.substr(0, path.find_last_of("\\/")+1);
}

/**
 * \brief: Get subfolders of a root directory  
 * 
 * \param: path - a root directory path
 * \param: sub_folders - subfolders in the root directory
*/
std::size_t FileOps::get_sub_folders(std::string& folder, std::vector<std::string>& sub_folders)
{
    sub_folders.clear();
    sub_folders.reserve(10000);

    bfs::path bst_path(folder);
    bfs::recursive_directory_iterator it_end;
    for (bfs::recursive_directory_iterator it(bst_path); it != it_end; ++it) {

        string sub_folder = it->path().string();
        if ( bfs::is_directory(it->path()) 
            && sub_folder != "." 
            && sub_folder != "..")

            sub_folders.push_back(sub_folder);
    }

    return sub_folders.size();
}

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
bool FileOps::check_ext(const std::string &file, const std::vector<std::string> &extensions)
{
    string file_ext = file.substr(file.find('.'));
    bool ext_match = false;

    for( std::vector<std::string>::const_iterator it = extensions.begin(); 
         it != extensions.end(); it++ ) { 

        if ( file_ext == *it ){
            ext_match = true;
            break;
        }
    }
    
    return ext_match;
}

/**
 * \brief: Get file paths from a directory with wildcard, NO SUBFOLDERS
 *         eg: "*.jpg, *.png, *.jpeg"
 * 
 * \param: path         a directory path
 * \param: paths        Image name in the directory path
 * \param: extensions   file extensions
 * 
 * \return: filenames number; 
*/
std::size_t FileOps::get_filepaths_in_a_folder( std::string &path, 
    std::vector<std::string> &paths,
    std::vector<std::string> &extensions )
{
    paths.clear();
    paths.reserve(10000);

    bfs::directory_iterator it_end;
    // bfs::recursive_directory_iterator it_end;
    bfs::path bst_path(path);
    // for (bfs::recursive_directory_iterator it(bst_path); it != it_end; ++it) {
    for (bfs::directory_iterator it(bst_path); it != it_end; ++it) {

        string file = it->path().string();
        if ( bfs::is_directory(it->path()) )
            continue;
        
        if(check_ext(file, extensions))
            paths.push_back(file);
    }

    return paths.size();
}

/**
 * \brief: Get file paths from a directory with extensions
 *         eg: imagfiles with "*.jpg, *.png,*.jpeg"
 * 
 * \param: path         a directory path
 * \param: paths        file name in the directory path,
 * \param: extensions   file extensions
 * \param: has_subdir   also get file paths in the subfolers  
 * 
 * \return: filenames number; 
 *
 * Note: MayBe WITH SUBFOLDERS
*/
std::size_t FileOps::get_filepaths(std::string& root_path, 
    std::vector<std::string> &paths, 
    std::vector<std::string> &extensions, 
    bool has_subdir)
{
    get_filepaths_in_a_folder(root_path, paths, extensions);

    if (has_subdir) {

        std::vector<std::string> sub_folders, names_in_sub_folder;
        std::size_t n_subfolders = get_sub_folders(root_path, sub_folders);

        // Get paths from each folder
        for (std::size_t i = 0; i < n_subfolders; i++){
            std::size_t n_subfolders = get_filepaths_in_a_folder(sub_folders[i], names_in_sub_folder, extensions);
            paths.insert(paths.end(), names_in_sub_folder.begin(), names_in_sub_folder.end());
        }
    }

    return paths.size();
}


std::size_t FileOps::get_image_paths(std::string& root_path, 
    std::vector<std::string> &paths, bool has_subdir)
{
    std::vector<std::string> extensions;
    extensions.push_back(".jpg");
    extensions.push_back(".png");
    extensions.push_back(".jpeg");

    return get_filepaths(root_path, paths, extensions, has_subdir);
}

std::size_t FileOps::get_image_names(std::string& root_path, 
    std::vector<std::string> &names, bool has_subdir)
{
    std::size_t fNum = get_image_paths(root_path, names, has_subdir);
    for (std::size_t i = 0; i < fNum; i++)
        names[i] = bfs::path(names[i]).filename().string(); 

    return fNum;
}


/**
 * \brief: Get file path names without extension 
 * 
 * \param: path - a directory path
 * \param: names - Image name in the directory path
 *
*/
std::size_t FileOps::get_image_paths_without_ext(std::string& root_path, 
    std::vector<std::string> &names, 
    bool has_subdir)
{
    std::size_t fNum = get_image_paths(root_path, names, has_subdir);
    for (std::size_t i = 0; i < fNum; i++)
        names[i] = bfs::path(names[i]).stem().string(); 

    return fNum;
}

/**
 * \brief: Get file names without extension 
 * 
 * \param: path - a directory path
 * \param: names - Image name in the directory path
 *
*/
std::size_t FileOps::get_image_names_without_ext(std::string& root_path, 
    std::vector<std::string> &names, 
    bool has_subdir)
{
    std::size_t fNum = get_image_paths(root_path, names, has_subdir);
    for (std::size_t i = 0; i < fNum; i++)
        names[i] = bfs::path(names[i]).filename().stem().string();

    return fNum;
}

/**
 * \brief: Get file names(not file paths) without extension 
 * 
 * \param: path - a directory path
 *
*/
std::string FileOps::GetNameNE(std::string& path)
{
    int start = path.find_last_of("\\/")+1;
    std::size_t pos = path.find_last_of('.');
    if (pos != std::string::npos)
        return path.substr(start, pos - start);
    else
        return path.substr(start, path.find_last_not_of(' ')+1 - start);
}


/**
 * \brief: Copy file once 
 * 
 * \param: src  source file path
 * \param: dst  destination file path
 *
*/
bool FileOps::copy_file(std::string &src, std::string &dst)
{
    boost::system::error_code ec;
    
    bfs::copy_file(bfs::path(src), bfs::path(dst), ec);

    if(ec)
        return false;
    else
        return true;
}

#if 0
bool FileOps::files_exists(std::string& fileW)
{
    std::vector<std::string> names;
    int fNum = get_filepaths(fileW, names);
    return fNum > 0;
}


int FileOps::GetNamesNoSuffix(std::string& root_path, std::vector<std::string> &namesNS, std::string suffix, string &dir)
{
    int fNum = FileOps::get_filepaths(root_path, namesNS, dir);
    for (int i = 0; i < fNum; i++)
        namesNS[i] = GetNameNoSuffix(namesNS[i], suffix);
    return fNum;
}
string FileOps::BrowseFolder()   
{
    static char Buffer[MAX_PATH];
    BROWSEINFOA bi;//Initial bi 	
    bi.hwndOwner = NULL; 
    bi.pidlRoot = NULL;
    bi.pszDisplayName = Buffer; // Dialog can't be shown if it's NULL
    bi.lpszTitle = "BrowseFolder";
    bi.ulFlags = 0;
    bi.lpfn = NULL;
    bi.iImage = NULL;


    LPITEMIDLIST pIDList = SHBrowseForFolderA(&bi); // Show dialog
    if(pIDList)	{	
        SHGetPathFromIDListA(pIDList, Buffer);
        if (Buffer[strlen(Buffer) - 1]  == '\\')
            Buffer[strlen(Buffer) - 1] = 0;

        return string(Buffer);
    }
    return string();   
}

string FileOps::BrowseFile(const char* strFilter, bool isOpen)
{
    static char Buffer[MAX_PATH];
    OPENFILENAMEA   ofn;  
    memset(&ofn, 0, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFile = Buffer;
    ofn.lpstrFile[0] = '\0';   
    ofn.nMaxFile = MAX_PATH;   
    ofn.lpstrFilter = strFilter;   
    ofn.nFilterIndex = 1;    
    ofn.Flags = OFN_PATHMUSTEXIST;   

    if (isOpen) {
        ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
        GetOpenFileNameA(&ofn);
        return Buffer;
    }

    GetSaveFileNameA(&ofn);
    return string(Buffer);

}

int FileOps::Rename(std::string& _srcNames, std::string& _dstDir, const char *nameCommon, const char *nameExt)
{
    std::vector<std::string> names;
    string inDir;
    int fNum = get_filepaths(_srcNames, names, inDir);
    for (int i = 0; i < fNum; i++) {
        string dstName = format("%s\\%.4d%s.%s", _S(_dstDir), i, nameCommon, nameExt);
        string srcName = inDir + names[i];
        ::CopyFileA(srcName.c_str(), dstName.c_str(), FALSE);
    }
    return fNum;
}

int FileOps::ChangeImgFormat(std::string &imgW, std::string dstW)
{
    std::vector<std::string> names;
    string inDir, ext = GetExtention(imgW);
    int iNum = get_filepaths(imgW, names, inDir);
#pragma omp parallel for
    for (int i = 0; i < iNum; i++) {
        Mat img = imread(inDir + names[i]);
        imwrite(format(_S(dstW), _S(GetNameNE(names[i]))), img);
    }
    return iNum;
}

void FileOps::RenameSuffix(std::string dir, std::string orgSuf, std::string dstSuf)
{
    std::vector<std::string> namesNS;
    int fNum = FileOps::GetNamesNoSuffix(dir + "*" + orgSuf, namesNS, orgSuf);
    for (int i = 0; i < fNum; i++)
        FileOps::Move(dir + namesNS[i] + orgSuf, dir + namesNS[i] + dstSuf);
}

void FileOps::RmFolder(std::string& dir)
{
    CleanFolder(dir);
    if (FolderExist(dir))
        RunProgram("Cmd.exe", format("/c rmdir /s /q \"%s\"", _S(dir)), true, false);
}

void FileOps::CleanFolder(std::string& dir, bool subFolder)
{
    std::vector<std::string> names;
    int fNum = FileOps::get_filepaths(dir + "/*.*", names);
    for (int i = 0; i < fNum; i++)
        RmFile(dir + "/" + names[i]);

    std::vector<std::string> sub_folders;
    int n_subfolders = get_sub_folders(dir, sub_folders);
    if (subFolder)
        for (int i = 0; i < n_subfolders; i++)
            CleanFolder(dir + "/" + sub_folders[i], true);
}




// Load mask image and threshold thus noisy by compression can be removed
Mat FileOps::LoadMask(std::string& fileName)
{
    Mat mask = imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);
    CV_Assert_(mask.data != NULL, ("Can't find mask image: %s", _S(fileName)));
    compare(mask, 128, mask, CV_CMP_GT);
    return mask;
}

bool FileOps::Move2Dir(std::string &srcW, std::string dstDir)
{
    std::vector<std::string> names;
    string inDir;
    int fNum = FileOps::get_filepaths(srcW, names, inDir);
    bool r = TRUE;
    for (int i = 0; i < fNum; i++)	
        if (Move(inDir + names[i], dstDir + names[i]) == FALSE)
            r = FALSE;
    return r;
}

bool FileOps::Copy2Dir(std::string &srcW, std::string dstDir)
{
    std::vector<std::string> names;
    string inDir;
    int fNum = FileOps::get_filepaths(srcW, names, inDir);
    bool r = TRUE;
    for (int i = 0; i < fNum; i++)	
        if (Copy(inDir + names[i], dstDir + names[i]) == FALSE)
            r = FALSE;
    return r;
}

void FileOps::ChkImgs(std::string &imgW)
{
    std::vector<std::string> names;
    string inDir;
    int imgNum = get_filepaths(imgW, names, inDir);
    printf("Checking %d images: %s\n", imgNum, _S(imgW));
    for (int i = 0; i < imgNum; i++){
        Mat img = imread(inDir + names[i]);
        if (img.data == NULL)
            printf("Loading file %s failed\t\t\n", _S(names[i]));
        if (i % 200 == 0)
            printf("Processing %2.1f%%\r", (i*100.0)/imgNum);
    }
    printf("\t\t\t\t\r");
}


void FileOps::AppendStr(std::string fileName, std::string str)
{
    FILE *f = fopen(_S(fileName), "a");
    if (f == NULL){
        printf("File %s can't be opened\n", _S(fileName));
        return;
    }
    fprintf(f, "%s", _S(str));
    fclose(f);
}


void FileOps::RunProgram(std::string &fileName, std::string &parameters, bool waiteF, bool showW)
{
    string runExeFile = fileName;
#ifdef _DEBUG
    runExeFile.insert(0, "..\\Debug\\");
#else
    runExeFile.insert(0, "..\\Release\\");
#endif // _DEBUG
    if (!FileOps::FileExist(_S(runExeFile)))
        runExeFile = fileName;

    SHELLEXECUTEINFOA  ShExecInfo  =  {0};  
    ShExecInfo.cbSize  =  sizeof(SHELLEXECUTEINFO);  
    ShExecInfo.fMask  =  SEE_MASK_NOCLOSEPROCESS;  
    ShExecInfo.hwnd  =  NULL;  
    ShExecInfo.lpVerb  =  NULL;  
    ShExecInfo.lpFile  =  _S(runExeFile);
    ShExecInfo.lpParameters  =  _S(parameters);         
    ShExecInfo.lpDirectory  =  NULL;  
    ShExecInfo.nShow  =  showW ? SW_SHOW : SW_HIDE;  
    ShExecInfo.hInstApp  =  NULL;              
    ShellExecuteExA(&ShExecInfo);  

    //printf("Run: %s %s\n", ShExecInfo.lpFile, ShExecInfo.lpParameters);

    if (waiteF)
        WaitForSingleObject(ShExecInfo.hProcess,INFINITE);
}

string FileOps::GetCompName() 
{
    char buf[1024];
    DWORD dwCompNameLen = 1024;
    GetComputerNameA(buf, &dwCompNameLen);
    return string(buf);
}

void FileOps::SegOmpThrdNum(double ratio /* = 0.8 */)
{
    int thrNum = omp_get_max_threads();
    int usedNum = cvRound(thrNum * ratio);
    usedNum = max(usedNum, 1);
    //printf("Number of CPU cores used is %d/%d\n", usedNum, thrNum);
    omp_set_num_threads(usedNum);
}


// Copy files and add suffix. e.g. copyAddSuffix("./*.jpg", "./Imgs/", "_Img.jpg")
void FileOps::copyAddSuffix(std::string &srcW, std::string &dstDir, std::string &dstSuffix)
{
    std::vector<std::string> namesNE;
    string srcDir, srcExt;
    int imgN = FileOps::get_image_names_without_ext(srcW, namesNE, srcDir, srcExt);
    FileOps::MkDir(dstDir);
    for (int i = 0; i < imgN; i++)
        FileOps::Copy(srcDir + namesNE[i] + srcExt, dstDir + namesNE[i] + dstSuffix);
}

std::vector<std::string> FileOps::loadStrList(std::string &fName)
{
    ifstream fIn(fName);
    string line;
    std::vector<std::string> strs;
    while(getline(fIn, line) && line.size())
        strs.push_back(line);
    return strs;
}


// Write matrix to binary file
bool FileOps::matWrite(std::string& filename, CMat& M){
    FILE* f = fopen(_S(filename), "wb");
    bool res = matWrite(f, M);
    if (f != NULL)
        fclose(f);	
    return res;
}

bool FileOps::matWrite(FILE *f, CMat& _M)
{
    Mat M;
    _M.copyTo(M);
    if (f == NULL || M.empty())
        return false;
    fwrite("CmMat", sizeof(char), 5, f);
    int headData[3] = {M.cols, M.rows, M.type()};
    fwrite(headData, sizeof(int), 3, f);
    fwrite(M.data, sizeof(char), M.step * M.rows, f);
    return true;
}

/****************************************************************************/
// Read matrix from binary file
bool FileOps::matRead( const string& filename, Mat& M){
    FILE* f = fopen(_S(filename), "rb");
    bool res = matRead(f, M);
    if (f != NULL)
        fclose(f);
    return res;
}

bool FileOps::matRead(FILE *f, Mat& M)
{
    if (f == NULL)
        return false;
    char buf[8];
    int pre = (int)fread(buf,sizeof(char), 5, f);
    if (strncmp(buf, "CmMat", 5) != 0)	{
        printf("Invalidate CvMat data file: %d:%s\n", __LINE__, __FILE__);
        return false;
    }
    int headData[3]; // Width, height, type
    fread(headData, sizeof(int), 3, f);
    M = Mat(headData[1], headData[0], headData[2]);
    fread(M.data, sizeof(char), M.step * M.rows, f);
    return true;
}

void FileOps::ZipFiles(std::string &filesW, std::string &zipFileName, int compressLevel)
{
    string param = format("u -tzip -mmt -mx%d \"%s\" \"%s\"", compressLevel, _S(zipFileName), _S(filesW));
    printf("Zip files: %s --> %s\n", _S(filesW), _S(zipFileName));
    RunProgram("7z.exe", param, true, false);
}


void FileOps::UnZipFiles(std::string &zipFileName, std::string &tgtDir, bool overwriteWarning/* = true*/)
{
    string param = format("e \"%s\" \"-o%s\" -r", _S(zipFileName), _S(tgtDir));
    if (!overwriteWarning)
        param += " -y";
    if (!FileExist(zipFileName))
        printf("File missing: %s\n", _S(zipFileName));

    if (overwriteWarning)
        printf("UnZip files: %s --> %s\n", _S(zipFileName), _S(tgtDir));
    FileOps::RunProgram("7z.exe", param, true, overwriteWarning);
}
#endif