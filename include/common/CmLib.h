#pragma once
#pragma warning(disable: 4996)
#pragma warning(disable: 4995)
#pragma warning(disable: 4805)
#pragma warning(disable: 4267)
#pragma warning(disable: 4819)

// CmLib common coding help
#include "common/my_common.h"
#include "common/file_ops.h"
#include "common/CmCv.h"
//#include "common/cv_timer.h"

// Clustering algorithms
// #include "cluster/ap_cluster.h"
// #include "cluster/color_quantize.h"
#include "cluster/gmm.h"

//////Segmentation algorithms
#include "segmentation/EfficientGraphBased/segment-image.h"
// #include "Segmentation/PlanarCut/code/CutPlanar.h" // For Planar Cut
// #include "Segmentation/PlanarCut/code/CutGrid.h"
// #include "Segmentation/PlanarCut/code/CutShape.h"
// #include "Segmentation/Maxflow/graph.h"
// #include "Segmentation/MeanShift/msImageProcessor.h"


#if 0
// For illustration
#include "Illustration/CmShow.h"
#include "Illustration/CmIllustr.h"
#include "Illustration/CmEvaluation.h"
#include "Illustration/CmIllu.h"
//
//// Shape matching algorithms
//#include "ShapMatch/CmAffine.h"
//#include "ShapMatch/CmShapeContext.h"
//#include "ShapMatch/CmShapePnts.h"
//#include "ShapMatch/CmShape.h"


// Other algorithms
#include "OtherAlg/CmCurveEx.h"
#include "OtherAlg/CmValStructVec.h"
//#include "OtherAlg/CmNNF.h"
//#include "OtherAlg/CmWebImg.h"
//#include "OtherAlg/CmMatch.h"
//#include "OtherAlg/CmPatchMatch.h"
//#include "OtherAlg/CmImgQuilt.h"
//
//






//#include ".Cholmod/CholmodInclude.h"
//#include "Mating/AlphaMatting.h"
//#include "Mating/CmMatingCF.h"
//
////Geometry
//#include "Geometry/CmGeometry.h"
//#include "Geometry/CmPolyFit.h"

// Saliency detection algorithms
#include "Saliency/CmSaliencyRC.h"
#include "Saliency/CmSaliencyGC.h"
#include "Saliency/CmSalCut.h"
//#include "Saliency/CmGrabCutUI.h"
//
//
//// Interface and speech recognition
//#include "Basic/CmQt.h"
//#include "Illustration/CmMatShowGL.h"
//
//#ifdef _USE_MATLAB
//#include "Matlab/CmMatlab.h"
//#endif // _USE_MATLAB

// CRFs
#include "CRF/fastmath.h"
#include "CRF/permutohedral.h"


#define ToDo printf("To be implemented, %d:%s\n", __LINE__, __FILE__)

extern bool dbgStop;
#define DBG_POINT if (dbgStop) printf("%d:%s\n", __LINE__, __FILE__);

// #pragma comment(lib, lnkLIB("CmLib"))
#define CM_CODE

#endif
