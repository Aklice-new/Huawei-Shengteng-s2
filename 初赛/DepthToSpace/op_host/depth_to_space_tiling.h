
#include "register/tilingdata_base.h"

/*
tiling 这里设置了多个参数：
size : int  

mode: int   0 is "DCR" ;  1 is "CRD"
data_format: int  0 is "NCHW" ; 1 is "NHWC"
block_size: int 
block_num:  int
*/
namespace optiling {
BEGIN_TILING_DATA_DEF(DepthToSpaceTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, size);
/* mode */
    TILING_DATA_FIELD_DEF(uint32_t, mode);
/* data_format */
    TILING_DATA_FIELD_DEF(uint32_t, dataFormat);
/* block_size */
    TILING_DATA_FIELD_DEF(uint32_t, blockSize);
/* block_num */
    TILING_DATA_FIELD_DEF(uint32_t, blockNum);
/* block length */
    TILING_DATA_FIELD_DEF(uint32_t, blockLength);
/* N C H W */
    TILING_DATA_FIELD_DEF(uint32_t, N);
    TILING_DATA_FIELD_DEF(uint32_t, C);
    TILING_DATA_FIELD_DEF(uint32_t, H);
    TILING_DATA_FIELD_DEF(uint32_t, W);
/* total length */
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
/* alignedTile */
    TILING_DATA_FIELD_DEF(uint32_t, alignedTile);
/* dataType */
    TILING_DATA_FIELD_DEF(uint32_t, dataType);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DepthToSpace, DepthToSpaceTilingData)
}
