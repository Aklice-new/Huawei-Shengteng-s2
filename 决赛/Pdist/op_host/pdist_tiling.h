
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(PdistTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, size);
    /*N, M, ALL_SIZE*/
    TILING_DATA_FIELD_DEF(uint32_t, N);
    TILING_DATA_FIELD_DEF(uint32_t, M);
    TILING_DATA_FIELD_DEF(uint32_t, allSize);
    /*dataType*/
    TILING_DATA_FIELD_DEF(uint32_t, dataType);
    /*tilingSize*/
    TILING_DATA_FIELD_DEF(uint32_t, tilingSize);
    /*pValue*/
    TILING_DATA_FIELD_DEF(float, pValue);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Pdist, PdistTilingData)
}
