
#include "pdist_tiling.h"

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

const uint32_t BLOCK_SIZE = 32;
const uint32_t BUFFER_NUM = 1;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  PdistTilingData tiling;
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  int32_t data_sz = 1;
  for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
    data_sz *= x1_shape->GetStorageShape().GetDim(i);
  tiling.set_size(data_sz);
  context->SetBlockDim(8);
    
/*  N  
    M 
    ALL_SIZE : N * (N - 1) / 2 */
    uint32_t N, M;
    N = x1_shape->GetStorageShape().GetDim(0);
    M = x1_shape->GetStorageShape().GetDim(1);
    uint32_t ALL_SIZE = N * (N - 1) / 2;
    tiling.set_N(N);
    tiling.set_M(M);
    tiling.set_allSize(ALL_SIZE);
    context->SetBlockDim(1);
/* pValue */
    float p = *(context->GetAttrs()->GetFloat(0));
    tiling.set_pValue(p);
/* data type */
    auto dt = context->GetInputDesc(0)->GetDataType();
    uint32_t sizeofdatatype;
    if (dt == 0 || dt == 3){ // float32
        sizeofdatatype = 4;
        tiling.set_dataType(0);
    }
    else if (dt == 1){ // float16
        sizeofdatatype = 2;
        tiling.set_dataType(1);
    }
     
/* ub_size */
    uint64_t ub_size;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size); 
    
/* ALIGN_NUM */
    uint32_t ubDataNumber = 16;
    uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;
    uint32_t tilingSize = (ub_size / BLOCK_SIZE / BUFFER_NUM) / ubDataNumber;
    tilingSize = tilingSize <= 8 ? tilingSize : tilingSize / 8 * 8;
    uint32_t blockSize = tilingSize * ALIGN_NUM;
    tiling.set_tilingSize(blockSize);
    


  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class Pdist : public OpDef {
public:
    explicit Pdist(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("p").AttrType(OPTIONAL).Float(2.0);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(Pdist);
}
