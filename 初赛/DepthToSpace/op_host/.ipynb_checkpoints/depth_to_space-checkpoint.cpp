
#include "depth_to_space_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


/*
    分析：
    对于一个depthtospace算子，reshape阶段不需要计算，只需要完成transpose过程中数据的搬运。
    对与dims[0,1,2,3,4,5]这几个维度来说，transpose前后变化的维度用b表示，维度保持不变的用a表示
    则四种情况为：
    [a, b, b, b, b, b]
    [a, a, b, b, b, b]
    [a, a, b, b, a, a]
    [a, a, b, b, b, b]
    可以将其分为三个部分，前，中，后，
    前面部分不变，则可以讲前面这些维度的划分到不同的AICORE上进行计算
    中间变的部分就是需要遍历计算的部分
    后面不变的部分，对应到内存中的数据一直保持连续，所以可以连续搬运
*/


namespace optiling {
const uint32_t BLOCK_SIZE = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  DepthToSpaceTilingData tiling;
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  int32_t data_sz = 1;
  // printf("Dim Num is %d \n ", x1_shape->GetStorageShape().GetDimNum());
  for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
    data_sz *= x1_shape->GetStorageShape().GetDim(i);
  tiling.set_size(data_sz);
  /*get ub_size and aivNum */
  uint64_t ub_size;
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
  auto aivNum = ascendcPlatform.GetCoreNumAiv();
  /* attr */
    int32_t block_size = *context->GetAttrs()->GetInt(0);
    const char *mode_s = context->GetAttrs()->GetStr(1);
    const char *data_format_s = context->GetAttrs()->GetStr(2);
    tiling.set_blockSize(block_size);
    
    uint32_t mode, dataFormat;
    
    if(strcmp(mode_s, "DCR") == 0){
        tiling.set_mode(0);
        mode = 0;
    }else{
        tiling.set_mode(1);
        mode = 1;
    }
    /* N C H W */
    int32_t N,C,H,W;
    if(strcmp(data_format_s, "NCHW") == 0){
        tiling.set_dataFormat(0);
        dataFormat = 0;
        N = x1_shape->GetStorageShape().GetDim(0);
        C = x1_shape->GetStorageShape().GetDim(1);
        H = x1_shape->GetStorageShape().GetDim(2);
        W = x1_shape->GetStorageShape().GetDim(3);
    }else{
        tiling.set_dataFormat(1);
        dataFormat = 1;
        N = x1_shape->GetStorageShape().GetDim(0);
        C = x1_shape->GetStorageShape().GetDim(3);
        H = x1_shape->GetStorageShape().GetDim(1);
        W = x1_shape->GetStorageShape().GetDim(2);
    }
    tiling.set_N(N);
    tiling.set_C(C);
    tiling.set_H(H);
    tiling.set_W(W);
  /* total length */
    uint32_t totalLengthAligned;
    /* multie core */
    context->SetBlockDim(N);
  /* data type */
    auto dt = context->GetInputDesc(0)->GetDataType();
    uint32_t sizeofdatatype;
    if (dt == 0 || dt == 3){ // float32 | int32
        sizeofdatatype = 4;
        tiling.set_dataType(0);
    }
    else if (dt == 1){ // float16
        sizeofdatatype = 2;
        tiling.set_dataType(1);
    }
    else if (dt == 2){ // int8
        sizeofdatatype = 1;
    }
    
    uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;
    if (data_sz % ALIGN_NUM != 0){
        totalLengthAligned =
            ((data_sz + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
    }else{
        totalLengthAligned = data_sz;
    }
    tiling.set_totalLength(totalLengthAligned);
    /* aligned tile */
    uint32_t alignedTile;
    uint32_t realTile;
    if(mode == 0 && dataFormat == 0){
        realTile = W;
        /* multie core */
        context->SetBlockDim(N);
        tiling.set_alignedTile(ALIGN_NUM);
    }
    else if(mode == 1 && dataFormat == 0){
        realTile = W;
        /* multie core */
        context->SetBlockDim(N * (C / (block_size * block_size)));
        tiling.set_alignedTile(ALIGN_NUM);
    }
    else if(mode == 0 && dataFormat == 1){
        realTile = block_size * (C / (block_size * block_size));
        /* multie core */
        context->SetBlockDim(N * H);
        auto beshu = (realTile + ALIGN_NUM - 1) / ALIGN_NUM;
        alignedTile = ALIGN_NUM * beshu;
        tiling.set_alignedTile(alignedTile);
    }
    else if(mode == 1 && dataFormat == 1){
        realTile = block_size;
        /* multie core */
        context->SetBlockDim(N * H);
        tiling.set_alignedTile(ALIGN_NUM);
    }
    
    
    
//     context->SetBlockDim(8);
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
class DepthToSpace : public OpDef {
public:
    explicit DepthToSpace(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("block_size").Int();
        this->Attr("mode").AttrType(OPTIONAL).String("DCR");
        this->Attr("data_format").AttrType(OPTIONAL).String("NCHW");

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(DepthToSpace);
}
