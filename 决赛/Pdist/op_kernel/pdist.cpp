#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 1;                                     // tensor num for each queue


class KernelPDist{
public:
    __aicore__ inline KernelPDist(){}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,uint32_t N, uint32_t M, uint32_t allSize, uint32_t tilingSize, float pValue, uint32_t dataType){
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        uint32_t blockIdx = GetBlockIdx();
        uint32_t core_size = M;
        this->blockLength = M;
        this->tileLength = tilingSize;
        // this->tileLength = 32;
        this->pValue = pValue;
        // uint32_t first_idx = blockIdx / N;
        // uint32_t second_idx = blockIdx % N;
        uint32_t first_idx = 0, second_idx = 1;
        int now = 0;
        for(int i = N - 1, j = 0;i >= 0;i --, j ++){
            if(now + i >= blockIdx + 1){
                first_idx = j;
                second_idx = blockIdx + 1 - now + first_idx;
                break;
            }
            now += i;
        }
        auto f_start_pointer = first_idx * core_size;
        auto s_start_pointer = second_idx * core_size;
        auto y_pointer = blockIdx;
        
        firstGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + f_start_pointer, this->blockLength);
        secondGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + s_start_pointer, this->blockLength);
        
        // if(dataType == 0){ // float32
        //      this->f_length = 8;
        // }else{             // half
        //      this->f_length = 16;
        // }
        this->f_length = 1;
        yGM_f.SetGlobalBuffer((__gm__ DTYPE_X*)y + blockIdx, this->f_length);
        
        this->tileNum = (this->blockLength + this->tileLength - 1) / this->tileLength;
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(tQue, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(workQue, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        
    }
    __aicore__ inline void Process(){
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount-1; i++) {
            // PRINTF("Now loop count is %d \n, ", i);
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }
        auto length = this->blockLength - this->tileLength * (loopCount - 1);
        CopyIn(loopCount - 1, length);
        Compute(loopCount - 1, length);
        CopyOut(loopCount - 1, length);
        
        /* CopyIn */
        LocalTensor<DTYPE_X> tLocal = tQue.AllocTensor<DTYPE_X>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(DTYPE_X)), 0, 0, 0};
        DataCopyPadExtParams<DTYPE_X> padParams{true, 0, 0, 0};
        DataCopyPad(tLocal, yGM_f, copyParams, padParams);
        tQue.EnQue(tLocal);
        /* Compute */
        LocalTensor<DTYPE_X> outLocal = outQueue.AllocTensor<DTYPE_X>();
        tLocal = tQue.DeQue<DTYPE_X>();
        // Ln(tLocal, tLocal, this->f_length);
        // Muls(tLocal, tLocal, (DTYPE_X)(1.0f / this->pValue), this->f_length);
        // Exp(outLocal, tLocal, this->f_length);
        Power(outLocal, tLocal,  (DTYPE_X)(1.0f / this->pValue));
        tQue.FreeTensor(tLocal);
        outQueue.EnQue(outLocal);
        /* CopytOut */
        outLocal = outQueue.DeQue<DTYPE_X>();
        DataCopyPad(yGM_f, outLocal, copyParams);
        outQueue.FreeTensor(outLocal);
    }
private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length){
        LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(DTYPE_X)), 0, 0, 0};
        DataCopyPadExtParams<DTYPE_X> padParams{true, 0, 0, 0};
        DataCopyPad(xLocal, firstGm[progress * this->tileLength], copyParams, padParams);
        
        LocalTensor<DTYPE_X> yLocal = inQueueY.AllocTensor<DTYPE_X>();
        DataCopyPad(yLocal, secondGm[progress * this->tileLength], copyParams, padParams);
        
        LocalTensor<DTYPE_X> tLocal = tQue.AllocTensor<DTYPE_X>();
        DataCopyExtParams copyParams2{1, static_cast<uint32_t>(this->f_length * sizeof(DTYPE_X)), 0, 0, 0};
        DataCopyPad(tLocal, yGM_f, copyParams2, padParams);
        if(progress == 0){
            tLocal.SetValue(0, (DTYPE_X)0);
        }
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
        tQue.EnQue(tLocal);
    }
    
    __aicore__ inline void Compute(int32_t progress, uint32_t length){
        LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        LocalTensor<DTYPE_X> yLocal = inQueueY.DeQue<DTYPE_X>();
        LocalTensor<DTYPE_X> sumLocal = workQue.AllocTensor<DTYPE_X>();
        LocalTensor<DTYPE_X> tLocal = tQue.DeQue<DTYPE_X>();
        
        LocalTensor<DTYPE_X> outLocal = outQueue.AllocTensor<DTYPE_Y>();
        /* (\sum (x - y)^p)^{1/p} */
        
        Muls(yLocal, yLocal, (DTYPE_X)(-1), length);
        Add(xLocal, xLocal, yLocal, length);
        Abs(xLocal, xLocal, length);
        Ln(xLocal, xLocal, length);
        Muls(xLocal, xLocal, (DTYPE_X)this->pValue, length);
        Exp(yLocal, xLocal, length);
        ReduceSum(outLocal, yLocal, sumLocal, length);
        Add(outLocal, outLocal, tLocal, this->f_length);
        
        // PRINTF("line 129: outLocal: %f   \n\n\n\n", outLocal);
        outQueue.EnQue<DTYPE_Y>(outLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
        tQue.FreeTensor(tLocal);
        workQue.FreeTensor(sumLocal);
    }
    
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length){
        LocalTensor<DTYPE_X> outLocal = outQueue.DeQue<DTYPE_Y>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(DTYPE_X)), 0, 0, 0};
        DataCopyPad(yGM_f, outLocal, copyParams);
        // yGM_f.SetValue(0, outLocal.GetValue(0));
        outQueue.FreeTensor(outLocal);    
    }
private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> sumBuffer;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY, tQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue, workQue;
    GlobalTensor<DTYPE_X> firstGm;
    GlobalTensor<DTYPE_X> secondGm;
    // GlobalTensor<DTYPE_X> yGM;
    GlobalTensor<DTYPE_X> yGM_f;
    uint32_t N, M, allSize, tilingSize, tileNum, blockLength, f_length, tileLength;
    float pValue;
};


extern "C" __global__ __aicore__ void pdist(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelPDist op;
    op.Init(x, y, tiling_data.N, tiling_data.M, tiling_data.allSize, tiling_data.tilingSize, tiling_data.pValue, tiling_data.dataType);
    op.Process();
}