//
//  MatmulBufExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#include <MNN/AutoTime.hpp>
#include "backend/opencl/execution/buffer/MatmulBufExecution.hpp"

namespace MNN {
namespace OpenCL {


    MatMulBufExecution::MatMulBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend,
                                           bool transposeA, bool transposeB) : Execution(backend)
                                            , mTransposeA(transposeA), mTransposeB(transposeB){
        mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
//        MNN_PRINT("creating Matmul-buffer-obj");
    }

#define CUSTOM_KERNEL

#ifdef CUSTOM_KERNEL

#define WIDTH 4
#define TS 32
//#define
    ErrorCode MatMulBufExecution::onResize(const std::vector<Tensor *> &inputs,
                                           const std::vector<Tensor *> &outputs) {
//        MNN_PRINT("called matmulbuf onresize");

        auto runtime = mOpenCLBackend->getOpenCLRuntime();


        Tensor *input0 = inputs[0];
        Tensor *input1 = inputs[1];
        Tensor *output = outputs[0];

        std::vector<int> input0Shape = tensorShapeFormat(input0);
        std::vector<int> input1Shape = tensorShapeFormat(input1);
        std::vector<int> outputShape = tensorShapeFormat(output);

        MNN::Timer t;

        if (mKernel.get() == nullptr) {
            std::set<std::string> buildOptions;

            if(mTransposeA) {
                mKernelName = mTransposeB ? "gemm4_transA_transB":"gemm4_transA";
            } else {
                mKernelName = mTransposeB ? "gemm4_transB":"gemm4";
            }

            if (inputs.size() > 2) {
                buildOptions.emplace("-DBIAS -DKERNEL=4 -DWIDTH=4 -DTS4=16");
            } else {
                buildOptions.emplace("-DKERNEL=4 -DWIDTH=4 -DTS4=32");
            }

            mKernel = runtime->buildKernel("myGEMM", mKernelName, buildOptions);
        }


//        if (inputs.size() > 2) {
//
//            Tensor *input2 = inputs[2];
//            std::vector<int> shape = tensorShapeFormat(input2);
//            MNN_PRINT("bias shape: %d, %d, %d, %d", shape.at(0), shape.at(1), shape.at(2), shape.at(3));
//        }

        cl_int res = CL_SUCCESS;

        const int M = (mTransposeA) ? input0Shape.at(3) : input0Shape.at(0);
        const int N = (mTransposeB) ? input1Shape.at(0) : input1Shape.at(3);
        const int K = (mTransposeA) ? input0Shape.at(0) : input0Shape.at(3);
        const int numTiles = UP_DIV(K, 16);

        int idx = 0;
        res |= mKernel.setArg(idx++, M);
        res |= mKernel.setArg(idx++, N);
        res |= mKernel.setArg(idx++, K);

        res |= mKernel.setArg(idx++, openCLBuffer(input0));
        res |= mKernel.setArg(idx++, openCLBuffer(input1));
        res |= mKernel.setArg(idx++, openCLBuffer(output));

        res |= mKernel.setArg(idx++, numTiles);

        MNN_CHECK_CL_SUCCESS(res, "matmul_buf");

        uint32_t global0 = M,
                 global1 = N / 4,
                 local0 = 16,
                 local1 = 16 / 4;

        mGlobalWorkSize = {global0, global1};
        mLocalWorkSize = {local0, local1};

        return NO_ERROR;
    }

    ErrorCode MatMulBufExecution::onExecute(const std::vector<Tensor *> &inputs,
                                            const std::vector<Tensor *> &outputs) {
//        MNN_PRINT("DEBUG: HERE");
#ifdef LOG_VERBOSE
        MNN_PRINT("Start MatMulBufExecution onExecute... \n");
#endif

        auto runtime = mOpenCLBackend->getOpenCLRuntime();

#ifdef ENABLE_OPENCL_TIME_PROFILER
        cl::Event event;
        runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize, runtime, &event);
        event.wait();

        double costTime = mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
        MNN_PRINT("kernel cost:%f    us MatmulBuf\n",costTime);
        MNN_PRINT("KERNEL submit: %f us", mOpenCLBackend->getOpenCLRuntime()->getSubmitTime(&event));
        MNN_PRINT("KERNEL queue: %f us", mOpenCLBackend->getOpenCLRuntime()->getQueuedTime(&event));
#else
        runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize, runtime, nullptr);
#endif

#ifdef LOG_VERBOSE
        MNN_PRINT("End MatMulBufExecution onExecute... \n");
#endif
        return NO_ERROR;
    }



#else
//
//MatMulBufExecution::MatMulBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend,
//                                 bool transposeA, bool transposeB) : Execution(backend)
//                                 , mTransposeA(transposeA), mTransposeB(transposeB){
//    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
//}
ErrorCode MatMulBufExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = mOpenCLBackend->getOpenCLRuntime();

    Tensor *input0 = inputs[0];
    Tensor *input1 = inputs[1];
    Tensor *output = outputs[0];

    std::vector<int> input0Shape = tensorShapeFormat(input0);
    std::vector<int> input1Shape = tensorShapeFormat(input1);
    std::vector<int> outputShape = tensorShapeFormat(output);
    
    if (mKernel.get() == nullptr) {
        std::set<std::string> buildOptions;
        if(mTransposeA) {
            mKernelName = mTransposeB ? "matmul_transA_transB_buf":"matmul_transA_buf";
        } else {
            mKernelName = mTransposeB ? "matmul_transB_buf":"matmul_buf";
        }

        if(inputs.size() > 2) {
            buildOptions.emplace("-DBIAS");
        }
        mKernel           = runtime->buildKernel("matmul_buf", mKernelName, buildOptions);
        mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
    }

    //处理二维矩阵相乘，N C相当于H W
    //二维矩阵相乘
    cl_int ret = CL_SUCCESS;
    if(mTransposeA) {
        const int height        = input0Shape.at(3);//input0 H
        const int outputChannel = input0Shape.at(0);//input0 W
        const int width         = mTransposeB ? input1Shape.at(0): input1Shape.at(3);//input1 WW
        const int outputChannelBlocks = UP_DIV(outputChannel, 4);
        const int widthblocks         = UP_DIV(width, 4);
        const int heightblocks        = UP_DIV(height, 4);
        
        mGlobalWorkSize = {static_cast<uint32_t>(widthblocks), static_cast<uint32_t>(heightblocks)};
        int idx            = 0;
        ret |= mKernel.setArg(idx++, mGlobalWorkSize[0]);
        ret |= mKernel.setArg(idx++, mGlobalWorkSize[1]);
        ret |= mKernel.setArg(idx++, openCLBuffer(input0));
        ret |= mKernel.setArg(idx++, openCLBuffer(input1));
        if(inputs.size() > 2) {
            ret |= mKernel.setArg(idx++, openCLBuffer(inputs[2]));
        }
        ret |= mKernel.setArg(idx++, openCLBuffer(output));
        ret |= mKernel.setArg(idx++, static_cast<int>(outputChannel));
        ret |= mKernel.setArg(idx++, static_cast<int>(outputChannelBlocks));
        ret |= mKernel.setArg(idx++, static_cast<int>(height));
        ret |= mKernel.setArg(idx++, static_cast<int>(heightblocks));
        ret |= mKernel.setArg(idx++, static_cast<int>(widthblocks));
        
        mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), mKernelName, mKernel).first;
    }
    else {
        const int height        = input0Shape.at(0);//input0 H
        const int outputChannel = input0Shape.at(3);//input0 W
        const int width         = mTransposeB ? input1Shape.at(0): input1Shape.at(3);//input1 W
        const int outputChannelBlocks = UP_DIV(outputChannel, 4);
        const int widthblocks         = UP_DIV(width, 4);
        
        mGlobalWorkSize = {static_cast<uint32_t>(widthblocks), static_cast<uint32_t>(height)};
        int idx            = 0;
        ret |= mKernel.setArg(idx++, mGlobalWorkSize[0]);
        ret |= mKernel.setArg(idx++, mGlobalWorkSize[1]);
        ret |= mKernel.setArg(idx++, openCLBuffer(input0));
        ret |= mKernel.setArg(idx++, openCLBuffer(input1));
        if(inputs.size() > 2) {
            ret |= mKernel.setArg(idx++, openCLBuffer(inputs[2]));
        }
        ret |= mKernel.setArg(idx++, openCLBuffer(output));
        ret |= mKernel.setArg(idx++, static_cast<int>(outputChannel));
        ret |= mKernel.setArg(idx++, static_cast<int>(outputChannelBlocks));
        ret |= mKernel.setArg(idx++, static_cast<int>(widthblocks));
        
        mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), mKernelName, mKernel).first;
    }
    MNN_CHECK_CL_SUCCESS(ret, "matmul_buf");
    return NO_ERROR;
}

ErrorCode MatMulBufExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

#ifdef LOG_VERBOSE
    MNN_PRINT("Start MatMulBufExecution onExecute... \n");
#endif

    auto runtime = mOpenCLBackend->getOpenCLRuntime();

    #ifdef ENABLE_OPENCL_TIME_PROFILER
        cl::Event event;
        runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize, runtime, &event);
        
        int costTime = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
        MNN_PRINT("kernel cost:%d    us MatmulBuf\n",costTime);
    #else
    runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize, runtime, nullptr);
    #endif
    
#ifdef LOG_VERBOSE
    MNN_PRINT("End MatMulBufExecution onExecute... \n");
#endif
    return NO_ERROR;
}




#endif // CUSTOM_KERNEL

        class MatMulBufCreator : public OpenCLBackend::Creator {
        public:
            virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                        const MNN::Op *op, Backend *backend) const override {
                auto param = op->main_as_MatMul();
                return new MatMulBufExecution(inputs, op, backend, param->transposeA(), param->transposeB());
            }
        };

        OpenCLCreatorRegister<MatMulBufCreator> __matmulBuf_op(OpType_MatMul, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
