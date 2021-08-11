#include <jni.h>
#include <string>

//General
#include "MNN/Tensor.hpp"
#include "MNN_generated.h"

//opencl resources
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/runtime/OpenCLRuntime.hpp"
#include "MNN/expr/Executor.hpp"
#include "backend/opencl/execution/buffer/MatmulBufExecution.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"


//CPU resources:
#include "backend/cpu/CPUMatMul.hpp"
#include "backend/cpu/CPUBackend.hpp"

#define WIDTH 8

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::OpenCL;


void cpu() {
    std::shared_ptr<Executor> executor = Executor::getGlobalExecutor();
    BackendConfig config;
    Backend::Info info;
    info.type = MNN_FORWARD_CPU;
    info.mode = Backend::Info::DIRECT;
    info.numThread = 1;
    info.user = (BackendConfig*)&config;

    CPURuntime runtime(info);
    CPUBackend backend(&runtime, config.precision);

    int width = 8;

    float hostA[width][width];
    float hostB[width][width];
    float hostC[width][width];

    for (int i = 0; i<width; i++) {
        for (int j = 0; j < width; j++) {
            hostA[i][j] = (float) (i*width + j);
            hostB[i][j] = (float) (i*width + j);
            hostC[i][j] = 0.0f;
        }
    }

    std::vector<int> shape({width, width});
    Tensor* tensorA = Tensor::create<float>(shape, hostA);
    Tensor* tensorB = Tensor::create<float>(shape, hostB);
    Tensor* tensorC = Tensor::create<float>(shape, hostC);

    std::vector<Tensor *> inputs({tensorA, tensorB}), outputs({tensorC});

    CPUMatMul matmul(&backend, false, false, false, false);

    matmul.onResize(inputs, outputs);
//    MNN_CHECK_CL_SUCCESS(code, "matmul.onResize");
//
    matmul.onExecute(inputs, outputs);

    tensorC->print();



}

double gpu(int width = 32) {
#define KERNEL 4

//defines for gemm2
#define TS2 32
//defines for gemm3
#define TS3 32
#define WPT (TS3 * TS3 / 256)
#define RTS TS3 / (WPT)
//defines for gemm4
#define TS4 32
#ifdef WIDTH
#undef WIDTH
#endif
#define WIDTH 8


    int res;
    BackendConfig config;
    std::shared_ptr<Executor> executor = Executor::getGlobalExecutor();
    executor->setGlobalExecutorConfig(MNN_FORWARD_OPENCL, config, 1);

//    BackendConfig config;
    Backend::Info info;
    info.type = MNN_FORWARD_OPENCL;
    info.mode = Backend::Info::DIRECT;
    info.numThread = 1;
    info.user = (BackendConfig*)&config;


//    OpenCL::CLRuntime cl_runtime(info);
//    OpenCL::OpenCLBackend backend(&cl_runtime);
//    OpenCLRuntime *runtime = backend.getOpenCLRuntime();
    OpenCLRuntime runtime_(config.precision,
                           MNN_GPU_MEMORY_BUFFER | MNN_GPU_TUNING_WIDE);
    OpenCLRuntime *runtime = &runtime_;
    runtime->setCommandQueueProfileEnable();

    cl::Context context = runtime->context();
    cl::CommandQueue commandQueue = runtime->commandQueue();


#if KERNEL == 1 || KERNEL == 2 || KERNEL == 3
    float hostA[width][width];
    float hostB[width][width];
    float hostC[width][width];

    for (int i = 0; i<width; i++) {
        for (int j = 0; j < width; j++) {
            hostA[i][j] = ((float) (i*width + j))/1000;
            hostB[i][j] = ((float) (i*width + j))/1000;
            hostC[i][j] = 0.0f;
        }
    }
#elif KERNEL == 4
    int widthTiles = width;
    widthTiles /= WIDTH;
    if (width % WIDTH != 0)
        widthTiles++;

    int adjustedWidth = widthTiles * WIDTH;
    float hostA[adjustedWidth][adjustedWidth];
    float hostB[adjustedWidth][adjustedWidth];
    float hostC[adjustedWidth][adjustedWidth];

//    for (int i = 0; i<width; i++) {
    for (int col = 0; col < adjustedWidth; col++) {
//        for (int j = 0; j < width; j++) {
        for (int row = 0; row < adjustedWidth; row++) {
            if (row < width && col < width) {
                hostA[col][row] = (float) 1;//(col * width + row);// / 1000.0f;
                hostB[col][row] = (float) 1;//(col * width + row);// / 1000.0f;
//                MNN_PRINT("%d ==  %f", col*width + row, hostA[col * adjustedWidth + row]);

            } else {
                hostA[col][row] = 0.0f;
                hostB[col][row] = 0.0f;
//                MNN_PRINT("removed: %d", col*adjustedWidth + row);
            }
            hostC[col][row] = 0.0f;
        }
    }
//    for (int i = 0; i < width*adjustedWidth/4; i++) {
//        float* temp= hostA[(4*i)/adjustedWidth];
//        int j = (4*i)%adjustedWidth;
//        MNN_PRINT("%f, %f, %f, %f", temp[j], temp[j+1], temp[j+2], temp[j+3]);
//    }
#endif

//    std::vector<int> shape({width, width});
//    Tensor* tensorA = Tensor::create<float>(shape, hostA);
//    Tensor* tensorB = Tensor::create<float>(shape, hostB);
//    Tensor* tensorC = Tensor::create<float>(shape, hostC);
//
//    std::vector<Tensor *> inputs({tensorA, tensorB}), outputs({tensorC});
//
//    MatMulBufExecution matmul(inputs, nullptr, &backend, false, false);
//    int res = matmul.onResize(inputs, outputs);
//    MNN_CHECK_CL_SUCCESS(res, "matmul.onResize");
//    res = matmul.onExecute(inputs, outputs);
//    MNN_CHECK_CL_SUCCESS(res, "matmul.onExecute");
//    commandQueue.finish();
//    tensorA->print();
//    tensorB->print();
//    tensorC->print();

//#define KERNEL3


    std::set<std::string> buildOptions;
#if KERNEL == 1
//#ifdef KERNEL1
    cl::Kernel kernel = runtime->buildKernel("myGEMM", "gemm1", buildOptions);
#elif KERNEL == 2
    cl::Kernel kernel = runtime->buildKernel("myGEMM", "gemm2", buildOptions);
#elif KERNEL == 3
    cl::Kernel kernel = runtime->buildKernel("myGEMM", "gemm3", buildOptions);
#elif KERNEL == 4
    cl::Kernel kernel = runtime->buildKernel("myGEMM", "gemm4", buildOptions);
#endif
//    MNN_PRINT("Max local size: %lu", runtime->getMaxWorkGroupSize(kernel));

    res = 0;
    cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;


#if KERNEL == 4

    cl::Buffer bufferA(context, flags, 4*adjustedWidth*width, hostA);
    cl::Buffer bufferB(context, flags, 4*adjustedWidth*width, hostB);
    cl::Buffer bufferC(context, flags, 4*adjustedWidth*width, hostC);

#else

    cl::Buffer bufferA(context, flags, 4*width*width, hostA);
    cl::Buffer bufferB(context, flags, 4*width*width, hostB);
    cl::Buffer bufferC(context, flags, 4*width*width, hostC);

#endif

    res = 0;
    int idx = 0;
//    uint32_t size = static_cast<uint32_t>(width);
    res |= kernel.setArg(idx++, width);
    res |= kernel.setArg(idx++, width);
    res |= kernel.setArg(idx++, width);
    res |= kernel.setArg(idx++, bufferA);
    res |= kernel.setArg(idx++, bufferB);//openCLBuffer(tensorB));
    res |= kernel.setArg(idx++, bufferC);//openCLBuffer(tensorC));
#if KERNEL == 2
#define TS TS2
#elif KERNEL==3
#define TS TS3
#elif KERNEL==4
#define TS TS4
#endif

#if KERNEL == 2 || KERNEL == 3 || KERNEL == 4

    int numTiles = width / TS;
    if (width % TS != 0)
        numTiles++;
    res |= kernel.setArg(idx++, numTiles);

#endif
//#if KERNEL == 4
//
//    res |= kernel.setArg(idx++, adjustedWidth);
//    res |= kernel.setArg(idx++, widthTiles);
//
//#endif

    MNN_CHECK_CL_SUCCESS(res, "setArg");

    uint32_t width_uint = (uint32_t) width;
    cl::NDRange offsetRange(0, 0);
    std::vector<uint32_t> offset({0, 0});


#if (KERNEL==1)
    std::vector<uint32_t> global({width_uint, width_uint}), local({32, 32});
    cl::NDRange globalRange(width_uint, width_uint), localRange(32, 32);
#elif (KERNEL==2)
    std::vector<uint32_t> global({width_uint, width_uint}), local({32, 32});
    cl::NDRange globalRange(width_uint, width_uint), localRange(32, 32);
//    MNN_PRINT("group_sizes")
#elif (KERNEL==3)

    MNN_PRINT("TS: %d, WPT: %d, RTS: %d", TS3, WPT, RTS);
    uint32_t globalColSize = width_uint / WPT;
    if (width % WPT != 0) globalColSize++;
    std::vector<uint32_t> global({width_uint, globalColSize}), local({TS3, RTS});
    cl::NDRange globalRange(width_uint, (width_uint / (WPT))), localRange(TS3, RTS);

#elif KERNEL == 4

//    uint32_t width = (uint32_t) width;
//    uint32_t WidthTiles = (uint32_t) widthTiles, Width = (uint32_t) width;
    uint32_t global0 = (uint32_t) widthTiles,
             global1 = (uint32_t) adjustedWidth,
             local1 = TS,
             local0 = TS/WIDTH;

    MNN_PRINT("global0=%d, global1=%d, local0=%d, local1=%d", global0, global1, local0, local1);

    std::vector<uint32_t> global({global0, global1}), local({local0, local1});
    cl::NDRange globalRange(global0, global1), localRange(local0, local1);

#endif

//#define PROFILING

#ifndef PROFILING
    cl::Event event1;
    cl::Event copy = event1;
    runKernel2D(kernel, global, local, runtime, &event1);
//    res = commandQueue.enqueueNDRangeKernel(kernel, offsetRange, globalRange, localRange,
//                                      nullptr, &event1);
//    commandQueue.finish();
    MNN_CHECK_CL_SUCCESS(res, "enqueueNDRangeKernel");
    if (res != 0) return -1.f;
    MNN_PRINT("event==copy: %d", &event1==&copy);

    int read_size = 4*width*width;
#if KERNEL == 4
    read_size = 4*width*adjustedWidth;
#endif
    commandQueue.enqueueReadBuffer(bufferC, CL_TRUE, 0, read_size, hostC, nullptr, nullptr);
    commandQueue.finish();

    MNN_PRINT("time: %f", runtime->getCostTime(&event1));
//
int width_ = width;
#if KERNEL==4
    width_ = adjustedWidth;
#endif
//    for (int i = 0; i < adjustedWidth * width; i++) {
//        MNN_PRINT(, i)
//    }
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j+=4) {
//            MNN_PRINT("(%2d, %2d):\t%f, %f, %f, %f", i, j,
//                      hostC[i][j], hostC[i][j+1],
//                      hostC[i][j+2], hostC[i][j+3]);
//            MNN_PRINT("%f, %f, %f, %f, %f, %f, %f, %f",
//                      hostC[i*width_ + j], hostC[i*width_ + j+1],
//                      hostC[i*width_ + j+2], hostC[i*width_ + j+3],
//                      hostC[i*width_ + j+4], hostC[i*width_ + j+5],
//                      hostC[i*width_ + j+6], hostC[i*width_ + j+7]);
//            if (i == j)
//                MNN_PRINT("hostC[%2d][%2d]||{(%5d), (%5d)} =\t%f",
//                          i, j, i*width + j, i*width_ + j,
//                          hostC[i][j]);//, hostA[i][j] == (float) i*width+j);


//            if (i-1 == j)
//                MNN_PRINT("hostC[%2d][%2d]: %5d %f", i, j, i*width + j, hostC[i][j]);
//            if (i == j-1)
//                MNN_PRINT("hostC[%2d][%2d]: %5d %f", i, j, i*width + j, hostC[i][j]);
        }
    }

    return 0.0f;
#else
    double avg_time = 0.0f;

    int warmup_step = 10, hot_runs = 50, last_runs = 5, overall_runs = 1;
    int total_runs = warmup_step + hot_runs + last_runs;
    std::vector<cl::Event> events;


    for (int k = 0; k < overall_runs; k++) {

//        for (int i = 0; i < total_runs; i++) {
//            if (i>=hot_runs && i < warmup_step + hot_runs) {
//                cl::Event event;
//                commandQueue.enqueueNDRangeKernel(kernel, offsetRange, globalRange, localRange,
//                                                  nullptr, &event);
//                events.push_back(event);
//            } else {
//                commandQueue.enqueueNDRangeKernel(kernel, offsetRange, globalRange, localRange,
//                                                  nullptr, nullptr);
//            }
//        }
//
//        commandQueue.finish();
        // warm up steps
        for (int i = 0; i < warmup_step; i++) {
            commandQueue.enqueueNDRangeKernel(kernel, offsetRange, globalRange, localRange,
                                              nullptr, nullptr);
        }

        // hot runs
        for (int i = 0; i < hot_runs; i++) {
            cl::Event event;
            res = commandQueue.enqueueNDRangeKernel(kernel, offsetRange, globalRange,
                                                    localRange, nullptr, &event);
            MNN_CHECK_CL_SUCCESS(res, "enqueueNDRangeKernel");
            events.push_back(event);
        }

        // cool down runs
        for (int i = 0; i < last_runs; i++) {
            commandQueue.enqueueNDRangeKernel(kernel, offsetRange, globalRange, localRange,
                                              nullptr, nullptr);
        }
        commandQueue.finish();
    }

    for (int i = 0; i < hot_runs*overall_runs; i++) {
//        if (&events[i]!=nullptr)
        avg_time += runtime->getCostTime(&(events[i]));
    }
    events.clear();

    avg_time /= hot_runs*overall_runs;
//    MNN_PRINT("Avg_time: %f", avg_time);

//    MNN_PRINT("done: %d", width);

//    delete bufferA, bufferB, bufferC
    return avg_time;
#endif
}


int starting = 224;

extern "C" JNIEXPORT jstring JNICALL
Java_com_ad945_mnn_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from C++";

#ifndef PROFILING

    MNN_PRINT("starting");
    gpu(33);

#else

//    MNN_PRINT("Starting CPU");
//    cpu();
//    MNN_PRINT("STarting GPU");
//    gpu();
    std::stringstream output;
//    output << "{";
//    for (int pow = 5; pow <= 9; pow++) {
//        double time = 0.0f;
//        int i = 1 << pow;
    int start = 15 , offset = 400;
    for (int i = start; i<=start+offset; i++) {
        double time = gpu(i);
        MNN_PRINT("%d-> %f", i, time);
        output << std::to_string(time) << ",";
//        if ((i - start) % 10 == 9) MNN_PRINT("%s", output.str().c_str());
    }
//    output << "}";

    MNN_PRINT("%s", output.str().c_str());
#endif

    MNN_PRINT("DONE!");










    return env->NewStringUTF(hello.c_str());
}