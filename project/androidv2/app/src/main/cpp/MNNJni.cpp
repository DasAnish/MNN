#define FCLAYER
//#define KERNEL_TRICKS

#ifdef FCLAYER
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
#include "backend/cpu/CPURuntime.hpp"
#include "MNN/AutoTime.hpp"

//NN resources:
#include "MNN/expr/Module.hpp"
#include "train/source/nn/NN.hpp"
#include "train/source/demo/MnistUtils.hpp"
#include "train/source/nn/RandomGenerator.hpp"
#include "train/source/models/Lenet.hpp"
#include "train/source/demo/MnistUtils.hpp"
#include "train/source/demo/mnistTrain.cpp"


using namespace MNN;
using namespace MNN::Express;
using namespace MNN::OpenCL;


static void printVar(VARP x) {
    auto size = x->getInfo()->size;
    auto ptr  = x->readMap<int32_t>();
    for (int i = 0; i < size; ++i) {
        MNN_PRINT("%d, ", ptr[i]);
    }
    MNN_PRINT("\n");
}

static void printVarFloat(VARP x) {
    auto size = x->getInfo()->size;
    auto ptr = x->readMap<float>();
    for (int i = 0; i < size; ++i) {
        MNN_PRINT("%f, ", ptr[i]);
    }
    MNN_PRINT("\n");
}

class Linear {

};

class SimpleFC : public Module  {
public:
    std::shared_ptr<Module> fc1; // 784 -> 16
    std::shared_ptr<Module> fc2; // 16 -> 16
    std::shared_ptr<Module> fc3; // 16 -> 10

public:

    SimpleFC(int size) {

//        int size = 128;
        fc1.reset(NN::Linear(size, size));
        fc2.reset(NN::Linear(size, size));
        fc3.reset(NN::Linear(size, size));

        registerModel({fc1, fc2, fc3});
    }

    std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) {
        VARP x = inputs[0];
//        MNN_PRINT("HERE-simpleFC");
//        x = _Reshape(x, {0, -1});
//        x = _Convert(x, NCHW);
//        MNN_PRINT("x.shape: (%d, %d, %d, %d)",
//                  x->getInfo()->dim[0], x->getInfo()->dim[1],
//                  x->getInfo()->dim[2], x->getInfo()->dim[3]);
        x = fc1->forward(x);
        x = _Relu(x);
        x = fc2->forward(x);
        x = _Relu(x);
        x = fc3->forward(x);
        x = _Softmax(x, 1);

        return {x};
    }

};

#define RUN_ON_CPU

void nn_run() {
    MNN_PRINT("STARTING");

    std::shared_ptr<Executor> executor = Executor::getGlobalExecutor();
    std::string root = "/data/local/tmp/mnist";
    RandomGenerator::generator(17);

//    for (int size = 4; size < 10; size++) {
//        int Size = 1 << size;
//        std::shared_ptr<Module> model1(new SimpleFC(Size));
//        MnistUtils::train(model1, root, MNN_FORWARD_CPU, Size, Size);
//        std::shared_ptr<Module> model2(new SimpleFC(Size));
//        MnistUtils::train(model2, root, MNN_FORWARD_OPENCL, Size, Size);
//    }
    int size = 16;
    auto fc = new SimpleFC(size);
    std::shared_ptr<Module> model1(fc);
    MnistUtils::train(model1, root, MNN_FORWARD_CPU);

    std::shared_ptr<Module> model2(new SimpleFC(size));
    MnistUtils::train(model2, root, MNN_FORWARD_OPENCL);
//#ifdef RUN_ON_CPU
//    MnistUtils::train(model, root, MNN_FORWARD_CPU);
//#else
//    MnistUtils::train(model, root, MNN_FORWARD_OPENCL);
//#endif


    MNN_PRINT("DONE");

}

extern "C" JNIEXPORT jstring JNICALL
Java_com_ad945_mnn_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from C++";

    MNN_PRINT("Here");
    nn_run();


    return env->NewStringUTF(hello.c_str());
}

#endif


#ifdef KERNEL_TRICKS

#include <jni.h>
#include <string>

//General
#include "MNN/Tensor.hpp"
#include "MNN_generated.h"
#include "train/source/nn/Initializer.hpp"

//opencl resources
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/runtime/OpenCLRuntime.hpp"
#include "MNN/expr/Executor.hpp"
#include "backend/opencl/execution/buffer/MatmulBufExecution.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"


//CPU resources:
#include "backend/cpu/CPUMatMul.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/CPURuntime.hpp"
#include "MNN/AutoTime.hpp"

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::OpenCL;


double cpu(int width = 16, int height=64, int common=784) {
    std::shared_ptr<Executor> executor = Executor::getGlobalExecutor();
    BackendConfig config;
    Backend::Info info;
    info.type = MNN_FORWARD_CPU;
    info.mode = Backend::Info::DIRECT;
    info.numThread = 1;
    info.user = (BackendConfig*)&config;

    CPURuntime runtime(info);
    CPUBackend backend(&runtime, config.precision);

//    int width = 8;

    float hostA[height][common];
    float hostB[common][width];
    float hostC[height][width];

    for (int i = 0; i<height; i++) {
        for (int j = 0; j < width; j++) {
//            hostA[i][j] = (float) (i*height + j);
//            hostB[i][j] = (float) (i*width + j);
            hostC[i][j] = 0.0f;
        }
    }

    for (int i = 0; i<height; i++) {
        for (int j = 0; j<common; j++) {
            hostA[i][j] = (float) (i*common +j) / 1000.f;
        }
    }

    for (int i = 0; i < common; i++) {
        for (int j = 0; j < width; j++) {
            hostB[i][j] = (float) (i*width + j) / 1000.f;
        }
    }

    std::vector<int> shape({width, width});
    Tensor* tensorA = Tensor::create<float>(shape, hostA);
    Tensor* tensorB = Tensor::create<float>(shape, hostB);
    Tensor* tensorC = Tensor::create<float>(shape, hostC);

    Timer timer;

    std::vector<Tensor *> inputs({tensorA, tensorB}), outputs({tensorC});

    CPUMatMul matmul(&backend, false, false, false, false);

//    MNN::MNNSetCPUThreadsMode()
    MNNSetCPUThreadsMode(MNN_CPU_MODE_BIG);
    matmul.onResize(inputs, outputs);
//    MNN_CHECK_CL_SUCCESS(code, "matmul.onResize");
//

    int warmup = 10, hot_run = 50, overall = 10;
    double avg_time = 0.f;
    for (int k = 0; k < overall; k++) {
        for (int i = 0; i < warmup; i++) {
            matmul.onExecute(inputs, outputs);
        }

        timer.reset();
        for (int i=0; i < hot_run; i++) {
            matmul.onExecute(inputs, outputs);
        }

        avg_time += (double) timer.durationInUs();
    }

    timer.reset();

//    tensorC->print();


    return avg_time / (hot_run * overall);
}


int starting = 0;
double gpu(int M = 32, int N = 32, int K=32, bool useTimer=false) {
#define KERNEL 4
#define TransA 0
#define TransB 0

#define PROFILING

//defines for gemm2
//#define TS2 32
    uint32_t TS2 = 32; // can't go up
//defines for gemm3
//#define TS3 32
    uint32_t TS3 = 32;
//#define WPT (TS3 * TS3 / 256)
    uint32_t WPT = 8;
//#define RTS TS3 / (WPT)
    uint32_t RTS = (TS3 / WPT);
//defines for gemm4
//#define TS4 32
    uint32_t TS4 = 16;
#ifdef WIDTH
#undef WIDTH
#endif
#define WIDTH 4

#if KERNEL == 5
    uint32_t TSM  = 32;                 // The tile-size in dimension M
uint32_t TSN  = TSM;                 // The tile-size in dimension N
uint32_t TSK  = 16;                 // The tile-size in dimension K
uint32_t WPTN = 4;                 // The work-per-thread in dimension N
uint32_t RTSN = (TSN/WPTN);        // The reduced tile-size in dimension N
uint32_t WPTM = 1;
uint32_t RTSM = (TSM/WPTM);
uint32_t LPT  = ((TSK*TSM)/(RTSM*RTSN)); // The loads-per-thread for a tile
#elif KERNEL == 6 || KERNEL == 7 || KERNEL == 8
    uint32_t TSM  = 32;                // The tile-size in dimension M
    uint32_t TSN  = TSM;                // The tile-size in dimension N
    uint32_t TSK  = 16;                 // The tile-size in dimension K
    uint32_t WPTM = 4;                // The work-per-thread in dimension M
    uint32_t WPTN = WPTM ;                // The work-per-thread in dimension N
//uint32_t RTSM = (TSM/WPTM)   ;     // The reduced tile-size in dimension M
//uint32_t RTSN = (TSN/WPTN)  ;      // The reduced tile-size in dimension N
//uint32_t LPTA = ((TSK*TSM)/(RTSM*RTSN)); // Loads-per-thread for A
//uint32_t LPTB = ((TSK*TSN)/(RTSM*RTSN)); // Loads-per-thread for B
#endif

    int res;
    BackendConfig config;
    std::shared_ptr<Executor> executor = Executor::getGlobalExecutor();
    executor->setGlobalExecutorConfig(MNN_FORWARD_OPENCL, config, 1);

//    BackendConfig config;


    Backend::Info info;
    info.type = MNN_FORWARD_OPENCL;
    info.mode = Backend::Info::DIRECT;
//    info.numThread = 1;
    info.gpuMode = MNN_GPU_TUNING_WIDE | MNN_GPU_MEMORY_BUFFER;
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



#if KERNEL == 2
    int adjustedM = M, adjustedN = N, adjustedK = K;

    int numTiles = K / TS2;
    if (K % TS2 != 0)
        numTiles++;
//    int adjustedK = numTiles * TS2;
//
//    numTiles = M / TS2;
//    if (M % TS2 != 0)
//        numTiles++;
//    int adjustedM = numTiles * TS2;
//
//    numTiles = N / TS2;
//    if (N % TS2 != 0)
//        numTiles++;
//    int adjustedN = numTiles * TS2;
#endif
#if KERNEL == 3
    int numTiles = width / TS3;
    if (width % TS3 != 0)
        numTiles++;
    int adjustedWidth = numTiles * TS3;
#endif
#if KERNEL == 4
//    int temp = K;
//    K = M;
//    N = temp;
//    M = temp;

    int numTiles = K / TS4;
    if (K % TS4 != 0)
        numTiles++;

//    int adjustedM = 32, adjustedN =32, adjustedK = 32;
    int adjustedK = K;//numTiles * TS4;
    int adjustedM = M;//UP_DIV(M, TS4)*TS4;
    int adjustedN = N;//UP_DIV(N, TS4)*TS4;

//    int t = K / WIDTH;
//    if (K % WIDTH != 0)
//        t ++;
//    adjustedK = t * WIDTH;
//
//
//    t = M / WIDTH;
//    if (M % WIDTH != 0)
//        t++;
//    adjustedM = t * WIDTH;
//
//    t = N / WIDTH;
//    if (N % WIDTH != 0)
//        t++;
//    adjustedN = t * WIDTH;

#endif
#if KERNEL == 6 || KERNEL == 5 || KERNEL == 7 || KERNEL == 8
    //    while (width < TSM/2) {
//        TSM /= 2;
//        TSN = TSM;
//    }

    if (TSM < TSK)
        TSK = TSM;

    int temp = width / TSM;
    if (width % TSM != 0)
        temp++;
    int adjustedWidth = temp * TSM;
    int numTiles = adjustedWidth / TSK;
#endif

#if KERNEL == 1
    float hostA[M][K];
    float hostB[K][N];
    float hostC[M][N];

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            hostA[i][j] = (float) (i*K + j) / 1000.f;
        }

        for (int j = 0; j < N; j++) {
            hostC[i][j] = 0.0f;
        }
    }

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            hostB[i][j] = (float) (i * N + j) / 1000.f;
        }
    }
#else

#if !TransA && !TransB
    float hostA[adjustedM][adjustedK];
    float hostB[adjustedK][adjustedN];
    float hostC[adjustedM][adjustedN];

    for (int i = 0; i < adjustedM; i++) {
        for (int j = 0; j < adjustedK; j++) {
            if (i < M && j < K) {
                hostA[i][j] = (float) (i*K + j) / 1000.f;
            }
            else {
                hostA[i][j] = 0.0f;
            }
        }
    }

    for (int i = 0; i < adjustedK; i++) {
        for (int j = 0; j < adjustedN; j++) {
            if (i < K && j < N) {
                hostB[i][j] = (float) (i*N + j) / 1000.f;
            }
            else {
                hostB[i][j] = 0.0f;
            }
        }
    }

    for (int i = 0; i < adjustedM; i++) {
        for (int j = 0; j < adjustedN; j++) {
            hostC[i][j] = 0.0f;
        }
    }

#endif

#if TransA && TransB
    float hostA[adjustedK][adjustedM];
    float hostB[adjustedN][adjustedK];
    float hostC[adjustedM][adjustedN];

    for (int i = 0; i < adjustedK; i++) {
        for (int j = 0; j < adjustedM; j++) {
            if (i < K && j < M)
                hostA[i][j] = (float) (j*K + i) / 1000.f;
            else
                hostA[i][j] = 0.0f;
        }
    }

    for (int i = 0; i < adjustedN; i++) {
        for (int j = 0; j < adjustedK; j++) {
            if (i < N && j < K)
                hostB[i][j] = (float) (j*N + i) / 1000.f;
            else
                hostB[i][j] = 0.0f;
        }
    }

    for (int i = 0; i < adjustedM; i++) {
        for (int j = 0; j < adjustedN; j++) {
            hostC[i][j] = 0.0f;
        }
    }

#endif

#if TransA && !TransB

    float hostA[adjustedK][adjustedM];
    float hostB[adjustedK][adjustedN];
    float hostC[adjustedM][adjustedN];

    for (int i = 0; i < adjustedK; i++) {
        for (int j = 0; j < adjustedM; j++) {
            if (i < K && j < M)
                hostA[i][j] = (float) (j*K + i) / 1000.f;
            else
                hostA[i][j] = 0.f;
        }
    }

    for (int i = 0; i < adjustedK; i++) {
        for (int j = 0; j < adjustedN; j++) {
            if (i < K && j < N)
                hostB[i][j] = (float) (i*N + j) / 1000.f;
            else
                hostB[i][j] = 0.f;
        }
    }

    for (int i = 0; i < adjustedM; i++) {
        for (int j = 0; j < adjustedN; j++) {
            hostC[i][j] = 0.f;
        }
    }

#endif

#if !TransA && TransB

    float hostA[adjustedM][adjustedK];
    float hostB[adjustedN][adjustedK];
    float hostC[adjustedM][adjustedN];

    for (int i = 0; i < adjustedM; i++) {
        for (int j = 0; j < adjustedK; j++) {
            if (i < M && j < K) {
                hostA[i][j] = (float) (i*K + j) / 1000.f;
            }
            else {
                hostA[i][j] = 0.0f;
            }
        }
    }

    for (int i = 0; i < adjustedN; i++) {
        for (int j = 0; j < adjustedK; j++) {
            if (i < N && j < K)
                hostB[i][j] = (float) (j*N + i) / 1000.f;
            else
                hostB[i][j] = 0.0f;
        }
    }

    for (int i = 0; i < adjustedM; i++) {
        for (int j = 0; j < adjustedN; j++) {
            hostC[i][j] = 0.f;
        }
    }

#endif

//    float hostA[adjustedWidth][adjustedWidth];
//    #if KERNEL == 5 || KERNEL == 6 || KERNEL == 7 || KERNEL == 8
//    float hostBtrans[adjustedWidth][adjustedWidth];
//    #endif
//    float hostB[adjustedWidth][adjustedWidth];
//    float hostC[adjustedWidth][adjustedWidth];
//
//    for (int i = 0; i<adjustedWidth; i++) {
//        for (int j = 0; j < adjustedWidth; j++) {
//            if (i < width && j < width) {
//                hostA[i][j] = (float) (i * width + j) / 1000.f;
//#if KERNEL == 5 || KERNEL == 6 || KERNEL == 7 || KERNEL == 8
//                hostBtrans[i][j] = (float) (i*width + j)/1000.f;
//                hostB[j][i] = (float) (i*width + j) / 1000.f;
//#else
//                hostB[i][j] = (float) (i * width + j) / 1000.f;
//#endif
//
//                hostC[i][j] = 0.0f;
//            } else{
//                hostA[i][j] = 0.f;
//                hostB[i][j] = 0.f;
//                hostC[i][j] = 0.f;
//#if KERNEL == 5 || KERNEL == 6 || KERNEL == 7 || KERNEL == 8
//                hostBtrans[i][j] = 0.f;
//#endif
//            }
//        }
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
    std::stringstream option;

#ifdef PROFILING
    option << "-DPROFILING=1 ";
#endif

//    buildOptions.emplace("-DTS3=32");
//    buildOptions.
#if KERNEL == 1
    //#ifdef KERNEL1
    buildOptions.emplace("-DKERNEL=1");
    cl::Kernel kernel = runtime->buildKernel("myGEMM", "gemm1", buildOptions);
#elif KERNEL == 2
    buildOptions.emplace("-DTS2=32 -DKERNEL=2");
    cl::Kernel kernel = runtime->buildKernel("myGEMM", "gemm2", buildOptions);
#elif KERNEL == 3
    option << "-DTS3=" << std::to_string(TS3) << " -DWPT=" << std::to_string(WPT)
           << " -DRTS=" << std::to_string(RTS) << " -DKERNEL=3 ";
    buildOptions.emplace(option.str());

    cl::Kernel kernel = runtime->buildKernel("myGEMM", "gemm3", buildOptions);
#elif KERNEL == 4
    option << " -DTS4=" << std::to_string(TS4)
           << " -DWIDTH=" << std::to_string(WIDTH)
           << " -DKERNEL=4 ";
    buildOptions.emplace(option.str());

    std::string kernelName;
#if TransA && TransB
    kernelName = "gemm4_transA_transB";
#elif TransA
    kernelName = "gemm4_transA";
#elif TransB
    kernelName = "gemm4_transB";
#else
    kernelName = "gemm4";
#endif

    cl::Kernel kernel = runtime->buildKernel("myGEMM", kernelName, buildOptions);
#elif KERNEL == 5
    option << "-DTSM=" << std::to_string(TSM) << " ";
    option << "-DTSN=" << std::to_string(TSN) << " ";
    option << "-DTSK=" << std::to_string(TSK) << " ";
    option << "-DWPTN=" << std::to_string(WPTN) << " ";
//    option << "-DWPTM=" << std::to_string(WPTM) << " ";
    option << "-DKERNEL=5 ";
//    option << "-DRTSN=" << std::to_string(TSN/WPTN) << " ";
//    option << "-DLPT=" << std::to_string(TSK/RTSN);
    buildOptions.emplace(option.str());

    cl:: Kernel kernel = runtime->buildKernel("myGEMM", "gemm5", buildOptions);
#elif KERNEL == 6

    option << "-DTSM=" << std::to_string(TSM) << " ";
    option << "-DTSN=" << std::to_string(TSM) << " ";
    option << "-DTSK=" << std::to_string(TSK) << " ";
    option << "-DWPTN=" << std::to_string(WPTN) << " ";
//    option << "-DRTSN=" << std::to_string(tsm / WPTN) << " ";
    option << "-DWPTM=" << std::to_string(WPTM) << " ";
//    option << "-DRTSM=" << std::to_string(tsm / WPTM) << " ";
    option << "-DKERNEL=6 ";

    buildOptions.emplace(option.str());

    cl::Kernel kernel = runtime->buildKernel("myGEMM", "gemm6", buildOptions);

#elif KERNEL == 7 || KERNEL == 8

    option << "-DTSM=" << std::to_string(TSM) << " ";
    option << "-DTSN=" << std::to_string(TSM) << " ";
    option << "-DTSK=" << std::to_string(TSK) << " ";
    option << "-DWPTN=" << std::to_string(WPTN) << " ";
//    option << "-DRTSN=" << std::to_string(tsm / WPTN) << " ";
    option << "-DWPTM=" << std::to_string(WPTM) << " ";
//    option << "-DRTSM=" << std::to_string(tsm / WPTM) << " ";
    option << "-DWIDTH=" << std::to_string(WIDTH) << " ";
    option << "-DKERNEL=" << std::to_string(KERNEL) << " ";

    buildOptions.emplace(option.str());

#if KERNEL == 7
    cl::Kernel kernel = runtime->buildKernel("myGEMM", "gemm7", buildOptions);
#else
    cl::Kernel kernel = runtime->buildKernel("myGEMM", "gemm8", buildOptions);
#endif
#endif

#ifndef PROFILING
    MNN_PRINT("%s", option.str().c_str());
#endif

#if KERNEL == 5 || KERNEL == 6 || KERNEL == 7 || KERNEL == 8
    option.clear();
    buildOptions.clear();

    option << "-DTRANSPOSEX=32 -DTRANSPOSEY=32";
    buildOptions.emplace(option.str());
    cl::Kernel transposeKernel = runtime->buildKernel("myGEMM", "transpose", buildOptions);
#endif


    res = 0;
    cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR;
    res = 0;
    int idx = 0;
//    MNN_PRINT("max_work_group_size: %lu", runtime->getMaxWorkGroupSize(kernel));


#if KERNEL == 1

    cl::Buffer bufferA(context, flags, 4*M*K, hostA);
    cl::Buffer bufferB(context, flags, 4*N*K, hostB);
    cl::Buffer bufferC(context, flags, 4*M*N, hostC);

    res |= kernel.setArg(idx++, M); // M
    res |= kernel.setArg(idx++, N); // N
    res |= kernel.setArg(idx++, K); // K



#else

    cl::Buffer bufferA(context, flags, 4*adjustedM*adjustedK, hostA);
    cl::Buffer bufferB(context, flags, 4*adjustedK*adjustedN, hostB);
    cl::Buffer bufferC(context, flags, 4*adjustedM*adjustedN, hostC);

    res |= kernel.setArg(idx++, adjustedM); // M
    res |= kernel.setArg(idx++, adjustedN); // N
    res |= kernel.setArg(idx++, adjustedK); // K

#endif

#if KERNEL == 5 || KERNEL == 6 || KERNEL == 7 || KERNEL == 8
    cl::Buffer bufferBtrans(context, flags, 4*adjustedWidth*adjustedWidth, hostBtrans);
#endif

    res |= kernel.setArg(idx++, bufferA);
    res |= kernel.setArg(idx++, bufferB);
    res |= kernel.setArg(idx++, bufferC);
    res |= kernel.setArg(idx++, numTiles);

//#if KERNEL == 2
//#define TS TS2
//#elif KERNEL==3
//#define TS TS3
//#elif KERNEL==4
//#define TS TS4
//#elif KERNEL==5
//#define TS TSM
//#elif KERNEL == 6
//#define TS tsm
//#endif

//#if KERNEL != 1
//
//    res |= kernel.setArg(idx++, numTiles);
//
//#endif

    MNN_CHECK_CL_SUCCESS(res, "setArg");

//#if KERNEL == 5 || KERNEL == 6 || KERNEL == 7 || KERNEL == 8
//    idx = 0;
//    res = 0;
//    res |= transposeKernel.setArg(idx++, adjustedWidth);
//    res |= transposeKernel.setArg(idx++, adjustedWidth);
//    res |= transposeKernel.setArg(idx++, bufferB);
//    res |= transposeKernel.setArg(idx++, bufferBtrans);
//    MNN_CHECK_CL_SUCCESS(res, "transposeKernel.setArg");
//
//    uint32_t tGlobal0 = adjustedWidth,
//             tGlobal1 = adjustedWidth,
//             tLocal0 = 32,
//             tLocal1 = 32;
//    cl::NDRange tGlobalRange(tGlobal0, tGlobal1), tLocalRange(tLocal0, tLocal1);
//    std::vector<uint32_t> tGlobal({tGlobal0, tGlobal1}), tLocal({tLocal0, tLocal1});
//
////    runKernel2D(transposeKernel, tGlobal, tLocal, runtime, nullptr);
////    commandQueue.finish();
//#endif

//    uint32_t width_uint = (uint32_t) width;
    cl::NDRange offsetRange(0, 0);
    std::vector<uint32_t> offset({0, 0});


#if (KERNEL==1)
    std::vector<uint32_t> global({(unsigned int) M, (unsigned int)N}), local({32, 32});
    cl::NDRange globalRange(M, N), localRange(32, 32);
#elif (KERNEL==2)
    std::vector<uint32_t> global({(unsigned int) adjustedM, (unsigned int) adjustedN}), local({TS2, TS2});
    cl::NDRange globalRange(adjustedM, adjustedN), localRange(TS2, TS2);
//    MNN_PRINT("group_sizes")
#elif (KERNEL==3)

    //    MNN_PRINT("TS: %d, WPT: %d, RTS: %d", TS3, WPT, RTS);
    uint32_t globalColSize = width_uint / WPT;
    if (width % WPT != 0) globalColSize++;
    std::vector<uint32_t> global({width_uint, globalColSize}), local({TS3, RTS});
//    MNN_PRINT("Max local size: %lu asked: %du", runtime->getMaxWorkGroupSize(kernel), globalColSize);
    cl::NDRange globalRange(width_uint, (width_uint / (WPT))), localRange(TS3, RTS);

#elif KERNEL == 4

    //    uint32_t width = (uint32_t) width;
//    uint32_t WidthTiles = (uint32_t) widthTiles, Width = (uint32_t) width;
//    uint32_t global0 = (uint32_t) adjustedM / WIDTH,
//             global1 = (uint32_t) adjustedN,
//             local0 = TS4 / WIDTH,
//             local1 = TS4;


#if TransA && TransB
    uint32_t global0 = adjustedM / WIDTH,
             global1 = adjustedN,
             local0 = TS4 / WIDTH,
             local1 = TS4;
#else
    uint32_t global0 = adjustedM,
            global1 = UP_DIV(adjustedN, WIDTH),
            local0 = TS4,
            local1 = UP_DIV(TS4, WIDTH);
#endif
//    MNN_PRINT("global0=%d, global1=%d, local0=%d, local1=%d", global0, global1, local0, local1);

    std::vector<uint32_t> global({global0, global1}), local({local0, local1});
    cl::NDRange globalRange(global0, global1), localRange(local0, local1);
#elif KERNEL == 5
    uint32_t global0 = width,
             global1 = width / WPTN,
             local0 = TSM,
             local1 = TSN / WPTN;
    if (width % WPTN != 0)
        global1++;

//    MNN_PRINT("global0=%d, global1=%d, local0=%d, local1=%d", global0, global1, local0, local1);

    std::vector<uint32_t> global({global0, global1}), local({local0, local1});
    cl::NDRange globalRange(global0, global1), localRange(local0, local1);
#elif KERNEL == 6 || KERNEL == 7 || KERNEL == 8
    uint32_t global0 = adjustedWidth / WPTM,
            global1 = adjustedWidth / WPTN,
            local0  = TSM / WPTM,
            local1  = TSN / WPTN;

//    if (width % WPTM != 0)
//        global0++;
//    if (width % WPTN != 0)
//        global1++;

//    MNN_PRINT("RTSM:%d | WPTM:%d | RTSN:%d | WPTN:%d", RTSM, WPTM, RTSN, WPTN);
//    MNN_PRINT("global0=%d, global1=%d, local0=%d, local1=%d", global0, global1, local0, local1);

    std::vector<uint32_t> global({global0, global1}), local({local0, local1});
#endif



#ifndef PROFILING
    cl::Event event1;
    cl::Event copy = event1;

#if KERNEL == 5 || KERNEL == 6 || KERNEL == 7 || KERNEL == 8
//    runKernel2D(transposeKernel, tGlobal, tLocal, runtime, nullptr);
//    commandQueue.finish();
#endif

    runKernel2D(kernel, global, local, runtime, &event1);
//    res = commandQueue.enqueueNDRangeKernel(kernel, offsetRange, globalRange, localRange,
//                                      nullptr, &event1);
//    commandQueue.finish();
    MNN_CHECK_CL_SUCCESS(res, "enqueueNDRangeKernel");
    if (res != 0) return -1.f;
    MNN_PRINT("event==copy: %d", &event1==&copy);

#if KERNEL == 1
    int read_size = M * N;
#else
    int read_size = adjustedM * adjustedN;
    MNN_PRINT("read_size: %d", read_size);
#endif
    read_size *= 4;
    commandQueue.enqueueReadBuffer(bufferC, CL_TRUE, 0, read_size, hostC, nullptr, nullptr);
    commandQueue.finish();

/////////////////////////////////////////////////////////////////////////////////////////////////////
//    MNN_PRINT("PRINTING A (%d, %d)", M, K);
//    for (int i = 0; i < 3; i++) {
//        for (int j = 0; j < 4; j+=4) {
//            MNN_PRINT("(%2d, %2d):\t%f, %f, %f, %f", i, j,
//                      hostA[i][j], hostA[i][j+1],
//                      hostA[i][j+2], hostA[i][j+3]);
//        }
////        MNN_PRINT(" ");
//    }
//
//    MNN_PRINT(" ");
//
//    for (int i = 0; i < 3; i++) {
//        for (int j = K-4; j < K; j+=4) {
//            MNN_PRINT("(%2d, %2d):\t%f, %f, %f, %f", i, j,
//                      hostA[i][j], hostA[i][j+1],
//                      hostA[i][j+2], hostA[i][j+3]);
//        }
////        MNN_PRINT(" ");
//    }
//
//    MNN_PRINT(" ");
//
//    for (int i = M-3; i < M; i++) {
//        for (int j = 0; j < 4; j+=4) {
//            MNN_PRINT("(%2d, %2d):\t%f, %f, %f, %f", i, j,
//                      hostA[i][j], hostA[i][j+1],
//                      hostA[i][j+2], hostA[i][j+3]);
//        }
////        MNN_PRINT(" ");
//    }
//
//    MNN_PRINT(" ");
//
//    for (int i = M-3; i < M; i++) {
//        for (int j = K-4; j < K; j+=4) {
//            MNN_PRINT("(%2d, %2d):\t%f, %f, %f, %f", i, j,
//                      hostA[i][j], hostA[i][j+1],
//                      hostA[i][j+2], hostA[i][j+3]);
//        }
////        MNN_PRINT(" ");
//    }
/////////////////////////////////////////////////////////////////////////////////////////////////
//
//    MNN_PRINT("PRINTING B (%d, %d)", K, N);
//    for (int i = 0; i < 3; i++) {
//        for (int j = 0; j < 4; j+=4) {
//            MNN_PRINT("(%2d, %2d):\t%f, %f, %f, %f", i, j,
//                      hostB[i][j], hostB[i][j+1],
//                      hostB[i][j+2], hostB[i][j+3]);
//        }
//        MNN_PRINT(" ");
//    }
//
//    MNN_PRINT(" ");
//
//    for (int i = 0; i < 3; i++) {
//        for (int j = N-4; j < N; j+=4) {
//            MNN_PRINT("(%2d, %2d):\t%f, %f, %f, %f", i, j,
//                      hostB[i][j], hostB[i][j+1],
//                      hostB[i][j+2], hostB[i][j+3]);
//        }
////        MNN_PRINT(" ");
//    }
//
//    MNN_PRINT(" ");
//
//    for (int i = K-3; i < K; i++) {
//        for (int j = 0; j < 4; j+=4) {
//            MNN_PRINT("(%2d, %2d):\t%f, %f, %f, %f", i, j,
//                      hostB[i][j], hostB[i][j+1],
//                      hostB[i][j+2], hostB[i][j+3]);
//        }
////        MNN_PRINT(" ");
//    }
//
//    MNN_PRINT(" ");
//
//    for (int i = K-3; i < K; i++) {
//        for (int j = N-4; j < N; j+=4) {
//            MNN_PRINT("(%2d, %2d):\t%f, %f, %f, %f", i, j,
//                      hostB[i][j], hostB[i][j+1],
//                      hostB[i][j+2], hostB[i][j+3]);
//        }
////        MNN_PRINT(" ");
//    }


////////////////////////////////////////////////////////////////////////////////////////////////
    MNN_PRINT("PRINTING C (%d, %d)", M, N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j+=4) {
            MNN_PRINT("(%2d, %2d):\t%f, %f, %f, %f", i, j,
//                        hostC[j][i], hostC[j+1][i],
//                        hostC[j+2][i], hostC[i]);
                      hostC[i][j], hostC[i][j+1],
                      hostC[i][j+2], hostC[i][j+3]);
        }
        MNN_PRINT(" ");
    }

//    MNN_PRINT(" ");
//
//    for (int i = 0; i < 3; i++) {
//        for (int j = N-4; j < N; j+=4) {
//            MNN_PRINT("(%2d, %2d):\t%f, %f, %f, %f", i, j,
//                      hostC[i][j], hostC[i][j+1],
//                      hostC[i][j+2], hostC[i][j+3]);
//        }
////        MNN_PRINT(" ");
//    }
//
//    MNN_PRINT(" ");
//
//    for (int i = K-3; i < K; i++) {
//        for (int j = 0; j < 4; j+=4) {
//            MNN_PRINT("(%2d, %2d):\t%f, %f, %f, %f", i, j,
//                      hostC[i][j], hostC[i][j+1],
//                      hostC[i][j+2], hostC[i][j+3]);
//        }
////        MNN_PRINT(" ");
//    }
//
//    MNN_PRINT(" ");
//
//    for (int i = K-3; i < K; i++) {
//        for (int j = N-4; j < N; j+=4) {
//            MNN_PRINT("(%2d, %2d):\t%f, %f, %f, %f", i, j,
//                      hostC[i][j], hostC[i][j+1],
//                      hostC[i][j+2], hostC[i][j+3]);
//        }
////        MNN_PRINT(" ");
//    }



    MNN_PRINT("time: %f", runtime->getCostTime(&event1));
    return 0.0f;
#else
    double avg_time = 0.0f;
    double avg = 0.0f;
    Timer timer;

    int warmup_step = 10, hot_runs = 50, last_runs = 5, overall = 10;
    int total_runs = warmup_step + hot_runs + last_runs;
    std::vector<cl::Event> events;


    for (int k = 0; k < overall; k++) {

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
//            commandQueue.enqueueNDRangeKernel(kernel, offsetRange, globalRange, localRange,
//                                              nullptr, nullptr);

            runKernel2D(kernel, global, local, runtime, nullptr);
        }

        // hot runs
        timer.reset();
        for (int i = 0; i < hot_runs; i++) {
            cl::Event event;
//            res = commandQueue.enqueueNDRangeKernel(kernel, offsetRange, globalRange,
//                                                    localRange, nullptr, &event);
//            MNN_CHECK_CL_SUCCESS(res, "enqueueNDRangeKernel");
//            timer.reset();

            if (useTimer) {
                idx = 0;
                res |= kernel.setArg(idx++, adjustedM); // M
                res |= kernel.setArg(idx++, adjustedN); // N
                res |= kernel.setArg(idx++, adjustedK); // K

                res |= kernel.setArg(idx++, bufferA);
                res |= kernel.setArg(idx++, bufferB);
                res |= kernel.setArg(idx++, bufferC);
                res |= kernel.setArg(idx++, numTiles);
            }


            runKernel2D(kernel, global, local, runtime, &event);
//            avg_time += timer.durationInUs();
            events.push_back(event);

        }
        commandQueue.finish();
        avg += timer.durationInUs();

        // cool down runs
        for (int i = 0; i < last_runs; i++) {
//            commandQueue.enqueueNDRangeKernel(kernel, offsetRange, globalRange, localRange,
//                                              nullptr, nullptr);
            runKernel2D(kernel, global, local, runtime, nullptr);
        }
        commandQueue.finish();
    }

    for (int i = 0; i < hot_runs*overall; i++) {
//        if (&events[i]!=nullptr)
        avg_time += runtime->getCostTime(&(events[i]));
    }
    events.clear();
//    MNN_PRINT("avg: %f", avg / (hot_runs * overall));

    avg_time /= hot_runs*overall;
    avg /= hot_runs * overall;

    if (!useTimer) {
        return avg_time;
    } else {
        return avg;
    }
#endif
}



extern "C" JNIEXPORT jstring JNICALL
Java_com_ad945_mnn_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from C++";

#ifndef PROFILING

    MNN_PRINT("starting");
    gpu(64, 16, 10);

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
//    MNN_PRINT("gpu-time: %f", gpu(64, 784, 16));
//    MNN_PRINT("cpu-time: %f", cpu());
    int start = 1 , offset = 31;
    for (int i = start; i<=16; i++) {
        int mat_size = i*16;
//        double time = gpu(mat_size, mat_size, mat_size);
//        double tcpu = cpu(mat_size, mat_size, mat_size);
//        double tgpu = gpu(mat_size, mat_size, mat_size);
//        double tgpu_timer = gpu(mat_size, mat_size, mat_size, true);

        MNN_PRINT("%d -> CPU: %f\t GPU: %f\t GPU-timer: %f", mat_size,
                  cpu(mat_size, mat_size, mat_size),
                  gpu(mat_size, mat_size, mat_size),
                  gpu(mat_size, mat_size, mat_size, true));

//        output << std::to_string(time) << ",";
//        if ((i - start) % 10 == 9) MNN_PRINT("%s", output.str().c_str());
    }
    output << "}";

    MNN_PRINT("%s", output.str().c_str());
#endif

    MNN_PRINT("DONE!");










    return env->NewStringUTF(hello.c_str());
}
#endif