//
//  MnistUtils.cpp
//  MNN
//
//  Created by MNN on 2020/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MnistUtils.hpp"
#include <MNN/expr/Executor.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include "DataLoader.hpp"
#include "DemoUnit.hpp"
#include "MnistDataset.hpp"
#include "NN.hpp"
#include "SGD.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "ADAM.hpp"
#include "LearningRateScheduler.hpp"
#include "Loss.hpp"
#include "RandomGenerator.hpp"
#include "Transformer.hpp"
#include "OpGrad.hpp"

//custom added
#include "Initializer.hpp"
#include "MNN/Tensor.hpp"
#include "backend/opencl/execution/buffer/MatmulBufExecution.hpp"
#include "backend/cpu/CPUMatMul.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/CPURuntime.hpp"

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Train;

//#define RUN_ON_CPU

void MnistUtils::train(std::shared_ptr<Module> model, std::string root, MNNForwardType forwardType) {
//    {
//        // Load snapshot
//        auto para = Variable::load("mnist.snapshot.mnn");
//        model->loadParameters(para);
//    }

    std::shared_ptr<Module> _model;
    _model.reset(NN::Linear(28*28, 16));

    auto exe = Executor::getGlobalExecutor();
    BackendConfig config;
    config.power = BackendConfig::Power_High;
    int numberThreads = 1;
    if (forwardType == MNN_FORWARD_OPENCL)
        numberThreads = MNN_GPU_TUNING_WIDE | MNN_GPU_MEMORY_BUFFER;
    exe->setGlobalExecutorConfig(forwardType, config, numberThreads);
    std::shared_ptr<SGD> sgd(new SGD(model));
    sgd->setMomentum(0.9f);
    // sgd->setMomentum2(0.99f);
    sgd->setWeightDecay(0.0005f);

    auto dataset = MnistDataset::create(root, MnistDataset::Mode::TRAIN);
    // the stack transform, stack [1, 28, 28] to [n, 1, 28, 28]
    const size_t batchSize  = 1 << 4;
    const size_t numWorkers = 0;
    bool shuffle            = false;

    auto dataLoader = std::shared_ptr<DataLoader>(dataset.createLoader(batchSize, true, shuffle, numWorkers));

    size_t iterations = dataLoader->iterNumber();

    auto testDataset            = MnistDataset::create(root, MnistDataset::Mode::TEST);
    const size_t testBatchSize  = 20;
    const size_t testNumWorkers = 0;
    shuffle                     = false;

    auto testDataLoader = std::shared_ptr<DataLoader>(testDataset.createLoader(testBatchSize, true, shuffle, testNumWorkers));

    size_t testIterations = testDataLoader->iterNumber();
    auto weightInit = Initializer::xavier();

//    auto trainData  = dataLoader->next();
//    auto example    = trainData[0];
//    auto cast       = _Cast<float>(example.first[0]);
//    cast = _Reshape(cast, {0, -1});
//    cast = _Convert(cast, NCHW);
//    example.first[0] = cast * _Const(1.0f / 255.0f);
//
//     Compute One-Hot
//                MNN_PRINT("HERE1");
//    auto newTarget = _OneHot(_Cast<int32_t>(example.second[0]),
//                             _Scalar<int>(16), _Scalar<float>(1.0f),
//                             _Scalar<float>(0.0f));
//
//    int t = 28*28, l = 16;
//    auto weightInit = Initializer::xavier();
//    auto _weight = weightInit->createConstVar({t, l}, NCHW);
////    MNN_PRINT(_weight->)
//    auto _fc1_weight = _weight->readMap<float>();
//    float* fc1_weight = (float*) _fc1_weight;
//    auto _fc2_weight = (weightInit->createConstVar({l, l}, NCHW))->readMap<float>();
//    float* fc2_weight = (float*) _fc2_weight;
//    auto _fc3_weight = (weightInit->createConstVar({l, l}, NCHW))->readMap<float>();
//    float* fc3_weight = (float*) _fc3_weight;
//
//    std::vector<int> shape1({t, 1, 1, l}), shape2({l, 1, 1, l});
//    Tensor *fc1_weight_tensor = Tensor::create<float>(shape1, fc1_weight);
//    Tensor *fc2_weight_tensor = Tensor::create<float>(shape2, fc2_weight);
//    Tensor *fc3_weight_tensor = Tensor::create<float>(shape2, fc3_weight);
//
//    float layer2[batchSize][16];
//    float layer3[batchSize][16];
//    float layer4[batchSize][16];
//
//    Tensor* layer1_tensor = Tensor::create<float>({64, 1, 1, 28*28});
//
//    std::vector<int> shapelayers({16, 1, 1, 16});
//    Tensor* layer2_tensor = Tensor::create<float>(shapelayers, layer2);
//    Tensor* layer3_tensor = Tensor::create<float>(shapelayers, layer3);
//    Tensor* layer4_tensor = Tensor::create<float>(shapelayers, layer4);
//
//    std::vector<Tensor*> fc1_output({layer2_tensor}), fc2_output({layer3_tensor}), fc3_output({layer4_tensor});
//    std::vector<Tensor*> fc2_inputs({layer2_tensor, fc2_weight_tensor}), fc3_inputs({layer3_tensor, fc3_weight_tensor});
//    std::vector<Tensor*> fc1_inputs({layer1_tensor, fc1_weight_tensor});
//
//
//
//#ifdef RUN_ON_CPU
//
////    BackendConfig config;
//    Backend::Info info;
//    info.type = MNN_FORWARD_CPU;
//    info.mode = Backend::Info::DIRECT;
//    info.numThread = 1;
//    info.user = (BackendConfig*)&config;
//
//    CPURuntime runtime(info);
//    CPUBackend backend(&runtime, config.precision);
//
//
//    CPUMatMul fc1_matmul(&backend, false, false, false, false);
//    CPUMatMul fc2_matmul(&backend, false, false, false, false);
//    CPUMatMul fc3_matmul(&backend, false, false, false, false);
//
//
//
//#else
//
//    config.power = BackendConfig::Power_High;
//    Backend::Info info;
//    info.type = MNN_FORWARD_OPENCL;
//    info.mode = Backend::Info::DIRECT;
//    info.gpuMode = MNN_GPU_TUNING_WIDE | MNN_GPU_MEMORY_BUFFER;
//    info.user = (BackendConfig*)&config;
//
//    OpenCL::CLRuntime cl_runtime(info);
//    OpenCL::OpenCLBackend backend(&cl_runtime);
//
//    MNN::OpenCL::MatMulBufExecution fc1_matmul(fc1_inputs, nullptr, &backend, false, false);
//    MNN::OpenCL::MatMulBufExecution fc2_matmul(fc2_inputs, nullptr, &backend, false, false);
//    MNN::OpenCL::MatMulBufExecution fc3_matmul(fc3_inputs, nullptr, &backend, false, false);
//#endif
//
//    MNN::Backend::StorageType dynamic = MNN::Backend::DYNAMIC;
//    backend.onAcquireBuffer(fc1_weight_tensor, dynamic);
//    backend.onAcquireBuffer(fc2_weight_tensor, dynamic);
//    backend.onAcquireBuffer(fc3_weight_tensor, dynamic);
//    backend.onAcquireBuffer(layer1_tensor, dynamic);
//    backend.onAcquireBuffer(layer2_tensor, dynamic);
//    backend.onAcquireBuffer(layer3_tensor, dynamic);
//    backend.onAcquireBuffer(layer4_tensor, dynamic);
//
//
//
//    fc1_matmul.onResize(fc1_inputs, fc1_output);
//    fc2_matmul.onResize(fc2_inputs, fc2_output);
//    fc3_matmul.onResize(fc3_inputs, fc3_output);


    for (int epoch = 0; epoch < 1; ++epoch) {
        model->clearCache();
        exe->gc(Executor::FULL);
        exe->resetProfile();
        {
            AUTOTIME;
            dataLoader->reset();
            model->setIsTraining(true);
            Timer _100Time;
            Timer _iterTimer;
            Timer reshapeTimer;
            auto meanTimeLost = 0.0f;
            int lastIndex = 0;
            int moveBatchSize = 0;
            auto meanForwardTime = 0.0f;
            auto meanBackwardTime = 0.0f;

            //custom timers
            auto forward1 = 0.0f, forward2=0.0f, forward3=0.0f;
            auto meanLossTime = 0.0f;
            auto meanDataLoadingTime = 0.0f;
            //-end

            for (int i = 0; i < 100; i++) {
                _iterTimer.reset();
//                MNN_PRINT("New Iteration %i\n", i);
                // AUTOTIME;
                auto trainData  = dataLoader->next();
                auto example    = trainData[0];
//                auto cast       = _Cast<float>(example.first[0]);
//                cast = _Reshape(cast, {0, -1});
//                cast = _Convert(cast, NCHW);
//                example.first[0] = cast * _Const(1.0f / 255.0f);

                int batch = example.first[0]->getInfo()->dim[0];
                int inputSize = 16;
                example.first[0] = weightInit->createConstVar({batch, inputSize});

//                auto input = example.first[0]->readMap<float>();
//                float* input_ptr = (float*) input;
//                void* host = layer1_tensor->map(Tensor::MAP_TENSOR_WRITE, Tensor::CAFFE);
//                layer1_tensor->unmap(Tensor::MAP_TENSOR_WRITE, Tensor::CAFFE, input_ptr);
//                backend.onReleaseBuffer(layer1_tensor, dynamic);
//                layer1_tensor = Tensor::create<float>({64, 1, 1, 784}, input_ptr);
//                backend.onAcquireBuffer(layer1_tensor, dynamic);
//                Tensor* input_tensor = Tensor::create<float>({64, 1, 1, 28*28}, input_ptr);
                float t = (float) _iterTimer.durationInUs();
                auto dataLoadingTime = (float) t / 1000.f;


//                model->forward(example.first[0]);
//                fc1_matmul.onExecute(fc1_inputs, fc1_output);
//                forward1 += (float) _iterTimer.durationInUs() - t;
//                t = (float) _iterTimer.durationInUs();
//                fc2_matmul.onExecute(fc2_inputs, fc2_output);
//                forward2 += (float) _iterTimer.durationInUs() - t;
//                t = (float) _iterTimer.durationInUs();
//                fc3_matmul.onExecute(fc3_inputs, fc3_output);
//                forward3 += (float) _iterTimer.durationInUs() - t;

                // Compute One-Hot
//                MNN_PRINT("HERE1");
                auto newTarget = _OneHot(_Cast<int32_t>(example.second[0]),
                        _Scalar<int>(inputSize),
                        _Scalar<float>(1.0f),
                                _Scalar<float>(0.0f));
                _iterTimer.reset();
                auto predict = model->onForward(example.first)[0];
                auto forwardTime = (float)_iterTimer.durationInUs() / 1000.0f;

                _iterTimer.reset();
                auto loss    = _CrossEntropy(predict, newTarget);
                auto lossTime = (float)_iterTimer.durationInUs() / 1000.f;

//                MNN_PRINT("predict: (%d, %d, %d, %d)",
//                          predict->getInfo()->dim[0],
//                          predict->getInfo()->dim[1],
//                          predict->getInfo()->dim[2],
//                          predict->getInfo()->dim[3]);

//                MNN_PRINT("HERE5");
//                MNN_PRINT("Forward Time %f", forwardTime);



//#define DEBUG_GRAD
#ifdef DEBUG_GRAD
                {
                    static bool init = false;
                    if (!init) {
                        init = true;
                        std::set<VARP> para;
                        example.first[0].fix(VARP::INPUT);
                        newTarget.fix(VARP::CONSTANT);
                        auto total = model->parameters();
                        for (auto p :total) {
                            para.insert(p);
                        }
                        auto grad = OpGrad::grad(loss, para);
                        total.clear();
                        for (auto iter : grad) {
                            total.emplace_back(iter.second);
                        }
                        Variable::save(total, ".temp.grad");
                    }
                }
#endif

                _iterTimer.reset();
                float rate   = LrScheduler::inv(0.01, epoch * iterations + i, 0.0001, 0.75);
                sgd->setLearningRate(rate);

                sgd->step(loss);
                auto backwardTime = (float)_iterTimer.durationInUs() / 1000.f;
                moveBatchSize += example.first[0]->getInfo()->dim[0];

                _iterTimer.reset();

                meanDataLoadingTime += dataLoadingTime / 10.0;
                meanForwardTime += (forwardTime)/10.0;
                meanLossTime += (lossTime) / 10.0;
                meanBackwardTime += (backwardTime) / 10.0;




                if (moveBatchSize % (10 * batchSize) == 0 || i == iterations - 1) {
#ifdef MNN_USE_LOGCAT
                    MNN_PRINT("LOSS: %f || dataLoad:%.2f | Forward:%.2f | LossCalc:%.2f | Backward:%.2f",
                              loss->readMap<float>()[0],
                              meanDataLoadingTime, meanForwardTime, meanLossTime, meanBackwardTime);
                    meanDataLoadingTime = 0.0f;
                    meanForwardTime = 0.0f;
                    meanLossTime = 0.0f;
//                    meanLossReadTime = 0.0f;
                    meanBackwardTime = 0.0f;
//                    MNN_PRINT("epoch: %i %i/%i\tloss: %f\tlr: %f\ttime: %f ms / %i iter",
//                              epoch, moveBatchSize, dataLoader->size(), loss->readMap<float>()[0], rate, (float)_100Time.durationInUs() / 1000.0f,
//                              (i - lastIndex));
//                    MNN_PRINT("Forward Time: %f ms\tBackward Time: %f ms\t", meanForwardTime, meanBackwardTime); // NOTE this is not correct if i == iterations - 1
                    _100Time.reset();
                    lastIndex = i;
                    meanForwardTime = 0;
                    meanBackwardTime = 0;
#else
                    std::cout << "epoch: " << (epoch);
                    std::cout << "  " << moveBatchSize << " / " << dataLoader->size();
                    std::cout << " loss: " << loss->readMap<float>()[0];
                    std::cout << " lr: " << rate;
                    std::cout << " time: " << (float)_100Time.durationInUs() / 1000.0f << " ms / " << (i - lastIndex) <<  " iter"  << std::endl;
                    std::cout.flush();
                    _100Time.reset();
                    lastIndex = i;
#endif
                }
            }
        }
//        Variable::save(model->parameters(), "mnist.snapshot.mnn");
//        {
//            model->setIsTraining(false);
//            auto forwardInput = _Input({1, 1, 28, 28}, NC4HW4);
//            forwardInput->setName("data");
//            auto predict = model->forward(forwardInput);
//            predict->setName("prob");
//            Transformer::turnModelToInfer()->onExecute({predict});
//            Variable::save({predict}, "temp.mnist.mnn");
//        }
//
//        int correct = 0;
//        testDataLoader->reset();
//        model->setIsTraining(false);
//        int moveBatchSize = 0;
//        for (int i = 0; i < testIterations; i++) {
//            auto data       = testDataLoader->next();
//            auto example    = data[0];
//            moveBatchSize += example.first[0]->getInfo()->dim[0];
//            if ((i + 1) % 100 == 0) {
//                std::cout << "test: " << moveBatchSize << " / " << testDataLoader->size() << std::endl;
//            }
//            auto cast       = _Cast<float>(example.first[0]);
//            example.first[0] = cast * _Const(1.0f / 255.0f);
//            auto predict    = model->forward(example.first[0]);
//            predict         = _ArgMax(predict, 1);
//            auto accu       = _Cast<int32_t>(_Equal(predict, _Cast<int32_t>(example.second[0]))).sum({});
//            correct += accu->readMap<int32_t>()[0];
//        }
//        auto accu = (float)correct / (float)testDataLoader->size();
//        std::cout << "epoch: " << epoch << "  accuracy: " << accu << std::endl;
//        exe->dumpProfile();
    }
}
