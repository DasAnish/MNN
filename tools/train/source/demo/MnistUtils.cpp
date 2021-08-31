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
using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Train;

void MnistUtils::train(std::shared_ptr<Module> model, std::string root, MNNForwardType forwardType) {
//    {
//        // Load snapshot
//        auto para = Variable::load("mnist.snapshot.mnn");
//        model->loadParameters(para);
//    }
    auto exe = Executor::getGlobalExecutor();
    BackendConfig config;
    exe->setGlobalExecutorConfig(forwardType, config, 1);
    std::shared_ptr<SGD> sgd(new SGD(model));
    sgd->setMomentum(0.9f);
    // sgd->setMomentum2(0.99f);
    sgd->setWeightDecay(0.0005f);

    auto dataset = MnistDataset::create(root, MnistDataset::Mode::TRAIN);
    // the stack transform, stack [1, 28, 28] to [n, 1, 28, 28]
    const size_t batchSize  = 64;
    const size_t numWorkers = 0;
    bool shuffle            = true;

    auto dataLoader = std::shared_ptr<DataLoader>(dataset.createLoader(batchSize, true, shuffle, numWorkers));

    size_t iterations = dataLoader->iterNumber();

    auto testDataset            = MnistDataset::create(root, MnistDataset::Mode::TEST);
    const size_t testBatchSize  = 20;
    const size_t testNumWorkers = 0;
    shuffle                     = false;

    auto testDataLoader = std::shared_ptr<DataLoader>(testDataset.createLoader(testBatchSize, true, shuffle, testNumWorkers));

    size_t testIterations = testDataLoader->iterNumber();

    for (int epoch = 0; epoch < 50; ++epoch) {
        model->clearCache();
        exe->gc(Executor::FULL);
        exe->resetProfile();
        {
            AUTOTIME;
            dataLoader->reset();
            model->setIsTraining(true);
            Timer _100Time;
            Timer _iterTimer;
            int lastIndex = 0;
            int moveBatchSize = 0;
            auto meanForwardTime = 0.0f;
            auto meanBackwardTime = 0.0f;
            for (int i = 0; i < iterations; i++) {
//                MNN_PRINT("New Iteration %i\n", i);
                // AUTOTIME;
                auto trainData  = dataLoader->next();
                auto example    = trainData[0];
                auto cast       = _Cast<float>(example.first[0]);
                example.first[0] = cast * _Const(1.0f / 255.0f);
                moveBatchSize += example.first[0]->getInfo()->dim[0];

                // Compute One-Hot
//                MNN_PRINT("HERE1");
                auto newTarget = _OneHot(_Cast<int32_t>(example.second[0]), _Scalar<int>(10), _Scalar<float>(1.0f),
                                         _Scalar<float>(0.0f));

//                int padding_data[] = {0, 0, 0, 0, 0, 0, 0, 0};
//                auto padding = _Const(padding_data, {4, 2}, NCHW, halide_type_of<int>());
//                MNN_PRINT("DEBUG: Model_forward");
//                MNN_PRINT("%d", example.first[0]->getInfo()->dim[2]);
//                int batch
                auto predict = model->forward(example.first[0]);
//                MNN_PRINT("HERE2");

                auto loss    = _CrossEntropy(predict, newTarget);
//                MNN_PRINT("HERE3");
                auto lossValue = loss->readMap<float>()[0];
//                MNN_PRINT("HERE4");
//                MNN_PRINT("LOSS = %f", lossValue);
                auto forwardTime = (float)_iterTimer.durationInUs() / 1000.0f;
//                MNN_PRINT("HERE5");
//                MNN_PRINT("Forward Time %f", forwardTime);
                meanForwardTime += forwardTime/10.0;
                _iterTimer.reset();
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
                float rate   = LrScheduler::inv(0.01, epoch * iterations + i, 0.0001, 0.75);
                sgd->setLearningRate(rate);

//                MNN_PRINT("Start SGD Step");
//                MNN_PRINT("HERE6");
                sgd->step(loss);
//                MNN_PRINT("HERE7");
                meanBackwardTime += (((float)_iterTimer.durationInUs() / 1000.0f) - forwardTime)/10.0;
                _iterTimer.reset();
//                MNN_PRINT("FIN SGD Step");



                if (moveBatchSize % (10 * batchSize) == 0 || i == iterations - 1) {
#ifdef MNN_USE_LOGCAT
                    MNN_PRINT("epoch: %i %i/%i\tloss: %f\tlr: %f\ttime: %f ms / %i iter",
                              epoch, moveBatchSize, dataLoader->size(), loss->readMap<float>()[0], rate, (float)_100Time.durationInUs() / 1000.0f,
                              (i - lastIndex));
                    MNN_PRINT("Forward Time: %f ms\tBackward Time: %f ms\t", meanForwardTime, meanBackwardTime); // NOTE this is not correct if i == iterations - 1
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
        Variable::save(model->parameters(), "mnist.snapshot.mnn");
        {
            model->setIsTraining(false);
            auto forwardInput = _Input({1, 1, 28, 28}, NC4HW4);
            forwardInput->setName("data");
            auto predict = model->forward(forwardInput);
            predict->setName("prob");
            Transformer::turnModelToInfer()->onExecute({predict});
            Variable::save({predict}, "temp.mnist.mnn");
        }

        int correct = 0;
        testDataLoader->reset();
        model->setIsTraining(false);
        int moveBatchSize = 0;
        for (int i = 0; i < testIterations; i++) {
            auto data       = testDataLoader->next();
            auto example    = data[0];
            moveBatchSize += example.first[0]->getInfo()->dim[0];
            if ((i + 1) % 100 == 0) {
                std::cout << "test: " << moveBatchSize << " / " << testDataLoader->size() << std::endl;
            }
            auto cast       = _Cast<float>(example.first[0]);
            example.first[0] = cast * _Const(1.0f / 255.0f);
            auto predict    = model->forward(example.first[0]);
            predict         = _ArgMax(predict, 1);
            auto accu       = _Cast<int32_t>(_Equal(predict, _Cast<int32_t>(example.second[0]))).sum({});
            correct += accu->readMap<int32_t>()[0];
        }
        auto accu = (float)correct / (float)testDataLoader->size();
        std::cout << "epoch: " << epoch << "  accuracy: " << accu << std::endl;
        exe->dumpProfile();
    }
}
