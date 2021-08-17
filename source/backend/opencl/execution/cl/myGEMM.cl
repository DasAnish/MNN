//#include "../../../../../../../../Android/Sdk/ndk/21.1.6352462/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/9.0.8/include/opencl-c.h"
// First naive implementation
#ifndef KERNEL
#define KERNEL 6
#endif
#if KERNEL == 1
__kernel void gemm1(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) {

    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)

    // Compute a single element (loop over K)
    float acc = 0.0f;
    for (int k=0; k<K; k++) {
        const int ai = k*M + globalRow,
                  bi = globalCol*K + k;

        //if (globalCol == 0 && globalRow == 0) {
        //    printf("acc: %f | ", acc);
        //}
        acc += A[ai] * B[bi];

        //if (globalCol == 0 && globalRow == 0) {
        //    printf("A[%d, %d]: %f | B[%d, %d]: %f | acc: %f ||| ", k, globalRow, A[ai], globalCol, k, B[bi], acc);
        //}

    }
    // Store the result
    C[globalCol*M + globalRow] = acc;


}

#elif KERNEL == 2
#define TS2 32
// Tiled and coalesced version
__kernel void gemm2(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C,
                      const int numTiles) {

    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS2)
    const int col = get_local_id(1); // Local col ID (max: TS2)
    const int globalRow = TS2*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS2*get_group_id(1) + col; // Col ID of C (0..N)

    if (get_global_id(0) == 0 && get_global_id(1) == 0)
        printf("GROUP Sizes: %ld, %ld", get_num_groups(0), get_num_groups(1));


    // Local memory to fit a tile of TS2*TS2 elements of A and B
    __local float Asub[TS2][TS2];
    __local float Bsub[TS2][TS2];

    // Initialise the accumulation register
    float acc = 0.0f;

    // Loop over all tiles
    //int numTiles = K/TS2;
    //if (K % TS2 != 0)
        //numTiles++;

    for (int t=0; t<numTiles; t++) {

        // Load one tile of A and B into local memory
        const int tiledRow = TS2*t + row;
        const int tiledCol = TS2*t + col;
        Asub[col][row] = A[tiledCol*M + globalRow];
        Bsub[col][row] = B[globalCol*K + tiledRow];

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        for (int k=0; k<TS2; k++) {
            acc += Asub[k][row] * Bsub[col][k];
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final result in C
    C[globalCol*M + globalRow] = acc;
}

#elif KERNEL == 3

#ifndef  TS3
#define TS3 32
#define WPT TS3 * TS3 / 256
#define RTS TS3 / (WPT)
#endif

// Increased the amount of work-per-thread by a factor WPT
__kernel void gemm3(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C,
                      const int numTiles) {

    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS3)
    const int col = get_local_id(1); // Local col ID (max: TS3/WPT == RTS)
    const int globalRow = TS3*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS3*get_group_id(1) + col; // Col ID of C (0..N)

    //if (get_global_id(0) == 0 && get_global_id(1) == 0) printf("GROUP_SIZE: %ld, %ld", get_num_groups(0), get_num_groups(1));

    // Local memory to fit a tile of TS3*TS3 elements of A and B
    __local float Asub[TS3][TS3];
    __local float Bsub[TS3][TS3];

    // Initialise the accumulation registers
    float acc[WPT];
    for (int w=0; w<WPT; w++) {
        acc[w] = 0.0f;
    }


    // Loop over all tiles
    //const int numTiles = get_num_groups(1);

    for (int t=0; t<numTiles; t++) {
        //PRINT_IF("LOOP0");
        // Load one tile of A and B into local memory
        for (int w=0; w<WPT; w++) {
            const int tiledRow = TS3*t + row;
            const int tiledCol = TS3*t + col;
            Asub[col + w*RTS][row] = A[(tiledCol + w*RTS)*M + globalRow];

            Bsub[col + w*RTS][row] = B[(globalCol + w*RTS)*K + tiledRow];
        }
        //PRINT_IF("LOOP1");

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        for (int k=0; k<TS3; k++) {
            for (int w=0; w<WPT; w++) {
                acc[w] += Asub[k][row] * Bsub[col + w*RTS][k];
            }
        }
        //PRINT_IF("LOOP2");

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }



    // Store the final results in C
    for (int w=0; w<WPT; w++) {
        C[(globalCol + w*RTS)*M + globalRow] = acc[w];
        //get_num_groups(0) * 10000 + get_num_groups(1);
        //A[(globalCol + w*RTS)*M + globalRow];
        //acc[w];//M*(globalCol + w*RTS) + globalRow;
    }
}

#elif KERNEL == 4

#ifndef WIDTH
#define TS4 32
#define WIDTH 8
#if WIDTH == 1
typedef float floatX;
#elif WIDTH == 2
typedef float2 floatX;
#elif WIDTH == 4
typedef float4 floatX;
#elif WIDTH == 8
typedef float8 floatX;
#elif WIDTH == 16
typedef float16 floatX;
#endif
#endif



// Use wider data types
__kernel void gemm4(const int M, const int N, const int K,
                    const __global floatX* A,
                    const __global floatX* B,
                    __global floatX* C,
                    const int numTiles) {

    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS4/WIDTH)
    const int col = get_local_id(1); // Local col ID (max: TS4)
    const int globalRow = (TS4/WIDTH)*get_group_id(0) + row; // 0..M/WIDTH
    const int globalCol = TS4*get_group_id(1) + col; // 0..N

    // Local memory to fit a tile of TS4*TS4 elements of A and B
    __local floatX Asub[TS4][TS4/WIDTH];
    __local floatX Bsub[TS4][TS4/WIDTH];

    // Initialise the accumulation registers
#if WIDTH == 1
    floatX acc = 0.0f;
    floatX float0 = 0.0f;
#elif WIDTH == 2
    floatX acc = { 0.0f, 0.0f };
    floatX float0 = (float2) (0, 0);
#elif WIDTH == 4
    floatX acc = { 0.0f, 0.0f, 0.0f, 0.0f };
    floatX float0 = (float4) (0, 0, 0, 0);
#elif WIDTH == 8
    floatX acc = (float8) (0, 0, 0, 0, 0, 0, 0, 0);
    floatX float0 = (float8) (0, 0, 0, 0, 0, 0, 0, 0);
#elif WIDTH == 16
    floatX acc = (float16) (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    floatX float0 = (float16) (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
#endif

    // Loop over all tiles
//    const int numTiles = K/TS4;
    for (int t=0; t<numTiles; t++) {

        // Load one tile of A and B into local memory
        const int tiledRow = (TS4/WIDTH)*t + row;
        const int tiledCol = TS4*t + col;
//        const int Ai = tiledCol * (M/WIDTH) + globalRow;
//        const int Bi = globalCol * (K/WIDTH) + tiledRow;
        if (tiledRow > M/WIDTH)
            Bsub[col][row] = float0;
        else
            Bsub[col][row] = B[globalCol*(K/WIDTH) + tiledRow];
        if (tiledCol > N)
            Asub[col][row] = float0;
        else
            Asub[col][row] = A[tiledCol*(M/WIDTH) + globalRow];

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
        // Perform the computation for a single tile

        floatX vecA, vecB;
        float valB;
        for (int k=0; k<TS4/WIDTH; k++) {
            vecB = Bsub[col][k];
#if WIDTH == 1

            float vecA = Asub[k][row];
            acc += vecB * vecA;

#elif WIDTH == 2

            float4 vecA0 = Asub[WIDTH*k + 0][row];
            float4 vecA1 = Asub[WIDTH*k + 1][row];

            float2 vecA0trans = (float2) (vecA0.s0, vecA1.s0);
            float2 vecA1trans = (float2) (vecA0.s1, vecA1.s1);

            acc.s0 += dot(vecB, vecA0trans);
            acc.s1 += dot(vecB, vecA1trans);

#elif WIDTH == 4

            float4 vecA0 = Asub[WIDTH*k + 0][row];
            float4 vecA1 = Asub[WIDTH*k + 1][row];
            float4 vecA2 = Asub[WIDTH*k + 2][row];
            float4 vecA3 = Asub[WIDTH*k + 3][row];

            float4 vecA0trans = (float4) (vecA0.s0, vecA1.s0, vecA2.s0, vecA3.s0);
            float4 vecA1trans = (float4) (vecA0.s1, vecA1.s1, vecA2.s1, vecA3.s1);
            float4 vecA2trans = (float4) (vecA0.s2, vecA1.s2, vecA2.s2, vecA3.s2);
            float4 vecA3trans = (float4) (vecA0.s3, vecA1.s3, vecA2.s3, vecA3.s3);

            acc.s0 += dot(vecB, vecA0trans);
            acc.s1 += dot(vecB, vecA1trans);
            acc.s2 += dot(vecB, vecA2trans);
            acc.s3 += dot(vecB, vecA3trans);

#elif WIDTH == 8


            float8 vecA0 = Asub[WIDTH*k + 0][row];
            float8 vecA1 = Asub[WIDTH*k + 1][row];
            float8 vecA2 = Asub[WIDTH*k + 2][row];
            float8 vecA3 = Asub[WIDTH*k + 3][row];
            float8 vecA4 = Asub[WIDTH*k + 4][row];
            float8 vecA5 = Asub[WIDTH*k + 5][row];
            float8 vecA6 = Asub[WIDTH*k + 6][row];
            float8 vecA7 = Asub[WIDTH*k + 7][row];

            float8 vecA0trans = (float8) (vecA0.s0, vecA1.s0, vecA2.s0, vecA3.s0, vecA4.s0, vecA5.s0, vecA6.s0, vecA7.s0);
            float8 vecA1trans = (float8) (vecA0.s1, vecA1.s1, vecA2.s1, vecA3.s1, vecA4.s1, vecA5.s1, vecA6.s1, vecA7.s1);
            float8 vecA2trans = (float8) (vecA0.s2, vecA1.s2, vecA2.s2, vecA3.s2, vecA4.s2, vecA5.s2, vecA6.s2, vecA7.s2);
            float8 vecA3trans = (float8) (vecA0.s3, vecA1.s3, vecA2.s3, vecA3.s3, vecA4.s3, vecA5.s3, vecA6.s3, vecA7.s3);
            float8 vecA4trans = (float8) (vecA0.s4, vecA1.s4, vecA2.s4, vecA3.s4, vecA4.s4, vecA5.s4, vecA6.s4, vecA7.s4);
            float8 vecA5trans = (float8) (vecA0.s5, vecA1.s5, vecA2.s5, vecA3.s5, vecA4.s5, vecA5.s5, vecA6.s5, vecA7.s5);
            float8 vecA6trans = (float8) (vecA0.s6, vecA1.s6, vecA2.s6, vecA3.s6, vecA4.s6, vecA5.s6, vecA6.s6, vecA7.s6);
            float8 vecA7trans = (float8) (vecA0.s7, vecA1.s7, vecA2.s7, vecA3.s7, vecA4.s7, vecA5.s7, vecA6.s7, vecA7.s7);

            acc.s0 += dot(vecB.s0123, vecA0trans.s0123);
            acc.s1 += dot(vecB.s0123, vecA1trans.s0123);
            acc.s2 += dot(vecB.s0123, vecA2trans.s0123);
            acc.s3 += dot(vecB.s0123, vecA3trans.s0123);
            acc.s4 += dot(vecB.s0123, vecA4trans.s0123);
            acc.s5 += dot(vecB.s0123, vecA5trans.s0123);
            acc.s6 += dot(vecB.s0123, vecA6trans.s0123);
            acc.s7 += dot(vecB.s0123, vecA7trans.s0123);

            acc.s0 += dot(vecB.s4567, vecA0trans.s4567);
            acc.s1 += dot(vecB.s4567, vecA1trans.s4567);
            acc.s2 += dot(vecB.s4567, vecA2trans.s4567);
            acc.s3 += dot(vecB.s4567, vecA3trans.s4567);
            acc.s4 += dot(vecB.s4567, vecA4trans.s4567);
            acc.s5 += dot(vecB.s4567, vecA5trans.s4567);
            acc.s6 += dot(vecB.s4567, vecA6trans.s4567);
            acc.s7 += dot(vecB.s4567, vecA7trans.s4567);

#elif WIDTH == 16

//loading corresponding values of vecA
float16 vecA0 = Asub[WIDTH*k + 0][row];
float16 vecA1 = Asub[WIDTH*k + 1][row];
float16 vecA2 = Asub[WIDTH*k + 2][row];
float16 vecA3 = Asub[WIDTH*k + 3][row];
float16 vecA4 = Asub[WIDTH*k + 4][row];
float16 vecA5 = Asub[WIDTH*k + 5][row];
float16 vecA6 = Asub[WIDTH*k + 6][row];
float16 vecA7 = Asub[WIDTH*k + 7][row];
float16 vecA8 = Asub[WIDTH*k + 8][row];
float16 vecA9 = Asub[WIDTH*k + 9][row];
float16 vecAa = Asub[WIDTH*k + 10][row];
float16 vecAb = Asub[WIDTH*k + 11][row];
float16 vecAc = Asub[WIDTH*k + 12][row];
float16 vecAd = Asub[WIDTH*k + 13][row];
float16 vecAe = Asub[WIDTH*k + 14][row];
float16 vecAf = Asub[WIDTH*k + 15][row];
//barrier(CLK_LOCAL_MEM_FENCE);

//the transpose lines
float16 vecA0trans = (float16) (vecA0.s0, vecA1.s0, vecA2.s0, vecA3.s0, vecA4.s0, vecA5.s0, vecA6.s0, vecA7.s0, vecA8.s0, vecA9.s0, vecAa.s0, vecAb.s0, vecAc.s0, vecAd.s0, vecAe.s0, vecAf.s0);
float16 vecA1trans = (float16) (vecA0.s1, vecA1.s1, vecA2.s1, vecA3.s1, vecA4.s1, vecA5.s1, vecA6.s1, vecA7.s1, vecA8.s1, vecA9.s1, vecAa.s1, vecAb.s1, vecAc.s1, vecAd.s1, vecAe.s1, vecAf.s1);
float16 vecA2trans = (float16) (vecA0.s2, vecA1.s2, vecA2.s2, vecA3.s2, vecA4.s2, vecA5.s2, vecA6.s2, vecA7.s2, vecA8.s2, vecA9.s2, vecAa.s2, vecAb.s2, vecAc.s2, vecAd.s2, vecAe.s2, vecAf.s2);
float16 vecA3trans = (float16) (vecA0.s3, vecA1.s3, vecA2.s3, vecA3.s3, vecA4.s3, vecA5.s3, vecA6.s3, vecA7.s3, vecA8.s3, vecA9.s3, vecAa.s3, vecAb.s3, vecAc.s3, vecAd.s3, vecAe.s3, vecAf.s3);
float16 vecA4trans = (float16) (vecA0.s4, vecA1.s4, vecA2.s4, vecA3.s4, vecA4.s4, vecA5.s4, vecA6.s4, vecA7.s4, vecA8.s4, vecA9.s4, vecAa.s4, vecAb.s4, vecAc.s4, vecAd.s4, vecAe.s4, vecAf.s4);
float16 vecA5trans = (float16) (vecA0.s5, vecA1.s5, vecA2.s5, vecA3.s5, vecA4.s5, vecA5.s5, vecA6.s5, vecA7.s5, vecA8.s5, vecA9.s5, vecAa.s5, vecAb.s5, vecAc.s5, vecAd.s5, vecAe.s5, vecAf.s5);
float16 vecA6trans = (float16) (vecA0.s6, vecA1.s6, vecA2.s6, vecA3.s6, vecA4.s6, vecA5.s6, vecA6.s6, vecA7.s6, vecA8.s6, vecA9.s6, vecAa.s6, vecAb.s6, vecAc.s6, vecAd.s6, vecAe.s6, vecAf.s6);
float16 vecA7trans = (float16) (vecA0.s7, vecA1.s7, vecA2.s7, vecA3.s7, vecA4.s7, vecA5.s7, vecA6.s7, vecA7.s7, vecA8.s7, vecA9.s7, vecAa.s7, vecAb.s7, vecAc.s7, vecAd.s7, vecAe.s7, vecAf.s7);
float16 vecA8trans = (float16) (vecA0.s8, vecA1.s8, vecA2.s8, vecA3.s8, vecA4.s8, vecA5.s8, vecA6.s8, vecA7.s8, vecA8.s8, vecA9.s8, vecAa.s8, vecAb.s8, vecAc.s8, vecAd.s8, vecAe.s8, vecAf.s8);
float16 vecA9trans = (float16) (vecA0.s9, vecA1.s9, vecA2.s9, vecA3.s9, vecA4.s9, vecA5.s9, vecA6.s9, vecA7.s9, vecA8.s9, vecA9.s9, vecAa.s9, vecAb.s9, vecAc.s9, vecAd.s9, vecAe.s9, vecAf.s9);
float16 vecAatrans = (float16) (vecA0.sa, vecA1.sa, vecA2.sa, vecA3.sa, vecA4.sa, vecA5.sa, vecA6.sa, vecA7.sa, vecA8.sa, vecA9.sa, vecAa.sa, vecAb.sa, vecAc.sa, vecAd.sa, vecAe.sa, vecAf.sa);
float16 vecAbtrans = (float16) (vecA0.sb, vecA1.sb, vecA2.sb, vecA3.sb, vecA4.sb, vecA5.sb, vecA6.sb, vecA7.sb, vecA8.sb, vecA9.sb, vecAa.sb, vecAb.sb, vecAc.sb, vecAd.sb, vecAe.sb, vecAf.sb);
float16 vecActrans = (float16) (vecA0.sc, vecA1.sc, vecA2.sc, vecA3.sc, vecA4.sc, vecA5.sc, vecA6.sc, vecA7.sc, vecA8.sc, vecA9.sc, vecAa.sc, vecAb.sc, vecAc.sc, vecAd.sc, vecAe.sc, vecAf.sc);
float16 vecAdtrans = (float16) (vecA0.sd, vecA1.sd, vecA2.sd, vecA3.sd, vecA4.sd, vecA5.sd, vecA6.sd, vecA7.sd, vecA8.sd, vecA9.sd, vecAa.sd, vecAb.sd, vecAc.sd, vecAd.sd, vecAe.sd, vecAf.sd);
float16 vecAetrans = (float16) (vecA0.se, vecA1.se, vecA2.se, vecA3.se, vecA4.se, vecA5.se, vecA6.se, vecA7.se, vecA8.se, vecA9.se, vecAa.se, vecAb.se, vecAc.se, vecAd.se, vecAe.se, vecAf.se);
float16 vecAftrans = (float16) (vecA0.sf, vecA1.sf, vecA2.sf, vecA3.sf, vecA4.sf, vecA5.sf, vecA6.sf, vecA7.sf, vecA8.sf, vecA9.sf, vecAa.sf, vecAb.sf, vecAc.sf, vecAd.sf, vecAe.sf, vecAf.sf);
//barrier(CLK_LOCAL_MEM_FENCE);
//Dot prods
acc.s0 += dot(vecB.s0123, vecA0trans.s0123);
acc.s0 += dot(vecB.s4567, vecA0trans.s4567);
acc.s0 += dot(vecB.s89ab, vecA0trans.s89ab);
acc.s0 += dot(vecB.scdef, vecA0trans.scdef);

acc.s1 += dot(vecB.s0123, vecA1trans.s0123);
acc.s1 += dot(vecB.s4567, vecA1trans.s4567);
acc.s1 += dot(vecB.s89ab, vecA1trans.s89ab);
acc.s1 += dot(vecB.scdef, vecA1trans.scdef);

acc.s2 += dot(vecB.s0123, vecA2trans.s0123);
acc.s2 += dot(vecB.s4567, vecA2trans.s4567);
acc.s2 += dot(vecB.s89ab, vecA2trans.s89ab);
acc.s2 += dot(vecB.scdef, vecA2trans.scdef);

acc.s3 += dot(vecB.s0123, vecA3trans.s0123);
acc.s3 += dot(vecB.s4567, vecA3trans.s4567);
acc.s3 += dot(vecB.s89ab, vecA3trans.s89ab);
acc.s3 += dot(vecB.scdef, vecA3trans.scdef);

acc.s4 += dot(vecB.s0123, vecA4trans.s0123);
acc.s4 += dot(vecB.s4567, vecA4trans.s4567);
acc.s4 += dot(vecB.s89ab, vecA4trans.s89ab);
acc.s4 += dot(vecB.scdef, vecA4trans.scdef);

acc.s5 += dot(vecB.s0123, vecA5trans.s0123);
acc.s5 += dot(vecB.s4567, vecA5trans.s4567);
acc.s5 += dot(vecB.s89ab, vecA5trans.s89ab);
acc.s5 += dot(vecB.scdef, vecA5trans.scdef);

acc.s6 += dot(vecB.s0123, vecA6trans.s0123);
acc.s6 += dot(vecB.s4567, vecA6trans.s4567);
acc.s6 += dot(vecB.s89ab, vecA6trans.s89ab);
acc.s6 += dot(vecB.scdef, vecA6trans.scdef);

acc.s7 += dot(vecB.s0123, vecA7trans.s0123);
acc.s7 += dot(vecB.s4567, vecA7trans.s4567);
acc.s7 += dot(vecB.s89ab, vecA7trans.s89ab);
acc.s7 += dot(vecB.scdef, vecA7trans.scdef);

acc.s8 += dot(vecB.s0123, vecA8trans.s0123);
acc.s8 += dot(vecB.s4567, vecA8trans.s4567);
acc.s8 += dot(vecB.s89ab, vecA8trans.s89ab);
acc.s8 += dot(vecB.scdef, vecA8trans.scdef);

acc.s9 += dot(vecB.s0123, vecA9trans.s0123);
acc.s9 += dot(vecB.s4567, vecA9trans.s4567);
acc.s9 += dot(vecB.s89ab, vecA9trans.s89ab);
acc.s9 += dot(vecB.scdef, vecA9trans.scdef);

acc.sa += dot(vecB.s0123, vecAatrans.s0123);
acc.sa += dot(vecB.s4567, vecAatrans.s4567);
acc.sa += dot(vecB.s89ab, vecAatrans.s89ab);
acc.sa += dot(vecB.scdef, vecAatrans.scdef);

acc.sb += dot(vecB.s0123, vecAbtrans.s0123);
acc.sb += dot(vecB.s4567, vecAbtrans.s4567);
acc.sb += dot(vecB.s89ab, vecAbtrans.s89ab);
acc.sb += dot(vecB.scdef, vecAbtrans.scdef);

acc.sc += dot(vecB.s0123, vecActrans.s0123);
acc.sc += dot(vecB.s4567, vecActrans.s4567);
acc.sc += dot(vecB.s89ab, vecActrans.s89ab);
acc.sc += dot(vecB.scdef, vecActrans.scdef);

acc.sd += dot(vecB.s0123, vecAdtrans.s0123);
acc.sd += dot(vecB.s4567, vecAdtrans.s4567);
acc.sd += dot(vecB.s89ab, vecAdtrans.s89ab);
acc.sd += dot(vecB.scdef, vecAdtrans.scdef);

acc.se += dot(vecB.s0123, vecAetrans.s0123);
acc.se += dot(vecB.s4567, vecAetrans.s4567);
acc.se += dot(vecB.s89ab, vecAetrans.s89ab);
acc.se += dot(vecB.scdef, vecAetrans.scdef);

acc.sf += dot(vecB.s0123, vecAftrans.s0123);
acc.sf += dot(vecB.s4567, vecAftrans.s4567);
acc.sf += dot(vecB.s89ab, vecAftrans.s89ab);
acc.sf += dot(vecB.scdef, vecAftrans.scdef);


#endif

        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }



    // Store the final results in C
    C[globalCol*(M/WIDTH) + globalRow] = acc;
//    vstore4(acc, globalCol * (M/WIDTH) + globalRow, C);
}







#elif KERNEL == 5
#ifndef TSM                // The tile-size in dimension M
#define TSM 64
#define TSN 64                 // The tile-size in dimension N
#define TSK 32                 // The tile-size in dimension K
#define WPTN 8                 // The work-per-thread in dimension N
#define WPTM 1
#endif
#define RTSN (TSN/WPTN)        // The reduced tile-size in dimension N
#define RTSM (TSM/WPTM)
#define LPT ((TSK)/(RTSN)) // The loads-per-thread for a tile

// Pre-transpose the input matrix B and use rectangular tiles
__kernel void gemm5(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C,
                      const int numTiles) {

    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TSM)
    const int col = get_local_id(1); // Local col ID (max: TSN/WPTN)
    const int globalRow = TSM*get_group_id(0) + row; // 0..M
    const int globalCol = TSN*get_group_id(1) + col; // 0..N

    // Local memory to fit a tile of A and B
    __local float Asub[TSK][TSM];
    __local float Bsub[TSN][TSK+2];

    // Initialise the accumulation registers
    float acc[WPTN];
    for (int w=0; w<WPTN; w++) {
        acc[w] = 0.0f;
    }

    // Loop over all tiles
//    int numTiles = K/TSK;
    for (int t=0; t<numTiles; t++) {

        // Load one tile of A and B into local memory
        for (int l=0; l<LPT; l++) {
            int tiledIndex = TSK*t + col + l*RTSN;
            int indexA = tiledIndex*M + TSM*get_group_id(0) + row;
            int indexB = tiledIndex*N + TSN*get_group_id(1) + row;
            Asub[col + l*RTSN][row] = A[indexA];
            Bsub[row][col + l*RTSN] = B[indexB];
        }

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        for (int k=0; k<TSK; k++) {
            for (int w=0; w<WPTN; w++) {
                acc[w] += Asub[k][row] * Bsub[col + w*RTSN][k];
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final results in C
    for (int w=0; w<WPTN; w++) {
        C[(globalCol + w*RTSN)*M + globalRow] = acc[w];
    }
}
#elif KERNEL == 6

#ifndef TSM
#define TSM 128                // The tile-size in dimension M
#define TSN 128                // The tile-size in dimension N
#define TSK 16                 // The tile-size in dimension K
#define WPTM 8                 // The work-per-thread in dimension M
#define WPTN 8                 // The work-per-thread in dimension N
#endif

#define RTSM (TSM/WPTM)        // The reduced tile-size in dimension M
#define RTSN (TSN/WPTN)        // The reduced tile-size in dimension N
#define LPTA ((TSK*TSM)/(RTSM*RTSN)) // Loads-per-thread for A
#define LPTB ((TSK*TSN)/(RTSM*RTSN)) // Loads-per-thread for B

// Use 2D register blocking (further increase in work per thread)
__kernel void gemm6(const int M, const int N, const int K,
                       __global float* A,
                       __global float* B,
                      __global float* C,
                      const int numTiles) {

    // Thread identifiers
    const int tidm = get_local_id(0); // Local row ID (max: TSM/WPTM)
    const int tidn = get_local_id(1); // Local col ID (max: TSN/WPTN)
    const int offsetM = TSM*get_group_id(0); // Work-group offset
    const int offsetN = TSN*get_group_id(1); // Work-group offset

    if (get_global_id(0) == 0 && get_global_id(1) == 0) {
        printf("6\nTSM: %d || WPTM: %d || RTSM: %d\n", TSM, WPTM, RTSM);
        printf("TSN: %d || WPTN: %d || RTSN: %d", TSN, WPTN, RTSN);
    }

    // Local memory to fit a tile of A and B
    __local float Asub[TSK][TSM+2];
    __local float Bsub[TSK][TSN+2];

    // Allocate register space
    float Areg;
    float Breg[WPTN];
    float acc[WPTM][WPTN];

    // Initialise the accumulation registers
    for (int wm=0; wm<WPTM; wm++) {
        for (int wn=0; wn<WPTN; wn++) {
            acc[wm][wn] = 0.0f;
        }
    }

    // Loop over all tiles
//    int numTiles = K/TSK;
    for (int t=0; t<numTiles; t++) {

        // Load one tile of A and B into local memory
        for (int la=0; la<LPTA; la++) {
            int tid = tidn*RTSM + tidm;
            int id = la*RTSN*RTSM + tid;
            int row = id % TSM;
            int col = id / TSM;
            int tiledIndex = TSK*t + col;
//            if (tiledIndex >= K && offsetM + row >= M)
//                Asub[col][row] = 0.0f;
//            else
                Asub[col][row] = A[tiledIndex*M + offsetM + row];
//            if (tiledIndex >= K && offsetN + row >= N)
//                Bsub[row][col] = 0.0f;
//            else
                Bsub[col][row] = B[tiledIndex*N + offsetN + row];
        }

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Loop over the values of a single tile
        for (int k=0; k<TSK; k++) {

            // Cache the values of Bsub in registers
            for (int wn=0; wn<WPTN; wn++) {
                int col = tidn + wn*RTSN;
                Breg[wn] = Bsub[k][col];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            // Perform the computation
            for (int wm=0; wm<WPTM; wm++) {
                int row = tidm + wm*RTSM;
                Areg = Asub[k][row];
                for (int wn=0; wn<WPTN; wn++) {
                    acc[wm][wn] += Areg * Breg[wn];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

        }

        // Synchronise before loading the next tile

    }

    // Store the final results in C
    for (int wm=0; wm<WPTM; wm++) {
        int globalRow = offsetM + tidm + wm*RTSM;
        for (int wn=0; wn<WPTN; wn++) {
            int globalCol = offsetN + tidn + wn*RTSN;
            C[globalCol*M + globalRow] = acc[wm][wn];
        }
    }
}

#endif

#if KERNEL == 5 || KERNEL == 6 || KERNEL == 7 || KERNEL == 8
#ifndef TRANSPOSEX
#define TRANSPOSEX 32
#define TRANSPOSEY 32
#endif

__kernel void transpose(const int P, const int Q,
                        const __global float* input,
                        __global float* output) {

    // Thread identifiers
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int ID0 = get_group_id(0)*TRANSPOSEX + tx; // 0..P
    const int ID1 = get_group_id(1)*TRANSPOSEY + ty; // 0..Q

    // Set-up the local memory for shuffling
    __local float buffer[TRANSPOSEX][TRANSPOSEY];

    // Swap the x and y coordinates to perform the rotation (coalesced)
    if (ID0 < P && ID1 < Q) {
        buffer[ty][tx] = input[ID1*P + ID0];
    }

    // Synchronise all threads
    barrier(CLK_LOCAL_MEM_FENCE);

    // We don't have to swap the x and y thread indices here,
    // because that's already done in the local memory
    const int newID0 = get_group_id(1)*TRANSPOSEY + tx;
    const int newID1 = get_group_id(0)*TRANSPOSEX + ty;

    // Store the transposed result (coalesced)
    if (newID0 < Q && newID1 < P) {
        output[newID1*Q + newID0] = buffer[tx][ty];
    }
}

#endif


#if KERNEL == 7


#ifndef WIDTH
#define TS4 32
#define WIDTH 4
#endif

#if WIDTH == 1
typedef float floatX;
#elif WIDTH == 2
typedef float2 floatX;
#elif WIDTH == 4
typedef float4 floatX;
#elif WIDTH == 8
typedef float8 floatX;
#elif WIDTH == 16
typedef float16 floatX;
#endif


#ifndef TSM
#define TSM 128                // The tile-size in dimension M
#define TSN 128                // The tile-size in dimension N
#define TSK 16                 // The tile-size in dimension K
#define WPTM 8                 // The work-per-thread in dimension M
#define WPTN 8                 // The work-per-thread in dimension N
#endif

#define RTSM (TSM/WPTM)        // The reduced tile-size in dimension M
#define RTSN (TSN/WPTN)        // The reduced tile-size in dimension N
#define LPTA ((TSK*TSM)/(RTSM*RTSN)) // Loads-per-thread for A
#define LPTB ((TSK*TSN)/(RTSM*RTSN)) // Loads-per-thread for B

// Wider loads combined with 2D register blocking
__kernel void gemm7(const int M, const int N, const int K,
                      const __global floatX* A,
                      const __global floatX* B,
                      __global float* C,
                      const int numTiles) {

    if (get_global_id(0) == 0 && get_global_id(1) == 0) {
        printf("7\nTSM: %d || WPTM: %d || RTSM: %d\n", TSM, WPTM, RTSM);
        printf("TSN: %d || WPTN: %d || RTSN: %d\n", TSN, WPTN, RTSN);
        printf("LPTA: %d || LPTB: %d", LPTA, LPTB);
    }
    
    // Thread identifiers
    const int tidm = get_local_id(0); // Local row ID (max: TSM/WPTM)
    const int tidn = get_local_id(1); // Local col ID (max: TSN/WPTN)
    const int offsetM = TSM*get_group_id(0); // Work-group offset
    const int offsetN = TSN*get_group_id(1); // Work-group offset
 
    // Local memory to fit a tile of A and B
    __local float Asub[TSK][TSM];
    __local float Bsub[TSK][TSN+2];
 
    // Allocate register space
    float Areg;
    float Breg[WPTN];
    float acc[WPTM][WPTN];
 
    // Initialise the accumulation registers
    for (int wm=0; wm<WPTM; wm++) {
        for (int wn=0; wn<WPTN; wn++) {
            acc[wm][wn] = 0.0f;
        }
    }
    
    // Loop over all tiles
//    int numTiles = K/TSK;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        for (int la=0; la<LPTA/WIDTH; la++) {
            int tid = tidn*RTSM + tidm;
            int id = la*RTSN*RTSM + tid;
            int row = id % (TSM/WIDTH);
            int col = id / (TSM/WIDTH);
 
            // Load the values (wide vector load)
            int tiledIndex = TSK*t + col;
            floatX vecA = A[tiledIndex*(M/WIDTH) + offsetM/WIDTH + row];
            floatX vecB = B[tiledIndex*(N/WIDTH) + offsetN/WIDTH + row];
 
            // Store the loaded vectors into local memory
            #if WIDTH == 1
                Asub[col][row] = vecA;
                Asub[col][row] = vecA;
            #elif WIDTH == 2
                Asub[col][WIDTH*row + 0] = vecA.x;
                Asub[col][WIDTH*row + 1] = vecA.y;
            #elif WIDTH == 4
                Asub[col][WIDTH*row + 0] = vecA.x;
                Asub[col][WIDTH*row + 1] = vecA.y;
                Asub[col][WIDTH*row + 2] = vecA.z;
                Asub[col][WIDTH*row + 3] = vecA.w;
            #endif
            #if WIDTH == 1
                Bsub[col][row] = vecB;
                Bsub[col][row] = vecB;
            #elif WIDTH == 2
                Bsub[col][WIDTH*row + 0] = vecB.x;
                Bsub[col][WIDTH*row + 1] = vecB.y;
            #elif WIDTH == 4
                Bsub[col][WIDTH*row + 0] = vecB.x;
                Bsub[col][WIDTH*row + 1] = vecB.y;
                Bsub[col][WIDTH*row + 2] = vecB.z;
                Bsub[col][WIDTH*row + 3] = vecB.w;
            #endif
        }
        
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Loop over the values of a single tile
        for (int k=0; k<TSK; k++) {
 
            // Cache the values of Bsub in registers
            for (int wn=0; wn<WPTN; wn++) {
                int col = tidn + wn*RTSN;
                Breg[wn] = Bsub[k][col];
            }
 
            // Perform the computation
            for (int wm=0; wm<WPTM; wm++) {
                int row = tidm + wm*RTSM;
                Areg = Asub[k][row];
                for (int wn=0; wn<WPTN; wn++) {
                    acc[wm][wn] += Areg * Breg[wn];
                }
            }
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final results in C
    for (int wm=0; wm<WPTM; wm++) {
        int globalRow = offsetM + tidm + wm*RTSM;
        for (int wn=0; wn<WPTN; wn++) {
            int globalCol = offsetN + tidn + wn*RTSN;
            C[globalCol*M + globalRow] = acc[wm][wn];
        }
    }
}

#endif



#if KERNEL == 8

#ifndef WIDTH
#define TS4 32
#define WIDTH 4
#endif

#if WIDTH == 1
typedef float floatX;
#elif WIDTH == 2
typedef float2 floatX;
#elif WIDTH == 4
typedef float4 floatX;
#elif WIDTH == 8
typedef float8 floatX;
#elif WIDTH == 16
typedef float16 floatX;
#endif


#ifndef TSM
#define TSM 128                // The tile-size in dimension M
#define TSN 128                // The tile-size in dimension N
#define TSK 16                 // The tile-size in dimension K
#define WPTM 8                 // The work-per-thread in dimension M
#define WPTN 8                 // The work-per-thread in dimension N
#endif

#define RTSM (TSM/WPTM)        // The reduced tile-size in dimension M
#define RTSN (TSN/WPTN)        // The reduced tile-size in dimension N
#define LPTA ((TSK*TSM)/(RTSM*RTSN)) // Loads-per-thread for A
#define LPTB ((TSK*TSN)/(RTSM*RTSN)) // Loads-per-thread for B

// Wider loads combined with 2D register blocking
__kernel void gemm8(const int M, const int N, const int K,
                      const __global floatX* A,
                      const __global floatX* B,
                      __global float* C,
                      const int numTiles) {

    if (get_global_id(0) == 0 && get_global_id(1) == 0) {
        printf("8\nTSM: %d || WPTM: %d || RTSM: %d\n", TSM, WPTM, RTSM);
        printf("TSN: %d || WPTN: %d || RTSN: %d\n", TSN, WPTN, RTSN);
        printf("LPTA: %d || LPTB: %d", LPTA, LPTB);
    }
    
    // Thread identifiers
    const int tidm = get_local_id(0); // Local row ID (max: TSM/WPTM)
    const int tidn = get_local_id(1); // Local col ID (max: TSN/WPTN)
    const int offsetM = TSM*get_group_id(0); // Work-group offset
    const int offsetN = TSN*get_group_id(1); // Work-group offset
 
    // Local memory to fit a tile of A and B
    __local float Asub[2][TSK*TSM];
    __local float Bsub[2][TSK*TSN];
 
    // Allocate register space
    float Areg;
    float Breg[WPTN];
    float acc[WPTM][WPTN];
 
    // Initialise the accumulation registers
    for (int wm=0; wm<WPTM; wm++) {
        for (int wn=0; wn<WPTN; wn++) {
            acc[wm][wn] = 0.0f;
        }
    }
    
    
    // Load one tile of A and B into local memory
        for (int la=0; la<LPTA/WIDTH; la++) {
            int tid = tidn*RTSM + tidm;
            int id = la*RTSN*RTSM + tid;
            int row = id % (TSM/WIDTH);
            int col = id / (TSM/WIDTH);
 
            // Load the values (wide vector load)
            int tiledIndex = col; // t = 0
            floatX vecA = A[tiledIndex*(M/WIDTH) + offsetM/WIDTH + row];
            floatX vecB = B[tiledIndex*(N/WIDTH) + offsetN/WIDTH + row];
 
            // Store the loaded vectors into local memory
            #if WIDTH == 1
                Asub[0][col*TSM + row] = vecA;
                Asub[0][col*TSM + row] = vecA;
            #elif WIDTH == 2
                Asub[0][col*TSM + WIDTH*row + 0] = vecA.x;
                Asub[0][col*TSM + WIDTH*row + 1] = vecA.y;
            #elif WIDTH == 4
                Asub[0][col*TSM + WIDTH*row + 0] = vecA.x;
                Asub[0][col*TSM + WIDTH*row + 1] = vecA.y;
                Asub[0][col*TSM + WIDTH*row + 2] = vecA.z;
                Asub[0][col*TSM + WIDTH*row + 3] = vecA.w;
            #endif
            #if WIDTH == 1
                Bsub[0][col*TSM + row] = vecB;
                Bsub[0][col*TSM + row] = vecB;
            #elif WIDTH == 2
                Bsub[0][col*TSM + WIDTH*row + 0] = vecB.x;
                Bsub[0][col*TSM + WIDTH*row + 1] = vecB.y;
            #elif WIDTH == 4
                Bsub[0][col*TSM + WIDTH*row + 0] = vecB.x;
                Bsub[0][col*TSM + WIDTH*row + 1] = vecB.y;
                Bsub[0][col*TSM + WIDTH*row + 2] = vecB.z;
                Bsub[0][col*TSM + WIDTH*row + 3] = vecB.w;
            #endif
        }
    
    // Loop over all tiles
//    int numTiles = K/TSK;
    for (int t=0; t<numTiles; t++) {

        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Load one tile of A and B into local memory
        int tt = t+1;
        if (tt < numTiles) {
        for (int la=0; la<LPTA/WIDTH; la++) {
            int tid = tidn*RTSM + tidm;
            int id = la*RTSN*RTSM + tid;
            int row = id % (TSM/WIDTH);
            int col = id / (TSM/WIDTH);

            // Load the values (wide vector load)
            int tiledIndex = TSK*tt + col;
            floatX vecA = A[tiledIndex*(M/WIDTH) + offsetM/WIDTH + row];
            floatX vecB = B[tiledIndex*(N/WIDTH) + offsetN/WIDTH + row];

            // Store the loaded vectors into local memory
            #if WIDTH == 1
                Asub[tt%2][col*TSM + row] = vecA;
                Asub[tt%2][col*TSM + row] = vecA;
            #elif WIDTH == 2
                Asub[tt%2][col*TSM + WIDTH*row + 0] = vecA.x;
                Asub[tt%2][col*TSM + WIDTH*row + 1] = vecA.y;
            #elif WIDTH == 4
                Asub[tt%2][col*TSM + WIDTH*row + 0] = vecA.x;
                Asub[tt%2][col*TSM + WIDTH*row + 1] = vecA.y;
                Asub[tt%2][col*TSM + WIDTH*row + 2] = vecA.z;
                Asub[tt%2][col*TSM + WIDTH*row + 3] = vecA.w;
            #endif
            #if WIDTH == 1
                Bsub[tt%2][col*TSN + row] = vecB;
                Bsub[tt%2][col*TSN + row] = vecB;
            #elif WIDTH == 2
                Bsub[tt%2][col*TSN + WIDTH*row + 0] = vecB.x;
                Bsub[tt%2][col*TSN + WIDTH*row + 1] = vecB.y;
            #elif WIDTH == 4
                Bsub[tt%2][col*TSN + WIDTH*row + 0] = vecB.x;
                Bsub[tt%2][col*TSN + WIDTH*row + 1] = vecB.y;
                Bsub[tt%2][col*TSN + WIDTH*row + 2] = vecB.z;
                Bsub[tt%2][col*TSN + WIDTH*row + 3] = vecB.w;
            #endif
        }
        }
        
        // Synchronise to make sure the tile is loaded

 
        // Loop over the values of a single tile
        for (int k=0; k<TSK; k++) {
 
            // Cache the values of Bsub in registers
            for (int wn=0; wn<WPTN; wn++) {
                int col = tidn + wn*RTSN;
                Breg[wn] = Bsub[t%2][k*TSN + col];
            }
 
            // Perform the computation
            for (int wm=0; wm<WPTM; wm++) {
                int row = tidm + wm*RTSM;
                Areg = Asub[t%2][k*TSM + row];
                for (int wn=0; wn<WPTN; wn++) {
                    acc[wm][wn] += Areg * Breg[wn];
                }
            }
        }
 
        // Synchronise before loading the next tile
//        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final results in C
    for (int wm=0; wm<WPTM; wm++) {
        int globalRow = offsetM + tidm + wm*RTSM;
        for (int wn=0; wn<WPTN; wn++) {
            int globalCol = offsetN + tidn + wn*RTSN;
            C[globalCol*M + globalRow] = acc[wm][wn];
        }
    }
}

#endif

