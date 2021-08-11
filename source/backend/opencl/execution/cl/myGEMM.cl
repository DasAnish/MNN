//#include "../../../../../../../../Android/Sdk/ndk/21.1.6352462/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/9.0.8/include/opencl-c.h"
// First naive implementation
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
    C[globalCol + globalRow*M] = acc;


}
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

#define TS3 32
#define WPT TS3 * TS3 / 256
#define RTS TS3 / (WPT)


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

#define TS4 32


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
//        if (tiledCol > N)
//            Asub[col][row] = float0;
//        else
            Asub[col][row] = A[tiledCol*(M/WIDTH) + globalRow];


//        if (globalRow == 0 && globalCol == 0) {
//            printf("t: %d || tiledRow: %d || row_index: %d || tiledCol: %d || col_index: %d",
//                   t, tiledRow, globalCol*(K/WIDTH)+tiledRow, tiledCol, globalRow * (M/WIDTH) + globalRow);
//        }

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

#endif

        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }



    // Store the final results in C
    C[globalCol*(M/WIDTH) + globalRow] = acc;
//    vstore4(acc, globalCol * (M/WIDTH) + globalRow, C);
}



























