//#include "../../../../../../../../Android/Sdk/ndk/21.1.6352462/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/9.0.8/include/opencl-c.h"
// First naive implementation

#define KERNEL 4


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
        acc += A[ai] * B[bi];

    }
    // Store the result
    C[globalCol*M + globalRow] = acc;


}

#elif KERNEL == 2

// Tiled and coalesced version
__kernel void gemm2(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C,
                      const int numTiles) {

//    if (get_global_id(0) == 0 && get_global_id(1) == 0) {
//        printf("%d\n", TS2);
//    }

    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS2)
    const int col = get_local_id(1); // Local col ID (max: TS2)
    const int globalRow = TS2*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS2*get_group_id(1) + col; // Col ID of C (0..N)

//    if (get_global_id(0) == 0 && get_global_id(1) == 0)
//        printf("GROUP Sizes: %ld, %ld", get_num_groups(0), get_num_groups(1));


    // Local memory to fit a tile of TS2*TS2 elements of A and B
    __local float Asub[TS2][TS2];
    __local float Bsub[TS2][TS2];
    Asub[col][row] = 0.0f;
    Bsub[col][row] = 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Initialise the accumulation register
    float acc = 0.0f;

//    const int numTiles = K / TS2;
    for (int t=0; t<numTiles; t++) {

        // Load one tile of A and B into local memory
        const int tiledRow = TS2*t + row;
        const int tiledCol = TS2*t + col;
//        if (tiledRow > M)
//            Asub[col][row] = 0.0f;
//        else
            Asub[col][row] = A[tiledCol + globalRow*K];

//        if (tiledCol > N)
//            Bsub[col][row] = 0.0f;
//        else
            Bsub[col][row] = B[globalCol + tiledRow*N];

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

    C[globalCol + globalRow*N] = acc;//globalRow;//globalCol*M + globalRow;
}

#elif KERNEL == 3

//#ifndef  TS3
//#define TS3 32
//#define WPT 8
//#define RTS TS3 / (WPT)
//#endif

// Increased the amount of work-per-thread by a factor WPT
__kernel void gemm3(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C,
                      const int numTiles) {
#ifndef PROFILING
    if (get_global_id(0) == 0 && get_global_id(1) == 0) {
        printf("3\n TS: %d, WPT: %d, RTS: %d\n", TS3, WPT, RTS);
    }
#endif

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
    }
}

#elif KERNEL == 4

#define WIDTH 4
#define TS4 16

#if WIDTH == 1
#define floatX float
#elif WIDTH == 2
#define floatX float2
#elif WIDTH == 4
#define floatX float4
#elif WIDTH == 8
#define floatX float8
#elif WIDTH == 16
#define floatX float16
#endif



// Use wider data types
__kernel void gemm4(const int M, const int N, const int K,
                    const __global floatX* A,
                    const __global floatX* B,
#ifdef BIAS
                    const __global floatX* bias,
#endif
                    __global floatX* C,
                    const int numTiles) {

    // Thread identifiers
    const int row = get_local_id(0); //max: TS4 // Local row ID (max: TS4/WIDTH)
    const int col = get_local_id(1); //max: TS4/WIDTH // Local col ID (max: TS4)
    const int globalRow = (TS4)*get_group_id(0) + row; // 0..M// 0..M/WIDTH
    const int globalCol = (TS4 / WIDTH)*get_group_id(1) + col; // 0..N/WIDTH // 0..N

//    if (globalRow == 0 && globalCol == 0) {
//        printf("(%d, %d)", get_global_size(0))
//    }

    // Local memory to fit a tile of TS4*TS4 elements of A and B
    //transed
    __local floatX Asub[TS4][TS4 / WIDTH];
    __local floatX Bsub[TS4][TS4 / WIDTH];

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

#ifdef BIAS
    acc = bias[globalCol];
#endif

    // Loop over all tiles
//    const int numTiles = K/TS4;
    for (int t=0; t<numTiles; t++) {

        // Load one tile of A and B into local memory
        //transed
        const int tiledRow = (TS4)*t + row;
        const int tiledCol = (TS4/WIDTH)*t + col;
//        const int Ai = tiledCol * (M/WIDTH) + globalRow;
//        const int Bi = globalCol * (K/WIDTH) + tiledRow;
//        if (tiledCol > K / WIDTH)
//            Asub[col][row] = float0;
//        else
            Asub[row][col] = A[tiledCol + globalRow*(K/WIDTH)]; //transed

//        if (tiledRow > K)
//            Bsub[col][row] = float0;
//        else
            Bsub[row][col] = B[globalCol + tiledRow*(N/WIDTH)]; // transed


        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
        // Perform the computation for a single tile

        floatX vecA;
        float valB;
        for (int k=0; k<TS4/WIDTH; k++) {
            vecA = Asub[row][k]; // transed
#if WIDTH == 1

            float vecA = Bsub[col][k];
            acc += vecB * vecA;

#elif WIDTH == 2

            float4 vecA0 = Bsub[col][WIDTH*k + 0];
            float4 vecA1 = Bsub[col][WIDTH*k + 1];

            float2 vecA0trans = (float2) (vecA0.s0, vecA1.s0);
            float2 vecA1trans = (float2) (vecA0.s1, vecA1.s1);

            acc.s0 += dot(vecA, vecA0trans);
            acc.s1 += dot(vecA, vecA1trans);

#elif WIDTH == 4

            //transed
            float4 vecB0 = Bsub[WIDTH*k + 0][col];
            float4 vecB1 = Bsub[WIDTH*k + 1][col];
            float4 vecB2 = Bsub[WIDTH*k + 2][col];
            float4 vecB3 = Bsub[WIDTH*k + 3][col];

            float4 vecB0trans = (float4) (vecB0.s0, vecB1.s0, vecB2.s0, vecB3.s0);
            float4 vecB1trans = (float4) (vecB0.s1, vecB1.s1, vecB2.s1, vecB3.s1);
            float4 vecB2trans = (float4) (vecB0.s2, vecB1.s2, vecB2.s2, vecB3.s2);
            float4 vecB3trans = (float4) (vecB0.s3, vecB1.s3, vecB2.s3, vecB3.s3);

            acc.s0 += dot(vecA, vecB0trans);
            acc.s1 += dot(vecA, vecB1trans);
            acc.s2 += dot(vecA, vecB2trans);
            acc.s3 += dot(vecA, vecB3trans);


#endif

        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // Store the final results in C
    int index = globalCol + globalRow * (N / WIDTH);
    C[globalCol + globalRow*(N/WIDTH)] = acc; //transed
}

// Use wider data types
__kernel void gemm4_transB(const int M, const int N, const int K,
                    const __global floatX* A,
                    const __global floatX* B,
#ifdef BIAS
                    const __global floatX* bias,
#endif
                    __global floatX* C,
                    const int numTiles) {

    // Thread identifiers
    const int row = get_local_id(0); //max: TS4 // Local row ID (max: TS4/WIDTH)
    const int col = get_local_id(1); //max: TS4/WIDTH // Local col ID (max: TS4)
    const int globalRow = (TS4)*get_group_id(0) + row; // 0..M// 0..M/WIDTH
    const int globalCol = (TS4 / WIDTH)*get_group_id(1) + col; // 0..N/WIDTH // 0..N

    const int transRow = row / WIDTH;
    const int transCol = col * WIDTH + row % WIDTH;
    const int globalRow2 = TS4 * get_group_id(1) + row;
//    const int globalTransRow = globalRow / WIDTH;
    const int globalTransCol = globalCol * WIDTH + globalRow % WIDTH;

//    if (globalRow == 0 && globalCol == 0) {
//        printf("(%d, %d)", get_global_size(0))
//    }



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
#endif

#ifdef BIAS
    acc = bias[globalCol];
#endif

    // Local memory to fit a tile of TS4*TS4 elements of A and B
    //transed
    __local floatX Asub[TS4][TS4 / WIDTH];
    __local floatX Bsub[TS4][TS4 / WIDTH];
//    if (row > )

    // Loop over all tiles
//    const int numTiles = K/TS4;
    for (int t=0; t<numTiles; t++) {

        // Load one tile of A and B into local memory
        //transed
        int tiledRow = (TS4)*t + row;
        int tiledCol = (TS4/WIDTH)*t + col;

            Asub[row][col] = A[tiledCol + globalRow*(K/WIDTH)]; //transed

            Bsub[row][col] = B[tiledCol + globalRow2*(K/WIDTH)]; // transed


        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
        // Perform the computation for a single tile

        floatX vecA;
        float valB;
        for (int k=0; k<TS4/WIDTH; k++) {
            vecA = Asub[row][k]; // transed
#if WIDTH == 1

            float vecB = Bsub[k][col];
            acc += vecB * vecA;

#elif WIDTH == 2

            float4 vecB0 = Bsub[k][WIDTH*col + 0];
            float4 vecB1 = Bsub[k][WIDTH*col + 1];

//            float2 vecA0trans = (float2) (vecA0.s0, vecA1.s0);
//            float2 vecA1trans = (float2) (vecA0.s1, vecA1.s1);

            acc.s0 += dot(vecA, vecA0);
            acc.s1 += dot(vecA, vecA1);

#elif WIDTH == 4

            //transed
            float4 vecB0 = Bsub[WIDTH*col + 0][k];
            float4 vecB1 = Bsub[WIDTH*col + 1][k];
            float4 vecB2 = Bsub[WIDTH*col + 2][k];
            float4 vecB3 = Bsub[WIDTH*col + 3][k];

//            float4 vecB0trans = (float4) (vecB0.s0, vecB1.s0, vecB2.s0, vecB3.s0);
//            float4 vecB1trans = (float4) (vecB0.s1, vecB1.s1, vecB2.s1, vecB3.s1);
//            float4 vecB2trans = (float4) (vecB0.s2, vecB1.s2, vecB2.s2, vecB3.s2);
//            float4 vecB3trans = (float4) (vecB0.s3, vecB1.s3, vecB2.s3, vecB3.s3);

            acc.s0 += dot(vecA, vecB0);
            acc.s1 += dot(vecA, vecB1);
            acc.s2 += dot(vecA, vecB2);
            acc.s3 += dot(vecA, vecB3);


#endif

        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // Store the final results in C
    int index = globalCol + globalRow * (N / WIDTH);
//    acc = (float4)(transRow, transCol, globalTransRow, globalTransCol);
//    acc = (float4) (row, col, transRow, transCol);
//    acc = Bsub[col][row];
    C[globalCol + globalRow*(N/WIDTH)] = acc; //transed
}


__kernel void gemm4_transA(const int M, const int N, const int K,
                    const __global floatX* A,
                    const __global floatX* B,
#ifdef BIAS
                    const __global floatX* bias,
#endif
                    __global floatX* C,
                    const int numTiles) {

    // Thread identifiers
    const int row = get_local_id(0); //max: TS4 // Local row ID (max: TS4/WIDTH)
    const int col = get_local_id(1); //max: TS4/WIDTH // Local col ID (max: TS4)
    const int globalRow = (TS4)*get_group_id(0) + row; // 0..M// 0..M/WIDTH
    const int globalCol = (TS4 / WIDTH)*get_group_id(1) + col; // 0..N/WIDTH // 0..N
    const int globalCol2 = (TS4 / WIDTH) * get_group_id(0) + col;

//    if (globalRow == 0 && globalCol == 0) {
//        printf("(%d, %d)", get_global_size(0))
//    }

    // Local memory to fit a tile of TS4*TS4 elements of A and B
    //transed
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
#endif

#ifdef BIAS
    acc = bias[globalCol];
#endif

    // Loop over all tiles
//    const int numTiles = K/TS4;
    for (int t=0; t<numTiles; t++) {

        // Load one tile of A and B into local memory
        //transed
        const int tiledRow = (TS4)*t + row;
        const int tiledCol = (TS4/WIDTH)*t + col;
//        const int Ai = tiledCol * (M/WIDTH) + globalRow;
//        const int Bi = globalCol * (K/WIDTH) + tiledRow;
//        if (tiledCol > K / WIDTH)
//            Asub[col][row] = float0;
//        else
            Asub[row][col] = A[globalCol2 + tiledRow*(M/WIDTH)]; //transed

//        if (tiledRow > K)
//            Bsub[col][row] = float0;
//        else
            Bsub[row][col] = B[globalCol + tiledRow*(N/WIDTH)]; // transed


        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
        // Perform the computation for a single tile

        floatX vecA;
        float valB;
        for (int k=0; k<TS4/WIDTH; k++) {
            switch (row % WIDTH) {
                case 0:
                    vecA.s0 = Asub[WIDTH*k + 0][row / WIDTH].s0; // transed
                    vecA.s1 = Asub[WIDTH*k + 1][row / WIDTH].s0;
                    vecA.s2 = Asub[WIDTH*k + 2][row / WIDTH].s0;
                    vecA.s3 = Asub[WIDTH*k + 3][row / WIDTH].s0;
                    break;
                case 1:
                    vecA.s0 = Asub[WIDTH*k + 0][row / WIDTH].s1; // transed
                    vecA.s1 = Asub[WIDTH*k + 1][row / WIDTH].s1;
                    vecA.s2 = Asub[WIDTH*k + 2][row / WIDTH].s1;
                    vecA.s3 = Asub[WIDTH*k + 3][row / WIDTH].s1;
                    break;
                case 2:
                    vecA.s0 = Asub[WIDTH*k + 0][row / WIDTH].s2; // transed
                    vecA.s1 = Asub[WIDTH*k + 1][row / WIDTH].s2;
                    vecA.s2 = Asub[WIDTH*k + 2][row / WIDTH].s2;
                    vecA.s3 = Asub[WIDTH*k + 3][row / WIDTH].s2;
                    break;
                case 3:
                    vecA.s0 = Asub[WIDTH*k + 0][row / WIDTH].s3; // transed
                    vecA.s1 = Asub[WIDTH*k + 1][row / WIDTH].s3;
                    vecA.s2 = Asub[WIDTH*k + 2][row / WIDTH].s3;
                    vecA.s3 = Asub[WIDTH*k + 3][row / WIDTH].s3;
                    break;
            }
#if WIDTH == 1

            float vecA = Bsub[col][k];
            acc += vecB * vecA;

#elif WIDTH == 2

            float4 vecA0 = Bsub[col][WIDTH*k + 0];
            float4 vecA1 = Bsub[col][WIDTH*k + 1];

            float2 vecA0trans = (float2) (vecA0.s0, vecA1.s0);
            float2 vecA1trans = (float2) (vecA0.s1, vecA1.s1);

            acc.s0 += dot(vecA, vecA0trans);
            acc.s1 += dot(vecA, vecA1trans);

#elif WIDTH == 4

            //transed
            float4 vecB0 = Bsub[WIDTH*k + 0][col];
            float4 vecB1 = Bsub[WIDTH*k + 1][col];
            float4 vecB2 = Bsub[WIDTH*k + 2][col];
            float4 vecB3 = Bsub[WIDTH*k + 3][col];

            float4 vecB0trans = (float4) (vecB0.s0, vecB1.s0, vecB2.s0, vecB3.s0);
            float4 vecB1trans = (float4) (vecB0.s1, vecB1.s1, vecB2.s1, vecB3.s1);
            float4 vecB2trans = (float4) (vecB0.s2, vecB1.s2, vecB2.s2, vecB3.s2);
            float4 vecB3trans = (float4) (vecB0.s3, vecB1.s3, vecB2.s3, vecB3.s3);

            acc.s0 += dot(vecA, vecB0trans);
            acc.s1 += dot(vecA, vecB1trans);
            acc.s2 += dot(vecA, vecB2trans);
            acc.s3 += dot(vecA, vecB3trans);


#endif

        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // Store the final results in C
    int index = globalCol + globalRow * (N / WIDTH);
//    acc = Asub[col][row];
    C[globalCol + globalRow*(N/WIDTH)] = acc; //transed
}




//// Use wider data types
//__kernel void gemm4_transA_transB(const int M, const int N, const int K,
//                    const __global floatX* A,
//                    const __global floatX* B,
//                    __global floatX* C,
//                    const int numTiles) {
//
//    // Thread identifiers
//    const int row = get_local_id(0); // Local row ID (max: TS4/WIDTH)
//    const int col = get_local_id(1); // Local col ID (max: TS4)
//    const int globalRow = (TS4/WIDTH)*get_group_id(0) + row; // 0..M/WIDTH
//    const int globalCol = TS4*get_group_id(1) + col; // 0..N
//
//    // Local memory to fit a tile of TS4*TS4 elements of A and B
//    __local floatX Asub[TS4][TS4/WIDTH];
//    __local floatX Bsub[TS4][TS4/WIDTH];
//
//    // Initialise the accumulation registers
//#if WIDTH == 1
//    floatX acc = 0.0f;
//    floatX float0 = 0.0f;
//#elif WIDTH == 2
//    floatX acc = { 0.0f, 0.0f };
//    floatX float0 = (float2) (0, 0);
//#elif WIDTH == 4
//    floatX acc = { 0.0f, 0.0f, 0.0f, 0.0f };
//    floatX float0 = (float4) (0, 0, 0, 0);
//#elif WIDTH == 8
//    floatX acc = (float8) (0, 0, 0, 0, 0, 0, 0, 0);
//    floatX float0 = (float8) (0, 0, 0, 0, 0, 0, 0, 0);
//#elif WIDTH == 16
//    floatX acc = (float16) (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
//    floatX float0 = (float16) (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
//#endif
//
//    // Loop over all tiles
////    const int numTiles = K/TS4;
//    for (int t=0; t<numTiles; t++) {
//
//        // Load one tile of A and B into local memory
//        const int tiledRow = (TS4/WIDTH)*t + row;
//        const int tiledCol = TS4*t + col;
//            Bsub[col][row] = B[globalCol*(K/WIDTH) + tiledRow];
//            Asub[col][row] = A[tiledCol*(M/WIDTH) + globalRow];
//
//        // Synchronise to make sure the tile is loaded
//        barrier(CLK_LOCAL_MEM_FENCE);
//        // Perform the computation for a single tile
//
//        floatX vecA, vecB;
//        float valB;
//        for (int k=0; k<TS4/WIDTH; k++) {
//            vecB = Bsub[col][k];
//#if WIDTH == 1
//
//            float vecA = Asub[k][row];
//            acc += vecB * vecA;
//
//#elif WIDTH == 2
//
//            float4 vecA0 = Asub[WIDTH*k + 0][row];
//            float4 vecA1 = Asub[WIDTH*k + 1][row];
//
//            float2 vecA0trans = (float2) (vecA0.s0, vecA1.s0);
//            float2 vecA1trans = (float2) (vecA0.s1, vecA1.s1);
//
//            acc.s0 += dot(vecB, vecA0trans);
//            acc.s1 += dot(vecB, vecA1trans);
//
//#elif WIDTH == 4
//
//            float4 vecA0 = Asub[WIDTH*k + 0][row];
//            float4 vecA1 = Asub[WIDTH*k + 1][row];
//            float4 vecA2 = Asub[WIDTH*k + 2][row];
//            float4 vecA3 = Asub[WIDTH*k + 3][row];
//
//            float4 vecA0trans = (float4) (vecA0.s0, vecA1.s0, vecA2.s0, vecA3.s0);
//            float4 vecA1trans = (float4) (vecA0.s1, vecA1.s1, vecA2.s1, vecA3.s1);
//            float4 vecA2trans = (float4) (vecA0.s2, vecA1.s2, vecA2.s2, vecA3.s2);
//            float4 vecA3trans = (float4) (vecA0.s3, vecA1.s3, vecA2.s3, vecA3.s3);
//
//            acc.s0 += dot(vecB, vecA0trans);
//            acc.s1 += dot(vecB, vecA1trans);
//            acc.s2 += dot(vecB, vecA2trans);
//            acc.s3 += dot(vecB, vecA3trans);
//#endif
//
//        }
//
//        // Synchronise before loading the next tile
//        barrier(CLK_LOCAL_MEM_FENCE);
//
////        Asub[col][row] = float0;
////        Bsub[col][row] = float0;
////        barrier(CLK_LOCAL_MEM_FENCE);
//    }
//
//
//
//    // Store the final results in C
//    C[globalCol * (M / WIDTH) + globalRow] = acc;
////    vstore4(acc, globalCol * (M/WIDTH) + globalRow, C);
//}







//__kernel void gemm4_transA(const int M, const int N, const int K,
//                    const __global floatX* A,
//                    const __global floatX* B,
//                    __global floatX* C) {
//
//    // Thread identifiers
//    const int row = get_local_id(0); //max: TS4 // Local row ID (max: TS4/WIDTH)
//    const int col = get_local_id(1); //max: TS4/WIDTH // Local col ID (max: TS4)
//    const int globalRow = (TS4)*get_group_id(0) + row; // 0..M// 0..M/WIDTH
//    const int globalCol = (TS4 / WIDTH)*get_group_id(1) + col; // 0..N/WIDTH // 0..N
//
//    // Local memory to fit a tile of TS4*TS4 elements of A and B
//    //transed
//    __local floatX Asub[TS4/WIDTH][TS4];
//    __local floatX Bsub[TS4/WIDTH][TS4];
//
//    // Initialise the accumulation registers
//#if WIDTH == 1
//    floatX acc = 0.0f;
//    floatX float0 = 0.0f;
//#elif WIDTH == 2
//    floatX acc = { 0.0f, 0.0f };
//    floatX float0 = (float2) (0, 0);
//#elif WIDTH == 4
//    floatX acc = { 0.0f, 0.0f, 0.0f, 0.0f };
//    floatX float0 = (float4) (0, 0, 0, 0);
//#elif WIDTH == 8
//    floatX acc = (float8) (0, 0, 0, 0, 0, 0, 0, 0);
//    floatX float0 = (float8) (0, 0, 0, 0, 0, 0, 0, 0);
//#elif WIDTH == 16
//    floatX acc = (float16) (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
//    floatX float0 = (float16) (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
//#endif
//
//    // Loop over all tiles
//    const int numTiles = K/TS4;
//    for (int t=0; t<numTiles; t++) {
//
//        // Load one tile of A and B into local memory
//        //transed
//        const int tiledRow = (TS4)*t + row;
//        const int tiledCol = (TS4/WIDTH)*t + col;
////        const int Ai = tiledCol * (M/WIDTH) + globalRow;
////        const int Bi = globalCol * (K/WIDTH) + tiledRow;
////        if (tiledCol > N)
////            Asub[col][row] = float0;
////        else
//            Asub[col][row] = A[tiledCol + globalRow*(K/WIDTH)]; //transed
//
////        if (tiledRow > M/WIDTH)
////            Bsub[col][row] = float0;
////        else
//            Bsub[col][row] = B[globalCol + tiledRow*(N/WIDTH)]; // transed
//
//
//        // Synchronise to make sure the tile is loaded
//        barrier(CLK_LOCAL_MEM_FENCE);
//        // Perform the computation for a single tile
//
//        floatX vecA;
//        float valB;
//        for (int k=0; k<TS4/WIDTH; k++) {
//            vecA = Asub[k][row]; // transed
//#if WIDTH == 1
//
//            float vecA = Bsub[col][k];
//            acc += vecB * vecA;
//
//#elif WIDTH == 2
//
//            float4 vecA0 = Bsub[col][WIDTH*k + 0];
//            float4 vecA1 = Bsub[col][WIDTH*k + 1];
//
////            float2 vecA0trans = (float2) (vecA0.s0, vecA1.s0);
////            float2 vecA1trans = (float2) (vecA0.s1, vecA1.s1);
//
//            acc.s0 += dot(vecA, vecA0);
//            acc.s1 += dot(vecA, vecA1);
//
//#elif WIDTH == 4
//
//            //transed
//            float4 vecB0 = Bsub[col][WIDTH*k + 0];
//            float4 vecB1 = Bsub[col][WIDTH*k + 1];
//            float4 vecB2 = Bsub[col][WIDTH*k + 2];
//            float4 vecB3 = Bsub[col][WIDTH*k + 3];
//
////            float4 vecB0trans = (float4) (vecB0.s0, vecB1.s0, vecB2.s0, vecB3.s0);
////            float4 vecB1trans = (float4) (vecB0.s1, vecB1.s1, vecB2.s1, vecB3.s1);
////            float4 vecB2trans = (float4) (vecB0.s2, vecB1.s2, vecB2.s2, vecB3.s2);
////            float4 vecB3trans = (float4) (vecB0.s3, vecB1.s3, vecB2.s3, vecB3.s3);
//
//            acc.s0 += dot(vecA, vecB0);
//            acc.s1 += dot(vecA, vecB1);
//            acc.s2 += dot(vecA, vecB2);
//            acc.s3 += dot(vecA, vecB3);
//#endif
//
//        }
//
//        // Synchronise before loading the next tile
//        barrier(CLK_LOCAL_MEM_FENCE);
//    }
//    // Store the final results in C
//    int index = globalCol + globalRow * (N / WIDTH);
//    C[globalCol + globalRow*(N/WIDTH)] = acc; //transed
//}
//
//

//
//// Use wider data types
//__kernel void gemm4_transB(const int M, const int N, const int K,
//                    const __global floatX* A,
//                    const __global floatX* B,
//                    __global floatX* C) {
//
//    // Thread identifiers
//    const int row = get_local_id(0); // Local row ID (max: TS4/WIDTH)
//    const int col = get_local_id(1); // Local col ID (max: TS4)
//    const int globalRow = (TS4/WIDTH)*get_group_id(0) + row; // 0..M/WIDTH
//    const int globalCol = TS4*get_group_id(1) + col; // 0..N
//
//    // Local memory to fit a tile of TS4*TS4 elements of A and B
//    __local floatX Asub[TS4][TS4/WIDTH];
//    __local floatX Bsub[TS4][TS4/WIDTH];
//
//    // Initialise the accumulation registers
//#if WIDTH == 1
//    floatX acc = 0.0f;
//    floatX float0 = 0.0f;
//#elif WIDTH == 2
//    floatX acc = { 0.0f, 0.0f };
//    floatX float0 = (float2) (0, 0);
//#elif WIDTH == 4
//    floatX acc = { 0.0f, 0.0f, 0.0f, 0.0f };
//    floatX float0 = (float4) (0, 0, 0, 0);
//#elif WIDTH == 8
//    floatX acc = (float8) (0, 0, 0, 0, 0, 0, 0, 0);
//    floatX float0 = (float8) (0, 0, 0, 0, 0, 0, 0, 0);
//#elif WIDTH == 16
//    floatX acc = (float16) (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
//    floatX float0 = (float16) (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
//#endif
//
//    // Loop over all tiles
//    const int numTiles = K/TS4;
//    for (int t=0; t<numTiles; t++) {
//
//        // Load one tile of A and B into local memory
//        const int tiledRow = (TS4/WIDTH)*t + row;
//        const int tiledCol = TS4*t + col;
////        const int Ai = tiledCol * (M/WIDTH) + globalRow;
////        const int Bi = globalCol * (K/WIDTH) + tiledRow;
////        if (tiledRow > M/WIDTH)
////            Bsub[col][row] = float0;
////        else
//            Bsub[col][row] = B[globalCol*(K/WIDTH) + tiledRow];
////        if (tiledCol > N)
////            Asub[col][row] = float0;
////        else
//            Asub[col][row] = A[tiledCol*(M/WIDTH) + globalRow];
//
//        // Synchronise to make sure the tile is loaded
//        barrier(CLK_LOCAL_MEM_FENCE);
//        // Perform the computation for a single tile
//
//        floatX vecA, vecB;
//        float valB;
//        for (int k=0; k<TS4/WIDTH; k++) {
//            vecB = Bsub[col][k];
//#if WIDTH == 1
//
//            float vecA = Asub[k][row];
//            acc += vecB * vecA;
//
//#elif WIDTH == 2
//
//            float4 vecA0 = Asub[WIDTH*k + 0][row];
//            float4 vecA1 = Asub[WIDTH*k + 1][row];
//
//            float2 vecA0trans = (float2) (vecA0.s0, vecA1.s0);
//            float2 vecA1trans = (float2) (vecA0.s1, vecA1.s1);
//
//            acc.s0 += dot(vecB, vecA0trans);
//            acc.s1 += dot(vecB, vecA1trans);
//
//#elif WIDTH == 4
//
//            float4 vecA0 = Asub[WIDTH*k + 0][row];
//            float4 vecA1 = Asub[WIDTH*k + 1][row];
//            float4 vecA2 = Asub[WIDTH*k + 2][row];
//            float4 vecA3 = Asub[WIDTH*k + 3][row];
//
////            float4 vecA0trans = (float4) (vecA0.s0, vecA1.s0, vecA2.s0, vecA3.s0);
////            float4 vecA1trans = (float4) (vecA0.s1, vecA1.s1, vecA2.s1, vecA3.s1);
////            float4 vecA2trans = (float4) (vecA0.s2, vecA1.s2, vecA2.s2, vecA3.s2);
////            float4 vecA3trans = (float4) (vecA0.s3, vecA1.s3, vecA2.s3, vecA3.s3);
//
//            acc.s0 += dot(vecB, vecA0);
//            acc.s1 += dot(vecB, vecA1);
//            acc.s2 += dot(vecB, vecA2);
//            acc.s3 += dot(vecB, vecA3);
//
//
//#endif
//
//        }
//
//        // Synchronise before loading the next tile
//        barrier(CLK_LOCAL_MEM_FENCE);
//    }
//
//    C[globalCol*(M/WIDTH) + globalRow] = acc;
//}




#elif KERNEL == 5

//#ifndef TSM                // The tile-size in dimension M
//#define TSM 64
//#define TSN 64                 // The tile-size in dimension N
//#define TSK 16                 // The tile-size in dimension K
//#define WPTN 8                 // The work-per-thread in dimension N
//#define WPTM 1
//#endif
#define RTSN (TSN/WPTN)        // The reduced tile-size in dimension N
//#define RTSM (TSM/WPTM)
#define LPT ((TSK)/(RTSN)) // The loads-per-thread for a tile

// Pre-transpose the input matrix B and use rectangular tiles
__kernel void gemm5(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C,
                      const int numTiles) {

#ifndef PROFILING
    if (get_global_id(0) == 0 && get_global_id(1) == 0)
        printf("5 || TS: %d | TSK: %d | WPT: %d\n", TSM, TSK, WPTN);
#endif

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

//#ifndef TSM
//#define TSM 32                // The tile-size in dimension M
//#define TSN TSM                // The tile-size in dimension N
//#define TSK 16                 // The tile-size in dimension K
//#define WPTM 4                 // The work-per-thread in dimension M
//#define WPTN WPTM                 // The work-per-thread in dimension N
//#endif
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

#ifndef PROFILING
    if (get_global_id(0) == 0 && get_global_id(1) == 0)
        printf("6 || TS: %d | TSK: %d | WPT: %d", TSM, TSK, WPTM);
#endif
    // Thread identifiers
    const int tidm = get_local_id(0); // Local row ID (max: TSM/WPTM)
    const int tidn = get_local_id(1); // Local col ID (max: TSN/WPTN)
    const int offsetM = TSM*get_group_id(0); // Work-group offset
    const int offsetN = TSN*get_group_id(1); // Work-group offset

//    if (get_global_id(0) == 0 && get_global_id(1) == 0) {
//        printf("6\nTSM: %d || WPTM: %d || RTSM: %d\n", TSM, WPTM, RTSM);
//        printf("TSN: %d || WPTN: %d || RTSN: %d", TSN, WPTN, RTSN);
//    }

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

#elif KERNEL == 7

//#ifndef WIDTH
//#define WIDTH 4
//#endif

#if WIDTH == 1
typedef float floatX;
#elif WIDTH == 2
typedef float2 floatX;
#elif WIDTH == 4
typedef float4 floatX;
#endif


//#ifndef TSM
//#define TSM 32
//#define TSN TSM                // The tile-size in dimension N
//#define TSK 32
//#define WPTM 4
//#define WPTN WPTM                 // The work-per-thread in dimension N
//#endif
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


#ifndef PROFILING
    if (get_global_id(0) == 0 && get_global_id(1) == 0)
        printf("*7* || TS: %d | TSK: %d | WPT: %d | WIDTH: %d", TSM, TSK, WPTM, WIDTH);
#endif

//    if (get_global_id(0) == 0 && get_global_id(1) == 0) {
//        printf("7\nTSM: %d || WPTM: %d || RTSM: %d\n", TSM, WPTM, RTSM);
//        printf("TSN: %d || WPTN: %d || RTSN: %d\n", TSN, WPTN, RTSN);
//        printf("LPTA: %d || LPTB: %d", LPTA, LPTB);
//    }

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

#elif KERNEL == 8

//#define WIDTH 4

#if WIDTH == 1
typedef float floatX;
#elif WIDTH == 2
typedef float2 floatX;
#elif WIDTH == 4
typedef float4 floatX;
#endif


//#define TSM 32                // The tile-size in dimension M
//#define TSN TSM                // The tile-size in dimension N
//#define TSK 16                 // The tile-size in dimension K
//#define WPTM 2                 // The work-per-thread in dimension M
//#define WPTN WPTM                 // The work-per-thread in dimension N

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

#ifndef PROFILING
    if (get_global_id(0) == 0 && get_global_id(1) == 0)
        printf("*7* || TS: %d | TSK: %d | WPT: %d | WIDTH: %d", TSM, TSK, WPTM, WIDTH);
#endif

//    if (get_global_id(0) == 0 && get_global_id(1) == 0) {
//        printf("8\nTSM: %d || WPTM: %d || RTSM: %d\n", TSM, WPTM, RTSM);
//        printf("TSN: %d || WPTN: %d || RTSN: %d\n", TSN, WPTN, RTSN);
//        printf("LPTA: %d || LPTB: %d", LPTA, LPTB);
//    }

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




