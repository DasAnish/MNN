loadline = "float16 vecA{i} = Asub[WIDTH*k + {j}][row];\n"
indexes = ['0123', '4567', '89ab', 'cdef']
range_16 = ['0', '1', '2', '3', '4', '5', '6', '7' , '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
range_= [i for i in range(16)]
transposeline = "float16 vecA{k}trans = (float16) ({vecs});\n"
vec = "vecA{i}.s{j}"
# indices = ["0_1_2_3", "4_5_6_7", "8_9_10_11", "12_13_14_15"]
# vecnames = "vecA{x1}.s{i}, vecA{x2}.s{i}, vecA{x3}.s{i}, vecA{x4}.s{i}"
# transposeline = "float4 vecA{k}trans_{index} = (float4) ({vecs});\n"
# vecBline = "float4 vecB_{i} = (float4) (vecB.s{x1}, vecB.s{x2}, vecB.s{x3}, vecB.s{x4});\n"
dotline = "acc.s{j} += dot(vecB.s{i}, vecA{j}trans.s{i});\n"

with open("write16.txt", "w") as f:
    f.write("\n//loading corresponding values of vecA\n")
    for i, j in zip(range_16, range_):
        f.write(loadline.format(i=i, j=j))

    f.write("\n//the transpose lines\n")
    for i in range_16:
        vec_ = ', '.join([vec.format(i=j, j=i) for j in range_16])
        f.write(transposeline.format(k=i, vecs=vec_))
        # for indice in indexes:
        #     # x1, x2, x3, x4 = indice.split("_")
        #     vec_ = ', '.join([vec.format(i=i, j=j) for j in indice])
        #     print(vec_)
        #     # vec = vecnames.format(i=i, x1=x1, x2=x2, x3=x3, x4=x4)
        #     f.write(transposeline.format(k=i, vecs=vec_))
        # f.write("\n")


    # f.write("\n//Splitting vecB into float4s\n")
    # for indice in indexes:
    #     x1, x2, x3, x4 = indice.split('_')
    #     vec = vecBline.format(i=indice, x1=x1, x2=x2, x3=x3, x4=x4)


    f.write("\n//Dot prods\n")
    for i in range_16:
        for indice in indexes:
            f.write(dotline.format(j=i, i=indice))
        f.write("\n")



        