# 算法原理:
# 1.copy一份stable diffusion的encoder部分
# 2.微调encoder部分，原始的unet参数是完全不变的
# 3.通过zero convolution去加到原来decoder里面去
# 这里的zero convolution是一种特殊的参数初始化，零卷积会逐渐变成非零权重的常规卷积层。
