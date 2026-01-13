import numpy as np


MAX_INIT_LATITUDE = 21.5


def hurricane_initialization():
    # Define mean and variance for translational speed
    meanTrans = 6
    varianceTrans = 2.1 ** 2
    # Calculate log-normal distribution parameters for translational speed
    muTrans = np.log((meanTrans ** 2) / np.sqrt(varianceTrans + meanTrans ** 2))
    sigmaTrans = np.sqrt(np.log(varianceTrans / (meanTrans ** 2) + 1))
    # Generate translational speed from log-normal distribution
    transSpeed = np.random.lognormal(muTrans, sigmaTrans)

    # Generate initial movement direction (theta) from normal distribution
    theta = np.random.normal(0, 4)

    # 使用正态分布生成初始位置，使其集中在南海-华南沿海附近
    while True:
        lat = np.random.normal(20.5, 1.5)
        if 17 <= lat <= MAX_INIT_LATITUDE:
            break

    while True:
        lng = np.random.normal(114.5, 2.0)
        if 110 <= lng <= 120:
            break

    # Define mean and variance for saturation vapor pressure deficit (deltaP)
    meandeltaP = 44  # 调整均值到区间中心
    variancedeltaP = 4 ** 2  # 减小方差以确保大部分值落在指定范围内
    # Calculate log-normal distribution parameters for deltaP
    mudeltaP = np.log((meandeltaP ** 2) / np.sqrt(variancedeltaP + meandeltaP ** 2))
    sigmadeltaP = np.sqrt(np.log(variancedeltaP / (meandeltaP ** 2) + 1))
    # Generate deltaP from log-normal distribution, ensuring it is within valid range
    deltaP = np.random.lognormal(mudeltaP, sigmadeltaP)
    while deltaP < 33 or deltaP > 55:
        deltaP = np.random.lognormal(mudeltaP, sigmadeltaP)

    # Define standard deviation for Rmw error
    singmaRmw = 0.39
    errorRmw = 100
    # Generate Rmw error from normal distribution within acceptable range
    while abs(errorRmw) >= 2 * singmaRmw:
        errorRmw = np.random.normal(0, singmaRmw)
    # Calculate Rmw using the generated error
    lnRmw = 3.859 - 7.7001e-5 * deltaP ** 2 + errorRmw
    Rmw = np.exp(lnRmw)

    # Generate IR within a specified range
    IR = np.random.uniform(0.45, 0.6)
    # Define hurricane occurrence date and time
    occurday = 231
    occurtime = 8
    # Compile all initial conditions into a list
    hurrintialcondition = [occurday, occurtime, lat, lng, deltaP, IR, Rmw, theta, transSpeed]

    return hurrintialcondition


__all__ = ["hurricane_initialization", "MAX_INIT_LATITUDE"]

