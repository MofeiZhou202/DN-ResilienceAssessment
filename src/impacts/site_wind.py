import numpy as np


class WindCalculator:
    """

    @brief 计算特定区域的风速。

    """

    def __init__(self):
        """
        @brief 初始化 WindCalculator 类。

        """
        pass

    def calculate_wind(self, radius, hurricanePara):
        """
        @brief 根据输入参数计算指定半径区域的风速。

        @param radius 距离台风中心的半径（单位：km）。
        @param hurricanePara 台风的参数，包括科里奥利力、气压差、Holland B参数、最大风速半径等。

        @return 返回计算得到的风速数组（单位：m/s）。
        """
        airRho = 1.15  # 空气密度 (kg/m^3)

        fc = hurricanePara[2]  # 科里奥利力
        deltap = hurricanePara[3]  # 气压差 (hPa)
        hollandB = hurricanePara[4]  # Holland B 参数
        Rwm = hurricanePara[5]  # 最大风速半径 (km)

        if Rwm > 50:  # 损坏半径
            damageRadius = 4 * Rwm
        else:
            damageRadius = 12.323 - 0.162 * Rwm

        wind10 = np.zeros_like(radius)

        tempa = (Rwm / radius) ** hollandB
        tempb = (1 / airRho) * np.exp(-tempa)
        tempc = fc * radius / 2

        tempd = hollandB * deltap * 100 * tempa * tempb + tempc ** 2

        windGradient = np.sqrt(tempd) - tempc

        # 垂直方向调整
        zheight = 10
        zgradient = 275
        alpha = 1 / 7
        wind10 = windGradient * (zheight / zgradient) ** alpha

        # 阵风因子调整
        gustf = 1.75
        wind10 = gustf * wind10
        wind10[wind10 < 0.01] = 0

        return wind10
