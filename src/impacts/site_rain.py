import numpy as np


class RainCalculator:
    """
    @class RainCalculator
    @brief 计算特定区域的降雨量。

    该类通过输入半径、台风强度和速度等参数，计算降雨量分布。
    """

    def __init__(self):
        """
        @brief 初始化 RainCalculator 类。
        """
        pass

    def calculate_rain(self, Rmw, radius, sector_angle, deltaP, dpdt, transspeed):
        """
        @brief 根据输入参数计算特定区域的降雨量。

        @param Rmw 台风最大风速半径（单位：km）。
        @param radius 从台风眼到观测点的半径（单位：km）。
        @param sector_angle 台风相对于观测点的方位角（单位：度）。
        @param deltaP 台风中心与外围的气压差（单位：hPa）。
        @param dpdt 台风气压变化率（单位：hPa/hour）。
        @param transspeed 台风的移动速度（单位：km/hour）。

        @return 返回计算得到的降雨量数组（单位：mm/hour）。
        """
        tempratio = Rmw / radius
        RR = np.zeros_like(radius)

        for i in range(len(tempratio)):
            if tempratio[i] > 1:
                RR[i] = -5.5 + 110 * tempratio[i] - 390 * tempratio[i] ** 2 + 550 * tempratio[i] ** 3 - 250 * tempratio[
                i] ** 4
                continue
            RR[i] = -5.5 + 110 * tempratio[i] - 390 * tempratio[i] ** 2 + 550 * tempratio[i] ** 3 - 250 * tempratio[
                i] ** 4

        k = max(1, 0.0319 * deltaP - 0.0395)
        k1 = max(1, 1 - dpdt / 100)

        sector_angle = np.where(sector_angle < 0, sector_angle + 360, sector_angle)
        rain_sector = np.floor(sector_angle / 45).astype(int)

        s = np.zeros_like(rain_sector, dtype=float)
        iffast = transspeed > 8
        ifslow = transspeed < 4

        for i in range(len(rain_sector)):
            if rain_sector[i] == 0:
                s[i] = 1.15 if iffast else (1.45 if ifslow else 1.0)
            elif rain_sector[i] == 1:
                s[i] = 1.15 if iffast else (1.05 if ifslow else 1.0)
            elif rain_sector[i] == 2:
                s[i] = 1.35 if iffast else (0.55 if ifslow else 1.0)
            elif rain_sector[i] == 3:
                s[i] = 1.35 if iffast else (0.65 if ifslow else 1.0)
            elif rain_sector[i] == 4:
                s[i] = 0.85
            elif rain_sector[i] == 5:
                s[i] = 0.65 if iffast else (0.95 if ifslow else 1.0)
            elif rain_sector[i] == 6:
                s[i] = 0.8 if iffast else (1.15 if ifslow else 1.0)
            elif rain_sector[i] == 7:
                s[i] = 0.95 if iffast else (1.35 if ifslow else 1.0)
            elif rain_sector[i] == 8:
                s[i] = 1.15 if iffast else (1.45 if ifslow else 1.0)

        rainfall = k * k1 * RR * s
        # 保证降雨量意义合理：非负，避免数值波动带来的负值
        return np.maximum(rainfall, 0.0)
