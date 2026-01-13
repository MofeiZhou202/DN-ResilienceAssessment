import numpy as np


class WindEquator:
    """
    @class WindEquator
    @brief 计算等效风速，考虑了风速与降雨的关系。

    该类用于考虑降雨的影响来计算等效风速。
    """

    def __init__(self):
        """
        @brief 初始化WindEquator类。

        类的构造函数，初始化所需的参数。
        """
        pass

    def equate_wind(self, windspeed10, rainfall):
        """
        @brief 根据降雨量调整风速。

        @param windspeed10 输入的风速（m/s）。
        @param rainfall 输入的降水量（mm）。

        @return 返回调整后的等效风速（m/s）。
        """
        # Convert 3-sec wind speed to 10-min wind speed and back
        windspeed10 = windspeed10 / 1.42
        f1v10 = np.exp(0.006462 * windspeed10) - 1.2486 * np.exp(-0.2769 * windspeed10)
        f2R = 0.09376 * (rainfall ** 0.7087)

        equalwind = f1v10 * f2R + windspeed10
        equalwind = equalwind * 1.42

        equalwind[f1v10 < 0] = windspeed10[f1v10 < 0]

        return equalwind
