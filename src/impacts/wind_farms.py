import numpy as np
import pandas as pd
import os

from src.config import (
    ORIGIN_LAT_LON,
    LEGACY_ORIGIN_LAT_LON,
    infrastructure_data_path,
    hurricane_data_path,
)
from src.impacts.site_wind import WindCalculator
from src.utils.coordinates import latlon_to_xy

class HurricaneImpactOnWindFarms:
    """
    冰暴对风电场影响的模拟类。

    属性:
        wind_farm_location (ndarray): 风电场的位置坐标。
        ice_storm (ndarray): 冰暴的时间步长数据。
        T (int): 风暴的总时间步长。
    """

    def __init__(self, wind_farm_location, ice_storm, T):
        """
        初始化风暴影响模拟类。

        参数:
            wind_farm_location (DataFrame): 风电场位置数据。
            ice_storm (ndarray): 冰暴的时间步长数据。
            T (int): 风暴的总时间步长。
        """
        self.wind_farm_location = wind_farm_location.to_numpy()
        self.ice_storm = ice_storm
        self.T = T
        self.wind_calculator = WindCalculator()

    def _convert_coordinates(self):
        """
        由于输入数据已经是平面坐标系，此方法不需要进行转换。
        保留此方法是为了保持接口兼容性。
        """
        # 输入数据已经是平面坐标系，不需要转换
        pass

    def calculate_wind_speeds(self):
        """
        计算风电场在每个时间步长的风速。

        返回:
            ndarray: 风电场在每个时间步长的风速矩阵。
        """
        nWF = self.wind_farm_location.shape[0]
        wind_farm_speed_10 = np.zeros((nWF, self.T))

        for t in range(self.T):
            latitude = float(self.ice_storm[t, 0])
            longitude = float(self.ice_storm[t, 1])
            origin = ORIGIN_LAT_LON if longitude >= 0 else LEGACY_ORIGIN_LAT_LON
            hurrLct = latlon_to_xy(latitude, longitude, origin)
            hureye_tower_vector = self.wind_farm_location - hurrLct
            dstTowerHrrcn = np.linalg.norm(hureye_tower_vector, axis=1)
            wind_farm_speed_10[:, t] = self.wind_calculator.calculate_wind(dstTowerHrrcn, self.ice_storm[t, :])

        return wind_farm_speed_10

    def calculate_power_output(self, wind_farm_speed_10):
        """
        根据风速计算风电场的输出功率。

        参数:
            wind_farm_speed_10 (ndarray): 风电场的风速矩阵。

        返回:
            ndarray: 风电场的输出功率矩阵。
        """
        nWF = wind_farm_speed_10.shape[0]
        P_win = np.zeros((nWF, self.T))
        c = 4200 / (11.2 ** 2)  # 功率计算因子

        for i in range(nWF):
            for j in range(self.T):
                speed = wind_farm_speed_10[i, j]
                if speed <= 2.5 or speed >= 25:
                    P_win[i, j] = 0
                elif 2.5 < speed <= 11.2:
                    P_win[i, j] = c * speed ** 2 / 1000
                elif 11.2 < speed < 25:
                    P_win[i, j] = 4.2

        wind_farms_output = P_win / 4.2  # 归一化功率
        return wind_farms_output

    def simulate(self):
        """
        执行冰暴对风电场的影响模拟。

        返回:
            ndarray: 风电场归一化输出功率矩阵。
        """
        self._convert_coordinates()
        wind_farm_speed_10 = self.calculate_wind_speeds()
        wind_farms_output = self.calculate_power_output(wind_farm_speed_10)
        return wind_farms_output


# 调用示例
if __name__ == "__main__":
    # 加载风电场位置数据
    from src.config import infrastructure_data_path
    wind_farm_location = os.path.join(infrastructure_data_path)
    wind_farm_location_df = pd.read_excel(infrastructure_data_path + 'wind_farms.xlsx', sheet_name='wind_farm_location', header=None)

    # 加载冰暴数据，读取所有工作表
    hurricane_dfs = pd.read_excel(hurricane_data_path+'hurricane.xlsx', sheet_name=None)

    # 删除之前的 wind_farms_output.xlsx 文件
    output_file_path = infrastructure_data_path + 'wind_farms_output.xlsx'
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    with pd.ExcelWriter(output_file_path) as writer:
        for sheet_name, hurricane_df in hurricane_dfs.items():
            hurricane = hurricane_df.to_numpy()
            T = hurricane.shape[0]

            # 初始化模拟类
            simulator = HurricaneImpactOnWindFarms(wind_farm_location_df, hurricane, T)

            # 执行模拟
            wind_farms_output = simulator.simulate()

            # 保存结果到 Excel 的对应工作表
            output_df = pd.DataFrame(wind_farms_output,
                                     columns=[f'Time Step {j + 1}' for j in range(wind_farms_output.shape[1])])
            output_df.index = [f'Farm {i + 1}' for i in range(wind_farms_output.shape[0])]
            output_df.to_excel(writer, sheet_name=sheet_name)

    print("已保存为 wind_farms_output.xlsx 文件！")