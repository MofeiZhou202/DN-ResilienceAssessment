import numpy as np
from scipy.stats import lognorm

from src.config import ORIGIN_LAT_LON, LEGACY_ORIGIN_LAT_LON
from src.impacts.equate_wind import WindEquator
from src.impacts.site_rain import RainCalculator
from src.impacts.site_wind import WindCalculator
from src.utils.coordinates import latlon_to_xy

class HurricaneImpactsOnTransmissionLines:
    """
    Simulate the impact of hurricanes on transmission lines.

    Attributes:
        tower_seg (list): Data structure for transmission lines and towers.
        hurricane (np.ndarray): Hurricane data.
        sample_hour (int): Number of hours to simulate.
    """

    def __init__(self, tower_seg, hurricane, sample_hour, verbose: bool = False):
        """
        Initialize the hurricane impact class.

        Args:
            tower_seg (list): Data structure for transmission lines and towers.
            hurricane (np.ndarray): Hurricane data.
            sample_hour (int): Number of hours to simulate.
        """
        self.tower_seg = tower_seg
        self.hurricane = hurricane
        self.sample_hour = sample_hour
        self.site_wind = WindCalculator()
        self.site_rain = RainCalculator()
        self.equator_wind = WindEquator()
        self.verbose = verbose
    def convert_hurricane_coordinates(self, latitude, longitude):
        """将经纬度转换为相对原点的平面坐标（km）。"""
        origin = ORIGIN_LAT_LON if longitude >= 0 else LEGACY_ORIGIN_LAT_LON
        return latlon_to_xy(latitude, longitude, origin)

    def calculate_impact(self):
        """
        Calculate the impact of hurricanes on transmission lines.

        Returns:
            list: Impact results for each transmission line, including failure probabilities.
        """
        results = []

        for line in self.tower_seg:
            # Extract tower and segment parameters
            tower_impactpara = line['tower']['impactpara']
            segment_impactpara = line['segment']['impactpara']

            ntower = tower_impactpara.shape[0]
            nseg = segment_impactpara.shape[0]

            towerLct = tower_impactpara[:, 0:2]
            dsgnWindTower = tower_impactpara[:, 2]
            collapswindTower = tower_impactpara[:, 3]
            mu_sigma_Tower = tower_impactpara[:, 6:20]

            segLct = segment_impactpara[:, 0:2]
            Seglength = segment_impactpara[:, 2]
            dsgnWindSeg = segment_impactpara[:, 3]
            dsgnRainSeg = segment_impactpara[:, 4]
            collapswindSeg = segment_impactpara[:, 5]
            aSeg = segment_impactpara[:, 8]
            bSeg = segment_impactpara[:, 9]
            cSeg = segment_impactpara[:, 10]

            # Initialize result matrices
            line['tower']['excollapsepwind'] = np.zeros((ntower, self.sample_hour), dtype=np.int8)
            line['tower']['failureprob'] = np.zeros((ntower, self.sample_hour), dtype=np.float64)
            line['tower']['wwi'] = np.ones((ntower, self.sample_hour), dtype=np.float64)
            line['tower']['windspeed'] = np.zeros((ntower, self.sample_hour), dtype=np.float64)

            line['segment']['excollapsepwind'] = np.zeros((nseg, self.sample_hour), dtype=np.int8)
            line['segment']['failureprob'] = np.zeros((nseg, self.sample_hour), dtype=np.float64)
            line['segment']['wwi'] = np.ones((nseg, self.sample_hour), dtype=np.float64)
            line['segment']['windspeed'] = np.zeros((nseg, self.sample_hour), dtype=np.float64)

            line['linefailprob'] = np.zeros((1, self.sample_hour), dtype=np.float64)
            line['linewwi'] = np.ones((1, self.sample_hour), dtype=np.float64)

            hurricane_xy_log = np.zeros((self.sample_hour, 2), dtype=np.float64)
            tower_wind_log = line['tower']['windspeed']

            lastsegfprob = np.zeros(nseg)
            lastsegupprob = np.ones(nseg)



            for j in range(self.sample_hour):
                dpdt = 1 if j == 0 else self.hurricane[j - 1, 3] - self.hurricane[j, 3]
                # 获取飓风位置并转换坐标
                hurricane_lat = self.hurricane[j, 0]  # 纬度
                hurricane_lon = self.hurricane[j, 1]  # 经度
                hurricane_x, hurricane_y = self.convert_hurricane_coordinates(hurricane_lat, hurricane_lon)
                if self.verbose:
                    print(f"[DEBUG] Line {line['lineid']} hour {j+1}: Hurricane XY=({hurricane_x:.2f}, {hurricane_y:.2f})")

                hurrLct = np.array([hurricane_x, hurricane_y])
                hureye_tower_vector = towerLct - hurrLct
                hureye_seg_vector = segLct - hurrLct

                dstTowerHrrcn = np.linalg.norm(hureye_tower_vector, axis=1)
                # 3.4.1 distance from each conductor segment to the hurricane center
                dstSegHrrcn = np.linalg.norm(hureye_seg_vector, axis=1)

                sector_angle_tower = np.arctan2(hureye_tower_vector[:, 1], hureye_tower_vector[:, 0]) * 180 / np.pi
                # 3.4.3 get sector_angle
                sector_angle_seg = np.zeros(nseg)
                yaxis_vector = np.array([0, 1])
                for k in range(nseg):
                    sector_angle_seg[k] = (np.arctan2(np.linalg.det([hureye_seg_vector[k, :], yaxis_vector]),
                                                      np.dot(hureye_seg_vector[k, :],
                                                             yaxis_vector)) * 180 / np.pi) % 360 + self.hurricane[j, 8]

                hurricane_xy_log[j] = [hurricane_x, hurricane_y]

                windTower = self.site_wind.calculate_wind(dstTowerHrrcn, self.hurricane[j, :])
                tower_wind_log[:, j] = windTower
                if self.verbose:
                    print(f"[DEBUG] Hour {j+1}: Tower wind speed range {windTower.min():.2f}~{windTower.max():.2f} m/s")
                # 3.4.2 wind at conductor segment locations  (10m height, 3 second)
                windSeg = self.site_wind.calculate_wind(dstSegHrrcn, self.hurricane[j, :])
                line['segment']['windspeed'][:, j] = windSeg

                rainTower = self.site_rain.calculate_rain(self.hurricane[j, 5], dstTowerHrrcn, sector_angle_tower,
                                                          self.hurricane[j, 3], dpdt, self.hurricane[j, 8])
                # 3.4.4 get the rainfall of conductor segment locations
                rainSeg = self.site_rain.calculate_rain(self.hurricane[j, 5], dstSegHrrcn, sector_angle_seg,
                                                        self.hurricane[j, 3], dpdt, self.hurricane[j, 8])

                equalwind = self.equator_wind.equate_wind(windTower, rainTower)

                exceed_collapse = equalwind > collapswindTower
                line['tower']['excollapsepwind'][exceed_collapse, j] = 1

                # 找出有风险的杆塔
                danger_towers = np.where(equalwind > dsgnWindTower)[0]

                for k in danger_towers:
                    # 计算风攻角所在的扇区
                    attack_sector = int(equalwind[k] / 7.5)

                    # 风攻角异常处理
                    if attack_sector > 12 or attack_sector < 0:
                        print("wrong wind attack angle, pls recheck!!!")
                        continue

                    # 根据风攻角扇区选择相应的 mu 和 sigma
                    if attack_sector < 1:  # 风攻角在 0 到 7.5 度之间 (mu0, sigma0)
                        mu = mu_sigma_Tower[k, 0]
                        sigma = mu_sigma_Tower[k, 1]
                    elif 1 <= attack_sector < 3:  # 风攻角在 7.5 到 22.5 度之间 (mu15, sigma15)
                        mu = mu_sigma_Tower[k, 2]
                        sigma = mu_sigma_Tower[k, 3]
                    elif 3 <= attack_sector < 5:  # 风攻角在 22.5 到 37.5 度之间 (mu30, sigma30)
                        mu = mu_sigma_Tower[k, 4]
                        sigma = mu_sigma_Tower[k, 5]
                    elif 11 <= attack_sector <= 12:  # 风攻角在 82.5 到 90 度之间 (mu90, sigma90)
                        mu = mu_sigma_Tower[k, 6]
                        sigma = mu_sigma_Tower[k, 7]
                    elif 9 <= attack_sector < 11:  # 风攻角在 67.5 到 82.5 度之间 (mu75, sigma75)
                        mu = mu_sigma_Tower[k, 8]
                        sigma = mu_sigma_Tower[k, 9]
                    elif 5 <= attack_sector < 7:  # 风攻角在 37.5 到 52.5 度之间 (mu45, sigma45)
                        mu = mu_sigma_Tower[k, 10]
                        sigma = mu_sigma_Tower[k, 11]
                    elif 7 <= attack_sector < 9:  # 风攻角在 52.5 到 67.5 度之间 (mu60, sigma60)
                        mu = mu_sigma_Tower[k, 12]
                        sigma = mu_sigma_Tower[k, 13]

                    # 计算对数正态分布的尺度参数
                    scale = np.exp(mu)

                    # 计算故障概率
                    fProbTower = lognorm.cdf(equalwind[k], s=sigma, loc=0, scale=scale)

                    # 记录故障概率
                    if fProbTower > 1e-8:
                        line['tower']['failureprob'][k, j] = fProbTower
                    else:
                        line['tower']['failureprob'][k, j] = 0

                    # 记录风荷载比（WWI）
                    line['tower']['wwi'][k, j] = equalwind[k] / dsgnWindTower[k]

                # 3.4.5 whether the wind at segment location exceeds the collapse wind
                exceed_collapse_seg = windSeg > collapswindSeg
                line['segment']['excollapsepwind'][exceed_collapse_seg, j] = 1

                # 3.4.6 segment failprob
                lambdaSeg = Seglength * np.exp(aSeg * windSeg / dsgnWindSeg + bSeg * rainSeg / dsgnRainSeg + cSeg)

                lambdaSeg = Seglength * lambdaSeg
                sampleinterval = 1
                tempfp = np.minimum(1 - np.exp(-lambdaSeg * sampleinterval), 1)
                fProbSeg = lastsegupprob * tempfp + lastsegfprob

                wind_exceed_design = windSeg > dsgnWindSeg
                line['segment']['failureprob'][wind_exceed_design, j] = fProbSeg[wind_exceed_design]

                lastsegfprob = line['segment']['failureprob'][:, j]
                lastsegupprob = 1 - lastsegfprob

                # 3.4.7 segment wwi hazard weght
                factorwind = windSeg / dsgnWindSeg
                line['segment']['wwi'][wind_exceed_design, j] *= factorwind[wind_exceed_design]

                rain_exceed_design = rainSeg > dsgnRainSeg
                factorrain = rainSeg / dsgnRainSeg
                line['segment']['wwi'][rain_exceed_design, j] *= factorrain[rain_exceed_design]

                alltowerupp = np.prod(1 - line['tower']['failureprob'][:, j].flatten())
                allsegupp = np.prod(1 - line['segment']['failureprob'][:, j].flatten())
                line['linefailprob'][0, j] = 1 - alltowerupp * allsegupp

                linewwi = np.prod(line['tower']['wwi'][:, j]) * np.prod(line['segment']['wwi'][:, j])
                line['linewwi'][0, j] = linewwi ** (1 / (ntower + nseg))


            results.append({
                'lineid': line['lineid'],
                'pairlineid': line['pairlineid'],
                'linefailprob': line['linefailprob'].flatten(),
                'linewwi': line['linewwi'].flatten(),
                'tower_failureprob': line['tower']['failureprob'],
                'segment_failureprob': line['segment']['failureprob'],
                'hurricane_xy': hurricane_xy_log,
                'tower_windspeed': tower_wind_log,
                'segment_windspeed': line['segment']['windspeed']
            })


        return results

