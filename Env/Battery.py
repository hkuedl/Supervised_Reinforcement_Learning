import numpy as np

class BatteryStorage:
    def __init__(self, E_ess=100, SoC_max=1.0, SoC_min=0.2, P_max=100, standby_loss=0.0, detailed = True):
        """
        初始化电池储能系统
        :param E_ess: 电池容量 (kWh)
        :param SoC_max: 最大荷电状态 (默认1.0)
        :param SoC_min: 最小荷电状态 (默认0.2)
        :param P_max: 最大充放电功率 (kW)
        :param standby_loss: 待机损耗 (kW)
        """
        self.E_ess = E_ess  # 电池容量 (kWh)
        self.SoC_max = SoC_max
        self.SoC_min = SoC_min
        self.P_max = P_max
        self.standby_loss = standby_loss
        self.SoC = 0.0  # 初始荷电状态设为50%
        self.N_series = 215
        self.N_parallel = 130
        self.detailed = detailed
        
        # 等效电路模型参数
        self.a = [-0.852, 63.867, 3.6297, 0.559, 0.51, 0.508]
        self.b = [0.1463, 30.27, 0.1037, 0.0584, 0.1747, 0.1288]
        self.c = [0.1063, 62.49, 0.0437]
        self.d = [0.0712, 61.40, 0.0288]

    def update_SoC(self, P_e, delta_t=1.0):
        """
        更新电池的荷电状态 (SoC)
        :param P_e: 充放电功率 (kW), 正为放电，负为充电
        :param delta_t: 时间步长 (小时)        
        """
        
        if self.detailed:                  # 按分钟颗粒度进行SOC更新
            num_ep = delta_t * 60
            for i in range(num_ep):
                if P_e < 0:  # 充电
                    eta = self.calculate_charging_efficiency(self.SoC, P_e)
                    delta_SoC = (-P_e / 60 * eta) / self.E_ess 
                elif P_e > 0:  # 放电
                    eta = self.calculate_discharging_efficiency(self.SoC, P_e)
                    delta_SoC = (-P_e / 60 ) / (self.E_ess * eta)
                else:  # 待机
                    delta_SoC = (-self.standby_loss / 60) / self.E_ess
                new_SoC = self.SoC + delta_SoC
                self.SoC = np.clip(new_SoC, self.SoC_min, self.SoC_max)
        else:
            if delta_t <1:
                num_ep = 1
                time_left = delta_t 
                flag = 0
            else:
                num_ep = int(delta_t)
                time_left = delta_t - num_ep
                flag = 1
            # update SOC
            if flag:
                for i in range(num_ep):
                    duration = 1
                    if P_e < 0:  # 充电
                        eta = self.calculate_charging_efficiency(self.SoC, P_e)
                        delta_SoC = (-P_e * duration * eta) / self.E_ess 
                    elif P_e > 0:  # 放电
                        eta = self.calculate_discharging_efficiency(self.SoC, P_e)
                        delta_SoC = (-P_e * duration ) / (self.E_ess * eta)
                    else:  # 待机
                        delta_SoC = (-self.standby_loss * duration) / self.E_ess
                    new_SoC = self.SoC + delta_SoC
                    self.SoC = np.clip(new_SoC, self.SoC_min, self.SoC_max)
            # 对剩余不满1小时的时间进行计算
            duration = time_left
            if P_e < 0:  # 充电
                eta = self.calculate_charging_efficiency(self.SoC, P_e)
                delta_SoC = (-P_e * duration * eta) / self.E_ess 
            elif P_e > 0:  # 放电
                eta = self.calculate_discharging_efficiency(self.SoC, P_e)
                delta_SoC = (-P_e * duration ) / (self.E_ess * eta)
            else:  # 待机
                delta_SoC = (-self.standby_loss * duration) / self.E_ess
            new_SoC = self.SoC + delta_SoC
            # 确保SoC在合理范围内
            self.SoC = np.clip(new_SoC, self.SoC_min, self.SoC_max)
        
        return self.SoC


    def calculate_equivalent_circuit_params(self, SoC):
        """
        计算等效电路参数
        :param SoC: 当前荷电状态
        :return: Voc, Rs, Rts, Rtl, Rtot
        """
        # 计算开路电压Voc
        Voc = (self.a[0] * np.exp(-self.a[1] * SoC) + self.a[2] + self.a[3] * SoC - \
              self.a[4] * SoC**2 + self.a[5] * SoC**3)
        
        # 计算串联电阻Rs
        Rs = (self.b[0] * np.exp(-self.b[1] * SoC) + self.b[2] + self.b[3] * SoC - \
              self.b[4] * SoC**2 + self.b[5] * SoC**3)
        
        # 计算电荷转移电阻Rts
        Rts = self.c[0] * np.exp(-self.c[1] * SoC) + self.c[2]
        
        # 计算膜扩散电阻Rtl
        Rtl = self.d[0] * np.exp(-self.d[1] * SoC) + self.d[2]
        
        # 总电阻
        Rtot = (Rs + Rts + Rtl) 
        
        Voc_sys = Voc * self.N_series
        Rtot_sys = (Rtot * self.N_series) / self.N_parallel
        
        return Voc_sys, Rs, Rts, Rtl, Rtot_sys

    def calculate_current(self, P_e, SoC):
        """
        计算电路电流
        :param P_e: 充放电功率 (kW)
        :param SoC: 当前荷电状态
        :return: 电流 (A)
        """
        Voc, _, _, _, Rtot = self.calculate_equivalent_circuit_params(SoC)
        # 解二次方程 Pe = I(Voc - Rtot*I)
        # 转换为标准形式: Rtot*I^2 - Voc*I + Pe = 0
        a_coeff = Rtot
        b_coeff = -Voc
        c_coeff = P_e * 1000    #转换为瓦特单位
        
        discriminant = b_coeff**2 - 4 * a_coeff * c_coeff
        if discriminant < 0:
            raise ValueError("无实数解")
        
        # 选择物理上合理的解
        I = (-b_coeff - np.sqrt(discriminant)) / (2 * a_coeff)
        
        return I

    def calculate_charging_efficiency(self, SoC, P_e):
        """
        计算充电效率
        :param SoC: 当前荷电状态
        :param P_e: 充电功率 (kW, 应为负值)
        """
        if P_e >= 0:
            raise ValueError("充电功率应为负值")
        
        I = self.calculate_current(P_e, SoC)
        Voc, _, _, _, Rtot = self.calculate_equivalent_circuit_params(SoC)
        eta_ch = Voc / (Voc - Rtot * I)
        return eta_ch

    def calculate_discharging_efficiency(self, SoC, P_e):
        """
        计算放电效率
        :param SoC: 当前荷电状态
        :param P_e: 放电功率 (kW, 应为正值)
        """
        if P_e <= 0:
            raise ValueError("放电功率应为正值")
        
        I = self.calculate_current(P_e, SoC)
        Voc, _, _, _, Rtot = self.calculate_equivalent_circuit_params(SoC)
        eta_dis = (Voc - Rtot * I) / Voc
        return eta_dis