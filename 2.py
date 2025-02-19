# config.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from functools import lru_cache
import akshare as ak

# 显示设置
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 200)
pd.set_option('expand_frame_repr', False)


@dataclass
class Config:
    """配置类"""
    DEFAULT_DAYS: int = 365
    RSI_WINDOW: int = 14
    MACD_SHORT_WINDOW: int = 12
    MACD_LONG_WINDOW: int = 26
    MACD_SIGNAL_WINDOW: int = 9
    KDJ_WINDOW: int = 9
    BBIBOLL_WINDOW: int = 11
    SAR_STEP: float = 0.02
    SAR_MAX: float = 0.2
    CACHE_HOURS: int = 24


class IndicatorWeight:
    """指标权重配置"""
    WEIGHTS = {
        'MACD': 0.15,  # 长期趋势
        'BBIBOLL': 0.15,  # 中期走势
        'RSI': 0.2,  # 超买超卖
        'KDJ': 0.2,  # 短期信号
        'SAR': 0.1,  # 趋势反转
        'CHIP': 0.2  # 筹码分布
    }


# data_fetcher.py
class DataFetcher:
    """数据获取类"""

    @staticmethod
    @lru_cache(maxsize=1024)
    def get_company_info(code: str) -> Dict[str, str]:
        """获取公司基本信息"""
        symbol = f"{code}"
        info_dict = {}

        try:
            base_info = ak.stock_individual_info_em(symbol=symbol)
            info_dict['股票简称'] = base_info.loc[base_info['item'] == '股票简称', 'value'].values[0]
            info_dict['行业'] = base_info.loc[base_info['item'] == '行业', 'value'].values[0]
            info_dict['上市时间'] = base_info.loc[base_info['item'] == '上市时间', 'value'].values[0]

            stock_ipo_info_df = ak.stock_ipo_info(stock=symbol)
            if not stock_ipo_info_df.empty:
                info_dict['发行价'] = stock_ipo_info_df.loc[stock_ipo_info_df['item'] == '发行价(元)', 'value'].values[
                    0]

            stock_history_dividend_df = ak.stock_history_dividend()
            dividend_info = stock_history_dividend_df[stock_history_dividend_df['代码'] == code]
            info_dict['分红次数'] = dividend_info['分红次数'].iloc[0] if not dividend_info.empty else 0

            jg_info = ak.stock_comment_detail_zlkp_jgcyd_em(symbol=symbol)
            info_dict['机构参与度'] = f"{jg_info['机构参与度'].values[0]}%"

            cost_info = ak.stock_comment_detail_scrd_cost_em(symbol=symbol)
            info_dict['市场成本'] = f"{cost_info['市场成本'].values[0]}元"

        except Exception as e:
            print(f"数据获取异常: {str(e)}")
            info_dict.update({
                '股票简称': '未知',
                '行业': '未知',
                '上市时间': '未知',
                '发行价': '暂无数据',
                '分红次数': 0,
                '机构参与度': '暂无数据',
                '市场成本': '暂无数据'
            })

        return info_dict

    @staticmethod
    @lru_cache(maxsize=1024)
    def get_history_data(code: str, days: int = Config.DEFAULT_DAYS) -> pd.DataFrame:
        """获取历史数据"""
        symbol = f"{code}"
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days)).strftime("%Y%m%d")

        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )
        print(f"历史{days}天数据获取完成，共获取{len(df)}条记录")
        return df

    @staticmethod
    @lru_cache(maxsize=1024)
    def get_chip_distribution(code: str) -> Tuple[Dict[str, float], str, float]:
        """获取筹码分布"""
        symbol = f"{code}"
        df = ak.stock_cyq_em(symbol=symbol, adjust="qfq")
        latest_chip = df.iloc[-1].to_dict()

        print(f"最新交易日筹码分布：获利比例={latest_chip['获利比例'] * 100:.2f}% "
              f"70集中度={(latest_chip['70集中度'] * 100):.2f}%")

        if latest_chip['获利比例'] <= 0.2:
            keypoint = "√"
        elif 0.15 < latest_chip['获利比例'] <= 0.5:
            keypoint = "-"
        else:
            keypoint = "×"

        return latest_chip, keypoint, latest_chip['平均成本']


# technical_indicators.py
class TechnicalIndicators:
    """技术指标计算类"""

    @staticmethod
    def calculate_bbiboll(df: pd.DataFrame) -> Tuple[Dict[str, float], str, float]:
        """计算BBIBOLL指标"""
        df['MA3'] = df['收盘'].rolling(3).mean()
        df['MA6'] = df['收盘'].rolling(6).mean()
        df['MA12'] = df['收盘'].rolling(12).mean()
        df['MA24'] = df['收盘'].rolling(24).mean()
        df['BBIBOLL'] = (df['MA3'] + df['MA6'] + df['MA12'] + df['MA24']) / 4
        df['UPPER'] = df['BBIBOLL'] + 2 * df['BBIBOLL'].rolling(Config.BBIBOLL_WINDOW).std()
        df['LOWER'] = df['BBIBOLL'] - 2 * df['BBIBOLL'].rolling(Config.BBIBOLL_WINDOW).std()

        latest = df.iloc[-1]
        latest_price = latest['收盘']

        bbiboll_data = {
            'BBIBOLL': latest['BBIBOLL'],
            'UPPER': latest['UPPER'],
            'LOWER': latest['LOWER'],
            'latest_price': latest_price
        }

        if latest_price > bbiboll_data['UPPER']:
            keypoint = "×"
        elif latest_price < bbiboll_data['LOWER']:
            keypoint = "√"
        else:
            keypoint = "-"

        return bbiboll_data, keypoint, latest_price

    @staticmethod
    def calculate_rsi(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """计算RSI指标"""
        delta = df['收盘'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=Config.RSI_WINDOW).mean()
        avg_loss = loss.rolling(window=Config.RSI_WINDOW).mean()

        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        latest_rsi = df['RSI'].iloc[-1]

        if latest_rsi > 70:
            keypoint = "×"
        elif latest_rsi < 30:
            keypoint = "√"
        else:
            keypoint = "-"

        return df, keypoint

    @staticmethod
    def calculate_macd(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """计算MACD指标"""
        df['EMA_short'] = df['收盘'].ewm(span=Config.MACD_SHORT_WINDOW, adjust=False).mean()
        df['EMA_long'] = df['收盘'].ewm(span=Config.MACD_LONG_WINDOW, adjust=False).mean()
        df['DIF'] = df['EMA_short'] - df['EMA_long']
        df['DEA'] = df['DIF'].ewm(span=Config.MACD_SIGNAL_WINDOW, adjust=False).mean()
        df['MACD'] = df['DIF'] - df['DEA']

        latest_dif = df['DIF'].iloc[-1]
        latest_dea = df['DEA'].iloc[-1]

        keypoint = "√" if latest_dif > latest_dea else "×"
        return df, keypoint

    @staticmethod
    def calculate_kdj(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """计算KDJ指标"""
        df['RSV'] = ((df['收盘'] - df['最低'].rolling(Config.KDJ_WINDOW).min()) /
                     (df['最高'].rolling(Config.KDJ_WINDOW).max() -
                      df['最低'].rolling(Config.KDJ_WINDOW).min())) * 100

        df['RSV'] = df['RSV'].replace([np.inf, -np.inf], np.nan).fillna(0)
        df['K'] = 50.0
        df['D'] = 50.0

        for i in range(1, len(df)):
            df.loc[df.index[i], 'K'] = (2 / 3) * df.loc[df.index[i - 1], 'K'] + (1 / 3) * df.loc[df.index[i], 'RSV']
            df.loc[df.index[i], 'D'] = (2 / 3) * df.loc[df.index[i - 1], 'D'] + (1 / 3) * df.loc[df.index[i], 'K']

        df['J'] = 3 * df['K'] - 2 * df['D']

        latest_k = df['K'].iloc[-1]
        latest_d = df['D'].iloc[-1]

        if latest_k > latest_d and latest_k < 20:
            keypoint = "√"
        elif latest_k < latest_d and latest_k > 80:
            keypoint = "×"
        else:
            keypoint = "-"

        return df, keypoint

    @staticmethod
    def calculate_sar(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """计算SAR指标"""
        df['SAR'] = 0.0
        df['trend'] = 0

        trend = 1
        af = Config.SAR_STEP
        ep = df['最高'].iloc[0]
        sar = df['最低'].iloc[0]

        for i in range(1, len(df)):
            prev_sar = sar

            if trend == 1:
                sar = prev_sar + af * (ep - prev_sar)
                sar = min(sar, df['最低'].iloc[max(0, i - 2):i].min())

                if sar > df['最低'].iloc[i]:
                    trend = -1
                    sar = ep
                    ep = df['最低'].iloc[i]
                    af = Config.SAR_STEP
                else:
                    if df['最高'].iloc[i] > ep:
                        ep = df['最高'].iloc[i]
                        af = min(af + Config.SAR_STEP, Config.SAR_MAX)
            else:
                sar = prev_sar - af * (prev_sar - ep)
                sar = max(sar, df['最高'].iloc[max(0, i - 2):i].max())

                if sar < df['最高'].iloc[i]:
                    trend = 1
                    sar = ep
                    ep = df['最高'].iloc[i]
                    af = Config.SAR_STEP
                else:
                    if df['最低'].iloc[i] < ep:
                        ep = df['最低'].iloc[i]
                        af = min(af + Config.SAR_STEP, Config.SAR_MAX)

            df.loc[df.index[i], 'SAR'] = sar
            df.loc[df.index[i], 'trend'] = trend

        latest_trend = df['trend'].iloc[-1]
        latest_sar = df['SAR'].iloc[-1]
        latest_price = df['收盘'].iloc[-1]

        if latest_trend == 1 and latest_price > latest_sar:
            keypoint = "√"
        elif latest_trend == -1 and latest_price < latest_sar:
            keypoint = "×"
        else:
            keypoint = "-"

        return df, keypoint


# analysis.py
class StockAnalyzer:
    """股票分析类"""

    def __init__(self, code: str):
        self.code = code
        self.data_fetcher = DataFetcher()
        self.technical_indicators = TechnicalIndicators()
        self.company_info = self.data_fetcher.get_company_info(code)
        self.history_data = self.data_fetcher.get_history_data(code)
        self.results = {}
        self.analysis_results = {}
        self.risk_assessment = None
        self.market_context = None

    def analyze_volume(self) -> str:
        """分析交易量"""
        volume_data = self.history_data['成交量'].astype(float)
        volume_mean = volume_data.mean()
        volume_std = volume_data.std()

        print(f"\n交易量分析:")
        print(f"交易量均值: {volume_mean:.2f}")
        print(f"交易量标准差: {volume_std:.2f}")

        if volume_std < (0.5 * volume_mean):
            print("交易量分布较为均匀，可能存在量化资金的控制。")
            return "警惕"
        else:
            print("交易量分布不均匀。")
            return "观望"

    def analyze_price_fluctuation(self) -> str:
        """分析价格波动"""
        self.history_data['日收益率'] = self.history_data['收盘'].pct_change()
        volatility = self.history_data['日收益率'].std()

        print(f"\n价格波动分析:")
        print(f"价格波动率（标准差）: {volatility:.2%}")

        if volatility < 0.015:
            print("价格波动较为规律，可能存在量化资金的控制。")
            return "警惕"
        else:
            print("价格波动较大。")
            return "观望"

    def analyze_market_context(self) -> Dict[str, Any]:
        """分析市场环境"""
        try:
            # 获取上证指数数据作为市场参考
            market_data = self.data_fetcher.get_history_data('000001')
            market_trend = market_data['收盘'].pct_change().mean()
            market_volatility = market_data['收盘'].pct_change().std()

            # 获取行业数据
            industry = self.company_info['行业']
            # 这里可以添加行业指数的分析

            self.market_context = {
                'market_trend': market_trend,
                'market_volatility': market_volatility,
                'industry': industry,
            }
            return self.market_context
        except Exception as e:
            print(f"市场环境分析失败: {str(e)}")
            return {}

    def calculate_risk_metrics(self) -> Dict[str, float]:
        """计算风险指标"""
        df = self.history_data

        # 计算波动率
        volatility = df['收盘'].pct_change().std() * np.sqrt(252)

        # 计算最大回撤
        df['rolling_max'] = df['收盘'].rolling(window=252, min_periods=1).max()
        df['drawdown'] = df['收盘'] / df['rolling_max'] - 1
        max_drawdown = df['drawdown'].min()

        # 计算流动性指标（日均成交量）
        avg_volume = df['成交量'].mean()

        self.risk_assessment = {
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'avg_volume': avg_volume
        }

        return self.risk_assessment

    def analyze_time_frames(self) -> Dict[str, str]:
        """多时间维度分析"""
        df = self.history_data

        # 计算不同时间周期的趋势
        df['yearly_ma'] = df['收盘'].rolling(window=252).mean()
        df['monthly_ma'] = df['收盘'].rolling(window=21).mean()
        df['weekly_ma'] = df['收盘'].rolling(window=5).mean()

        latest = df.iloc[-1]

        time_frames = {
            'long_term': "√" if latest['收盘'] > latest['yearly_ma'] else "×",
            'medium_term': "√" if latest['收盘'] > latest['monthly_ma'] else "×",
            'short_term': "√" if latest['收盘'] > latest['weekly_ma'] else "×"
        }

        return time_frames

    def calculate_weighted_score(self) -> float:
        """计算加权评分"""
        score_map = {'√': 1.0, '-': 0.5, '×': 0.0}
        weighted_score = 0.0

        for indicator, value in self.results.items():
            if indicator in IndicatorWeight.WEIGHTS:
                weighted_score += score_map[value] * IndicatorWeight.WEIGHTS[indicator]

        return weighted_score

    def generate_recommendations(self) -> List[str]:
        """生成投资建议"""
        recommendations = []
        score = self.calculate_weighted_score()

        # 基于综合得分的建议
        if score >= 0.8:
            recommendations.append("各项指标表现优异，适合积极建仓（0.8-1.0）")
        elif score >= 0.6:
            recommendations.append("指标表现良好，可以考虑逐步建仓（0.6-0.8）")
        elif score >= 0.4:
            recommendations.append("指标表现一般，建议观望（0.4-0.6）")
        else:
            recommendations.append("指标表现不佳，建议保持谨慎（0.0-0.4）")

        # 基于风险评估的建议
        if self.risk_assessment:
            if self.risk_assessment['volatility'] > 0.4:
                recommendations.append("波动率较高，注意控制仓位（>0.4）")
            if self.risk_assessment['max_drawdown'] < -0.3:
                recommendations.append("历史最大回撤较大，建议设置止损（<-0.3）")

        # 基于市场环境的建议
        if self.market_context:
            if self.market_context['market_trend'] < -0.001:
                recommendations.append("大盘处于下跌趋势，建议谨慎操作")

        return recommendations

    def run_analysis(self) -> Dict[str, Any]:
        """运行完整分析"""
        # 获取基础数据
        _, chip_keypoint, mean_chip = self.data_fetcher.get_chip_distribution(self.code)
        _, bbiboll_keypoint, latest_price = self.technical_indicators.calculate_bbiboll(self.history_data)

        # 技术指标分析
        self.history_data, rsi_keypoint = self.technical_indicators.calculate_rsi(self.history_data)
        self.history_data, macd_keypoint = self.technical_indicators.calculate_macd(self.history_data)
        self.history_data, kdj_keypoint = self.technical_indicators.calculate_kdj(self.history_data)
        self.history_data, sar_keypoint = self.technical_indicators.calculate_sar(self.history_data)

        # 交易量和价格波动分析
        volume_keypoint = self.analyze_volume()
        price_keypoint = self.analyze_price_fluctuation()

        # 记录所有指标结果
        self.results = {
            'CHIP': chip_keypoint,
            'BBIBOLL': bbiboll_keypoint,
            'RSI': rsi_keypoint,
            'MACD': macd_keypoint,
            'KDJ': kdj_keypoint,
            'SAR': sar_keypoint,
            'VOL': volume_keypoint,
            'PRICE': price_keypoint
        }

        # 计算风险指标
        self.calculate_risk_metrics()

        # 分析市场环境
        self.analyze_market_context()

        # 分析不同时间维度
        time_frames = self.analyze_time_frames()

        # 生成建议
        recommendations = self.generate_recommendations()

        # 汇总分析结果
        self.analysis_results = {
            'code': self.code,
            'company_info': self.company_info,
            'latest_price': latest_price,
            'mean_chip': mean_chip,
            'indicators': self.results,
            'risk_assessment': self.risk_assessment,
            'market_context': self.market_context,
            'time_frames': time_frames,
            'weighted_score': self.calculate_weighted_score(),
            'recommendations': recommendations
        }

        return self.analysis_results

    def print_analysis_report(self):
        """打印分析报告"""
        if not self.analysis_results:
            print("请先运行分析 (run_analysis)")
            return

        print("\n========== 股票分析报告 ==========")
        print(f"股票代码：{self.code}")
        print(f"股票名称：{self.company_info['股票简称']}")
        print(f"所属行业：{self.company_info['行业']}")
        print(f"上市时间：{self.company_info['上市时间']}")
        print(f"发行价格：{self.company_info['发行价']}")
        print(f"分红次数：{self.company_info['分红次数']}")
        print(f"机构参与：{self.company_info['机构参与度']}")
        print(f"市场成本：{self.company_info['市场成本']}")
        print(f"\n当前价格：{self.analysis_results['latest_price']:.2f}")
        print(f"筹码成本：{self.analysis_results['mean_chip']:.2f}")

        print("------指标评级：------")
        for indicator, value in self.results.items():
            print(f"{indicator}: {value}")

        print(f"------综合得分：{self.analysis_results['weighted_score']:.2f}------")

        print("\n风险评估：")
        if self.risk_assessment:
            print(f"波动率：{self.risk_assessment['volatility']:.2%}")
            print(f"最大回撤：{self.risk_assessment['max_drawdown']:.2%}")
            print(f"日均成交量：{self.risk_assessment['avg_volume']:.0f}")

        print("------投资建议：------")
        for recommendation in self.analysis_results['recommendations']:
            print(f"- {recommendation}")

        print("==================================")


def main():
    """主程序"""
    print("欢迎使用股票分析系统")
    while True:
        stock_code = input("请输入6位股票代码（输入q退出）: ")
        if stock_code.lower() == 'q':
            break

        try:
            analyzer = StockAnalyzer(stock_code)
            analyzer.run_analysis()
            analyzer.print_analysis_report()
        except Exception as e:
            print(f"分析过程出现错误: {str(e)}")
            continue

    print("感谢使用股票分析系统！")


if __name__ == "__main__":
    main()