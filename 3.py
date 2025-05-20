from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from datetime import datetime, timedelta
from functools import lru_cache
import akshare as ak

from sklearn.preprocessing import RobustScaler  # 使用RobustScaler代替MinMaxScaler
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

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
        'MACD': 0.10,  # 中期趋势确认，中短期交易中价值适中
        'BBIBOLL': 0.15,  # 波动区间判断，维持权重
        'RSI': 0.15,  # 超买超卖，略降低但仍重要
        'KDJ': 0.20,  # 短期信号，增强权重（5-35日范围内非常有效）
        'SAR': 0.15,  # 趋势反转，在中短期交易中更为重要
        'CHIP': 0.10,  # 筹码分布，中短期内重要性略降
        'ICHIMOKU': 0.10,  # 一目均衡表，对中期趋势判断有帮助
        'OBV': 0.05,  # 能量潮，辅助判断，权重保持不变
    }


# data_fetcher.py
class DataFetcher:
    """数据获取类"""

    @staticmethod
    @lru_cache(maxsize=1024)
    def get_company_info(code: str) -> Dict[str, str]:
        """获取公司基本信息"""
        symbol = f"{code}"
        info_dict = {
            '股票简称': '未知',
            '行业': '未知',
            '上市时间': '未知',
            '发行价': '暂无数据',
            '分红次数': 0,
            '机构参与度': '暂无数据',
            '市场成本': '暂无数据'
        }  # 提前初始化默认值

        try:
            base_info = ak.stock_individual_info_em(symbol=symbol)
            # 添加检查，确保数据存在
            if not base_info.empty and '股票简称' in base_info['item'].values:
                info_dict['股票简称'] = base_info.loc[base_info['item'] == '股票简称', 'value'].values[0]
            if not base_info.empty and '行业' in base_info['item'].values:
                info_dict['行业'] = base_info.loc[base_info['item'] == '行业', 'value'].values[0]
            if not base_info.empty and '上市时间' in base_info['item'].values:
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


# 添加高级技术指标计算类
class AdvancedTechnicalIndicators(TechnicalIndicators):
    """高级技术指标计算类"""

    @staticmethod
    def calculate_ichimoku(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """计算一目均衡表"""
        # 转换线 (Conversion Line)
        df['转换线'] = (df['最高'].rolling(window=9).max() + df['最低'].rolling(window=9).min()) / 2

        # 基准线 (Base Line)
        df['基准线'] = (df['最高'].rolling(window=26).max() + df['最低'].rolling(window=26).min()) / 2

        # 先行带1 (Leading Span A)
        df['先行带A'] = ((df['转换线'] + df['基准线']) / 2).shift(26)

        # 先行带2 (Leading Span B)
        df['先行带B'] = ((df['最高'].rolling(window=52).max() + df['最低'].rolling(window=52).min()) / 2).shift(26)

        # 延迟线 (Lagging Span)
        df['延迟线'] = df['收盘'].shift(-26)

        # 判断信号
        latest = df.iloc[-1]
        if (latest['收盘'] > latest['先行带A'] and latest['收盘'] > latest['先行带B']):
            keypoint = "√"
        elif (latest['收盘'] < latest['先行带A'] and latest['收盘'] < latest['先行带B']):
            keypoint = "×"
        else:
            keypoint = "-"

        return df, keypoint

    @staticmethod
    def calculate_obv(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """计算OBV(能量潮)指标"""
        df['OBV'] = 0
        df.loc[0, 'OBV'] = df.loc[0, '成交量']

        for i in range(1, len(df)):
            if df.loc[df.index[i], '收盘'] > df.loc[df.index[i - 1], '收盘']:
                df.loc[df.index[i], 'OBV'] = df.loc[df.index[i - 1], 'OBV'] + df.loc[df.index[i], '成交量']
            elif df.loc[df.index[i], '收盘'] < df.loc[df.index[i - 1], '收盘']:
                df.loc[df.index[i], 'OBV'] = df.loc[df.index[i - 1], 'OBV'] - df.loc[df.index[i], '成交量']
            else:
                df.loc[df.index[i], 'OBV'] = df.loc[df.index[i - 1], 'OBV']

        # 计算OBV的移动平均线
        df['OBV_MA'] = df['OBV'].rolling(window=10).mean()

        # 判断信号
        if df['OBV'].iloc[-1] > df['OBV_MA'].iloc[-1] and df['OBV'].iloc[-2] <= df['OBV_MA'].iloc[-2]:
            keypoint = "√"
        elif df['OBV'].iloc[-1] < df['OBV_MA'].iloc[-1] and df['OBV'].iloc[-2] >= df['OBV_MA'].iloc[-2]:
            keypoint = "×"
        else:
            keypoint = "-"

        return df, keypoint

    @staticmethod
    def calculate_liquidity_indicators(df: pd.DataFrame) -> Dict[str, float]:
        """计算流动性指标"""
        # 计算成交量加权平均价格 (VWAP)
        df['VWAP'] = (df['成交量'] * df['收盘']).cumsum() / df['成交量'].cumsum()

        # 计算Amihud非流动性指标
        df['daily_return'] = df['收盘'].pct_change().abs()
        df['amihud'] = df['daily_return'] / (df['成交量'] * df['收盘'])

        # 计算买卖价差估计
        df['price_range'] = (df['最高'] - df['最低']) / df['收盘']

        liquidity_metrics = {
            'vwap': df['VWAP'].iloc[-1],
            'amihud': df['amihud'].iloc[-10:].mean(),  # 最近10天的平均值
            'price_range': df['price_range'].iloc[-10:].mean()
        }

        return liquidity_metrics


# 添加宏观经济数据获取类
class MacroDataFetcher:
    """宏观经济数据获取类"""

    @staticmethod
    @lru_cache(maxsize=128)
    def get_economic_indicators() -> Dict[str, float]:
        """获取宏观经济指标"""
        result = {}  # 初始化空结果
        try:
            # 获取GDP数据
            gdp_data = ak.macro_china_gdp()
            if not gdp_data.empty and 'GDP累计同比' in gdp_data.columns:
                result['gdp_yoy'] = float(gdp_data['GDP累计同比'].iloc[-1])
            else:
                result['gdp_yoy'] = 0.0

            # 获取CPI数据
            cpi_data = ak.macro_china_cpi_yearly()
            if not cpi_data.empty and '同比' in cpi_data.columns:
                result['cpi'] = float(cpi_data['同比'].iloc[-1])
            else:
                result['cpi'] = 0.0

            # 获取货币供应量
            money_supply = ak.macro_china_money_supply()
            if not money_supply.empty and '指标' in money_supply.columns and '数值' in money_supply.columns:
                m2_data = money_supply[money_supply['指标'] == 'M2同比']
                if not m2_data.empty:
                    result['m2_yoy'] = float(m2_data.iloc[-1]['数值'])
                else:
                    result['m2_yoy'] = 0.0
            else:
                result['m2_yoy'] = 0.0

            # 获取社会融资规模
            social_finance = ak.macro_china_shrzgm()
            if not social_finance.empty and '当月同比' in social_finance.columns:
                result['social_finance_yoy'] = float(social_finance['当月同比'].iloc[-1])
            else:
                result['social_finance_yoy'] = 0.0

        except Exception as e:
            print(f"宏观数据获取异常: {str(e)}")
            # 设置默认值
            result = {
                'gdp_yoy': 0.0,
                'cpi': 0.0,
                'm2_yoy': 0.0,
                'social_finance_yoy': 0.0
            }

        return result

    @staticmethod
    def get_industry_relation(industry: str) -> Dict[str, float]:
        """获取行业相关系数"""
        # 可以基于行业获取相关系数
        industry_betas = {
            '银行': 0.8,
            '保险': 0.9,
            '房地产': 1.2,
            '医药生物': 0.7,
            '计算机': 1.3,
            '电子': 1.4,
            # 可以添加更多行业
        }

        return {
            'market_beta': industry_betas.get(industry, 1.0),
            'cyclical_score': 0.5  # 周期性得分，可以进一步细化
        }


# 添加机器学习预测类
class MLPredictor:
    """机器学习预测类"""

    def __init__(self):
        self.models = {}
        self.features = [
            'RSI', 'DIF', 'DEA', 'MACD', 'K', 'D', 'J',
            '收盘', '最高', '最低', '开盘', '成交量', '成交额',
            'MA5', 'MA10', 'MA20', 'MA60'
        ]

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """准备特征数据"""
        feature_df = df.copy()

        try:
            # 计算移动平均线作为特征
            feature_df['MA5'] = feature_df['收盘'].rolling(window=5).mean()
            feature_df['MA10'] = feature_df['收盘'].rolling(window=10).mean()
            feature_df['MA20'] = feature_df['收盘'].rolling(window=20).mean()
            feature_df['MA60'] = feature_df['收盘'].rolling(window=60).mean()

            # 计算价格差异
            feature_df['收盘_开盘'] = feature_df['收盘'] - feature_df['开盘']
            feature_df['最高_最低'] = feature_df['最高'] - feature_df['最低']

            # 波动率
            feature_df['波动率'] = feature_df['收盘'].pct_change().rolling(window=10).std()

            # 只计算RSI存在时的相对值
            if 'RSI' in feature_df.columns:
                feature_df['RSI相对'] = feature_df['RSI'] / 50 - 1

            # 只计算MACD存在时的相对值
            if 'MACD' in feature_df.columns:
                # 检查MACD是否有足够的非NaN值
                if feature_df['MACD'].notna().sum() > 10:
                    feature_df['MACD相对'] = feature_df['MACD'].rolling(window=10).mean() / (
                            feature_df['MACD'].rolling(window=10).std() + 1e-10)  # 避免除以0

            # 成交量相对变化
            feature_df['成交量_变化'] = feature_df['成交量'].pct_change()
            feature_df['成交量_MA5'] = feature_df['成交量'].rolling(window=5).mean()
            feature_df['成交量_相对'] = feature_df['成交量'] / (feature_df['成交量_MA5'] + 1e-10)  # 避免除以0

            # 安全删除NaN值
            for col in feature_df.columns:
                if feature_df[col].isnull().all():
                    print(f"警告: 列 '{col}' 全部为空值，将被移除")
                    feature_df = feature_df.drop(columns=[col])
                    # 如果这个特征在self.features中，也要移除
                    if col in self.features:
                        self.features.remove(col)

            # 打印特征准备后的数据量
            print(f"特征准备完成，数据量: {len(feature_df.dropna())}行")

            return feature_df.dropna()
        except Exception as e:
            print(f"特征准备异常: {str(e)}")
            # 返回原始数据框，只移除完全为空的列
            for col in df.columns:
                if df[col].isnull().all():
                    df = df.drop(columns=[col])
            return df

    def train_model(self, df: pd.DataFrame, target_days: int = 5):
        """训练模型"""
        try:


            # 准备特征
            feature_df = self.prepare_features(df)

            # 创建目标变量：未来N天收盘价变化百分比
            feature_df['future_return'] = feature_df['收盘'].shift(-target_days) / feature_df['收盘'] - 1

            # 删除NaN值
            feature_df = feature_df.dropna()

            # 保证有足够的数据进行训练 - 降低要求
            if len(feature_df) < 50:  # 从100改为50
                print(f"数据不足，无法训练模型 (仅有{len(feature_df)}行)")
                # 设置一个简单的默认预测
                self.models = {
                    'default': {
                        'model': None,
                        'scaler': None,
                        'mse': 0,
                        'mean_return': feature_df['future_return'].mean() if not feature_df.empty else 0
                    }
                }
                return

            # 准备特征和目标
            X = feature_df[self.features]
            y = feature_df['future_return']

            # 特征标准化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # 分割训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

            # 训练多个模型
            models = {
                'rf': RandomForestRegressor(n_estimators=100, random_state=42),
                'gbm': GradientBoostingRegressor(n_estimators=100, random_state=42)
            }

            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                print(f"模型 {name} 的MSE: {mse}")
                self.models[name] = {
                    'model': model,
                    'scaler': scaler,
                    'mse': mse
                }
        except Exception as e:
            print(f"模型训练异常: {str(e)}")

    def predict(self, df: pd.DataFrame, target_days: int = 5) -> Dict[str, float]:
        """预测未来收益率"""
        if not self.models:
            print("模型尚未训练")
            return {}

        try:
            # 检查是否有默认预测
            if 'default' in self.models and self.models['default']['model'] is None:
                # 返回默认的平均收益率
                return {
                    'default': self.models['default']['mean_return'] if 'mean_return' in self.models['default'] else 0}

            # 准备特征
            feature_df = self.prepare_features(df)

            if feature_df.empty:
                print("特征准备后数据为空")
                return {'default': 0}

            # 确保所有需要的特征都存在
            available_features = [f for f in self.features if f in feature_df.columns]
            if len(available_features) < len(self.features) / 2:  # 如果可用特征不到一半
                print(f"可用特征不足: {len(available_features)}/{len(self.features)}")
                return {'default': 0}

            # 获取最新数据点
            latest_data = feature_df.iloc[-1:][available_features]

            predictions = {}

            for name, model_info in self.models.items():
                if name == 'default':
                    continue

                try:
                    # 标准化数据
                    if model_info['model'] is not None and model_info['scaler'] is not None:
                        X_scaled = model_info['scaler'].transform(latest_data)
                        # 预测
                        pred = model_info['model'].predict(X_scaled)[0]
                        predictions[name] = pred
                except Exception as e:
                    print(f"模型 {name} 预测异常: {str(e)}")

            # 综合多个模型的预测
            if predictions:
                ensemble_pred = sum(predictions.values()) / len(predictions)
                predictions['ensemble'] = ensemble_pred
            else:
                # 如果没有模型能成功预测，使用默认值
                predictions['ensemble'] = 0

            return predictions
        except Exception as e:
            print(f"预测总体异常: {str(e)}")
            return {'ensemble': 0}


# 添加深度学习预测类
class DeepLearningPredictor:
    """深度学习预测类"""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_list = None
        self.window_size = 20  # 使用过去20天的数据预测

    def create_model(self, input_shape):
        """创建LSTM模型"""
        try:


            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
            model.add(Dropout(0.2))
            model.add(LSTM(50, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(25, activation='relu'))  # 添加激活函数
            model.add(Dense(1))

            # 使用较小的学习率
            optimizer = Adam(learning_rate=0.001)
            model.compile(optimizer=optimizer, loss='mean_squared_error')

            return model
        except Exception as e:
            print(f"创建深度学习模型异常: {str(e)}")
            return None

    def prepare_sequences(self, df: pd.DataFrame, features: List[str]):
        """将数据转换为序列"""
        try:


            # 移除异常值
            # 对于每个特征，移除超出3个标准差的数据点
            clean_df = df.copy()
            for feature in features:
                if feature in clean_df.columns:
                    mean = clean_df[feature].mean()
                    std = clean_df[feature].std()
                    clean_df = clean_df[(clean_df[feature] > mean - 3 * std) &
                                        (clean_df[feature] < mean + 3 * std)]

            # 确保没有NaN值
            clean_df = clean_df.dropna(subset=features)

            # 检查数据是否足够
            if len(clean_df) < self.window_size + 10:
                print(f"警告: 清洗后数据不足({len(clean_df)}行)")
                return None, None, None

            # 使用RobustScaler，对异常值不敏感
            scaler = RobustScaler()
            scaled_data = scaler.fit_transform(clean_df[features])

            X, y = [], []
            for i in range(self.window_size, len(scaled_data)):
                X.append(scaled_data[i - self.window_size:i])
                # 预测下一天的收盘价
                y.append(scaled_data[i, features.index('收盘')])

            return np.array(X), np.array(y), scaler
        except Exception as e:
            print(f"准备深度学习数据异常: {str(e)}")
            return None, None, None

    def train_model(self, df: pd.DataFrame):
        """训练深度学习模型"""
        try:
            from sklearn.model_selection import train_test_split
            import numpy as np

            # 打印训练起点信息
            print(f"开始训练深度学习模型，数据量: {len(df)}行")

            # 确保数据充足
            if len(df) < 200:
                print("数据不足，无法训练深度学习模型")
                return

            # 确保所有必需特征存在
            features = ['开盘', '最高', '最低', '收盘', '成交量']
            optional_features = ['RSI', 'DIF', 'DEA', 'K', 'D']

            # 只使用存在的可选特征
            available_optional = [f for f in optional_features if f in df.columns]
            final_features = features + available_optional

            print(f"使用特征: {final_features}")

            # 准备数据
            X, y, scaler = self.prepare_sequences(df, final_features)

            if X is None or y is None:
                print("序列准备失败，退出训练")
                return

            print(f"准备好的序列数据形状: X={X.shape}, y={y.shape}")

            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            # 创建并训练模型
            self.model = self.create_model((X.shape[1], X.shape[2]))

            if self.model is None:
                print("模型创建失败")
                return

            # 使用较少的 epoch
            self.model.fit(
                X_train, y_train,
                batch_size=min(32, len(X_train)),  # 确保batch_size不大于样本数
                epochs=min(20, max(5, int(len(X_train) / 10))),  # 根据样本量动态调整epochs
                validation_data=(X_test, y_test),
                verbose=1
            )

            self.feature_list = final_features
            self.scaler = scaler

        except Exception as e:
            print(f"训练深度学习模型异常: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def predict_next_day(self, df: pd.DataFrame) -> float:
        """预测下一个交易日的收盘价"""
        if self.model is None or self.scaler is None:
            print("模型尚未训练")
            return None

        try:
            # 确保所有必要的特征都存在
            for feature in self.feature_list:
                if feature not in df.columns:
                    print(f"缺少必要特征: {feature}")
                    return None

            # 获取最近的数据
            if len(df) < self.window_size:
                print(f"数据不足，需要至少{self.window_size}行数据")
                return None

            recent_data = df[self.feature_list].iloc[-self.window_size:]

            # 检查数据是否有NaN
            if recent_data.isnull().values.any():
                print("输入数据含有NaN值")
                recent_data = recent_data.fillna(method='ffill').fillna(method='bfill')

            # 标准化
            scaled_data = self.scaler.transform(recent_data)

            # 创建输入序列
            X_test = np.array([scaled_data])

            # 预测
            scaled_prediction = self.model.predict(X_test)

            # 反标准化获取实际价格
            dummy_array = np.zeros((1, len(self.feature_list)))
            dummy_array[0, self.feature_list.index('收盘')] = scaled_prediction[0, 0]
            predicted_price = self.scaler.inverse_transform(dummy_array)[0, self.feature_list.index('收盘')]

            # 验证预测结果是否合理
            latest_price = df['收盘'].iloc[-1]
            if abs(predicted_price / latest_price - 1) > 0.1:
                print(f"预测值异常，超出当前价格10%: {predicted_price}")
                return latest_price  # 返回最新价格作为预测值

            return predicted_price
        except Exception as e:
            print(f"预测异常: {str(e)}")
            return None


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


# 高级股票分析类，整合了机器学习和深度学习
class AdvancedStockAnalyzer(StockAnalyzer):
    """高级股票分析类"""

    def __init__(self, code: str):
        super().__init__(code)
        self.advanced_indicators = AdvancedTechnicalIndicators()
        self.macro_fetcher = MacroDataFetcher()
        self.ml_predictor = MLPredictor()
        self.dl_predictor = DeepLearningPredictor()

        # 扩展分析结果字典
        self.analysis_results.update({
            'advanced_indicators': {},
            'macro_data': {},
            'ml_predictions': {},
            'dl_predictions': {}
        })

    def run_advanced_analysis(self) -> Dict[str, Any]:
        """运行高级分析"""
        # 首先运行基础分析
        self.run_analysis()

        # 计算高级技术指标
        self.history_data, ichimoku_keypoint = self.advanced_indicators.calculate_ichimoku(self.history_data)
        self.history_data, obv_keypoint = self.advanced_indicators.calculate_obv(self.history_data)
        liquidity_metrics = self.advanced_indicators.calculate_liquidity_indicators(self.history_data)

        # 添加到结果中
        self.results.update({
            'ICHIMOKU': ichimoku_keypoint,
            'OBV': obv_keypoint
        })

        # 获取宏观经济数据
        macro_data = self.macro_fetcher.get_economic_indicators()
        industry_relation = self.macro_fetcher.get_industry_relation(self.company_info['行业'])

        # 训练模型并获取预测
        try:
            self.ml_predictor.train_model(self.history_data)
            ml_predictions = self.ml_predictor.predict(self.history_data)

            # 只在数据足够的情况下训练深度学习模型
            if len(self.history_data) > 200:
                self.dl_predictor.train_model(self.history_data)
                dl_prediction = self.dl_predictor.predict_next_day(self.history_data)
            else:
                dl_prediction = None
        except Exception as e:
            print(f"模型训练异常: {str(e)}")
            ml_predictions = {}
            dl_prediction = None

        # 更新分析结果
        self.analysis_results.update({
            'advanced_indicators': {
                'ICHIMOKU': ichimoku_keypoint,
                'OBV': obv_keypoint,
                'liquidity': liquidity_metrics
            },
            'macro_data': {
                'economic': macro_data,
                'industry': industry_relation
            },
            'ml_predictions': ml_predictions,
            'dl_predictions': {'next_day': dl_prediction}
        })

        # 更新投资建议，考虑新增的指标
        self.update_recommendations()

        return self.analysis_results

    def update_recommendations(self):
        """更新投资建议，考虑高级指标和预测"""
        recommendations = self.analysis_results['recommendations'].copy()

        # 添加基于机器学习预测的建议
        ml_preds = self.analysis_results['ml_predictions']
        if ml_preds and 'ensemble' in ml_preds:
            pred_return = ml_preds['ensemble']
            if pred_return > 0.05:  # 预测5天后涨幅超过5%
                recommendations.append(f"机器学习模型预测5天后涨幅 {pred_return:.2%}，建议积极关注")
            elif pred_return < -0.03:  # 预测5天后跌幅超过3%
                recommendations.append(f"机器学习模型预测5天后跌幅 {abs(pred_return):.2%}，建议谨慎操作")

        # 添加基于深度学习的建议
        dl_pred = self.analysis_results['dl_predictions'].get('next_day')
        if dl_pred:
            current_price = self.history_data['收盘'].iloc[-1]
            change_rate = (dl_pred / current_price - 1)
            if change_rate > 0.03:  # 预测涨幅超过3%
                recommendations.append(f"深度学习模型预测明日涨幅 {change_rate:.2%}，建议关注")
            elif change_rate < -0.02:  # 预测跌幅超过2%
                recommendations.append(f"深度学习模型预测明日跌幅 {abs(change_rate):.2%}，建议注意风险")

        # 添加基于宏观数据的建议
        macro_data = self.analysis_results['macro_data']
        if 'economic' in macro_data and 'gdp_yoy' in macro_data['economic']:
            gdp_yoy = macro_data['economic']['gdp_yoy']
            if gdp_yoy < 5:  # GDP同比增长低于5%
                recommendations.append("宏观经济增速放缓，建议关注防御性板块")

        # 更新推荐列表
        self.analysis_results['recommendations'] = recommendations

    def print_advanced_report(self):
        """打印高级分析报告"""
        # 先打印基础报告
        self.print_analysis_report()

        print("\n========== 高级分析报告 ==========")

        # 打印高级技术指标
        print("高级技术指标：")
        adv_indicators = self.analysis_results.get('advanced_indicators', {})
        for k, v in adv_indicators.items():
            if k != 'liquidity':
                print(f"{k}: {v}")

        # 打印流动性指标
        if 'liquidity' in adv_indicators:
            print("\n流动性分析：")
            for k, v in adv_indicators['liquidity'].items():
                print(f"{k}: {v:.4f}")

        # 打印宏观数据
        print("\n宏观经济环境：")
        macro_data = self.analysis_results.get('macro_data', {})
        if 'economic' in macro_data:
            for k, v in macro_data['economic'].items():
                print(f"{k}: {v}")

        # 打印行业相关性
        if 'industry' in macro_data:
            print("\n行业特性：")
            for k, v in macro_data['industry'].items():
                print(f"{k}: {v:.2f}")

        # 打印机器学习预测
        print("\n机器学习预测（5天后收益率）：")
        ml_preds = self.analysis_results.get('ml_predictions', {})
        for model, pred in ml_preds.items():
            print(f"{model}: {pred:.2%}")

        # 打印深度学习预测
        dl_pred = self.analysis_results.get('dl_predictions', {}).get('next_day')
        if dl_pred:
            current_price = self.history_data['收盘'].iloc[-1]
            print(f"\n深度学习预测下一交易日价格: {dl_pred:.2f} (变化率: {(dl_pred / current_price - 1):.2%})")

        print("==================================")


def main():
    """主程序"""
    print("欢迎使用高级股票分析系统")
    while True:
        stock_code = input("请输入6位股票代码（输入q退出）: ")
        if stock_code.lower() == 'q':
            break

        try:
            # 询问是否使用高级分析
            use_advanced = input("是否使用高级分析模式？(y/n): ").lower() == 'y'

            if use_advanced:
                analyzer = AdvancedStockAnalyzer(stock_code)
                analyzer.run_advanced_analysis()
                analyzer.print_advanced_report()
            else:
                analyzer = StockAnalyzer(stock_code)
                analyzer.run_analysis()
                analyzer.print_analysis_report()
        except Exception as e:
            print(f"分析过程出现错误: {str(e)}")
            continue

    print("感谢使用高级股票分析系统！")


if __name__ == "__main__":
    main()

