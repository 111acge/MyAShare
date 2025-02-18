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

    # 新增配置参数
    VOLUME_MA_WINDOW: int = 20
    INDUSTRY_COMPARE_WINDOW: int = 90
    SENTIMENT_WINDOW: int = 30
    ESG_UPDATE_DAYS: int = 180
    MACRO_UPDATE_DAYS: int = 30


class FactorWeight:
    """因子权重配置"""

    # 技术面因子权重
    TECHNICAL_WEIGHTS = {
        'MACD': 0.2,  # 长期趋势
        'BBIBOLL': 0.15,  # 中期走势
        'RSI': 0.15,  # 超买超卖
        'KDJ': 0.15,  # 短期信号
        'SAR': 0.1,  # 趋势反转
        'VOL': 0.15,  # 成交量
        'CHIP': 0.1  # 筹码分布
    }

    # 基本面因子权重
    FUNDAMENTAL_WEIGHTS = {
        'PE': 0.15,  # 市盈率
        'PB': 0.15,  # 市净率
        'ROE': 0.2,  # 净资产收益率
        'GROWTH': 0.2,  # 营收增长率
        'DEBT': 0.15,  # 资产负债率
        'CASH': 0.15  # 经营性现金流
    }

    # 资金流向因子权重
    MONEY_FLOW_WEIGHTS = {
        'BIG_MONEY': 0.3,  # 大单资金流向
        'NORTHBOUND': 0.3,  # 北向资金
        'INSTITUTIONAL': 0.2,  # 机构持股
        'MARGIN_TRADING': 0.2  # 融资融券
    }

    # 情绪因子权重
    SENTIMENT_WEIGHTS = {
        'IMPLIED_VOL': 0.3,  # 期权隐含波动率
        'FORUM_SENTIMENT': 0.2,  # 股吧情绪
        'TOP_TRADERS': 0.3,  # 龙虎榜
        'SHAREHOLDERS': 0.2  # 股东户数
    }

    # 行业因子权重
    INDUSTRY_WEIGHTS = {
        'PROSPERITY': 0.3,  # 行业景气度
        'CONCENTRATION': 0.2,  # 行业集中度
        'POLICY': 0.3,  # 政策敏感度
        'SUPPLY_CHAIN': 0.2  # 产业链地位
    }

    # 宏观因子权重
    MACRO_WEIGHTS = {
        'GDP': 0.3,  # GDP增速
        'INFLATION': 0.2,  # 通货膨胀
        'PMI': 0.3,  # PMI指数
        'INTEREST_RATE': 0.2  # 利率水平
    }

    # 创新因子权重
    INNOVATION_WEIGHTS = {
        'RD': 0.3,  # 研发投入
        'PATENT': 0.3,  # 专利
        'TALENT': 0.2,  # 人才
        'BUSINESS_MODEL': 0.2  # 商业模式
    }

    # ESG因子权重
    ESG_WEIGHTS = {
        'ENVIRONMENT': 0.4,  # 环境
        'SOCIAL': 0.3,  # 社会
        'GOVERNANCE': 0.3  # 治理
    }

    # 风格因子权重
    STYLE_WEIGHTS = {
        'SIZE': 0.25,  # 规模
        'VALUE': 0.25,  # 价值
        'MOMENTUM': 0.25,  # 动量
        'VOLATILITY': 0.25  # 波动
    }

    # 综合权重
    OVERALL_WEIGHTS = {
        'TECHNICAL': 0.2,  # 技术面
        'FUNDAMENTAL': 0.2,  # 基本面
        'MONEY_FLOW': 0.1,  # 资金流向
        'SENTIMENT': 0.1,  # 市场情绪
        'INDUSTRY': 0.1,  # 行业
        'MACRO': 0.1,  # 宏观
        'INNOVATION': 0.1,  # 创新
        'ESG': 0.05,  # ESG
        'STYLE': 0.05  # 风格
    }


class DataFetcher:
    """数据获取类"""

    def __init__(self):
        self.cache = {}
        self.cache_lifetime = timedelta(hours=Config.CACHE_HOURS)

    @lru_cache(maxsize=1024)
    def get_company_info(self, code: str) -> Dict[str, str]:
        """获取公司基本信息"""
        try:
            # 添加市场标识
            if code.startswith('6'):
                symbol = f"{code}.SH"
            else:
                symbol = f"{code}.SZ"

            info_dict = {}

            # 获取基本信息
            base_info = ak.stock_individual_info_em(symbol=code)
            info_dict['股票简称'] = base_info.loc[base_info['item'] == '股票简称', 'value'].values[0]
            info_dict['行业'] = base_info.loc[base_info['item'] == '行业', 'value'].values[0]
            info_dict['上市时间'] = base_info.loc[base_info['item'] == '上市时间', 'value'].values[0]

            # 获取机构持股信息
            inst_info = ak.stock_institute_hold_detail(symbol=symbol)
            if not inst_info.empty:
                total_inst_ratio = inst_info['持股比例'].astype(float).sum()
                info_dict['机构持股比例'] = f"{total_inst_ratio:.2f}%"
            else:
                info_dict['机构持股比例'] = "0.00%"

            # 获取市值信息
            quote_info = ak.stock_zh_a_spot_em()
            stock_info = quote_info[quote_info['代码'] == code].iloc[0]
            info_dict['总市值'] = f"{float(stock_info['总市值']) / 100000000:.2f}亿"
            info_dict['流通市值'] = f"{float(stock_info['流通市值']) / 100000000:.2f}亿"

            # 获取财务指标
            financial_info = ak.stock_financial_analysis_indicator(symbol=code)
            if not financial_info.empty:
                latest_data = financial_info.iloc[-1]
                info_dict['市盈率'] = f"{latest_data.get('市盈率', 0):.2f}"
                info_dict['市净率'] = f"{latest_data.get('市净率', 0):.2f}"
                info_dict['营收增长率'] = f"{latest_data.get('营业收入同比增长', 0):.2f}%"
                info_dict['净利润增长率'] = f"{latest_data.get('净利润同比增长', 0):.2f}%"

            # 获取ESG评级
            try:
                esg_info = ak.stock_esg_hz_summary()
                stock_esg = esg_info[esg_info['股票代码'] == code]
                if not stock_esg.empty:
                    info_dict['ESG评级'] = stock_esg['ESG评级'].iloc[0]
                else:
                    info_dict['ESG评级'] = "未评级"
            except:
                info_dict['ESG评级'] = "未评级"

            # 获取研发投入
            try:
                rd_info = ak.stock_research_report_em(symbol=code)
                if not rd_info.empty:
                    info_dict['研发投入'] = f"{float(rd_info['研发投入'].iloc[-1]) / 100000000:.2f}亿"
                else:
                    info_dict['研发投入'] = "未披露"
            except:
                info_dict['研发投入'] = "未披露"

            return info_dict

        except Exception as e:
            print(f"获取公司信息异常: {str(e)}")
            return {
                '股票简称': '未知',
                '行业': '未知',
                '上市时间': '未知',
                '机构持股比例': '0.00%',
                '总市值': '0.00亿',
                '流通市值': '0.00亿',
                '市盈率': '0.00',
                '市净率': '0.00',
                '营收增长率': '0.00%',
                '净利润增长率': '0.00%',
                'ESG评级': '未评级',
                '研发投入': '未披露'
            }

    @lru_cache(maxsize=1024)
    def get_history_data(self, code: str, days: int = Config.DEFAULT_DAYS) -> pd.DataFrame:
        """获取历史行情数据"""
        try:
            # 计算日期范围
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

            # 获取行情数据
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )

            # 重命名列
            df.columns = ['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额',
                          '换手率']

            # 设置日期索引
            df['日期'] = pd.to_datetime(df['日期'])
            df.set_index('日期', inplace=True)

            # 确保数据类型正确
            numeric_columns = ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            print(f"已获取{len(df)}条历史数据记录")
            return df

        except Exception as e:
            print(f"获取历史数据异常: {str(e)}")
            return pd.DataFrame()

    @lru_cache(maxsize=1024)
    def get_chip_distribution(self, code: str) -> Tuple[Dict[str, float], str, float]:
        """获取筹码分布数据"""
        try:
            # 获取筹码数据
            df = ak.stock_cyq_em(symbol=code)
            latest_chip = df.iloc[-1].to_dict()

            # 计算套牢比例
            profit_ratio = latest_chip.get('获利比例', 0)
            if profit_ratio <= 0.2:
                keypoint = "√"  # 低套牢率，可能有反弹机会
            elif 0.2 < profit_ratio <= 0.5:
                keypoint = "-"  # 中等套牢率
            else:
                keypoint = "×"  # 高套牢率，上涨压力大

            # 计算平均成本
            avg_cost = latest_chip.get('平均成本', 0)

            # 打印分析信息
            print(f"最新交易日筹码分布：")
            print(f"获利比例: {profit_ratio * 100:.2f}%")
            print(f"平均成本: {avg_cost:.2f}元")

            return latest_chip, keypoint, avg_cost

        except Exception as e:
            print(f"获取筹码分布异常: {str(e)}")
            return {}, "-", 0.0

    def get_real_time_quotes(self, code: str) -> Dict[str, float]:
        """获取实时行情数据"""
        try:
            # 获取实时行情
            quote_df = ak.stock_zh_a_spot_em()
            stock_quote = quote_df[quote_df['代码'] == code].iloc[0]

            quotes = {
                '最新价': float(stock_quote['最新价']),
                '涨跌幅': float(stock_quote['涨跌幅']),
                '成交量': float(stock_quote['成交量']),
                '成交额': float(stock_quote['成交额']),
                '换手率': float(stock_quote['换手率']),
                '振幅': float(stock_quote['振幅'])
            }

            return quotes

        except Exception as e:
            print(f"获取实时行情异常: {str(e)}")
            return {}

    def get_industry_data(self, industry: str) -> pd.DataFrame:
        """获取行业数据"""
        try:
            # 获取行业所有股票
            industry_stocks = ak.stock_industry_summary_em()
            industry_stocks = industry_stocks[industry_stocks['行业'] == industry]

            # 获取行业指标
            industry_data = {
                '股票数量': len(industry_stocks),
                '总市值': industry_stocks['总市值'].sum(),
                '平均市盈率': industry_stocks['市盈率'].mean(),
                '平均市净率': industry_stocks['市净率'].mean(),
                '行业涨跌幅': industry_stocks['涨跌幅'].mean()
            }

            return pd.DataFrame([industry_data])

        except Exception as e:
            print(f"获取行业数据异常: {str(e)}")
            return pd.DataFrame()

    def get_market_status(self) -> Dict[str, Any]:
        """获取市场状态"""
        try:
            # 获取大盘指数
            indices = ak.stock_zh_index_spot()

            # 获取北向资金
            north_money = ak.stock_hsgt_north_net_flow_em()

            # 获取两市成交额
            market_volume = ak.stock_market_volume_em()

            market_status = {
                '上证指数': indices[indices['名称'] == '上证指数']['最新价'].values[0],
                '深证成指': indices[indices['名称'] == '深证成指']['最新价'].values[0],
                '北向资金': north_money['值'].sum(),
                '成交总额': market_volume['两市总成交额'].sum()
            }

            return market_status

        except Exception as e:
            print(f"获取市场状态异常: {str(e)}")
            return {}


class FundamentalFactors:
    """基本面因子分析"""

    def __init__(self):
        self.data_fetcher = DataFetcher()

    def calculate_pe(self, code: str) -> Tuple[float, str]:
        """计算市盈率"""
        try:
            stock_financial = ak.stock_financial_analysis_indicator(code)
            latest_pe = stock_financial['总市值/净利润'].iloc[-1]
            industry_pe = self._get_industry_pe(code)

            if latest_pe < industry_pe * 0.7:
                return latest_pe, "√"
            elif latest_pe > industry_pe * 1.3:
                return latest_pe, "×"
            else:
                return latest_pe, "-"
        except Exception as e:
            print(f"PE计算异常: {str(e)}")
            return 0.0, "-"

    def calculate_pb(self, code: str) -> Tuple[float, str]:
        """计算市净率"""
        try:
            stock_financial = ak.stock_financial_analysis_indicator(code)
            latest_pb = stock_financial['市净率'].iloc[-1]
            industry_pb = self._get_industry_pb(code)

            if latest_pb < industry_pb * 0.7:
                return latest_pb, "√"
            elif latest_pb > industry_pb * 1.3:
                return latest_pb, "×"
            else:
                return latest_pb, "-"
        except Exception as e:
            print(f"PB计算异常: {str(e)}")
            return 0.0, "-"

    def calculate_roe(self, code: str) -> Tuple[float, str]:
        """计算净资产收益率"""
        try:
            stock_financial = ak.stock_financial_analysis_indicator(code)
            latest_roe = stock_financial['净资产收益率'].iloc[-1]
            industry_roe = self._get_industry_roe(code)

            if latest_roe > industry_roe * 1.2:
                return latest_roe, "√"
            elif latest_roe < industry_roe * 0.8:
                return latest_roe, "×"
            else:
                return latest_roe, "-"
        except Exception as e:
            print(f"ROE计算异常: {str(e)}")
            return 0.0, "-"

    def calculate_growth(self, code: str) -> Tuple[float, str]:
        """计算营收增长率"""
        try:
            stock_financial = ak.stock_financial_analysis_indicator(code)
            revenue_growth = stock_financial['营业收入同比增长'].iloc[-1]
            industry_growth = self._get_industry_growth(code)

            if revenue_growth > industry_growth * 1.2:
                return revenue_growth, "√"
            elif revenue_growth < industry_growth * 0.8:
                return revenue_growth, "×"
            else:
                return revenue_growth, "-"
        except Exception as e:
            print(f"增长率计算异常: {str(e)}")
            return 0.0, "-"

    def calculate_debt(self, code: str) -> Tuple[float, str]:
        """计算资产负债率"""
        try:
            stock_financial = ak.stock_financial_analysis_indicator(code)
            debt_ratio = stock_financial['资产负债率'].iloc[-1]
            industry_debt = self._get_industry_debt(code)

            if debt_ratio < industry_debt * 0.8:
                return debt_ratio, "√"
            elif debt_ratio > industry_debt * 1.2:
                return debt_ratio, "×"
            else:
                return debt_ratio, "-"
        except Exception as e:
            print(f"负债率计算异常: {str(e)}")
            return 0.0, "-"

    def calculate_cash_flow(self, code: str) -> Tuple[float, str]:
        """计算经营性现金流"""
        try:
            stock_financial = ak.stock_financial_analysis_indicator(code)
            operating_cash_flow = stock_financial['经营活动产生的现金流量净额'].iloc[-1]
            net_profit = stock_financial['净利润'].iloc[-1]

            cash_flow_ratio = operating_cash_flow / net_profit if net_profit != 0 else 0

            if cash_flow_ratio > 1.2:
                return cash_flow_ratio, "√"
            elif cash_flow_ratio < 0.8:
                return cash_flow_ratio, "×"
            else:
                return cash_flow_ratio, "-"
        except Exception as e:
            print(f"现金流计算异常: {str(e)}")
            return 0.0, "-"

    def _get_industry_pe(self, code: str) -> float:
        """获取行业平均市盈率"""
        try:
            industry_info = self.data_fetcher.get_company_info(code)
            industry = industry_info['行业']
            industry_data = ak.stock_industry_pe_ratio()
            return industry_data[industry_data['行业'] == industry]['市盈率'].iloc[0]
        except:
            return 0.0

    def _get_industry_pb(self, code: str) -> float:
        """获取行业平均市净率"""
        try:
            industry_info = self.data_fetcher.get_company_info(code)
            industry = industry_info['行业']
            industry_data = ak.stock_industry_pe_ratio()
            return industry_data[industry_data['行业'] == industry]['市净率'].iloc[0]
        except:
            return 0.0

    def _get_industry_roe(self, code: str) -> float:
        """获取行业平均净资产收益率"""
        try:
            industry_info = self.data_fetcher.get_company_info(code)
            industry = industry_info['行业']
            industry_data = ak.stock_industry_pe_ratio()
            return industry_data[industry_data['行业'] == industry]['净资产收益率'].iloc[0]
        except:
            return 0.0

    def _get_industry_growth(self, code: str) -> float:
        """获取行业平均增长率"""
        try:
            industry_info = self.data_fetcher.get_company_info(code)
            industry = industry_info['行业']
            industry_data = ak.stock_industry_pe_ratio()
            return industry_data[industry_data['行业'] == industry]['营收增长率'].iloc[0]
        except:
            return 0.0

    def _get_industry_debt(self, code: str) -> float:
        """获取行业平均负债率"""
        try:
            industry_info = self.data_fetcher.get_company_info(code)
            industry = industry_info['行业']
            industry_data = ak.stock_industry_pe_ratio()
            return industry_data[industry_data['行业'] == industry]['资产负债率'].iloc[0]
        except:
            return 0.0


class MoneyFlowFactors:
    """资金流向因子分析"""

    def __init__(self):
        self.data_fetcher = DataFetcher()

    def calculate_big_money_flow(self, code: str) -> Tuple[float, str]:
        """分析大单资金流向"""
        try:
            # 获取最近5日大单资金流向数据
            money_flow_data = ak.stock_cash_flow_individual_em(symbol=code)

            # 计算主力净流入
            net_inflow = money_flow_data['主力净流入'].sum()
            total_amount = money_flow_data['成交额'].sum()

            # 计算主力净流入占比
            inflow_ratio = net_inflow / total_amount if total_amount != 0 else 0

            if inflow_ratio > 0.1:  # 主力净流入超过10%
                return inflow_ratio, "√"
            elif inflow_ratio < -0.1:  # 主力净流出超过10%
                return inflow_ratio, "×"
            else:
                return inflow_ratio, "-"

        except Exception as e:
            print(f"大单资金流向分析异常: {str(e)}")
            return 0.0, "-"

    def analyze_northbound_flow(self, code: str) -> Tuple[float, str]:
        """分析北向资金持股"""
        try:
            # 获取北向资金持股数据
            northbound_data = ak.stock_hsgt_individual_em(symbol=code)

            # 计算最近5日北向资金持股变化
            recent_changes = northbound_data['持股数量'].diff().tail(5).sum()
            total_shares = northbound_data['持股数量'].iloc[-1]

            # 计算变化率
            change_ratio = recent_changes / total_shares if total_shares != 0 else 0

            if change_ratio > 0.02:  # 增持超过2%
                return change_ratio, "√"
            elif change_ratio < -0.02:  # 减持超过2%
                return change_ratio, "×"
            else:
                return change_ratio, "-"

        except Exception as e:
            print(f"北向资金分析异常: {str(e)}")
            return 0.0, "-"

    def analyze_institutional_holdings(self, code: str) -> Tuple[float, str]:
        """分析机构持股变动"""
        try:
            # 获取机构持股数据
            institutional_data = ak.stock_report_fund_hold_detail(symbol=code)

            # 计算机构持股比例变化
            current_ratio = institutional_data['持股数量'].sum() / institutional_data['总股本'].iloc[0]
            previous_ratio = institutional_data['上期持股数量'].sum() / institutional_data['总股本'].iloc[0]

            # 计算变化幅度
            change_ratio = (current_ratio - previous_ratio) / previous_ratio if previous_ratio != 0 else 0

            if change_ratio > 0.05:  # 增持超过5%
                return change_ratio, "√"
            elif change_ratio < -0.05:  # 减持超过5%
                return change_ratio, "×"
            else:
                return change_ratio, "-"

        except Exception as e:
            print(f"机构持股分析异常: {str(e)}")
            return 0.0, "-"

    def analyze_margin_trading(self, code: str) -> Tuple[float, str]:
        """分析融资融券数据"""
        try:
            # 获取融资融券数据
            margin_data = ak.stock_margin_detail_em(symbol=code)

            # 计算融资融券余额变化
            recent_changes = margin_data['融资余额'].diff().tail(5).sum()
            total_amount = margin_data['融资余额'].iloc[-1]

            # 计算变化率
            change_ratio = recent_changes / total_amount if total_amount != 0 else 0

            if change_ratio > 0.05:  # 融资增加超过5%
                return change_ratio, "√"
            elif change_ratio < -0.05:  # 融资减少超过5%
                return change_ratio, "×"
            else:
                return change_ratio, "-"

        except Exception as e:
            print(f"融资融券分析异常: {str(e)}")
            return 0.0, "-"


class SentimentFactors:
    """情绪因子分析"""

    def __init__(self):
        self.data_fetcher = DataFetcher()

    def calculate_implied_volatility(self, code: str) -> Tuple[float, str]:
        """计算期权隐含波动率"""
        try:
            # 获取期权数据
            option_data = ak.option_50etf_volatility_analysis()

            # 计算隐含波动率
            current_iv = option_data['已实现波动率'].iloc[-1]
            historical_mean = option_data['已实现波动率'].mean()

            # 计算波动率偏离度
            deviation = (current_iv - historical_mean) / historical_mean if historical_mean != 0 else 0

            if deviation < -0.2:  # 波动率显著低于历史均值
                return deviation, "√"
            elif deviation > 0.2:  # 波动率显著高于历史均值
                return deviation, "×"
            else:
                return deviation, "-"

        except Exception as e:
            print(f"隐含波动率计算异常: {str(e)}")
            return 0.0, "-"

    def analyze_forum_sentiment(self, code: str) -> Tuple[float, str]:
        """分析股吧情绪"""
        try:
            # 获取股吧帖子数据
            forum_data = ak.stock_guba_em(symbol=code)

            # 简单的情绪分析（可以使用更复杂的NLP模型）
            positive_words = ['买入', '看多', '上涨', '利好', '突破']
            negative_words = ['卖出', '看空', '下跌', '利空', '跌破']

            # 计算情绪得分
            sentiment_score = 0
            total_posts = len(forum_data)

            for _, post in forum_data.iterrows():
                title = str(post['标题'])
                positive_count = sum(1 for word in positive_words if word in title)
                negative_count = sum(1 for word in negative_words if word in title)
                sentiment_score += (positive_count - negative_count)

            # 归一化情绪得分
            normalized_score = sentiment_score / total_posts if total_posts > 0 else 0

            if normalized_score > 0.2:
                return normalized_score, "√"
            elif normalized_score < -0.2:
                return normalized_score, "×"
            else:
                return normalized_score, "-"

        except Exception as e:
            print(f"股吧情绪分析异常: {str(e)}")
            return 0.0, "-"

    def analyze_top_traders(self, code: str) -> Tuple[float, str]:
        """分析龙虎榜活跃度"""
        try:
            # 获取龙虎榜数据
            top_traders_data = ak.stock_lhb_detail_em(symbol=code)

            # 计算机构参与度
            total_records = len(top_traders_data)
            institution_records = len(top_traders_data[top_traders_data['营业部类型'].str.contains('机构')])

            # 计算机构参与率
            institution_ratio = institution_records / total_records if total_records > 0 else 0

            if institution_ratio > 0.5:  # 机构参与度高
                return institution_ratio, "√"
            elif institution_ratio < 0.2:  # 机构参与度低
                return institution_ratio, "×"
            else:
                return institution_ratio, "-"

        except Exception as e:
            print(f"龙虎榜分析异常: {str(e)}")
            return 0.0, "-"

    def analyze_shareholder_changes(self, code: str) -> Tuple[float, str]:
        """分析股东户数变化"""
        try:
            # 获取股东户数数据
            shareholder_data = ak.stock_holder_num_detail_em(symbol=code)

            # 计算户数变化率
            current_holders = shareholder_data['股东户数'].iloc[-1]
            previous_holders = shareholder_data['股东户数'].iloc[-2]

            # 计算变化率
            change_ratio = (current_holders - previous_holders) / previous_holders if previous_holders != 0 else 0

            if change_ratio < -0.05:  # 户数减少超过5%
                return change_ratio, "√"
            elif change_ratio > 0.05:  # 户数增加超过5%
                return change_ratio, "×"
            else:
                return change_ratio, "-"

        except Exception as e:
            print(f"股东户数分析异常: {str(e)}")
            return 0.0, "-"


class IndustryFactors:
    """行业因子分析"""

    def __init__(self):
        self.data_fetcher = DataFetcher()

    def analyze_industry_prosperity(self, code: str) -> Tuple[float, str]:
        """分析行业景气度"""
        try:
            # 获取公司所属行业
            company_info = self.data_fetcher.get_company_info(code)
            industry = company_info['行业']

            # 获取行业整体数据
            industry_data = ak.stock_industry_change_em()
            industry_info = industry_data[industry_data['板块名称'] == industry].iloc[0]

            # 计算行业景气度指标
            industry_change = float(industry_info['涨跌幅'].strip('%')) / 100
            industry_turnover = float(industry_info['换手率'].strip('%')) / 100
            industry_pe = float(industry_info['市盈率'])

            # 综合评分
            prosperity_score = (
                    0.4 * (industry_change + 1) +  # 行业涨跌幅
                    0.3 * (industry_turnover / 0.1) +  # 换手率（相对于10%基准）
                    0.3 * (15 / industry_pe if industry_pe > 0 else 0)  # PE估值（相对于15倍PE基准）
            )

            if prosperity_score > 1.2:
                return prosperity_score, "√"
            elif prosperity_score < 0.8:
                return prosperity_score, "×"
            else:
                return prosperity_score, "-"

        except Exception as e:
            print(f"行业景气度分析异常: {str(e)}")
            return 0.0, "-"

    def analyze_industry_concentration(self, code: str) -> Tuple[float, str]:
        """分析行业集中度"""
        try:
            # 获取行业所有公司数据
            company_info = self.data_fetcher.get_company_info(code)
            industry = company_info['行业']
            industry_stocks = ak.stock_industry_summary_em()
            industry_stocks = industry_stocks[industry_stocks['行业'] == industry]

            # 计算公司市值占行业总市值的比例
            total_market_cap = industry_stocks['总市值'].sum()
            company_market_cap = industry_stocks[industry_stocks['代码'] == code]['总市值'].iloc[0]
            market_share = company_market_cap / total_market_cap if total_market_cap > 0 else 0

            # 计算行业集中度（CR4）
            top_4_share = industry_stocks.nlargest(4, '总市值')['总市值'].sum() / total_market_cap

            # 综合评估公司的行业地位
            if market_share > 0.1 and top_4_share > 0.6:  # 龙头公司
                return market_share, "√"
            elif market_share < 0.01:  # 边缘公司
                return market_share, "×"
            else:
                return market_share, "-"

        except Exception as e:
            print(f"行业集中度分析异常: {str(e)}")
            return 0.0, "-"

    def analyze_policy_sensitivity(self, code: str) -> Tuple[float, str]:
        """分析政策敏感度"""
        try:
            # 获取行业政策新闻
            company_info = self.data_fetcher.get_company_info(code)
            industry = company_info['行业']

            # 获取行业新闻数据
            news_data = ak.stock_news_em()
            industry_news = news_data[news_data['新闻标题'].str.contains(industry, na=False)]

            # 统计政策相关新闻占比
            policy_keywords = ['政策', '规范', '监管', '发改委', '国务院', '部委']
            policy_news_count = sum(
                1 for _, news in industry_news.iterrows()
                if any(keyword in news['新闻标题'] for keyword in policy_keywords)
            )

            policy_sensitivity = policy_news_count / len(industry_news) if len(industry_news) > 0 else 0

            if policy_sensitivity < 0.1:  # 政策敏感度低
                return policy_sensitivity, "√"
            elif policy_sensitivity > 0.3:  # 政策敏感度高
                return policy_sensitivity, "×"
            else:
                return policy_sensitivity, "-"

        except Exception as e:
            print(f"政策敏感度分析异常: {str(e)}")
            return 0.0, "-"

    def analyze_supply_chain(self, code: str) -> Tuple[float, str]:
        """分析产业链地位"""
        try:
            # 获取上下游数据
            company_info = self.data_fetcher.get_company_info(code)
            financial_data = ak.stock_financial_analysis_indicator(code)

            # 计算毛利率
            gross_margin = financial_data['销售毛利率'].iloc[-1]

            # 获取应收账款周转率和存货周转率
            receivables_turnover = financial_data['应收账款周转率'].iloc[-1]
            inventory_turnover = financial_data['存货周转率'].iloc[-1]

            # 综合评估产业链地位
            supply_chain_score = (
                    0.4 * (gross_margin / 30) +  # 毛利率（相对于30%基准）
                    0.3 * (receivables_turnover / 4) +  # 应收账款周转（相对于4次/年基准）
                    0.3 * (inventory_turnover / 4)  # 存货周转（相对于4次/年基准）
            )

            if supply_chain_score > 1.2:  # 产业链地位强
                return supply_chain_score, "√"
            elif supply_chain_score < 0.8:  # 产业链地位弱
                return supply_chain_score, "×"
            else:
                return supply_chain_score, "-"

        except Exception as e:
            print(f"产业链地位分析异常: {str(e)}")
            return 0.0, "-"


class MacroFactors:
    """宏观因子分析"""

    def __init__(self):
        self.data_fetcher = DataFetcher()

    def analyze_gdp_impact(self, code: str) -> Tuple[float, str]:
        """分析GDP增速影响"""
        try:
            # 获取GDP数据
            gdp_data = ak.macro_china_gdp()
            latest_gdp_growth = gdp_data['value'].iloc[-1]

            # 获取公司收入增速
            company_info = self.data_fetcher.get_company_info(code)
            financial_data = ak.stock_financial_analysis_indicator(code)
            revenue_growth = financial_data['营业收入同比增长'].iloc[-1]

            # 计算收入增速对GDP增速的敏感度
            gdp_sensitivity = revenue_growth / latest_gdp_growth if latest_gdp_growth != 0 else 0

            if gdp_sensitivity > 2:  # 高度受益于GDP增长
                return gdp_sensitivity, "√"
            elif gdp_sensitivity < 0.5:  # 受GDP影响小
                return gdp_sensitivity, "×"
            else:
                return gdp_sensitivity, "-"

        except Exception as e:
            print(f"GDP影响分析异常: {str(e)}")
            return 0.0, "-"

    def analyze_inflation_impact(self, code: str) -> Tuple[float, str]:
        """分析通货膨胀影响"""
        try:
            # 获取CPI数据
            cpi_data = ak.macro_china_cpi()
            latest_cpi = cpi_data['value'].iloc[-1]

            # 获取公司毛利率数据
            financial_data = ak.stock_financial_analysis_indicator(code)
            gross_margin = financial_data['销售毛利率'].iloc[-1]
            gross_margin_prev = financial_data['销售毛利率'].iloc[-2]

            # 计算毛利率对CPI的敏感度
            margin_change = (gross_margin - gross_margin_prev) / gross_margin_prev if gross_margin_prev != 0 else 0
            inflation_sensitivity = margin_change / (latest_cpi / 100) if latest_cpi != 0 else 0

            if inflation_sensitivity > 0:  # 能够转嫁成本
                return inflation_sensitivity, "√"
            elif inflation_sensitivity < -1:  # 成本转嫁能力弱
                return inflation_sensitivity, "×"
            else:
                return inflation_sensitivity, "-"

        except Exception as e:
            print(f"通胀影响分析异常: {str(e)}")
            return 0.0, "-"

    def analyze_pmi_impact(self, code: str) -> Tuple[float, str]:
        """分析PMI影响"""
        try:
            # 获取PMI数据
            pmi_data = ak.macro_china_pmi()
            latest_pmi = pmi_data['value'].iloc[-1]

            # 获取公司所属行业
            company_info = self.data_fetcher.get_company_info(code)
            industry = company_info['行业']

            # 判断行业是否属于制造业
            manufacturing_industries = ['制造业', '工业', '机械', '电子', '化工']
            is_manufacturing = any(ind in industry for ind in manufacturing_industries)

            if is_manufacturing:
                if latest_pmi > 50:  # 制造业景气
                    return latest_pmi, "√"
                else:  # 制造业低迷
                    return latest_pmi, "×"
            else:
                return latest_pmi, "-"  # 非制造业

        except Exception as e:
            print(f"PMI影响分析异常: {str(e)}")
            return 0.0, "-"

    def analyze_interest_rate_sensitivity(self, code: str) -> Tuple[float, str]:
        """分析利率敏感度"""
        try:
            # 获取LPR数据
            lpr_data = ak.macro_china_lpr()
            latest_lpr = lpr_data['value'].iloc[-1]

            # 获取公司资产负债率
            financial_data = ak.stock_financial_analysis_indicator(code)
            debt_ratio = financial_data['资产负债率'].iloc[-1]

            # 计算利息保障倍数
            interest_coverage = financial_data['息税前利润'].iloc[-1] / financial_data['财务费用'].iloc[-1] \
                if financial_data['财务费用'].iloc[-1] != 0 else float('inf')

            # 综合评估利率敏感度
            if debt_ratio < 30 or interest_coverage > 5:  # 利率敏感度低
                return interest_coverage, "√"
            elif debt_ratio > 60 or interest_coverage < 2:  # 利率敏感度高
                return interest_coverage, "×"
            else:
                return interest_coverage, "-"

        except Exception as e:
            print(f"利率敏感度分析异常: {str(e)}")
            return 0.0, "-"


class InnovationFactors:
    """创新因子分析"""

    def __init__(self):
        self.data_fetcher = DataFetcher()

    def analyze_rd_investment(self, code: str) -> Tuple[float, str]:
        """分析研发投入"""
        try:
            # 获取研发投入数据
            financial_data = ak.stock_financial_analysis_indicator(code)

            # 计算研发投入占营收比例
            revenue = financial_data['营业收入'].iloc[-1]
            rd_expense = financial_data['研发费用'].iloc[-1] if '研发费用' in financial_data.columns else 0
            rd_ratio = rd_expense / revenue if revenue != 0 else 0

            # 获取行业平均研发投入比例
            company_info = self.data_fetcher.get_company_info(code)
            industry = company_info['行业']
            industry_data = ak.stock_industry_summary_em()
            industry_stocks = industry_data[industry_data['行业'] == industry]

            # 计算行业平均研发投入比例
            industry_rd_ratio = industry_stocks[
                '研发投入占比'].mean() if '研发投入占比' in industry_stocks.columns else 0.03

            if rd_ratio > industry_rd_ratio * 1.5:  # 显著高于行业平均
                return rd_ratio, "√"
            elif rd_ratio < industry_rd_ratio * 0.5:  # 显著低于行业平均
                return rd_ratio, "×"
            else:
                return rd_ratio, "-"

        except Exception as e:
            print(f"研发投入分析异常: {str(e)}")
            return 0.0, "-"

    def analyze_patents(self, code: str) -> Tuple[float, str]:
        """分析专利情况"""
        try:
            # 获取公司专利数据
            company_info = self.data_fetcher.get_company_info(code)

            # 获取年报中的专利信息
            financial_report = ak.stock_financial_report_em(code)

            # 提取专利相关信息（简单示例，实际可能需要更复杂的文本分析）
            patent_keywords = ['专利', '知识产权', '技术创新']
            patent_score = 0

            for _, report in financial_report.iterrows():
                content = str(report['报告内容'])
                patent_count = sum(content.count(keyword) for keyword in patent_keywords)
                patent_score += patent_count

            # 获取行业平均专利得分
            industry = company_info['行业']
            industry_patent_score = self._get_industry_patent_score(industry)

            if patent_score > industry_patent_score * 1.5:
                return patent_score, "√"
            elif patent_score < industry_patent_score * 0.5:
                return patent_score, "×"
            else:
                return patent_score, "-"

        except Exception as e:
            print(f"专利分析异常: {str(e)}")
            return 0.0, "-"

    def analyze_talent_density(self, code: str) -> Tuple[float, str]:
        """分析人才密度"""
        try:
            # 获取公司员工信息
            company_info = self.data_fetcher.get_company_info(code)
            employee_data = ak.stock_employee_information_em(code)

            # 计算高学历员工占比
            total_employees = employee_data['员工总数'].iloc[-1]
            high_edu_employees = employee_data['本科及以上人数'].iloc[-1]
            talent_ratio = high_edu_employees / total_employees if total_employees > 0 else 0

            # 获取行业平均人才密度
            industry = company_info['行业']
            industry_talent_ratio = self._get_industry_talent_ratio(industry)

            if talent_ratio > industry_talent_ratio * 1.3:
                return talent_ratio, "√"
            elif talent_ratio < industry_talent_ratio * 0.7:
                return talent_ratio, "×"
            else:
                return talent_ratio, "-"

        except Exception as e:
            print(f"人才密度分析异常: {str(e)}")
            return 0.0, "-"

    def analyze_business_model(self, code: str) -> Tuple[float, str]:
        """分析商业模式创新性"""
        try:
            # 获取公司年报信息
            financial_report = ak.stock_financial_report_em(code)

            # 分析商业模式创新关键词
            innovation_keywords = [
                '商业模式', '创新', '转型', '数字化', '智能化',
                '平台', '生态', '用户体验', '新技术', '新产品'
            ]

            # 计算创新相关描述的频率
            innovation_score = 0
            for _, report in financial_report.iterrows():
                content = str(report['报告内容'])
                innovation_count = sum(content.count(keyword) for keyword in innovation_keywords)
                innovation_score += innovation_count

            # 获取行业平均创新得分
            company_info = self.data_fetcher.get_company_info(code)
            industry = company_info['行业']
            industry_innovation_score = self._get_industry_innovation_score(industry)

            if innovation_score > industry_innovation_score * 1.5:
                return innovation_score, "√"
            elif innovation_score < industry_innovation_score * 0.5:
                return innovation_score, "×"
            else:
                return innovation_score, "-"

        except Exception as e:
            print(f"商业模式创新性分析异常: {str(e)}")
            return 0.0, "-"

    def _get_industry_patent_score(self, industry: str) -> float:
        """获取行业平均专利得分"""
        # 这里可以实现更复杂的行业专利数据获取逻辑
        industry_scores = {
            '计算机': 100,
            '电子': 80,
            '医药': 70,
            '机械': 50,
            '化工': 40,
            '房地产': 20,
            '金融': 15
        }
        return industry_scores.get(industry, 30)

    def _get_industry_talent_ratio(self, industry: str) -> float:
        """获取行业平均人才密度"""
        # 这里可以实现更复杂的行业人才密度数据获取逻辑
        industry_ratios = {
            '计算机': 0.8,
            '电子': 0.7,
            '医药': 0.6,
            '金融': 0.5,
            '机械': 0.4,
            '化工': 0.3,
            '房地产': 0.2
        }
        return industry_ratios.get(industry, 0.3)

    def _get_industry_innovation_score(self, industry: str) -> float:
        """获取行业平均创新得分"""
        # 这里可以实现更复杂的行业创新得分数据获取逻辑
        industry_scores = {
            '计算机': 50,
            '电子': 45,
            '医药': 40,
            '机械': 30,
            '化工': 25,
            '金融': 20,
            '房地产': 15
        }
        return industry_scores.get(industry, 20)


class ESGFactors:
    """ESG因子分析"""

    def __init__(self):
        self.data_fetcher = DataFetcher()

    def analyze_environmental_impact(self, code: str) -> Tuple[float, str]:
        """分析环境影响"""
        try:
            # 获取公司ESG评级数据
            company_info = self.data_fetcher.get_company_info(code)
            financial_report = ak.stock_financial_report_em(code)

            # 环境相关关键词
            env_keywords = [
                '节能', '减排', '环保', '绿色', '可持续',
                '碳中和', '污染防治', '生态', '清洁能源'
            ]

            # 计算环境相关描述的频率和投入
            env_score = 0
            env_investment = 0

            for _, report in financial_report.iterrows():
                content = str(report['报告内容'])
                env_count = sum(content.count(keyword) for keyword in env_keywords)
                env_score += env_count

                # 提取环保投入金额（实际实现需要更复杂的文本分析）
                if '环保投入' in content or '环境投入' in content:
                    # 简单示例，实际需要更准确的提取方法
                    env_investment += 1

            # 综合评分
            total_score = (env_score * 0.6 + env_investment * 0.4) / 2

            if total_score > 30:
                return total_score, "√"
            elif total_score < 10:
                return total_score, "×"
            else:
                return total_score, "-"

        except Exception as e:
            print(f"环境影响分析异常: {str(e)}")
            return 0.0, "-"

    def analyze_social_responsibility(self, code: str) -> Tuple[float, str]:
        """分析社会责任"""
        try:
            # 获取公司社会责任报告
            financial_report = ak.stock_financial_report_em(code)

            # 社会责任相关关键词
            social_keywords = [
                '社会责任', '公益', '扶贫', '就业', '员工福利',
                '社区发展', '慈善', '教育支持', '医疗援助'
            ]

            # 计算社会责任相关描述的频率
            social_score = 0
            donation_amount = 0

            for _, report in financial_report.iterrows():
                content = str(report['报告内容'])
                social_count = sum(content.count(keyword) for keyword in social_keywords)
                social_score += social_count

                # 提取捐赠金额（实际实现需要更复杂的文本分析）
                if '捐赠' in content or '公益支出' in content:
                    # 简单示例，实际需要更准确的提取方法
                    donation_amount += 1

            # 综合评分
            total_score = (social_score * 0.7 + donation_amount * 0.3) / 2

            if total_score > 25:
                return total_score, "√"
            elif total_score < 8:
                return total_score, "×"
            else:
                return total_score, "-"

        except Exception as e:
            print(f"社会责任分析异常: {str(e)}")
            return 0.0, "-"

    def analyze_governance_quality(self, code: str) -> Tuple[float, str]:
        """分析公司治理"""
        try:
            # 获取公司治理相关数据
            company_info = self.data_fetcher.get_company_info(code)
            financial_data = ak.stock_financial_analysis_indicator(code)

            # 计算治理相关指标
            # 1. 董事会独立性
            board_data = ak.stock_board_information_em(code)
            independent_directors = board_data['独立董事人数'].iloc[-1] if '独立董事人数' in board_data.columns else 0
            total_directors = board_data['董事会人数'].iloc[-1] if '董事会人数' in board_data.columns else 1
            independence_ratio = independent_directors / total_directors if total_directors > 0 else 0

            # 2. 股权集中度
            ownership_data = ak.stock_main_stock_holder(code)
            top_holder_ratio = float(
                ownership_data['持股比例'].iloc[0].strip('%')) / 100 if not ownership_data.empty else 1

            # 3. 信息披露质量
            disclosure_score = self._analyze_disclosure_quality(code)

            # 综合评分
            governance_score = (
                    0.4 * (1 - top_holder_ratio) +  # 股权分散度
                    0.3 * independence_ratio +  # 董事会独立性
                    0.3 * disclosure_score  # 信息披露质量
            )

            if governance_score > 0.7:
                return governance_score, "√"
            elif governance_score < 0.4:
                return governance_score, "×"
            else:
                return governance_score, "-"

        except Exception as e:
            print(f"公司治理分析异常: {str(e)}")
            return 0.0, "-"

    def _analyze_disclosure_quality(self, code: str) -> float:
        """分析信息披露质量"""
        try:
            # 获取公司公告数据
            announcements = ak.stock_announcement_em(code)

            # 计算年度公告数量
            annual_count = len(announcements)

            # 设定基准值
            benchmark = 50  # 假设年度合理公告数量为50个

            # 计算得分
            if annual_count > benchmark * 1.5:  # 过度披露
                score = 0.6
            elif annual_count < benchmark * 0.5:  # 披露不足
                score = 0.3
            else:  # 适度披露
                score = 0.8

            return score
        except:
            return 0.5  # 默认中等水平


class StyleFactors:
    """风格因子分析"""

    def __init__(self):
        self.data_fetcher = DataFetcher()

    def analyze_size_factor(self, code: str) -> Tuple[float, str]:
        """分析规模因子"""
        try:
            # 获取市值数据
            stock_info = ak.stock_individual_info_em(symbol=code)
            market_cap = float(stock_info.loc[stock_info['item'] == '总市值', 'value'].values[0])

            # 获取行业市值分布
            company_info = self.data_fetcher.get_company_info(code)
            industry = company_info['行业']
            industry_data = ak.stock_industry_summary_em()
            industry_stocks = industry_data[industry_data['行业'] == industry]

            # 计算市值分位数
            industry_caps = industry_stocks['总市值'].astype(float)
            percentile = sum(market_cap >= cap for cap in industry_caps) / len(industry_caps)

            if percentile > 0.8:  # 大市值
                return percentile, "√"
            elif percentile < 0.2:  # 小市值
                return percentile, "×"
            else:
                return percentile, "-"

        except Exception as e:
            print(f"规模因子分析异常: {str(e)}")
            return 0.0, "-"

    def analyze_value_factor(self, code: str) -> Tuple[float, str]:
        """分析价值因子"""
        try:
            # 获取估值指标
            financial_data = ak.stock_financial_analysis_indicator(code)

            # 计算综合价值分数
            pe = financial_data['市盈率'].iloc[-1]
            pb = financial_data['市净率'].iloc[-1]
            ps = financial_data['市销率'].iloc[-1]

            # 获取行业平均水平
            company_info = self.data_fetcher.get_company_info(code)
            industry = company_info['行业']
            industry_avg = self._get_industry_valuation(industry)

            # 计算相对估值水平
            relative_pe = pe / industry_avg['pe'] if industry_avg['pe'] != 0 else 1
            relative_pb = pb / industry_avg['pb'] if industry_avg['pb'] != 0 else 1
            relative_ps = ps / industry_avg['ps'] if industry_avg['ps'] != 0 else 1

            # 综合评分
            value_score = (1 / relative_pe + 1 / relative_pb + 1 / relative_ps) / 3

            if value_score > 1.2:  # 低估
                return value_score, "√"
            elif value_score < 0.8:  # 高估
                return value_score, "×"
            else:
                return value_score, "-"

        except Exception as e:
            print(f"价值因子分析异常: {str(e)}")
            return 0.0, "-"

    def analyze_momentum_factor(self, code: str) -> Tuple[float, str]:
        """分析动量因子"""
        try:
            # 获取历史价格数据
            stock_data = self.data_fetcher.get_history_data(code)

            # 计算不同期限收益率
            returns = {
                'monthly': stock_data['收盘'].pct_change(20).iloc[-1],
                'quarterly': stock_data['收盘'].pct_change(60).iloc[-1],
                'semiannual': stock_data['收盘'].pct_change(120).iloc[-1]
            }

            # 计算动量得分
            momentum_score = (
                    0.5 * returns['monthly'] +
                    0.3 * returns['quarterly'] +
                    0.2 * returns['semiannual']
            )

            if momentum_score > 0.1:  # 强势动量
                return momentum_score, "√"
            elif momentum_score < -0.1:  # 弱势动量
                return momentum_score, "×"
            else:
                return momentum_score, "-"

        except Exception as e:
            print(f"动量因子分析异常: {str(e)}")
            return 0.0, "-"

    def analyze_volatility_factor(self, code: str) -> Tuple[float, str]:
        """分析波动因子"""
        try:
            # 获取历史价格数据
            stock_data = self.data_fetcher.get_history_data(code)

            # 计算波动率指标
            returns = stock_data['收盘'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # 年化波动率

            # 获取行业波动率
            company_info = self.data_fetcher.get_company_info(code)
            industry = company_info['行业']
            industry_volatility = self._get_industry_volatility(industry)

            # 计算相对波动率
            relative_volatility = volatility / industry_volatility if industry_volatility != 0 else 1

            if relative_volatility < 0.8:  # 低波动
                return relative_volatility, "√"
            elif relative_volatility > 1.2:  # 高波动
                return relative_volatility, "×"
            else:
                return relative_volatility, "-"

        except Exception as e:
            print(f"波动因子分析异常: {str(e)}")
            return 0.0, "-"

    def _get_industry_valuation(self, industry: str) -> Dict[str, float]:
        """获取行业平均估值水平"""
        industry_vals = {
            '计算机': {'pe': 35, 'pb': 3.5, 'ps': 2.5},
            '医药': {'pe': 40, 'pb': 4.0, 'ps': 3.0},
            '消费': {'pe': 30, 'pb': 3.0, 'ps': 2.0},
            '金融': {'pe': 10, 'pb': 1.2, 'ps': 3.0},
            '地产': {'pe': 12, 'pb': 1.5, 'ps': 2.0},
            '工业': {'pe': 20, 'pb': 2.0, 'ps': 1.5}
        }
        return industry_vals.get(industry, {'pe': 25, 'pb': 2.5, 'ps': 2.0})

    def _get_industry_volatility(self, industry: str) -> float:
        """获取行业平均波动率"""
        industry_volatility = {
            '计算机': 0.35,
            '医药': 0.30,
            '消费': 0.25,
            '金融': 0.20,
            '地产': 0.28,
            '工业': 0.26
        }
        return industry_volatility.get(industry, 0.28)


class TechnicalIndicators:
    """技术指标计算类"""

    @staticmethod
    def calculate_bbiboll(df: pd.DataFrame) -> Tuple[Dict[str, float], str, float]:
        """计算BBIBOLL指标"""
        try:
            df = df.copy()

            # 计算多个移动平均线
            df['MA3'] = df['收盘'].rolling(3).mean()
            df['MA6'] = df['收盘'].rolling(6).mean()
            df['MA12'] = df['收盘'].rolling(12).mean()
            df['MA24'] = df['收盘'].rolling(24).mean()

            # 计算BBIBOLL
            df['BBIBOLL'] = (df['MA3'] + df['MA6'] + df['MA12'] + df['MA24']) / 4

            # 计算标准差
            df['BBIBOLL_STD'] = df['BBIBOLL'].rolling(Config.BBIBOLL_WINDOW).std()

            # 计算上下轨
            df['UPPER'] = df['BBIBOLL'] + 2 * df['BBIBOLL_STD']
            df['LOWER'] = df['BBIBOLL'] - 2 * df['BBIBOLL_STD']

            # 获取最新数据
            latest = df.iloc[-1]
            latest_price = latest['收盘']

            bbiboll_data = {
                'BBIBOLL': latest['BBIBOLL'],
                'UPPER': latest['UPPER'],
                'LOWER': latest['LOWER'],
                'latest_price': latest_price
            }

            # 生成评级
            if latest_price > latest['UPPER']:
                keypoint = "×"  # 超买
            elif latest_price < latest['LOWER']:
                keypoint = "√"  # 超卖
            else:
                keypoint = "-"  # 盘整

            return bbiboll_data, keypoint, latest_price

        except Exception as e:
            print(f"BBIBOLL计算异常: {str(e)}")
            return {}, "-", 0.0

    @staticmethod
    def calculate_rsi(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """计算RSI指标"""
        try:
            df = df.copy()

            # 计算价格变化
            delta = df['收盘'].diff()

            # 分离上涨和下跌
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)

            # 计算平均涨跌幅
            avg_gain = gain.rolling(window=Config.RSI_WINDOW).mean()
            avg_loss = loss.rolling(window=Config.RSI_WINDOW).mean()

            # 计算相对强弱指标
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # 获取最新RSI值
            latest_rsi = df['RSI'].iloc[-1]

            # 生成评级
            if latest_rsi > 70:
                keypoint = "×"  # 超买
            elif latest_rsi < 30:
                keypoint = "√"  # 超卖
            else:
                keypoint = "-"  # 盘整

            return df, keypoint

        except Exception as e:
            print(f"RSI计算异常: {str(e)}")
            return df, "-"

    @staticmethod
    def calculate_macd(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """计算MACD指标"""
        try:
            df = df.copy()

            # 计算快速和慢速EMA
            df['EMA_short'] = df['收盘'].ewm(span=Config.MACD_SHORT_WINDOW, adjust=False).mean()
            df['EMA_long'] = df['收盘'].ewm(span=Config.MACD_LONG_WINDOW, adjust=False).mean()

            # 计算DIF
            df['DIF'] = df['EMA_short'] - df['EMA_long']

            # 计算DEA
            df['DEA'] = df['DIF'].ewm(span=Config.MACD_SIGNAL_WINDOW, adjust=False).mean()

            # 计算MACD
            df['MACD'] = 2 * (df['DIF'] - df['DEA'])

            # 获取最新值
            latest_dif = df['DIF'].iloc[-1]
            latest_dea = df['DEA'].iloc[-1]

            # 生成评级
            if latest_dif > latest_dea:
                keypoint = "√"  # 多头
            else:
                keypoint = "×"  # 空头

            return df, keypoint

        except Exception as e:
            print(f"MACD计算异常: {str(e)}")
            return df, "-"

    @staticmethod
    def calculate_kdj(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """计算KDJ指标"""
        try:
            df = df.copy()

            # 计算RSV
            low_list = df['最低'].rolling(window=Config.KDJ_WINDOW, min_periods=1).min()
            high_list = df['最高'].rolling(window=Config.KDJ_WINDOW, min_periods=1).max()

            rsv = (df['收盘'] - low_list) / (high_list - low_list) * 100

            # 计算K值
            df['K'] = rsv.ewm(com=2, adjust=False).mean()

            # 计算D值
            df['D'] = df['K'].ewm(com=2, adjust=False).mean()

            # 计算J值
            df['J'] = 3 * df['K'] - 2 * df['D']

            # 获取最新值
            latest_k = df['K'].iloc[-1]
            latest_d = df['D'].iloc[-1]

            # 生成评级
            if latest_k > latest_d and latest_k < 20:
                keypoint = "√"  # 超卖反转
            elif latest_k < latest_d and latest_k > 80:
                keypoint = "×"  # 超买反转
            else:
                keypoint = "-"  # 盘整

            return df, keypoint

        except Exception as e:
            print(f"KDJ计算异常: {str(e)}")
            return df, "-"

    @staticmethod
    def calculate_sar(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """计算SAR指标（抛物线指标）"""
        try:
            df = df.copy()

            # 初始化SAR列
            df['SAR'] = 0.0
            df['SAR_trend'] = 0

            # 设置初始值
            trend = 1  # 1为上涨趋势，-1为下跌趋势
            af = Config.SAR_STEP  # 加速因子
            ep = df['最高'].iloc[0]  # 极值点
            sar = df['最低'].iloc[0]  # SAR初始值

            # 逐日计算SAR
            for i in range(1, len(df)):
                prev_sar = sar

                if trend == 1:  # 上涨趋势
                    # 更新SAR值
                    sar = prev_sar + af * (ep - prev_sar)
                    # 限制SAR不高于前两天的最低价
                    sar = min(sar, df['最低'].iloc[max(0, i - 2):i].min())

                    # 检查趋势是否改变
                    if sar > df['最低'].iloc[i]:
                        trend = -1
                        sar = ep
                        ep = df['最低'].iloc[i]
                        af = Config.SAR_STEP
                    else:
                        if df['最高'].iloc[i] > ep:
                            ep = df['最高'].iloc[i]
                            af = min(af + Config.SAR_STEP, Config.SAR_MAX)

                else:  # 下跌趋势
                    # 更新SAR值
                    sar = prev_sar - af * (prev_sar - ep)
                    # 限制SAR不低于前两天的最高价
                    sar = max(sar, df['最高'].iloc[max(0, i - 2):i].max())

                    # 检查趋势是否改变
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
                df.loc[df.index[i], 'SAR_trend'] = trend

            # 获取最新值
            latest = df.iloc[-1]
            latest_trend = latest['SAR_trend']
            latest_sar = latest['SAR']
            latest_price = latest['收盘']

            # 生成评级
            if latest_trend == 1 and latest_price > latest_sar:
                keypoint = "√"  # 上涨趋势
            elif latest_trend == -1 and latest_price < latest_sar:
                keypoint = "×"  # 下跌趋势
            else:
                keypoint = "-"  # 趋势转换中

            return df, keypoint

        except Exception as e:
            print(f"SAR计算异常: {str(e)}")
            return df, "-"

    @staticmethod
    def calculate_volume_analysis(df: pd.DataFrame) -> Tuple[Dict[str, float], str]:
        """分析成交量"""
        try:
            df = df.copy()

            # 计算成交量移动平均
            df['VOL_MA5'] = df['成交量'].rolling(window=5).mean()
            df['VOL_MA10'] = df['成交量'].rolling(window=10).mean()

            # 计算量比
            latest_volume = df['成交量'].iloc[-1]
            avg_volume = df['成交量'].rolling(window=5).mean().iloc[-1]
            volume_ratio = latest_volume / avg_volume if avg_volume != 0 else 1

            # 计算量价背离
            price_trend = df['收盘'].iloc[-1] > df['收盘'].iloc[-5]  # 价格是否上涨
            volume_trend = latest_volume > avg_volume  # 成交量是否放大

            # 生成评级
            if price_trend and volume_trend:
                keypoint = "√"  # 量价齐升
            elif not price_trend and volume_trend:
                keypoint = "×"  # 放量下跌
            else:
                keypoint = "-"  # 量价平稳

            volume_data = {
                'latest_volume': latest_volume,
                'volume_ratio': volume_ratio,
                'ma5_volume': df['VOL_MA5'].iloc[-1],
                'ma10_volume': df['VOL_MA10'].iloc[-1]
            }

            return volume_data, keypoint

        except Exception as e:
            print(f"成交量分析异常: {str(e)}")
            return {}, "-"


class StockAnalyzer:
    """增强版股票分析器"""

    def __init__(self, code: str):
        self.code = code
        self.data_fetcher = DataFetcher()
        self.technical_indicators = TechnicalIndicators()
        self.fundamental_factors = FundamentalFactors()
        self.money_flow_factors = MoneyFlowFactors()
        self.sentiment_factors = SentimentFactors()
        self.industry_factors = IndustryFactors()
        self.macro_factors = MacroFactors()
        self.innovation_factors = InnovationFactors()
        self.esg_factors = ESGFactors()
        self.style_factors = StyleFactors()

        self.company_info = self.data_fetcher.get_company_info(code)
        self.history_data = self.data_fetcher.get_history_data(code)
        self.results = {}
        self.analysis_results = {}

    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """运行综合分析"""
        try:
            print(f"\n开始分析股票 {self.code}...")

            # 1. 技术面分析
            print("\n执行技术面分析...")
            self._analyze_technical_factors()

            # 2. 基本面分析
            print("\n执行基本面分析...")
            self._analyze_fundamental_factors()

            # 3. 资金流向分析
            print("\n执行资金流向分析...")
            self._analyze_money_flow_factors()

            # 4. 情绪面分析
            print("\n执行情绪面分析...")
            self._analyze_sentiment_factors()

            # 5. 行业分析
            print("\n执行行业分析...")
            self._analyze_industry_factors()

            # 6. 宏观分析
            print("\n执行宏观分析...")
            self._analyze_macro_factors()

            # 7. 创新能力分析
            print("\n执行创新能力分析...")
            self._analyze_innovation_factors()

            # 8. ESG分析
            print("\n执行ESG分析...")
            self._analyze_esg_factors()

            # 9. 风格分析
            print("\n执行风格分析...")
            self._analyze_style_factors()

            # 10. 生成综合评分和建议
            print("\n生成综合评估...")
            self._generate_comprehensive_recommendations()

            print("\n分析完成！")
            return self.analysis_results

        except Exception as e:
            print(f"分析过程出现错误: {str(e)}")
            return {}

    def _analyze_technical_factors(self):
        """分析技术面因子"""
        _, chip_keypoint, mean_chip = self.data_fetcher.get_chip_distribution(self.code)
        _, bbiboll_keypoint, latest_price = self.technical_indicators.calculate_bbiboll(self.history_data)
        self.history_data, rsi_keypoint = self.technical_indicators.calculate_rsi(self.history_data)
        self.history_data, macd_keypoint = self.technical_indicators.calculate_macd(self.history_data)
        self.history_data, kdj_keypoint = self.technical_indicators.calculate_kdj(self.history_data)
        self.history_data, sar_keypoint = self.technical_indicators.calculate_sar(self.history_data)

        self.results['technical'] = {
            'CHIP': chip_keypoint,
            'BBIBOLL': bbiboll_keypoint,
            'RSI': rsi_keypoint,
            'MACD': macd_keypoint,
            'KDJ': kdj_keypoint,
            'SAR': sar_keypoint
        }

    def _analyze_fundamental_factors(self):
        """分析基本面因子"""
        pe_ratio, pe_key = self.fundamental_factors.calculate_pe(self.code)
        pb_ratio, pb_key = self.fundamental_factors.calculate_pb(self.code)
        roe, roe_key = self.fundamental_factors.calculate_roe(self.code)
        growth, growth_key = self.fundamental_factors.calculate_growth(self.code)
        debt, debt_key = self.fundamental_factors.calculate_debt(self.code)
        cash_flow, cash_key = self.fundamental_factors.calculate_cash_flow(self.code)

        self.results['fundamental'] = {
            'PE': pe_key,
            'PB': pb_key,
            'ROE': roe_key,
            'GROWTH': growth_key,
            'DEBT': debt_key,
            'CASH': cash_key
        }

    def _analyze_money_flow_factors(self):
        """分析资金流向因子"""
        big_money, big_money_key = self.money_flow_factors.calculate_big_money_flow(self.code)
        north_flow, north_key = self.money_flow_factors.analyze_northbound_flow(self.code)
        inst_holding, inst_key = self.money_flow_factors.analyze_institutional_holdings(self.code)
        margin, margin_key = self.money_flow_factors.analyze_margin_trading(self.code)

        self.results['money_flow'] = {
            'BIG_MONEY': big_money_key,
            'NORTHBOUND': north_key,
            'INSTITUTIONAL': inst_key,
            'MARGIN': margin_key
        }

    def _analyze_sentiment_factors(self):
        """分析情绪因子"""
        volatility, vol_key = self.sentiment_factors.calculate_implied_volatility(self.code)
        forum, forum_key = self.sentiment_factors.analyze_forum_sentiment(self.code)
        traders, traders_key = self.sentiment_factors.analyze_top_traders(self.code)
        holders, holders_key = self.sentiment_factors.analyze_shareholder_changes(self.code)

        self.results['sentiment'] = {
            'VOLATILITY': vol_key,
            'FORUM': forum_key,
            'TRADERS': traders_key,
            'HOLDERS': holders_key
        }

    def _analyze_industry_factors(self):
        """分析行业因子"""
        prosperity, pros_key = self.industry_factors.analyze_industry_prosperity(self.code)
        concentration, conc_key = self.industry_factors.analyze_industry_concentration(self.code)
        policy, policy_key = self.industry_factors.analyze_policy_sensitivity(self.code)
        supply_chain, supply_key = self.industry_factors.analyze_supply_chain(self.code)

        self.results['industry'] = {
            'PROSPERITY': pros_key,
            'CONCENTRATION': conc_key,
            'POLICY': policy_key,
            'SUPPLY_CHAIN': supply_key
        }

    def _analyze_macro_factors(self):
        """分析宏观因子"""
        gdp, gdp_key = self.macro_factors.analyze_gdp_impact(self.code)
        inflation, inf_key = self.macro_factors.analyze_inflation_impact(self.code)
        pmi, pmi_key = self.macro_factors.analyze_pmi_impact(self.code)
        interest, int_key = self.macro_factors.analyze_interest_rate_sensitivity(self.code)

        self.results['macro'] = {
            'GDP': gdp_key,
            'INFLATION': inf_key,
            'PMI': pmi_key,
            'INTEREST': int_key
        }

    def _analyze_innovation_factors(self):
        """分析创新因子"""
        rd, rd_key = self.innovation_factors.analyze_rd_investment(self.code)
        patent, patent_key = self.innovation_factors.analyze_patents(self.code)
        talent, talent_key = self.innovation_factors.analyze_talent_density(self.code)
        business, business_key = self.innovation_factors.analyze_business_model(self.code)

        self.results['innovation'] = {
            'RD': rd_key,
            'PATENT': patent_key,
            'TALENT': talent_key,
            'BUSINESS': business_key
        }

    def _analyze_esg_factors(self):
        """分析ESG因子"""
        env, env_key = self.esg_factors.analyze_environmental_impact(self.code)
        social, social_key = self.esg_factors.analyze_social_responsibility(self.code)
        governance, gov_key = self.esg_factors.analyze_governance_quality(self.code)

        self.results['esg'] = {
            'ENVIRONMENTAL': env_key,
            'SOCIAL': social_key,
            'GOVERNANCE': gov_key
        }

    def _analyze_style_factors(self):
        """分析风格因子"""
        size, size_key = self.style_factors.analyze_size_factor(self.code)
        value, value_key = self.style_factors.analyze_value_factor(self.code)
        momentum, momentum_key = self.style_factors.analyze_momentum_factor(self.code)
        volatility, vol_key = self.style_factors.analyze_volatility_factor(self.code)

        self.results['style'] = {
            'SIZE': size_key,
            'VALUE': value_key,
            'MOMENTUM': momentum_key,
            'VOLATILITY': vol_key
        }

    def _calculate_factor_scores(self) -> Dict[str, float]:
        """计算各因子得分"""
        score_map = {'√': 1.0, '-': 0.5, '×': 0.0}
        factor_scores = {}

        for factor_type, factors in self.results.items():
            total_score = sum(score_map[value] * FactorWeight.OVERALL_WEIGHTS[factor_type.upper()]
                              for name, value in factors.items())
            factor_scores[factor_type] = total_score

        return factor_scores

    def _generate_comprehensive_recommendations(self):
        """生成综合评分和建议"""
        # 计算各因子得分
        factor_scores = self._calculate_factor_scores()
        total_score = sum(factor_scores.values())

        # 生成详细分析报告
        self.analysis_results = {
            'code': self.code,
            'company_info': self.company_info,
            'factor_scores': factor_scores,
            'total_score': total_score,
            'recommendations': self._generate_specific_recommendations(factor_scores),
            'risk_assessment': self._assess_risks(factor_scores)
        }

    def _generate_specific_recommendations(self, factor_scores: Dict[str, float]) -> List[str]:
        """生成具体投资建议"""
        recommendations = []

        # 总体建议
        if factor_scores['total_score'] >= 0.8:
            recommendations.append("综合评分优异，建议积极关注")
        elif factor_scores['total_score'] >= 0.6:
            recommendations.append("综合表现良好，可以逐步建仓")
        elif factor_scores['total_score'] >= 0.4:
            recommendations.append("综合表现一般，建议观望")
        else:
            recommendations.append("综合表现欠佳，建议规避")

        # 具体因子建议
        low_score_factors = [factor for factor, score in factor_scores.items() if score < 0.4]
        if low_score_factors:
            recommendations.append(f"需要重点关注以下方面的改善：{', '.join(low_score_factors)}")

        high_score_factors = [factor for factor, score in factor_scores.items() if score > 0.8]
        if high_score_factors:
            recommendations.append(f"具有显著优势的方面：{', '.join(high_score_factors)}")

        return recommendations

    def _assess_risks(self, factor_scores: Dict[str, float]) -> Dict[str, str]:
        """评估各类风险"""
        risk_assessment = {}

        # 评估技术风险
        if factor_scores['technical'] < 0.4:
            risk_assessment['technical_risk'] = "高"
        elif factor_scores['technical'] < 0.6:
            risk_assessment['technical_risk'] = "中"
        else:
            risk_assessment['technical_risk'] = "低"

        # 评估基本面风险
        if factor_scores['fundamental'] < 0.4:
            risk_assessment['fundamental_risk'] = "高"
        elif factor_scores['fundamental'] < 0.6:
            risk_assessment['fundamental_risk'] = "中"
        else:
            risk_assessment['fundamental_risk'] = "低"

        # 评估市场风险
        market_risk_score = (factor_scores['money_flow'] + factor_scores['sentiment']) / 2
        if market_risk_score < 0.4:
            risk_assessment['market_risk'] = "高"
        elif market_risk_score < 0.6:
            risk_assessment['market_risk'] = "中"
        else:
            risk_assessment['market_risk'] = "低"

        # 评估行业风险
        if factor_scores['industry'] < 0.4:
            risk_assessment['industry_risk'] = "高"
        elif factor_scores['industry'] < 0.6:
            risk_assessment['industry_risk'] = "中"
        else:
            risk_assessment['industry_risk'] = "低"

        return risk_assessment

    def print_analysis_report(self):
        """打印分析报告"""
        if not self.analysis_results:
            print("请先运行分析 (run_comprehensive_analysis)")
            return

        print("\n============== 股票综合分析报告 ==============")
        print(f"股票代码：{self.code}")
        print(f"股票名称：{self.company_info['股票简称']}")
        print(f"所属行业：{self.company_info['行业']}")
        print(f"上市时间：{self.company_info['上市时间']}")
        print(f"机构参与度：{self.company_info['机构参与度']}")

        print("\n一、各维度得分：")
        print("-" * 40)
        for factor_type, score in self.analysis_results['factor_scores'].items():
            print(f"{factor_type.upper():<15}: {score:.2f}")
        print(f"\n综合得分：{self.analysis_results['total_score']:.2f}")

        print("\n二、风险评估：")
        print("-" * 40)
        for risk_type, level in self.analysis_results['risk_assessment'].items():
            print(f"{risk_type:<20}: {level}")

        print("\n三、投资建议：")
        print("-" * 40)
        for idx, recommendation in enumerate(self.analysis_results['recommendations'], 1):
            print(f"{idx}. {recommendation}")

        print("\n四、详细指标评级：")
        print("-" * 40)
        for factor_type, factors in self.results.items():
            print(f"\n{factor_type.upper()}类指标：")
            for name, value in factors.items():
                print(f"{name:<15}: {value}")

        print("\n五、技术指标详情：")
        print("-" * 40)
        latest_data = self.history_data.iloc[-1]
        print(f"最新收盘价：{latest_data['收盘']:.2f}")
        print(f"RSI指标：{latest_data['RSI']:.2f}")
        if 'MACD' in latest_data:
            print(f"MACD指标：{latest_data['MACD']:.4f}")
        if 'KDJ' in latest_data:
            print(f"KDJ指标：K={latest_data['K']:.2f}, D={latest_data['D']:.2f}, J={latest_data['J']:.2f}")

        print("\n六、估值与成长：")
        print("-" * 40)
        print(f"市盈率(PE)：{self.company_info.get('市盈率', '未知')}")
        print(f"市净率(PB)：{self.company_info.get('市净率', '未知')}")
        print(f"营收增长率：{self.company_info.get('营收增长率', '未知')}")

        print("\n七、ESG评估：")
        print("-" * 40)
        esg_factors = self.results.get('esg', {})
        print(f"环境(E)：{esg_factors.get('ENVIRONMENTAL', '未知')}")
        print(f"社会(S)：{esg_factors.get('SOCIAL', '未知')}")
        print(f"治理(G)：{esg_factors.get('GOVERNANCE', '未知')}")

        print("\n============== 报告结束 ==============")


def main():
    """主程序"""
    print("欢迎使用增强版股票分析系统")
    print("本系统整合了技术面、基本面、资金面、情绪面等多维度分析")
    print("支持ESG分析、创新能力评估、行业分析等深度研究")

    while True:
        print("\n" + "=" * 50)
        stock_code = input("\n请输入6位股票代码（输入q退出）: ")

        if stock_code.lower() == 'q':
            print("\n感谢使用股票分析系统！")
            break

        try:
            # 创建分析器实例
            analyzer = StockAnalyzer(stock_code)

            # 运行综合分析
            print("\n开始进行多维度分析，请稍候...")
            analyzer.run_comprehensive_analysis()

            # 打印分析报告
            analyzer.print_analysis_report()

            # 询问是否需要保存报告
            save_report = input("\n是否需要保存分析报告？(y/n): ")
            if save_report.lower() == 'y':
                file_name = f"stock_analysis_{stock_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                try:
                    with open(file_name, 'w', encoding='utf-8') as f:
                        # 重定向标准输出到文件
                        import sys
                        original_stdout = sys.stdout
                        sys.stdout = f
                        analyzer.print_analysis_report()
                        sys.stdout = original_stdout
                    print(f"\n报告已保存至: {file_name}")
                except Exception as e:
                    print(f"保存报告时出现错误: {str(e)}")

        except Exception as e:
            print(f"\n分析过程出现错误: {str(e)}")
            print("请检查股票代码是否正确，或稍后重试")
            continue

        # 询问是否继续分析其他股票
        continue_analysis = input("\n是否继续分析其他股票？(y/n): ")
        if continue_analysis.lower() != 'y':
            print("\n感谢使用股票分析系统！")
            break


if __name__ == "__main__":
    main()
