# 自动安装缺失的库
import subprocess
import sys

required_packages = {
    'akshare': 'akshare',
    'pandas': 'pandas',
    'numpy': 'numpy',
    'torch': 'torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124',
    'sklearn': 'scikit-learn'
}

for package, pip_name in required_packages.items():
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pip_name])

# 现在正常导入所有库
import akshare as ak
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
from functools import wraps


########################
### 装饰器 ###
### 输出当前执行的函数名 ###
########################
def func_name(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"\n{func.__name__}>>>{'=' * 30}")
        result = func(*args, **kwargs)
        return result

    return wrapper


#####################
###  公司基本信息  ###
#####################
@func_name
def get_company_info(code):
    # 获取市场前缀
    symbol = f"{code}"
    info_dict = {}

    try:
        # 基础信息
        base_info = ak.stock_individual_info_em(symbol=symbol)
        info_dict['股票简称'] = base_info.loc[base_info['item'] == '股票简称', 'value'].values[0]
        info_dict['行业'] = base_info.loc[base_info['item'] == '行业', 'value'].values[0]
        info_dict['上市时间'] = base_info.loc[base_info['item'] == '上市时间', 'value'].values[0]
    except Exception as e:
        print(f"基础信息获取失败: {str(e)}")
        info_dict.update({'股票简称': '未知', '行业': '未知', '上市时间': '未知'})

    try:
        # 发行信息
        stock_ipo_info_df = ak.stock_ipo_info(stock=symbol)
        if not stock_ipo_info_df.empty:
            info_dict['发行价'] = stock_ipo_info_df.loc[stock_ipo_info_df['item'] == '发行价(元)', 'value'].values[0]
        else:
            info_dict['发行价'] = '暂无数据'
    except Exception as e:
        print(f"发行价获取失败: {str(e)}")
        info_dict['发行价'] = '暂无数据'

    try:
        # 分红信息
        stock_history_dividend_df = ak.stock_history_dividend()
        dividend_info = stock_history_dividend_df[stock_history_dividend_df['代码'] == code]
        # 如果找到记录，获取分红次数列的值；如果没找到记录，则为0
        info_dict['分红次数'] = dividend_info['分红次数'].iloc[0] if not dividend_info.empty else 0
    except Exception as e:
        print(f"分红信息获取失败: {str(e)}")
        info_dict['分红次数'] = 0

    try:
        # 机构参与度
        jg_info = ak.stock_comment_detail_zlkp_jgcyd_em(symbol=symbol)
        info_dict['机构参与度'] = f"{jg_info['机构参与度'].values[0]}%"
    except Exception as e:
        print(f"机构参与度获取失败: {str(e)}")
        info_dict['机构参与度'] = '暂无数据'

    try:
        # 市场成本
        cost_info = ak.stock_comment_detail_scrd_cost_em(symbol=symbol)
        info_dict['市场成本'] = f"{cost_info['市场成本'].values[0]}元"
    except Exception as e:
        print(f"市场成本获取失败: {str(e)}")
        info_dict['市场成本'] = '暂无数据'

    # 格式化输出
    print(f"----- {code} 公司基本信息 -----")
    print(f"股票简称：{info_dict['股票简称']}")
    print(f"所属行业：{info_dict['行业']}")
    print(f"上市时间：{info_dict['上市时间']}")
    print(f"发行价格：{info_dict['发行价']}")
    print(f"分红次数：{info_dict['分红次数']}次")
    print(f"机构参与：{info_dict['机构参与度']}")
    print(f"成本均价：{info_dict['市场成本']}")
    print(f"-----------------------------")

    return info_dict


#####################
###  获取历史数据   ###
#####################
@func_name
def get_history_data(code):
    symbol = f"{code}"
    days = 365
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days)).strftime("%Y%m%d")
    df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
    print(f"历史{days}天数据获取完成，共获取{len(df)}条记录")
    return df


#####################
###   获取筹码分布  ###
#####################
@func_name
def get_chip_distribution(code):
    symbol = f"{code}"
    df = ak.stock_cyq_em(symbol=symbol)
    latest_chip = df.iloc[-1].to_dict()
    print(
        f"最新交易日筹码分布：获利比例={latest_chip['获利比例'] * 100:.2f}% 70集中度={(latest_chip['70集中度'] * 100):.2f}%")
    return latest_chip


#####################
###  BBIBOLL计算   ###
#####################
@func_name
def calculate_bbiboll(df):
    df['MA3'] = df['收盘'].rolling(3).mean()
    df['MA6'] = df['收盘'].rolling(6).mean()
    df['MA12'] = df['收盘'].rolling(12).mean()
    df['MA24'] = df['收盘'].rolling(24).mean()

    df['BBIBOLL'] = (df['MA3'] + df['MA6'] + df['MA12'] + df['MA24']) / 4
    df['UPPER'] = df['BBIBOLL'] + 2 * df['BBIBOLL'].rolling(11).std()
    df['LOWER'] = df['BBIBOLL'] - 2 * df['BBIBOLL'].rolling(11).std()

    latest = df.iloc[-1][['BBIBOLL', 'UPPER', 'LOWER']].to_dict()
    print(f"最新BBIBOLL值: mid={latest['BBIBOLL']:.2f} upper={latest['UPPER']:.2f} lower={latest['LOWER']:.2f}")
    return latest


#####################
###  LSTM预测   ###
#####################
class EnhancedLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 双向LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)

        # LSTM输出
        lstm_out, _ = self.lstm(x, (h0, c0))

        # 注意力权重
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)

        # 输出层
        out = self.fc(context)
        return out


@func_name
def enhanced_predict(df, target_col='收盘'):
    """综合多特征、参数搜索和高级LSTM结构的增强预测函数"""
    # 准备多特征数据
    feature_cols = ['开盘', '最高', '最低', '收盘', '成交量']
    target_idx = feature_cols.index(target_col)

    # 数据标准化
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_cols])

    # 参数网格配置
    param_grid = {
        'hidden_size': [64, 128],
        'num_layers': [2, 3],
        'dropout': [0.2, 0.3],
        'look_back': [30, 60],
        'learning_rate': [0.001, 0.0005],
        'epochs': [150],
        'patience': [15]
    }

    best_rmse = float('inf')
    best_params = {}
    best_model_state = None
    best_scaler = None
    best_look_back = 0
    best_predictions = None
    best_actuals = None

    # 参数搜索
    for params in ParameterGrid(param_grid):
        look_back = params['look_back']

        # 创建序列数据集
        X, y = [], []
        for i in range(len(scaled_data) - look_back):
            X.append(scaled_data[i:i + look_back])
            y.append(scaled_data[i + look_back, target_idx])

        if len(X) < 10:  # 跳过数据量不足的参数组合
            continue

        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y).view(-1, 1)

        # 模型初始化
        model = EnhancedLSTM(
            input_size=len(feature_cols),
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout']
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        criterion = nn.MSELoss()

        # 训练循环
        best_train_loss = float('inf')
        stop_counter = 0

        for epoch in range(params['epochs']):
            model.train()
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # 早停机制
            if loss.item() < best_train_loss:
                best_train_loss = loss.item()
                stop_counter = 0
            else:
                stop_counter += 1
                if stop_counter >= params['patience']:
                    break

        # 评估模型
        model.eval()
        with torch.no_grad():
            preds = model(X)
            n_samples = len(preds)

            # 反标准化处理
            dummy_preds = np.zeros((n_samples, len(feature_cols)))
            dummy_preds[:, target_idx] = preds.numpy().flatten()
            preds_denorm = scaler.inverse_transform(dummy_preds)[:, target_idx]

            dummy_y = np.zeros((n_samples, len(feature_cols)))
            dummy_y[:, target_idx] = y.numpy().flatten()
            y_denorm = scaler.inverse_transform(dummy_y)[:, target_idx]

            rmse = np.sqrt(mean_squared_error(y_denorm, preds_denorm))

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params
                best_model_state = model.state_dict()
                best_scaler = scaler
                best_look_back = look_back
                best_predictions = preds_denorm
                best_actuals = y_denorm

    # 最终预测
    final_model = EnhancedLSTM(
        input_size=len(feature_cols),
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout']
    )
    final_model.load_state_dict(best_model_state)
    final_model.eval()

    last_sequence = scaled_data[-best_look_back:]
    input_tensor = torch.FloatTensor(last_sequence).view(1, best_look_back, -1)

    with torch.no_grad():
        next_pred_scaled = final_model(input_tensor).item()
        dummy_next = np.zeros((1, len(feature_cols)))
        dummy_next[0, target_idx] = next_pred_scaled
        next_pred = best_scaler.inverse_transform(dummy_next)[0, target_idx]

    # 输出结果
    print(f"\n{target_col}价预测最佳参数: {best_params}")
    print(f"训练集RMSE: {best_rmse:.2f}")
    print(f"实际值={best_actuals[-1]:.2f} 预测值={best_predictions[-1]:.2f} (最新数据点)")
    print(f"预测下一个交易日的{target_col}价可能为：{next_pred:.2f} 元")

    return {
        'best_params': best_params,
        'rmse': best_rmse,
        'last_actual': best_actuals[-1],
        'last_pred': best_predictions[-1],
        'next_pred': next_pred
    }


if __name__ == '__main__':
    stock_code = input("请输入6位股票代码: ")
    company_info = get_company_info(stock_code)
    history_df = get_history_data(stock_code)
    get_chip_distribution(stock_code)
    calculate_bbiboll(history_df)
    # 执行预测
    print("增强版多特征LSTM预测结果：")
    for col in ['开盘', '收盘', '最低', '最高']:
        enhanced_predict(history_df, col)
