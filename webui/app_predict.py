import os
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.utils
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import warnings
import datetime
import requests
import threading
warnings.filterwarnings('ignore')
from typing import Optional, Dict, Any, Tuple, List


# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
# 导入模型管理模块
import app_model

# 创建全局预测锁，确保同一时间只能有一个预测任务运行
prediction_lock = threading.Lock()

app = Flask(__name__)
CORS(app)

def save_prediction_results(file_path, prediction_type, prediction_results, actual_data, input_data, prediction_params):
    """Save prediction results to file"""
    try:
        # Create prediction results directory
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'prediction_{timestamp}.json'
        filepath = os.path.join(results_dir, filename)
        
        # Prepare data for saving
        save_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'file_path': file_path,
            'prediction_type': prediction_type,
            'prediction_params': prediction_params,
            'input_data_summary': {
                'rows': len(input_data),
                'columns': list(input_data.columns),
                'price_range': {
                    'open': {'min': float(input_data['open'].min()), 'max': float(input_data['open'].max())},
                    'high': {'min': float(input_data['high'].min()), 'max': float(input_data['high'].max())},
                    'low': {'min': float(input_data['low'].min()), 'max': float(input_data['low'].max())},
                    'close': {'min': float(input_data['close'].min()), 'max': float(input_data['close'].max())}
                },
                'last_values': {
                    'open': float(input_data['open'].iloc[-1]),
                    'high': float(input_data['high'].iloc[-1]),
                    'low': float(input_data['low'].iloc[-1]),
                    'close': float(input_data['close'].iloc[-1])
                }
            },
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'analysis': {}
        }
        
        # If actual data exists, perform comparison analysis
        if actual_data and len(actual_data) > 0:
            # Calculate continuity analysis
            if len(prediction_results) > 0 and len(actual_data) > 0:
                last_pred = prediction_results[0]  # First prediction point
                first_actual = actual_data[0]      # First actual point
                
                save_data['analysis']['continuity'] = {
                    'last_prediction': {
                        'open': last_pred['open'],
                        'high': last_pred['high'],
                        'low': last_pred['low'],
                        'close': last_pred['close']
                    },
                    'first_actual': {
                        'open': first_actual['open'],
                        'high': first_actual['high'],
                        'low': first_actual['low'],
                        'close': first_actual['close']
                    },
                    'gaps': {
                        'open_gap': abs(last_pred['open'] - first_actual['open']),
                        'high_gap': abs(last_pred['high'] - first_actual['high']),
                        'low_gap': abs(last_pred['low'] - first_actual['low']),
                        'close_gap': abs(last_pred['close'] - first_actual['close'])
                    },
                    'gap_percentages': {
                        'open_gap_pct': (abs(last_pred['open'] - first_actual['open']) / first_actual['open']) * 100,
                        'high_gap_pct': (abs(last_pred['high'] - first_actual['high']) / first_actual['high']) * 100,
                        'low_gap_pct': (abs(last_pred['low'] - first_actual['low']) / first_actual['low']) * 100,
                        'close_gap_pct': (abs(last_pred['close'] - first_actual['close']) / first_actual['close']) * 100
                    }
                }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Prediction results saved to: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"Failed to save prediction results: {e}")
        return None

def create_prediction_chart(df, pred_df, lookback, pred_len, actual_df=None, historical_start_idx=0):
    """Create prediction chart"""
    # Use specified historical data start position, not always from the beginning of df
    if historical_start_idx + lookback + pred_len <= len(df):
        # Display lookback historical points + pred_len prediction points starting from specified position
        historical_df = df.iloc[historical_start_idx:historical_start_idx+lookback]
        prediction_range = range(historical_start_idx+lookback, historical_start_idx+lookback+pred_len)
    else:
        # If data is insufficient, adjust to maximum available range
        available_lookback = min(lookback, len(df) - historical_start_idx)
        available_pred_len = min(pred_len, max(0, len(df) - historical_start_idx - available_lookback))
        historical_df = df.iloc[historical_start_idx:historical_start_idx+available_lookback]
        prediction_range = range(historical_start_idx+available_lookback, historical_start_idx+available_lookback+available_pred_len)
    
    # Create chart
    fig = go.Figure()
    
    # Add historical data (candlestick chart)
    fig.add_trace(go.Candlestick(
        x=historical_df['timestamps'] if 'timestamps' in historical_df.columns else historical_df.index,
        open=historical_df['open'],
        high=historical_df['high'],
        low=historical_df['low'],
        close=historical_df['close'],
        name='Historical Data (400 data points)',
        increasing_line_color='#26A69A',
        decreasing_line_color='#EF5350'
    ))
    
    # Add prediction data (candlestick chart)
    if pred_df is not None and len(pred_df) > 0:
        # Calculate prediction data timestamps - ensure continuity with historical data
        if 'timestamps' in df.columns and len(historical_df) > 0:
            # Start from the last timestamp of historical data, create prediction timestamps with the same time interval
            last_timestamp = historical_df['timestamps'].iloc[-1]
            time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0] if len(df) > 1 else pd.Timedelta(hours=1)
            
            pred_timestamps = pd.date_range(
                start=last_timestamp + time_diff,
                periods=len(pred_df),
                freq=time_diff
            )
        else:
            # 如果没有时间戳，使用索引
            pred_timestamps = range(len(historical_df), len(historical_df) + len(pred_df))
        
        fig.add_trace(go.Candlestick(
            x=pred_timestamps,
            open=pred_df['open'],
            high=pred_df['high'],
            low=pred_df['low'],
            close=pred_df['close'],
            name=f'预测数据 ({len(pred_df)} 个数据点)',
            increasing_line_color='#66BB6A',
            decreasing_line_color='#FF7043'
        ))
    
    # 添加实际数据用于对比（如果存在）
    if actual_df is not None and len(actual_df) > 0:
        # 实际数据应该与预测数据在相同的时间段
        if 'timestamps' in df.columns:
            # 实际数据应该使用与预测数据相同的时间戳以确保时间对齐
            if 'pred_timestamps' in locals():
                actual_timestamps = pred_timestamps
            else:
                # 如果没有预测时间戳，从历史数据的最后时间戳计算
                if len(historical_df) > 0:
                    last_timestamp = historical_df['timestamps'].iloc[-1]
                    time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0] if len(df) > 1 else pd.Timedelta(hours=1)
                    actual_timestamps = pd.date_range(
                        start=last_timestamp + time_diff,
                        periods=len(actual_df),
                        freq=time_diff
                    )
                else:
                    actual_timestamps = range(len(historical_df), len(historical_df) + len(actual_df))
        else:
            actual_timestamps = range(len(historical_df), len(historical_df) + len(actual_df))
        
        fig.add_trace(go.Candlestick(
            x=actual_timestamps,
            open=actual_df['open'],
            high=actual_df['high'],
            low=actual_df['low'],
            close=actual_df['close'],
            name=f'实际数据 ({len(actual_df)} 个数据点)',
            increasing_line_color='#FF9800',
            decreasing_line_color='#F44336'
        ))
    
    # 更新图表布局
    fig.update_layout(
        title='Kronos 金融预测结果 - 历史数据 + 预测数据 vs 实际数据',
        xaxis_title='时间',
        yaxis_title='价格',
        template='plotly_white',
        height=600,
        showlegend=True
    )
    
    # 确保x轴时间连续性
    if 'timestamps' in historical_df.columns:
        # 获取所有时间戳并排序
        all_timestamps = []
        if len(historical_df) > 0:
            all_timestamps.extend(historical_df['timestamps'])
        if 'pred_timestamps' in locals():
            all_timestamps.extend(pred_timestamps)
        if 'actual_timestamps' in locals():
            all_timestamps.extend(actual_timestamps)
        
        if all_timestamps:
            all_timestamps = sorted(all_timestamps)
            fig.update_xaxes(
                range=[all_timestamps[0], all_timestamps[-1]],
                rangeslider_visible=False,
                type='date'
            )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def predict_from_file(file_path: str, lookback: int = 400, pred_len: int = 120, 
                     temperature: float = 0.7, top_p: float = 0.9, 
                     sample_count: int = 1, start_date: Optional[str] = None) -> Dict[str, Any]:
    """
    基于文件进行预测分析
    
    功能说明:
    - 从文件加载数据并进行预测
    - 支持自定义时间段选择
    - 生成预测图表和结果分析
    
    Args:
        file_path (str): 数据文件路径
        lookback (int): 历史数据回看长度，默认400
        pred_len (int): 预测长度，默认120
        temperature (float): 预测温度参数，默认0.7
        top_p (float): 预测top_p参数，默认0.9
        sample_count (int): 采样次数，默认1
        start_date (Optional[str]): 开始日期，格式为ISO字符串
        
    Returns:
        Dict[str, Any]: 预测结果字典
    """
    try:
        if not file_path:
            return {
                'success': False,
                'error': '文件路径不能为空'
            }
        
        # 加载数据 - 使用app_model模块的函数
        if not app_model:
            return {
                'success': False,
                'error': 'app_model模块未加载，无法进行数据处理'
            }
        
        df, error = app_model.load_data_file(file_path)
        if error:
            return {
                'success': False,
                'error': error
            }
        
        if len(df) < lookback:
            return {
                'success': False,
                'error': f'数据长度不足，需要至少 {lookback} 行数据，当前只有 {len(df)} 行'
            }
        
        # 执行预测
        if app_model and hasattr(app_model, 'predictor') and app_model.predictor is not None:
            try:
                # 使用真实的Kronos模型
                # 只使用必要的列：OHLCV，排除amount
                required_cols = ['open', 'high', 'low', 'close']
                if 'volume' in df.columns:
                    required_cols.append('volume')
                
                # 处理时间段选择
                if start_date:
                    # 自定义时间段 - 修复逻辑：使用选定窗口内的数据
                    start_dt = pd.to_datetime(start_date)
                    
                    # 找到开始时间之后的数据
                    mask = df['timestamps'] >= start_dt
                    time_range_df = df[mask]
                    
                    # 确保有足够的数据：lookback + pred_len
                    if len(time_range_df) < lookback + pred_len:
                        return {
                            'success': False,
                            'error': f'从开始时间 {start_dt.strftime("%Y-%m-%d %H:%M")} 开始的数据不足，需要至少 {lookback + pred_len} 个数据点，当前只有 {len(time_range_df)} 个可用'
                        }
                    
                    # 使用选定窗口内的前lookback个数据点进行预测
                    x_df = time_range_df.iloc[:lookback][required_cols]
                    x_timestamp = time_range_df.iloc[:lookback]['timestamps']
                    
                    # 使用选定窗口内的后pred_len个数据点作为实际值
                    y_timestamp = time_range_df.iloc[lookback:lookback+pred_len]['timestamps']
                    
                    # 计算实际时间段长度
                    start_timestamp = time_range_df['timestamps'].iloc[0]
                    end_timestamp = time_range_df['timestamps'].iloc[lookback+pred_len-1]
                    time_span = end_timestamp - start_timestamp
                    
                    prediction_type = f"Kronos模型预测（选定窗口内：前{lookback}个数据点用于预测，后{pred_len}个数据点用于对比，时间跨度：{time_span}）"
                else:
                    # 使用最新数据
                    x_df = df.iloc[-lookback:][required_cols]
                    x_timestamp = df.iloc[-lookback:]['timestamps']
                    
                    # 生成未来时间戳用于预测
                    if len(df) > 1:
                        time_diff = df['timestamps'].iloc[-1] - df['timestamps'].iloc[-2]
                        last_timestamp = df['timestamps'].iloc[-1]
                        y_timestamp = pd.date_range(
                            start=last_timestamp + time_diff,
                            periods=pred_len,
                            freq=time_diff
                        )
                        y_timestamp = pd.Series(y_timestamp, name='timestamps')
                    else:
                        # 如果只有一个数据点，使用默认的小时间隔
                        last_timestamp = df['timestamps'].iloc[-1]
                        y_timestamp = pd.date_range(
                            start=last_timestamp + pd.Timedelta(hours=1),
                            periods=pred_len,
                            freq='1H'
                        )
                        y_timestamp = pd.Series(y_timestamp, name='timestamps')
                    
                    prediction_type = "Kronos模型预测（最新数据）"
                
                # 确保时间戳是Series格式，不是DatetimeIndex，以避免Kronos模型中的.dt属性错误
                if isinstance(x_timestamp, pd.DatetimeIndex):
                    x_timestamp = pd.Series(x_timestamp, name='timestamps')
                if isinstance(y_timestamp, pd.DatetimeIndex):
                    y_timestamp = pd.Series(y_timestamp, name='timestamps')
                
                # 使用锁保护模型预测调用
                if not prediction_lock.acquire(blocking=False):
                    return {
                        'success': False,
                        'error': '预测服务正忙，请稍后再试。同一时间只能运行一个预测任务。'
                    }
                
                try:
                    # 调用模型预测
                    pred_df = app_model.predictor.predict(
                        df=x_df,
                        x_timestamp=x_timestamp,
                        y_timestamp=y_timestamp,
                        pred_len=pred_len,
                        T=temperature,
                        top_p=top_p,
                        sample_count=sample_count
                    )
                finally:
                    prediction_lock.release()
                
            except Exception as e:
                return {
                    'success': False,
                    'error': f'Kronos模型预测失败: {str(e)}'
                }
        else:
            return {
                'success': False,
                'error': 'Kronos模型未加载，请先加载模型'
            }
        
        # 准备实际数据用于对比（如果存在）
        actual_data = []
        actual_df = None
        
        if start_date:  # 自定义时间段
            # 修复逻辑：使用选定窗口内的数据
            # 预测使用选定窗口内的前400个数据点
            # 实际数据应该是选定窗口内的后120个数据点
            start_dt = pd.to_datetime(start_date)
            
            # 找到从start_date开始的数据
            mask = df['timestamps'] >= start_dt
            time_range_df = df[mask]
            
            if len(time_range_df) >= lookback + pred_len:
                # 获取选定窗口内的后120个数据点作为实际值
                actual_df = time_range_df.iloc[lookback:lookback+pred_len]
                
                for i, (_, row) in enumerate(actual_df.iterrows()):
                    timestamp_obj = row['timestamps']
                    timestamp_seconds = int(timestamp_obj.timestamp())
                    datetime_str = timestamp_obj.strftime("%Y-%m-%d %H:%M:%S")
                    
                    actual_data.append({
                        'timestamp': timestamp_seconds,  # 秒级时间戳
                        'datetime': datetime_str,        # 格式化时间
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume']) if 'volume' in row else 0,
                        'amount': float(row['amount']) if 'amount' in row else 0
                    })
        
        # 创建图表 - 传递历史数据起始位置
        if start_date:
            # 自定义时间段：在原始df中找到历史数据的起始位置
            start_dt = pd.to_datetime(start_date)
            mask = df['timestamps'] >= start_dt
            historical_start_idx = df[mask].index[0] if len(df[mask]) > 0 else 0
        else:
            # 最新数据：从最后lookback个数据点开始
            historical_start_idx = max(0, len(df) - lookback)
        
        chart_json = create_prediction_chart(df, pred_df, lookback, pred_len, actual_df, historical_start_idx)
        
        # 准备预测结果数据 - 修复时间戳计算逻辑
        prediction_results = []
        for i, (_, row) in enumerate(pred_df.iterrows()):
            if i < len(y_timestamp):
                timestamp_obj = y_timestamp.iloc[i]
                timestamp_seconds = int(timestamp_obj.timestamp())
                datetime_str = timestamp_obj.strftime("%Y-%m-%d %H:%M:%S")
            else:
                # 备用时间戳
                timestamp_seconds = None
                datetime_str = f"T{i}"
            
            prediction_results.append({
                'timestamp': timestamp_seconds,  # 秒级时间戳
                'datetime': datetime_str,        # 格式化时间
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']) if 'volume' in row else 0,
                'amount': float(row['amount']) if 'amount' in row else 0
            })
        
        # 保存预测结果到文件
        try:
            save_prediction_results(
                file_path=file_path,
                prediction_type=prediction_type,
                prediction_results=prediction_results,
                actual_data=actual_data,
                input_data=x_df,
                prediction_params={
                    'lookback': lookback,
                    'pred_len': pred_len,
                    'temperature': temperature,
                    'top_p': top_p,
                    'sample_count': sample_count,
                    'start_date': start_date if start_date else 'latest'
                }
            )
        except Exception as e:
            print(f"保存预测结果失败: {e}")
        
        return {
            'success': True,
            'prediction_type': prediction_type,
            'chart': chart_json,
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'has_comparison': len(actual_data) > 0,
            'message': f'预测完成，生成了 {pred_len} 个预测点' + (f'，包含 {len(actual_data)} 个实际数据点用于对比' if len(actual_data) > 0 else '')
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'预测失败: {str(e)}'
        }

def predict_from_kline(kline_data: List[Dict], lookback: int = 400, pred_len: int = 120, 
                      temperature: float = 0.7, top_p: float = 0.9, 
                      sample_count: int = 1) -> Dict[str, Any]:
    """
    基于K线数据进行直接预测
    
    功能说明:
    - 接受K线数据列表进行预测
    - 自动处理时间戳和数据格式
    - 生成预测图表和结果
    
    Args:
        kline_data (List[Dict]): K线数据列表
        lookback (int): 历史数据回看长度，默认400
        pred_len (int): 预测长度，默认120
        temperature (float): 预测温度参数，默认0.7
        top_p (float): 预测top_p参数，默认0.9
        sample_count (int): 采样次数，默认1
        
    Returns:
        Dict[str, Any]: 预测结果字典
    """
    try:
        if not kline_data or len(kline_data) == 0:
            return {
                'success': False,
                'error': 'K线数据不能为空'
            }
        
        # 将K线数据转换为DataFrame
        try:
            df = pd.DataFrame(kline_data)
            
            # 检查必需的列
            required_cols_base = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols_base):
                return {
                    'success': False,
                    'error': f'缺少必需的列: {required_cols_base}'
                }
            
            # 处理时间戳列
            timestamp_processed = False
            if 'timestamp' in df.columns:
                df['timestamps'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
                # 检查是否创建了有效的时间戳
                if not df['timestamps'].isna().all():
                    timestamp_processed = True
            
            if not timestamp_processed and 'datetime' in df.columns:
                df['timestamps'] = pd.to_datetime(df['datetime'], errors='coerce')
                if not df['timestamps'].isna().all():
                    timestamp_processed = True
            
            if not timestamp_processed and 'time' in df.columns:
                df['timestamps'] = pd.to_datetime(df['time'], unit='s', errors='coerce')
                if not df['timestamps'].isna().all():
                    timestamp_processed = True
            
            if not timestamp_processed:
                # 如果没有有效的时间戳列或所有时间戳都无效，创建一个默认的小时间隔时间序列
                df['timestamps'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1H')
            
            # 确保数值列是数值类型
            for col in required_cols_base:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 处理成交量列（可选）
            if 'volume' in df.columns:
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            
            # 处理成交额列（可选）
            if 'amount' in df.columns:
                df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            
            # 只从OHLC列中删除包含NaN值的行
            # 不要仅仅因为时间戳转换失败就删除行
            ohlc_cols = ['open', 'high', 'low', 'close']
            if 'volume' in df.columns:
                ohlc_cols.append('volume')
            if 'amount' in df.columns:
                ohlc_cols.append('amount')
            
            df = df.dropna(subset=ohlc_cols)
            
            # 如果由于无效的OHLC数据导致所有行都被删除，返回错误
            if len(df) == 0:
                return {
                    'success': False,
                    'error': '所有K线数据都包含无效的OHLC值'
                }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'处理K线数据失败: {str(e)}'
            }
        
        if len(df) < lookback:
            return {
                'success': False,
                'error': f'数据长度不足，需要至少 {lookback} 行数据，当前只有 {len(df)} 行'
            }
        
        # 执行预测
        if app_model and hasattr(app_model, 'predictor') and app_model.predictor is not None:
            try:
                # 使用真实的Kronos模型
                # 只使用必要的列：OHLCV，排除amount
                required_cols = ['open', 'high', 'low', 'close']
                if 'volume' in df.columns:
                    required_cols.append('volume')
                
                # 使用最后lookback个数据点进行预测
                x_df = df.iloc[-lookback:][required_cols]
                x_timestamp = df.iloc[-lookback:]['timestamps']
                
                # 生成未来时间戳用于预测
                if len(df) > 1:
                    time_diff = df['timestamps'].iloc[-1] - df['timestamps'].iloc[-2]
                    last_timestamp = df['timestamps'].iloc[-1]
                    y_timestamp = pd.date_range(
                        start=last_timestamp + time_diff,
                        periods=pred_len,
                        freq=time_diff
                    )
                    y_timestamp = pd.Series(y_timestamp, name='timestamps')
                else:
                    # 如果只有一个数据点，使用默认的小时间隔
                    last_timestamp = df['timestamps'].iloc[-1]
                    y_timestamp = pd.date_range(
                        start=last_timestamp + pd.Timedelta(hours=1),
                        periods=pred_len,
                        freq='1H'
                    )
                    y_timestamp = pd.Series(y_timestamp, name='timestamps')
                
                # 确保时间戳是Series格式
                if isinstance(x_timestamp, pd.DatetimeIndex):
                    x_timestamp = pd.Series(x_timestamp, name='timestamps')
                if isinstance(y_timestamp, pd.DatetimeIndex):
                    y_timestamp = pd.Series(y_timestamp, name='timestamps')
                
                # 使用锁保护模型预测调用
                if not prediction_lock.acquire(blocking=False):
                    return {
                        'success': False,
                        'error': '预测服务正忙，请稍后再试。同一时间只能运行一个预测任务。'
                    }
                
                try:
                    # 调用模型预测
                    pred_df = app_model.predictor.predict(
                        df=x_df,
                        x_timestamp=x_timestamp,
                        y_timestamp=y_timestamp,
                        pred_len=pred_len,
                        T=temperature,
                        top_p=top_p,
                        sample_count=sample_count
                    )
                finally:
                    prediction_lock.release()
                
                prediction_type = f"Kronos模型预测（直接K线输入，{lookback}个历史点）"
                
            except Exception as e:
                return {
                    'success': False,
                    'error': f'Kronos模型预测失败: {str(e)}'
                }
        else:
            return {
                'success': False,
                'error': 'Kronos模型未加载，请先加载模型'
            }
        
        # 使用输入数据创建图表
        chart_json = create_prediction_chart(df, pred_df, lookback, pred_len, None, max(0, len(df) - lookback))
        
        # 准备预测结果数据
        prediction_results = []
        for i, (_, row) in enumerate(pred_df.iterrows()):
            if i < len(y_timestamp):
                timestamp_obj = y_timestamp.iloc[i]
                timestamp_seconds = int(timestamp_obj.timestamp())
                datetime_str = timestamp_obj.strftime("%Y-%m-%d %H:%M:%S")
            else:
                # 备用时间戳
                timestamp_seconds = None
                datetime_str = f"T{i}"
            
            prediction_results.append({
                'timestamp': timestamp_seconds,  # 秒级时间戳
                'datetime': datetime_str,        # 格式化时间
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']) if 'volume' in row else 0,
                'amount': float(row['amount']) if 'amount' in row else 0
            })
        
        return {
            'success': True,
            'prediction_type': prediction_type,
            'chart': chart_json,
            'prediction_results': prediction_results,
            'actual_data': [],
            'has_comparison': False,
            'input_data_info': {
                'total_points': len(df),
                'used_points': lookback,
                'timeframe': 'auto-detected',
                'price_range': {
                    'min': float(df[['open', 'high', 'low', 'close']].min().min()),
                    'max': float(df[['open', 'high', 'low', 'close']].max().max())
                }
            },
            'message': f'预测完成，使用了 {lookback} 个K线数据点，生成了 {pred_len} 个预测点'
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'预测失败: {str(e)}'
        }


def test_prediction_functions():
    """
    测试预测功能
    
    测试内容:
    - 测试模型状态检查
    - 测试数据文件预测功能
    - 测试K线数据预测功能
    - 测试图表生成功能
    """
    print("\n" + "="*60)
    print("测试预测功能")
    print("="*60)
    
    # 测试模型状态
    print("\n1. 检查模型状态...")
    if app_model:
        model_status = app_model.get_model_status()
        print(f"  模型可用性: {model_status['data']['model_available']}")
        print(f"  模型已加载: {model_status['data']['model_loaded']}")
        print(f"  分词器已加载: {model_status['data']['tokenizer_loaded']}")
        print(f"  预测器就绪: {model_status['data']['predictor_ready']}")
        
        if not model_status['data']['predictor_ready']:
            print("  ⚠️  预测器未就绪，预测功能测试将跳过")
            return
    else:
        print("  ❌ app_model模块未加载")
        return
    
    # 测试数据文件扫描
    print("\n2. 扫描数据文件...")
    data_files = app_model.load_data_files()
    print(f"  找到 {len(data_files)} 个数据文件")
    
    if data_files:
        for i, file_info in enumerate(data_files[:3], 1):  # 只显示前3个文件
            print(f"    {i}. {file_info['name']} ({file_info['size']})")
        
        # 测试文件预测
        print("\n3. 测试文件预测功能...")
        test_file = data_files[0]['path']
        print(f"  使用文件: {os.path.basename(test_file)}")
        
        # 执行预测（使用较小的参数以加快测试）
        result = predict_from_file(
            file_path=test_file,
            lookback=50,  # 减少到50个历史点
            pred_len=10,  # 减少到10个预测点
            temperature=0.7,
            top_p=0.9,
            sample_count=1
        )
        
        if result['success']:
            print("  ✅ 文件预测测试成功")
            print(f"    预测类型: {result['prediction_type']}")
            print(f"    预测点数: {len(result['prediction_results'])}")
            print(f"    包含对比数据: {result['has_comparison']}")
            print(f"    消息: {result['message']}")
        else:
            print(f"  ❌ 文件预测测试失败: {result['error']}")
    else:
        print("  ⚠️  没有找到数据文件，跳过文件预测测试")
    
    # 测试K线数据预测
    print("\n4. 测试K线数据预测功能...")
    
    # 创建模拟K线数据
    test_kline_data = []
    base_price = 100.0
    base_time = 1704067200  # 2024-01-01 00:00:00 UTC
    
    for i in range(60):  # 创建60个数据点
        # 模拟价格波动
        price_change = np.random.normal(0, 0.02)  # 2%的标准差
        open_price = base_price * (1 + price_change)
        high_price = open_price * (1 + abs(np.random.normal(0, 0.01)))
        low_price = open_price * (1 - abs(np.random.normal(0, 0.01)))
        close_price = open_price + np.random.normal(0, open_price * 0.01)
        
        test_kline_data.append({
            'timestamp': base_time + i * 3600,  # 每小时一个数据点
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': round(np.random.uniform(1000, 10000), 2)
        })
        
        base_price = close_price  # 下一个数据点的基础价格
    
    print(f"  创建了 {len(test_kline_data)} 个模拟K线数据点")
    
    # 执行K线预测
    result = predict_from_kline(
        kline_data=test_kline_data,
        lookback=30,  # 使用30个历史点
        pred_len=5,   # 预测5个点
        temperature=0.7,
        top_p=0.9,
        sample_count=1
    )
    
    if result['success']:
        print("  ✅ K线数据预测测试成功")
        print(f"    预测类型: {result['prediction_type']}")
        print(f"    预测点数: {len(result['prediction_results'])}")
        print(f"    输入数据信息: {result['input_data_info']}")
        print(f"    消息: {result['message']}")
        
        # 显示前几个预测结果
        print("    前3个预测结果:")
        for i, pred in enumerate(result['prediction_results'][:3]):
            print(f"      {i+1}. {pred['datetime']}: O={pred['open']:.2f}, H={pred['high']:.2f}, L={pred['low']:.2f}, C={pred['close']:.2f}")
    else:
        print(f"  ❌ K线数据预测测试失败: {result['error']}")
    
    print("\n" + "="*60)
    print("预测功能测试完成")
    print("="*60)


def test_chart_generation():
    """
    测试图表生成功能
    
    测试内容:
    - 创建模拟数据
    - 测试图表生成
    - 验证图表JSON格式
    """
    print("\n" + "="*60)
    print("测试图表生成功能")
    print("="*60)
    
    try:
        # 创建模拟历史数据
        print("\n1. 创建模拟数据...")
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        
        # 生成模拟OHLC数据
        np.random.seed(42)  # 设置随机种子以获得可重复的结果
        base_price = 100.0
        ohlc_data = []
        
        for i in range(len(dates)):
            # 模拟价格随机游走
            price_change = np.random.normal(0, 0.02)
            open_price = base_price * (1 + price_change)
            high_price = open_price * (1 + abs(np.random.normal(0, 0.01)))
            low_price = open_price * (1 - abs(np.random.normal(0, 0.01)))
            close_price = open_price + np.random.normal(0, open_price * 0.01)
            
            ohlc_data.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': np.random.uniform(1000, 10000)
            })
            
            base_price = close_price
        
        # 创建DataFrame
        df = pd.DataFrame(ohlc_data)
        df['timestamps'] = dates
        
        print(f"  创建了 {len(df)} 个历史数据点")
        print(f"  时间范围: {df['timestamps'].min()} 到 {df['timestamps'].max()}")
        print(f"  价格范围: {df['close'].min():.2f} - {df['close'].max():.2f}")
        
        # 创建模拟预测数据
        print("\n2. 创建模拟预测数据...")
        pred_dates = pd.date_range(start=dates[-1] + pd.Timedelta(hours=1), periods=20, freq='1H')
        pred_data = []
        
        last_price = df['close'].iloc[-1]
        for i in range(len(pred_dates)):
            price_change = np.random.normal(0, 0.015)
            open_price = last_price * (1 + price_change)
            high_price = open_price * (1 + abs(np.random.normal(0, 0.008)))
            low_price = open_price * (1 - abs(np.random.normal(0, 0.008)))
            close_price = open_price + np.random.normal(0, open_price * 0.008)
            
            pred_data.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': np.random.uniform(800, 8000)
            })
            
            last_price = close_price
        
        pred_df = pd.DataFrame(pred_data)
        print(f"  创建了 {len(pred_df)} 个预测数据点")
        
        # 测试图表生成
        print("\n3. 生成预测图表...")
        chart_json = create_prediction_chart(
            df=df,
            pred_df=pred_df,
            lookback=50,
            pred_len=20,
            actual_df=None,
            historical_start_idx=30
        )
        
        # 验证图表JSON
        try:
            chart_data = json.loads(chart_json)
            print("  ✅ 图表JSON格式验证成功")
            print(f"    图表类型: {chart_data.get('data', [{}])[0].get('type', 'unknown')}")
            print(f"    数据轨迹数量: {len(chart_data.get('data', []))}")
            print(f"    图表标题: {chart_data.get('layout', {}).get('title', 'N/A')}")
            
            # 检查数据轨迹
            traces = chart_data.get('data', [])
            for i, trace in enumerate(traces):
                trace_name = trace.get('name', f'Trace {i}')
                data_points = len(trace.get('x', []))
                print(f"    轨迹 {i+1}: {trace_name} ({data_points} 个数据点)")
                
        except json.JSONDecodeError as e:
            print(f"  ❌ 图表JSON格式验证失败: {e}")
        
        print("\n4. 测试带实际数据的图表生成...")
        
        # 创建模拟实际数据（用于对比）
        actual_data = []
        for i in range(15):  # 创建15个实际数据点
            price_change = np.random.normal(0, 0.012)
            open_price = last_price * (1 + price_change)
            high_price = open_price * (1 + abs(np.random.normal(0, 0.006)))
            low_price = open_price * (1 - abs(np.random.normal(0, 0.006)))
            close_price = open_price + np.random.normal(0, open_price * 0.006)
            
            actual_data.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': np.random.uniform(900, 9000)
            })
            
            last_price = close_price
        
        actual_df = pd.DataFrame(actual_data)
        
        # 生成包含实际数据的图表
        chart_with_actual = create_prediction_chart(
            df=df,
            pred_df=pred_df,
            lookback=40,
            pred_len=20,
            actual_df=actual_df,
            historical_start_idx=40
        )
        
        # 验证包含实际数据的图表
        try:
            chart_data_with_actual = json.loads(chart_with_actual)
            traces_with_actual = chart_data_with_actual.get('data', [])
            print(f"  ✅ 包含实际数据的图表生成成功")
            print(f"    数据轨迹数量: {len(traces_with_actual)}")
            
            for i, trace in enumerate(traces_with_actual):
                trace_name = trace.get('name', f'Trace {i}')
                data_points = len(trace.get('x', []))
                print(f"    轨迹 {i+1}: {trace_name} ({data_points} 个数据点)")
                
        except json.JSONDecodeError as e:
            print(f"  ❌ 包含实际数据的图表JSON验证失败: {e}")
        
    except Exception as e:
        print(f"  ❌ 图表生成测试失败: {str(e)}")
    
    print("\n" + "="*60)
    print("图表生成功能测试完成")
    print("="*60)


def test_utility_functions():
    """
    测试工具函数
    
    测试内容:
    - 测试预测结果保存功能
    - 测试数据处理功能
    - 测试错误处理
    """
    print("\n" + "="*60)
    print("测试工具函数")
    print("="*60)
    
    # 测试预测结果保存
    print("\n1. 测试预测结果保存功能...")
    
    try:
        # 创建模拟输入数据
        input_data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [105.0, 106.0, 107.0],
            'low': [98.0, 99.0, 100.0],
            'close': [103.0, 104.0, 105.0],
            'volume': [1000, 1100, 1200]
        })
        
        # 创建模拟预测结果
        prediction_results = [
            {
                'timestamp': '2024-01-01T10:00:00',
                'open': 106.0,
                'high': 108.0,
                'low': 104.0,
                'close': 107.0,
                'volume': 1300
            },
            {
                'timestamp': '2024-01-01T11:00:00',
                'open': 107.0,
                'high': 109.0,
                'low': 105.0,
                'close': 108.0,
                'volume': 1400
            }
        ]
        
        # 创建模拟实际数据
        actual_data = [
            {
                'timestamp': '2024-01-01T10:00:00',
                'open': 105.5,
                'high': 107.5,
                'low': 103.5,
                'close': 106.5,
                'volume': 1250
            }
        ]
        
        # 测试保存功能
        saved_path = save_prediction_results(
            file_path="test_data.csv",
            prediction_type="测试预测",
            prediction_results=prediction_results,
            actual_data=actual_data,
            input_data=input_data,
            prediction_params={
                'lookback': 3,
                'pred_len': 2,
                'temperature': 0.7,
                'top_p': 0.9,
                'sample_count': 1
            }
        )
        
        if saved_path:
            print("  ✅ 预测结果保存测试成功")
            print(f"    保存路径: {saved_path}")
            
            # 验证保存的文件
            if os.path.exists(saved_path):
                with open(saved_path, 'r', encoding='utf-8') as f:
                    saved_data = json.load(f)
                
                print(f"    文件大小: {os.path.getsize(saved_path)} 字节")
                print(f"    包含预测结果: {len(saved_data.get('prediction_results', []))} 个")
                print(f"    包含实际数据: {len(saved_data.get('actual_data', []))} 个")
                print(f"    包含分析数据: {'analysis' in saved_data}")
                
                # 检查分析数据
                if 'analysis' in saved_data and 'continuity' in saved_data['analysis']:
                    continuity = saved_data['analysis']['continuity']
                    
                    try:
                        if 'overall_accuracy' in continuity and 'avg_relative_error' in continuity['overall_accuracy']:
                            avg_error = continuity['overall_accuracy']['avg_relative_error']
                        elif 'gap_percentages' in continuity:
                            gap_pcts = continuity['gap_percentages']
                            avg_error = (gap_pcts.get('open_gap_pct', 0) + 
                                        gap_pcts.get('high_gap_pct', 0) + 
                                        gap_pcts.get('low_gap_pct', 0) + 
                                        gap_pcts.get('close_gap_pct', 0)) / 4
                        else:
                            avg_error = 0
                            
                        print(f"    连续性分析: 平均相对误差 {avg_error:.2f}%")
                    except (KeyError, TypeError, ZeroDivisionError) as e:
                        print(f"    连续性分析: 计算错误 - {e}")
                else:
                    print("    连续性分析: 未找到分析数据")
            else:
                print("  ⚠️  保存的文件不存在")
        else:
            print("  ❌ 预测结果保存测试失败")
            
    except Exception as e:
        print(f"  ❌ 预测结果保存测试异常: {str(e)}")
    
    # 测试错误处理
    print("\n2. 测试错误处理...")
    
    # 测试空数据处理
    result = predict_from_kline([], 10, 5)
    if not result['success'] and 'K线数据不能为空' in result['error']:
        print("  ✅ 空数据错误处理测试通过")
    else:
        print("  ❌ 空数据错误处理测试失败")
    
    # 测试无效数据处理
    invalid_kline = [{'invalid': 'data'}]
    result = predict_from_kline(invalid_kline, 10, 5)
    if not result['success'] and '缺少必需的列' in result['error']:
        print("  ✅ 无效数据错误处理测试通过")
    else:
        print("  ❌ 无效数据错误处理测试失败")
    
    # 测试数据长度不足处理
    short_kline = [{
        'open': 100, 'high': 105, 'low': 95, 'close': 102,
        'timestamp': 1704067200
    }]
    result = predict_from_kline(short_kline, 10, 5)  # 需要10个数据点，但只有1个
    if not result['success'] and '数据长度不足' in result['error']:
        print("  ✅ 数据长度不足错误处理测试通过")
    else:
        print("  ❌ 数据长度不足错误处理测试失败")
    
    print("\n" + "="*60)
    print("工具函数测试完成")
    print("="*60)


def run_all_tests():
    """
    运行所有测试函数
    
    测试流程:
    1. 预测功能测试
    2. 图表生成测试
    3. 工具函数测试
    """
    print("\n" + "="*80)
    print("Kronos 预测模块 - 完整功能测试")
    print("="*80)
    print(f"测试开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        app_model.load_model('kronos-mini')
        # 运行各项测试
        test_prediction_functions()
        test_chart_generation()
        test_utility_functions()
        
        print("\n" + "="*80)
        print("所有测试完成")
        print("="*80)
        print(f"测试结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\n❌ 测试过程中发生异常: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    """
    主函数 - 运行测试
    """
    
    run_all_tests()
