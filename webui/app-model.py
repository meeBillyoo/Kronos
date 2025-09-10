#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kronos 模型管理模块
提供模型加载、预测和状态管理功能

主要功能:
- 模型配置和加载管理
- 数据文件处理和验证
- 预测结果保存和分析
- 模型状态监控

作者: Kronos Team
版本: 1.0.0
创建时间: 2024
"""

import os
import pandas as pd
import numpy as np
import json
import sys
import warnings
import datetime
from typing import Optional, Dict, Any, Tuple, List

# 忽略警告信息
warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径，以便导入模型模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 尝试导入Kronos模型相关类
try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("警告: 无法导入 Kronos 模型，将使用模拟数据进行演示")

# 全局变量存储已加载的模型组件
tokenizer = None  # 分词器实例
model = None      # 模型实例
predictor = None  # 预测器实例

# 计算模型目录的绝对路径
MODEL_BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')

# 可用模型配置字典
# 包含不同规模的Kronos模型配置信息
AVAILABLE_MODELS = {
    'kronos-mini': {
        'name': 'Kronos-mini',
        'model_id': 'NeoQuasar/Kronos-mini',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-2k',
        'context_length': 2048,
        'params': '4.1M',
        'description': '轻量级模型，适合快速预测，参数量小，推理速度快'
    },
    'kronos-small': {
        'name': 'Kronos-small',
        'model_id': 'NeoQuasar/Kronos-small',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '24.7M',
        'description': '小型模型，平衡性能和速度，适合一般预测任务'
    },
    'kronos-base': {
        'name': 'Kronos-base',
        'model_id': 'NeoQuasar/Kronos-base',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '102.3M',
        'description': '基础模型，提供更好的预测质量，适合高精度预测需求'
    }
}


def load_data_files() -> List[Dict[str, str]]:
    """
    扫描数据目录并返回可用的数据文件列表
    
    功能说明:
    - 扫描webui/data目录下的所有数据文件
    - 支持CSV和Feather格式文件
    - 计算文件大小并格式化显示
    
    Returns:
        List[Dict[str, str]]: 包含文件信息的字典列表
                             每个字典包含name、path、size字段
    """
    # 构建数据目录路径
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    data_files = []
    
    # 检查数据目录是否存在
    if os.path.exists(data_dir):
        # 遍历目录中的所有文件
        for file in os.listdir(data_dir):
            # 只处理支持的文件格式
            if file.endswith(('.csv', '.feather')):
                file_path = os.path.join(data_dir, file)
                file_size = os.path.getsize(file_path)
                
                # 格式化文件大小显示
                size_str = (
                    f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 
                    else f"{file_size / (1024*1024):.1f} MB"
                )
                
                data_files.append({
                    'name': file,
                    'path': file_path,
                    'size': size_str
                })
    
    return data_files


def load_data_file(file_path: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    加载数据文件并进行预处理
    
    功能说明:
    - 支持CSV和Feather格式文件加载
    - 验证必需的OHLC列是否存在
    - 处理时间戳列的多种格式
    - 数据类型转换和清理
    
    Args:
        file_path (str): 数据文件的完整路径
        
    Returns:
        Tuple[Optional[pd.DataFrame], Optional[str]]: 
            - 成功时返回(DataFrame, None)
            - 失败时返回(None, 错误信息)
    """
    try:
        # 根据文件扩展名选择加载方法
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.feather'):
            df = pd.read_feather(file_path)
        else:
            return None, "不支持的文件格式，仅支持CSV和Feather格式"
        
        # 检查必需的OHLC列是否存在
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return None, f"缺少必需的列: {missing_cols}"
        
        # 处理时间戳列，支持多种列名格式
        timestamp_cols = ['timestamps', 'timestamp', 'date', 'time']
        timestamp_col = None
        
        for col in timestamp_cols:
            if col in df.columns:
                timestamp_col = col
                break
        
        if timestamp_col:
            # 转换现有的时间戳列
            df['timestamps'] = pd.to_datetime(df[timestamp_col])
            # 如果原列名不是timestamps，删除原列
            if timestamp_col != 'timestamps':
                df = df.drop(columns=[timestamp_col])
        else:
            # 如果没有时间戳列，创建一个默认的时间序列
            print("警告: 未找到时间戳列，将创建默认时间序列")
            df['timestamps'] = pd.date_range(
                start='2024-01-01', 
                periods=len(df), 
                freq='1H'
            )
        
        # 确保OHLC列是数值类型
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 处理可选的成交量列
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # 处理可选的成交额列（通常不用于预测）
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # 删除包含NaN值的行
        original_len = len(df)
        df = df.dropna()
        
        if len(df) < original_len:
            print(f"警告: 删除了 {original_len - len(df)} 行包含缺失值的数据")
        
        # 按时间戳排序
        df = df.sort_values('timestamps').reset_index(drop=True)
        
        return df, None
        
    except Exception as e:
        return None, f"加载文件失败: {str(e)}"


def save_prediction_results(file_path: str, prediction_type: str, prediction_results: List[Dict], 
                          actual_data: List[Dict], input_data: pd.DataFrame, 
                          prediction_params: Dict[str, Any]) -> Optional[str]:
    """
    保存预测结果到JSON文件
    
    功能说明:
    - 创建预测结果目录
    - 生成带时间戳的文件名
    - 保存完整的预测信息和分析结果
    - 计算预测与实际数据的差异分析
    
    Args:
        file_path (str): 原始数据文件路径
        prediction_type (str): 预测类型标识
        prediction_results (List[Dict]): 预测结果列表
        actual_data (List[Dict]): 实际数据列表（用于对比）
        input_data (pd.DataFrame): 输入数据
        prediction_params (Dict[str, Any]): 预测参数
        
    Returns:
        Optional[str]: 成功时返回保存的文件路径，失败时返回None
    """
    try:
        # 创建预测结果存储目录
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # 生成带时间戳的文件名
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'prediction_{timestamp}.json'
        filepath = os.path.join(results_dir, filename)
        
        # 构建保存数据结构
        save_data = {
            'metadata': {
                'timestamp': datetime.datetime.now().isoformat(),
                'file_path': file_path,
                'prediction_type': prediction_type,
                'prediction_params': prediction_params
            },
            'input_data_summary': {
                'total_rows': len(input_data),
                'columns': list(input_data.columns),
                'time_range': {
                    'start': input_data['timestamps'].min().isoformat() if 'timestamps' in input_data.columns else None,
                    'end': input_data['timestamps'].max().isoformat() if 'timestamps' in input_data.columns else None
                },
                'price_statistics': {
                    'open': {
                        'min': float(input_data['open'].min()), 
                        'max': float(input_data['open'].max()),
                        'mean': float(input_data['open'].mean())
                    },
                    'high': {
                        'min': float(input_data['high'].min()), 
                        'max': float(input_data['high'].max()),
                        'mean': float(input_data['high'].mean())
                    },
                    'low': {
                        'min': float(input_data['low'].min()), 
                        'max': float(input_data['low'].max()),
                        'mean': float(input_data['low'].mean())
                    },
                    'close': {
                        'min': float(input_data['close'].min()), 
                        'max': float(input_data['close'].max()),
                        'mean': float(input_data['close'].mean())
                    }
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
        
        # 如果存在实际数据，进行预测准确性分析
        if actual_data and len(actual_data) > 0 and len(prediction_results) > 0:
            # 计算预测与实际数据的连续性分析
            last_pred = prediction_results[0]  # 第一个预测点
            first_actual = actual_data[0]      # 第一个实际点
            
            # 计算各项指标的绝对差异和相对差异
            gaps = {}
            gap_percentages = {}
            
            for metric in ['open', 'high', 'low', 'close']:
                if metric in last_pred and metric in first_actual:
                    abs_gap = abs(last_pred[metric] - first_actual[metric])
                    rel_gap = (abs_gap / first_actual[metric]) * 100 if first_actual[metric] != 0 else 0
                    
                    gaps[f'{metric}_gap'] = abs_gap
                    gap_percentages[f'{metric}_gap_pct'] = rel_gap
            
            save_data['analysis']['continuity'] = {
                'description': '预测值与实际值的连续性分析',
                'last_prediction': last_pred,
                'first_actual': first_actual,
                'absolute_gaps': gaps,
                'relative_gaps_percent': gap_percentages,
                'overall_accuracy': {
                    'avg_relative_error': np.mean(list(gap_percentages.values())),
                    'max_relative_error': max(gap_percentages.values()) if gap_percentages else 0
                }
            }
        
        # 保存到JSON文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"预测结果已保存到: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"保存预测结果失败: {e}")
        return None


def load_model(model_key: str, device: str = 'cpu') -> Dict[str, Any]:
    """
    加载指定的Kronos模型
    
    功能说明:
    - 验证模型可用性和模型键的有效性
    - 加载分词器和模型
    - 创建预测器实例
    - 更新全局模型状态
    
    Args:
        model_key (str): 模型键名，必须在AVAILABLE_MODELS中
        device (str): 运行设备，默认为'cpu'，可选'cuda'
        
    Returns:
        Dict[str, Any]: 包含加载结果的字典
                       成功时code=0，失败时code!=0
    """
    global tokenizer, model, predictor
    
    try:
        # 检查模型是否可用
        if not MODEL_AVAILABLE:
            return {
                'code': 500,
                'message': '模型不可用，请检查模型模块是否正确安装',
                'data': None
            }
        
        # 验证模型键是否有效
        if model_key not in AVAILABLE_MODELS:
            available_keys = list(AVAILABLE_MODELS.keys())
            return {
                'code': 400,
                'message': f'不支持的模型: {model_key}，可用模型: {available_keys}',
                'data': None
            }
        
        # 获取模型配置
        model_config = AVAILABLE_MODELS[model_key]
        print(f"开始加载模型: {model_config['name']}...")
        
        # 加载分词器
        print("正在加载分词器...")
        tokenizer = KronosTokenizer.from_pretrained(model_config['tokenizer_id'])
        
        # 加载模型
        print("正在加载模型...")
        model = Kronos.from_pretrained(model_config['model_id'])
        
        # 创建预测器实例
        print("正在创建预测器...")
        predictor = KronosPredictor(
            model, 
            tokenizer, 
            device=device, 
            max_context=model_config['context_length']
        )
        
        success_msg = f"模型加载成功: {model_config['name']} ({model_config['params']}) 在 {device} 设备上"
        print(success_msg)
        
        return {
            'code': 0,
            'message': success_msg,
            'data': {
                'success': True,
                'model_info': {
                    'key': model_key,
                    'name': model_config['name'],
                    'params': model_config['params'],
                    'context_length': model_config['context_length'],
                    'description': model_config['description'],
                    'device': device
                }
            }
        }
        
    except Exception as e:
        error_msg = f"模型加载失败: {str(e)}"
        print(error_msg)
        return {
            'code': 500,
            'message': error_msg,
            'data': None
        }


def get_available_models() -> Dict[str, Any]:
    """
    获取所有可用模型的配置信息
    
    Returns:
        Dict[str, Any]: 包含所有可用模型信息的字典
    """
    try:
        return {
            'code': 0,
            'message': '获取可用模型列表成功',
            'data': {
                'models': AVAILABLE_MODELS,
                'model_available': MODEL_AVAILABLE,
                'total_models': len(AVAILABLE_MODELS)
            }
        }
    except Exception as e:
        return {
            'code': 500,
            'message': f'获取可用模型失败: {str(e)}',
            'data': None
        }


def get_model_status() -> Dict[str, Any]:
    """
    获取当前模型的加载状态
    
    功能说明:
    - 检查模型组件的加载状态
    - 返回当前模型的详细信息
    - 提供设备和上下文长度信息
    
    Returns:
        Dict[str, Any]: 包含模型状态信息的字典
    """
    try:
        if MODEL_AVAILABLE:
            # 构建详细的状态信息
            status = {
                'model_available': True,
                'model_loaded': model is not None,
                'tokenizer_loaded': tokenizer is not None,
                'predictor_ready': predictor is not None,
                'components_status': {
                    'tokenizer': 'loaded' if tokenizer is not None else 'not_loaded',
                    'model': 'loaded' if model is not None else 'not_loaded',
                    'predictor': 'ready' if predictor is not None else 'not_ready'
                }
            }
            
            # 如果预测器已就绪，添加详细信息
            if predictor is not None:
                status['current_model'] = {
                    'device': str(predictor.device),
                    'max_context': predictor.max_context,
                    'model_type': type(predictor.model).__name__,
                    'tokenizer_type': type(predictor.tokenizer).__name__
                }
        else:
            # 模型不可用时的状态
            status = {
                'model_available': False,
                'model_loaded': False,
                'tokenizer_loaded': False,
                'predictor_ready': False,
                'error_reason': '模型模块导入失败'
            }
        
        return {
            'code': 0,
            'message': '获取模型状态成功',
            'data': status
        }
        
    except Exception as e:
        return {
            'code': 500,
            'message': f'获取模型状态失败: {str(e)}',
            'data': None
        }


def get_model_instance() -> Dict[str, Any]:
    """
    获取模型实例的状态信息（用于内部调用）
    
    Returns:
        Dict[str, Any]: 模型实例状态字典
    """
    status = {
        'model_available': MODEL_AVAILABLE,
        'model_loaded': model,
        'tokenizer_loaded': tokenizer,
        'predictor_ready': predictor
    }
    
    # 如果预测器存在，添加当前模型信息
    if predictor is not None:
        status['current_model'] = {
            'device': str(predictor.device),
            'max_context': predictor.max_context
        }
    
    return status


def test_data_loading():
    """
    测试数据加载功能
    
    测试内容:
    - 扫描数据目录
    - 加载第一个可用的数据文件
    - 验证数据格式和内容
    """
    print("\n" + "="*50)
    print("测试数据加载功能")
    print("="*50)
    
    # 测试数据文件扫描
    print("\n1. 扫描数据文件...")
    data_files = load_data_files()
    print(f"找到 {len(data_files)} 个数据文件:")
    
    if not data_files:
        print("  ❌ 没有找到任何数据文件")
        print("  💡 请在 webui/data/ 目录下放置CSV或Feather格式的数据文件")
        return
    
    # 显示找到的文件
    for i, file_info in enumerate(data_files, 1):
        print(f"  {i}. {file_info['name']} ({file_info['size']})")
    
    # 测试加载第一个文件
    print("\n2. 测试加载第一个数据文件...")
    first_file = data_files[0]['path']
    print(f"正在加载: {os.path.basename(first_file)}")
    
    df, error = load_data_file(first_file)
    if error:
        print(f"  ❌ 加载失败: {error}")
    else:
        print(f"  ✅ 加载成功!")
        print(f"  📊 数据行数: {len(df)}")
        print(f"  📋 列名: {list(df.columns)}")
        
        if len(df) > 0:
            print(f"  📅 时间范围: {df['timestamps'].min()} 到 {df['timestamps'].max()}")
            print(f"  💰 价格范围: {df['close'].min():.2f} - {df['close'].max():.2f}")
            print(f"  📈 最后一行数据:")
            last_row = df.iloc[-1]
            for col in ['timestamps', 'open', 'high', 'low', 'close']:
                if col in last_row:
                    value = last_row[col]
                    if col == 'timestamps':
                        print(f"    {col}: {value}")
                    else:
                        print(f"    {col}: {value:.4f}")


def test_model_functions():
    """
    测试模型相关功能
    
    测试内容:
    - 获取可用模型列表
    - 检查模型状态
    - 尝试加载模型（如果可用）
    """
    print("\n" + "="*50)
    print("测试模型功能")
    print("="*50)
    
    # 测试获取可用模型
    print("\n1. 获取可用模型列表...")
    models_result = get_available_models()
    
    if models_result['code'] == 0:
        print(f"  ✅ {models_result['message']}")
        data = models_result['data']
        print(f"  🤖 模型可用性: {'✅ 可用' if data['model_available'] else '❌ 不可用'}")
        print(f"  📦 模型数量: {data['total_models']}")
        
        print("\n  可用模型详情:")
        for key, model_info in data['models'].items():
            print(f"    🔹 {key}:")
            print(f"      名称: {model_info['name']}")
            print(f"      参数: {model_info['params']}")
            print(f"      上下文: {model_info['context_length']}")
            print(f"      描述: {model_info['description']}")
    else:
        print(f"  ❌ {models_result['message']}")
    
    # 测试获取模型状态
    print("\n2. 检查模型状态...")
    status_result = get_model_status()
    
    if status_result['code'] == 0:
        print(f"  ✅ {status_result['message']}")
        status = status_result['data']
        print(f"  🔧 模型可用: {'✅' if status['model_available'] else '❌'}")
        print(f"  📦 模型已加载: {'✅' if status['model_loaded'] else '❌'}")
        print(f"  🔤 分词器已加载: {'✅' if status['tokenizer_loaded'] else '❌'}")
        print(f"  🚀 预测器就绪: {'✅' if status['predictor_ready'] else '❌'}")
        
        if 'current_model' in status:
            current = status['current_model']
            print(f"  💻 当前设备: {current['device']}")
            print(f"  📏 最大上下文: {current['max_context']}")
    else:
        print(f"  ❌ {status_result['message']}")
    
    # 如果模型可用，测试加载
    if MODEL_AVAILABLE:
        print("\n3. 测试模型加载...")
        print("  正在尝试加载 kronos-mini 模型...")
        
        load_result = load_model('kronos-mini', 'cpu')
        if load_result['code'] == 0:
            print(f"  ✅ {load_result['message']}")
            model_info = load_result['data']['model_info']
            print(f"  📋 模型信息:")
            print(f"    名称: {model_info['name']}")
            print(f"    参数量: {model_info['params']}")
            print(f"    设备: {model_info['device']}")
            print(f"    上下文长度: {model_info['context_length']}")
        else:
            print(f"  ❌ {load_result['message']}")
    else:
        print("\n3. 跳过模型加载测试")
        print("  ⚠️  模型不可用，无法进行加载测试")
        print("  💡 请确保已正确安装Kronos模型模块")


def test_utility_functions():
    """
    测试工具函数
    
    测试内容:
    - 模型实例状态获取
    - 预测结果保存功能
    """
    print("\n" + "="*50)
    print("测试工具函数")
    print("="*50)
    
    # 测试模型实例状态获取
    print("\n1. 测试模型实例状态获取...")
    instance_status = get_model_instance()
    print(f"  📊 模型实例状态:")
    print(f"    模型可用: {'✅' if instance_status['model_available'] else '❌'}")
    print(f"    模型已加载: {'✅' if instance_status['model_loaded'] is not None else '❌'}")
    print(f"    分词器已加载: {'✅' if instance_status['tokenizer_loaded'] is not None else '❌'}")
    print(f"    预测器就绪: {'✅' if instance_status['predictor_ready'] is not None else '❌'}")
    
    if 'current_model' in instance_status:
        current = instance_status['current_model']
        print(f"    当前设备: {current['device']}")
        print(f"    最大上下文: {current['max_context']}")
    
    # 测试预测结果保存功能
    print("\n2. 测试预测结果保存功能...")
    print("  创建模拟数据进行测试...")
    
    # 创建模拟输入数据
    mock_input_data = pd.DataFrame({
        'open': [100.0, 101.0, 102.0],
        'high': [105.0, 106.0, 107.0],
        'low': [95.0, 96.0, 97.0],
        'close': [103.0, 104.0, 105.0],
        'volume': [1000, 1100, 1200],
        'timestamps': pd.date_range('2024-01-01', periods=3, freq='1H')
    })
    
    # 创建模拟预测结果
    mock_prediction = [
        {
            'open': 106.0, 'high': 108.0, 'low': 104.0, 'close': 107.0, 
            'timestamp': '2024-01-01T04:00:00'
        },
        {
            'open': 107.0, 'high': 109.0, 'low': 105.0, 'close': 108.0, 
            'timestamp': '2024-01-01T05:00:00'
        }
    ]
    
    # 创建模拟实际数据（用于对比）
    mock_actual = [
        {
            'open': 105.5, 'high': 107.5, 'low': 103.5, 'close': 106.5, 
            'timestamp': '2024-01-01T04:00:00'
        }
    ]
    
    # 保存测试结果
    saved_path = save_prediction_results(
        file_path='test_data.csv',
        prediction_type='unit_test',
        prediction_results=mock_prediction,
        actual_data=mock_actual,
        input_data=mock_input_data,
        prediction_params={
            'model': 'kronos-mini',
            'steps': 2,
            'device': 'cpu',
            'test_mode': True
        }
    )
    
    if saved_path:
        print(f"  ✅ 预测结果保存成功")
        print(f"  📁 保存路径: {saved_path}")
        print(f"  📄 文件名: {os.path.basename(saved_path)}")
        
        # 验证保存的文件
        try:
            with open(saved_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            print(f"  📊 保存的数据包含:")
            print(f"    元数据: ✅")
            print(f"    输入数据摘要: ✅")
            print(f"    预测结果: {len(saved_data.get('prediction_results', []))} 条")
            print(f"    实际数据: {len(saved_data.get('actual_data', []))} 条")
            print(f"    分析结果: {'✅' if saved_data.get('analysis') else '❌'}")
        except Exception as e:
            print(f"  ⚠️  验证保存文件时出错: {e}")
    else:
        print(f"  ❌ 预测结果保存失败")


if __name__ == '__main__':
    """
    主程序入口
    运行所有测试函数以验证模块功能
    """
    print("🚀 启动 Kronos 模型管理模块测试")
    print(f"📦 模型可用性: {'✅ 可用' if MODEL_AVAILABLE else '❌ 不可用'}")
    
    if not MODEL_AVAILABLE:
        print("⚠️  警告: Kronos模型模块不可用")
        print("💡 这可能是因为:")
        print("   1. 模型模块未正确安装")
        print("   2. Python路径配置问题")
        print("   3. 依赖库缺失")
        print("\n🔄 将继续运行其他功能的测试...")
    
    # 运行所有测试
    try:
        test_data_loading()
        test_model_functions()
        test_utility_functions()
        
        print("\n" + "="*50)
        print("🎉 所有测试完成")
        print("="*50)
        print("✅ 数据加载功能测试完成")
        print("✅ 模型功能测试完成")
        print("✅ 工具函数测试完成")
        print("\n💡 如需启动Flask Web服务，请运行 app.py")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  测试被用户中断")
    except Exception as e:
        print(f"\n\n❌ 测试过程中出现错误: {e}")
        print("\n🔍 详细错误信息:")
        import traceback
        traceback.print_exc()
        print("\n💡 请检查错误信息并修复相关问题")
